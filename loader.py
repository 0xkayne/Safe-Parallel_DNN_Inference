import csv
import json
import networkx as nx
from common import DNNLayer

class ModelLoader:
    @staticmethod
    def load_model_from_csv(file_path):
        G = nx.DiGraph()
        layers_map = {} # name -> DNNLayer object
        
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
            name_to_id = {}
            
            # Helper to get value from row with multiple possible keys
            def get_val(row, keys, default=None):
                for k in keys:
                    if k in row:
                        return row[k]
                return default

            for idx, row in enumerate(rows):
                name = get_val(row, ['name', 'LayerName'])
                if not name:
                    continue
                    
                name_to_id[name] = idx
                
                try:
                    # Time
                    enclave_time = float(get_val(row, ['enclave_time_mean', 'EnclaveTime_mean'], 0))
                    cpu_time = float(get_val(row, ['cpu_time_mean', 'CPUTime_mean'], 0))
                    
                    # Memory
                    mem_bytes = float(get_val(row, ['tee_total_memory_bytes', 'TEE_Total_Memory_Bytes'], 0))
                    mem_mb = mem_bytes / (1024 * 1024)
                    
                    # Communication
                    out_bytes = float(get_val(row, ['output_bytes', 'OutputBytes'], 0))
                    
                except ValueError:
                    continue 
                
                layer = DNNLayer(idx, name, mem_mb, cpu_time, enclave_time, out_bytes)
                layers_map[idx] = layer
                G.add_node(idx, layer=layer)
            
            # Edges
            # ---- Runtime virtual splitting for ViT attention ----
            # Build a mapping from layer name to its row for easy lookup
            name_to_row = {get_val(row, ['name', 'LayerName']): row for row in rows if get_val(row, ['name', 'LayerName'])}
            # Keep track of new virtual rows to be added
            virtual_rows = []
            for name, row in list(name_to_row.items()):
                if name.endswith('_attn_qkv_proj'):
                    base = name[:-len('_attn_qkv_proj')]
                    # Original dependencies (usually from norm layer)
                    orig_deps = get_val(row, ['dependencies', 'Dependencies'], "[]")
                    # Create virtual Q, K, V projection rows
                    for suffix in ['_attn_q_proj', '_attn_k_proj', '_attn_v_proj']:
                        virt_name = base + suffix
                        virt_row = {
                            'name': virt_name,
                            'LayerName': virt_name,
                            'type': 'Linear',
                            'group': row.get('group', ''),
                            'enclave_time_mean': '0',
                            'cpu_time_mean': '0',
                            'tee_total_memory_bytes': '0',
                            'output_bytes': '0',
                            'dependencies': orig_deps,
                        }
                        virtual_rows.append(virt_row)
                        name_to_row[virt_name] = virt_row
            # Append virtual rows to the original rows list so they are processed later
            rows.extend(virtual_rows)
            # Adjust dependencies for qk_matmul and v_matmul to use the virtual Q/K/V nodes
            for name, row in name_to_row.items():
                if name.endswith('_attn_qk_matmul'):
                    base = name[:-len('_attn_qk_matmul')]
                    q_name = base + '_attn_q_proj'
                    k_name = base + '_attn_k_proj'
                    row['dependencies'] = f"[{q_name}, {k_name}]"
                elif name.endswith('_attn_v_matmul'):
                    base = name[:-len('_attn_v_matmul')]
                    softmax_name = base + '_attn_softmax'
                    v_name = base + '_attn_v_proj'
                    row['dependencies'] = f"[{softmax_name}, {v_name}]"
            # End of runtime virtual splitting
            # ---- Existing attention head fix for models that already have separate q/k/v ----
            for name, row in name_to_row.items():
                if name.endswith('_attn_q_proj'):
                    base = name[:-len('_attn_q_proj')]
                    k_name = base + '_attn_k_proj'
                    v_name = base + '_attn_v_proj'
                    q_deps = get_val(row, ['dependencies', 'Dependencies'], "[]")
                    for dep_name in (k_name, v_name):
                        if dep_name in name_to_row:
                            name_to_row[dep_name]['dependencies'] = q_deps
            # End of attention dependency fix
            # ---- Fix attention head dependencies for transformer models ----
            # Build a mapping from layer name to its row for easy lookup
            name_to_row = {get_val(row, ['name', 'LayerName']): row for row in rows if get_val(row, ['name', 'LayerName'])}
            for name, row in name_to_row.items():
                if name.endswith('_attn_q_proj'):
                    base = name[:-len('_attn_q_proj')]
                    k_name = base + '_attn_k_proj'
                    v_name = base + '_attn_v_proj'
                    # Get dependencies of Q projection
                    q_deps = get_val(row, ['dependencies', 'Dependencies'], "[]")
                    # Apply same dependencies to K and V rows if they exist
                    for dep_name in (k_name, v_name):
                        if dep_name in name_to_row:
                            name_to_row[dep_name]['dependencies'] = q_deps
            # End of attention dependency fix
            for idx, row in enumerate(rows):
                name = get_val(row, ['name', 'LayerName'])
                if not name or name not in name_to_id:
                    continue
                    
                curr_id = name_to_id[name]
                
                deps_str = get_val(row, ['dependencies', 'Dependencies'], "[]")
                
                deps_list = []
                try:
                    if deps_str:
                        # InceptionV3 might use ";", others use python list string
                        if deps_str.startswith('[') and deps_str.endswith(']'):
                             deps_list = json.loads(deps_str.replace("'", '"'))
                        elif ';' in deps_str:
                             deps_list = deps_str.split(';')
                        else:
                             # Single item not in list brackets? or empty
                             if deps_str.strip():
                                 deps_list = [deps_str]
                except:
                    deps_list = []
                
                for dep_name in deps_list:
                    dep_name = dep_name.strip()
                    if dep_name in name_to_id:
                        prev_id = name_to_id[dep_name]
                        comm_size_mb = layers_map[prev_id].output_bytes / (1024 * 1024)
                        G.add_edge(prev_id, curr_id, weight=comm_size_mb)
        
        return G, layers_map
