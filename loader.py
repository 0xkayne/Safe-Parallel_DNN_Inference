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
            
            # Helper to get value from row with multiple possible keys
            def get_val(row, keys, default=None):
                for k in keys:
                    if k in row:
                        return row[k]
                return default

            # =========================================================
            # PHASE 1: PRE-PROCESS & VIRTUAL LAYER GENERATION
            # We must do this BEFORE creating nodes so virtual nodes 
            # are indexed correctly.
            # =========================================================
            
            # Build a mapping from layer name to its row for easy lookup
            name_to_row = {get_val(row, ['name', 'LayerName']): row for row in rows if get_val(row, ['name', 'LayerName'])}
            
            # Detect if dataset requires virtual QKV splitting (old format) or has per-head layers (new format)
            requires_virtual_qkv = any(name.endswith('_attn_qkv_proj') for name in name_to_row.keys())
            has_per_head_layers = any('_head' in name.lower() for name in name_to_row.keys())
            
            virtual_rows = []
            
            # 1.1 Virtual Splitting for QKV projection (Single QKV -> Virtual Q, K, V)
            # Only apply for old datasets without per-head layers
            if requires_virtual_qkv and not has_per_head_layers:
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
                                'enclave_time_mean': '0', # Virtual split, minimal overhead assumed or amortized
                                'cpu_time_mean': '0',
                                'tee_total_memory_bytes': '0',
                                'output_bytes': '0', # Will be set by splitting original output? or kept 0 for virtual
                                'dependencies': orig_deps,
                            }
                            virtual_rows.append(virt_row)
                            name_to_row[virt_name] = virt_row
            
            # Add virtual rows to main list
            rows.extend(virtual_rows)
            
            # 1.2 Fix dependencies for MatMul layers to point to new Virtual Layers
            # Only needed if we created virtual layers
            if requires_virtual_qkv and not has_per_head_layers:
                for name, row in name_to_row.items():
                    if name.endswith('_attn_qk_matmul'):
                        base = name[:-len('_attn_qk_matmul')]
                        q_name = base + '_attn_q_proj'
                        k_name = base + '_attn_k_proj'
                        # Update dependency to point to virtual Q and K
                        # Use standard list string format with quotes for parser compatibility
                        new_deps = f"['{q_name}', '{k_name}']"
                        row['dependencies'] = new_deps
                        row['Dependencies'] = new_deps # Ensure both keys updated
                    elif name.endswith('_attn_v_matmul'):
                        base = name[:-len('_attn_v_matmul')]
                        softmax_name = base + '_attn_softmax'
                        v_name = base + '_attn_v_proj'
                        # Update dependency to point to softmax and virtual V
                        new_deps = f"['{softmax_name}', '{v_name}']"
                        row['dependencies'] = new_deps
                        row['Dependencies'] = new_deps

            # 1.3 Fix dependencies for existing separate Q/K/V heads (Propagate dependencies)
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
                            target_row = name_to_row[dep_name]
                            target_row['dependencies'] = q_deps
                            target_row['Dependencies'] = q_deps

            # =========================================================
            # PHASE 2: NODE CREATION
            # Now iterating over ALL rows (including virtual ones)
            # =========================================================
            name_to_id = {}
            
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
                    
                    # Granular Memory Components (for dynamic peak memory calculation)
                    weight_bytes = float(get_val(row, ['weight_bytes'], 0))
                    bias_bytes = float(get_val(row, ['bias_bytes'], 0))
                    activation_bytes = float(get_val(row, ['activation_bytes'], 0))
                    encryption_bytes = float(get_val(row, ['tee_encryption_overhead'], 0))
                    
                    weight_mb = weight_bytes / (1024 * 1024)
                    bias_mb = bias_bytes / (1024 * 1024)
                    activation_mb = activation_bytes / (1024 * 1024)
                    encryption_mb = encryption_bytes / (1024 * 1024)
                    
                    # Communication
                    out_bytes = float(get_val(row, ['output_bytes', 'OutputBytes'], 0))
                    
                    # Execution mode (new dataset format)
                    execution_mode = get_val(row, ['execution_mode', 'ExecutionMode'], 'Unknown')
                    
                except ValueError:
                    continue 
                
                layer = DNNLayer(idx, name, mem_mb, cpu_time, enclave_time, out_bytes, execution_mode,
                                 weight_memory=weight_mb,
                                 bias_memory=bias_mb,
                                 activation_memory=activation_mb,
                                 encryption_overhead=encryption_mb)
                layers_map[idx] = layer
                G.add_node(idx, layer=layer)
            
            # =========================================================
            # PHASE 3: EDGE CREATION
            # =========================================================
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
                        # Calc weight
                        if prev_id in layers_map:
                             comm_size_mb = layers_map[prev_id].output_bytes / (1024 * 1024)
                        else:
                             comm_size_mb = 0.0
                        G.add_edge(prev_id, curr_id, weight=comm_size_mb)
            
            # =========================================================
            # PHASE 4: CYCLE DETECTION
            # =========================================================
            if not nx.is_directed_acyclic_graph(G):
                try:
                    cycle = nx.find_cycle(G, orientation="original")
                    # cycle is a list of edges [(u, v, ...), ...]
                    cycle_path = []
                    for edge in cycle:
                        u = edge[0]
                        name = layers_map[u].name if u in layers_map else str(u)
                        cycle_path.append(name)
                    
                    # Close the loop for display
                    start_node = cycle[0][0]
                    start_name = layers_map[start_node].name if start_node in layers_map else str(start_node)
                    cycle_path.append(start_name)
                    
                    cycle_str = " -> ".join(cycle_path)
                    raise ValueError(f"Input graph contains a cycle: {cycle_str}")
                except nx.NetworkXNoCycle:
                    pass # logic error if is_directed_acyclic_graph says False but find_cycle finds nothing

        return G, layers_map
