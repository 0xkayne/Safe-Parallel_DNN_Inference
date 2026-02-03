import networkx as nx
import math
from common import (Partition, EPC_EFFECTIVE_MB, calculate_penalty, PAGING_BANDWIDTH_MB_PER_MS,
                    PAGE_SIZE_KB, PAGE_FAULT_OVERHEAD_MS, ENCLAVE_ENTRY_EXIT_OVERHEAD_MS,
                    ScheduleResult, network_latency)

class DINAAlgorithm:
    def __init__(self, G, layers_map, servers, bandwidth_mbps):
        self.G = G
        self.layers_map = layers_map
        self.servers = servers
        self.bandwidth_mbps = bandwidth_mbps
        self.bandwidth_per_ms = (bandwidth_mbps / 8.0) / 1000.0

    def run(self):
        topo_order = list(nx.topological_sort(self.G))
        partitions = []
        current_part_layers = []
        current_mem = 0.0
        
        for node_id in topo_order:
            layer = self.layers_map[node_id]
            
            # Try adding this layer to the current partition
            test_layers = current_part_layers + [layer]
            # Create temporary partition to calculate accurate peak memory
            test_partition = Partition(-1, test_layers, self.G)
            
            if test_partition.total_memory > EPC_EFFECTIVE_MB:
                if current_part_layers:
                     partitions.append(Partition(len(partitions), current_part_layers, self.G))
                     current_part_layers = [layer]
                else:
                    current_part_layers = [layer]
            else:
                current_part_layers.append(layer)
                
        if current_part_layers:
            partitions.append(Partition(len(partitions), current_part_layers, self.G))
        return partitions

    def schedule(self, partitions):
        if not partitions:
            return ScheduleResult("DINA", 0.0, {}, [])
        
        n_servers = len(self.servers)
        server_free_time = {s.id: 0.0 for s in self.servers}
        server_schedule = {s.id: [] for s in self.servers}
        last_partition_info = {}
        
        n_servers = len(self.servers)
        for i, p in enumerate(partitions):
            best_server = None
            best_finish_t = float('inf')
            final_start_t = 0.0
            
            # Universal SGX Paging Model
            swap_bytes_mb = p.get_static_memory()
            num_pages = math.ceil(swap_bytes_mb * 1024 / PAGE_SIZE_KB)
            paging_overhead = (num_pages * PAGE_FAULT_OVERHEAD_MS + 
                                swap_bytes_mb / PAGING_BANDWIDTH_MB_PER_MS + 
                                ENCLAVE_ENTRY_EXIT_OVERHEAD_MS)
            
            prev_end = 0.0
            prev_server_id = -1
            if i > 0:
                prev_info = last_partition_info[i - 1]
                prev_end = prev_info['end']
                prev_server_id = prev_info['server']

            for s in self.servers:
                # CONSTRAINT: Must switch server after every partition to force network overhead
                if i > 0 and s.id == prev_server_id and len(self.servers) > 1:
                    continue
                    
                comm = 0.0
                if i > 0 and prev_server_id != s.id:
                    prev_p = partitions[i - 1]
                    vol = 0.0
                    for u in prev_p.layers:
                        for v in p.layers:
                            if self.G.has_edge(u.id, v.id):
                                vol += self.G[u.id][v.id]['weight']
                    vol_mb = vol / (1024 * 1024)  # Convert bytes to MB
                    comm = network_latency(vol_mb, self.bandwidth_mbps, is_first_hop=(i == 1))
                
                start_loading = max(server_free_time[s.id], prev_end + comm)
                start_exec = start_loading + paging_overhead
                
                penalty_factor = calculate_penalty(p.total_memory)
                exec_t = (p.total_workload * penalty_factor) / s.power_ratio
                finish_t = start_exec + exec_t
                
                if finish_t < best_finish_t:
                    best_finish_t = finish_t
                    best_server = s
                    final_start_t = start_exec

            # Assign to best alternative server
            server_free_time[best_server.id] = best_finish_t
            server_schedule[best_server.id].append({
                'start': final_start_t,
                'end': best_finish_t,
                'partition_id': p.id,
                'partition': p
            })
            last_partition_info[i] = {'end': best_finish_t, 'server': best_server.id}
        
        total_latency = max(info['end'] for info in last_partition_info.values())
        return ScheduleResult("DINA", total_latency, server_schedule, partitions)
