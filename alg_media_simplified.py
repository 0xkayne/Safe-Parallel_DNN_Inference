import networkx as nx
from common import Partition, EPC_EFFECTIVE_MB, calculate_penalty, PAGING_BANDWIDTH_MB_PER_MS

class MEDIAAlgorithm:
    def __init__(self, G, layers_map, servers, bandwidth_mbps):
        self.G = G
        self.layers_map = layers_map
        self.servers = servers
        self.bandwidth_per_ms = (bandwidth_mbps / 8.0) / 1000.0
        
        # Fixing Heuristic: 
        # Partitioner assumes Split -> Comm. But Scheduler often does Split -> SameServer -> NoComm.
        # We weigh the communication cost by probability of it actually happening.
        self.COMM_WEIGHT = 1.0

    def run(self):
        # MEDIA Serial: Linearize the graph first to enforce serial execution
        topo_order_ids = list(nx.topological_sort(self.G))
        
        # Partitioning Strategy: Merge layers as long as it benefits us
        # Heuristic: Merge if (Memory < EPC) OR (Communication Cost > EPC Paging Penalty)
        # Since it's serial, we can just iterate linearly.
        
        partitions = []
        if not topo_order_ids:
            return partitions
            
        current_layers = [self.layers_map[topo_order_ids[0]]]
        
        for i in range(1, len(topo_order_ids)):
            node_id = topo_order_ids[i]
            layer = self.layers_map[node_id]
            prev_layer_id = topo_order_ids[i-1]
            
            # Calculate metrics for current partition candidate
            # Setup temp partition
            # Metrics:
            # 1. Memory if merged
            # 2. Communication if NOT merged (edge weight between prev and curr)
            
            curr_mem = sum(l.memory for l in current_layers)
            
            # Simple greedy merge check similar to standard MEDIA but strictly linear
            # If we merge:
            merged_mem = curr_mem + layer.memory
            
            # Check edge weight (comm cost if we split)
            # Find edge from ANY layer in current_layers to new layer (in DAG)
            edge_weight_mb = 0
            for existing_l in current_layers:
                if self.G.has_edge(existing_l.id, layer.id):
                    edge_weight_mb += self.G[existing_l.id][layer.id]['weight']
            
            comm_time = edge_weight_mb / self.bandwidth_per_ms
            
            # Execution time penalty estimation
            # Workload
            merged_work = sum(l.workload for l in current_layers) + layer.workload
            
            # This logic mimics 'merge_check' in the original code but simplified for linear scan.
            # We merge if:
            # 1. Total Mem <= EPC (Always merge to save comms)
            # 2. if Total Mem > EPC, only merge if EXEC_Penalty < Comm_Time
            
            should_merge = False
            if merged_mem <= EPC_EFFECTIVE_MB:
                should_merge = True
            else:
                # Calculate Overhead Time
                # Baseline time (fast)
                t_fast = merged_work 
                # Penalized time
                penalty_factor = calculate_penalty(merged_mem)
                t_slow = t_fast * penalty_factor
                penalty_delta = t_slow - t_fast
                
                # If penalty < communication cost saved * PROBABILITY, then we still merge
                if penalty_delta < comm_time * self.COMM_WEIGHT:
                    should_merge = True
            
            if should_merge:
                # print(f"    [MEDIA] Merging {layer.name} into partition. Mem: {merged_mem:.2f}MB > EPC")
                current_layers.append(layer)
            else:
                # if merged_mem > EPC_EFFECTIVE_MB:
                #     print(f"    [MEDIA] NOT merging {layer.name}. Penalty {penalty_delta:.2f}ms > Comm {comm_time:.2f}ms")
                # Finalize current
                partitions.append(Partition(len(partitions), current_layers))
                current_layers = [layer]
                
        if current_layers:
            partitions.append(Partition(len(partitions), current_layers))
            
        return partitions

    def schedule(self, partitions):
        """
        Round-Robin Scheduling:
        Partition i is assigned to server (i % n_servers).
        This forces network communication between consecutive partitions.
        
        Also includes SGX context switch paging overhead when partitions change.
        """
        if not partitions:
            return 0.0
        
        n_servers = len(self.servers)
        server_free_time = {s.id: 0.0 for s in self.servers}
        last_partition_info = {}
        
        for i, p in enumerate(partitions):
            # Round-robin: partition i → server i % n_servers
            assigned_server = self.servers[i % n_servers]
            
            # Communication cost (if previous partition was on different server)
            comm = 0.0
            paging_overhead = 0.0
            
            if i > 0:
                prev_info = last_partition_info[i - 1]
                prev_p = partitions[i - 1]
                
                if prev_info['server'] != assigned_server.id:
                    # Cross-server: network communication
                    vol = 0.0
                    for u in prev_p.layers:
                        for v in p.layers:
                            if self.G.has_edge(u.id, v.id):
                                vol += self.G[u.id][v.id]['weight']
                    comm = vol / self.bandwidth_per_ms
                else:
                    # Same server: SGX context switch paging overhead
                    # Swap out previous partition + swap in current partition
                    swap_out = min(prev_p.total_memory, EPC_EFFECTIVE_MB)
                    swap_in = min(p.total_memory, EPC_EFFECTIVE_MB)
                    paging_overhead = (swap_out + swap_in) / PAGING_BANDWIDTH_MB_PER_MS
                
                data_ready = prev_info['end'] + comm + paging_overhead
            else:
                data_ready = 0.0
            
            start_t = max(server_free_time[assigned_server.id], data_ready)
            
            penalty_factor = calculate_penalty(p.total_memory)
            exec_t = (p.total_workload * penalty_factor) / assigned_server.power_ratio
            
            finish_t = start_t + exec_t
            
            server_free_time[assigned_server.id] = finish_t
            last_partition_info[i] = {'end': finish_t, 'server': assigned_server.id}
        
        return max(info['end'] for info in last_partition_info.values())
