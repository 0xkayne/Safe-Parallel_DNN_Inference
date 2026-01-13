import networkx as nx
from common import Partition, EPC_EFFECTIVE_MB, calculate_penalty, PAGING_BANDWIDTH_MB_PER_MS

class DINAAlgorithm:
    def __init__(self, G, layers_map, servers, bandwidth_mbps):
        self.G = G
        self.layers_map = layers_map
        self.servers = servers
        self.bandwidth_mbps = bandwidth_mbps # Mbps
        # Convert Bandwidth to MB/ms for simulation
        # 1 Byte = 8 bits
        # 1 MB = 8 * 10^6 bits
        # MB/s = mbps / 8
        # MB/ms = (mbps / 8) / 1000
        self.bandwidth_per_ms = (bandwidth_mbps / 8.0) / 1000.0

    def run(self):
        # DINA: Strict Partitioning < EPC
        # Simple greedy strategy following topological sort
        
        topo_order = list(nx.topological_sort(self.G))
        partitions = []
        
        current_part_layers = []
        current_mem = 0.0
        
        for node_id in topo_order:
            layer = self.layers_map[node_id]
            
            # If adding this layer exceeds EPC, finalize current partition and start new
            if current_mem + layer.memory > EPC_EFFECTIVE_MB:
                if current_part_layers:
                    partitions.append(Partition(len(partitions), current_part_layers))
                
                # Start new partition
                current_part_layers = [layer]
                current_mem = layer.memory
                
                # Edge case: Single layer > EPC (Should theoretically not happen in DINA hypothesis, but we must handle it)
                # If single layer > EPC, DINA forces it to be its own partition (and suffers penalty later)
            else:
                current_part_layers.append(layer)
                current_mem += layer.memory
        
        # Add last partition
        if current_part_layers:
            partitions.append(Partition(len(partitions), current_part_layers))
            
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

