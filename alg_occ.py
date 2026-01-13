import networkx as nx
from common import Partition, EPC_EFFECTIVE_MB, calculate_penalty

class OCCAlgorithm:
    """
    OCC: Single-server Oblivious Context-switch Computation.
    Runs entire model on ONE server via multi-partition paging.
    """
    
    # Paging bandwidth: EPC ↔ DRAM crypto copy (≈ 2 GB/s typical)
    PAGING_BANDWIDTH_MBPS = 2000  # 2 GB/s = 2000 MB/s
    
    def __init__(self, G, layers_map, servers, bandwidth_mbps):
        self.G = G
        self.layers_map = layers_map
        self.servers = servers
        self.bandwidth_mbps = bandwidth_mbps
        # Paging bandwidth in MB/ms (2 GB/s = 2.0 MB/ms)
        self.paging_bw_per_ms = self.PAGING_BANDWIDTH_MBPS / 1000.0
    
    def run(self):
        """
        Partition model so each partition fits in EPC.
        Greedy bin-packing following topological order.
        """
        topo_order = list(nx.topological_sort(self.G))
        partitions = []
        
        current_layers = []
        current_mem = 0.0
        
        for node_id in topo_order:
            layer = self.layers_map[node_id]
            
            if current_mem + layer.memory > EPC_EFFECTIVE_MB:
                if current_layers:
                    partitions.append(Partition(len(partitions), current_layers))
                current_layers = [layer]
                current_mem = layer.memory
            else:
                current_layers.append(layer)
                current_mem += layer.memory
        
        if current_layers:
            partitions.append(Partition(len(partitions), current_layers))
        
        return partitions
    
    def schedule(self, partitions):
        """
        Single-server serial execution with paging overhead between partitions.
        
        Paging overhead per switch:
          swap_time = (prev_partition.memory + next_partition.memory) / paging_bw
        
        Total = Σ exec_time + (n-1) * avg_swap_time
        """
        if not partitions:
            return 0.0
        
        total_time = 0.0
        
        for i, part in enumerate(partitions):
            # Execution time (with potential penalty if single layer > EPC)
            penalty = calculate_penalty(part.total_memory)
            exec_time = part.total_workload * penalty
            total_time += exec_time
            
            # Paging overhead to next partition
            if i < len(partitions) - 1:
                next_part = partitions[i + 1]
                # Swap out current partition, swap in next
                swap_bytes_mb = part.total_memory + next_part.total_memory
                swap_time = swap_bytes_mb / self.paging_bw_per_ms
                total_time += swap_time
        
        return total_time
