import math
import networkx as nx
from common import Partition, EPC_EFFECTIVE_MB, calculate_penalty, ScheduleResult

class OCCAlgorithm:
    """
    OCC: Single-server Oblivious Context-switch Computation.
    Runs entire model on ONE server via multi-partition paging.
    """
    
    # SGX Paging Parameters (calibrated to realistic values)
    DEFAULT_PAGING_BW_MBPS = 1000
    PAGE_SIZE_KB = 4
    PAGE_FAULT_OVERHEAD_MS = 0.03
    ENCLAVE_ENTRY_EXIT_OVERHEAD_MS = 0.005
    
    def __init__(self, G, layers_map, servers, bandwidth_mbps):
        self.G = G
        self.layers_map = layers_map
        self.servers = servers
        self.bandwidth_mbps = bandwidth_mbps
        self.paging_bw_per_ms = self.DEFAULT_PAGING_BW_MBPS / 1000.0
    
    def run(self):
        topo_order = list(nx.topological_sort(self.G))
        partitions = []
        current_layers = []
        current_mem = 0.0
        
        for node_id in topo_order:
            layer = self.layers_map[node_id]
            
            # Try adding this layer to the current partition
            test_layers = current_layers + [layer]
            # Create temporary partition to calculate accurate peak memory
            test_partition = Partition(-1, test_layers, self.G)
            
            if test_partition.total_memory > EPC_EFFECTIVE_MB:
                if current_layers:
                    # Previous layers formed a valid partition, finalize it
                    partitions.append(Partition(len(partitions), current_layers, self.G))
                    current_layers = [layer]
                else:
                    # Single layer exceeds EPC, must accept it
                    current_layers = [layer]
            else:
                current_layers.append(layer)
        
        if current_layers:
            partitions.append(Partition(len(partitions), current_layers, self.G))
        return partitions
    
    def schedule(self, partitions):
        if not partitions:
            return ScheduleResult("OCC", 0.0, {}, [])
        
        best_server = max(self.servers, key=lambda s: s.power_ratio) if self.servers else None
        max_power_ratio = best_server.power_ratio if best_server else 1.0
        s_id = best_server.id if best_server else 0
        
        current_time = 0.0
        server_schedule = {s.id: [] for s in self.servers}

        for i, part in enumerate(partitions):
            # Universal SGX Paging Model: Every partition must be loaded (Cold Start / Context Switch)
            # CRITICAL FIX: Only load Weight + Bias (Static Data). Activations are generated runtime.
            swap_bytes_mb = part.get_static_memory()
            num_pages = int(math.ceil(swap_bytes_mb * 1024 / self.PAGE_SIZE_KB))
            paging_overhead = (num_pages * self.PAGE_FAULT_OVERHEAD_MS + swap_bytes_mb / self.paging_bw_per_ms)
            current_time += paging_overhead

            start_t = current_time
            current_time += self.ENCLAVE_ENTRY_EXIT_OVERHEAD_MS
            penalty = calculate_penalty(part.total_memory)
            exec_time = (part.total_workload * penalty) / max_power_ratio
            current_time += exec_time
            finish_t = current_time
            
            server_schedule[s_id].append({
                'start': start_t,
                'end': finish_t,
                'partition_id': part.id,
                'partition': part
            })
        
        return ScheduleResult("OCC", current_time, server_schedule, partitions)
