import networkx as nx
from .common import (Partition, EPC_EFFECTIVE_MB, calculate_penalty, ScheduleResult,
                     enclave_init_cost, ENABLE_ENCLAVE_INIT)

class OCCAlgorithm:
    """
    OCC: Occlumency-style single-server inference.
    Weights stored in UNPROTECTED memory, loaded on-demand via OCALL (DDR memcpy ~10 GB/s).
    EPC contains only activations + ring buffer — no EPC paging for weights.
    Three-thread pipeline: loading + hash-checking + inference run in parallel.
    """

    # DDR memcpy bandwidth for weight loading via OCALL (conservative 10 GB/s)
    WEIGHT_COPY_BW_MB_PER_MS = 10.0
    # HMAC-SHA256 hash verification bandwidth in SGX enclave (~0.5 GB/s = 0.5 MB/ms)
    HASH_VERIFY_BW_MB_PER_MS = 0.5
    # EPC ring buffer overhead for weight staging (fixed per partition)
    RING_BUFFER_EPC_MB = 20.0
    
    def __init__(self, G, layers_map, servers, bandwidth_mbps):
        self.G = G
        self.layers_map = layers_map
        self.servers = servers
        self.bandwidth_mbps = bandwidth_mbps
    
    def run(self):
        """
        Partition layers sequentially such that each partition's activation
        memory fits within the EPC budget.

        Per the Occlumency paper (MobiCom'19 §4-6):
        - Weights are stored in UNPROTECTED memory outside EPC
        - EPC holds only: activations (feature maps) + ring buffer (~20 MB)
        - Available EPC for activations = EPC_EFFECTIVE_MB - RING_BUFFER_EPC_MB
        """
        topo_order = list(nx.topological_sort(self.G))
        partitions = []
        current_layers = []

        # EPC budget for activations only (ring buffer reserves space for weight staging)
        activation_epc_budget = EPC_EFFECTIVE_MB - self.RING_BUFFER_EPC_MB

        for node_id in topo_order:
            layer = self.layers_map[node_id]

            # Try adding this layer to the current partition
            test_layers = current_layers + [layer]
            # Create temporary partition to calculate accurate peak activation memory
            test_partition = Partition(-1, test_layers, self.G)

            # Check activation-only peak memory against EPC budget
            # Weights are outside EPC, so only activations matter for partitioning
            peak_activation = test_partition._calculate_peak_activation()

            if peak_activation > activation_epc_budget:
                if current_layers:
                    # Previous layers formed a valid partition, finalize it
                    partitions.append(Partition(len(partitions), current_layers, self.G))
                    current_layers = [layer]
                else:
                    # Single layer exceeds EPC, must accept it (will use partitioned convolution)
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
        
        # Enclave Initialization Overhead (Cold Start, one-time per inference)
        if ENABLE_ENCLAVE_INIT:
            # OCC runs on single server, init cost = cost of loading the enclave binary
            enclave_binary_size_mb = 10.0  # Typical DNN inference enclave size
            current_time += enclave_init_cost(enclave_binary_size_mb)
        
        server_schedule = {s.id: [] for s in self.servers}

        for part in partitions:
            # Per-layer 3-thread pipeline (Occlumency paper §4-6):
            # For each layer, Thread1 loads weights from untrusted memory,
            # Thread2 verifies HMAC-SHA256, Thread3 computes using verified weights.
            # The bottleneck per layer is max(load, hash, compute).
            # A layer's weights must be fully verified before its computation begins,
            # so heavy-weight layers (e.g. fc1) cannot hide hash cost behind other layers.
            partition_time = 0.0
            for layer in part.layers:
                layer_weight_mb = layer.weight_memory
                layer_load_time = layer_weight_mb / self.WEIGHT_COPY_BW_MB_PER_MS
                layer_hash_time = layer_weight_mb / self.HASH_VERIFY_BW_MB_PER_MS
                layer_exec_time = layer.workload / max_power_ratio
                partition_time += max(layer_exec_time, layer_load_time, layer_hash_time)

            start_t = current_time
            current_time += partition_time
            server_schedule[s_id].append({
                'start': start_t,
                'end': current_time,
                'partition_id': part.id,
                'partition': part
            })
        
        return ScheduleResult("OCC", current_time, server_schedule, partitions)
