import networkx as nx

# Global Constants (Defaults, can be overridden during experiment)
EPC_EFFECTIVE_MB = 93.0  # (128 - 35) MB
# Dynamic SGX EPC-Overflow Initialization Penalty Model
# Based on ICDCS'22 and MobiCom'19 measurements
# This penalty represents ONLY the one-time initialization cost when memory > EPC
# (EINIT, metadata reconstruction, initial page-fault burst)
# Per-page paging costs are handled separately in the swap model
def calculate_penalty(memory_mb):
    """
    Calculate execution penalty for partitions exceeding EPC.
    
    Args:
        memory_mb: Total memory requirement of the partition in MB
    
    Returns:
        float: Penalty multiplier (1.0 = no penalty)
    
    Breakdown:
    - memory <= EPC: No overflow, penalty = 1.0
    - EPC < memory <= 2*EPC: First overflow, large initialization cost ≈ 4.5×
    - memory > 2*EPC: Additional linear growth ≈ 0.25× per extra EPC
    
    Note: This does NOT include per-page paging costs (those are in swap_time).
    """
    if memory_mb <= EPC_EFFECTIVE_MB:
        return 1.0
    elif memory_mb <= 2 * EPC_EFFECTIVE_MB:
        # First EPC overflow – large one-time EINIT + metadata cost
        # Empirical value from ICDCS'22 measurements
        return 4.5
    else:
        # Additional EPCs – modest linear growth for continued paging
        extra_ratio = (memory_mb - 2 * EPC_EFFECTIVE_MB) / EPC_EFFECTIVE_MB
        return 4.5 + 0.25 * extra_ratio

EPC_EFFECTIVE_MB = 93.0  # (128 - 35) MB

# SGX Paging Bandwidth (EPC ↔ DRAM with AES encryption)
# Typical value: ~1 GB/s = 1000 MB/s
DEFAULT_PAGING_BW_MBPS = 1000
PAGING_BANDWIDTH_MB_PER_MS = DEFAULT_PAGING_BW_MBPS / 1000.0  # 1.0 MB/ms

# Shared SGX Overhead Constants
PAGE_SIZE_KB = 4               # 4 KB per page
PAGE_FAULT_OVERHEAD_MS = 0.03  # 30 µs per page fault
ENCLAVE_ENTRY_EXIT_OVERHEAD_MS = 0.005  # 5 µs per ecall/ocall

# CPU Benchmark Scores (PassMark) for Heterogeneous Compute Scaling
# Baseline: Intel Xeon Platinum 8380 (Ice Lake) ~ 62318
SERVER_TYPES = {
    "Xeon_IceLake": 1.00, # Baseline (Power Ratio = 1.0)
    "Celeron G4930": 0.11,     # Mid-range desktop
    "i5-6500": 0.93,      # Entry-level desktop
    "i3-10100": 1.03,       # Legacy (Estimated as i5-6500)
    "i5-11600": 1.97   # Low-power edge node
}

# Baseline: Intel Xeon Platinum 8380 (Ice Lake) ~ 1.00
BASELINE_COMPUTE = 1.00

class DNNLayer:
    def __init__(self, layer_id, name, memory, cpu_time, enclave_time, output_bytes, execution_mode='Unknown',
                 weight_memory=0.0, bias_memory=0.0, activation_memory=0.0, encryption_overhead=0.0):
        self.id = layer_id
        self.name = name
        self.memory = memory          # MB (Total memory footprint)
        
        # Granular memory components (MB)
        self.weight_memory = weight_memory
        self.bias_memory = bias_memory
        self.activation_memory = activation_memory
        self.encryption_overhead = encryption_overhead
        
        self.cpu_time = cpu_time      # ms
        self.enclave_time = enclave_time # ms
        self.output_bytes = output_bytes # Bytes
        self.execution_mode = execution_mode  # Execution mode (e.g., 'Enclave', 'CPU')
        
        # Workload is treated as measured execution time
        self.workload = enclave_time 

    def __repr__(self):
        return f"Layer({self.name}, mem={self.memory:.2f})"

class Partition:
    def __init__(self, partition_id, layers, dag=None):
        self.id = partition_id
        self.dag = dag # NetworkX DAG for dependency tracking
        
        # Ensure layers are in topological order for correct memory simulation
        if self.dag is not None and layers:
            try:
                # 1. Get IDs from input layers
                layer_ids = [l.id for l in layers]
                # 2. Create induced subgraph to respect dependencies
                subgraph = self.dag.subgraph(layer_ids)
                # 3. Sort IDs topologically
                sorted_ids = list(nx.topological_sort(subgraph))
                # 4. Reconstruct layers list in order
                id_to_layer = {l.id: l for l in layers}
                self.layers = [id_to_layer[nid] for nid in sorted_ids]
            except Exception:
                # Fallback if sorting fails (should not happen on DAG)
                self.layers = layers
        else:
            self.layers = layers

        # Calculate peak memory dynamically if DAG is provided and granular info exists
        # Otherwise fall back to sum (backward compatibility)
        if self.dag is not None and any(l.weight_memory > 0 for l in self.layers):
            self.total_memory = self._calculate_peak_memory()
        else:
            self.total_memory = sum(l.memory for l in self.layers)
            
        # Simple sum of workloads
        self.total_workload = sum(l.workload for l in self.layers)
        
        self.assigned_server = None
        self.start_time = 0.0
        self.finish_time = 0.0
        self.ready_time = 0.0

    def _calculate_peak_memory(self):
        """Calculate peak memory requirement considering activation liveness."""
        if not self.layers:
            return 0.0
        
        # Persistent memory: weights + biases + encryption overhead
        # These must be loaded and stay in memory during the partition execution
        persistent_memory = sum(
            l.weight_memory + l.bias_memory + l.encryption_overhead 
            for l in self.layers
        )
        
        # Dynamic activation memory peak
        peak_activation = self._calculate_peak_activation()
        
        return persistent_memory + peak_activation

    def get_static_memory(self):
        """Calculate implementation-static memory (Weights + Bias + Encryption Overhead)."""
        # This is the amount of data that must be securely loaded (Swap-In) from DRAM.
        # It does NOT include activations, which are generated at runtime.
        if not self.layers:
            return 0.0
        return sum(l.weight_memory + l.bias_memory + l.encryption_overhead for l in self.layers)

    def _calculate_peak_activation(self):
        """Calculate peak activation memory usage during execution."""
        peak = 0.0
        
        # Fast lookup set for layer IDs in this partition
        partition_layer_ids = {l.id for l in self.layers}
        
        # Map layer ID to its index in self.layers for order checking
        layer_indices = {l.id: i for i, l in enumerate(self.layers)}
        
        # We verify memory state at the END of each layer's execution
        # For each step i (after layer i finishes):
        for i in range(len(self.layers)):
            current_activation = 0.0
            
            # Check all layers processed so far (0 to i) to see if their activation is still needed
            for j in range(i + 1):
                layer = self.layers[j]
                
                # Check if layer j's output is needed by any future layer
                is_needed = False
                
                for succ_id in self.dag.successors(layer.id):
                    if succ_id in partition_layer_ids:
                        # Successor is in the same partition
                        succ_idx = layer_indices[succ_id]
                        # If successor hasn't executed yet (index > i), we need to keep activation
                        if succ_idx > i:
                            is_needed = True
                            # Optimization: Just need one reason to keep it
                            break 
                    else:
                        # Successor is outside the partition (or on another node)
                        # We must assume it's needed until partition finishes (or transferred)
                        # In this simple model, we assume it consumes memory until end of partition execution
                        is_needed = True
                        break
                
                if is_needed:
                    current_activation += layer.activation_memory
            
            # Update peak found so far
            peak = max(peak, current_activation)
            
        return peak

    def __repr__(self):
        return f"Part#{self.id}(n={len(self.layers)}, mem={self.total_memory:.1f})"

    def __deepcopy__(self, memo):
        """Custom deepcopy to avoid copying the DAG (which is read-only)."""
        import copy
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k == 'dag':
                # Shallow copy DAG reference - crucial for performance!
                setattr(result, k, self.dag)
            else:
                setattr(result, k, copy.deepcopy(v, memo))
        return result

class Server:
    def __init__(self, server_id, server_type="Xeon_IceLake"):
        self.id = server_id
        self.server_type = server_type
        
        # Calculate power ratio relative to baseline
        if server_type in SERVER_TYPES:
            self.compute_score = SERVER_TYPES[server_type]
        else:
            self.compute_score = BASELINE_COMPUTE
            
        self.power_ratio = self.compute_score / BASELINE_COMPUTE
        
        # Schedule entries: (start_time, end_time, partition_id, partition_obj)
        self.schedule = []
        self.assigned_memory = 0.0 

    def add_event(self, start, end, partition):
        self.schedule.append({
            'start': start,
            'end': end,
            'partition_id': partition.id,
            'partition': partition
        })

    def __repr__(self):
        return f"Server#{self.id}({self.server_type}, x{self.power_ratio:.2f})"

class ScheduleResult:
    """
    Structured result of a scheduling algorithm.
    """
    def __init__(self, algorithm_name, latency, server_schedules, partitions):
        self.algorithm_name = algorithm_name
        self.latency = latency
        self.server_schedules = server_schedules # Map: server_id -> list of events
        self.partitions = partitions # All partitions formed

