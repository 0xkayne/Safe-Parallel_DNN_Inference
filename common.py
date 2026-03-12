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

    Model: Gradual linear growth beyond EPC threshold.
    penalty = 1.0 + slope * (overflow / EPC)
    slope = 2.0 (calibrated to match MEDIA paper Eq.1 paging behavior)
    """
    if memory_mb <= EPC_EFFECTIVE_MB:
        return 1.0
    overflow_ratio = (memory_mb - EPC_EFFECTIVE_MB) / EPC_EFFECTIVE_MB
    return 1.0 + 2.0 * overflow_ratio

EPC_EFFECTIVE_MB = 93.0  # (128 - 35) MB

# SGX Paging Bandwidth (EPC ↔ DRAM with AES encryption)
# Typical value: ~1 GB/s = 1000 MB/s
DEFAULT_PAGING_BW_MBPS = 1000
PAGING_BANDWIDTH_MB_PER_MS = DEFAULT_PAGING_BW_MBPS / 1000.0  # 1.0 MB/ms

# Shared SGX Overhead Constants
PAGE_SIZE_KB = 4               # 4 KB per page
PAGE_FAULT_OVERHEAD_MS = 0.03  # 30 µs per page fault
ENCLAVE_ENTRY_EXIT_OVERHEAD_MS = 0.005  # 5 µs per ecall/ocall

# SGX Enclave Memory Realism Parameters
# Heap fragmentation: dlmalloc/jemalloc inside enclave cannot perfectly reuse freed
# virtual pages — different tensor sizes cause internal/external fragmentation.
# Typical overhead: 10-20% (conservative 15%, based on jemalloc benchmarks on DNN workloads)
HEAP_FRAGMENTATION_FACTOR = 1.15

# Framework runtime overhead: inference engine metadata, graph scheduler, thread stacks,
# crypto context (AES-GCM state), and SGX SDK internal structures.
# Measured range: 5-20 MB for ONNX Runtime / TFLite inside SGX (take conservative 10 MB)
FRAMEWORK_RUNTIME_OVERHEAD_MB = 10.0

# ============================================
# Distributed Multi-TEE Inference Parameters
# (Calibrated based on real-world measurements)
# ============================================

# 1. Cross-Node Network Latency (分场景配置)
RTT_DATACENTER_MS = 1.0        # 数据中心内 (同机架/跨机架)
RTT_EDGE_MS = 5.0              # 边缘网络 (园区 LAN) - 默认
RTT_WAN_MS = 30.0              # 广域边缘 (跨城市)
RTT_MS = RTT_EDGE_MS           # 当前使用的 RTT (可切换)
TLS_HANDSHAKE_OVERHEAD_MS = 10.0  # 首次 TLS 连接开销

# 2. SGX Remote Attestation (首次 Enclave 间通信)
# 基于 DCAP 本地验证模式 (30-150 ms 实测范围，取保守值)
ATTESTATION_OVERHEAD_MS = 80.0    # DCAP Quote 生成 + 验证 (保守)
SIGMA_HANDSHAKE_MS = 20.0         # SIGMA 密钥协商 (ECDH P-256 + ECDSA)
FIRST_HOP_OVERHEAD_MS = ATTESTATION_OVERHEAD_MS + SIGMA_HANDSHAKE_MS  # ~100 ms

# 3. Enclave Initialization (冷启动)
# 基于 SGX1/SGX2 实测数据 (20-100 ms 范围)
ENCLAVE_INIT_BASE_MS = 50.0       # ECREATE + EINIT 固定开销 (保守)
EADD_PER_PAGE_MS = 0.001          # 1 µs/page (实测典型值)

# 4. Simulation Mode Flags
ENABLE_RTT = True                 # 启用 RTT 网络延迟
ENABLE_ATTESTATION = False        # 启用 Remote Attestation 开销 (默认关闭)
ENABLE_ENCLAVE_INIT = False       # 启用 Enclave 初始化开销 (默认关闭)

def enclave_init_cost(enclave_size_mb):
    """计算 Enclave 初始化开销 (ms)
    
    基于 SGX 指令执行时间:
    - ECREATE: ~0.1 ms
    - EADD: ~1 µs/page (批量优化)
    - EINIT: 10-50 ms (签名验证 + Launch Enclave)
    """
    if not ENABLE_ENCLAVE_INIT:
        return 0.0
    num_pages = enclave_size_mb * 1024 / 4  # 4 KB/page
    return ENCLAVE_INIT_BASE_MS + num_pages * EADD_PER_PAGE_MS

def network_latency(data_mb, bandwidth_mbps, is_first_hop=False):
    """计算完整网络通信延迟 (ms)
    
    T_network = RTT + T_transmission + T_attestation (optional)
    """
    bandwidth_per_ms = (bandwidth_mbps / 8.0) / 1000.0  # MB/ms
    transmission_time = data_mb / bandwidth_per_ms if bandwidth_per_ms > 0 else 0
    
    rtt = RTT_MS if ENABLE_RTT else 0.0
    attestation = FIRST_HOP_OVERHEAD_MS if (is_first_hop and ENABLE_ATTESTATION) else 0.0
    
    return rtt + transmission_time + attestation

def hpa_cost(
    layer, 
    k: int, 
    bandwidth_mbps: float, 
    efficiency_gamma: float = 0.9,
    activation_split_ratio: float = 1.0,  # NEW: 激活切分比例 (1.0=完全切分, 0.0=完全复制)
    sync_probability: float = 0.5         # NEW: 同步概率 (0.5=双层摊销, 1.0=每层同步)
) -> float:
    """
    Compute HPA cost for splitting a layer into k parallel shards.
    
    Args:
        layer: DNNLayer object
        k: Parallelism degree (number of shards)
        bandwidth_mbps: Network bandwidth in Mbps
        efficiency_gamma: Parallel efficiency factor (default 0.9)
        activation_split_ratio: Fraction of activation memory that is split (default 1.0)
            - 1.0: Activation fully split (e.g., Column Parallel FC output)
            - 0.0: Activation fully replicated (e.g., BatchNorm requiring full statistics)
        sync_probability: Probability that this layer requires AllReduce sync (default 0.5)
            - 1.0: Always sync (isolated TP layer)
            - 0.5: Amortized sync (Megatron-style: 2 layers share 1 AllReduce)
            - 0.0: No sync (intermediate layer in TP group)
    
    Returns:
        float: Total cost in ms = compute + paging + sync
    
    Cost Model (Optimized):
        Cost(v, k) = T_comp/k^γ + Penalty_mem(M_shard) + T_sync(k) * P_sync
        
        Where M_shard = (M_weight/k) + M_activation * (1 - α + α/k)
              α = activation_split_ratio
    """
    # 1. Compute cost with efficiency factor
    t_comp_original = layer.workload  # ms
    t_comp = t_comp_original / (k ** efficiency_gamma)
    
    # 2. Memory penalty (FIXED: activation memory is also split)
    m_weight = layer.weight_memory + layer.bias_memory
    m_activation = layer.activation_memory
    
    # Calculate per-shard memory based on split strategy:
    # - Weights: always split equally
    # - Activation: split according to activation_split_ratio
    #   - If ratio=1.0: m_activation / k (fully split)
    #   - If ratio=0.0: m_activation (fully replicated)
    #   - General: m_activation * (1 - ratio) + m_activation * ratio / k
    m_activation_shard = m_activation * (1.0 - activation_split_ratio) + \
                         m_activation * activation_split_ratio / k
    m_split = (m_weight / k) + m_activation_shard
    
    penalty = calculate_penalty(m_split)
    
    # Separate paging costs into runtime penalty and init overhead
    # a) Runtime penalty: scaling factor on compute time (only if penalty > 1)
    t_runtime_penalty = (penalty - 1.0) * t_comp if penalty > 1.0 else 0.0
    
    # b) Init overhead: fixed cost based on memory size (optional)
    t_init = enclave_init_cost(m_split) if ENABLE_ENCLAVE_INIT else 0.0
    
    t_paging = t_runtime_penalty + t_init
    
    # 3. Sync cost (FIXED: apply sync_probability to amortize AllReduce)
    if k > 1:
        # Ring AllReduce: 2 * (k-1)/k * data_volume
        sync_bytes = layer.output_bytes * 2 * (k - 1) / k
        sync_mb = sync_bytes / (1024 * 1024)
        
        # Apply sync probability factor (default 0.5 for Megatron-style TP)
        t_sync = network_latency(sync_mb, bandwidth_mbps) * sync_probability
    else:
        t_sync = 0.0
    
    return t_comp + t_paging + t_sync

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
        """Calculate peak memory requirement considering activation liveness,
        workspace memory, heap fragmentation, and framework runtime overhead.

        Total = persistent + peak_activation * fragmentation + framework_overhead
        """
        if not self.layers:
            return 0.0

        # Persistent memory: weights + biases + encryption overhead
        # These must be loaded and stay in memory during the partition execution
        persistent_memory = sum(
            l.weight_memory + l.bias_memory + l.encryption_overhead
            for l in self.layers
        )

        # Dynamic activation memory peak (includes workspace transients)
        peak_activation = self._calculate_peak_activation()

        # Heap fragmentation: real allocators cannot perfectly reuse freed pages
        peak_activation *= HEAP_FRAGMENTATION_FACTOR

        # Framework runtime overhead: inference engine internals per enclave
        return persistent_memory + peak_activation + FRAMEWORK_RUNTIME_OVERHEAD_MB

    def get_static_memory(self):
        """Calculate implementation-static memory (Weights + Bias + Encryption Overhead)."""
        # This is the amount of data that must be securely loaded (Swap-In) from DRAM.
        # It does NOT include activations, which are generated at runtime.
        if not self.layers:
            return 0.0
        return sum(l.weight_memory + l.bias_memory + l.encryption_overhead for l in self.layers)

    def _calculate_peak_activation(self):
        """Calculate peak activation memory using DAG liveness analysis.

        Tracks which output tensors are "live" (still needed by a downstream
        layer in this partition) at each execution step.  An output becomes
        live when its producing layer executes and dies when all of its
        in-partition consumers have executed.

        Additionally models **workspace memory**: temporary buffers needed
        during a layer's computation (e.g., im2col for Conv, intermediate
        matmul products).  Workspace = activation_memory − output_mb; it is
        live only during that layer's execution step and freed immediately
        after.  This matches real DNN runtime behaviour where frameworks
        allocate scratch space, compute, then free before the next op.

        This models real SGX behaviour: heap allocators (dlmalloc / jemalloc)
        inside the enclave reuse freed virtual pages, so the EPC footprint
        equals the high-water mark of concurrently live tensors — not the
        cumulative sum.  EPC physical pages stay committed (no EREMOVE), but
        virtual-address reuse means no *additional* pages are needed once a
        tensor is freed and a same-sized tensor is allocated in its place.
        """
        if not self.layers:
            return 0.0

        layer_ids = set(l.id for l in self.layers)
        id_to_layer = {l.id: l for l in self.layers}

        # Count how many in-partition successors each layer has
        remaining_consumers = {}
        for lid in layer_ids:
            count = 0
            if self.dag is not None:
                for succ in self.dag.successors(lid):
                    if succ in layer_ids:
                        count += 1
            remaining_consumers[lid] = count

        # Simulate execution in topological order, tracking live set
        live_memory = 0.0  # MB currently live (persistent output tensors)
        peak = 0.0

        for layer in self.layers:  # already in topological order
            # This layer's output becomes live
            out_mb = layer.output_bytes / (1024 * 1024)
            live_memory += out_mb

            # Workspace memory: temporary buffers during this layer's execution
            # (e.g., im2col, intermediate matmul, BatchNorm statistics)
            # activation_memory captures the layer's full activation footprint;
            # subtract the output tensor to isolate the transient workspace.
            workspace_mb = max(0.0, layer.activation_memory - out_mb)

            # Peak check: during execution, both output + workspace are live
            peak = max(peak, live_memory + workspace_mb)
            # workspace is freed immediately after the layer finishes

            # Consume predecessors: decrement their remaining consumer count
            if self.dag is not None:
                for pred in self.dag.predecessors(layer.id):
                    if pred in layer_ids:
                        remaining_consumers[pred] -= 1
                        if remaining_consumers[pred] == 0:
                            # This predecessor's output is no longer needed
                            live_memory -= id_to_layer[pred].output_bytes / (1024 * 1024)

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

