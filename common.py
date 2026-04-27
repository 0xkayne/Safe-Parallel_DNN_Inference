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

# ═══════════════════════════════════════════════════════════════════════════════
# Unified Weights-Outside-EPC Memory Model (Occlumency, MobiCom'19 §4-6)
# ═══════════════════════════════════════════════════════════════════════════════
# All methods share this model: weights stored encrypted in untrusted DRAM,
# loaded on-demand into EPC via OCALL with HMAC integrity verification.
# EPC holds only activations + ring buffer for weight staging.

# DDR memcpy bandwidth for OCALL weight loading (conservative 10 GB/s)
DDR_COPY_BW_MB_PER_MS = 10.0

# HMAC-SHA256 verification bandwidth in SGX enclave (~0.5 GB/s)
# This is the software SHA-256 bottleneck — dominant for weight-heavy FC layers
HMAC_VERIFY_BW_MB_PER_MS = 0.5

# EPC ring buffer for weight staging (OCC paper Fig.9)
RING_BUFFER_EPC_MB = 20.0


def partition_exec_cost(partition, server_power_ratio,
                         ddr_bw=DDR_COPY_BW_MB_PER_MS,
                         hmac_bw=HMAC_VERIFY_BW_MB_PER_MS,
                         ring_buf=RING_BUFFER_EPC_MB):
    """Compute execution cost of a partition under the unified weights-outside-EPC model.

    Three-thread pipeline (OCC §4-6):
      Thread 1 — DDR load:   OCALL memcpy(untrusted DRAM → EPC ring buffer)
      Thread 2 — HMAC check: integrity verification in enclave
      Thread 3 — Compute:    matrix multiply / convolution

    Per-layer bottleneck: max(T_compute, T_load, T_hmac)
    Paging penalty is charged on activation memory ONLY (weights are outside EPC).

    Args:
        partition: Partition object
        server_power_ratio: Compute power ratio of the target server
        ddr_bw: DDR copy bandwidth in MB/ms
        hmac_bw: HMAC verification bandwidth in MB/ms
        ring_buf: EPC ring buffer size in MB

    Returns:
        (eff_time_ms, load_time_ms, hash_time_ms, penalty):
          eff_time_ms = max(T_compute, T_load, T_hmac) per layer, summed over layers
          load_time_ms = total DDR copy time (before pipeline overlap)
          hash_time_ms = total HMAC time (before pipeline overlap)
          penalty = paging penalty multiplier (≥1.0, activation-only)
    """
    weight_mb = partition.get_static_memory()
    peak_act = partition.total_memory - weight_mb

    # Paging penalty: activation memory only (weights are outside EPC)
    penalty = calculate_penalty(peak_act + ring_buf)

    total_exe = 0.0
    total_load = 0.0
    total_hash = 0.0

    for layer in partition.layers:
        t_comp = layer.workload / server_power_ratio
        t_load = layer.weight_memory / ddr_bw
        t_hash = layer.weight_memory / hmac_bw
        # Pipeline: bottleneck is the slowest of the three threads
        total_exe += max(t_comp, t_load, t_hash)
        total_load += t_load
        total_hash += t_hash

    return total_exe, total_load, total_hash, penalty

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

def is_conv_layer(layer) -> bool:
    """Determine if a layer uses filter parallelism (Conv) vs column parallelism (FC).

    Conv layers use filter parallelism: each shard computes independent output
    channels, requiring only AllGather (concatenation) for synchronisation.
    FC layers use column parallelism: each shard computes partial sums,
    requiring AllReduce (reduce-scatter + all-gather) for synchronisation.
    AllGather costs (k-1)/k × output_bytes; AllReduce costs 2(k-1)/k × output_bytes.

    Falls back to name-based heuristics when the CSV lacks a 'type' column.
    """
    lt = (layer.layer_type or '').lower()
    if lt:
        if 'conv' in lt:
            return True
        if lt in ('linear', 'matmul'):
            return False
    # Infer from name for datasets without type column (e.g. InceptionV3)
    name = layer.name.lower()
    if any(p in name for p in ('conv', '1x1', '3x3', '5x5', '7x7', 'dwconv')):
        return True
    if any(p in name for p in ('fc', 'linear', 'proj', 'dense', 'matmul', 'ffn')):
        return False
    return False  # Default: FC (conservative — AllReduce has higher cost)


def hpa_cost(
    layer,
    k: int,
    bandwidth_mbps: float,
    efficiency_gamma: float = 0.9,
    activation_split_ratio: float = None,  # None = auto-infer by layer type
    sync_probability: float = None,        # None = use paper formula (no amortization)
    avg_power: float = 1.0                 # M3: expected compute power π̄
) -> float:
    """
    Compute HPA cost for splitting a layer into k parallel shards.

    Args:
        layer: DNNLayer object
        k: Parallelism degree (number of shards)
        bandwidth_mbps: Network bandwidth in Mbps
        efficiency_gamma: Parallel efficiency factor (default 0.9)
        activation_split_ratio: Fraction of activation memory that is split
            - None (default): auto-infer — Conv→1.0, FC→0.0
            - 1.0: Activation fully split (e.g., spatial parallelism for Conv)
            - 0.0: Activation fully replicated (e.g., Column Parallel FC input)
        sync_probability: DEPRECATED — kept for backward compat, ignored when None
        avg_power: Expected compute power ratio π̄ (default 1.0, backward compat)

    Returns:
        float: Total cost in ms = compute + paging + sync

    Cost Model (§4.4.2):
        Cost(v, k) = T_exec_shard + T_sync
        T_exec_shard = (W_v / k^γ) / π̄ · Φ(M_v/k, μ̄)
        T_sync = RTT · (k-1) + D_sync / (β̄/8)       [§4.4.2]

        Sync primitive depends on layer type:
            Conv → AllGather (filter parallelism):   (k-1)/k × output_bytes
            FC   → AllReduce (column parallelism): 2(k-1)/k × output_bytes
    """
    # M2e: Auto-infer activation_split_ratio by layer type
    if activation_split_ratio is None:
        activation_split_ratio = 1.0 if is_conv_layer(layer) else 0.0

    # 1. Compute cost with efficiency factor + expected power (§4.4.2)
    t_comp_original = layer.workload  # ms
    t_comp = t_comp_original / (k ** efficiency_gamma) / avg_power  # M3: / π̄

    # 2. Memory penalty (per-shard model)
    m_weight = layer.weight_memory + layer.bias_memory
    m_activation = layer.activation_memory

    # Calculate per-shard memory based on split strategy:
    # - Weights: always split equally
    # - Activation: split according to activation_split_ratio
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

    # 3. Sync cost — Ring collective model (§4.4.2):
    #    Ring uses one-way hops between adjacent nodes, latency per hop = RTT/2.
    #    AllGather  (1 phase):  T_sync = (RTT/2)·(k-1) + D_sync/β
    #    AllReduce  (2 phases): T_sync = RTT·(k-1)     + D_sync/β
    if k > 1:
        if is_conv_layer(layer):
            # Filter parallelism → AllGather (1 phase, concatenate output channels)
            sync_bytes = layer.output_bytes * (k - 1) / k
            ring_latency = (RTT_MS / 2.0) * (k - 1)
        else:
            # Column parallelism → AllReduce (2 phases, reduce-scatter + all-gather)
            sync_bytes = layer.output_bytes * 2 * (k - 1) / k
            ring_latency = RTT_MS * (k - 1)
        sync_mb = sync_bytes / (1024 * 1024)

        bandwidth_per_ms = (bandwidth_mbps / 8.0) / 1000.0
        t_transmission = sync_mb / bandwidth_per_ms if bandwidth_per_ms > 0 else 0.0
        t_sync = ring_latency + t_transmission
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
                 weight_memory=0.0, bias_memory=0.0, activation_memory=0.0, encryption_overhead=0.0,
                 layer_type=''):
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
        self.layer_type = layer_type  # e.g., 'Conv2d', 'Linear', 'MatMul'

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

