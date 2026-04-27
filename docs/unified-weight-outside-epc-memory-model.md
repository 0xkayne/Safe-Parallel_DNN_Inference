# 统一 Weights-Outside-EPC 内存模型设计文档

**版本**: v1  
**日期**: 2026-04-27  
**目的**: 为论文实验对比建立统一、物理可实现、安全性等价的内存成本模型，消除当前四种方法因内存模型差异导致的不公平对比。

---

## 1. 问题陈述

当前四种方法在内存/换页成本模型上存在不一致：

| 方法 | 权重位置 | paging penalty 参数 | 权重加载开销 |
|------|---------|-------------------|------------|
| OCC | EPC 外 | 无 penalty（权重不占 EPC） | DDR copy + HMAC pipeline |
| DINA | EPC 内 | `total_memory`（权重 + 激活） | 无 |
| MEDIA | EPC 内 | `_sum_memory(layers)`（权重 + 激活） | 无 |
| Ours | 不一致 | schedule() 用 `peak_activation`，priority 用 `total_memory` | DDR copy（缺少 HMAC） |

**核心问题**：OCC 享有"权重免 paging"特权的唯一原因是它把权重放在 EPC 外。但 DINA/MEDIA/Ours 同样可以实现这一策略——这不是 OCC 的专利，而是 SGX OCALL 机制的通用能力。当前对比混合了"分区策略"和"内存管理实现选择"两个独立变量，导致低带宽下 MEDIA/Ours 必然劣于 OCC，**这不是分区策略的问题，是内存模型的不公平负担**。

---

## 2. 统一模型定义

### 2.1 核心约束

所有方法在统一模型下遵循相同的物理约束：

```
EPC 有效容量: 93 MB (128 MB 物理 - 35 MB SGX 保留)
EPC 内驻留数据 (持久): 无
EPC 内驻留数据 (运行时):
  - 当前活跃的 activation tensors (由 DAG liveness 决定的高水位)
  - Ring buffer 用于权重 staging (~20 MB)
  - Framework runtime overhead (~10 MB)
  - × heap fragmentation factor (1.15×)

权重存放: EPC 外（不可信 DRAM），AES-GCM 加密存储
权重加载: OCALL DDR memcpy → HMAC 完整性校验 → (AES-GCM 解密，HW 加速可忽略)
激活值存放: EPC 内（MEE 加密 + 完整性保护）
```

### 2.2 Per-Partition 执行成本模型

每个 partition 在服务器上的执行时间：

```
T_partition = Σ over layers { max( T_compute, T_load_ddr, T_hmac ) }
              + T_paging(peak_activation + ring_buffer)

其中:
  T_compute   = layer.workload / (server.power_ratio × k^γ)          [单 shard 计算]
  T_load_ddr  = weight_bytes / DDR_COPY_BW                           [DDR memcpy]
  T_hmac      = weight_bytes / HMAC_VERIFY_BW                        [完整性校验]
  T_paging(m) = (calculate_penalty(m) - 1.0) × T_compute             [仅激活超 EPC 时]

关键: 权重不进入 calculate_penalty()。分区间不可 pipeline 重叠（后一分区依赖前一分区的输出 activation）。
```

### 2.3 常数标定

| 常数 | 值 | 来源 |
|------|-----|------|
| `EPC_EFFECTIVE_MB` | 93 MB | 128-35, SGX 架构文档 |
| `RING_BUFFER_EPC_MB` | 20 MB | OCC 论文 Fig.9 ring buffer 配置 |
| `FRAMEWORK_RUNTIME_OVERHEAD_MB` | 10 MB | ONNX Runtime / TFLite SGX 实测 |
| `HEAP_FRAGMENTATION_FACTOR` | 1.15× | jemalloc DNN workload benchmark |
| `DDR_COPY_BW_MB_PER_MS` | 10.0 MB/ms (≈10 GB/s) | DDR4-3200 实测（OCC 论文保守值） |
| `HMAC_VERIFY_BW_MB_PER_MS` | 0.5 MB/ms (≈0.5 GB/s) | SGX enclave 内软件 SHA-256 实测（OCC 论文） |
| `ENCLAVE_ENTRY_EXIT_OVERHEAD_MS` | 0.005 ms | SGX EENTER/EEXIT 指令周期 |

### 2.4 为什么 AES-GCM 解密开销可忽略

权重在不可信 DRAM 中以 AES-GCM 加密存储。但解密使用 Intel AES-NI 硬件指令，吞吐量约 13 GB/s，远超 DDR 带宽。DDR memcpy (10 GB/s) 和 HMAC (0.5 GB/s) 构成瓶颈后，AES-GCM 解密在流水线中完全被覆盖，不增加有效延迟。**这是工程事实，不是建模简化。**

---

## 3. 安全性论证

### 3.1 Threat Model（SGX 标准）

- **可信**: CPU 封装 + Enclave 代码/数据（EPC 内）
- **不可信**: OS/Hypervisor、DRAM、网络、其他 enclave
- **不在范围内的威胁**: 侧信道攻击（OCC/DINA/MEDIA 均有此问题，正交研究）

### 3.2 各类数据的安全保障

#### 权重（EPC 外）

```
存储: AES-GCM 加密(权重, key=SGX sealing key)
      → 不可信 OS 无法读取明文权重
      → 不可信 OS 无法篡改（GCM 认证标签，篡改立即可检测）

加载流程:
  1. OCALL 发起 DDR memcpy: 加密权重 DRAM → EPC ring buffer
  2. Enclave 内 HMAC-SHA256 逐块校验密文 → 不匹配则 enclave ABORT
  3. AES-NI 解密 → 明文权重进入计算区
  4. 计算完成后，权重明文从 EPC 中释放
```

**安全性等价性**: 权重在 EPC 内 vs EPC 外 + OCALL + HMAC:
- 机密性: 相同（AES-GCM 加密 vs MEE 加密，均为 AES）
- 完整性: 相同（HMAC 校验 vs MEE integrity tree）
- 差异: EPC 外方案暴露权重访问模式（地址、时序、大小）→ 侧信道。但 EPC 内同样暴露页级访问模式（MEE 在 cacheline 粒度工作），两者均可被 OS 观测。此问题正交于分区策略。

#### 输入数据（用户→服务器）

```
TLS/RA 安全通道: ECDH key exchange (SIGMA protocol) → shared session key
AES-GCM 加密传输 → 仅目标 enclave 可解密
```

#### 中间激活值（EPC 内 + 网络传输中）

```
EPC 内: MEE 硬件加密 + integrity tree 保护
网络: 安全通道加密（基于 RA 建立的共享密钥）
      跨服务器时，发送方 enclave 加密 activation
      → 接收方 enclave 解密后继续计算
```

### 3.3 与 MEDIA/OCC 原论文的安全性等价

| 安全属性 | OCC (MobiCom'19) | MEDIA | Ours (统一模型) |
|---------|-----------------|-------|----------------|
| 权重机密性 | ✅ (AES-GCM + OCALL) | ✅ (EPC MEE) | ✅ (AES-GCM + OCALL，同 OCC) |
| 权重完整性 | ✅ (HMAC verify) | ✅ (MEE integrity tree) | ✅ (HMAC verify，同 OCC) |
| 输入机密性 | ✅ (RA + TLS) | ✅ (RA + TLS) | ✅ (RA + TLS) |
| 激活值机密性 | ✅ (EPC MEE) | ✅ (EPC MEE) | ✅ (EPC MEE) |
| 网络机密性 | ✅ (安全通道) | ✅ (安全通道) | ✅ (安全通道) |
| 侧信道（访问模式） | ⚠️ 公开 | ⚠️ 公开 | ⚠️ 公开（正交） |

**结论**: 统一模型不降低任何安全保障，安全性等价于 OCC 和 MEDIA 原论文。

---

## 4. 完整 TEE 推理流程（论文 Design 章节用）

### Phase 0: 离线部署

```
1. 模型权重用 enclave-sealing-key 加密 (AES-GCM)
2. 加密权重部署到各边缘服务器的不可信文件系统
3. Enclave binary 签名后部署到各服务器
4. 各服务器启动 SGX enclave (ECREATE + EINIT)
```

### Phase 1: 推理请求 (Remote Attestation, ~100ms, 推理开始一次)

```
Client (用户)                          Edge Server (SGX Enclave)
    │                                         │
    │──── RA challenge ────────────────────→│
    │     (nonce + 验证 enclave 身份)          │
    │                                         │
    │←─── MRENCLAVE + 签名 Quote ───────────│
    │     (证明: 代码未被篡改, 运行在真实 SGX 中)   │
    │                                         │
    │──── ECDH key exchange (SIGMA) ────────→│
    │←─── shared session key established ────│
    │                                         │
    │──── encrypted input data ─────────────→│
    │     (进入 EPC, 解密)                      │
```

### Phase 2: 逐分区推理 (所有方法统一)

```
对每个 partition P 在分配的服务器 S 上，按拓扑序:

  ╔══════════════════════════════════════════════════════════════════╗
  ║ Step A: 数据就绪                                               ║
  ╠══════════════════════════════════════════════════════════════════╣
  ║ 等待 P 的所有前驱分区完成:                                       ║
  ║   - 同服务器(S): 前驱输出已在 EPC 中，直接可用，延迟 = 0         ║
  ║   - 跨服务器: 前驱输出通过安全通道传输                            ║
  ║     T_comm = RTT + data_bytes / (bandwidth_Mbps / 8)             ║
  ║     首次跳额外 +100ms (RA + SIGMA handshake, 可配置)              ║
  ╠══════════════════════════════════════════════════════════════════╣
  ║ Step B: 权重加载 + 校验 (per layer, pipeline 可重叠)             ║
  ╠══════════════════════════════════════════════════════════════════╣
  ║ 对 P 中的每个 layer L:                                          ║
  ║                                                                 ║
  ║   ┌── Thread 1: DDR Load ──────────────────────────────────┐   ║
  ║   │  OCALL memcpy(不可信DRAM → EPC ring buffer)             │   ║
  ║   │  带宽: 10 GB/s (DDR4)                                   │   ║
  ║   │  T_load = weight_bytes / (10 × 1024³) s                 │   ║
  ║   └────────────────────────────────────────────────────────┘   ║
  ║                                                                 ║
  ║   ┌── Thread 2: HMAC Verify ──────────────────────────────┐   ║
  ║   │  HMAC-SHA256 逐块校验 (integrity check)                 │   ║
  ║   │  带宽: 0.5 GB/s (SGX 内软件 SHA-256)                    │   ║
  ║   │  T_hash = weight_bytes / (0.5 × 1024³) s               │   ║
  ║   │  校验失败 → enclave ABORT (检测到篡改)                   │   ║
  ║   └────────────────────────────────────────────────────────┘   ║
  ║                                                                 ║
  ║   ┌── Thread 3: Compute ──────────────────────────────────┐   ║
  ║   │  AES-NI 解密权重 → 矩阵乘 / 卷积                         │   ║
  ║   │  使用已验证的权重 + 前驱 activation                      │   ║
  ║   │  T_compute = layer.workload / server_power_ratio       │   ║
  ║   │  输出: 本层 activation (保留在 EPC 中)                   │   ║
  ║   └────────────────────────────────────────────────────────┘   ║
  ║                                                                 ║
  ║   三线程流水线: T_layer = max(T_load, T_hash, T_compute)       ║
  ║                                                                 ║
  ╠══════════════════════════════════════════════════════════════════╣
  ║ Step C: Paging (仅当 peak_activation + ring_buffer > EPC 时)     ║
  ╠══════════════════════════════════════════════════════════════════╣
  ║ 若 peak_activation + RING_BUF_EPC > EPC_EFFECTIVE_MB:           ║
  ║   overflow = peak_activation + RING_BUF_EPC - EPC_EFFECTIVE_MB   ║
  ║   T_paging = 2.0 × (overflow / EPC) × T_compute [runtime penalty] ║
  ║                                                                 ║
  ║ 权重不参与此计算 (权重不在 EPC 中，由 OCALL 加载)                ║
  ╚══════════════════════════════════════════════════════════════════╝
```

### Phase 3: 输出返回

```
最终 partition 产生推理输出:
  ├─ 加密输出 (session key)
  └─ 通过安全通道传回 Client
Client 解密 → 获得推理结果
```

---

## 5. 当前代码差距分析

### 5.1 成本项覆盖矩阵

下表中:
- ✅ = 已正确实现
- ⚠️ = 部分实现或有 Bug
- ❌ = 完全缺失

| 成本项 | 物理含义 | OCC | DINA | MEDIA | Ours |
|--------|---------|-----|------|-------|------|
| `T_compute` | 矩阵乘/卷积计算 | ✅ | ✅ | ✅ | ✅ |
| `T_load_ddr` | DDR memcpy 加载权重 | ✅ | ❌ | ❌ | ✅ |
| `T_hmac` | HMAC-SHA256 完整性校验 | ✅ | ❌ | ❌ | **❌** |
| `T_paging(activation)` | 激活值超 EPC 的换页惩罚 | N/A (无 penalty) | ❌ 用 total_memory | ❌ 用 _sum_memory | ✅ schedule() 中正确 |
| `T_paging(weights)` | 权重超 EPC 的换页惩罚 | N/A (权重在 EPC 外) | ❌ 错误计入 | ❌ 错误计入 | ⚠️ merge_check/priority 中仍计入 |
| OCALL 边界开销 | EENTER/EEXIT | 隐含 | ❌ | ❌ | ❌ |
| `T_network` | 跨服务器数据传输 | N/A (单机) | ✅ | ✅ | ✅ |
| `T_enclave_init` | ECREATE+EINIT | ✅ | ✅ | ❌ | ❌ |
| `T_attestation` | 首次跳 RA 开销 | ❌ | ❌ | ❌ | ❌ |
| `T_decrypt` | AES-GCM 解密权重 | ✅ (HW 加速, 隐含) | ❌ | ❌ | ❌ |
| Pipeline 重叠 | 加载/校验/计算 三线流水 | ✅ | ❌ | ❌ | ⚠️ 仅 load/compute 双线 |

### 5.2 各方法具体问题

#### OCC (`alg_occ.py`)

**状态**: ✅ 已正确实现 weights-outside-EPC 模型。无需修改。

- `run()`: 使用 `_calculate_peak_activation()` 做 EPC 预算检查
- `schedule()`: 三线程 pipeline `max(compute, load, hash)`，无 paging penalty
- **缺失**: 无（已经是最正确的实现）

#### DINA (`alg_dina.py`)

**状态**: ❌ 完全未实现 weights-outside-EPC 模型。

- `_partition_cost()` (line 117-121): 使用 `partition.total_memory`（权重+激活）计算 paging penalty
- 无权重加载 pipeline（DDR copy + HMAC）
- 需修改:
  - `_partition_cost()`: 改为 `peak_activation + ring_buffer`，加 weight loading pipeline
  - 分区策略也需要适配（当前用 `total_memory` 判断分区大小）

#### MEDIA (`alg_media.py`)

**状态**: ❌ 完全未实现 weights-outside-EPC 模型。

- `_sum_memory()` (line 18-26): 返回权重+激活总和
- `_merge_check()` (line 159-202): 所有 `calculate_penalty()` 用 `_sum_memory()`
- `schedule()` (line 233-295): `calculate_penalty(self._sum_memory(p.layers))`
- `_compute_priorities()` (line 297-310): 同上
- 无权重加载 pipeline
- 需修改:
  - 新增 `_peak_activation_memory()` 方法（或复用 `Partition._calculate_peak_activation()`）
  - `_merge_check()`: 用 `peak_activation + ring_buffer` 替代 `_sum_memory()`
  - `schedule()`: 加 weight loading pipeline `max(compute, load, hash)`
  - `_compute_priorities()`: 同上

#### Ours (`alg_ours.py`)

**状态**: ⚠️ 不一致。schedule() 已部分正确，其他方法仍有问题。

正确的地方:
- `schedule()` (line 438-462): `peak_act = p.total_memory - weight_mb` + `calculate_penalty(peak_act + RING_BUF_EPC)` + `max(exec, load_t)` ✅
- `_single_server_time()` (line 494): 同上 ✅
- `_build_single_server_result()` (line 508): 同上 ✅

有问题的地方:
- `_merge_check()` (line 322-353): 使用 `calculate_penalty(part.total_memory)` — **仍包含权重** ❌
- `_compute_priorities()` (line 519-529): 使用 `calculate_penalty(p.total_memory)` — **仍包含权重** ❌
- **缺失 HMAC 步骤**: `eff_t = max(exec_t, load_t)` 应改为 `eff_t = max(exec_t, load_t, hash_t)` ❌

### 5.3 缺失开销的影响估计

#### HMAC 校验开销（最关键缺失）

以 VGG-16 fc1 为例（392 MB 权重）:

```
当前 Ours 模型:
  load_t = 392 / 10.0  = 39.2 ms      ← DDR copy
  exec_t 取决于计算量                    ← compute
  eff_t = max(exec_t, 39.2)

正确模型 (加 HMAC):
  load_t  = 392 / 10.0 = 39.2 ms
  hash_t  = 392 / 0.5  = 784.0 ms     ← HMAC 是真实瓶颈!
  exec_t 取决于计算量
  eff_t = max(exec_t, 39.2, 784.0)
```

对于 weight-heavy 的 FC 层，HMAC 可能成为整个 partition 的瓶颈。对 Conv 层（权重小，计算大），HMAC 可被计算覆盖，影响不大。

#### AES-GCM 解密

Intel AES-NI: ~13 GB/s = 13 MB/ms。VGG-16 fc1: 392 / 13 = 30 ms。**远小于 HMAC (784 ms)**，且在流水线中与 HMAC 并行。省略不影响结果准确性。

#### OCALL 边界开销

每次 OCALL ~5 µs = 0.005 ms。整个推理期间 OCALL 次数 = 层数（最多几百层），总开销 < 1 ms。**可忽略**。

---

## 6. 需修改的代码清单

### 6.1 `common.py` — 添加共享常数和方法

```python
# Weights-Outside-EPC model parameters (from OCC/MobiCom'19)
DDR_COPY_BW_MB_PER_MS = 10.0       # DDR memcpy for OCALL weight loading
HMAC_VERIFY_BW_MB_PER_MS = 0.5      # HMAC-SHA256 integrity verification in SGX
RING_BUFFER_EPC_MB = 20.0           # Ring buffer for weight staging in EPC

def partition_activation_paging_cost(partition, server):
    """Activation-only paging penalty (weights are outside EPC).
    
    Returns: (penalty_multiplier, load_time_ms, hash_time_ms)
    """
    weight_mb = partition.get_static_memory()
    peak_act = partition.total_memory - weight_mb
    penalty = calculate_penalty(peak_act + RING_BUFFER_EPC_MB)
    return penalty
```

### 6.2 `alg_media.py` — 全面改造

1. `_sum_memory()`: 改为返回 `peak_activation + ring_buffer`，或直接废弃用 `Partition` 方法
2. `_merge_check()`: `calculate_penalty(peak_activation + ring_buffer)`
3. `schedule()`: 每 partition 加 `max(exec, load, hash)` pipeline
4. `_compute_priorities()`: 同上

### 6.3 `alg_ours.py` — 修复不一致

1. `_merge_check()`: 使用 `calculate_penalty(peak_act + RING_BUF_EPC)`
2. `_compute_priorities()`: 同上
3. `schedule()`: `eff_t = max(exec_t, load_t, hash_t)` 加入 HMAC

### 6.4 `alg_dina.py` — 全面改造

1. `_partition_cost()`: 使用 `peak_activation + ring_buffer`
2. 加 weight loading pipeline

---

## 7. 论文论证要点

### 7.1 统一模型的合理性（必需在论文中论证）

> "We adopt a unified weights-outside-EPC memory model for all methods, following Occlumency's approach (MobiCom'19 §4-6). In SGX, weights can be stored in untrusted DRAM (AES-GCM encrypted) and loaded into EPC on-demand via OCALL with HMAC integrity verification. This approach preserves all SGX security guarantees: confidentiality (AES-GCM encryption), integrity (HMAC verification aborts on tampering), and freshness (sealing key derivation). Adopting a uniform model isolates the comparison variable to partitioning and scheduling strategy alone."

### 7.2 下限保证

> "Under the unified model, when any distributed method degenerates to single-server single-partition execution, its workflow is equivalent to OCC's, guaranteeing that OCC serves as the performance lower bound for all methods."

### 7.3 安全的等价性

> "Weights are encrypted with AES-GCM using keys derived from the SGX sealing key. Integrity is verified via HMAC-SHA256 on each load. Activation tensors remain within EPC protected by MEE. Cross-server activation transfers are encrypted via session keys established through remote attestation (SIGMA key exchange). The security guarantees are identical to those of OCC and MEDIA's original designs."

---

## 附录 A: 为什么 AES-GCM 解密可忽略

Intel AES-NI 硬件指令吞吐量: ~13 GB/s (≈13 MB/ms)
SGX 内 SHA-256 软件吞吐量: ~0.5 GB/s (≈0.5 MB/ms)

在流水线中，HMAC (0.5 GB/s) 是比解密 (13 GB/s) 慢约 26 倍的瓶颈。
AES-GCM 解密在 HMAC 等待期间有大量空闲周期完成，不增加有效关键路径延迟。
这是 SGX 架构的物理事实，有 Intel 白皮书支撑。

## 附录 B: 侧信道攻击的声明

统一模型下，权重加载通过 OCALL 暴露访问模式（地址、大小、时序）。
这与所有基于 SGX 的 DNN 推理方案（包括 OCC 和 MEDIA）面临相同的侧信道威胁。
现有防御（ORAM、Oblivious RAM、恒定时间加载）可叠加在此模型之上，
属于正交研究方向，不在本文讨论范围内。
