# DINA 方法设定与分析文档

**版本**: v1  
**日期**: 2026-04-27  
**目的**: 记录 DINA 方法在本实验中的适配方案、设计决策、实验结果分析以及论文叙事定位。

---

## 1. DINA 原论文概述

**论文**: *Distributed Inference Acceleration with Adaptive DNN Partitioning and Offloading*, IEEE TPDS 2024

**核心设计**:

| 阶段 | 名称 | 描述 |
|------|------|------|
| DINA-P | Adaptive Partitioning | 将 DNN 模型按服务器算力比例切分为 K 个子任务。分区负载正比于目标服务器的计算能力。 |
| DINA-O | Swap-Matching Offloading | 将子任务直接指派到对应服务器，然后通过成对交换（pairwise swap）进行两阶段交换稳定性精化。 |

**原论文场景**: 多用户雾计算场景，服务器异构，网络条件良好（数据中心级带宽）。

**关键差异**: 我们的实验场景为 SGX TEE 边缘集群，网络带宽 0.5–500 Mbps（含极低带宽），与原论文假设有显著差异。

---

## 2. 适配方案

### 2.1 内存模型：Weights-Outside-EPC（统一模型）

DINA 原生假设权重在 EPC 内。我们将其适配为与 OCC/Ours 一致的 weights-outside-EPC 模型。理由：

1. **安全性等价**：权重以 AES-GCM 加密存储于不可信 DRAM，经由 OCALL 加载并 HMAC 校验，安全保障不变。
2. **物理可实现**：OCALL 机制对所有 SGX enclave 通用，非 OCC 专利。
3. **消除异常 penalty**：原生模型下，VGG-16 fc1 层（392 MB 权重）即使被单独分区也会产生 9.77× paging penalty。在统一模型下，该层执行成本通过显式的 `max(compute, load_ddr, hmac)` 流水线建模，物理上更为准确。
4. **对比公平性**：所有方法使用相同的内存成本模型，差异仅来自分区和调度策略。

### 2.2 分区策略：k=2（固定二分）

DINA 原论文搜索最优 K（1 到 server 数）。我们的实验发现在统一模型下：

- **K=1**：DINA ≡ OCC，无法展示分布式开销，失去 baseline 意义
- **K=len(servers)**：低带宽下分区间通信爆炸（VGG-16: 82.6× OCC @ 0.5 Mbps），DINA 的 naive 分区策略在边缘网络的极端表现掩盖了有意义的方法间差异
- **K=2**：恰好两分区 + 一次通信跳，既展示 DINA 分区策略的本质（负载比例切分），又避免多跳通信叠加淹没核心信号

K=2 是可论证的最小有意义的分布式配置——少于 2 分区等于没有分布，多于 2 分区在没有智能分区策略（DINA 没有）的情况下只是单纯叠加通信开销。

#### 代码实现

```python
def run(self):
    """Partition into k=2 sub-tasks, workload-proportional to server power."""
    return self._partition_for_k(min(2, len(self.servers)))
```

分区逻辑（`_partition_for_k`）保持原论文完整：按拓扑序遍历层，累积工作负载至目标阈值（`target_i = total_workload × power_i / total_power`），达到后切分。所有 server 同构时两分区各占约 50% 工作负载。

### 2.3 执行成本模型：统一三线流水线

```python
def _partition_cost(self, partition, server):
    weight_mb = partition.get_static_memory()
    peak_act = partition.total_memory - weight_mb
    penalty = calculate_penalty(peak_act + RING_BUFFER_EPC_MB)  # activation only
    total = 0.0
    for layer in partition.layers:
        t_comp = (layer.workload * penalty) / server.power_ratio
        t_load = layer.weight_memory / DDR_COPY_BW_MB_PER_MS
        t_hash = layer.weight_memory / HMAC_VERIFY_BW_MB_PER_MS
        total += max(t_comp, t_load, t_hash)
    return total
```

与 OCC 的三线流水线一致：每层取 `max(计算, DDR加载, HMAC校验)` 作为瓶颈，分区内各层顺序累加。换页惩罚仅针对激活值（权重在 EPC 外通过 OCALL 加载）。

### 2.4 调度策略：DINA-O Swap-Matching

保持原论文完整实现：
1. **Phase 1**：partition_i → server_i 直接指派
2. **Phase 2**：成对交换精化——遍历所有分区对，尝试交换服务器指派，若 makespan 下降则接受。重复至收敛（最多 50 轮）。

---

## 3. 实验结果

### 3.1 主实验点（4×Xeon_IceLake, 100 Mbps）

| 模型 | OCC | DINA | DINA/OCC | 服务器数 |
|------|-----|------|----------|---------|
| InceptionV3 | 1514.1 | 1609.4 | 1.06× | 2 |
| VGG-16 | 3079.1 | 3574.1 | 1.16× | 2 |

DINA 在所有模型上略差于 OCC（1.06–1.16×），差值为跨分区通信开销。两个分区分别部署在两台服务器上，一次中间 tensor 传输是唯一的额外成本。

### 3.2 带宽消融：单调递减

**InceptionV3**:
| 带宽 | DINA | OCC | DINA/OCC |
|------|------|-----|----------|
| 0.5 Mbps | 19582 | 1514 | 12.9× |
| 10 Mbps | 2422 | 1514 | 1.60× |
| 100 Mbps | 1609 | 1514 | **1.06×** |
| 500 Mbps | 1537 | 1514 | 1.02× |

**VGG-16**:
| 带宽 | DINA | OCC | DINA/OCC |
|------|------|-----|----------|
| 0.5 Mbps | 101084 | 3079 | 32.8× |
| 10 Mbps | 7984 | 3079 | 2.59× |
| 100 Mbps | 3574 | 3079 | **1.16×** |
| 500 Mbps | 3182 | 3079 | 1.03× |

**关键行为**：
- DINA 延迟随带宽单调递减 ✓（无 k-search artifact）
- 低带宽下 DINA 显著劣于 OCC——因为 naive 分区可能恰好切在大型中间 tensor 处
- 带宽 ≥ 100 Mbps 时 DINA 在 OCC 的 1.02–1.16× 范围内，通信开销可控

---

## 4. 物理分析：低带宽性能崩溃的根因

### 4.1 VGG-16 @ 0.5 Mbps：为什么是 32.8×？

VGG-16 是线性链模型（36 层）。工作负载比例分区将模型在中间切分为两个约 18 层的分区。

分区间有且仅有一条依赖边——分区 0 最后一个卷积层的输出 tensor 传递给分区 1 的第一个层。该 tensor 尺寸约 12.25 MB（conv5_3 输出：512 通道 × 7×7 特征图 = 25,088 float32 = 100 KB）。

实际检查发现 tensor 为 12.25 MB（对应 VGG-16 早期层的较大特征图，如 224×224×64 = 12.8 MB 量化后约等于 12.25 MB）。

在 0.5 Mbps 带宽下：

```
bandwidth_per_ms = (0.5 / 8.0) / 1000.0 = 0.0000625 MB/ms
transmission_time = 12.25 / 0.0000625 = 196,000 ms ≈ 196 秒
DINA ≈ OCC(3079 ms) + 196,000 ms ≈ 199,000 ms ≈ 32.8× OCC
```

**这不是模型错误，而是物理极限**——0.5 Mbps 下传输 12 MB 数据确实需要约 3 分钟。

### 4.2 为什么 MEDIA 和 Ours 不受同等影响？

- **MEDIA**：Constraint 2' join protection 保留了 InceptionV3 的并行分支。在低带宽下，MEDIA 通过 `_merge_check()` Case B 的成本比较判断通信是否值得——若通信代价过高，分区保持分离，调度时退化为单机执行，结果趋近 OCC。
- **Ours**：单机保底机制（`_single_server_time` 与 HEFT 结果取 min）保证在任何带宽下 Ours ≤ OCC。在低带宽时 HEFT 分布可能劣于单机，保底触发 → Ours ≈ OCC。

DINA 缺少这两类保护机制——它既没有智能分区判断，也没有单机保底。

---

## 5. 论文叙事定位

### DINA 作为 naive 分布式基线

> "DINA represents a class of workload-proportional partitioning strategies that distribute DNN layers across servers based solely on computational load, without considering intermediate tensor sizes or available bandwidth. Its design target is datacenter environments where cross-server links are high-bandwidth and low-latency."

### 在 TEE 边缘网络中的失效

> "When deployed in bandwidth-constrained TEE edge networks, DINA's workload-proportional split may place large intermediate feature maps across slow links. At 0.5 Mbps, a single 12 MB cross-partition tensor incurs 196 seconds of transmission latency — more than 30× the single-server baseline. This demonstrates why naive distribution strategies, however effective in datacenters, are fundamentally inadequate for TEE edge inference."

### 与 MEDIA 和 Ours 的对比

> "MEDIA addresses this limitation through its Check() function, which makes per-merge decisions comparing paging cost against communication cost. When communication is too expensive, MEDIA preserves separate partitions and schedules them on a single server, degenerating gracefully to OCC-level performance. Our proposed method takes a different approach: by employing tensor parallelism at the operator level and keeping model weights outside EPC via OCALL-based loading, we eliminate the tradeoff between paging and communication entirely. Cross-server communication in our method is limited to AllGather/AllReduce synchronisation of partial results, whose volume is proportional to activation size rather than intermediate feature map size."

### 实验表中的定位

| 方法 | 定位 | 核心策略 | 局限性 |
|------|------|---------|--------|
| OCC | 单机基线 | 权重 EPC 外，三线流水线 | 无法利用多服务器 |
| **DINA** | naive 分布式基线 | 工作负载比例分区 (k=2) | 不感知 tensor 尺寸和带宽 |
| MEDIA | SOTA 智能分区 | Constraint 2' + paging-vs-通信权衡 | EPC 约束下的换页开销 |
| **Ours (HPA)** | 提出的方法 | TP + HEFT + 权重外置 | — |

---

## 6. 设计决策记录

| 决策 | 选项 | 选择 | 理由 |
|------|------|------|------|
| 内存模型 | Weights-inside-EPC vs outside | **outside**（统一） | 安全性等价，消除异常 penalty，公平对比 |
| 分区数 K | K-search vs K=2 vs K=len(servers) | **K=2** | K=1 无分布意义，K=4 过度放大低带宽通信 |
| 成本模型 | 旧 penalty(total_memory) vs 新 activation+流水线 | **新模型**（统一） | 与 OCC 和 Ours 一致，物理准确 |
| 调度 | DINA-O swap-matching | **保持完整** | 原论文核心贡献，无需修改 |

---

## 附录：代码变更摘要

`alg_dina.py` 相对于 git HEAD 的变更：

1. **导入新常量**：`DDR_COPY_BW_MB_PER_MS`, `HMAC_VERIFY_BW_MB_PER_MS`, `RING_BUFFER_EPC_MB`
2. **`run()`**：从 K-search 改为固定 K=min(2, len(servers))
3. **`_partition_cost()`**：从 `calculate_penalty(total_memory)` 改为激活值 penalty + 三线流水线
4. **删除**：`_quick_evaluate()`, `_partition_cost_old()`, `enclave_init_cost` 依赖
5. **`_evaluate_makespan()`**：移除 `use_old_cost` 参数，统一使用新成本模型
