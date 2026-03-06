# Phase 4：算法理论分析

---

### [2026-03-03] MEDIA ≡ OCC 完整根因分析

**类型**：`关键发现` / `素材`

**现象**：所有 12 个 DNN 模型，在所有网络带宽和服务器配置下，MEDIA 的推理延迟与 OCC 几乎完全相等（差值 < 0.1%，仅浮点精度）。

#### 根因一：MEDIA 分区结果退化为 OCC

MEDIA 的 `_merge_check` 函数有两个 Case：

- **Case A**（`merged_mem ≤ EPC`）：**无条件合并**。这与 OCC 的贪心填充逻辑（"能装就装"）在行为上完全等价。
- **Case B**（`merged_mem > EPC`）：才执行 MEDIA 特有的通信-分页权衡。

对于 BERT-base（每层 0.3-2 MB）：层层贪心合并，最终每个分区约 88-92 MB（接近 EPC 上限），这与 OCC 的贪心填充产生相同结构的分区。

`run()` 末尾的**强制后处理**（`lines 202-237`）进一步将所有相邻且合并后 ≤ EPC 的分区强制合并，使分区结构更接近 OCC。

实测数据：
- OCC：4 个分区（BERT-base）
- MEDIA：5 个分区（边界划分不同，但同为线性链）

**关键**：不同的分区划分无法带来差异，因为调度阶段消除了一切可能的差别（见根因二）。

#### 根因二：MEDIA 调度器的 RTT 粘性

调度器追踪（BERT-base，4 台服务器，100 Mbps）：

```
调度 P4（paging=743ms, exec=201ms）：
  S0（P3 所在）: dep_ready=640ms,  ft = 640+743+201 = 1585ms  <- 选中
  S1（空闲）:    需等通信 5.25MB@100Mbps = 425ms
                dep_ready=640+425=1065ms, ft = 1065+743+201 = 2010ms
  差值：S1 比 S0 多 425ms（通信惩罚）
```

**RTT 粘性的数学本质**：

对于分区 P_i 依赖 P_{i-1}（在 S0 完成于时刻 t_prev）：
- 同服务器 S0：`finish = t_prev + paging + exec`
- 其他服务器 Sk：`finish = t_prev + comm_time + paging + exec`

由于 `comm_time = RTT(5ms) + data/BW > 0`，**同服务器永远优先**，差值精确等于 comm_time。

在 100 Mbps 下，BERT-base 各分区间通信量 2.56–5.25 MB，对应 210–425ms 额外惩罚。即使 Sk 完全空闲，也无法弥补这一差距。

**结论**：所有 5 个分区全部分配到 S0，MEDIA 退化为 OCC（单服务器串行执行）。

#### 根因三：12 个 benchmark 模型均不满足 MEDIA 的前提假设

MEDIA 的设计假设："存在多条大型独立计算路径，每条路径的内存需求接近或超过 EPC（93 MB）"。

| 模型类型 | 并行结构 | 为何 MEDIA 无效 |
|---------|---------|----------------|
| BERT/ViT/ALBERT 等 Transformer | 多头注意力（12 heads） | 每个 head ≈ 2-3 MB，所有 heads 立即在 concat 汇合，MEDIA Case A 将其合并进同一分区 |
| InceptionV3 | 每个 inception 模块 4 路并行 | 总静态内存 89 MB，全部分支合并进 P0（90 MB），只生成 2 个分区构成线性链 |

**Transformer 的"假并行"**：BERT-base 有 348 个高度节点（in_degree>1 或 out_degree>1），表面上存在大量并行分支，但这些并行是**层级（layer-level）并行**，不是**分区级（partition-level）并行**。每个注意力头极小，被合并进单个 EPC 分区后，调度层面看不到任何可并行的独立分区。

**MEDIA 真正能发挥优势的场景**：模型具有多条独立的重量级并行路径，每条路径的内存 ≥ 93 MB（EPC 上限），使得它们必须成为不同的分区，才能被调度到不同服务器上并发执行。当前 12 个 benchmark 无一满足此条件。

> **素材标注（论文核心论点）**：
> - 这是论文攻击 MEDIA 的核心技术论点，**直接可写入 Evaluation 的 Discussion 部分**
> - 论点逻辑链：MEDIA 的调度优势依赖于"分区级并行机会" → Transformer 模型仅有"层级并行"（attention heads）→ MEDIA 的分区器将所有 heads 合并进单个 EPC 分区 → 调度器看不到可并行的独立分区 → MEDIA 退化为 OCC
> - 实验数据（12 个模型全部 MEDIA≡OCC）是强有力的实证支撑

---

### [2026-03-03] 为什么 Ours 优于 MEDIA：机制差异分析

**类型**：`关键发现` / `素材`

#### 并行粒度的根本差异

| 维度 | MEDIA | Ours (HPA) |
|------|-------|-----------|
| 并行粒度 | **分区级**（整个分区移到另一台服务器） | **算子级/层级**（单层 tensor 切分到 k 台服务器） |
| 并行机制 | 流水线并行（Pipeline Parallelism） | 张量并行（Tensor Parallelism） |
| 并行条件 | 需要多条独立的大型计算路径 | 只需单个 compute-heavy 算子（Linear/Conv 层） |
| 通信模式 | 分区输出 tensor 传输（one-shot，大数据量） | AllReduce（分散-规约，数据量 = output 的 2×(k-1)/k） |

#### Ours 绕过 RTT 粘性的原因

MEDIA 的每个分区通信是"完成后再传输"（sequential），必须等待整个分区执行完才能发出数据。这个等待本身就拉平了并行的收益。

Ours 的 AllReduce 是**层内同步**（intra-layer），执行 k 个 shard 并发计算后立即规约，下一层在同一批服务器上继续。**不存在分区级的依赖等待**，RTT 粘性问题天然被规避。

#### 自适应带宽决策（HPA 的 DP 选择器）

HPA 通过 DP 为每个算子选择最优并行度 k：

```
Cost(v, k) = T_comp / k^γ + Penalty_mem(M_shard/k) + T_AllReduce(k) × P_sync
```

- 低带宽（0.5-5 Mbps）：T_AllReduce 大 → DP 选 k=1（不并行），Ours 退化为 OCC
- 高带宽（≥ 20 Mbps）：T_AllReduce 小 → DP 选 k=4~8，并行收益超过通信开销

实验验证：
- BERT-large @ 0.5 Mbps：Ours=82,908ms（比 OCC 慢 6.7×，AllReduce 代价极高）
- BERT-large @ 100 Mbps：Ours=3,500ms（比 OCC 快 3.5×）
- BERT-large @ 500 Mbps：Ours=3,103ms（比 OCC 快 4.0×）

**HPA 的隐式决策边界（BERT-large 约在 10-20 Mbps）**：低于阈值时 k=1（无并行），高于阈值时 k>1（激活并行）。这个阈值取决于模型参数量、层级别计算量和 AllReduce 数据量。

> **素材标注**：
> - "算子级并行 vs 分区级并行"是论文 Design 章节的核心区分点，可画一张对比图
> - HPA 的带宽自适应 DP 决策是**核心可专利技术**：在 SGX EPC 约束下，根据当前网络带宽动态选择张量并行度的方法
> - "自适应带宽阈值"现象（低带宽 k=1，高带宽 k>1）是 Evaluation 中 bandwidth sensitivity 实验的解释机制

---

### [2026-03-03] OCC 算法分析：为何它是合理的 baseline

**类型**：`关键发现`

OCC（Oblivious Context-switch Computation）的定义：

1. **分区策略**：贪心拓扑顺序填充，每个分区的 `total_memory` 不超过 EPC（93 MB）
2. **调度策略**：所有分区串行执行在**最快的单台服务器**上（`max(s.power_ratio)`）
3. **paging 开销**：仅计算静态内存（`get_static_memory() = weight + bias + encryption`）

**OCC 代表的实际意义**：在没有任何跨服务器通信的前提下，单个 SGX 节点能达到的最优推理延迟。它是一个**通信成本为零的理想化 baseline**（每次分区切换无通信开销）。

任何分布式方法在低带宽下都可能劣于 OCC（通信代价过高），这是物理约束，不是算法缺陷。

**实测 paging 占比（BERT-base）**：
- 静态内存 paging：~2856ms（79% 总延迟）
- 计算执行：~757ms（21% 总延迟）
- 结论：BERT-base 在 SGX 下是**I/O 密集型**（paging 主导），而非计算密集型

> **素材标注**：
> - "paging 占 79%"是说明 SGX EPC 约束严重性的关键数据，可进入 paper Background 章节
> - OCC 的单机最优 baseline 定义，澄清了比较的意义：任何分布式方法（含 Ours）在低带宽时都可能不如 OCC

---

### [2026-03-03] DINA 强制轮换策略的理论局限

**类型**：`关键发现`

DINA 在 `schedule()` 中有一个硬约束（`alg_dina.py:73`）：

```python
# CONSTRAINT: Must switch server after every partition to force network overhead
if i > 0 and s.id == prev_server_id and len(self.servers) > 1:
    continue
```

即每个连续分区必须使用不同的服务器。这个设计的初衷是避免单机负载过重，但在实验中带来了严重问题：

1. **通信惩罚不可避免**：无论带宽多低，DINA 都强制发生跨服务器传输
2. **在低带宽下劣化剧烈**：ViT-large @ 0.5 Mbps = 1,208,017 ms（是 OCC 的 88×）
3. **分区数量 × 通信 = 累积惩罚**：ViT-large 有 15 个分区，14 次强制轮换，每次都付出巨大通信代价

**DINA 的设计假设**：网络带宽充裕（通信时间可忽略），分区执行时间是主要瓶颈，轮换服务器可并行化 paging 准备。在高带宽（≥ 500 Mbps）下，DINA 接近 OCC，验证了这一假设在特定条件下成立。

> **素材标注**：
> - DINA 强制轮换作为一种"盲目强制分布"的负面示例，可在 paper Introduction 或 Related Work 中引用，说明"分布式不总是更好"
> - DINA @ 低带宽的极端劣化数据是说明本文问题 significance 的有力素材

---
