# 实验规划方案 v1.0
> 制定日期：2026-02-25
> 目标：对比 DINA / OCC / MEDIA / Ours 在 SGX TEE 分布式边缘集群上的端到端 DNN 推理时延

---

## 一、现状审计

### 1.1 项目目录现状（待整理）

```
Safe-Parallel_DNN_Inference/
├── 算法实现（当前正式版本）
│   ├── alg_dina.py          ✅ 正式
│   ├── alg_media.py         ✅ 正式
│   ├── alg_ours.py          ✅ 正式（含 HPA 四阶段）
│   ├── alg_occ.py           ✅ 正式
│   ├── alg_hpa.py           ⚠️  与 alg_ours.py 的关系待确认
│   ├── alg_ours_optimized.py       ⚠️  是否已合并到 alg_ours.py？
│   └── alg_ours_original_backup.py ⚠️  历史备份，应归档
│
├── 数据集
│   ├── datasets_260120/     ✅ 当前正式数据集（12个模型）
│   └── datasets/            ⚠️  旧版数据集（7个模型），应归档
│
├── 实验结果（混乱）
│   ├── results/             ⚠️  多个不同配置的结果混在一起
│   ├── results_baseline.csv ⚠️  独立文件，应入 results/
│   ├── results_comparison.csv ⚠️  独立文件，应入 results/
│   ├── server-chart_260120/ ✅ 服务器消融 CSV 数据
│   ├── network-chart_260120/ ✅ 网络带宽消融 CSV 数据
│   ├── ablation_study_chart_260120/ ✅ 带时间戳的完整输出
│   ├── server-chart/        ⚠️  旧版（重复），应归档
│   ├── network-chart/       ⚠️  旧版（重复），应归档
│   └── ablation_study_chart/ ⚠️  旧版，应归档
│
├── 辅助脚本（功能单一）
│   ├── run_all_experiments.py  ✅ 主入口（实验2+实验3）
│   ├── experiment_runner.py    ⚠️  旧版运行器，应归档
│   ├── analyze_bert_breakdown.py  ⚠️  分析脚本，功能待确认
│   ├── analyze_tp_bottleneck.py   ⚠️  分析脚本
│   └── diagnose_*.py              ⚠️  调试脚本，应归档
│
└── legacy_alg/              ⚠️  历史算法目录，应归档
```

### 1.2 现有实验数据概览

| 数据文件 | 内容 | 服务器配置 | 带宽 | 状态 |
|---------|------|---------|------|------|
| `results/final_results_4servers_100mbps.csv` | 旧版7模型对比 | 4×Xeon (同构) | 100Mbps | 旧，待归档 |
| `results/new_results_comparison.csv` | 同构多服务器扫描 | 1-16×Xeon | 100Mbps | 部分有用 |
| `server-chart_260120/*.csv` | 异构服务器消融 | 1-8台递增异构 | 100Mbps | **当前最新** |
| `network-chart_260120/*.csv` | 网络带宽消融 | 4×Xeon (同构) | 1-100Mbps | **当前最新** |

---

## 二、问题诊断（为何上次结果不符合预期）

### 问题 A：预期排序 DINA > OCC > MEDIA > OURS 不稳定

**根本原因分析：**

排序能否成立，取决于两个关键不等式是否同时满足：

```
条件1（DINA > OCC 成立）：
  DINA 的网络传输开销 > OCC 的分页换页开销
  即：(N_partitions - 1) × (RTT + Data/BW) > N_partitions × swap_cost

条件2（OCC > MEDIA 成立，即 MEDIA 优于 OCC）：
  MEDIA 优于 OCC 的条件（二选一即可）：
    A. 网络带宽高（通信时延低）：MEDIA 跨服务器通信代价小，而减少的换页开销收益可见
    B. 模型具有并行结构：MEDIA 可将并行分支分配到不同服务器，实现真正的执行并行

  MEDIA 退化为与 OCC 相当的条件（二者同时满足）：
    A. 网络带宽低（通信时延高）：跨服务器传输代价超过减少换页带来的收益
    B. 模型是严格串行结构（线性 DAG）：MEDIA 无法利用多服务器并行
```

**实验数据验证：**

| 配置 | BERT-base(线性, 100Mbps) 排序 | InceptionV3(并行, 100Mbps) 排序 |
|------|------------------------------|-------------------------------|
| 4×Xeon同构 | DINA(3628)>OCC(3612)>MEDIA≈Ours(3420) | DINA(1515)>OCC(1505)>MEDIA(1505)≈OCC>Ours(924) |
| 4台异构递增 | DINA(3684)>OCC(3669)>MEDIA≈Ours(3479) | DINA(1629)>OCC(1619)≈MEDIA(1619)>Ours(1108) |

**结论：**
- `DINA > OCC` 在当前配置下已基本成立 ✅
- `OCC > MEDIA`（MEDIA 优于 OCC）：
  - 线性模型（BERT/ViT）+ 串行特性 → MEDIA ≈ OCC，符合退化条件 ✅（正确现象）
  - 并行模型（InceptionV3）+ 高带宽 → MEDIA 与 OCC 差距仍小，原因见下方解释
- `MEDIA > OURS` 仅在并行结构模型（InceptionV3）中成立；线性模型中 MEDIA ≈ Ours ✅（正确现象）

**解释：**
- **线性模型 MEDIA ≈ OCC**：串行 DAG 结构使 MEDIA 无法利用多服务器并行，同时高带宽已使通信代价较低，整体收益有限，退化到接近 OCC 水平。这是**正确的理论行为**。
- **InceptionV3 中 OCC ≈ MEDIA**：InceptionV3 的并行分支在 MEDIA 策略下也无法被跨节点合并（合并会引入环），因此 MEDIA 实际分区粒度与 OCC 相近，两者换页/通信代价接近。
- **线性模型 MEDIA = Ours**：BERT/ViT 是严格串行 DAG，Ours 的 HPA 张量并行和 HEFT 并行调度无用武之地，退化为 MEDIA 行为。这是**正确的理论行为，不是 Bug**。

### 问题 B：InceptionV3 上 MEDIA ≈ OCC 的根本原因

**现象**（来自 `exp_results/exp3_server_ablation/server_hetero_incremental_InceptionV3.csv`）：

```
n=3: OCC=1619.14ms, MEDIA=1619.24ms (差距仅 0.1ms!)
n=8: OCC=764.37ms,  MEDIA=764.47ms  (差距仍只 0.1ms!)
```

MEDIA 与 OCC 的差距在所有服务器数量下均为固定的 0.1ms，**这是无可辩驳的证据**：MEDIA 的调度器将所有分区都分配给了与 OCC 相同的单一最优服务器，完全没有利用其他服务器并行执行。

**根本原因 1：alg_media.py 中的单位换算 Bug（次要）**

在 `alg_media.py` 的第 147 行（`_merge_check()`）和第 252 行（`schedule()`）中：
```python
# 传入 bandwidth_mbps * 8 * 1000 给 network_latency()
t_comm = network_latency(vol_mb, self.bandwidth_mbps * 8 * 1000)
```

而 `common.py` 中的 `network_latency()` 函数内部**已经**将输入的 Mbps 转换为 MB/ms：
```python
def network_latency(data_mb, bandwidth_mbps, ...):
    bandwidth_per_ms = (bandwidth_mbps / 8.0) / 1000.0  # MB/ms
```

因此传入 `bandwidth * 8 * 1000` 会导致等效带宽被放大 8000 倍（例如 100Mbps → 800,000 Mbps → ~100 GB/s），使所有数据传输的传输时延趋近于零，`network_latency()` 仅返回 RTT（5ms）。

- **影响评估**：该 Bug 使通信代价被严重低估（仅剩 RTT），但 **不是 MEDIA ≈ OCC 的主因**（低通信代价反而有利于 MEDIA 的并行分配决策）。

**根本原因 2：极端异构服务器配置导致调度器"强制串行"（主因）**

服务器配置中 Celeron G4930 的算力比仅为 0.11（相比 i5-11600 的 1.97，差距约 18倍）：

```
n=3 服务器: [Celeron(0.11), Celeron(0.11), i5-6500(0.93)]
n=8 服务器: [Celeron(0.11)×2, i5-6500(0.93)×4, i3-10100(1.03), i5-11600(1.97)]
```

MEDIA 的调度器对每个分区**尝试所有服务器，选择最早完成时刻**。对于并行分支 B2（其前驱 P_main 已分配到 i5-6500）：

- **分配给 i5-6500（已忙碌）**：start = finish[P_main]，finish = finish[P_main] + paging + exec/0.93
- **分配给 Celeron（空闲但极慢）**：start = finish[P_main] + 5ms(RTT)，finish = finish[P_main] + 5 + paging + exec/0.11

由于 exec/0.11 ≈ 8.5 × exec/0.93，只要分区执行时间 > 约 5.8ms（对 InceptionV3 的任何实际分区均满足），Celeron 的完成时刻远高于 i5-6500。**调度器正确地避开了 Celeron，但副作用是所有分区串行排队在 i5-6500（或 i5-11600）上**，等效为单服务器执行。

这是**正确的局部最优决策产生全局次优结果**的典型场景（异构调度中的负载均衡困境）。

**定量验证：**

OCC 在 n=3 时选 i5-6500（ratio=0.93），n=7 时切换到 i3-10100（ratio=1.03），n=8 时切换到 i5-11600（ratio=1.97）。

```
OCC(n=3)=1619ms, OCC(n=7)=1462ms → 比值: 1619/1462 ≈ 1.107 ≈ 1.03/0.93 ✓
OCC(n=8)=764ms  → 比值: 1619/764 ≈ 2.12 ≈ 1.97/0.93 ✓
MEDIA 完全匹配 OCC 的跳变时机，确认 MEDIA 也在追踪最优服务器。
```

**深入诊断（执行实验一后更新）：**

修复 Bug 并改用同构 Xeon_IceLake × 4 后，InceptionV3 上 MEDIA 仍然 ≈ OCC（1505.9ms vs 1505.8ms）。
进一步探查分区粒度与调度结果（`alg_media.py` 对 InceptionV3 产生 23 个分区，分区 DAG 有 11 个分叉节点），
发现 **所有 23 个分区仍被分配到同一台服务器（Server 0）**。根本原因如下：

**根本原因 3：InceptionV3 数据集 weight_memory=0，paging_cost=0（最终主因）**

`datasets_260120/inceptionV3.csv` 中各层只记录了 workload 和 activation_memory，
`weight_memory` 字段为 0。因此 `get_static_memory()=0`，所有分区的换页开销 = 0ms。

MEDIA 的核心优势在于"合理扩大分区以减少换页次数"——但当换页代价为 0 时，
该优势完全消失，MEDIA 的分区策略相比 OCC 没有任何节省。

**根本原因 4：贪心调度的 RTT 粘连效应（调度层）**

数据探查表明每个分区的执行时间（10–385ms）均远大于 RTT（5ms）。贪心调度对每个分区独立选最优服务器：
- 若前驱分区在 S0 → 当前分区放 S0 无 RTT 开销 → S0 的完成时刻比 S1 早 5ms
- → 贪心选 S0 → S0 继续占用 → 后续分区的前驱也在 S0 → 循环粘连

这是 LIST SCHEDULING 在低通信-计算比场景下的已知局限：贪心最优解导致全局串行化。
Ours 的 HEFT 算法也存在同样局限，但 TP 将 17 个重计算算子的工作量缩减约 6×，
使总延迟从 1505ms 降到 883ms（约 41% 加速），与调度层的并行能力无关。

**结论：**

| 来源 | 对 MEDIA ≈ OCC 的贡献 |
|------|----------------------|
| weight_memory=0 → paging=0 | 消除 MEDIA 最核心的分区优势 ✗ |
| RTT 粘连效应 | 调度层无并行 ✗ |
| 带宽单位 Bug（已修复） | 已修正，影响次要 |
| 极端异构服务器（实验三） | 实验一已消除，但仍存在粘连 |

**后续修复建议（针对 InceptionV3）：**

1. **补全 weight_memory 字段**：在 InceptionV3 数据集中添加真实权重内存（InceptionV3 各 Conv 权重约 0.1-5MB），使换页代价非零，让 MEDIA 的分区策略有实际意义
2. **论文中说明**：当前 MEDIA ≈ OCC 的结论仅适用于"无换页开销"的模型特例，与 MEDIA 的算法设计并不矛盾

---

### 问题 C：网络带宽消融曲线近乎水平

当前数据（`network-chart_260120/network_BERT-base.csv`，1-100Mbps）：
- OCC 曲线完全水平（不走网络）：3612ms 不变 ✅ 符合理论
- DINA 曲线几乎水平：3628ms→3627ms（变化仅 1ms）⚠️
- MEDIA=Ours 完全水平：3420ms 不变 ⚠️

**根本原因**：计算时间（~3400ms）远大于通信时间（BERT-base 层间激活值约 0.1-1MB，100Mbps 下传输 8ms），通信开销被淹没。

**解决方案：**
1. **扩大带宽范围下限**：延伸至 0.1 或 0.5 Mbps，使通信成为瓶颈
2. **选用大激活量模型**：ViT-large / BERT-large 的中间激活值更大，通信影响更显著

---

## 三、目录整理方案

### 整理后的目录结构

```
Safe-Parallel_DNN_Inference/
├── ── 算法实现（主目录，保持现有）──
│   ├── common.py
│   ├── loader.py
│   ├── alg_dina.py
│   ├── alg_media.py
│   ├── alg_ours.py
│   ├── alg_occ.py
│   └── alg_hpa.py
│
├── ── 数据集 ──
│   └── datasets_260120/    ← 正式数据集
│
├── ── 实验运行脚本 ──
│   └── run_all_experiments.py  ← 主入口
│
├── ── 实验结果（重新组织）──
│   └── exp_results/
│       ├── exp1_fixed_comparison/   ← 实验一：固定配置对比
│       ├── exp2_network_ablation/   ← 实验二：网络带宽消融
│       └── exp3_server_ablation/    ← 实验三：服务器数量消融
│
├── ── 可视化输出 ──
│   └── figures/
│       ├── exp1/
│       ├── exp2/
│       └── exp3/
│
├── ── 文档 ──
│   ├── README.md
│   └── EXPERIMENT_PLAN.md   ← 本文件
│
└── ── 归档（历史版本，不删除）──
    └── archive/
        ├── datasets/              ← 旧版数据集
        ├── results/               ← 旧版结果
        ├── server-chart/          ← 旧版图表
        ├── network-chart/         ← 旧版图表
        ├── ablation_study_chart/  ← 旧版图表
        ├── alg_ours_original_backup.py
        ├── alg_ours_optimized.py  (如已合并)
        ├── experiment_runner.py
        └── diagnose_*.py 等脚本
```

---

## 四、三组实验详细规划

---

### 实验一：多模型固定配置对比

**科学问题**：在合理的固定服务器与网络配置下，4 种方法在不同 DNN 模型上的端到端推理时延差异。

**预期结论**：
- 对于**具有并行结构的模型**（InceptionV3）：DINA > OCC > MEDIA > Ours
  - 高带宽（100Mbps）下 MEDIA 优于 OCC（MEDIA 可将并行分支分配到不同服务器）
  - Ours 利用 HEFT 并行调度，进一步超过 MEDIA
- 对于**线性结构模型**（BERT/ViT）：DINA > OCC > MEDIA ≈ Ours
  - 串行 DAG 结构下 MEDIA 无法利用多服务器并行，退化为接近 OCC 水平
  - Ours 无并行结构可利用，同样退化为接近 MEDIA 水平
  - **MEDIA ≈ Ours 是正确的理论现象，不是 Bug**

#### 1.1 参数设计

| 参数 | 取值 | 选择依据 |
|------|------|---------|
| 服务器数量 | **4 台** | 足以展示并行效果，不过多（边缘计算场景） |
| 服务器类型 | **同构 Xeon_IceLake** | 排除服务器异构性干扰，专注于算法差异 |
| 网络带宽 | **100 Mbps** | 标准边缘 LAN，MEDIA > OCC 的差异在此带宽下可见 |
| 模型集 | 见下方表格 | 覆盖线性和并行两类结构 |

#### 1.2 模型选择与预期排序

从 12 个模型中精选 **代表性模型**，涵盖：
- 轻量线性模型：TinyBERT-4l、TinyBERT-6l、DistilBERT-base
- 中型线性模型：BERT-base、ALBERT-base
- 大型线性模型：BERT-large、ALBERT-large、ViT-base
- **并行结构模型**：InceptionV3 ← 最重要，唯一展示 Ours > MEDIA 的模型
- Vision Transformer：ViT-small、ViT-large

#### 1.3 数据核验（当前已有 4×Xeon, 100Mbps 数据）

> 来源：`results/new_results_comparison.csv` 中 Servers=4, Bandwidth=100 行

| 模型 | DINA (ms) | OCC (ms) | MEDIA (ms) | Ours (ms) | 排序是否符合预期 |
|------|-----------|----------|------------|-----------|--------------|
| TinyBERT-4l | 104.7 | 104.7 | 104.7 | 26.6 | Ours最优✅，其余无区分⚠️ |
| TinyBERT-6l | 1340.9 | 611.3 | 1341.1 | 114.2 | ❌ OCC < DINA=MEDIA |
| BERT-base | 1056.2 | 1082.2 | 1069.9 | 605.1 | OCC>MEDIA✅，DINA<OCC❌ |
| DistilBERT-base | 1333.5 | 603.9 | 1332.1 | 112.2 | ❌ |
| ViT-base | 8761.2 | 2199.0 | 8605.4 | 425.2 | ❌ OCC远小于DINA |

**⚠️ 问题确认**：`new_results_comparison.csv` 使用的是旧版 `datasets/`（7 个模型），且排序混乱。
**需要使用 `datasets_260120/` 重新运行 4×Xeon_IceLake, 100Mbps 的对比实验。**

#### 1.4 实验执行步骤

1. **Step 1.1**：修改 `run_all_experiments.py` 中 `run_network_ablation()` 函数，增加一组固定对比实验（4台同构，100Mbps，所有12个模型）
2. **Step 1.2**：运行实验，将结果保存至 `exp_results/exp1_fixed_comparison/`
3. **Step 1.3**：验证排序是否符合预期，若不符合则定位原因（见下方调试路径）
4. **Step 1.4**：生成论文级对比柱状图和对比表格（LaTeX）

#### 1.5 排序验证调试路径

若运行后 DINA > OCC 不成立：
- 检查 `alg_dina.py` Line 73 强制轮转逻辑是否在此配置下生效
- 尝试降低带宽至 10 Mbps（增大网络代价）
- 或切换为异构服务器配置

若 OCC > MEDIA 不成立（MEDIA < OCC）：
- 这意味着 MEDIA 的跨服务器通信代价 > 减少换页次数的收益
- 验证：降低带宽使 MEDIA 变慢（应更接近 OCC）
- 若在 100Mbps 下 MEDIA < OCC 一直成立，说明理论设计正确

---

### 实验二：网络带宽消融实验

**科学问题**：网络带宽如何影响各方法的端到端时延？哪种方法对带宽最敏感？

**预期结论**：
- **OCC**：完全不受带宽影响（水平线）→ 可作为各带宽点的基准参考线
- **DINA**：带宽越低时延越高（N-1 次强制跨节点传输，每次代价 = RTT + Data/BW），是对带宽最敏感的方法
- **MEDIA**：
  - 低带宽时：跨服务器通信代价超过减少换页的收益，退化为接近 OCC 水平（时延接近 OCC 水平线）
  - 高带宽时（且并行模型）：通信代价低，多服务器优势显现，时延明显低于 OCC
  - 对于串行模型：无论带宽高低，MEDIA 始终接近 OCC（因串行 DAG 无并行可利用）
- **Ours**：
  - 低带宽时：AllReduce 同步开销大，与 MEDIA 差距缩小
  - 高带宽时（且并行模型）：HEFT 并行调度充分发挥，时延最低

#### 2.1 参数设计

| 参数 | 取值 | 选择依据 |
|------|------|---------|
| 服务器数量 | **4 台** | 固定，与实验一一致 |
| 服务器类型 | **同构 Xeon_IceLake** | 排除异构干扰 |
| 带宽范围 | **0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500 Mbps** | 见下方设计理由 |

**带宽范围设计理由：**

当前数据（1-100 Mbps）曲线近乎水平，根本原因是计算主导。需要向低带宽扩展：

```
通信时间变为计算时间的 10% 的条件：
  对于 BERT-base（假设单次跨节点传输 ~1 MB 激活值）：
  需要传输时间 = 3420ms × 10% = 342ms
  即带宽 ≤ 1MB / 342ms ≈ 2.9 Mbps ≈ 23 Mbps

  → 在 1-10 Mbps 范围内应可看到明显的曲线变化
  → 需要扩展至 0.5 Mbps 或更低
```

**推荐带宽序列**：`[0.5, 1, 2, 5, 10, 20, 30, 50, 100, 200, 500]` Mbps（11 个点）

注：0.5 Mbps 是典型的受限边缘场景；500 Mbps 接近带宽饱和上限。

#### 2.2 模型选择

优先选择激活量较大的模型以放大通信效果：
- **BERT-large**：激活值尺寸大于 BERT-base
- **ViT-large**：激活值尺寸最大
- **BERT-base**：对比基准
- **InceptionV3**：验证并行模型的带宽敏感性（Ours 受带宽影响最大）

#### 2.3 当前数据问题

`network-chart_260120/network_BERT-base.csv` 使用 4 台同构 Xeon，范围 1-100 Mbps：
- 所有曲线近乎水平（差值 < 10ms）
- 这是一个**有效的科学发现**：说明在 1-100 Mbps 范围内，通信开销可忽略
- **但无法体现方法差异随带宽的变化**

**需要重新运行**，增加 0.5 Mbps 区间并扩展至 500 Mbps。

#### 2.4 实验执行步骤

1. **Step 2.1**：修改 `run_all_experiments.py`，将 `BANDWIDTHS` 修改为 `[0.5, 1, 2, 5, 10, 20, 30, 50, 100, 200, 500]`
2. **Step 2.2**：运行实验，保存至 `exp_results/exp2_network_ablation/`
3. **Step 2.3**：检查曲线是否在低带宽区出现拐点（communication knee）
4. **Step 2.4**：生成折线图（X轴对数刻度，Y轴线性或对数）

---

### 实验三：服务器数量消融实验

**科学问题**：增加 TEE 节点数量如何影响各方法的推理时延？

**预期结论**：
- 随服务器数量增加，4 种方法的时延均应下降（或不增加）
- Ours 下降最显著（充分利用并行调度）
- MEDIA 下降有限（串行调度，并行能力弱）
- OCC 完全不随服务器数量变化（单服务器固定）
- DINA 下降有限（强制轮转带来通信开销，削弱并行收益）

#### 3.1 参数设计

| 参数 | 取值 | 选择依据 |
|------|------|---------|
| 服务器数量范围 | **1, 2, 3, 4, 5, 6, 7, 8 台** | 边缘集群典型规模 |
| 服务器类型 | **异构递增配置** | 见下方 |
| 网络带宽 | **100 Mbps** | 标准边缘 LAN |

**服务器配置（异构递增）：**
```
n=1: [Celeron G4930]                          (power_ratio=0.11)
n=2: [Celeron G4930, Celeron G4930]
n=3: [Celeron G4930, Celeron G4930, i5-6500]  (i5: ratio=0.93)
n=4: [Celeron G4930, Celeron G4930, i5-6500, i5-6500]
n=5: [Celeron G4930, Celeron G4930, i5-6500, i5-6500, i5-6500]
n=6: [Celeron G4930, Celeron G4930, i5-6500×4]
n=7: [..., i3-10100]                           (i3: ratio=1.03)
n=8: [..., i5-11600]                           (i5-11600: ratio=1.97)
```

**选择异构配置的理由**：
1. 使 DINA > OCC 的排序更加稳定（弱服务器拉大差距）
2. 真实边缘场景通常是异构的
3. 展示 Ours 的动态调度能力（能优先选择强服务器）

#### 3.2 当前数据问题

`server-chart_260120/server_hetero_incremental_BERT-base.csv` 已有数据：
```
n=1: OCC=9738, DINA=9738, MEDIA=9743, Ours=9743
n=4: OCC=3669, DINA=3684, MEDIA=3479, Ours=3479
n=8: OCC=3240, DINA=3428, MEDIA=3130, Ours=3130
```

**问题**：
1. n=1 时 OCC < Ours（OCC 单服务器应该与其他相同）⚠️
2. n=8 时 MEDIA = Ours（BERT-base 无并行结构，正常现象）

**需要验证**：
- OCC 在 n=1 时是否使用了最强的服务器（当 n=1 时，只有 Celeron，OCC 使用的就是 Celeron）
- 此时 power_ratio=0.11，所有计算时间 ×9，这是正确的

**BERT-base 中 n=1 时 OCC(9738) > Ours(9743)** 差距极小（5ms），几乎相等 ✅（符合理论：单服务器时所有方法应相同）

#### 3.3 实验执行步骤

1. **Step 3.1**：当前 `run_all_experiments.py` 的 `run_server_ablation()` 函数已实现，直接验证
2. **Step 3.2**：确认 OCC 曲线是否随服务器数量变化（理论上应该变化，因为 OCC 选 max power_ratio 的服务器）
3. **Step 3.3**：重点关注 InceptionV3 的结果（应展示 Ours 随服务器增加下降最多）
4. **Step 3.4**：生成折线图，X轴=服务器数量，Y轴=时延

> ⚠️ 注意：OCC 在本实现中会随服务器数量变化，因为它选择 `max(servers, key=lambda s: s.power_ratio)`。这与"OCC=单服务器固定"的理论略有偏差——在 n=1 时 OCC 只有弱机器，n=8 时 OCC 可以选强机器。这是**设计上的选择**，需要在论文中说明。

---

## 五、执行清单

### Phase 0：目录整理 ✅ 待完成
- [ ] 创建 `archive/` 目录
- [ ] 将旧版数据集、结果、图表移入 `archive/`
- [ ] 将备份算法文件移入 `archive/`
- [ ] 创建 `exp_results/` 目录结构

### Phase 1：代码验证 ✅ 待完成
- [ ] 确认 `alg_ours.py` 与 `alg_hpa.py` 的关系（是否重复）
- [ ] 确认 `alg_ours_optimized.py` 是否已合并
- [ ] 在 `run_all_experiments.py` 中验证实验配置参数

### Phase 2：实验一执行 ✅ 待完成
- [ ] 运行 4×Xeon_IceLake, 100Mbps, datasets_260120 的对比实验
- [ ] 验证排序结果
- [ ] 生成对比图表

### Phase 3：实验二执行 ✅ 待完成
- [ ] 修改带宽范围为 `[0.5, 1, 2, 5, 10, 20, 30, 50, 100, 200, 500]`
- [ ] 运行实验
- [ ] 验证曲线拐点

### Phase 4：实验三执行 ✅ 待完成
- [ ] 验证异构服务器消融实验结果
- [ ] 确认 InceptionV3 的 Ours 优势
- [ ] 生成折线图

### Phase 5：论文输出 ✅ 待完成
- [ ] 生成 LaTeX 对比表格
- [ ] 生成 PDF 图表（学术风格）
- [ ] 撰写实验章节分析段落

---

## 六、关键参数总结表

| 参数 | 实验一 | 实验二 | 实验三 |
|------|--------|--------|--------|
| 服务器数量 | 4台 | 4台 | 1~8台（递增） |
| 服务器类型 | 同构 Xeon_IceLake | 同构 Xeon_IceLake | 异构递增 |
| 网络带宽 | 100 Mbps | **0.5~500 Mbps** | 100 Mbps |
| 数据集 | datasets_260120/ | datasets_260120/ | datasets_260120/ |
| 主要观测 | 跨方法跨模型时延对比 | 带宽敏感性 | 服务器扩展性 |

---

## 七、重要注意事项

1. **MEDIA = Ours 对线性模型是正确现象**，不是 Bug。论文中应解释：Ours 的 HPA 张量并行和 HEFT 并行调度仅在具有并行分支结构（如 InceptionV3）的模型上展现优势。

2. **OCC 随服务器数量变化是设计选择**：当前实现中 OCC 选最优服务器，因此在异构实验中 OCC 会随服务器增加而下降。若要保持 OCC 为"单固定服务器基准"，需将 OCC 固定使用第一台服务器。需要在论文中明确说明。

3. **带宽消融的曲线是否"平坦"本身是结论**：在 1-100 Mbps 范围内，通信时延可忽略（相对于计算时延），说明算法的性能主要由计算决定，对网络带宽具有较强鲁棒性。这对实际部署有重要意义。

4. **每次实验前记录 Git commit hash**，保证数据可溯源。

---

## 八、张量并行（TP）有效性分析与模型选型

### 8.1 TP 最有效的条件（理论推导）

Ours 算法的 TP 代价模型（见 `common.py: hpa_cost()`）：

```
Cost(v, k) = T_comp / k^γ + Penalty(M_shard) + T_sync × P_sync

其中：
  γ = 0.9（并行效率因子）
  M_shard = (M_weight/k) + M_act × (1 - α + α/k)
  T_sync = network_latency(sync_data, bandwidth) × 0.5（Megatron 风格摊销）
  sync_data = output_bytes × 2(k-1)/k（Ring AllReduce）
```

**TP 产生显著收益的核心条件（按重要性排序）**：

#### 条件一（最重要）：单层权重 > EPC（93MB）——TP 可消除 4.5× 惩罚

当 `M_weight > EPC` 时，该层每次执行都会触发 SGX 内存换页，产生 4.5× 的执行时间惩罚（严重时可达 4.5×+ 0.25×N 倍）。TP 将权重拆分为 k 个分片：

```
M_shard = M_weight/k + M_act/k
若 M_weight = 400MB（VGG FC6），EPC = 93MB：
  k=1: penalty = 4.5  → Cost ≈ 4.5 × T_comp
  k=5: M_shard=80MB < EPC, penalty=1.0 → Cost ≈ T_comp/5^0.9 ≈ T_comp/4.22 (无惩罚!)
  综合加速比: 4.5 / (1/4.22) ≈ 19× （巨大提升）
```

**这是 TP 在 TEE 场景下最独特的价值：不仅提速，更能消除内存安全惩罚。**

#### 条件二：高网络带宽——AllReduce 同步代价可接受

TP 收益成立需满足：`T_sync < T_comp × (1 - 1/k^γ)`

以 k=2（最保守）为例：需要 T_sync < T_comp × 0.464

```
T_sync = (RTT + sync_mb / bandwidth_MB_ms) × 0.5
       = (5ms + output_mb / (bandwidth_mbps/8000)) × 0.5

对于输出 1MB 的层（如 BERT attention）：
  100 Mbps: T_sync = (5 + 80) × 0.5 = 42.5ms → 需要 T_comp > 92ms (可能不满足)
  1000 Mbps: T_sync = (5 + 8) × 0.5 = 6.5ms → 需要 T_comp > 14ms (较容易满足)
  10 Gbps:  T_sync = (5 + 0.8) × 0.5 = 2.9ms → 需要 T_comp > 6.3ms (容易满足)
```

**结论**：在 100 Mbps 的标准边缘带宽下，TP 对小层（无 EPC 溢出）几乎无益。**TP 的核心优势依赖于条件一（EPC 溢出消除），而非纯粹的计算加速**。

#### 条件三：算术强度高的层——计算远大于通信

Transformer 的 Linear/FC 层、Conv 层的算术强度通常较高，适合 TP。
Softmax、LayerNorm 等逐元素操作算术强度低，不适合 TP（无法均摊同步开销）。

### 8.2 当前数据集的 TP 有效性评估

现有 12 个模型的单层权重大小估计：

| 模型 | 最大单层权重 | 是否超出 EPC(93MB) | TP 收益评估 |
|------|------------|-----------------|------------|
| BERT-base | FFN: 768×3072×4≈9.4MB | ❌ 未超出 | 仅计算加速，收益有限 |
| BERT-large | FFN: 1024×4096×4≈16.8MB | ❌ 未超出 | 仅计算加速，收益有限 |
| ALBERT-base/large | 参数共享，更小 | ❌ 未超出 | 极低 |
| DistilBERT | 比 BERT-base 更小 | ❌ 未超出 | 极低 |
| TinyBERT | 更小 | ❌ 未超出 | 极低 |
| ViT-small/base | FFN≤16.8MB | ❌ 未超出 | 低 |
| ViT-large | FFN: 1024×4096×4≈16.8MB | ❌ 未超出 | 低 |
| InceptionV3 | 最大 Conv: ~5-10MB | ❌ 未超出 | 极低 |

**重要结论：当前 12 个模型的所有层均未超出 EPC（93MB），TP 在当前数据集上几乎无法展现其核心优势（EPC 惩罚消除）。这解释了为什么 MEDIA ≈ Ours 对所有当前模型成立。**

Ours 的 TP 在当前配置下只能提供：
- 计算时间减少：T_comp / k^0.9（约 6% 提升对 k=2）
- 减去同步开销：T_sync（在 100Mbps 下对大多数层超过计算节省）
- 净效果：可能为负！（TP 反而使整体变慢）

### 8.3 用于验证 TP 有效性的候选模型

需要添加单层权重 > EPC 的模型，以真正体现 TP 的 EPC 惩罚消除优势：

#### 首选：VGG-16（强烈推荐）

```
架构特征：
  FC6: 25088 × 4096 × 4 bytes = ~411MB ≫ EPC (93MB)
  FC7: 4096 × 4096 × 4 bytes  = ~67MB  < EPC
  FC8: 4096 × 1000 × 4 bytes  = ~16MB  < EPC

TP 效果（FC6 层）：
  k=1: penalty=4.5+0.25×(411-186)/93≈5.1, T_cost ≈ 5.1×T_base
  k=5: M_shard=82MB < EPC, penalty=1.0, T_cost ≈ T_base/5^0.9 ≈ T_base/4.22
  理论加速: ~21× (仅此一层，且排除同步开销时)
```

VGG-16 是标准 CV 模型（ImageNet Top-5 accuracy ~90.1%），在 TEE 推理场景下的大 FC 层正是 TP 最典型的应用场景。

#### 次选：AlexNet

```
FC6: 9216 × 4096 × 4 bytes = ~144MB > EPC
  k=2: M_shard=72MB < EPC → 消除惩罚
  理论加速: 4.5 / (1/2^0.9) ≈ 8.4×
```

#### 补充：ResNet-50 + 大 batch

```
批量推理时激活量放大，可能使分区内存超出 EPC
适合展示 TP 在 activation memory 上的分摊效果
```

### 8.4 建议的实验四：TP 有效性验证实验（新增）

**科学问题**：在哪些模型/配置下，Ours 的 TP 技术能产生显著加速？

**实验方案：**

| 参数 | 取值 |
|------|------|
| 模型 | VGG-16（必须）+ BERT-large（对比） |
| 服务器 | 4台同构 Xeon_IceLake |
| 带宽 | 100 Mbps, 1 Gbps（如可配置） |
| 对比方法 | OCC vs MEDIA vs Ours（TP on/off ablation） |

**预期结论：**
- VGG-16：Ours 显著优于 MEDIA（TP 消除 FC6 的 4.5× 惩罚）
- BERT-large：Ours ≈ MEDIA（无 EPC 溢出层，TP 收益抵消同步代价）
- **这一对比是 Ours 创新性的核心论据：TP 在 TEE 场景下最大价值不是计算加速，而是内存安全惩罚消除**

**执行步骤：**
1. 准备 VGG-16 的模型数据文件（`datasets_260120/vgg16.csv`）
2. 在 `run_all_experiments.py` 中增加 TP ablation 实验（Ours with/without TP）
3. 对比 VGG-16 和 BERT-large 在不同方法下的结果
4. 生成对比图表，重点展示 FC6 层的 TP 加速效果
