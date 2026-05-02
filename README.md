# 分布式 DNN 推理仿真系统

本项目模拟了在资源受限（特别是 SGX EPC 内存限制）的边缘设备集群上进行分布式 DNN 推理的过程。系统对比了三种不同的任务分割与调度算法：DINA、MEDIA 以及我们提出的优化算法（Ours）。

## 算法实现详解

### 1. OCC (Occlumency) — 单机基线
**来源**：MobiCom 2019

**核心思想**：权重存储在 EPC 外（不可信 DRAM），通过 OCALL 逐层加载并 HMAC 校验后计算。EPC 仅存放激活值和 ring buffer。

*   **分区策略**：按拓扑序遍历层，以 `peak_activation + ring_buffer` 为 EPC 约束切分分区。
*   **调度策略**：单服务器三线流水线——DDR 加载 / HMAC 校验 / 计算，瓶颈为 `max(compute, load, hash)`。
*   **无网络通信**：所有分区在同一服务器顺序执行。

### 2. DINA — naive 分布式基线
**来源**：IEEE TPDS 2024

**核心思想**：工作负载比例分区——将模型按服务器算力比例切分为 k=2 个子任务，通过 swap-matching 调度精化指派。

*   **分区策略**：按拓扑序累积工作负载至目标阈值后切分，k 固定为 2（一个通信跳）。
*   **调度策略**：partition_i → server_i 直接指派 + 成对交换精化（DINA-O）。
*   **内存模型**：weights-inside-EPC，`calculate_penalty(total_memory)`。

### 3. MEDIA — SOTA 智能分区
**来源**：ICDCS 相关论文

**核心思想**：通过度约束边选择 + Check() 函数在换页开销与通信开销之间做权衡。保留 Constraint 2'（join protection）以保护不等长并行分支。

*   **分区策略**：Algorithm 1（边选择，Constraint 1+2+2'）+ Algorithm 2（M 边收缩，三种 Case + Check）。
*   **调度策略**：优先级列表调度（bottom-up critical path rank）。
*   **内存模型**：weights-inside-EPC，`_sum_memory` + `calculate_penalty`。

### 4. Ours (HPA) — 本文方法
**核心思想**：算子级张量并行（TP）+ MEDIA 分区 + HEFT 调度。TP 将单算子拆分为 k 个 shard 并行执行，通过 AllGather（Conv）/AllReduce（FC）同步。MAX_PART_WL=150ms 防止分区过度合并，保留 CSP 并行分支。

*   **Stage 0-2**：TP 决策——`_filter_candidates` → `_build_cost_surface` → `_select_best_k`，server-aware 瓶颈算力 + 双路径评估（TP vs no-TP 选优）。
*   **Stage 3**：图增强——拆分算子为 shard + 插入 sync barrier，TP 边界保护防止同算子 shard 被合并。
*   **Stage 4**：MEDIA 分区——度约束合并 + EPC 检查 + workload 上限。
*   **Stage 5**：HEFT 调度——upward rank 优先级 + 最早完成时间贪心分配 + OCC 单机保底。
*   **内存模型**：weights-inside-EPC，`_partition_cost = workload × calculate_penalty(total_memory) / power`。

#### SGX 时间开销拆解

本实现对 Intel SGX 环境下的各项时间开销进行了精细建模，确保仿真结果贴近真实硬件行为。以下是端到端推理时延的组成部分：

| 时间组件 | 物理来源 | 建模方式 | 典型值 |
|----------|----------|----------|--------|
| **Enclave Entry/Exit** | `ecall`/`ocall` 系统调用开销（保存/恢复寄存器、页表切换、MAC 校验） | 每个分区固定开销 | **0.005 ms** / 分区 |
| **执行时间（基准）** | 分区内所有层的计算时间（enclave_time） | `Σ layer.workload` | 取决于模型 |
| **EPC 溢出惩罚** | 当分区内存 > EPC 时的**一次性初始化开销**（EINIT、元数据重建、初始页面错误爆发） | `penalty = calculate_penalty(memory)`<br>- memory ≤ EPC: `1.0×`<br>- EPC < memory ≤ 2×EPC: `4.5×`<br>- memory > 2×EPC: `4.5 + 0.25×(额外EPC)` | **4.5×** (首次溢出) |
| **页面错误处理** | 每次访问不在 EPC 的页面触发 `#PF`，进入 SGX 页面错误处理例程 | `num_pages × 0.03 ms`<br>页面数 = `swap_bytes_mb × 1024 / 4 KB` | **30 µs** / 4 KB 页面 |
| **分页数据传输** | EPC ↔ DRAM 的加密/解密传输（受 AES-NI 引擎限制） | `swap_bytes_mb / paging_bandwidth`<br>带宽 = **1 GB/s** (0.8–1.2 GB/s) | **1 MB/ms** |
| **上下文切换** | 分区间切换时换出前一分区、换入下一分区的综合开销 | `(prev_mem + next_mem) × (页面错误 + 传输)` | 取决于分区大小 |

**关键假设与配置**：

1. **EPC 大小**：`EPC_EFFECTIVE_MB = 93 MB`（Intel SGX1 平台典型值：128 MB 总容量 - 35 MB 元数据）
2. **页面大小**：`PAGE_SIZE = 4 KB`（SGX 标准页面大小）
3. **分页带宽**：`PAGING_BANDWIDTH = 1 GB/s`（实测范围 0.8–1.2 GB/s，受 CPU 加密单元限制）
4. **页面错误开销**：`PAGE_FAULT_OVERHEAD = 30 µs`（包含上下文切换、页表更新、MAC 校验）
5. **Enclave Entry/Exit**：`ENCLAVE_OVERHEAD = 5 µs`（基于 Intel SGX SDK 测量）

**建模合理性说明**：

- **惩罚因子与分页开销的分离**：`calculate_penalty()` **仅代表 EPC 溢出时的一次性初始化成本**（EINIT、元数据重建），**不包含**运行时的逐页换页开销（后者在 `swap_time` 中单独计算），避免重复计费。
- **分段惩罚模型**：基于 ICDCS'22《DNN Partitioning and Assignment for Distributed Inference in SGX-Empowered Edge Cloud》的实验数据，当内存首次超出 EPC 时会出现 **突发性** 的初始化开销（约 4.5×），而继续增长时呈现 **缓慢线性** 增长（约 0.25×/EPC），该模型比简单的线性惩罚更符合真实硬件行为。
- **页面粒度建模**：将换页时间细化为 `页面数 × 单页固定开销 + 数据传输时间`，能够反映 **页面碎片** 与 **访问局部性** 对性能的影响。

**参考文献**  
Lee, T., Lin, Z., Pushp, S., Li, C., Liu, Y., Lee, Y., Xu, F., Xu, C., Zhang, L., & Song, J. *Occlumency: Privacy‑preserving Remote Deep‑learning Inference Using SGX*. MobiCom 2019. DOI: 10.1145/3300061.3345447.

---

## MEDIA 论文算法原文分析与实现对比

### 一、论文原文算法步骤

#### Algorithm 1: Edge Selection for MEDIA Partitioning

**输入**：模型图 G=(V,E)；节点层级 L(v)；边权重 w(e)
**输出**：边子集 M

```
1  M ← ∅, P ← ∅
2  for u ∈ V following increasing order of level do
3      for v ∈ Succ(u) following the priority on edges do
4          if (|Pre(v)| ≠ 1) and (|Succ(u)| ≠ 1) then continue
5          M ← M ∪ {(u,v)}
6          for w ∈ succ(u) do
7              if L(u) = L(w)−1 and there is an (w',w) ∈ M then
8                  M ← M \ {(u,v)}
9              end
10         end
11     end
12 end
```

**核心逻辑**：
- **Constraint 1**（第 4 行）：只选择满足 `|Pre(v)|==1 OR |Succ(u)|==1` 的边——即 v 只有一个前驱，或 u 只有一个后继。
- **Constraint 2**（第 6-9 行，fork 保护）：将 (u,v) 加入 M 后，检查 u 的所有后继 w：若 L(u)=L(w)−1（w 是 u 的直接下一层级）且已有另一条边 (w',w) ∈ M，则移除 (u,v)。防止 fork 点的多条分支同时被选入 M。
- **遍历顺序**：按节点层级递增遍历 u；对每个 u 的后继 v 按**边权重优先级**遍历。

#### Algorithm 2: Memory-aware MEDIA Partitioning

**输入**：边子集 M
**输出**：分区结果 P

```
1  for (u,v) ∈ M do
2      if Both vertices in (u,v) are not collapsed then
3          if Check({u}, {v}) then
4              Generate a new partition P = {u,v}
5              P ← P ∪ P
6          end
7      else if Both vertices in (u,v) are collapsed then
8          Find partition P ∈ P includes u
9          Find partition P' ∈ P includes v
10         if Check(P, P') then
11             P ← P ∪ P'
12         end
13     else if Only one vertex in (u,v) is not collapsed then
14         Find partition P ∈ P with the collapsed vertex
15         Let w denote the uncollapsed vertex in (u,v)
16         if Check(P, {w}) then
17             P ← P ∪ {w}
18         end
19     end
20 end
```

**Check() 函数**（第 21-24 行）：
```
Function Check(P, P'):
    if T̄(P∪P') ≤ T̄(P) + T̄(P,P') + T̄(P')  OR  m(P∪P') ≤ E then
        return true
    return false
```

**核心逻辑**：
- 遍历 M 中的每条边 (u,v)，根据 u 和 v 的"折叠"（collapsed）状态分三种情况：
  - **Case 1**（两个都未折叠）：Check 两个单节点，通过则创建新分区
  - **Case 2**（两个都已在分区中）：找到各自分区 P 和 P'，Check 后合并
  - **Case 3**（一个已折叠，一个未折叠）：将未折叠节点加入已有分区
- **Check() 使用 OR 条件**：满足**任一**条件即合并：
  - `m(P∪P') ≤ E`：合并后内存不超 EPC，无换页代价
  - `T̄(P∪P') ≤ T̄(P) + T̄(P,P') + T̄(P')`：换页代价 < 通信 + 分开执行代价
- **关键：Algorithm 2 是论文中唯一的分区合并步骤。论文中没有第二阶段的额外合并。**

#### Algorithm 3: MEDIA Assignment Algorithm

```
1  for P ∈ P do
2      Calculate the priority of partition P following (11)
3  end
4  for P ∈ P following decreasing order of priorities do
5      T ← ∅
6      for n ∈ N do
7          Calculate FT(P) on server n via (5)-(7)
8          T ← T ∪ {FT(P)}
9      end
10     Find the minimum FT(P) from set T and assign P to server s
11 end
```

**核心逻辑**：
- **优先级**（Eq.11）：`Priority(P) = max_{P' ∈ succ(P)} { T(P) + T(P,P') + Priority(P') }`，叶分区 Priority = T(P)（bottom-up critical path，类似 HEFT upward rank）。
- **贪心调度**：按优先级降序，对每个分区遍历所有服务器计算最早完成时间 FT，选 FT 最小的服务器分配。

### 二、当前实现 `alg_media.py` 与论文的差异

| # | 方面 | 论文原文 | 当前实现 | 严重程度 |
|---|------|---------|---------|---------|
| **D1** | 是否有 Stage 2 合并 | **没有**。Algorithm 2 是唯一的分区步骤 | 有 `_stage2_merge()`：在 M-edge contraction 后，对分区 DAG 全部链式边做额外 Check()-based merge | **严重** — 论文不存在此步骤，导致过度合并 |
| **D2** | Algorithm 1 是否有 Constraint 2'（join protection） | **没有**。只有 Constraint 2（fork protection） | 添加了 Constraint 2'：对 v 的前驱做对称检查 | **中等** — 论文中不存在，减少 M 中有效边数 |
| **D3** | M 边处理顺序 | Algorithm 2 按 Algorithm 1 的插入顺序处理 M 边（层级递增 × 边权优先级） | `_contract_M_edges()` 将所有 M 边按权重**全局降序排序** | **中等** — 不同处理顺序导致不同合并路径 |
| **D4** | Check() 逻辑 | `T̄(P∪P') ≤ T̄(P) + T̄(P,P') + T̄(P') OR m(P∪P') ≤ E` | Case A: `m ≤ E → true`; Case B: `t_merged ≤ t_p1 + t_p2 + t_comm` | **无差异** — 逻辑等价 |
| **D5** | Cycle check | 未显式提及 | `_would_cause_cycle()` 每次合并前检测 | **无差异** — DAG 合并隐含需要 |
| **D6** | Algorithm 3 (调度) | Priority-based list scheduling | `schedule()` + `_compute_priorities()` | **无差异** — 实现正确 |

### 三、关键问题分析

**D1 是根本原因**：论文的完整流程是 `Algorithm 1 → Algorithm 2 → Algorithm 3`，没有额外的合并阶段。当前实现在 Algorithm 2 之后插入了 `_stage2_merge()`，该函数在分区 DAG 上寻找链式边并反复合并，将 Algorithm 2 留下的多个小分区压缩成少数几个大分区。即使限制为 AND 条件（out_degree==1 AND in_degree==1），分支内部的连续分区仍然是链式的，会被逐步合并。

**D2 的影响次要但存在**：Constraint 2' 过度保护了 join 点，使 M 中边数减少。但即使去掉它，Stage 2 仍会把分区合并回去。

**D3 值得关注**：论文在 Algorithm 1 中按 "priority on edges" 处理后继，M 边在 Algorithm 2 中按插入顺序处理。当前实现的全局排序可能改变合并路径。

### 四、修复建议

1. **删除 `_stage2_merge()`** — 论文中不存在的步骤，是过度合并的根源
2. **删除 Constraint 2'** — 恢复论文 Algorithm 1 原始逻辑
3. **修正 M 边处理顺序** — Algorithm 2 应保持 Algorithm 1 的插入顺序，而非全局排序

---

## 项目结构

```
├── alg_occ.py                           # OCC 算法 (单机基线)
├── alg_dina.py                          # DINA 算法 (负载比例分区)
├── alg_media.py                         # MEDIA 算法 (度约束合并)
├── alg_ours.py                          # Ours 算法 (TP + HEFT)
├── common.py                            # 共享数据结构 + 成本模型
├── loader.py                            # CSV 模型加载器
├── run_all_experiments.py               # 实验主程序 (一键运行 3 个实验 + 生成图表)
├── requirements.txt
├── datasets_260120/                     # 模型 CSV (7 个模型)
├── exp_results/                         # 实验结果 CSV
│   ├── exp1_fixed_comparison/
│   ├── exp2_network_ablation/
│   └── exp3_server_ablation/
├── figures/                             # 图表 PNG + PDF
│   ├── exp1/
│   ├── exp2/
│   └── exp3/
├── docs/                                # 设计文档
│   ├── unified-weight-outside-epc-memory-model.md
│   └── dina-method-configuration-analysis.md
├── model_struct_visualization/          # DAG 可视化工具
│   ├── visualize_model.py
│   ├── visualize_alg.py
│   └── batch_visualize.py
├── lab-notebook/                        # 实验笔记
└── paper_reference/                     # 论文参考图
```

## 模型可视化工具

`model_struct_visualization/visualize_model.py` 可将 DNN 模型的层级依赖关系可视化为交互式 HTML 图形。

### 功能特性

- **交互式图形**：支持缩放、平移、拖拽节点
- **动态着色**：可根据任意列（如 `group`、`type` 或未来的 `partition_id`）对节点着色
- **悬停信息**：鼠标悬停显示层的详细性能指标
- **层级布局**：清晰展示数据流方向

### 使用方法

```bash
# 批量处理所有模型（推荐）
python model_struct_visualization/batch_visualize.py

# 基础用法（按 group 着色，处理单文件）
python model_struct_visualization/visualize_model.py --input datasets_260120/bert_base.csv

# 按操作类型着色
python model_struct_visualization/visualize_model.py --input datasets_260120/bert_base.csv --color-by type

# 指定输出文件路径
python model_struct_visualization/visualize_model.py --input datasets_260120/bert_base.csv --output custom_viz.html
```

### 可视化模型分区

当使用分区算法处理模型后，可将 `partition_id` 列添加到 CSV 中，然后使用相同工具可视化分区结果：

```bash
# 分区后的可视化
python model_struct_visualization/visualize_model.py --input partitioned_model.csv --color-by partition_id
```

### 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--input, -i` | 输入 CSV 文件路径 | (必需) |
| `--output, -o` | 输出 HTML 文件路径 | `<input>_viz.html` |
| `--color-by, -c` | 用于节点着色的列名 | `group` |
| `--layout, -l` | 布局算法 (`hierarchical` 或 `physics`) | `hierarchical` |

## 算法分区结果可视化

`model_struct_visualization/visualize_alg.py` 提供了一个集成入口，可以直接运行指定的模型分割算法并实时生成分区后的可视化 HTML。

### 使用方法

```bash
# 运行并查看 Ours 算法的分区结果（默认 4 服务器, 100Mbps）
python model_struct_visualization/visualize_alg.py --model datasets_260120/bert_base.csv --alg ours

# 查看 DINA 算法（严格受限）的分区结果
python model_struct_visualization/visualize_alg.py --model datasets_260120/bert_base.csv --alg dina --servers 8

# 查看 MEDIA 算法结果
python model_struct_visualization/visualize_alg.py --model datasets_260120/bert_base.csv --alg media --bw 10
```

### 命令行参数 (`visualize_alg.py`)

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model, -m` | 模型 CSV 路径 | (必需) |
| `--alg, -a` | 算法选择 (`ours`, `media`, `dina`, `occ`) | `ours` |
| `--servers, -s` | 集群服务器数量 | 4 |
| `--bw, -b` | 网络带宽 (Mbps) | 100 |
| `--output, -o` | 输出 HTML 路径 | (自动生成) |

### 批量可视化算法分区结果

如果您希望一次性运行所有算法（DINA, MEDIA, Ours, OCC）并为所有模型生成可视化报告，可以使用以下脚本：

```bash
python model_struct_visualization/batch_alg_visualize.py
```

该脚本将结果按模型存储在 `model_struct_visualization/outputs/<model_name>/partitions/` 目录下。


在引入了基于 Intel SGX 加密带宽瓶颈的 **动态惩罚模型 ($Penalty = 1.5 + 3 \times (Ratio-1)$)** 后，实验结果更加贴近真实硬件表现：

### 1. 单服务器基准 (Consistency Check)
*   **现象**: 当 $N=1$ 时，修正 SGX 换页开销模型后，所有算法（OCC, DINA, Ours）的延迟完全一致（例如 ViT @ 2199ms）。
*   **解读**: 这验证了物理建模的正确性——在单机环境下，无论调度策略如何，都无法通过并行获益，且必须通过换页机制（Demand Paging 或 Context Switch）来处理大模型，因此开销底线是固定的。

### 2. MEDIA 与 DINA 的博弈
*   **线性模型 (BERT, ViT)**: 
    *   随着服务器增加，DINA 和 DINA 性能反而下降（例如 ViT 从 2199ms -> 8761ms）。
    *   **原因**: 盲目的 Round-Robin 调度在线性依赖链上引入了巨大的跨服务器通信开销，远超并行计算带来的微弱收益。这证实了在低带宽边缘环境下，粗粒度的分布策略往往适得其反。

### 2. Ours 的独特优势
*   **InceptionV3 (并行模型)**:
    *   Ours 依然在 InceptionV3 上保持了最优性能 (**1367ms** vs DINA 1505ms)。
    *   这证明了 Ours 的优势**不依赖于冒险的内存超限**，而是源于其对 **DAG 并行结构** 的有效利用。
    *   在 54 个细粒度分区中，并行计算节省的时间成功抵消了通信开销。

### 总结
本仿真证明了在 SGX 边缘集群中：
1.  简单地用“换页”换“通信”（MEDIA 策略）在高惩罚系数下往往得不偿失。
2.  **真正的优化空间在于并行计算**（Ours 策略）：通过拓扑感知的细粒度划分，最大化利用集群算力才是打破时延瓶颈的关键。

## 运行方法

### 1. 环境配置

使用 [uv](https://docs.astral.sh/uv/) 管理 Python 环境与依赖。

```bash
# 安装 uv（如尚未安装）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 同步依赖（自动创建虚拟环境）
uv sync
```

### 2. 运行单次实验

```bash
python run_all_experiments.py
```

### 3. 运行完整实验流水线（推荐）

使用统一脚本一次性完成所有实验和图表生成：

```bash
python run_all_experiments.py
```

该脚本将自动执行以下任务：

| 步骤 | 说明 | 输出位置 |
|------|------|----------|
| 1 | 固定配置对比 (Exp1) | `exp_results/exp1_fixed_comparison/` |
| 2 | 网络带宽消融 (Exp2) | `exp_results/exp2_network_ablation/` |
| 3 | 服务器异构消融 (Exp3) | `exp_results/exp3_server_ablation/` |
| 4 | 生成全部图表 | `figures/exp1/`, `exp2/`, `exp3/` |

**实验配置**（可在脚本中修改）：
- 服务器数量 (Exp3)：1–8 台异构递增
- 网络带宽 (Exp2)：0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500 Mbps
- Exp1 固定：4×Xeon_IceLake, 100 Mbps
- 测试模型：InceptionV3, VGG-16, YOLOv5, BERT-large, ALBERT-large, ViT-large, ResNet-50

图表合并功能已集成在 `run_all_experiments.py` 的 `generate_combined_charts()` 中，运行时自动生成。

---

## 分布式多 TEE 节点推理工作流

本节详细描述了在真实分布式 SGX 集群中执行 DNN 推理的完整工作流程，包括系统初始化、安全信道建立、分区调度与执行等全部阶段。

### 1. 系统架构总览

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                               Distributed SGX Inference System                           │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                          │
│   ┌──────────────────┐         ┌──────────────────┐         ┌──────────────────┐        │
│   │   Edge Server 1   │◄───────►│   Edge Server 2   │◄───────►│   Edge Server N   │        │
│   │  ┌────────────┐  │   TLS   │  ┌────────────┐  │   TLS   │  ┌────────────┐  │        │
│   │  │  Enclave 1  │  │◄───────►│  │  Enclave 2  │  │◄───────►│  │  Enclave N  │  │        │
│   │  │ Partition A │  │  SIGMA  │  │ Partition B │  │  SIGMA  │  │ Partition C │  │        │
│   │  └────────────┘  │         │  └────────────┘  │         │  └────────────┘  │        │
│   │       │ EPC      │         │       │ EPC      │         │       │ EPC      │        │
│   └───────┼──────────┘         └───────┼──────────┘         └───────┼──────────┘        │
│           ▼                            ▼                            ▼                    │
│   ┌───────────────┐            ┌───────────────┐            ┌───────────────┐           │
│   │   Local DRAM   │            │   Local DRAM   │            │   Local DRAM   │           │
│   │ (Encrypted Swap)│            │ (Encrypted Swap)│            │ (Encrypted Swap)│           │
│   └───────────────┘            └───────────────┘            └───────────────┘           │
│                                                                                          │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

### 2. 端到端推理时序图

一次完整的分布式推理请求经历以下阶段：

```
时间轴 ──────────────────────────────────────────────────────────────────────────────────────►

阶段 1: 系统初始化 (一次性)
├─ Enclave 创建 (ECREATE/EADD/EINIT) ──────────────────────┤
│  各节点: 50-200 ms                                        │

阶段 2: 安全信道建立 (首次通信)
├─ Remote Attestation (DCAP/EPID) ─┤
│  每对节点: 100-500 ms             │
├─ SIGMA 密钥协商 ──────────────────┤
│  每对节点: 10-50 ms               │

阶段 3: 模型分发与加载 (每次推理)
├─ 分区权重传输 ───────────────────────────────────────────┤
│  取决于模型大小和网络带宽                                  │
├─ EPC 换入 (静态内存) ────────────────────────────────────┤
│  swap_time = static_mem / 1.0 MB/ms                       │

阶段 4: 分区执行与数据流转
├─ Partition A @ Server 1 ─────────┤
│  exec_time = workload × penalty   │
│                      ├─ 激活值传输 (RTT + BW) ─┤
│                      │  latency = RTT + data/BW │
│                                   ├─ Partition B @ Server 2 ─────────┤
│                                   │  exec_time = workload × penalty   │
│                                                        ├─ 激活值传输 ─┤
│                                                                       ├─ Partition C ─┤
阶段 5: 结果聚合
                                                                                        ├─ 返回 ─┤
```

### 3. 分阶段开销建模

#### 3.1 跨节点网络延迟建模

在真实边缘网络中，跨节点通信的时延不仅包含数据传输时间，还需要考虑网络协议栈的固有开销：

| 开销组件 | 物理来源 | 建模公式 | 典型值 |
|----------|----------|----------|--------|
| **RTT (往返延迟)** | 物理链路传播 + 交换机排队 + 协议处理 | 固定值 `RTT_MS` | **1-50 ms** (边缘网络) |
| **传输时延** | 数据量 / 有效带宽 | `data_bytes / (bandwidth_mbps / 8)` | 取决于配置 |
| **TCP 握手** | 三次握手 (首次连接) | `1.5 × RTT` | 按需 |
| **TLS 握手** | 证书验证 + 密钥交换 (首次安全连接) | `2 × RTT + crypto_time` | 5-20 ms |

**完整网络通信时延公式**：

```
T_network = RTT + T_transmission + T_tls_overhead

其中：
- T_transmission = data_mb / (bandwidth_mbps / 8.0 / 1000.0)  [ms]
- T_tls_overhead = 0 (如果连接已建立) 或 ~10 ms (首次连接)
```

**关键参数建议值**：

| 网络类型 | RTT (ms) | 典型带宽 | 使用场景 |
|----------|----------|----------|----------|
| 同机架 (Intra-Rack) | 0.1-0.5 | 1-10 Gbps | 数据中心内部 |
| 同数据中心 | 0.5-2 | 100 Mbps-1 Gbps | 机房内跨机架 |
| 边缘集群 (LAN) | 1-10 | 10-100 Mbps | 园区/工厂边缘 |
| 广域边缘 (WAN) | 10-100 | 1-50 Mbps | 跨城市边缘节点 |

#### 3.2 SGX Remote Attestation 开销建模

在分布式 SGX 系统中，**每对 Enclave 首次通信前**必须完成远程证明（Remote Attestation），以验证对方运行在真实的 SGX 硬件上且代码未被篡改。

##### 3.2.1 远程证明流程

```
┌─────────────────┐                                    ┌─────────────────┐
│   Enclave A     │                                    │   Enclave B     │
│   (Prover)      │                                    │   (Verifier)    │
└────────┬────────┘                                    └────────┬────────┘
         │                                                      │
         │  ──────────────── 1. Challenge (Nonce) ────────────► │
         │                                                      │
         │  ◄───────────── 2. Quote (REPORT + Signature) ────── │
         │     包含: MRENCLAVE, MRSIGNER, ISV_SVN, 用户数据      │
         │                                                      │
         │  ──────────── 3. 验证 Quote (IAS/PCCS) ────────────► │
         │     EPID: 联系 Intel Attestation Service              │
         │     DCAP: 本地验证 + Quoting Enclave                  │
         │                                                      │
         │  ◄────────────── 4. 验证结果 ─────────────────────── │
         │                                                      │
         │  ══════════════ 5. SIGMA 密钥协商 ═══════════════════ │
         │     建立 AES-GCM 加密信道                             │
         │                                                      │
         ▼                                                      ▼
    ┌─────────────────────────────────────────────────────────────┐
    │              Secure Channel Established                      │
    │         后续通信使用协商的会话密钥加密                         │
    └─────────────────────────────────────────────────────────────┘
```

##### 3.2.2 各阶段时延分解

| 阶段 | EPID (SGX1) | DCAP (SGX2/云) | 说明 |
|------|-------------|----------------|------|
| **Quote 生成** | 10-30 ms | 5-15 ms | QE 生成签名 |
| **Quote 传输** | ~RTT | ~RTT | 网络往返 |
| **Quote 验证** | 100-500 ms | 10-50 ms | EPID 需联网; DCAP 本地 |
| **SIGMA 协商** | 10-30 ms | 10-30 ms | Diffie-Hellman + 签名 |
| **会话密钥导出** | 1-5 ms | 1-5 ms | HKDF/AES 密钥扩展 |

**远程证明总开销公式**：

```
T_attestation = T_quote_gen + T_quote_verify + T_sigma + T_key_derive

典型值：
- EPID 模式: 150-600 ms (需联系 Intel 服务器)
- DCAP 模式: 30-100 ms (本地验证)
```

**建模参数建议**：

```python
# DCAP 模式 (推荐用于边缘集群)
ATTESTATION_OVERHEAD_MS = 50.0  # Quote 生成 + 本地验证

# SIGMA 密钥协商
SIGMA_HANDSHAKE_MS = 20.0       # ECDH + 签名验证

# 首次跨节点通信总开销
FIRST_HOP_OVERHEAD_MS = ATTESTATION_OVERHEAD_MS + SIGMA_HANDSHAKE_MS  # ~70 ms
```

> **重要假设**：本仿真假设所有节点在系统启动时完成相互证明，因此推理阶段不计入证明开销。若需要模拟动态加入节点的场景，需将 `FIRST_HOP_OVERHEAD_MS` 加入首次通信时延。

##### 3.2.3 会话密钥复用

建立安全信道后，后续通信仅需使用协商的 AES-GCM 密钥加解密，开销极低（~1 µs/KB）。本仿真假设：

- **单次推理内**：同一节点对之间的多次数据传输复用已建立的安全信道
- **跨推理请求**：会话保持有效，无需重新证明

#### 3.3 Enclave 初始化开销建模

每个 SGX 节点在首次加载 Enclave 时需要执行昂贵的初始化操作：

##### 3.3.1 Enclave 生命周期

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Enclave Lifecycle                                 │
├──────────────────┬──────────────────┬──────────────────┬────────────────┤
│   1. ECREATE     │   2. EADD × N    │   3. EINIT       │  4. EENTER     │
│   创建 SECS      │   逐页加载代码/数据│   完成初始化     │  进入执行      │
│   ~0.1 ms        │   ~0.1 ms/page   │   ~10-50 ms      │  ~5 µs         │
└──────────────────┴──────────────────┴──────────────────┴────────────────┘
                                │
                                ▼
                    ┌─────────────────────────┐
                    │  Enclave Ready State    │
                    │  可接受 ecall 调用       │
                    └─────────────────────────┘
```

##### 3.3.2 初始化阶段详解

| 阶段 | 指令/操作 | 时延 | 说明 |
|------|-----------|------|------|
| **ECREATE** | 创建 Enclave 控制结构 (SECS) | ~0.1 ms | 分配 EPC 元数据页 |
| **EADD** | 添加页面到 Enclave | ~0.1 ms/page | 包含代码、全局数据、堆栈 |
| **EINIT** | 验证签名并初始化 | 10-50 ms | Launch Enclave 验证、MRENCLAVE 计算 |
| **EENTER** (首次) | 首次进入 Enclave | ~5-10 µs | TCS 初始化 |

**Enclave 初始化总开销公式**：

```
T_enclave_init = T_ecreate + N_pages × T_eadd + T_einit

其中：
- T_ecreate ≈ 0.1 ms
- T_eadd ≈ 0.1 ms/page
- T_einit ≈ 20-50 ms (取决于 Enclave 大小和 Launch Policy)
- N_pages = Enclave 代码/数据页数 (通常 100-1000 页)

典型值：
- 小型 Enclave (< 10 MB): 30-50 ms
- 中型 Enclave (10-50 MB): 50-100 ms
- 大型 Enclave (> 50 MB): 100-200 ms
```

**建模参数建议**：

```python
# 基础 Enclave 初始化开销
ENCLAVE_INIT_BASE_MS = 30.0      # ECREATE + EINIT 固定开销

# 按页加载开销
EADD_PER_PAGE_MS = 0.0001        # 0.1 µs/page (批量优化后)

# 总初始化开销
def enclave_init_cost(enclave_size_mb):
    num_pages = enclave_size_mb * 1024 / 4  # 4 KB/page
    return ENCLAVE_INIT_BASE_MS + num_pages * EADD_PER_PAGE_MS
```

> **重要假设**：本仿真假设所有节点的 Enclave 在系统启动时预初始化完成，每次推理不重新创建 Enclave。若需模拟冷启动场景，需将 `enclave_init_cost()` 加入首分区执行前的时延。

### 4. 统一时延模型

综合上述建模，一次分布式推理的端到端时延由以下部分组成：

```
T_total = T_init + T_attestation + Σ(T_partition) + Σ(T_network)

其中：
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ T_init (一次性/冷启动)                                                               │
│   = Σ enclave_init_cost(node_i)                                  [每节点 30-200 ms] │
├─────────────────────────────────────────────────────────────────────────────────────┤
│ T_attestation (首次通信)                                                            │
│   = num_node_pairs × (ATTESTATION_OVERHEAD + SIGMA_HANDSHAKE)      [每对 50-100 ms] │
├─────────────────────────────────────────────────────────────────────────────────────┤
│ T_partition (每分区)                                                                │
│   = T_swap_in + T_exec × penalty + T_enclave_entry                                  │
│   = (static_mem / PAGING_BW) + (workload × penalty) + 0.005 ms                      │
├─────────────────────────────────────────────────────────────────────────────────────┤
│ T_network (每次跨节点传输)                                                           │
│   = RTT + (data_bytes / bandwidth) + T_tls_overhead                                 │
│   注: 同节点内分区切换无网络开销，仅有 Context Switch 开销                            │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

### 5. 建模参数汇总

以下参数定义在 `common.py` 中（或建议添加）：

| 参数 | 含义 | 默认值 | 来源 |
|------|------|--------|------|
| `EPC_EFFECTIVE_MB` | SGX EPC 可用内存 | 93.0 MB | Intel SGX1 规格 - 元数据 |
| `PAGING_BANDWIDTH_MB_PER_MS` | EPC 换页带宽 | 1.0 MB/ms | AES-NI 加密瓶颈实测 |
| `PAGE_FAULT_OVERHEAD_MS` | 单页错误处理 | 0.03 ms | SGX 异常处理实测 |
| `ENCLAVE_ENTRY_EXIT_OVERHEAD_MS` | ecall/ocall 开销 | 0.005 ms | Intel SGX SDK 测量 |
| `RTT_EDGE_MS` | 边缘网络 RTT | 5.0 ms | 典型园区网络 |
| `ATTESTATION_OVERHEAD_MS` | 远程证明开销 (DCAP) | 50.0 ms | DCAP 本地验证 |
| `SIGMA_HANDSHAKE_MS` | SIGMA 密钥协商 | 20.0 ms | ECDH + 签名 |
| `ENCLAVE_INIT_BASE_MS` | Enclave 初始化基础开销 | 30.0 ms | ECREATE + EINIT |

### 6. 仿真简化说明

为聚焦于调度算法对比，本仿真采用以下简化假设：

| 简化项 | 假设 | 影响 |
|--------|------|------|
| **Enclave 预初始化** | 所有节点 Enclave 在推理前已初始化 | 不计 `T_init` |
| **证明已完成** | 节点间已完成相互证明，安全信道已建立 | 不计 `T_attestation` |
| **RTT 忽略** | 网络延迟仅考虑带宽，忽略固定 RTT | 适用于高吞吐场景 |
| **TLS 开销忽略** | 安全信道加解密开销计入带宽限制 | 已隐式建模 |

> 若需完整建模冷启动或动态节点加入场景，可在调度器中启用上述开销参数。

---

## Intel SGX EPC 换页开销建模

本仿真系统对 SGX 的两种换页开销进行了统一建模，确保不同算法之间的一致性。

### 1. SGX 内存架构

```
┌─────────────────────────────────────────────────────────┐
│                      System DRAM                        │
│  ┌─────────────────────────────────────────────────┐   │
│  │         PRM (Processor Reserved Memory)          │   │
│  │  ┌─────────────────────────────────────────┐    │   │
│  │  │              EPC (128 MB)               │    │   │
│  │  │  ├── Usable Pages (~93 MB)              │    │   │
│  │  │  └── Metadata (~35 MB)                  │    │   │
│  │  └─────────────────────────────────────────┘    │   │
│  └─────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────┐   │
│  │        Encrypted Swap Area (VA Pages)           │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

### 2. 两种换页开销

| 开销类型 | 触发场景 | 建模方式 | 物理机制 |
|----------|---------|---------|---------|
| **Demand Paging** (按需换页) | 分区内存 > EPC | `calculate_penalty()` | 运行时持续发生的 page fault |
| **Context Switch** (上下文切换) | 分区之间切换 | `swap_time = (mem1+mem2) / 1.0 MB/ms` | 一次性完整的 swap out + swap in |

### 3. Demand Paging（惩罚因子）

当分区工作集超过 EPC 时，CPU 访问不在 EPC 中的页面会触发 page fault，导致：
- **Swap Out**: 将 EPC 中的页加密后写入 DRAM
- **Swap In**: 将 DRAM 中的页解密后加载到 EPC

惩罚公式：
```
Penalty = 1.0 + δ + γ × (ratio - 1)
其中：
- δ = 0.5（基础开销：上下文切换等）
- γ = 3.0（加密带宽瓶颈系数）
- ratio = partition_memory / EPC_EFFECTIVE_MB
```

### 4. Context Switch（分区切换开销）

当执行从一个分区切换到另一个分区时，需要：
1. 将前一分区的数据从 EPC 换出到 DRAM（加密）
2. 将下一分区的数据从 DRAM 加载到 EPC（解密）

公式：
```
swap_time = (prev_partition.memory + next_partition.memory) / PAGING_BANDWIDTH
PAGING_BANDWIDTH ≈ 1 GB/s = 1.0 MB/ms（受 AES 加密引擎限制）
```

### 5. 算法中的应用

| 算法 | Demand Paging | Context Switch | 网络通信 |
|------|---------------|----------------|----------|
| **DINA** | ✅ (分区 > EPC 时) | ✅ (同服务器分区切换) | ✅ (跨服务器) |
| **MEDIA** | ✅ (分区 > EPC 时) | ✅ (同服务器分区切换) | ✅ (跨服务器) |
| **Ours** | ✅ (分区 > EPC 时) | ✅ (HEFT 调度考虑) | ✅ (跨服务器) |
| **OCC** | ✅ (分区 > EPC 时) | ✅ (单服务器串行) | ❌ (无网络) |

### 6. 关键参数

定义在 `common.py` 中：
```python
EPC_EFFECTIVE_MB = 93.0           # SGX EPC 可用内存 (128 - 35 MB)
PAGING_BANDWIDTH_MB_PER_MS = 2.0  # 换页带宽 (2 GB/s)
```

此统一建模确保了：
- **单服务器时**：所有算法的时延一致（都包含 context switch 开销）
- **多服务器时**：网络通信与本地换页开销互斥（不会重复计算）

---

## HPA 张量并行 (Tensor Parallelism) 仿真模型

本节详细介绍 `common.py` 中 `hpa_cost()` 函数所实现的 **张量并行代价模型**。该模型用于 HPA (Hybrid Parallel Algorithm) 算法决策"是否将某个算子拆分为多个并行分片"。

### 1. 张量并行基本原理

**张量并行 (Tensor Parallelism, TP)** 是将单个算子（如矩阵乘法）的计算沿某一张量维度切分到多个设备上并行执行的技术。

```
                    ┌──────────────────────────────────────────────────────┐
                    │           原始算子 (单节点)                           │
                    │                                                       │
                    │   X [M×K] × W [K×N] = Y [M×N]                         │
                    │   计算量: O(M×K×N)                                    │
                    │   内存: O(K×N) 权重 + O(M×N) 激活                     │
                    └──────────────────────────────────────────────────────┘
                                           │
                                           │ 张量并行 (k=2)
                                           ▼
    ┌─────────────────────────────────┐         ┌─────────────────────────────────┐
    │         分片 0                   │         │         分片 1                   │
    │                                  │         │                                  │
    │  X × W₀ [K×N/2] = Y₀ [M×N/2]     │         │  X × W₁ [K×N/2] = Y₁ [M×N/2]     │
    │  计算量: O(M×K×N/2)              │         │  计算量: O(M×K×N/2)              │
    │  内存: O(K×N/2) + O(M×N/2)       │         │  内存: O(K×N/2) + O(M×N/2)       │
    └───────────────┬─────────────────┘         └───────────────┬─────────────────┘
                    │                                           │
                    └──────────────────┬────────────────────────┘
                                       │ AllReduce 同步
                                       ▼
                              Y = Concat(Y₀, Y₁) 或 Y = Sum(Y₀, Y₁)
```

### 2. HPA 代价模型公式

`hpa_cost()` 函数计算将层拆分为 $k$ 个并行分片后的**总执行代价**：

$$
\text{Cost}(v, k) = T_{\text{comp}} + T_{\text{paging}} + T_{\text{sync}}
$$

#### 2.1 计算时间 ($T_{\text{comp}}$)

```python
t_comp = layer.workload / (k ** efficiency_gamma)
```

- **$\gamma$ (efficiency_gamma)**：并行效率因子，默认 0.9
- **物理含义**：由于 Amdahl 定律、负载不均、启动开销等因素，实际加速比无法达到理想的 $k$ 倍
- **示例**：$k=2, \gamma=0.9$ → 加速比 $= 2^{0.9} \approx 1.87$（而非 2.0）

#### 2.2 内存惩罚 ($T_{\text{paging}}$)

```python
m_activation_shard = m_activation * (1 - α) + m_activation * α / k
m_split = (m_weight / k) + m_activation_shard
penalty = calculate_penalty(m_split)
t_paging = (penalty - 1.0) * t_comp  # if penalty > 1
```

**内存切分模型**：

| 内存组件 | 切分方式 | 每分片占用 |
|---------|---------|-----------|
| 权重 (Weight) | 总是按 $k$ 等分 | $M_w / k$ |
| 激活 (Activation) | 按 `activation_split_ratio` ($\alpha$) 控制 | $M_a \times (1-\alpha) + M_a \times \alpha / k$ |

- **$\alpha = 1.0$**（默认）：激活完全切分，适用于 Column Parallel FC
- **$\alpha = 0.0$**：激活完全复制，适用于 BatchNorm 等需要完整统计的层

**惩罚因子**：当 $M_{\text{split}} > \text{EPC}$ 时，触发 SGX 换页，`calculate_penalty()` 返回 > 1.0 的惩罚乘数。

#### 2.3 同步开销 ($T_{\text{sync}}$)

```python
sync_bytes = layer.output_bytes * 2 * (k - 1) / k
t_sync = network_latency(sync_mb, bandwidth) * sync_probability
```

**Ring AllReduce 通信量公式**：

$$
\text{Sync Data} = 2 \times \frac{k-1}{k} \times \text{Output Size}
$$

| $k$ | 通信量系数 | 说明 |
|-----|-----------|------|
| 2 | $1.0 \times$ | 两节点直接交换 |
| 4 | $1.5 \times$ | 3 个步骤的 Ring 交换 |
| 8 | $1.75 \times$ | 7 个步骤的 Ring 交换 |

### 3. `sync_probability` 参数深度解析

**核心问题**：在实际的 Tensor Parallelism 策略中，是否每个被拆分的层都需要进行 AllReduce？

**答案**：**不是**。现代 TP 策略（如 Megatron-LM）采用 **Column-Row 双层结构**，两层共享一次 AllReduce。

#### 3.1 Megatron-LM 风格 FFN 模块

```
输入 X (完整，需复制到各分片)
        │
        ├─────────────────────────────────────────────┐
        │                                              │
        ▼                                              ▼
┌───────────────────┐                      ┌───────────────────┐
│  Column FC1 (分片0) │                      │  Column FC1 (分片1) │
│  X × W₁₀ = H₀       │                      │  X × W₁₁ = H₁       │
│  (无需同步)          │                      │  (无需同步)          │
└─────────┬─────────┘                      └─────────┬─────────┘
          │ 部分激活 H₀                              │ 部分激活 H₁
          ▼                                          ▼
┌───────────────────┐                      ┌───────────────────┐
│   Row FC2 (分片0)   │                      │   Row FC2 (分片1)   │
│  H₀ × W₂₀ = P₀      │                      │  H₁ × W₂₁ = P₁      │
│  输出: 部分和        │                      │  输出: 部分和        │
└─────────┬─────────┘                      └─────────┬─────────┘
          │ Partial Sum P₀                           │ Partial Sum P₁
          │                                          │
          └──────────────┬───────────────────────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │     AllReduce       │  ← 仅在此处同步 1 次
              │    P₀ + P₁ = Y      │
              └─────────────────────┘
                         │
                         ▼
                    输出 Y (完整)
```

**关键观察**：
- **Column FC1**：输出是独立的部分激活（Partial Activation），**无需同步**
- **Row FC2**：输出是部分和（Partial Sum），**需要 AllReduce**
- **两层共享一次同步**，摊销后每层平均同步概率 = 0.5

#### 3.2 `sync_probability` 取值指南

| 值 | 物理含义 | 适用场景 |
|----|---------|---------|
| **1.0** | 每层独立同步 | 孤立的 TP 层、保守估计 |
| **0.5** | 双层共享一次同步 | **Megatron-style FFN (默认)** |
| **0.0** | 无需同步 | TP 组中间层（如 Column FC1 到 Row FC2） |

**为什么默认值是 0.5**：

大多数 Transformer 模型的 FFN 模块采用 Column-Row 结构，设置 `sync_probability=0.5` 能够：
- ✅ 准确反映平均同步开销
- ✅ 避免过度悲观的代价估算
- ✅ 让 HPA 算法更积极地选择 Tensor Parallelism

### 4. 完整代价公式

综合上述，`hpa_cost()` 的完整数学表达式为：

$$
\text{Cost}(v, k) = \underbrace{\frac{T_{\text{comp}}}{k^\gamma}}_{\text{并行计算}} + \underbrace{(\text{Penalty}(M_{\text{shard}}) - 1) \times T_{\text{comp}}}_{\text{内存惩罚}} + \underbrace{\frac{2(k-1)}{k} \times \frac{\text{Output}}{BW} \times P_{\text{sync}}}_{\text{同步通信}}
$$

其中：
- $M_{\text{shard}} = \frac{M_w}{k} + M_a \times (1 - \alpha + \frac{\alpha}{k})$
- $\alpha$ = `activation_split_ratio`
- $P_{\text{sync}}$ = `sync_probability`
- $\gamma$ = `efficiency_gamma`
- $BW$ = `bandwidth_mbps` (转换为 MB/ms)

### 5. 函数接口

```python
def hpa_cost(
    layer,                              # DNNLayer 对象
    k: int,                             # 并行度 (1, 2, 4, 8...)
    bandwidth_mbps: float,              # 网络带宽 (Mbps)
    efficiency_gamma: float = 0.9,      # 并行效率因子
    activation_split_ratio: float = 1.0,# 激活切分比例 (0.0~1.0)
    sync_probability: float = 0.5       # 同步概率 (0.0~1.0)
) -> float:                             # 返回: 总代价 (ms)
```

### 6. 使用示例

```python
from common import hpa_cost

# 场景1: 保守估计 (旧模型行为)
cost_old = hpa_cost(layer, k=2, bw=500, 
                    activation_split_ratio=0.0, sync_probability=1.0)

# 场景2: Megatron-style 优化估计 (默认)
cost_new = hpa_cost(layer, k=2, bw=500)  
# 等价于: activation_split_ratio=1.0, sync_probability=0.5

# 场景3: 自定义 TP 策略 (如 4 层共享 1 次同步)
cost_custom = hpa_cost(layer, k=4, bw=500, sync_probability=0.25)
```

### 7. 建模验证

以 BERT encoder0_ffn_fc1 层为例 (Workload=20.67ms, Output=1.5MB, BW=500Mbps):

| 模型配置 | k=1 | k=2 | k=4 | k=8 |
|---------|-----|-----|-----|-----|
| **OLD** (α=0, P=1.0) | 20.67 ms | 40.08 ms | 52.37 ms | 56.10 ms |
| **NEW** (α=1, P=0.5) | 20.67 ms | 25.58 ms | 26.44 ms | 26.68 ms |

**结论**：
- OLD 模型：k=2 代价为 k=1 的 **194%**（高估通信开销，TP 看起来"不值得"）
- NEW 模型：k=2 代价为 k=1 的 **124%**（合理估算，TP 虽有开销但显著降低内存压力）

