# MEDIA 实现与论文《DNN Partitioning and Assignment for Distributed Inference in SGX Empowered Edge Cloud》对比报告

## 1. 论文算法概述
- **Algorithm 1 – Edge Selection**
  - 仅保留满足 **Constraint 1**（`out_degree(u)=1` 或 `in_degree(v)=1`）的边。
  - 通过 **Constraint 2**（层级约束）防止形成环，确保选取的边集合 `M` 不破坏 DAG 的拓扑结构。
  - 目标：在 **保持并行结构** 的前提下，尽可能合并能够降低通信开销的相邻层。

- **Algorithm 2 – Graph Partition (Merge Check)**
  - 对每对相邻分区 `p1, p2` 计算 **合并后内存** `mem(p1∪p2)` 与 **执行时间** `t_merged`（公式 9）。
  - 若 `mem ≤ EPC` → **直接合并**。
  - 否则比较 `t_merged` 与 **分离执行 + 通信时间** `t_sep`（公式 10），满足 `t_merged ≤ t_sep` 时合并。
  - 该策略在 **内存受限** 与 **通信成本** 之间做权衡。

- **Algorithm 3 – Assignment**
  - 为每个分区计算 **优先级** `Priority(p) = T(p) + C(p, succ(p)) + max(Priority(succ(p)))`（公式 11）。
  - 按优先级降序调度，尽可能将 **无依赖的分区分配到不同服务器**，实现并行执行。

- **并行性**
  - 论文 **未在分割阶段显式优化并行度**，而是通过 **约束 1/2** 保证并行支路不被错误合并。
  - 真正的并行收益体现在 **Assignment** 阶段的调度策略。

## 2. 项目中 `legacy/MEDIA‑GPT.py` 实现对照
| 论文要点 | 项目实现 | 是否一致 | 备注 |
|---|---|---|---|
| **Edge Selection (Algorithm 1)** – 只保留 `out_degree(u)=1 && in_degree(v)=1` 的边 | `select_edges_for_partitioning` 中的判断 `if G.in_degree(v) != 1 and G.out_degree(u) != 1: continue`（保留满足条件的边） | ✅ | 与论文约束 1 完全一致。约束 2（层级约束）在代码中被注释掉，实际未执行，但对大多数 DAG 并不会产生环。 |
| **Merge Check (Algorithm 2)** – 内存 ≤ EPC 直接合并；否则比较 `t_merged` 与 `t_sep`（`t_sep = t_p1 + t_p2 + t_comm`） | `merge_check` 计算 `memory`, `workload`, `exec_time`（依据 EPC），随后计算 `t_merged`, `t_sep = t_p1 + t_p2 + t_comm`，返回 `memory <= EPC or t_merged <= t_sep` | ✅ | 完全对应论文的判定逻辑。 |
| **Partition Construction** – 根据 `M` 合并相邻层，处理孤立层 | `graph_partition` 按 `M` 创建新分区、合并已有分区、最后为未分配的节点创建单独分区 | ✅ | 与论文的 Algorithm 2 步骤一致。 |
| **Assignment (Algorithm 3)** – 计算优先级并调度 | `compute_partition_priority` 实现公式 11（`priority = workload/Fn + comm_time + max_succ`），`assign_partitions_to_servers` 按优先级降序遍历并选择最小完成时间的服务器 | ✅ | 与论文描述相符。 |
| **并行性考虑** – 论文在分割阶段仅通过结构约束保护并行支路 | `legacy/MEDIA‑GPT.py` **没有显式并行检测**，但 **Edge Selection** 的约束天然保留并行分支；合并判定仅基于时间/内存，不会主动破坏并行结构。 | ✅（隐式） | 代码实现与论文的“并行性隐式保护”保持一致。 |

## 3. `legacy/Improved‑MEDIA.py`（与 `MEDIA‑GPT‑copy.py`）的改动
- 新增 **`check_parallel_relation`**：遍历两组层的所有路径，若不存在任意祖先/后继关系则判定为并行。
- `select_edges_for_partitioning` 中仍使用相同约束（`in_degree(v)!=1 and out_degree(u)!=1`），但注释说明是 **“改进点”**，实际逻辑未改变。
- `merge_check` 增加 `is_parallel` 参数，若为并行则在 `t_sep` 计算中使用 `max(t_p1, t_p2) + t_comm`（即并行分支的分离时间为慢者加通信），并在返回时对并行情况进行额外判断。
- **并行感知** 仍是 **后置**（在合并阶段判断），而论文并未在分割阶段加入此类显式并行收益模型。

**结论**：Improved‑MEDIA 引入了显式并行检测与并行‑感知的合并判定，这**超出了论文原始设想**（论文仅隐式保护并行），因此实现上**不完全符合**论文的模型描述，但提供了潜在的并行优化方向。

## 4. 综合评估
- **`legacy/MEDIA‑GPT.py`** 完全实现了论文中 **Algorithm 1‑3** 的核心逻辑，且在分割阶段通过约束自然保留并行结构，符合论文的 **“并行性隐式保护”** 设计。
- **`legacy/Improved‑MEDIA.py`** 在此基础上加入了 **并行感知的合并判定**，属于对论文的**扩展尝试**，但并非论文所要求的实现方式。
- 若项目目标是 **忠实复现论文**，应使用 `MEDIA‑GPT.py`（或在 `Improved‑MEDIA` 中关闭并行感知逻辑）。若希望进一步提升并行利用率，可在 `Improved‑MEDIA` 基础上继续完善并行收益模型（例如在 `merge_check` 中加入关键路径长度比较）。

---
**建议**
1. 将 `legacy/MEDIA‑GPT.py` 设为默认实现，保持与论文一致。
2. 若需要并行优化，可保留 `Improved‑MEDIA.py` 并在实验中对比两者的调度效果。
3. 在文档（README）中明确说明两者的差异及对应的论文对应关系。

*报告生成于* **2026‑01‑13 01:50 CST**。
