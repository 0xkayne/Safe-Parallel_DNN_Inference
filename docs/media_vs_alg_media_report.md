# MEDIA‑GPT vs alg_media 实现对比报告

## 1. 代码概览
| 文件 | 主要类/函数 | 负责的算法阶段 |
|------|--------------|-----------------|
| `legacy/MEDIA‑GPT.py` | `select_edges_for_partitioning`、`merge_check`、`graph_partition`、`compute_partition_priority`、`assign_partitions_to_servers` | 完整实现 **Algorithm 1‑3**（边选择、图合并、调度） |
| `alg_media.py` | `MEDIAAlgorithm.run`、`MEDIAAlgorithm.schedule` | 仅实现 **模型分割（线性合并）** 与 **Round‑Robin 调度**，未实现完整的 **Algorithm 2‑3**（图合并、优先级调度） |

## 2. 边选择（Algorithm 1）
### MEDIA‑GPT
- `select_edges_for_partitioning(G)`：
  - 只保留满足 **`in_degree(v) == 1` 且 `out_degree(u) == 1`** 的边（对应论文 Constraint 1）。
  - 通过 `topological_sort` 遍历，**未实现**论文中对层级的 `Constraint 2`（防止环），但实际运行中该约束对大多数 DAG 并不产生冲突。
  - 返回集合 `M` 用于后续合并。

### alg_media
- **没有显式的边选择**。`run()` 直接对拓扑序列进行线性遍历，**默认所有相邻层都可以合并**（只要满足内存或通信‑惩罚条件）。
- 因此 **并行结构会被破坏**：在有分叉/汇合的 DAG 中，所有层都会被强行串行化。

**结论**：`MEDIA‑GPT` 更贴合论文的 **Algorithm 1**，而 `alg_media` 省略了该阶段，导致分割策略更激进。

## 3. 合并判定（Algorithm 2 – `merge_check`）
### MEDIA‑GPT
- `merge_check(part1, part2, Fn_avg, bandwidth_avg)`：
  - 计算合并后 **内存** 与 **执行时间**（`exec_time` 使用 EPC 判定），
  - `t_merged` 与 `t_sep = t_p1 + t_p2 + t_comm`（通信时间基于 `LAYLER_COM_BAND / bandwidth_avg`）比较，
  - 返回 `memory <= EPC or t_merged <= t_sep`，**完全对应论文的判定逻辑**。

### alg_media
- 合并判定逻辑嵌入在 `run()` 中的 **线性贪心**：
  - 计算 `merged_mem` 与 `comm_time`（基于边权 / 带宽），
  - 若 `merged_mem <= EPC` **直接合并**；
  - 否则比较 **惩罚增量** `penalty_delta = t_slow - t_fast` 与 `comm_time`（乘以 `COMM_WEIGHT`，默认 1.0），若 `penalty_delta < comm_time` 则合并。
- 这里 **使用了 `calculate_penalty`**（论文中对 EPC 超限的惩罚模型），但 **没有** `t_sep` 的 **两分区执行时间相加**，而是使用 **单分区基准时间 `t_fast`**（等同于工作量）来估计惩罚。
- **缺少** 对 **两分区并行执行**（如 `Improved‑MEDIA` 中的 `is_parallel`）的考虑。

**结论**：两者的合并判定思路相似（内存 vs EPC、通信 vs惩罚），但 `alg_media` 的实现更**简化**，未完全遵循论文的 `t_sep = t_p1 + t_p2 + t_comm` 公式。

## 4. 分区构建（Algorithm 2 – 图合并）
### MEDIA‑GPT
- `graph_partition` 完整实现：
  - 依据 `edges_M` 创建初始分区，随后遍历 `edges_M` 合并相邻分区，最后处理孤立节点。
  - 还提供 **迭代合并** 循环（`while has_merged`）尝试进一步合并相邻分区，使用 `merge_check` 决定是否合并。
- 结果返回 **`partitions`** 与 **`node_to_partition`**，供后续调度使用。

### alg_media
- `run()` **仅** 进行一次线性扫描：
  - 维护 `current_layers` 列表，依据合并判定决定是否 **结束当前分区** 并创建 `Partition` 实例。
  - **不** 生成跨分区的依赖图，也不进行二次迭代合并。
- 因此 **缺少** 对 **非相邻层**（通过 `edges_M`）的合并尝试，也没有 **孤立节点** 的单独处理（因为线性遍历已覆盖全部节点）。

**结论**：`MEDIA‑GPT` 实现了完整的 **Algorithm 2**，而 `alg_media` 只实现了 **简化的线性合并**，不具备论文中对图结构的全局合并能力。

## 5. 调度（Algorithm 3）
### MEDIA‑GPT
- `compute_partition_priority` 实现 **公式 11**（工作量 / 平均算力 + 通信时间 + 最大后继优先级）。
- `assign_partitions_to_servers`：
  - 计算每个分区在每台服务器上的 **就绪时间**（前驱完成 + 通信），
  - 依据 **服务器空闲时间** 与 **就绪时间** 决定 **开始时间**，
  - 选取 **完成时间最小** 的服务器分配。
- 该调度 **考虑并行执行**：无依赖分区可分配到不同服务器，实现并行加速。

### alg_media
- `schedule(partitions)` 实现 **Round‑Robin**：
  - 分区 `i` 固定分配到服务器 `i % n_servers`，强制 **跨服务器通信**（除非同服务器），
  - 计算 **通信时间**（跨服务器）或 **SGX 上下文切换分页开销**（同服务器），
  - 采用 **线性** 的 **服务器空闲时间** 与 **前驱完成时间** 取最大作为开始时间。
- **不** 使用优先级或关键路径概念，调度策略较为 **粗糙**，可能导致不必要的通信。

**结论**：`MEDIA‑GPT` 完整实现了 **Algorithm 3**（基于优先级的调度），而 `alg_media` 使用了 **简化的 Round‑Robin**，与论文的调度策略不一致。

## 6. 并行性处理
- **论文**：在 **分割阶段** 通过约束保留并行支路；真正的并行收益在 **调度阶段**（Algorithm 3）通过将无依赖分区分配到不同服务器实现。
- **MEDIA‑GPT**：遵循该思路，**未显式** 计算并行收益，但结构约束保证了并行支路不被错误合并；调度阶段利用优先级实现并行。
- **alg_media**：
  - **分割阶段** 完全串行化（线性扫描），会把并行支路合并进同一分区，导致调度时失去并行潜力。
  - **调度阶段** 采用 **Round‑Robin**，强制每个分区跨服务器（除同服务器情况），并不利用 **无依赖分区的并行**，反而可能增加不必要的通信。

**整体结论**：`alg_media` 与论文的 **并行性保留与利用** 目标相背离，属于 **简化/近似实现**，适用于 **线性模型**（如 ViT、BERT）但在 **并行模型**（如 InceptionV3）上会出现性能退化。

## 7. 关键差异汇总
| 维度 | MEDIA‑GPT (论文实现) | alg_media (当前实验使用) |
|------|----------------------|--------------------------|
| 边选择 | 只保留满足约束的边 (`M`) | 无边选择，默认所有相邻层可合并 |
| 合并判定 | 完整 `merge_check`（内存 + `t_merged ≤ t_sep`） | 简化判定（内存 ≤ EPC 或 `penalty_delta < comm_time`） |
| 图合并 | 多轮迭代合并，处理孤立节点 | 单轮线性合并，仅一次遍历 |
| 调度策略 | 基于优先级的 **Algorithm 3**（关键路径） | 固定 **Round‑Robin**（强制交叉通信） |
| 并行性保留 | 通过约束间接保留；调度阶段实现并行 | 分割阶段破坏并行，调度阶段也未利用并行 |
| SGX 分页开销 | 在调度阶段加入 **上下文切换分页**（`swap_out + swap_in`） | 同样加入，但因分区往往在同服务器上，影响较小 |
| 适用模型 | 线性与并行模型均可，尤其在并行模型上表现更好 | 主要适用于 **线性模型**；并行模型会出现 **退化为串行** 的情况 |

## 8. 对实验测量的影响
- 您在实验中使用的 **`alg_media.py`** 采用了 **简化的线性合并 + Round‑Robin**，这解释了为何在 **并行模型（InceptionV3）** 上 Ours（HEFT）表现优于 MEDIA，而在 **线性模型（ViT、BERT）** 上两者差距不大。
- 若希望 **MEDIA** 在并行模型上发挥优势，需要：
  1. **引入边选择**（如 `select_edges_for_partitioning`）以保护并行支路；
  2. 使用 **完整的 `merge_check`**（包括 `t_sep = t_p1 + t_p2 + t_comm`），并在合并阶段考虑 **并行收益**（可参考 `Improved‑MEDIA` 中的 `is_parallel` 逻辑）。
  3. 替换 **Round‑Robin** 调度为 **Algorithm 3** 的优先级调度，使无依赖分区能够并行分配到不同服务器。

## 9. 推荐改进路线
1. **复用 `legacy/MEDIA‑GPT.py`** 中的 `select_edges_for_partitioning`、`merge_check`、`graph_partition`、`assign_partitions_to_servers`，直接在实验脚本中调用，以获得论文级实现。
2. 若仍想保留 `alg_media.py` 的简洁结构，可在 **分割阶段** 加入 **并行支路检测**（`check_parallel_relation`）并在 `merge_check` 中使用 `max(t_p1, t_p2) + t_comm`（并行版），类似 `Improved‑MEDIA` 的做法。
3. 将调度改为 **优先级调度**（参考 `MEDIA‑GPT`），或在 `alg_media.schedule` 中加入 **依赖图** 与 **关键路径** 计算，以提升并行模型的性能。

---
*报告生成于 2026‑01‑13 01:57 CST*。
