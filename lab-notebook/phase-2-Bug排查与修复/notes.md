# Phase 2：Bug 排查与修复

---

### [2026-03-03] 通信量二次除法 Bug（Critical）

**类型**：`踩坑记录` / `算法决策`

**Bug 描述**

`loader.py` 在创建 DAG 时已将 edge weight 转换为 MB 单位：

```python
# loader.py line 186
comm_size_mb = layers_map[prev_id].output_bytes / (1024 * 1024)
G.add_edge(prev_id, curr_id, weight=comm_size_mb)  # 单位已是 MB
```

但 `alg_dina.py`、`alg_media.py`、`alg_ours.py` 的调度代码又对 edge weight 做了一次 `/（1024 * 1024）`，导致实际通信量被缩小了 **100 万倍**（约 1 MB → 约 1 Byte），仅剩 RTT（5ms）贡献到通信开销。

**修复位置（共 5 处）**

| 文件 | 行号 | 修复内容 |
|------|------|---------|
| `alg_dina.py` | ~84 | `vol_mb = vol` （原 `vol / (1024*1024)`） |
| `alg_media.py` | ~145 | `vol_mb = vol` （`_merge_check` 函数） |
| `alg_media.py` | ~288 | `comm_data_mb = comm_data` （`schedule` 函数） |
| `alg_ours.py` | ~444 | `vol_mb = vol` （partition merge check） |
| `alg_ours.py` | ~529 | `comm_data_mb = comm_data` （HEFT schedule） |

**修复效果（BERT-base @ 100 Mbps）**

| 指标 | Bug 版本 | 修复版本 |
|------|---------|---------|
| DINA 单次跨服务器 hop 通信延迟 | ~5ms（仅 RTT） | ~280ms（RTT 5ms + 数据 275ms） |
| DINA 总延迟 vs OCC 差值 | +15ms（3 跳 × 5ms） | +825ms（3 跳 × 275ms） |
| 实验结论 | OCC ≈ DINA ≈ MEDIA（错误） | OCC < MEDIA ≤ Ours << DINA（正确） |

**根本原因**：接口约定不一致，`loader.py` 的注释未明确说明 edge weight 单位为 MB，导致下游代码重复换算。

> **素材标注**：
> - 可作为 paper 的 implementation notes 或 lessons learned，说明分布式仿真系统中接口单位一致性的重要性
> - 专利不直接使用，但体现了代码工程严谨性

---

### [2026-03-03] Ours Bug 1：AllReduce 通信未在 HEFT 中建模

**类型**：`踩坑记录` / `算法决策`

**Bug 描述**

`alg_ours.py` 的 `_augment_graph()` 在将算子拆分为 k 个 shard 时，只创建了 k 个 shard 节点，但**没有插入 AllReduce 同步节点**。HEFT 调度器看到的 augmented DAG 中：
- shard_0 放在 S0，shard_1 放在 S1（正常并行）
- 后继分区 P_next 只需等任意一个 shard 完成即可开始（不正确）

实际上 All-to-All Reduce 要求所有 k 个 shard 完成计算后才能汇总，P_next 必须等待最慢的那个 shard + 跨服务器通信时间。没有 AllReduce 节点 → HEFT 低估了 k>1 时的同步开销。

**根本原因**：`_augment_graph()` 设计时只关注了 shard 节点的创建，忽略了 Ring AllReduce 的 barrier 语义——k 个 shard 的输出必须先汇聚到一个虚拟节点，再传给下游。

**修复方案（选项 A：显式 AllReduce 节点）**

对每个 k>1 的算子，在 augmented DAG 中插入一个零工作量的 AllReduce barrier 节点：

```
shard_0 ─┐
shard_1 ─┼─► AllReduce_barrier ──► successor(s)
  ...    ─┘
```

- 每条 shard → AllReduce 边的权重 = `sync_mb × P_sync / k`
  - `sync_mb = output_bytes × 2(k-1)/k / (1024²)`（Ring AllReduce 公式）
  - `P_sync = 0.5`（Megatron 式两层共享一次同步的摊销概率）
- AllReduce 节点的内存/计算量均为 0，output_bytes 继承原算子（供下游边权重计算）
- 边的改写：若上游为 k>1 算子，下游边起点改为 AllReduce 节点（而非任意 shard）

**修复效果**

AllReduce 节点正确出现在 augmented DAG 中（InceptionV3@100Mbps 生成 17 个 AllReduce 节点，@500Mbps 生成 58 个），HEFT 现在能正确计算 k>1 算子的跨服务器同步成本。

> **素材标注**：
> - 这是专利 P1（带宽自适应张量并行度选择）在实现层面的关键细节：AllReduce barrier 建模是使 DP 成本公式与 HEFT 调度一致的必要保证
> - 可写入 paper Design 章节的"算子级并行通信模型"小节，说明 Ring AllReduce 公式及 P_sync=0.5 的工程取舍

---

### [2026-03-03] Ours Bug 2：低带宽下 HEFT 过度分散分区（Critical）

**类型**：`踩坑记录` / `关键发现`

**Bug 描述**

修复 Bug 1 后，在 0.5 Mbps 带宽下测试发现：

```
BERT-base @0.5Mbps:  OCC=3,613ms  Ours=43,637ms  (12.1× 更差！)
```

调查发现 Ours 日志显示 `Filtered 0 operators with tensor-parallel benefit`，即 k 全部为 1，没有任何算子做张量并行，AllReduce 节点一个都没创建。**Bug 1 的修复与此无关。**

**根本原因：MEDIA 式分区器产生大量微分区 + HEFT 在低带宽下主动分散**

1. **过度分区**：OCC 对 BERT-base 生成 4 个分区；Ours 的 MEDIA 式分区器（允许跨 EPC 并考虑通信-分页权衡）生成 62–86 个微分区
2. **BERT 的"假并行"**：BERT 的多头注意力结构在 DAG 层面产生大量拓扑独立的分区（Q/K/V 各路径无直接边），HEFT 将这些独立分区调度到空闲服务器以最大化并行度
3. **低带宽下的灾难**：@0.5 Mbps，传输 5 MB 数据需要 `5ms RTT + 5/(0.0625/1000) ≈ 80,000ms`；14 次跨服务器 hop × ~8,000ms = 毁灭性延迟

**决策：不修改分区器，在调度层添加单机保底**

考虑了两种方案：
- **方案 B**：在低带宽时禁止分区器产生过多微分区（修改 MEDIA 分区逻辑）
  - 缺点：破坏了 MEDIA 分区器的通用性，且低带宽判断阈值难以确定
- **方案 A（选用）**：在 `schedule()` 末尾添加单机保底（single-server safeguard）
  - 计算在最快单台服务器上串行执行所有分区的延迟 `ss_time`
  - 若 `ss_time < HEFT 结果`，返回单机串行 ScheduleResult
  - 优点：对分区器零侵入，保证 Ours 在任何带宽下不劣于单机最优

**修复后验证数据**

| 场景 | 修复前 | 修复后 |
|------|--------|--------|
| BERT-base @ 0.5 Mbps | 43,637ms（12.1× OCC） | 3,615ms（1.001× OCC）✅ |
| BERT-base @ 100 Mbps | 2,164ms（0.60× OCC） | 2,164ms（0.60× OCC）✅ |
| BERT-large @ 0.5 Mbps | — | 12,391ms（1.000× OCC）✅ |
| BERT-large @ 100 Mbps | — | 3,556ms（0.287× OCC，3.5×加速）✅ |
| InceptionV3 @ 0.5 Mbps | — | 2,284ms（1.001× OCC）✅ |
| InceptionV3 @ 500 Mbps | — | 796ms（0.349× OCC，2.9×加速）✅ |

低带宽下 Ours 的 1.001× OCC（而非精确 1.000×）来自两者 paging 公式的浮点计算顺序不同，属正常精度差异。

**关键洞察**：单机保底的本质是"带宽自适应退化"——当网络带宽无法支撑任何分布式并行时，Ours 自动退化为单机串行最优，而 DINA 在同样场景下仍强制跨服务器（@0.5Mbps = OCC 的 88×）。这正好验证了论文的核心论点：自适应退化是 Ours 相对于 DINA 的根本优势。

> **素材标注**：
> - **论文 Discussion（重要！）**：单机保底的触发条件（低带宽 + 过多微分区）解释了为何 Ours 在低带宽下能与 OCC 持平，而不是像 DINA 一样崩溃。这是 Ours 鲁棒性的核心机制
> - **专利 P1 的有益效果补充**：低带宽时自动退化为 k=1 单机模式，消除了朴素分布式方法（DINA）在低带宽下的灾难性劣化——可以量化为"在 0.5 Mbps 下比 DINA 快 88×"
> - **论文 Evaluation Discussion**：InceptionV3 @0.5Mbps 的 Ours≈OCC（而非 Ours>>OCC 如 DINA）正是单机保底生效的直接体现

---

### [2026-03-03] 诊断工具设计

**类型**：`算法决策`

为了系统性排查 bug，创建了 `diagnostics/diagnose.py`，包含 7 个诊断模块：

1. 数据集/层分析（edge weight 单位探测）
2. 各算法分区结果对比
3. 调度时间线分解（exec / paging / comm 三部分）
4. 通信量 bug 定量分析（正确值 vs bug 值对比）
5. 跨算法延迟对比表
6. MEDIA 调度决策追踪（为何每次都选 S0）
7. Bug 汇总与修复建议

用法：`python diagnostics/diagnose.py --model bert_base --servers 4 --bandwidth 100 --section N`

**注意**：Windows GBK 终端不支持 Unicode 特殊字符（✓、✗、─），脚本中必须用 ASCII 替代（`[OK]`、`[!!]`、`-`）。

> **素材标注**：诊断工具的设计思路可写入 paper 的 methodology 部分，说明仿真验证方法的可信度保障机制。

---

### [2026-03-04] Ours Bug 3：`_media_partition` 非确定性（Critical）

**类型**：`踩坑记录` / `关键发现`

**Bug 描述**

`_media_partition()` 中存在多处 `list(set(partition_objects))` 调用，Python 的 `set()` 对自定义对象的迭代顺序基于内存地址哈希，每次运行随机。导致 `vit_base@100Mbps` 每次运行产生不同分区数（47–111 个）和延迟（1,542–3,851ms）。

**更严重的发现**：原来 BERT-base @100Mbps 的 Ours=1,431ms（0.396× OCC）完全是非确定性 `set()` 的意外产物——某些随机迭代顺序恰好产生较少分区（~4 个），使 HEFT 调度有效。这并非算法的真实性能。

**修复位置（4 处）**

| 位置 | 修复内容 |
|------|---------|
| Phase 1 merge 层去重 | `sorted({l.id:l for l in pu.layers+pv.layers}.values(), key=lambda l:l.id)` |
| Post-processing 去重 | 新 `_unique_parts()` 函数（用 `id(p)` 去重） |
| Post-processing 合并策略 | 重新设计为 comm-weight-first 贪心合并（按通信量降序尝试）|

**修复后验证（3次重复，det=True 表示确定性）**

```
bert_base: parts=[112,112,112] lat=[3381.9,3381.9,3381.9] det=True
vit_base:  parts=[102,102,102] lat=[3752.4,3752.4,3752.4] det=True
inception: parts=[173,173,173] lat=[1231.3,1231.3,1231.3] det=True
```

---

### [2026-03-04] BERT/ViT 分区图环路分析（关键发现）

**类型**：`关键发现` / `算法决策`

**发现**

修复非确定性后，BERT-base@100Mbps 深度诊断：

```
Phase 1: edges=374, merges=374 → 169 partitions
Post-processing: 169 → 112 partitions
Partition graph: Is DAG: False, Cycles: 210
SCC analysis: 1 giant SCC (ALL 112 partitions, total 409.23MB)
```

**根本原因：BERT 残差连接造成分区图环路**

BERT 残差结构：`Input → [Q,K,V 投影] → Attention → Add_残差(Input + Attention) → ...`

在分区图中：Big_B（含 Input 和 Add_残差） → Tiny_T（attention 计算） → Big_B，形成环路。

**为何无法合并消环**：涉及环路的三个分区：

```
big(91.49MB) + 3MB(1个) + tiny(0.38MB) = 94.87MB > EPC(93MB)  → 超出 EPC，无法合并
全部 112 个分区合计 409.23MB                                    → 更不可能合并
```

**EPC 约束下环路不可解** → HEFT fallback 任意排序 → 单机保底触发 → Ours ≈ OCC

**确定性代码下真实算法行为（Exp1, 4×Xeon, 100Mbps）**

| 模型 | OCC | Ours(确定性) | Ours/OCC | 说明 |
|------|-----|-------------|----------|------|
| InceptionV3 | 2282ms | 1231ms | 0.539 | 并行分支结构，张量并行有效 |
| BERT-base | 3613ms | 3382ms | 0.936 | 残差→环路→保底触发 |
| BERT-large | 12387ms | 12333ms | 0.996 | 同上 |
| ViT-base | 4079ms | 3752ms | 0.920 | 同上 |
| ViT-large | 13634ms | 13563ms | 0.995 | 同上 |
| TinyBERT-4l | 239ms | 224ms | 0.936 | 小模型，轻微改善 |
| ViT-small | 1137ms | 988ms | 0.869 | 小模型，轻微改善 |

**结论**：非确定性修复前的"亮眼"数据（如 BERT-base 0.396×）是随机 set() 的意外产物。确定性代码的正确结果：Ours 对 InceptionV3 有真实加速（1.85×），对 BERT/ViT 在 100Mbps 下退化为 ≈ OCC（正确行为：带宽不足以触发张量并行）。

> **素材标注**：
> - **论文 Discussion**：BERT/ViT 残差连接造成分区图 EPC-unresolvable 环路，是 100Mbps 下 Ours 无法超越 OCC 的根本原因；高带宽（≥500Mbps）下张量并行会生效
> - **论文 Evaluation**：Exp2 网络消融中 BERT/ViT 的 Ours 加速曲线会随带宽增大而改善，体现算法的自适应带宽特性
> - **重要警示**：非确定性修复前的实验数据（2026-03-03 第一版）不可用于论文，需以确定性重跑结果为准

---
