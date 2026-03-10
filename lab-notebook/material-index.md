# 论文 / 专利素材索引

> 写论文和专利时从这里开始检索。每条素材标注了：
> - 来源笔记（链接）
> - 可用于论文的哪个章节
> - 是否具有专利潜力

---

## 一、按论文章节分类

### Abstract（摘要关键数据）

> **v7 版本（2026-03-09，当前最终）**：Baseline 正确性审计完成（OCC/DINA/MEDIA 均按原论文实现）。关键变化：OCC=DINA=MEDIA（同构服务器），只有 Ours 能打破单服务器瓶颈。所有 Exp1/2/3 数据完整更新并交叉验证通过。

| 素材 | 数据 | 来源 |
|------|------|------|
| **OCC=DINA=MEDIA（核心论点）** | 4×Xeon_IceLake 100Mbps：所有12模型 OCC≡DINA≡MEDIA。串行DAG无并行收益，只有Ours（层内TP）能加速 | [Exp1 v7 结果](phase-3-实验结果分析/notes.md) |
| **Ours 主要加速（100Mbps）** | InceptionV3: **1.62×** vs baseline（930ms vs 1506ms）| [Exp1 v7 结果](phase-3-实验结果分析/notes.md) |
| Ours 高带宽加速（InceptionV3） | InceptionV3 @ 500Mbps: **2.71×**（557ms vs 1506ms）| [Exp2 v7 结果](phase-3-实验结果分析/notes.md) |
| Ours 大型模型高带宽加速 | BERT-large @ 500Mbps: **1.27×**（1817ms vs 2307ms）；ViT-large: **1.30×**（2748ms vs 3563ms）| [Exp2 v7 结果](phase-3-实验结果分析/notes.md) |
| Ours 低带宽鲁棒性 | 任意模型 @ 0.5Mbps: Ours ≈ OCC（单机保底触发）；DINA/MEDIA 也 ≈ OCC（v7正确实现后不再强制分发）| [Exp2 v7 结果](phase-3-实验结果分析/notes.md) |
| 线性模型 baseline 等价（确认） | BERT/ViT 所有模型、所有带宽：OCC ≡ DINA ≡ MEDIA（无并行分支可利用）| [Exp1 v7 结果](phase-3-实验结果分析/notes.md) |

---

### Introduction / Motivation

| 素材 | 要点 | 来源 |
|------|------|------|
| SGX paging 开销极高 | BERT-base: paging 占推理延迟 79%（2856ms out of 3613ms） | [Phase 1 模型数据](phase-1-系统搭建与baseline验证/notes.md) |
| 分布式不总是更好 | DINA @ 0.5Mbps 比 OCC 慢 88×，说明朴素分布式方法有严重局限 | [DINA 分析](phase-4-算法理论分析/notes.md) |
| 现有方法 MEDIA 受限于并行结构 | MEDIA 仅对有并行分支的模型有效（InceptionV3: 5.3%加速）；线性 Transformer 等价 OCC | [MEDIA 分析](phase-4-算法理论分析/notes.md) |
| EPC 是关键约束 | EPC 93MB 限制导致模型必须拆分为 4-15 个分区 | [Phase 1 模型数据](phase-1-系统搭建与baseline验证/notes.md) |

---

### Background / Related Work

| 素材 | 要点 | 来源 |
|------|------|------|
| SGX paging cost model | `penalty = 4.5×` for first EPC overflow，线性增长之后 | [Phase 1 架构概述](phase-1-系统搭建与baseline验证/notes.md) |
| EPC 有效大小 93MB | 128MB 物理 - 35MB OS 保留 | [Phase 1 架构概述](phase-1-系统搭建与baseline验证/notes.md) |
| DINA 强制轮换的缺陷 | 在带宽 < 50Mbps 时系统性劣于 OCC | [DINA 分析](phase-4-算法理论分析/notes.md) |
| MEDIA 的前提假设 | 需要多条"EPC 级别"的独立大型并行路径 | [MEDIA 分析](phase-4-算法理论分析/notes.md) |

---

### Design / Methodology（Ours 方法）

| 素材 | 要点 | 来源 |
|------|------|------|
| HPA 并行度 DP 决策 | `Cost(v,k) = T_comp/k^γ + Penalty(M/k) + T_AllReduce × P_sync` | [Ours vs MEDIA 分析](phase-4-算法理论分析/notes.md) |
| AllReduce 通信模型 | Ring AllReduce: `2(k-1)/k × output_bytes`，同步概率 0.5 | [Ours vs MEDIA 分析](phase-4-算法理论分析/notes.md) |
| DAG liveness 峰值内存 | `total_mem = persistent + peak_activation`，DAG 遍历追踪激活存活窗口 | [内存模型验证](phase-3-实验结果分析/notes.md) |
| 算子级 vs 分区级并行 | Ours=张量并行（层内），MEDIA=流水并行（分区间），根本差异 | [Ours vs MEDIA 分析](phase-4-算法理论分析/notes.md) |
| 带宽自适应的意义 | 低带宽禁用 HPA（k=1），高带宽激活，平滑退化 | [Ours vs MEDIA 分析](phase-4-算法理论分析/notes.md) |

---

### Evaluation

#### 主对比表（RQ1：固定配置下 4 方法对比）

来源：[Exp1 结果](phase-3-实验结果分析/notes.md)，配置：4×Xeon_IceLake，100Mbps

```
建议 Table 1：12 模型 × 4 方法的延迟对比（单位 ms）
加粗每行最优值（Ours）
底部行：平均加速比
```

#### 网络带宽消融（RQ2：带宽敏感性）

来源：[Exp2 结果](phase-3-实验结果分析/notes.md)

```
建议 Figure：
- X 轴：带宽（对数坐标，0.5-500 Mbps）
- Y 轴：推理延迟（ms，对数）
- 4 条线：OCC（水平基准线）/ DINA（高带宽趋近 OCC）/ MEDIA（与 OCC 重合）/ Ours（高带宽大幅低于 OCC）
- 重点标注 InceptionV3 @ 0.5Mbps 的 Ours vs OCC 交叉点
```

#### 服务器数量消融（RQ3：可扩展性）

来源：[Exp3 结果](phase-3-实验结果分析/notes.md)

```
建议 Figure：
- X 轴：服务器数（1-8，标注服务器类型变化点）
- Y 轴：推理延迟（ms）
- 重点：Ours 随 i5-6500 增加呈近线性下降；OCC 在同类服务器增加时不改善
```

---

### Discussion

| 素材 | 要点 | 来源 |
|------|------|------|
| MEDIA 的条件有效性（v6 精确） | 有并行分支（InceptionV3）时 MEDIA < OCC（5-18% at 100Mbps Exp3）；线性模型 RTT 粘性导致 MEDIA = OCC | [MEDIA 根因分析](phase-4-算法理论分析/notes.md) |
| Ours 在低带宽的退化 | InceptionV3 @ 0.5Mbps: Ours≈OCC（单机保底触发，非 AllReduce 过贵） | [Exp2 结果](phase-3-实验结果分析/notes.md) |
| Ours 低带宽鲁棒性 vs DINA | @ 0.5Mbps：Ours≈1×OCC，DINA=**219-446×OCC**（v6）；单机保底是核心机制 | [Bug 2 修复](phase-2-Bug排查与修复/notes.md) |
| Ours 的适用边界 | 建议带宽 ≥ 10Mbps（InceptionV3），50Mbps（BERT-base），500Mbps（BERT-large/ViT-large） | [Ours vs MEDIA 分析](phase-4-算法理论分析/notes.md) |
| Ours 在 100Mbps 下的 n 扩展性 | InceptionV3: Ours/OCC 随 n 单调改善（0.667-0.909×）；BERT/ViT 线性模型：保底主导，Ours/OCC ≈ 0.877-0.991× 且不随 n 变化 | [Exp3 v6 结果](phase-3-实验结果分析/notes.md) |
| OCC 不能利用多台同类服务器 | OCC@n=4 = OCC@n=2（同类型），Ours: InceptionV3 随 n 持续改善（v6 确认）；BERT/ViT 因保底主导无法利用额外算力 | [Exp3 v6 结果](phase-3-实验结果分析/notes.md) |

---

## 二、潜在可专利点

### 专利点 P1（核心）：带宽自适应张量并行度动态选择方法

**技术方案**：在 SGX TEE 约束下，针对 DNN 推理中的计算密集型算子，通过动态规划（DP）方法，根据当前网络带宽、算子计算量、EPC 内存约束，为每个算子选择最优张量并行度 k（k ∈ {1, 2, 4, 8}），并通过 AllReduce 协议在 k 个 TEE 节点间同步中间结果。

**新颖性要点**：
- 在 EPC 内存约束（93MB）和网络带宽动态变化的双重约束下，联合优化并行度选择
- 低带宽时自动退化为 k=1（单机），避免 AllReduce 开销超过并行收益
- AllReduce 同步概率建模（`P_sync=0.5`，对应 Megatron 式两层共享一次同步）

**技术效果**（v6 最终数据，4×Xeon_IceLake，100Mbps）：
- InceptionV3 推理延迟从 1,506ms（OCC）降至 930ms（Ours），降低 38%（**1.62×**）；500Mbps 下进一步至约 556ms（**~2.7×**）
- BERT-large/ViT-large 在 100Mbps 下 Ours≈OCC（残差环，保底触发）；500Mbps 下约 **1.27-1.30×** 加速（v6）
- 低带宽（0.5Mbps）下所有模型 Ours≈OCC，而 DINA 崩溃到 **219-446×OCC**（v6，OCC 基线降低后比值更大）——自适应退化是核心优势
- ⚠️ v3 旧值（InceptionV3 OCC=2282ms，1.85× 加速）因 OCC 未修正 paging 模型而偏高，已废弃

**权利要求主题**：方法权利要求 + 系统权利要求 + 存储介质

来源：[HPA 决策机制](phase-4-算法理论分析/notes.md) + [Exp2 带宽实验](phase-3-实验结果分析/notes.md)

---

### 专利点 P2：基于 DAG 活跃性分析的 EPC 峰值内存估计方法

**技术方案**：针对 DNN 模型分区的 EPC 内存需求估计问题，提出基于 DAG 拓扑排序的激活内存活跃性（liveness）分析方法。对于分区内的每一步执行，追踪哪些前序层的输出张量仍被后续层依赖（即"仍存活"），计算各时刻的激活内存累计量，取峰值作为分区的动态内存需求。

**EPC 内存组成**：`peak_memory = Σ(weight+bias+encryption) + max_t(Σ_live activation_t)`

**新颖性要点**：
- 区分"静态内存"（权重，用于 paging cost）和"峰值内存"（权重+激活，用于 EPC 约束判断）
- 在 DAG 子图上做活跃性分析，正确处理并行分支（如 inception 模块多路并行激活同时存活的情况）
- 与分区策略紧密结合：分区切割决策基于峰值内存而非参数量

**技术效果**：避免因忽略激活内存导致的分区过小（InceptionV3 P0 激活峰值 10.6MB，占总内存 11.8%）。

来源：[内存模型验证](phase-3-实验结果分析/notes.md)

---

### 专利点 P3：面向异构 TEE 边缘集群的 HEFT 感知推理调度方法

**技术方案**：将 HPA 张量并行扩展后的 DNN 算子图（包含虚拟并行 shard 节点和 AllReduce 同步节点）映射到异构 TEE 边缘服务器集群（含不同计算能力的 SGX 节点），采用改进的 HEFT（Heterogeneous Earliest Finish Time）调度算法，在考虑 SGX paging 开销、网络通信延迟、首跳 attestation 开销的前提下，最小化端到端推理延迟。

**新颖性要点**：
- SGX-aware HEFT：在传统 HEFT 的任务权重中引入 paging overhead（依赖静态内存量 × paging带宽），而非仅考虑计算时间
- 首跳 attestation（Remote Attestation，~100ms）与后续跳的区分处理
- 异构节点的算力归一化（power_ratio：Celeron 0.11×，i5-6500 0.93×，i5-11600 1.97×）

**技术效果**（v6 最终数据，Exp3 异构服务器，100Mbps）：
- **InceptionV3**：Ours < MEDIA < OCC 于所有 n≥2（Ours 0.60-0.91× OCC；MEDIA 0.83-0.99× OCC）
- **线性模型**（BERT/ViT）：HEFT 退化，单机保底触发 → Ours ≈ 0.877-0.991× OCC（恒定，不随 n 变化）
- n=3（首次出现 i5-6500）时 OCC 大幅跳变（各模型下降 80%+）；Ours 跟随跳变，相对比值稳定
- 500Mbps 下 BERT-large/ViT-large：~**1.27-1.30×** 加速（v6：BERT-large=1817ms/2307ms=0.787×；ViT-large=2748ms/3563ms=0.771×）

来源：[Exp3 结果](phase-3-实验结果分析/notes.md) + [Ours vs MEDIA 分析](phase-4-算法理论分析/notes.md)

---

## 三、论文写作 Gap 分析

当前素材覆盖情况：

| 章节 | 素材充分度 | 缺少的内容 |
|------|----------|----------|
| Abstract | ✅ 充分 | - |
| Introduction | ✅ 充分 | 需要更多 motivation 的实际场景描述 |
| Background | ✅ 充分 | 需要引用原始文献（ICDCS'22, MobiCom'19） |
| Related Work | ⚠️ 部分 | 需要梳理 DINA/MEDIA/OCC 原论文，并给出正式引用 |
| Design (Ours) | ✅ 充分 | 需要补充 HPA DP 算法的伪代码 |
| Evaluation | ✅ 充分 | 图表需要美化（当前在 figures/ 目录下有原始图） |
| Discussion | ✅ 充分 | MEDIA 有效性条件（并行分支 vs 线性）已有精确数据；需引用模型架构文献 |
| Conclusion | ⚠️ 待写 | 基于 Evaluation 结论汇总 |

---

## 四、待补充实验

- [x] **对照实验**：InceptionV3 k 分布（2026-03-04 完成）→ [实验结果](phase-3-实验结果分析/notes.md)
  - 0.5-2Mbps: k=0，保底触发；5Mbps: k=0 但 HEFT 并行分支生效（0.883×）；10Mbps: 1 op k=8；100Mbps: 17 ops k=8；500Mbps: 58 ops k=8
  - BERT-large@500Mbps: 24 ops k=8（40→388 分区），HEFT 突破 ss_time → 0.729× OCC
  - BERT-base 永不获得 k>1（层太小），所有带宽保底触发
- [x] **消融实验**：`P_sync=0.5` vs `P_sync=1.0`（2026-03-04 完成）→ [完整数据见 phase-3 笔记]
  - 100Mbps 主实验点：P_sync 影响温和（InceptionV3 仅 +3%），P_sync=0.5 选择合理
  - 500Mbps 极限：P_sync=1.0 让 BERT-large 完全失去加速（9025→12333ms，+37%）；InceptionV3 +39%
  - BERT-base：k=0 始终，P_sync 无影响
- [ ] **模型规模扩展**：更大模型（如 BERT-xxlarge）是否会有更显著的 Ours 优势？
- [ ] **安全性分析**：不同分区切割策略下的 attack surface 分析（专利和论文的安全性声明需要支撑）
