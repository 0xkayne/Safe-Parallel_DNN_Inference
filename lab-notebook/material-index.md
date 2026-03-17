# 论文 / 专利素材索引

> 写论文和专利时从这里开始检索。每条素材标注了：
> - 来源笔记（链接）
> - 可用于论文的哪个章节
> - 是否具有专利潜力

---

## 一、按论文章节分类

### Abstract（摘要关键数据）

> **v12 版本（2026-03-12，当前最终）**：MEDIA 论文忠实复现（删除 Stage 2 + Constraint 2 不等长分支缺陷修复）。MEDIA: 41 分区 → 3 server / 1403ms（v11 仅 1 server）；DINA: 2 server ✅；OCC: 1 server/35.8MB ✅；Ours: 4 server / 930ms（最优）。

| 素材 | 数据 | 来源 |
|------|------|------|
| **SGX 内存真实性增强（v10 核心）** | workspace（activation−output）+ 碎片 ×1.15 + 框架 10MB；InceptionV3: 100.0→117.7MB | [Exp1 v10 结果](phase-3-实验结果分析/notes.md) |
| **MEDIA Check() 被迫选 paging（v10 关键发现）** | BERT-large@100Mbps: 层间传输 16.5MB→1325ms >> paging 163ms → Check() 合并 → 6 分区全超 EPC → M/O=4.57× | [Exp1 v10 结果](phase-3-实验结果分析/notes.md) |
| **MEDIA 退化** | v10 大模型 3-5× 慢于 OCC（BERT-large: 4.57×, ViT-large: 5.09×）；合并 = 累积更多激活 = 更严重 paging | [Exp1 v10 结果](phase-3-实验结果分析/notes.md) |
| **Ours 主要加速（100Mbps）** | InceptionV3: **1.70×** vs OCC（891ms vs 1514ms），YOLOv5: **1.40×**（8304ms vs 11613ms），VGG-16: **2.28×**（1353ms vs 3079ms） | [Exp1 v13 结果](phase-3-实验结果分析/notes.md) |
| OCC 天然适配 | Weights outside EPC → 仅 activation+ring buffer 在 EPC → 分区小 → 无 paging | [v9 分析](phase-4-算法理论分析/notes.md) |
| Ours 低带宽鲁棒性 | 任意模型 @ 0.5Mbps: Ours ≈ OCC（单机保底触发） | [Exp2 结果](phase-3-实验结果分析/notes.md) |

---

### Introduction / Motivation

| 素材 | 要点 | 来源 |
|------|------|------|
| SGX paging 开销极高 | BERT-base: paging 占推理延迟 79%（2856ms out of 3613ms） | [Phase 1 模型数据](phase-1-系统搭建与baseline验证/notes.md) |
| **TEE 中 free() 不释放 EPC 页（v9）** | SGX enclave 的 malloc arena 碎片 + 4KB 页粒度 + LibOS 内存池 → 激活内存只增不减 → peak-liveness 模型低估 8× | [v9 内存模型分析](phase-4-算法理论分析/notes.md) |
| **MEDIA 合并策略在 TEE 下失效（v9）** | 累积激活模型下合并 = 单调增加内存 → MEDIA 2-5× 慢于 OCC | [v9 分析](phase-4-算法理论分析/notes.md) |
| 分布式不总是更好 | DINA @ 0.5Mbps 比 OCC 慢 88×，说明朴素分布式方法有严重局限 | [DINA 分析](phase-4-算法理论分析/notes.md) |
| EPC 是关键约束 | EPC 93MB 限制导致模型必须拆分为 4-15 个分区 | [Phase 1 模型数据](phase-1-系统搭建与baseline验证/notes.md) |

---

### Background / Related Work

| 素材 | 要点 | 来源 |
|------|------|------|
| SGX paging cost model | v9: 渐进模型 `penalty = 1.0 + 2.0 × (overflow/EPC)`（取代旧 4.5× 阶跃） | [v9 分析](phase-4-算法理论分析/notes.md) |
| EPC 有效大小 93MB | 128MB 物理 - 35MB OS 保留 | [Phase 1 架构概述](phase-1-系统搭建与baseline验证/notes.md) |
| DINA 强制轮换的缺陷 | 在带宽 < 50Mbps 时系统性劣于 OCC | [DINA 分析](phase-4-算法理论分析/notes.md) |
| MEDIA 的前提假设 | 需要多条"EPC 级别"的独立大型并行路径 | [MEDIA 分析](phase-4-算法理论分析/notes.md) |
| **MEDIA Constraint 2 结构缺陷（v12）** | `L(u)==L(w)-1` 级别检查仅保护等长分支；InceptionV3 不等长分支（1-3层）绕过检查 → 并行结构坍缩为链式 DAG → 仅 1 server | [v12 分析](phase-3-实验结果分析/notes.md) |

---

### Design / Methodology（Ours 方法）

| 素材 | 要点 | 来源 |
|------|------|------|
| HPA 并行度 DP 决策 | `Cost(v,k) = T_comp/k^γ + Penalty(M/k) + T_sync × P_sync` | [Ours vs MEDIA 分析](phase-4-算法理论分析/notes.md) |
| **类型感知同步原语（v13）** | Conv→AllGather `(k-1)/k × output`（滤波器并行）；FC→AllReduce `2(k-1)/k × output`（列并行），通信量差 2× | [v13 Conv/FC 分析](phase-3-实验结果分析/notes.md) |
| **TEE 累积激活内存模型（v9）** | `peak_activation = Σ output_bytes`（不释放），取代 DAG liveness 追踪。物理依据：SGX free()→arena 碎片→EPC 页不回收 | [v9 内存模型分析](phase-4-算法理论分析/notes.md) |
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

#### Per-Server Peak Memory（RQ4：baseline 论文对齐验证）

来源：[v11 对齐验证](phase-3-实验结果分析/notes.md)，`diagnostics/server_peak_memory.py`

```
建议 Figure：InceptionV3 per-server peak memory 3D 图
- 对比论文 Fig.(e)：OCC 1srv/35.8MB, DINA 2srv/38+94MB, MEDIA 3srv/29+18+12MB
- EPC threshold plane at 93MB
- 说明 MEDIA 3 vs 4 server 差异源自 M-edge 处理顺序
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
| **MEDIA Check() 双内存模型设计（v11）** | 分区决策用 Σ layer.memory（保守启发式），调度用 peak-liveness（物理准确）；论文的 Check() 刻意保守以产生更多分区 | [v11 对齐验证](phase-3-实验结果分析/notes.md) |
| **MEDIA 宽度 3 vs 4 差异（v11）** | M-edge 按通信量降序处理，第一条分支在 fork 保护触发前合并 → inception width=3（论文 4）；数据依赖的处理顺序效应 | [v11 对齐验证](phase-3-实验结果分析/notes.md) |
| **MEDIA 在 TEE 下失效（v9）** | 累积激活模型下 MEDIA 2-5× 慢于 OCC（BERT 2.14×, ViT-large 3.48×）；合并策略单调增加内存 → paging 恶化；仅在 ≥1Gbps 时趋近 OCC | [v9 分析](phase-4-算法理论分析/notes.md) |
| Ours 在低带宽的退化 | InceptionV3 @ 0.5Mbps: Ours≈OCC（单机保底触发，非 AllReduce 过贵） | [Exp2 结果](phase-3-实验结果分析/notes.md) |
| Ours 低带宽鲁棒性 vs DINA | @ 0.5Mbps：Ours≈1×OCC，DINA=**219-446×OCC**（v6）；单机保底是核心机制 | [Bug 2 修复](phase-2-Bug排查与修复/notes.md) |
| Ours 的适用边界 | 建议带宽 ≥ 10Mbps（InceptionV3），50Mbps（BERT-base），500Mbps（BERT-large/ViT-large） | [Ours vs MEDIA 分析](phase-4-算法理论分析/notes.md) |
| Ours 在 100Mbps 下的 n 扩展性 | InceptionV3: Ours/OCC 随 n 单调改善（0.667-0.909×）；BERT/ViT 线性模型：保底主导，Ours/OCC ≈ 0.877-0.991× 且不随 n 变化 | [Exp3 v6 结果](phase-3-实验结果分析/notes.md) |
| OCC 不能利用多台同类服务器 | OCC@n=4 = OCC@n=2（同类型），Ours: InceptionV3 随 n 持续改善（v6 确认）；BERT/ViT 因保底主导无法利用额外算力 | [Exp3 v6 结果](phase-3-实验结果分析/notes.md) |

---

## 二、潜在可专利点

### 专利点 P1（核心）：带宽自适应张量并行度动态选择方法

**技术方案**：在 SGX TEE 约束下，针对 DNN 推理中的计算密集型算子，通过动态规划（DP）方法，根据当前网络带宽、算子计算量、EPC 内存约束、算子类型（卷积/全连接），为每个算子选择最优张量并行度 k（k ∈ {1, 2, 4, 8}），并根据算子类型选择最优同步原语——卷积层使用 AllGather（滤波器并行），全连接层使用 AllReduce（列并行）——在 k 个 TEE 节点间同步中间结果。

**新颖性要点**：
- 在 EPC 内存约束（93MB）和网络带宽动态变化的双重约束下，联合优化并行度选择
- **类型感知同步原语**：Conv 滤波器并行产生独立输出通道，仅需 AllGather（拼接），通信量 `(k-1)/k × output`；FC 列并行产生部分和，需 AllReduce（归约），通信量 `2(k-1)/k × output`——差 2 倍
- 低带宽时自动退化为 k=1（单机），避免同步开销超过并行收益
- AllReduce 同步概率建模（`P_sync=0.5`，对应 Megatron 式两层共享一次同步）

**技术效果**（v6 最终数据，4×Xeon_IceLake，100Mbps）：
- InceptionV3 推理延迟从 1,506ms（OCC）降至 930ms（Ours），降低 38%（**1.62×**）；500Mbps 下进一步至约 556ms（**~2.7×**）
- BERT-large/ViT-large 在 100Mbps 下 Ours≈OCC（残差环，保底触发）；500Mbps 下约 **1.27-1.30×** 加速（v6）
- 低带宽（0.5Mbps）下所有模型 Ours≈OCC，而 DINA 崩溃到 **219-446×OCC**（v6，OCC 基线降低后比值更大）——自适应退化是核心优势
- ⚠️ v3 旧值（InceptionV3 OCC=2282ms，1.85× 加速）因 OCC 未修正 paging 模型而偏高，已废弃

**权利要求主题**：方法权利要求 + 系统权利要求 + 存储介质

来源：[HPA 决策机制](phase-4-算法理论分析/notes.md) + [Exp2 带宽实验](phase-3-实验结果分析/notes.md)

---

### 专利点 P2：基于 TEE 物理约束的 EPC 峰值内存估计方法

**技术方案**：针对 SGX TEE 环境下 DNN 推理分区的 EPC 内存需求估计问题，提出基于 "累积 output_bytes" 的激活内存模型。该模型基于 SGX 物理约束（free() 不释放 EPC 页、malloc arena 碎片、4KB 页粒度）建模，认为分区执行期间所有层的输出激活累积不释放，以此估算分区峰值内存。

**EPC 内存组成**：`peak_memory = Σ(weight+bias+encryption) + Σ(output_bytes)`

**新颖性要点**：
- 基于 SGX EPC 物理机制（EREMOVE 不由 free() 触发、arena 碎片、LibOS 内存池）推导累积模型
- 区分"静态内存"（权重，用于 paging cost / swap cost）和"峰值内存"（权重+累积激活，用于 EPC 约束判断和 paging penalty 计算）
- 使用 `output_bytes`（仅自身输出）而非 `activation_memory`（含前驱输出，双重计数），避免层间重复计算
- 配合渐进 penalty 函数 `1.0 + 2.0 × (overflow/EPC)` 模拟连续的 EPC 换页代价

**技术效果**（v9 验证数据）：
- InceptionV3 峰值激活：peak-liveness 10.6MB → 累积模型 83.1MB（更接近真实 ~100MB）
- 揭示 MEDIA 合并策略在 TEE 下的根本缺陷：合并 = 累积激活增加 = paging 恶化（2-5× 慢于 OCC）
- OCC "weights outside EPC" 方案与累积模型天然兼容

来源：[v9 内存模型分析](phase-4-算法理论分析/notes.md) + [Exp1 v9 结果](phase-3-实验结果分析/notes.md)

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
