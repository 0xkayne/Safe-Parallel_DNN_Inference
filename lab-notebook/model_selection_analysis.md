# DNN 模型选型分析：补充实验架构覆盖度

> 日期：2026-03-12
> 目的：分析当前实验模型集的架构覆盖缺口，推荐补充模型以有效验证分支结构并行和张量并行效果

---

## 当前覆盖分析

当前 12 个模型从 DAG 拓扑角度看只有两种结构：

| 拓扑类型 | 模型 | DAG 特征 | 分支并行 | 张量并行 |
|----------|------|---------|---------|---------|
| **纯链式** | BERT/ALBERT/DistilBERT/TinyBERT/ViT (10个) | width=1，无分支 | 无 | 有（大 attention/FFN 算子）|
| **局部多路 fork-join** | InceptionV3 (1个) | inception module 内 4 路并行，module 间串行 | 有（4-way） | 有 |

缺失的架构类型至少有 4 种，每种对算法的考验不同。

---

## 推荐补充的架构类型与模型

### 1. 残差跳跃连接（2-way fork-join）— ResNet

**代表模型**：ResNet-50、ResNet-152

**DAG 特征**：每个 residual block 是一个 2-way fork-join：主路径（2-3 层 conv）与 skip connection 并行，在 element-wise add 处汇合。整个网络由 ~16-50 个这样的 block 串联。

**为什么需要**：
- 残差连接是当前最普遍的 CNN 设计范式，缺少 ResNet 会被审稿人质疑实验覆盖度
- 2-way 分支比 Inception 的 4-way 简单得多，可以测试 **HPA 在浅分支并行下是否仍有收益**
- ResNet 的 skip connection 传输量极小（只传 feature map 本身），与 Inception 的大激活传输形成对比 → 测试通信开销敏感度
- 预期结果：Ours 在 ResNet 上的加速比应介于 Transformer（几乎无加速）和 InceptionV3（显著加速）之间，形成**连续梯度**，比二元对比更有说服力

**建议规模**：ResNet-50（~25M params, 50 层）和 ResNet-152（~60M params, 152 层）分别代表中等和大规模

---

### 2. 密集连接（dense fan-in）— DenseNet

**代表模型**：DenseNet-121

**DAG 特征**：Dense block 内每层接收所有前序层的输出作为输入。一个 6 层的 dense block，第 6 层有 5 个输入边。整体 DAG 的 fan-in 远大于其他架构。

**为什么需要**：
- 高连接度 DAG 对分区算法是严峻考验：边多 → 跨分区通信多 → MEDIA 的 edge selection 和 Ours 的 HEFT 面临完全不同的 tradeoff
- DenseNet 的 concat 操作（不是 add）会累积通道数，使得**中间激活内存随深度快速增长** → 可能触发 EPC paging，测试各算法的内存管理能力
- 密集连接限制了分区自由度（切任何边都切断大量信息流），预期各算法差距缩小 → 验证 Ours 在**不利拓扑下不退化**

---

### 3. 多尺度特征金字塔（multi-scale lateral connections）— YOLOv3 / FPN

**代表模型**：YOLOv3（或 RetinaNet with FPN backbone）

**DAG 特征**：backbone 提取多尺度特征 → top-down pathway 逐级上采样 + lateral connection 融合 → 多个检测头并行输出。DAG 形状是"倒三角 + 横向连接 + 多输出"，与前面所有架构都不同。

**为什么需要**：
- **多输出头**是全新的 DAG 模式：3 个检测头可以真正并行在不同服务器上执行
- FPN 的 lateral connection 创造了**跨层级的长距离依赖**，不同于 Inception 的局部 fork-join
- 目标检测是边缘推理的核心场景（自动驾驶、监控），增加应用说服力
- 预期结果：Ours 可以将 3 个独立输出头分配到不同服务器 → 接近 InceptionV3 级别的加速

---

### 4. 持续多分辨率并行（sustained parallelism）— HRNet

**代表模型**：HRNet-W32

**DAG 特征**：从第一个 stage 开始就维持 2-4 条不同分辨率的并行流，stage 之间有跨分辨率的信息交换（多对多连接）。与 Inception 的"fork → 几层 → join → fork"不同，HRNet 的并行是**全程持续**的。

**为什么需要**：
- **持续并行**是 Ours(HPA) 最理想的场景：多条并行流可以常驻不同服务器，减少通信次数
- 跨分辨率交换产生的通信模式（高分辨率→低分辨率：下采样；低→高：上采样）与 Inception 的 concat 不同
- 预期效果：HRNet 上 Ours 的加速比可能**超过 InceptionV3**，因为并行度更持久
- 这个结果可以作为论文的亮点论据

---

### 5. 混合 CNN-Transformer — EfficientNet 或 Swin Transformer

**代表模型**：EfficientNet-B3 或 Swin-T

**为什么需要**：
- EfficientNet 的 MBConv + SE block 形成微小的 fork-join（SE 分支是 squeeze→excite→scale），加上 skip connection → **双层嵌套分支**
- Swin Transformer 使用 window attention + shifted window，与标准 ViT 的全局 attention 不同，且有分层结构（patch merging） → 测试非标准 Transformer
- 代表"2020 年后的现代架构"，增加时效性

---

## 优先级排序

| 优先级 | 模型 | 核心理由 |
|--------|------|---------|
| **P0（必须加）** | **ResNet-50** | 最经典 CNN，缺它实验不完整；2-way branch 填补 Inception(4-way) 和 Transformer(0-way) 之间的空白 |
| **P0（必须加）** | **YOLOv3 或 FPN** | 唯一的多输出头 + 多尺度架构；边缘推理核心场景 |
| **P1（强烈建议）** | **DenseNet-121** | 高连接度 DAG，对分区算法的压力测试；验证 Ours 在不利拓扑下的鲁棒性 |
| **P1（强烈建议）** | **HRNet-W32** | 持续多流并行，预期是 Ours 的最佳展示场景；可作论文亮点 |
| **P2（锦上添花）** | **ResNet-152** | 与 ResNet-50 形成规模对比（50 vs 152 层） |
| **P2（锦上添花）** | **EfficientNet-B3** | 现代高效架构代表；嵌套分支结构 |

---

## 补充后的架构覆盖谱

加上 P0+P1 四个模型后，架构覆盖变为：

```
纯链式(10) → 2-way残差(1-2) → 密集连接(1) → 局部多路fork-join(1) → 持续多流并行(1) → 多尺度多输出(1)
```

从"无分支"到"最大分支"形成完整的拓扑复杂度谱，实验论证会扎实很多。
