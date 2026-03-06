# Phase 1：系统搭建与 Baseline 验证

---

### [2026-03-03] 仿真系统架构概述

**类型**：`算法决策`

**设计决策**：采用纯 Python 仿真（无真实 SGX 硬件），所有延迟从层级 profiling 数据分析计算得出。

**理由**：

- 真实 SGX 环境下运行 12 个模型 × 3 组实验的完整对比，硬件配置和重复测量代价极高
- 仿真允许精确控制变量（带宽、服务器配置、EPC 大小），真实环境难以做到
- 数据集来自真实 SGX 硬件 profiling（`enclave_time_mean` 是真实测量值），保证了计算成本的真实性

**核心 cost model 参数**（均来自文献实测）：


| 参数            | 数值                         | 来源                 |
| ------------- | -------------------------- | ------------------ |
| EPC 有效大小      | 93 MB（128-35）              | SGX 实测（OS 保留 35MB） |
| Paging 带宽     | 1000 MB/s（EPC↔DRAM，AES 加密） | ICDCS'22           |
| Page fault 开销 | 0.03 ms/page（30 µs）        | 实测                 |
| 首次超 EPC 惩罚    | 4.5×（执行时间乘数）               | MobiCom'19         |
| RTT（边缘网络）     | 5 ms（园区 LAN）               | 实测                 |


**数据集**：`datasets_260120/`，12 个模型，每层包含：

- `enclave_time_mean`：SGX enclave 内执行时间（ms）
- `weight_bytes`/`bias_bytes`/`activation_bytes`：内存分量（Bytes）
- `output_bytes`：层输出大小（用于通信量计算）
- `dependencies`：层间依赖（用于构建 DAG）

**DAG edge weight 约定**（关键！）：

- `loader.py` 将 `output_bytes / (1024^2)` 存储为 edge weight，**单位 MB**
- 所有下游算法代码必须直接使用此值，**不得再次换算**

> **素材标注**：可作为 paper Implementation 章节中仿真方法可信度的论证，特别是"数据来自真实 SGX 硬件 profiling"这一点

---

### [2026-03-03] 模型结构关键数据

**类型**：`实验结果`

通过 `loader.py` + `alg_occ.py` 分析各模型拓扑：


| 模型          | 总层数  | 有参数层 | OCC 分区数 | 单分区最大内存 |
| ----------- | ---- | ---- | ------- | ------- |
| BERT-base   | 543  | 99   | 4       | 91.5 MB |
| BERT-large  | 1371 | 195  | 14      | 93.0 MB |
| ViT-large   | 1371 | 195  | 15      | 92.4 MB |
| InceptionV3 | 181  | 77   | 2       | 90.0 MB |
| TinyBERT-4l | ~100 | ~30  | 1       | <93 MB  |
| ViT-small   | ~400 | ~60  | 1       | <93 MB  |


**BERT-base 分区内存构成**（paging 主导分析）：

- 静态（weights + bias）：87.9 MB（96%）
- 动态峰值激活：3.6 MB（4%）
- paging 开销（加载 87.9MB）：~700 ms/分区
- 执行时间：~190 ms/分区
- **paging 占比 79%**：高度 I/O 密集型

**InceptionV3 的特殊性**：

- 仅 2 个 OCC 分区（174 层 + 7 层），原因：模型总权重仅 89 MB，接近单 EPC 大小
- 有真实并行分支（4 路 inception 模块）
- 但分支内存（每路约 5-20 MB）远小于 EPC 限制，MEDIA 将其合并

> **素材标注**：
>
> - "paging 占 79%"可作为 paper Background 中 SGX EPC 约束严重性的关键数据
> - InceptionV3 作为"唯一有真实并行结构的模型"在 paper 中值得单独分析

---

