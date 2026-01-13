# 分布式 DNN 推理仿真系统

本项目模拟了在资源受限（特别是 SGX EPC 内存限制）的边缘设备集群上进行分布式 DNN 推理的过程。系统对比了三种不同的任务分割与调度算法：DINA、MEDIA 以及我们提出的优化算法（Ours）。

## 算法实现详解

### 1. DINA (Strict Partitioning)
**核心思想**：
DINA 算法严格遵守 SGX 的 EPC（Enclave Page Cache）内存限制，旨在避免昂贵的 EPC 换页开销。

*   **分区策略 (Partitioning)**：
    *   采用贪心策略遍历图的拓扑序列。
    *   **严格约束**：在构建分区时，如果加入当前层会导致分区总内存超过 EPC 有效上限（`EPC_EFFECTIVE_MB`），则强制截断，开启一个新的分区。
    *   结果是产生大量较小的分区，确保大多数计算都在 EPC 内高速运行。
*   **调度策略 (Scheduling)**：
    *   使用标准的列表调度（List Scheduling）。
    *   虽然分区大多小于 EPC，但极少数单层内存即超过 EPC 的情况仍会发生，此时会计算性能惩罚。

### 2. MEDIA (Serial / Linearized)
**核心思想**：
MEDIA 算法允许分区大小超过 EPC，通过权衡“跨服务器通信开销”与“本地 EPC 换页开销”来决定是否合并。为简化调度，该实现假设执行流是串行化的。

*   **分区策略**：
    *   首先将 DNN 的 DAG 图线性化（Topological Sort）。
    *   **代价模型权衡**：遍历线性序列，尝试合并相邻层。
        *   如果合并后内存 <= EPC：总是合并（减少通信）。
        *   如果合并后内存 > EPC：计算 **换页惩罚时间** vs **节省的通信时间**。如果换页代价小于传输代价，则仍然合并。
*   **调度策略**：
    *   **串行调度**：由于分区基于线性序列构建，调度也被简化为全局串行执行。前一个分区执行完毕后，下一个分区才开始调度，无法利用多服务器的并行计算能力。

### 3. Ours (Topology-Aware Parallel)
**核心思想**：
结合了 MEDIA 的灵活内存管理与 DINA 的 DAG 调度能力，并引入了对模型并行性的显式支持。

*   **分区策略**：
    *   **拓扑感知**：在选择合并边时，仅合并“线性”连接（出度为1且入度为1的边），**严格保留图中的分叉（Fork）和汇聚（Join）结构**。这确保了像 Inception 模块这样的并行分支不会被强制串行化合并。
    *   **灵活合并**：与 MEDIA 类似，允许分区超 EPC。在判断线性边合并时，同样比较 $T_{merged}$（含惩罚）与 $T_{separate}$（含通信），择优合并。
*   **调度策略**：
    *   **优先级列表调度**：计算分区的 Rank（基于计算量和关键路径长度）。
    *   **并行执行**：调度器能够识别互不依赖的分区（例如 Inception 模块中的不同分支），并将它们调度到不同的服务器上同时运行。
    *   **优势**：在 InceptionV3 等具有复杂分支结构的模型上，能够显著降低端到端延迟。

## 项目结构

*   `datasets/`: 存放模型数据的 CSV 文件。
*   `loader.py`: **数据加载器**。解析 CSV 文件，构建 NetworkX DAG 图，自动处理不同列名格式。
*   `common.py`: **公共类定义**。包含 `DNNLayer`, `Partition`, `Server` 等基础数据结构及全局常量（如 EPC 大小）。
*   `alg_dina.py`: DINA 算法实现。
*   `alg_media.py`: MEDIA 算法实现。
*   `alg_ours.py`: Ours 算法实现。
*   `experiment_runner.py`: **实验主程序**。自动遍历所有模型、服务器配置和带宽配置，运行三种算法并输出对比结果。
*   `results_comparison.csv`: 实验结果汇总。

## 性能分析与结果解读（基于动态惩罚模型）

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

#### 创建虚拟环境

**Windows (PowerShell)**
```powershell
# 创建虚拟环境
python -m venv .venv

# 激活虚拟环境
.\.venv\Scripts\Activate.ps1

# 如果遇到执行策略错误，先运行：
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Windows (CMD)**
```cmd
python -m venv .venv
.\.venv\Scripts\activate.bat
```

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### 安装依赖

```bash
pip install -r requirements.txt
```

> **提示**：激活虚拟环境后，命令行前面会显示 `(.venv)` 前缀。退出虚拟环境使用 `deactivate` 命令。

### 2. 运行单次实验

```bash
python experiment_runner.py
```

### 3. 运行完整实验流水线（推荐）

使用统一脚本一次性完成所有实验和图表生成：

```bash
python run_all_experiments.py
```

该脚本将自动执行以下任务：

| 步骤 | 说明 | 输出位置 |
|------|------|----------|
| 1 | 服务器数量消融实验 | `server-chart/server_*.csv` |
| 2 | 网络带宽消融实验 | `network-chart/network_*.csv` |
| 3 | 生成服务器消融图表 | `server-chart/*.png, *.pdf` |
| 4 | 生成网络带宽图表 | `network-chart/*.png, *.pdf` |

**实验配置**（可在脚本中修改）：
- 服务器数量：1, 2, 4, 8, 12, 16
- 网络带宽：1, 10, 50, 100, 500, 1000 Mbps
- 默认带宽（服务器实验）：100 Mbps
- 默认服务器数（网络实验）：4

### 4. 辅助工具：图表合并

为了方便直接对比查看所有模型的实验结果，可以使用合并脚本将生成的 PNG 图表拼接为一张大图：

```bash
python combine_charts.py
```

输出文件：
- `combined_server_charts_grid.png`: 服务器消融实验对比图（网格排列）
- `combined_network_charts_grid.png`: 网络带宽消融实验对比图（网格排列）

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
| **Context Switch** (上下文切换) | 分区之间切换 | `swap_time = (mem1+mem2) / 2.0 MB/ms` | 一次性完整的 swap out + swap in |

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
PAGING_BANDWIDTH ≈ 2 GB/s = 2.0 MB/ms（受 AES 加密引擎限制）
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
