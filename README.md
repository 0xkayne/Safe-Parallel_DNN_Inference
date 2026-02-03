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

### 4. OCC (Occlumency)

**核心思想**：基于 Intel SGX 的安全飞地，使用 SGX enclave 在云端执行深度学习推理，确保用户数据在整个推理过程中的机密性与完整性。该方法在单服务器上执行，严格遵守 EPC（约 93 MB）容量限制，采用 **严格分区**（每个分区的内存不超过 EPC）并采用 **串行调度**，不涉及跨服务器通信。

* **分区策略**：在模型 DAG 上进行拓扑排序后，将每个层依次放入当前分区，若加入后内存超出 EPC，则开启新分区。每个分区在同一服务器上顺序执行。
* **调度策略**：所有分区在单服务器上按顺序执行，避免网络通信开销，唯一的开销来自 SGX 换页（Demand Paging）和上下文切换。
* **与其他算法的区别**：相较于 DINA、MEDIA、Ours，OCC 不利用多服务器并行，也不进行跨服务器通信，仅在单服务器环境下提供安全推理基准。

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

## 项目结构

*   `datasets/`: 存放模型数据的 CSV 文件。
*   `loader.py`: **数据加载器**。解析 CSV 文件，构建 NetworkX DAG 图，自动处理不同列名格式。
*   `common.py`: **公共类定义**。包含 `DNNLayer`, `Partition`, `Server` 等基础数据结构及全局常量（如 EPC 大小）。
*   `alg_dina.py`: DINA 算法实现。
*   `alg_media.py`: MEDIA 算法实现。
*   `alg_ours.py`: Ours 算法实现。
*   `experiment_runner.py`: **实验主程序**。自动遍历所有模型、服务器配置和带宽配置，运行三种算法并输出对比结果。
*   `model_struct_visualization/`: **模型结构可视化模块**。包含代码及生成的结果。
    *   `visualize_model.py`: 核心可视化脚本，支持单文件处理。
    *   `batch_visualize.py`: 批量可视化处理脚本，自动遍历数据集。
    *   `outputs/`: 生成的可视化文件存放目录（按模型分子文件夹）。
*   `results_comparison.csv`: 实验结果汇总。

## 模型可视化工具

`model_struct_visualization/visualize_model.py` 可将 DNN 模型的层级依赖关系可视化为交互式 HTML 图形。

### 功能特性

- **交互式图形**：支持缩放、平移、拖拽节点
- **动态着色**：可根据任意列（如 `group`、`type` 或未来的 `partition_id`）对节点着色
- **悬停信息**：鼠标悬停显示层的详细性能指标
- **层级布局**：清晰展示数据流方向

### 使用方法

```bash
# 安装依赖（如尚未安装）
pip install pyvis pandas networkx

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
