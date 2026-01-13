# 项目代码结构说明

本文档描述了分布式安全推理调度模拟器项目的代码组织结构、各文件功能及其协作方式。

---

## 一、项目目录结构

```
pythonProject/
│
├── 📄 核心代码
│   ├── common.py              # 公共数据结构与惩罚函数
│   ├── loader.py              # 模型DAG加载与依赖解析
│   ├── alg_dina.py            # DINA算法实现
│   ├── alg_media.py           # MEDIA算法实现
│   ├── alg_ours.py            # Ours算法实现（HEFT调度）
│   ├── alg_occ.py             # OCC基线算法实现
│   └── experiment_runner.py   # 实验执行入口
│
├── 📁 datasets/               # 模型数据集（CSV格式）
│   ├── SafeDnnInferenceExp - ALBERT.csv
│   ├── SafeDnnInferenceExp - BERT-base.csv
│   ├── SafeDnnInferenceExp - DistillBERT.csv
│   ├── SafeDnnInferenceExp - InceptionV3.csv
│   ├── SafeDnnInferenceExp - TinyBERT-4l.csv
│   ├── SafeDnnInferenceExp - TinyBERT-6l.csv
│   └── SafeDnnInferenceExp - ViT-base.csv
│
├── 📁 results/                # 实验结果输出
│   ├── exp1_4servers_100mbps.csv
│   ├── exp2_bandwidth_experiment.csv
│   ├── exp3_server_comparison.csv
│   ├── final_results_*.csv
│   ├── new_results_*.csv
│   └── results_*.csv
│
├── 📁 docs/                   # 分析文档
│   ├── algorithm_analysis.md
│   ├── bert_dependency_issue.md
│   ├── convergence_analysis.md
│   ├── dynamic_penalty_theory.md
│   ├── ours_regression_analysis.md
│   ├── parallelism_analysis.md
│   ├── result_analysis_and_fix.md
│   └── sgx_paging_analysis.md
│
├── 📁 legacy/                 # 旧版/废弃代码
│   ├── DINA.py
│   ├── MEDIA-GPT.py
│   ├── MEDIA-GPT-copy.py
│   ├── Improved-MEDIA.py
│   ├── MIDEA.py
│   ├── qiongju_HIGH.py
│   └── setup.py
│
├── 📁 network-chart/          # 网络拓扑图表数据
├── 📁 server-chart/           # 服务器调度图表数据
│
├── README.md                  # 项目说明
├── CODE_STRUCTURE.md          # 本文件
└── requirements.txt           # Python依赖
```

---

## 二、核心代码文件功能

### 1. `common.py` — 公共数据结构

| 组件 | 功能 |
|------|------|
| `EPC_EFFECTIVE_MB` | SGX EPC有效容量常量（93 MB） |
| `calculate_penalty()` | 动态换页惩罚计算函数 |
| `DNNLayer` | DNN层的数据类（内存、计算时间、输出大小） |
| `Partition` | 分区类（包含多个层、总内存、总工作量） |
| `Server` | 服务器类（算力比、调度队列） |

---

### 2. `loader.py` — 模型加载器

| 函数 | 功能 |
|------|------|
| `ModelLoader.load_model_from_csv()` | 从CSV加载模型，构建DAG图 |

**关键逻辑**：
- 解析层信息（内存、计算时间、通信量）
- 修复Transformer Q/K/V并行依赖
- 处理ViT的虚拟QKV分离节点
- 构建`networkx.DiGraph`，边权重为通信量(MB)

---

### 3. `alg_dina.py` — DINA算法

| 方法 | 功能 |
|------|------|
| `run()` | 严格EPC约束分区（贪心装箱） |
| `schedule()` | Round-Robin调度 |

**分区策略**：每个分区内存 ≤ EPC，超出则切分

---

### 4. `alg_media.py` — MEDIA算法

| 方法 | 功能 |
|------|------|
| `run()` | 通信感知合并分区 |
| `schedule()` | Round-Robin调度 |

**核心逻辑**：若换页惩罚 < 通信开销，则允许超EPC合并

---

### 5. `alg_ours.py` — Ours算法（本文方法）

| 方法 | 功能 |
|------|------|
| `run()` | DAG感知分区（保留拓扑结构） |
| `schedule()` | HEFT调度（计算-通信全局优化） |

**特点**：基于Rank-U的全局最优调度

---

### 6. `alg_occ.py` — OCC基线

| 方法 | 功能 |
|------|------|
| `run()` | EPC约束分区 |
| `schedule()` | 单服务器串行执行 + 换页开销 |

---

### 7. `experiment_runner.py` — 实验执行器

批量运行所有模型×服务器数×带宽组合，输出CSV结果。

---

## 三、数据流与协作关系

```
datasets/*.csv
      │
      ▼
  loader.py ──────► (G: DiGraph, layers_map)
      │
      ▼
  ┌───────────────────────────────────┐
  │  alg_dina / alg_media / alg_ours  │
  │  / alg_occ                        │
  └───────────────────────────────────┘
      │
      ├── run()  ──► partitions
      │
      └── schedule() ──► latency_ms
                              │
                              ▼
                       results/*.csv
```

---

## 四、快速使用示例

```python
from loader import ModelLoader
from common import Server
from alg_ours import OursAlgorithm

# 1. 加载模型
G, layers_map = ModelLoader.load_model_from_csv('datasets/SafeDnnInferenceExp - ViT-base.csv')

# 2. 创建服务器
servers = [Server(i, 1.0) for i in range(4)]

# 3. 运行算法
ours = OursAlgorithm(G, layers_map, servers, bandwidth_mbps=100)
partitions = ours.run()
latency = ours.schedule(partitions)

print(f"Inference latency: {latency:.2f} ms")
```

---

## 五、文件依赖关系

| 文件 | 依赖 | 被依赖 |
|------|------|--------|
| `common.py` | networkx | loader, 所有alg_* |
| `loader.py` | common | experiment_runner |
| `alg_*.py` | common, networkx | experiment_runner |
| `experiment_runner.py` | 所有上述模块, pandas | — |

---

## 六、扩展指南

### 添加新算法
1. 创建 `alg_xxx.py`
2. 实现 `XXXAlgorithm` 类，包含 `run()` 和 `schedule()` 方法
3. 在 `experiment_runner.py` 中导入并添加到实验循环

### 添加新模型
将模型CSV放入 `datasets/` 目录，格式需包含：
- `name`, `enclave_time_mean`, `tee_total_memory_bytes`, `output_bytes`, `dependencies`

### 修改惩罚模型
编辑 `common.py` 中的 `calculate_penalty()` 函数
