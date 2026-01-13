# 基线实验设计文档

## 实验目的

在固定的网络和服务器配置下，系统性地评估四种算法（DINA、MEDIA、Ours、OCC）在不同DNN模型上的性能表现，为后续的消融实验提供基线数据。

## 实验背景

本实验是分布式DNN推理系统性能评估的第一步。通过固定配置参数，我们可以：

1. **建立性能基线**：为每个模型获得标准化的性能数据
2. **对比算法优劣**：在相同条件下比较四种算法的表现
3. **验证设计假设**：验证Ours算法在并行模型上的优势
4. **准备消融实验**：为后续的服务器数量和带宽消融实验提供参考

## 实验配置

### 固定参数

| 参数 | 值 | 说明 |
|------|-----|------|
| **服务器数量** | 4台 | 中等规模的边缘集群 |
| **网络带宽** | 100 Mbps | 典型的边缘网络带宽 |
| **服务器类型** | 同构 | 所有服务器算力比例均为1.0 |
| **EPC内存** | 93 MB | Intel SGX可用内存 |
| **换页带宽** | 2 GB/s | SGX加密带宽限制 |

### 测试模型

基于 `datasets/` 目录中的CSV文件，共7个模型：

| 模型 | 类型 | 特点 |
|------|------|------|
| **ALBERT** | Transformer | 轻量级BERT变体 |
| **BERT-base** | Transformer | 标准BERT模型 |
| **DistillBERT** | Transformer | 知识蒸馏BERT |
| **InceptionV3** | CNN | 多分支并行结构 |
| **TinyBERT-4l** | Transformer | 4层超轻量BERT |
| **TinyBERT-6l** | Transformer | 6层超轻量BERT |
| **ViT-base** | Vision Transformer | 视觉Transformer |

### 测试算法

| 算法 | 核心策略 | 预期优势 |
|------|----------|----------|
| **DINA** | 严格EPC限制，细粒度分区 | 避免换页开销 |
| **MEDIA** | 允许超限，权衡换页与通信 | 减少通信次数 |
| **Ours** | 拓扑感知，保留并行结构 | 利用模型并行性 |
| **OCC** | 单服务器串行执行 | 性能下界基线 |

## 实验流程

```
┌─────────────────────────────────────────────────────────────┐
│                     基线实验流程                             │
└─────────────────────────────────────────────────────────────┘

1. 初始化
   ├── 加载所有模型数据集 (7个CSV文件)
   └── 创建4台同构服务器实例

2. 对每个模型执行
   ├── 加载模型DAG图和层信息
   │
   ├── 运行 DINA 算法
   │   ├── 分区 (严格EPC限制)
   │   ├── 调度 (列表调度)
   │   └── 记录: 时延 + 分区数
   │
   ├── 运行 MEDIA 算法
   │   ├── 分区 (允许超限)
   │   ├── 调度 (串行执行)
   │   └── 记录: 时延 + 分区数
   │
   ├── 运行 Ours 算法
   │   ├── 分区 (拓扑感知)
   │   ├── 调度 (HEFT并行)
   │   └── 记录: 时延 + 分区数
   │
   └── 运行 OCC 算法
       ├── 分区 (单服务器)
       ├── 调度 (串行执行)
       └── 记录: 时延 + 分区数

3. 结果输出
   ├── 保存CSV文件 (results_baseline.csv)
   ├── 打印性能统计
   └── 生成对比摘要
```

## 预期输出

### 输出文件: `results_baseline.csv`

| 列名 | 类型 | 说明 |
|------|------|------|
| `Model` | string | 模型名称 |
| `Servers` | int | 服务器数量 (固定为4) |
| `Bandwidth_Mbps` | int | 网络带宽 (固定为100) |
| `DINA_Latency` | float | DINA算法延迟 (ms) |
| `MEDIA_Latency` | float | MEDIA算法延迟 (ms) |
| `Ours_Latency` | float | Ours算法延迟 (ms) |
| `OCC_Latency` | float | OCC算法延迟 (ms) |
| `DINA_Partitions` | int | DINA分区数量 |
| `MEDIA_Partitions` | int | MEDIA分区数量 |
| `Ours_Partitions` | int | Ours分区数量 |
| `OCC_Partitions` | int | OCC分区数量 |

### 示例数据

```csv
Model,Servers,Bandwidth_Mbps,DINA_Latency,MEDIA_Latency,Ours_Latency,OCC_Latency,DINA_Partitions,MEDIA_Partitions,Ours_Partitions,OCC_Partitions
InceptionV3,4,100,1505.23,1618.45,1367.89,1420.12,42,28,54,39
BERT-base,4,100,2234.67,2245.12,2240.33,2180.45,36,32,38,35
...
```

## 结果解读

### 关键性能指标

1. **端到端延迟 (Latency)**
   - 越低越好
   - 包含计算时间、通信时间、换页开销

2. **分区数量 (Partitions)**
   - 影响通信次数
   - 影响调度灵活性

### 预期观察

基于之前的实验结果，我们预期：

1. **InceptionV3 (并行模型)**
   ```
   Ours < DINA ≈ MEDIA < OCC
   ```
   - Ours利用并行结构，性能最优
   - DINA和MEDIA性能接近（动态惩罚下）

2. **BERT/ViT (线性模型)**
   ```
   DINA ≈ MEDIA ≈ Ours < OCC
   ```
   - 三种分布式算法性能接近
   - 都优于单服务器OCC

3. **分区数量趋势**
   ```
   Ours > DINA > MEDIA
   ```
   - Ours保留并行结构，分区最多
   - DINA严格限制，分区较多
   - MEDIA允许合并，分区最少

## 使用方法

### 前置条件

1. 激活虚拟环境
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

2. 确保依赖已安装
   ```bash
   pip install -r requirements.txt
   ```

3. 确保数据集存在
   ```
   datasets/
   ├── SafeDnnInferenceExp - ALBERT.csv
   ├── SafeDnnInferenceExp - BERT-base.csv
   ├── ...
   └── SafeDnnInferenceExp - ViT-base.csv
   ```

### 运行实验

```bash
python run_baseline_experiment.py
```

### 预期运行时间

- **单个模型**: 约30-60秒
- **全部7个模型**: 约5-8分钟

### 查看结果

```bash
# 查看CSV文件
cat results_baseline.csv

# 或使用Python/Pandas分析
python
>>> import pandas as pd
>>> df = pd.read_csv('results_baseline.csv')
>>> print(df)
```

## 故障排除

### 问题1: 模块未找到

```
ModuleNotFoundError: No module named 'xxx'
```

**解决方案**:
```bash
pip install -r requirements.txt
```

### 问题2: 数据集未找到

```
错误: 在 datasets 目录中未找到任何CSV文件！
```

**解决方案**:
确保 `datasets/` 目录存在且包含模型CSV文件。

### 问题3: 单个模型失败

如果某个模型处理失败，脚本会继续处理其他模型，最后的CSV会包含成功的结果。检查控制台输出中的错误信息。

## 后续实验

基于本基线实验的结果，可以进行：

1. **服务器数量消融实验**
   - 测试 1, 2, 4, 8, 12, 16 台服务器
   - 分析算法的可扩展性

2. **网络带宽消融实验**
   - 测试 1, 10, 50, 100, 500, 1000 Mbps
   - 分析网络瓶颈影响

3. **性能可视化**
   - 生成对比柱状图
   - 生成性能提升热力图
