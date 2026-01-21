# 内存模型错误分析与修复方案

## 问题描述

### 当前实现的错误

在 `common.py` 的 `Partition` 类中（第86行），内存计算方式为：

```python
self.total_memory = sum(l.memory for l in layers)
```

这里 `l.memory` 来自 `tee_total_memory_bytes`，其组成为：
```
tee_total_memory_bytes = weight_bytes + bias_bytes + activation_bytes + tee_encryption_overhead
```

**问题**：简单求和假设所有层的权重、偏置和激活值都需要同时驻留在内存中，这是**完全错误**的。

### 正确的内存模型

在 DNN 推理的流水线执行中，内存占用是**动态**的：

#### 内存生命周期

1. **权重 (weights) 和偏置 (bias)**
   - 在层执行期间必须驻留在内存中
   - 执行结束后继续驻留（可复用）

2. **激活值 (activations)**
   - 层 L 的输出激活 → 传递给后继层作为输入
   - 当所有依赖层 L 输出的层都执行完毕后，L 的激活可以被释放

#### Peak Memory 计算

对于一个包含 n 层的分区 `[L₀, L₁, ..., Lₙ₋₁]`，peak memory 应该是：

```
peak_memory = max over all time points t (
    sum of (weight + bias) for all executed layers +
    sum of activations still needed by future layers
)
```

**简化计算（对于顺序执行的分区）**：

由于分区内的层按拓扑顺序执行，我们可以计算在执行每一层时的内存峰值：

```python
def calculate_peak_memory(layers, dag):
    """
    计算分区的 peak memory
    
    Args:
        layers: 分区中的层列表（已按拓扑排序）
        dag: 完整的DAG图
    
    Returns:
        peak_memory_mb: 峰值内存（MB）
    """
    peak = 0.0
    
    # 持久内存：所有层的 weight + bias
    persistent_memory = sum(l.weight_memory + l.bias_memory for l in layers)
    
    # 对于每个执行点，计算需要保留的激活值
    layer_ids = {l.id for l in layers}
    
    for i, current_layer in enumerate(layers):
        # 当前时刻需要保留的激活值
        active_activations = 0.0
        
        # 检查已执行的层（包括当前层）的激活是否还需要
        for j in range(i + 1):
            layer = layers[j]
            activation_needed = False
            
            # 检查是否有后续层依赖这个激活
            for succ_id in dag.successors(layer.id):
                # 如果后继层还未执行（在分区内）或在分区外
                if succ_id in layer_ids:
                    # 分区内后继
                    succ_idx = next((k for k, l in enumerate(layers) if l.id == succ_id), None)
                    if succ_idx is not None and succ_idx > i:
                        activation_needed = True
                        break
                else:
                    # 分区外后继，激活需要保留到分区结束
                    activation_needed = True
                    break
            
            if activation_needed:
                active_activations += layer.activation_memory
        
        # 当前时刻的总内存 = 持久内存 + 活跃激活值
        current_memory = persistent_memory + active_activations
        peak = max(peak, current_memory)
    
    return peak
```

## 修复方案

### 第一步：扩展 DNNLayer 类

需要将 `tee_total_memory_bytes` 拆分为独立的字段：

```python
class DNNLayer:
    def __init__(self, layer_id, name, weight_memory, bias_memory, activation_memory, 
                 encryption_overhead, cpu_time, enclave_time, output_bytes, execution_mode='Unknown'):
        self.id = layer_id
        self.name = name
        
        # 内存组成部分（MB）
        self.weight_memory = weight_memory
        self.bias_memory = bias_memory
        self.activation_memory = activation_memory
        self.encryption_overhead = encryption_overhead
        
        # 总内存（兼容性）
        self.memory = weight_memory + bias_memory + activation_memory + encryption_overhead
        
        self.cpu_time = cpu_time
        self.enclave_time = enclave_time
        self.output_bytes = output_bytes
        self.execution_mode = execution_mode
        self.workload = enclave_time
```

### 第二步：修改 loader.py

读取 CSV 时分别提取各个内存组件：

```python
# Memory components
weight_bytes = float(get_val(row, ['weight_bytes'], 0))
bias_bytes = float(get_val(row, ['bias_bytes'], 0))
activation_bytes = float(get_val(row, ['activation_bytes'], 0))
encryption_overhead = float(get_val(row, ['tee_encryption_overhead'], 0))

# Convert to MB
weight_mb = weight_bytes / (1024 * 1024)
bias_mb = bias_bytes / (1024 * 1024)
activation_mb = activation_bytes / (1024 * 1024)
encryption_mb = encryption_overhead / (1024 * 1024)

layer = DNNLayer(idx, name, weight_mb, bias_mb, activation_mb, encryption_mb,
                cpu_time, enclave_time, out_bytes, execution_mode)
```

### 第三步：修改 Partition 类

```python
class Partition:
    def __init__(self, partition_id, layers, dag=None):
        self.id = partition_id
        self.layers = layers
        self.dag = dag  # 需要 DAG 来计算 peak memory
        
        # 正确计算 peak memory
        if dag is not None:
            self.total_memory = self._calculate_peak_memory()
        else:
            # 回退到旧的简单求和（不推荐）
            self.total_memory = sum(l.memory for l in layers)
        
        self.total_workload = sum(l.workload for l in layers)
        self.assigned_server = None
        self.start_time = 0.0
        self.finish_time = 0.0
        self.ready_time = 0.0
    
    def _calculate_peak_memory(self):
        """计算分区的峰值内存需求"""
        if not self.layers:
            return 0.0
        
        # 持久内存：weight + bias + encryption overhead
        persistent_memory = sum(
            l.weight_memory + l.bias_memory + l.encryption_overhead 
            for l in self.layers
        )
        
        # 计算激活值的峰值
        peak_activation = self._calculate_peak_activation()
        
        return persistent_memory + peak_activation
    
    def _calculate_peak_activation(self):
        """计算激活值的峰值需求"""
        if not self.dag:
            # 保守估计：所有激活同时存在
            return sum(l.activation_memory for l in self.layers)
        
        peak = 0.0
        layer_ids = {l.id for l in self.layers}
        
        # 对于每个执行时刻
        for i in range(len(self.layers)):
            current_activation = 0.0
            
            # 检查每个已执行的层的激活是否还需要
            for j in range(i + 1):
                layer = self.layers[j]
                
                # 检查是否有后继层需要这个激活
                has_successor_in_partition = False
                has_successor_outside = False
                
                for succ_id in self.dag.successors(layer.id):
                    if succ_id in layer_ids:
                        # 分区内后继
                        succ_idx = next((k for k, l in enumerate(self.layers) if l.id == succ_id), None)
                        if succ_idx is not None and succ_idx > i:
                            has_successor_in_partition = True
                            break
                    else:
                        # 分区外后继
                        has_successor_outside = True
                
                if has_successor_in_partition or has_successor_outside:
                    current_activation += layer.activation_memory
            
            peak = max(peak, current_activation)
        
        return peak
```

### 第四步：修改所有算法

所有算法在创建 Partition 时都需要传入 DAG：

#### OCC 算法

```python
# 修改前
partitions.append(Partition(len(partitions), current_layers))

# 修改后
partitions.append(Partition(len(partitions), current_layers, self.G))
```

#### DINA 算法

```python
# 修改前
partitions.append(Partition(len(partitions), current_part_layers))

# 修改后
partitions.append(Partition(len(partitions), current_part_layers, self.G))
```

#### MEDIA & Ours 算法

类似的修改，确保所有 Partition 创建时都传入 `self.G`。

## 影响分析

### 对实验结果的影响

1. **分区数量可能增加**
   - 旧模型：过度估计内存，导致过多的小分区
   - 新模型：更准确的内存计算，可能允许更大的分区

2. **端到端时延可能降低**
   - 更少的分区 → 更少的分区间通信/切换开销
   - 更准确的内存约束 → 更少的不必要的 penalty

3. **不同算法的差异可能缩小**
   - 旧模型下，内存计算的不准确性可能夸大了算法差异
   - 新模型下，差异主要来自调度策略，更加公平

## 实施计划

1. ✅ **验证分析正确性**（本文档）
2. ⬜ **实现新的内存模型**
   - 修改 `common.py`
   - 修改 `loader.py`
3. ⬜ **更新所有算法**
   - `alg_occ.py`
   - `alg_dina.py`  
   - `alg_media.py`
   - `alg_ours.py`
4. ⬜ **添加向后兼容性**
   - 支持旧数据格式（没有拆分的内存字段）
5. ⬜ **测试验证**
   - 单元测试
   - 回归测试
6. ⬜ **重新运行所有实验**

## 注意事项

### 边界情况

1. **单层分区**
   - Peak memory = weight + bias + activation + encryption overhead
   - 与旧模型一致

2. **线性链分区**
   - Peak memory = 所有 weight/bias + 最多2个激活值
   - 显著低于旧模型

3. **有分支的分区**
   - 需要仔细跟踪哪些激活还被需要
   - 这是最复杂的情况

### 性能考虑

新的 peak memory 计算需要遍历 DAG，时间复杂度为 O(n²)，其中 n 是分区中的层数。对于大分区，这可能会成为瓶颈。

**优化建议**：
- 缓存计算结果
- 在分区合并时增量更新 peak memory
- 对于特定拓扑（如链式），使用快速路径
