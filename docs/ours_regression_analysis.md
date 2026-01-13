# Ours 算法在 BERT-base 上性能退化的根因分析

## 1. 问题现象
| 模型 | DINA | MEDIA | Ours |
|------|------|-------|------|
| BERT-base | 666.24 ms | 666.24 ms | **739.97 ms (慢 11%)** |

Ours 本应是"最优算法"，但在 BERT-base 上竟然比 DINA 慢了 73ms。这明显是一个缺陷。

## 2. 根因定位：`COMM_WEIGHT` 修复遗漏

在之前修复 MEDIA 性能倒挂时，我们引入了 `COMM_WEIGHT = 0.2` 来抑制"悲观通信预期"导致的过度合并。

**检查发现**：该修复只在 `alg_media.py` 中生效，**`alg_ours.py` 的合并逻辑中漏掉了这个关键修正**。

### 2.1 MEDIA 的代码（已修复 ✔）
```python
# alg_media.py, line 73
if penalty_delta < comm_time * self.COMM_WEIGHT:
    should_merge = True
```
MEDIA 会认为："除非惩罚小于预期通信的 20%，否则不值得冒险合并。" 这让它在 BERT-base 上保守地选择了切分，回归了 DINA 的最优策略。

### 2.2 Ours 的代码（未修复 ✘）
```python
# alg_ours.py, line 80
t_sep = (p_u.total_workload + p_v.total_workload) + t_comm  # <-- 缺少 * self.COMM_WEIGHT
```
Ours 依然在用**完整的 100% 通信代价**来评估切分成本。这使得它高估了切分的代价，于是做出了"合并更划算"的错误决策。

## 3. 因果链推演

1.  **BERT-base 特性**: 是典型的线性模型（每层依赖前一层）。Ours 的"拓扑保护"在此处不起作用（因为没有分支可保护）。
2.  **Ours 的决策流程**:
    *   遍历线性边 $u \to v$。
    *   检查合并后内存是否超过 EPC (93MB)。
    *   **假设**某次合并使内存达到 110MB (超限 18%)。
    *   计算 $t_{merged} = Work \times 1.74$ (动态惩罚)。
    *   计算 $t_{sep} = Work + t_{comm} \times 1.0$ (无权重)。
    *   由于 $t_{comm}$ 在 100Mbps 下可能是几十到上百毫秒，算法误判 $t_{merged} < t_{sep}$。
3.  **结果**: Ours 选择了合并，承受了 74% 的性能惩罚（换页开销）。
4.  **累积效应**: 多次错误合并导致整体时延膨胀，最终 739ms > 666ms。

## 4. 修复方案

在 `alg_ours.py` 的第 80 行，将：
```python
t_sep = (p_u.total_workload + p_v.total_workload) + t_comm
```
改为：
```python
t_sep = (p_u.total_workload + p_v.total_workload) + t_comm * self.COMM_WEIGHT
```

## 5. 预期修复效果
*   **BERT-base**: Ours 会正确识别出"超限合并不划算"，退守 DINA 策略，达到 666ms。
*   **InceptionV3**: 不受影响，因为 Ours 的优势来自并行调度，不是超限合并。
