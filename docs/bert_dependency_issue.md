# 关键发现：BERT 数据集的依赖关系存在错误

## 1. 问题定位

通过分析 BERT-base 的 CSV 数据，我发现：

**数据中的依赖关系（当前）：**
```
embedding -> q_proj -> k_proj -> v_proj -> softmax -> out_proj -> ...
```

**正确的 Transformer 架构依赖（理论）：**
```
embedding -> q_proj ----+
          -> k_proj ----+-> softmax -> out_proj -> ...
          -> v_proj ----+
```

## 2. 问题本质

CSV 数据文件将 **Q/K/V 投影错误地标记为串行依赖**：
| 层名 | 当前依赖 (错误) | 正确依赖 |
|------|-----------------|----------|
| `encoder0_attn_q_proj` | `['embedding']` | `['embedding']` ✔ |
| `encoder0_attn_k_proj` | `['encoder0_attn_q_proj']` ❌ | `['embedding']` |
| `encoder0_attn_v_proj` | `['encoder0_attn_k_proj']` ❌ | `['embedding']` |

*   **Q/K/V 投影是三个独立的矩阵乘法**，它们只依赖于前一层的输出（embedding 或 norm1），彼此之间无依赖。
*   **但当前数据将它们串成了链**：Q 完成后才做 K，K 完成后才做 V。

## 3. 这为何会破坏并行性？

*   当 `ModelLoader` 加载此数据时，它忠实地构建了一个 **纯线性 DAG**（122 边 / 123 节点 = 每个节点只有 1 个前驱）。
*   Ours 算法扫描 DAG，发现 **没有任何分叉点（Fork）或汇聚点（Join）**，因此无并行可利用。
*   结果：Ours 退化为 DINA 的串行执行。

## 4. 修复方向

### 方案 A：修正 CSV 数据（推荐）
修改 BERT CSV 文件中的 `dependencies` 列，使 Q/K/V 都直接依赖上一个 norm 层：
```csv
encoder0_attn_q_proj, ..., "['embedding']"
encoder0_attn_k_proj, ..., "['embedding']"  # 修复: 原为 q_proj
encoder0_attn_v_proj, ..., "['embedding']"  # 修复: 原为 k_proj
encoder0_attn_softmax, ..., "['encoder0_attn_q_proj', 'encoder0_attn_k_proj', 'encoder0_attn_v_proj']"
```

### 方案 B：在 Loader 中自动修正
在 `loader.py` 中增加逻辑，自动识别 `*_attn_q_proj`, `*_attn_k_proj`, `*_attn_v_proj` 模式，并将它们的依赖统一指向同一个前驱。

## 5. 预期修复效果

修正后，BERT 的 DAG 将拥有：
*   **12 个分叉点**（每个 Encoder 层的 norm -> Q/K/V）
*   **12 个汇聚点**（Q/K/V -> softmax）

这将使 Ours 算法能够识别到这 12 组并行机会，并将 Q/K/V 分配到不同服务器上执行，从而实现类似 InceptionV3 的加速效果。
