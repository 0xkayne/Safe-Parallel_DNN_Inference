# Safe-Parallel DNN Inference

DNN 推理调度仿真系统，在 SGX TEE 边缘集群上对比 4 种分布式推理算法。

## 算法

| 方法 | 文件 | 来源 | 核心策略 |
|------|------|------|---------|
| **OCC** | `alg_occ.py` | MobiCom 2019 | 单机基线，权重 EPC 外，三线流水线 |
| **DINA** | `alg_dina.py` | IEEE TPDS 2024 | 工作负载比例分区 (k=2)，swap-matching 调度 |
| **MEDIA** | `alg_media.py` | — | 度约束边合并 + EPC-aware 分区 + 优先级调度 |
| **Ours** | `alg_ours.py` | 本文 | 算子级 TP + MEDIA 分区 + HEFT 调度 |

## 关键设计

- **Ours/MEDIA/DINA 统一使用 weights-inside-EPC 模型**：权重在 EPC 内，换页由 `calculate_penalty(total_memory)` 处理。分布式 TP 使每台服务器仅持有部分权重，自然适配 EPC。
- **OCC 使用 weights-outside-EPC 模型**：权重经 OCALL 从不可信 DRAM 加载，HMAC 校验后计算。
- **MAX_PART_WL=150ms**：阻止 MEDIA 合并将 CSP 并行分支吞并，释放 YOLOv5/InceptionV3 的结构并行性。
- **DINA k=2**：一个通信跳，既展示分布开销又避免多跳通信爆炸。
- **MEDIA v12 Constraint 2'**：join protection 保留 InceptionV3 不等长分支的并行性。

## 项目结构

```
├── alg_occ.py              # OCC 算法 (单机)
├── alg_dina.py             # DINA 算法 (负载比例分区)
├── alg_media.py            # MEDIA 算法 (度约束合并)
├── alg_ours.py             # Ours 算法 (TP + HEFT)
├── common.py               # 共享数据结构 + 成本模型
├── loader.py               # CSV 模型加载器
├── run_all_experiments.py  # 实验主程序
├── requirements.txt
├── datasets_260120/        # 模型 CSV (7 个模型)
├── exp_results/            # 实验结果 CSV
│   ├── exp1_fixed_comparison/
│   ├── exp2_network_ablation/
│   └── exp3_server_ablation/
├── figures/                # 图表 PNG + PDF
│   ├── exp1/
│   ├── exp2/
│   └── exp3/
├── docs/                   # 设计文档
│   ├── unified-weight-outside-epc-memory-model.md
│   └── dina-method-configuration-analysis.md
├── model_struct_visualization/  # DAG 可视化工具
├── lab-notebook/           # 实验笔记
└── paper_reference/        # 论文参考图
```

## 运行

```bash
pip install -r requirements.txt
python run_all_experiments.py
```

三个实验：
- **Exp1**：固定配置对比 (4×Xeon_IceLake, 100Mbps, 7 模型)
- **Exp2**：网络带宽消融 (0.5–500 Mbps, 4×Xeon)
- **Exp3**：异构服务器消融 (1–8 台: Celeron→i5-6500→i3→i5-11600, 100Mbps)

图表生成在 `figures/` 下。

## 模型

`datasets_260120/` 中的 7 个模型：InceptionV3, VGG-16, YOLOv5, BERT-large, ALBERT-large, ViT-large, ResNet-50。
