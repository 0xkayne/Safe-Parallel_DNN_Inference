# 实验笔记总览

**研究课题**：面向 SGX TEE 边缘集群的安全并行 DNN 推理调度

**核心问题**：在 SGX Enclave（EPC 内存 93MB）约束下，如何将大规模 DNN 模型高效分布到多个异构边缘服务器，最小化端到端推理延迟，同时保证安全性。

**方法对比**：OCC（单机串行 paging 基线）/ DINA（强制服务器轮换）/ MEDIA（图分割 + 贪心调度）/ Ours（HPA 张量并行 + HEFT 调度）

**实验平台**：Python 3.12 纯仿真，12 个 DNN 模型（BERT/ViT/ALBERT/DistilBERT/TinyBERT/InceptionV3）

---

## 当前研究进展

| 阶段 | 状态 | 核心内容 |
|------|------|---------|
| Phase 1：系统搭建与 baseline 验证 | 完成 | 仿真框架、数据集、4 算法接口 |
| Phase 2：Bug 排查与修复 | 完成 | 通信量二次除法 bug，5 处修复 |
| Phase 3：实验结果分析 | 完成 | 3 组实验数据，结果符合理论预期 |
| Phase 4：算法理论分析 | 完成 | MEDIA≡OCC 根因，Ours 优势机制 |

---

## 快速链接

- [Phase 1 笔记](phase-1-系统搭建与baseline验证/notes.md)
- [Phase 2 笔记](phase-2-Bug排查与修复/notes.md)
- [Phase 3 笔记](phase-3-实验结果分析/notes.md)
- [Phase 4 笔记](phase-4-算法理论分析/notes.md)
- [**论文/专利素材索引**](material-index.md) ← 写论文/专利从这里开始

---

## 关键文件路径

```
datasets_260120/          -- 12 个 DNN 模型数据集（CSV）
alg_occ.py                -- OCC 算法
alg_dina.py               -- DINA 算法（含 bug 修复记录）
alg_media.py              -- MEDIA 算法（含 bug 修复记录）
alg_ours.py               -- 我们的 HPA 算法
common.py                 -- 共享数据结构与 cost model
loader.py                 -- 模型 CSV 加载（edge weight 以 MB 为单位）
run_all_experiments.py    -- 实验驱动脚本
exp_results/              -- 实验结果 CSV
figures/                  -- 生成的图表（PNG + PDF）
diagnostics/diagnose.py   -- 算法诊断工具
```
