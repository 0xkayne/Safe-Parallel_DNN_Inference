# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Simulation system for a graduation thesis comparing 4 DNN inference scheduling algorithms on SGX TEE edge clusters. The goal is to measure end-to-end inference latency under varying server counts and network bandwidths. **This is a pure simulation** — no actual SGX hardware is involved; all costs are computed analytically from profiled layer data.

Language: Python 3.12. No build step required.

## Running Experiments

```bash
# Run all 3 experiments + generate all charts
python run_all_experiments.py

# Individual experiment functions can be called from Python:
#   run_fixed_comparison()    — Exp1: 4×Xeon, 100Mbps, all models
#   run_server_ablation()     — Exp3: heterogeneous 1-8 servers
#   run_network_ablation()    — Exp2: 0.5-500 Mbps bandwidth sweep
#   generate_server_charts()  — Exp3 figures
#   generate_network_charts() — Exp2 figures
#   generate_combined_charts()— Combined grid images
```

Dependencies: `pip install networkx pandas numpy matplotlib pyvis pillow`

## Architecture

### Data Flow

```
datasets_260120/*.csv  →  loader.py (ModelLoader)  →  (nx.DiGraph, layers_map)
                                                           ↓
                                              alg_*.py  .run()   → partitions
                                              alg_*.py  .schedule(partitions) → ScheduleResult
                                                           ↓
                                              run_all_experiments.py  → exp_results/ CSVs + figures/
```

All four algorithms share the same interface: `__init__(G, layers_map, servers, bandwidth_mbps)`, then `.run()` returns partitions, `.schedule(partitions)` returns a `ScheduleResult`.

### Core Modules

- **`common.py`** — Shared simulation primitives:
  - `DNNLayer`, `Partition`, `Server`, `ScheduleResult` data classes
  - SGX cost model: `calculate_penalty()`, `network_latency()`, `hpa_cost()`, `enclave_init_cost()`
  - Key constants: `EPC_EFFECTIVE_MB=93`, `RTT_MS=5`, `PAGING_BANDWIDTH_MB_PER_MS=1.0`
  - `SERVER_TYPES` dict maps CPU names to compute power ratios (baseline Xeon=1.0)
  - `Partition._calculate_peak_memory()` tracks activation liveness via DAG analysis

- **`loader.py`** — `ModelLoader.load_model_from_csv()` parses dataset CSVs into a NetworkX DAG. Handles virtual QKV splitting for old-format datasets and dependency edge creation. Returns `(G: nx.DiGraph, layers_map: dict[int, DNNLayer])`.

- **`run_all_experiments.py`** — Experiment orchestrator. Configures server clusters (homogeneous/heterogeneous), sweeps parameters, collects results into CSVs, generates matplotlib charts (PNG+PDF), and creates combined grid images.

### The Four Algorithms

| File | Class | Strategy | Key Trait |
|------|-------|----------|-----------|
| `alg_occ.py` | `OCCAlgorithm` | Strict ≤EPC partitioning | Single-server serial baseline; no network cost |
| `alg_dina.py` | `DINAAlgorithm` | Strict ≤EPC partitioning | Forced server rotation (line 73): consecutive partitions must use different servers |
| `alg_media.py` | `MEDIAAlgorithm` | Allows >EPC (paging vs communication tradeoff) | MEDIA-style edge selection + greedy merge + priority scheduling |
| `alg_ours.py` | `OursAlgorithm` | HPA: tensor parallelism + MEDIA partitioning + HEFT scheduling | 5-stage pipeline: candidate filtering → cost surface → DAG DP → graph augmentation → HEFT |

### Dataset Format

CSVs in `datasets_260120/` (12 models: BERT/ALBERT/DistilBERT/TinyBERT/ViT variants + InceptionV3). Key columns per layer:
- `name`, `type`, `group`, `dependencies` (JSON list of parent layer names)
- `enclave_time_mean` (ms), `output_bytes`, `tee_total_memory_bytes`
- `weight_bytes`, `bias_bytes`, `activation_bytes` (granular memory breakdown)

### Output Structure

```
exp_results/
  exp1_fixed_comparison/   — Single CSV: all models × all methods
  exp2_network_ablation/   — Per-model CSVs: latency vs bandwidth
  exp3_server_ablation/    — Per-model CSVs: latency vs server count
figures/
  exp1/  — Bar chart (PNG+PDF)
  exp2/  — Line charts per model + combined grid
  exp3/  — Line charts per model + combined grid
```

## Important Implementation Details

- **Partition memory**: Uses peak memory (weights + peak live activations via DAG liveness analysis), not sum of all layers. See `Partition._calculate_peak_memory()` and `Partition.get_static_memory()` (weights-only, used for swap cost).
- **Heterogeneous servers**: Exp3 uses incremental addition order: `[2×Celeron, 4×i5-6500, 1×i3-10100, 1×i5-11600]`. Power ratios range from 0.11 to 1.97.
- **InceptionV3** is the only model with significant parallel branch structure — it's the key model for demonstrating Ours(HPA) advantage.
- **Linear models** (BERT/ViT): MEDIA ≈ Ours is expected behavior (no parallel structure to exploit).
- **`archive/`** contains legacy algorithm implementations and old scripts — not used in current experiments.
