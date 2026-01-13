# Legacy Code Analysis Report

This document analyzes the legacy Python files found in the `legacy/` directory, detailing their functionality, evolution, and relationships.

## 1. File Overview

| File | Status | Description |
|------|--------|-------------|
| **`MIDEA.py`** | ❌ Failed/Immature | Early attempt at implementing MEDIA algorithm. Contains basic class structures but logic is marked as "failed/immature". Has Chinese comments. |
| **`MEDIA-GPT.py`** | ✅ Functional | The main legacy implementation of the MEDIA algorithm. Contains full implementations of Algorithm 1 (Edge Selection), 2 (Partitioning), and 3 (Scheduling). Includes detailed debugging prints and a NiN model test case. |
| **`MEDIA-GPT-copy.py`** | ⚠️ Duplicate | Exact copy of `Improved-MEDIA.py`. It seems to be a backup or a misnamed file. It contains the "Improved" logic comments at the top. |
| **`Improved-MEDIA.py`** | 🧪 Experimental | An attempt to improve `MEDIA-GPT.py` by adding **parallelism awareness**. It introduces a `check_parallel_relation` function and modifies edge selection to strictly avoid merging parallel branches. However, comments indicate it "did not succeed" and was "put aside". |

## 2. Relationships & Evolution

### Evolution Path
`MIDEA.py` (Prototype) -> `MEDIA-GPT.py` (Baseline) -> `Improved-MEDIA.py` / `MEDIA-GPT-copy.py` (Attempted Improvement)

### Key Comparisons

#### A. `MIDEA.py` vs. `MEDIA-GPT.py`
*   **Structure**: `MIDEA.py` is object-oriented with a `MEDIAPartitioner` class. `MEDIA-GPT.py` is more procedural with standalone functions (`select_edges`, `graph_partition`, `assign_partitions`).
*   **Completeness**: `MEDIA-GPT.py` is much more complete, including detailed scheduling logic (Algorithm 3) and test runners, whereas `MIDEA.py` focuses mainly on partitioning.
*   **Penalty Model**: Both use a simple static penalty model (e.g., performance halves if > EPC).

#### B. `MEDIA-GPT.py` vs. `Improved-MEDIA.py`
*   **Goal**: `Improved-MEDIA.py` tries to fix a flaw in `MEDIA-GPT.py` where parallel branches (e.g., Inception blocks) were being merged into serial partitions, destroying parallelism.
*   **Changes**:
    *   **Edge Selection**: Changed logic from `if in!=1 AND out!=1` to `if in!=1 OR out!=1`. This is stricter: it refuses to pre-merge *any* branching or merging point, forcing them to be handled by the dynamic check later.
    *   **Parallel Check**: Added `check_parallel_relation` to detect independent branches.
    *   **Merge Check**: Modified `merge_check` to accept an `is_parallel` flag. If parallel, it penalizes merging even more (comparing against `max(t1, t2)` instead of `sum(t1, t2)`).
*   **Outcome**: The file header explicitly states: *"This code attempts to add parallel consideration via heuristics, but did not succeed. Put aside."* This suggests the heuristic approach was insufficient, leading to the development of the current `OursAlgorithm` (which uses HEFT) in the main project.

## 3. Summary of Functionality

### `MEDIA-GPT.py` (The "Standard" Legacy MEDIA)
1.  **Algorithm 1 (Edge Selection)**: Greedily selects edges $(u, v)$ where $u$ has out-degree 1 and $v$ has in-degree 1.
2.  **Algorithm 2 (Partitioning)**: Merges selected edges into partitions.
    *   Checks if `Memory < EPC` OR `Merge_Time < Sep_Time`.
    *   `Sep_Time` assumes serial execution ($T_{sep} = T_1 + T_2 + Comm$).
3.  **Algorithm 3 (Scheduling)**: Assigns partitions to servers using a priority-based list scheduler.

### `Improved-MEDIA.py` (The "Parallel" Attempt)
1.  **Modified Edge Selection**: Stricter. Skips any node with degree != 1.
2.  **Parallel Awareness**: Tries to recognize that for parallel nodes, $T_{sep} = \max(T_1, T_2) + Comm$.
    *   *Bug/Issue*: The simple heuristic likely wasn't robust enough for complex graphs or the overhead modeling was off.

## 4. Conclusion
The current project (root directory) has evolved from these files:
*   `alg_media.py` in the root is a modernized, cleaner version of `MEDIA-GPT.py` logic, integrated with the new `common.py` and `loader.py`.
*   `alg_ours.py` in the root is the successful realization of the goals of `Improved-MEDIA.py`, using a formal HEFT (Heterogeneous Earliest Finish Time) approach instead of ad-hoc heuristics.
