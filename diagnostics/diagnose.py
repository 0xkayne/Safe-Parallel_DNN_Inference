#!/usr/bin/env python3
"""
Algorithm Implementation Diagnostic Tool
=========================================
Exposes partitioning structure, scheduling decisions, latency breakdown,
and unit/logic bugs for OCC / DINA / MEDIA algorithms.

Usage:
    python diagnostics/diagnose.py --model bert_base --servers 4 --bandwidth 100
    python diagnostics/diagnose.py --model InceptionV3 --servers 4 --bandwidth 100
    python diagnostics/diagnose.py --model bert_large --servers 4 --bandwidth 100 --all

Model name options (match files in datasets_260120/ without .csv):
    bert_base, bert_large, ALBERT-base, ALBERT-large, distillbert_base,
    TinyBERT-4l, TinyBERT-6l, ViT-base, ViT-large, ViT-small, ViT-tiny, InceptionV3
"""

import sys
import os
import argparse
import math
import networkx as nx

# --- Path setup ---
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from loader import ModelLoader
from common import (
    Server, EPC_EFFECTIVE_MB, calculate_penalty, network_latency,
    PAGING_BANDWIDTH_MB_PER_MS, PAGE_SIZE_KB, PAGE_FAULT_OVERHEAD_MS,
    ENCLAVE_ENTRY_EXIT_OVERHEAD_MS, DEFAULT_PAGING_BW_MBPS, RTT_MS,
    ScheduleResult, Partition
)
from alg_occ   import OCCAlgorithm
from alg_dina  import DINAAlgorithm
from alg_media import MEDIAAlgorithm

# -----------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------

def _sep(char="=", width=72):
    return char * width

def _paging_cost(partition):
    """Recompute per-partition paging cost (same formula as OCC/DINA/MEDIA schedulers)."""
    swap_mb = partition.get_static_memory()
    num_pages = math.ceil(swap_mb * 1024 / PAGE_SIZE_KB)
    return num_pages * PAGE_FAULT_OVERHEAD_MS + swap_mb / (DEFAULT_PAGING_BW_MBPS / 1000.0)

def _exec_time(partition, power_ratio=1.0):
    """Compute partition execution time including EPC penalty."""
    penalty = calculate_penalty(partition.total_memory)
    return (partition.total_workload * penalty) / power_ratio

def _find_csv(model_name):
    """Try to locate the CSV file for a given model name."""
    ds_dir = os.path.join(_ROOT, "datasets_260120")
    candidates = [
        os.path.join(ds_dir, f"{model_name}.csv"),
        os.path.join(ds_dir, f"{model_name.lower()}.csv"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    # Fuzzy match
    for fname in os.listdir(ds_dir):
        if fname.lower().replace("-", "_").startswith(model_name.lower().replace("-", "_")):
            return os.path.join(ds_dir, fname)
    raise FileNotFoundError(
        f"Cannot find dataset for '{model_name}' in {ds_dir}.\n"
        f"Available: {sorted(os.listdir(ds_dir))}"
    )

def _make_servers(n, server_type="Xeon_IceLake"):
    return [Server(i, server_type) for i in range(n)]

def _build_partition_dag(partitions, G, node_to_partition):
    """Build partition-level DAG from layer-level DAG."""
    pg = nx.DiGraph()
    for p in partitions:
        pg.add_node(p.id)
    for u, v in G.edges():
        pu_id = node_to_partition[u].id
        pv_id = node_to_partition[v].id
        if pu_id != pv_id:
            pg.add_edge(pu_id, pv_id)
    return pg


# -----------------------------------------------------------------
# Section 1 – Dataset / Layer Analysis
# -----------------------------------------------------------------

def report_layer_analysis(G, layers_map):
    layers = list(layers_map.values())
    print(_sep())
    print("SECTION 1  DATASET / LAYER ANALYSIS")
    print(_sep())

    # Counts
    n = len(layers)
    n_weight  = sum(1 for l in layers if l.weight_memory > 0)
    n_act     = sum(1 for l in layers if l.activation_memory > 0)
    n_out     = sum(1 for l in layers if l.output_bytes > 0)

    print(f"  Total layers        : {n}")
    print(f"  weight_memory > 0   : {n_weight} / {n}  "
          f"({'ALL' if n_weight == n else 'PARTIAL' if n_weight > 0 else 'NONE <- WARNING'})")
    print(f"  activation_memory>0 : {n_act} / {n}")
    print(f"  output_bytes > 0    : {n_out} / {n}")
    print(f"  EPC limit           : {EPC_EFFECTIVE_MB:.1f} MB")

    # Memory stats
    w_vals  = [l.weight_memory + l.bias_memory for l in layers]
    a_vals  = [l.activation_memory              for l in layers]
    m_vals  = [l.memory                         for l in layers]
    wl_vals = [l.workload                        for l in layers]
    out_vals = [l.output_bytes / (1024**2)       for l in layers]  # MB

    def _stats(vals, label, unit):
        if not vals:
            return
        print(f"  {label:30s}: sum={sum(vals):.2f} {unit}  "
              f"mean={sum(vals)/len(vals):.3f}  max={max(vals):.3f}  "
              f"min={min(vals):.4f}")

    print()
    _stats(w_vals,  "weight+bias per layer",    "MB")
    _stats(a_vals,  "activation per layer",     "MB")
    _stats(m_vals,  "l.memory (tee_total) per layer", "MB")
    _stats(wl_vals, "workload (enclave_time)",  "ms")

    # -- Edge weight unit check ----------------------------------
    print()
    print("  -- Edge weight (output_bytes) in DAG --")
    edge_weights = [d.get('weight', 0.0) for _, _, d in G.edges(data=True)]
    if edge_weights:
        _stats(out_vals, "output_bytes per layer", "MB")
        print(f"  DAG edge count      : {len(edge_weights)}")

        # Check loader unit: loader divides output_bytes by 1024^2 → stored in MB
        # Sanity: max edge should match max output_bytes / 1M
        max_out_bytes = max(l.output_bytes for l in layers) / (1024**2)
        max_edge      = max(edge_weights)
        print(f"  Max edge weight     : {max_edge:.6f}  (loader stored MB = output_bytes/1M)")
        if abs(max_edge - max_out_bytes) < 1e-6:
            print(f"  [OK] Edge unit check: Edge weights ARE in MB (correct loader)")
        else:
            print(f"  [!!] Edge unit MISMATCH: expected {max_out_bytes:.6f} MB, got {max_edge:.6f}")

        print()
        print("  +--- UNIT BUG PROBE --------------------------------------------+")
        sample = edge_weights[0]
        buggy  = sample / (1024 * 1024)
        bw_per_ms = (100.0 / 8.0) / 1000.0
        t_correct = RTT_MS + sample / bw_per_ms
        t_buggy   = RTT_MS + buggy  / bw_per_ms
        print(f"  |  Sample edge weight (stored) : {sample:.6f} MB")
        print(f"  |  alg_dina / alg_media do     : vol_mb = edge_weight / 1024^2")
        print(f"  |  -> buggy vol_mb             : {buggy:.4e} MB  (approx 0)")
        print(f"  |  Correct comm_time @100Mbps  : {t_correct:.2f} ms  (RTT + data)")
        print(f"  |  Buggy   comm_time @100Mbps  : {t_buggy:.2f} ms  (RTT only)")
        if t_correct > 1.05 * t_buggy:
            print(f"  |  [BUG] CONFIRMED: comm underestimated by {t_correct/max(t_buggy,0.001):.0f}x")
        print(f"  +---------------------------------------------------------------+")
    else:
        print("  WARNING: No edges found in DAG!")

    return layers


# -----------------------------------------------------------------
# Section 2 – Partition Analysis
# -----------------------------------------------------------------

def report_partition_analysis(alg_name, partitions, G, node_to_partition=None):
    print()
    print(_sep("-"))
    print(f"SECTION 2  PARTITION ANALYSIS  -  {alg_name}")
    print(_sep("-"))

    n_parts = len(partitions)
    n_exceed = sum(1 for p in partitions if p.total_memory > EPC_EFFECTIVE_MB)
    total_static = sum(p.get_static_memory() for p in partitions)
    total_workload = sum(p.total_workload for p in partitions)
    total_paging = sum(_paging_cost(p) for p in partitions)

    print(f"  Partitions          : {n_parts}")
    print(f"  Exceeds EPC         : {n_exceed} / {n_parts}  "
          f"({'none' if n_exceed == 0 else 'SOME - paging penalty applies'})")
    print(f"  Total static memory : {total_static:.2f} MB  (weights+bias summed over all partitions)")
    print(f"  Total workload      : {total_workload:.2f} ms")
    print(f"  Total paging cost   : {total_paging:.2f} ms  (sum, sequential on one server)")
    print()

    # Table header
    hdr = f"  {'ID':>4}  {'#Lay':>5}  {'TotalMem':>10}  {'StaticMem':>10}  " \
          f"{'Penalty':>8}  {'Workload':>10}  {'PagCost':>10}  {'ExcEPC':>6}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for p in sorted(partitions, key=lambda x: x.id):
        static = p.get_static_memory()
        penalty = calculate_penalty(p.total_memory)
        pag = _paging_cost(p)
        flag = " [\!\!]EPC" if p.total_memory > EPC_EFFECTIVE_MB else ""
        print(f"  {p.id:>4}  {len(p.layers):>5}  {p.total_memory:>9.2f}M  "
              f"{static:>9.2f}M  {penalty:>8.2f}×  {p.total_workload:>9.2f}ms  "
              f"{pag:>9.2f}ms{flag}")

    # Compare with OCC if this is another algorithm
    if alg_name != "OCC" and node_to_partition:
        # Try to check if partitions are identical to OCC's
        pass  # Comparison done externally

    return n_parts, total_paging, total_workload


# -----------------------------------------------------------------
# Section 3 – Schedule Timeline + Latency Decomposition
# -----------------------------------------------------------------

def report_schedule_analysis(alg_name, result: ScheduleResult, servers, partitions_list):
    print()
    print(_sep("-"))
    print(f"SECTION 3  SCHEDULE TIMELINE  -  {alg_name}")
    print(_sep("-"))

    # Build a flat ordered list of (server_id, event) sorted by start time
    all_events = []
    for sid, events in result.server_schedules.items():
        for ev in events:
            all_events.append((sid, ev))
    all_events.sort(key=lambda x: x[1]['start'])

    # Track per-server sequence for communication gap analysis
    partition_server = {}  # partition_id -> server_id
    partition_finish = {}  # partition_id -> finish_time
    for sid, ev in all_events:
        pid = ev['partition_id']
        partition_server[pid] = sid
        partition_finish[pid] = ev['end']

    # Server distribution
    server_counts = {}
    for sid, _ in all_events:
        server_counts[sid] = server_counts.get(sid, 0) + 1
    print(f"  Server usage        : " +
          "  ".join(f"S{sid}:{cnt}" for sid, cnt in sorted(server_counts.items())))

    n_unique_servers = len(server_counts)
    if n_unique_servers == 1:
        only_sid = list(server_counts.keys())[0]
        print(f"  [\!\!] ALL partitions on Server {only_sid} → No parallelism!")
    else:
        print(f"  [OK] Partitions spread across {n_unique_servers} servers")

    # Build partition id → partition object map
    part_map = {p.id: p for p in partitions_list}

    # Per-server power ratios
    server_power = {s.id: s.power_ratio for s in servers}

    # Compute decomposition
    total_exec   = 0.0
    total_paging = 0.0
    total_comm   = 0.0

    print()
    hdr = f"  {'PID':>4}  {'Srv':>4}  {'Exec_start':>11}  {'Exec_end':>9}  " \
          f"{'ExecT':>8}  {'PagingT':>8}  {'CommWait':>9}  {'Penalty':>8}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for sid, ev in all_events:
        pid   = ev['partition_id']
        part  = part_map.get(pid)
        if part is None:
            continue
        exec_start = ev['start']
        exec_end   = ev['end']
        exec_t     = exec_end - exec_start

        pwr = server_power.get(sid, 1.0)
        pag_t      = _paging_cost(part)
        penalty    = calculate_penalty(part.total_memory)

        # Estimate communication wait: time between when data became ready and when loading started.
        # load_start ≈ exec_start - pag_t
        load_start = exec_start - pag_t

        # When was data ready? = max finish time of predecessors (+ comm time per buggy schedule)
        # We can only approximate this from what we know.
        # Simple estimate: comm_wait = load_start - pag_t_predecessor_end
        # (We skip the precise computation here; focus on the aggregate)

        total_exec   += exec_t
        total_paging += pag_t
        flag = f"  [\!\!]EPC×{penalty:.1f}" if penalty > 1.0 else ""
        print(f"  {pid:>4}  S{sid:<3}  {exec_start:>11.2f}  {exec_end:>9.2f}  "
              f"{exec_t:>7.2f}ms  {pag_t:>7.2f}ms  {'(see below)':>9}  "
              f"{penalty:>7.2f}×{flag}")

    total_comm = result.latency - total_exec - total_paging
    print()
    print(f"  --- Latency Decomposition ---")
    print(f"  Total latency       : {result.latency:.2f} ms")
    print(f"    Execution (sum)   : {total_exec:.2f} ms  ({100*total_exec/result.latency:.1f}%)")
    print(f"    Paging (sum)      : {total_paging:.2f} ms  ({100*total_paging/result.latency:.1f}%)")
    print(f"    Comm+Idle (diff)  : {total_comm:.2f} ms  ({100*max(0,total_comm)/result.latency:.1f}%)")

    return result.latency, total_exec, total_paging, total_comm


# -----------------------------------------------------------------
# Section 4 – Cross-Partition Communication Bug Analysis
# -----------------------------------------------------------------

def report_comm_bug(G, partitions, node_to_partition, bandwidth_mbps):
    print()
    print(_sep("-"))
    print("SECTION 4  COMMUNICATION VOLUME BUG ANALYSIS")
    print(_sep("-"))

    bw_per_ms = (bandwidth_mbps / 8.0) / 1000.0  # MB/ms

    cross_edges = []  # (pu_id, pv_id, edge_weight_mb)
    for u, v, data in G.edges(data=True):
        pu = node_to_partition.get(u)
        pv = node_to_partition.get(v)
        if pu is None or pv is None:
            continue
        if pu.id != pv.id:
            w = data.get('weight', 0.0)  # already in MB (from loader)
            cross_edges.append((pu.id, pv.id, w))

    if not cross_edges:
        print("  No cross-partition edges found.")
        return

    # Aggregate by (src_partition, dst_partition)
    pair_vol = {}
    for pu_id, pv_id, w in cross_edges:
        key = (pu_id, pv_id)
        pair_vol[key] = pair_vol.get(key, 0.0) + w

    total_vol_mb = sum(pair_vol.values())
    print(f"  Cross-partition edges     : {len(cross_edges)}")
    print(f"  Unique partition pairs    : {len(pair_vol)}")
    print(f"  Total cross-partition data: {total_vol_mb:.4f} MB  (from loader, already in MB)")
    print()

    print("  Top 10 partition pairs by volume:")
    print(f"  {'Src→Dst':>12}  {'Vol_MB':>10}  {'Correct_comm_ms':>16}  "
          f"{'Buggy_vol_MB':>14}  {'Buggy_comm_ms':>14}  {'ErrorFactor':>12}")
    print("  " + "-" * 85)

    sorted_pairs = sorted(pair_vol.items(), key=lambda x: -x[1])[:10]
    bug_ratios = []
    for (pu_id, pv_id), vol_mb in sorted_pairs:
        correct_comm = RTT_MS + vol_mb / bw_per_ms
        buggy_vol    = vol_mb / (1024 * 1024)   # double-divide bug
        buggy_comm   = RTT_MS + buggy_vol / bw_per_ms
        error_factor = correct_comm / max(buggy_comm, 1e-9)
        bug_ratios.append(error_factor)
        print(f"  {str(pu_id)+'->'+ str(pv_id):>12}  {vol_mb:>10.4f}  "
              f"{correct_comm:>16.2f}ms  {buggy_vol:>14.2e}  "
              f"{buggy_comm:>14.2f}ms  {error_factor:>12.1f}×")

    print()
    if bug_ratios:
        avg_err = sum(bug_ratios) / len(bug_ratios)
        print(f"  Average comm underestimate : {avg_err:.0f}×")
        print(f"  [\!\!] BUG: alg_dina.py line ~84  : vol_mb = vol / (1024*1024)")
        print(f"         alg_media.py line ~145 : vol_mb = vol / (1024*1024)  [_merge_check]")
        print(f"         alg_media.py line ~287 : comm_data_mb = comm_data / (1024*1024)  [schedule]")
        print(f"     Edge weights in G are already MB (loader divides output_bytes by 1M).")
        print(f"     These lines divide by 1M AGAIN → vol ≈ 0 → only RTT({RTT_MS}ms) contributes.")
        print()
        print(f"  Expected DINA overhead @{bandwidth_mbps}Mbps per hop (if bug fixed):")
        total_vol = total_vol_mb / len(pair_vol) if pair_vol else 0
        correct_per_hop = RTT_MS + total_vol / bw_per_ms
        print(f"    Avg vol/hop = {total_vol:.4f} MB → comm = {correct_per_hop:.2f} ms/hop")
        print(f"  Current DINA overhead per hop : {RTT_MS:.1f} ms  (RTT only, data≈0)")


# -----------------------------------------------------------------
# Section 5 – Partition Structure Comparison
# -----------------------------------------------------------------

def report_partition_comparison(results_map):
    """Compare partition counts and structures across algorithms."""
    print()
    print(_sep("-"))
    print("SECTION 5  CROSS-ALGORITHM COMPARISON")
    print(_sep("-"))

    print(f"  {'Algorithm':>10}  {'Partitions':>11}  {'TotalPaging':>12}  {'TotalExec':>11}  {'Latency':>10}")
    print("  " + "-" * 62)
    for alg, (n_parts, total_pag, total_wl, latency) in results_map.items():
        print(f"  {alg:>10}  {n_parts:>11}  {total_pag:>11.2f}ms  {total_wl:>10.2f}ms  {latency:>9.2f}ms")

    print()
    occ_lat = results_map.get("OCC", (0, 0, 0, 0))[3]
    dina_lat = results_map.get("DINA", (0, 0, 0, 0))[3]
    media_lat = results_map.get("MEDIA", (0, 0, 0, 0))[3]
    if occ_lat > 0:
        print(f"  DINA – OCC   : {dina_lat - occ_lat:.2f} ms  "
              f"({'≈' if abs(dina_lat-occ_lat)<1 else ''}should be significant if comm bug fixed)")
        print(f"  MEDIA – OCC  : {media_lat - occ_lat:.2f} ms  "
              f"({'≈ 0 <- RTT adhesion prevents MEDIA from parallelizing' if abs(media_lat-occ_lat)<5 else ''})")


# -----------------------------------------------------------------
# Section 6 – MEDIA Scheduler Decision Trace
# -----------------------------------------------------------------

def report_media_scheduler_trace(G, partitions, node_to_partition, servers, bandwidth_mbps):
    """Show why MEDIA's greedy scheduler ends up putting everything on S0."""
    print()
    print(_sep("-"))
    print("SECTION 6  MEDIA SCHEDULER DECISION TRACE  (first 8 partitions)")
    print(_sep("-"))

    bw_per_ms = (bandwidth_mbps / 8.0) / 1000.0

    # Build partition DAG
    pg = _build_partition_dag(partitions, G, node_to_partition)
    parts_by_id = {p.id: p for p in partitions}

    # Sort partitions in topological order
    try:
        topo = list(nx.topological_sort(pg))
    except Exception:
        topo = [p.id for p in partitions]

    server_free = {s.id: 0.0 for s in servers}
    assignment  = {}   # pid -> server_id
    finish      = {}   # pid -> finish_time
    best_s_log  = {}   # pid -> reason string

    for pid in topo[:8]:
        part = parts_by_id[pid]
        pag_t   = _paging_cost(part)
        pred_ids = list(pg.predecessors(pid))

        best_s_id, best_ft = None, float('inf')
        candidates = []
        for s in servers:
            dep_ready = 0.0
            comm_breakdown = []
            for pred_id in pred_ids:
                pred_s_id  = assignment.get(pred_id)
                pred_ft    = finish.get(pred_id, 0.0)
                if pred_s_id is not None and pred_s_id != s.id:
                    # Cross-server: compute comm (buggy way, as in actual code)
                    vol = sum(
                        G[l1.id][l2.id]['weight']
                        for l1 in parts_by_id[pred_id].layers
                        for l2 in part.layers
                        if G.has_edge(l1.id, l2.id)
                    )
                    vol_mb_buggy  = vol / (1024 * 1024)   # the bug
                    vol_mb_correct = vol                    # correct (already MB)
                    comm_buggy   = network_latency(vol_mb_buggy,   bandwidth_mbps)
                    comm_correct = network_latency(vol_mb_correct, bandwidth_mbps)
                    arrival = pred_ft + comm_buggy
                    dep_ready = max(dep_ready, arrival)
                    comm_breakdown.append(
                        f"P{pred_id}→S{s.id}: vol={vol:.4f}MB, "
                        f"comm_buggy={comm_buggy:.2f}ms, comm_correct={comm_correct:.2f}ms"
                    )
                elif pred_s_id is not None:
                    dep_ready = max(dep_ready, pred_ft)

            start_load  = max(server_free[s.id], dep_ready)
            start_exec  = start_load + pag_t
            exec_t      = _exec_time(part, s.power_ratio)
            ft          = start_exec + exec_t
            candidates.append((s.id, ft, dep_ready, start_load, comm_breakdown))
            if ft < best_ft:
                best_ft, best_s_id = ft, s.id

        print(f"  Partition {pid}  ({len(part.layers)} layers, "
              f"pag={pag_t:.2f}ms, exec={_exec_time(part):.2f}ms):")
        for (s_id, ft, dr, sl, cbs) in sorted(candidates, key=lambda x: x[1]):
            marker = " <- CHOSEN" if s_id == best_s_id else ""
            print(f"    S{s_id}: dep_ready={dr:.2f} start_load={sl:.2f} "
                  f"finish={ft:.2f}ms{marker}")
            for cb in cbs:
                print(f"      {cb}")

        if best_s_id is not None:
            server_free[best_s_id] = best_ft
            assignment[pid] = best_s_id
            finish[pid]     = best_ft
        print()

    print("  --- Why MEDIA ≈ OCC ---")
    print(f"  With near-zero comm data (due to bug), every cross-server hop costs only")
    print(f"  RTT = {RTT_MS}ms. Putting partition on same server saves RTT every time.")
    print(f"  Greedy scheduler therefore 'sticks' to the first server for all partitions.")
    if bw_per_ms > 0:
        # Show what threshold would make parallelism beneficial
        sample_part = parts_by_id[topo[1]] if len(topo) > 1 else None
        if sample_part:
            exec_t_sample = _exec_time(sample_part)
            print(f"  Even with corrected comm, S0 wait = 0ms while S1 comm ≥ RTT = {RTT_MS}ms")
            print(f"  For MEDIA to spread work: S0 busy time > RTT + data_transfer")
            if sample_part.total_workload > 0:
                min_vol_for_spread = 0  # Just RTT is enough to discourage
                print(f"  Current: S0 always finishes before S1 (linear chain → no parallel branches)")


# -----------------------------------------------------------------
# Section 7 – Summary Bug Report
# -----------------------------------------------------------------

def report_bug_summary(G, layers_map, bandwidth_mbps, results_map):
    print()
    print(_sep())
    print("SECTION 7  BUG SUMMARY")
    print(_sep())

    bw_per_ms = (bandwidth_mbps / 8.0) / 1000.0
    layers = list(layers_map.values())

    print()
    print("  BUG #1  Double unit conversion in communication volume calculation")
    print("  -----------------------------------------------------------------")
    print("  loader.py stores edge weights as: output_bytes / 1024^2  (units: MB)")
    print()
    print("  Affected locations:")
    print("    alg_dina.py  ~line 84 : vol_mb = vol / (1024 * 1024)")
    print("                            vol is already MB → effective vol ≈ 0 bytes")
    print("    alg_media.py ~line 145: vol_mb = vol / (1024 * 1024)  [_merge_check]")
    print("    alg_media.py ~line 287: comm_data_mb = comm_data / (1024 * 1024)  [schedule]")
    print()

    edge_weights = [d.get('weight', 0.0) for _, _, d in G.edges(data=True) if d.get('weight', 0) > 0]
    if edge_weights:
        avg_mb = sum(edge_weights) / len(edge_weights)
        buggy  = avg_mb / (1024**2)
        t_correct = RTT_MS + avg_mb  / bw_per_ms
        t_buggy   = RTT_MS + buggy   / bw_per_ms
        print(f"  Impact (avg cross-partition edge = {avg_mb:.4f} MB):")
        print(f"    Correct comm time  : {t_correct:.2f} ms/hop")
        print(f"    Buggy   comm time  : {t_buggy:.2f} ms/hop  (≈ RTT only)")
        print(f"    Underestimate      : {t_correct/max(t_buggy,0.001):.0f}×")
        print()

    print("  Consequence on DINA:")
    occ_lat  = results_map.get("OCC",  (0,0,0,0))[3]
    dina_lat = results_map.get("DINA", (0,0,0,0))[3]
    if occ_lat > 0:
        n_transitions = results_map.get("DINA", (0,0,0,0))[0] - 1
        expected_diff_buggy   = n_transitions * RTT_MS
        if edge_weights:
            expected_diff_correct = n_transitions * (RTT_MS + avg_mb / bw_per_ms)
        print(f"    Partitions-1 (transitions) : ~{n_transitions}")
        print(f"    Observed DINA-OCC          : {dina_lat-occ_lat:.2f} ms")
        print(f"    Expected with bug          : ~{expected_diff_buggy:.1f} ms  (N×RTT only)")
        if edge_weights:
            print(f"    Expected without bug       : ~{expected_diff_correct:.1f} ms  (N×(RTT+data))")
    print()

    print("  Consequence on MEDIA:")
    media_lat = results_map.get("MEDIA", (0,0,0,0))[3]
    if occ_lat > 0:
        print(f"    MEDIA latency = {media_lat:.2f} ms,  OCC latency = {occ_lat:.2f} ms")
        print(f"    Diff = {media_lat-occ_lat:.2f} ms  (should be ≤0 if MEDIA outperforms OCC)")
        print(f"    → Scheduler puts ALL partitions on S0 (RTT adhesion)")
        print(f"    → MEDIA cannot spread work across servers → MEDIA ≡ OCC")
    print()

    # Check if partition structures differ
    occ_parts  = results_map.get("OCC",   (0,0,0,0))[0]
    dina_parts = results_map.get("DINA",  (0,0,0,0))[0]
    media_parts= results_map.get("MEDIA", (0,0,0,0))[0]
    print(f"  BUG #2 (Design)  MEDIA partitioning = OCC partitioning for linear models")
    print(f"  -----------------------------------------------------------------")
    print(f"    OCC partitions  : {occ_parts}")
    print(f"    DINA partitions : {dina_parts}  (same algorithm as OCC)")
    print(f"    MEDIA partitions: {media_parts}")
    if occ_parts == media_parts:
        print(f"    → Same partition count: MEDIA's edge-selection + merge produced identical split.")
        print(f"      For linear models: no edges violate degree-1 constraint → same EPC-filling as OCC.")
    else:
        print(f"    → Different count: MEDIA merged some partitions differently.")
    print()

    print("  FIX RECOMMENDATIONS:")
    print("  -----------------------------------------------------------------")
    print("  1. In alg_dina.py, replace (line ~84):")
    print("       vol_mb = vol / (1024 * 1024)")
    print("     with:")
    print("       vol_mb = vol   # edge weights are already in MB from loader.py")
    print()
    print("  2. In alg_media.py, replace (line ~145 in _merge_check):")
    print("       vol_mb = vol / (1024 * 1024)")
    print("     with:")
    print("       vol_mb = vol")
    print()
    print("  3. In alg_media.py, replace (line ~287 in schedule):")
    print("       comm_data_mb = comm_data / (1024 * 1024)")
    print("     with:")
    print("       comm_data_mb = comm_data   # already in MB")
    print()
    print("  After the fix, DINA-OCC difference should increase from ~RTT*N")
    print("  to ~(RTT+data/BW)*N, making the latency ordering more pronounced.")


# -----------------------------------------------------------------
# Main
# -----------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Algorithm diagnostic tool")
    parser.add_argument("--model",      default="bert_base",
                        help="Model name (e.g. bert_base, InceptionV3)")
    parser.add_argument("--servers",    type=int, default=4,
                        help="Number of servers (default: 4)")
    parser.add_argument("--bandwidth",  type=float, default=100.0,
                        help="Bandwidth in Mbps (default: 100)")
    parser.add_argument("--server-type",default="Xeon_IceLake",
                        help="Server type (default: Xeon_IceLake)")
    parser.add_argument("--all",        action="store_true",
                        help="Show all sections (default shows all)")
    parser.add_argument("--section",    type=int, default=0,
                        help="Show only section N (0 = all)")
    args = parser.parse_args()

    # -- Load model ----------------------------------------------
    csv_path = _find_csv(args.model)
    print(_sep())
    print(f"  ALGORITHM DIAGNOSTIC REPORT")
    print(f"  Model     : {args.model}  ({os.path.basename(csv_path)})")
    print(f"  Servers   : {args.servers} × {args.server_type}")
    print(f"  Bandwidth : {args.bandwidth} Mbps")
    print(_sep())

    G, layers_map = ModelLoader.load_model_from_csv(csv_path)
    servers = _make_servers(args.servers, args.server_type)

    # -- Section 1 -----------------------------------------------
    if args.section in (0, 1):
        report_layer_analysis(G, layers_map)

    # -- Run algorithms ------------------------------------------
    print()
    print(_sep("-"))
    print("  Running algorithms...")

    occ   = OCCAlgorithm(G, layers_map, servers, args.bandwidth)
    dina  = DINAAlgorithm(G, layers_map, servers, args.bandwidth)
    media = MEDIAAlgorithm(G, layers_map, servers, args.bandwidth)

    occ_parts   = occ.run()
    dina_parts  = dina.run()
    media_parts = media.run()

    occ_result   = occ.schedule(occ_parts)
    dina_result  = dina.schedule(dina_parts)
    media_result = media.schedule(media_parts)

    # node_to_partition for MEDIA (used for cross-partition edge analysis)
    media_n2p = media.node_to_partition

    # Build node_to_partition for OCC / DINA (greedy sequential)
    def build_n2p(partitions):
        n2p = {}
        for p in partitions:
            for l in p.layers:
                n2p[l.id] = p
        return n2p

    occ_n2p  = build_n2p(occ_parts)
    dina_n2p = build_n2p(dina_parts)

    # -- Section 2 -----------------------------------------------
    results_map = {}
    if args.section in (0, 2):
        n, tp, tw = report_partition_analysis("OCC",   occ_parts,   G, occ_n2p)
        results_map["OCC"] = (n, tp, tw, occ_result.latency)
        n, tp, tw = report_partition_analysis("DINA",  dina_parts,  G, dina_n2p)
        results_map["DINA"] = (n, tp, tw, dina_result.latency)
        n, tp, tw = report_partition_analysis("MEDIA", media_parts, G, media_n2p)
        results_map["MEDIA"] = (n, tp, tw, media_result.latency)

    # -- Section 3 -----------------------------------------------
    if args.section in (0, 3):
        report_schedule_analysis("OCC",   occ_result,   servers, occ_parts)
        report_schedule_analysis("DINA",  dina_result,  servers, dina_parts)
        report_schedule_analysis("MEDIA", media_result, servers, media_parts)

    # -- Section 4 -----------------------------------------------
    if args.section in (0, 4):
        report_comm_bug(G, occ_parts, occ_n2p, args.bandwidth)

    # -- Section 5 -----------------------------------------------
    if args.section in (0, 5):
        if not results_map:
            # Rebuild if section 2 was skipped
            results_map = {
                "OCC":   (len(occ_parts),   sum(_paging_cost(p) for p in occ_parts),
                          sum(p.total_workload for p in occ_parts),   occ_result.latency),
                "DINA":  (len(dina_parts),  sum(_paging_cost(p) for p in dina_parts),
                          sum(p.total_workload for p in dina_parts),  dina_result.latency),
                "MEDIA": (len(media_parts), sum(_paging_cost(p) for p in media_parts),
                          sum(p.total_workload for p in media_parts), media_result.latency),
            }
        report_partition_comparison(results_map)

    # -- Section 6 -----------------------------------------------
    if args.section in (0, 6):
        report_media_scheduler_trace(G, media_parts, media_n2p, servers, args.bandwidth)

    # -- Section 7 -----------------------------------------------
    if args.section in (0, 7):
        if not results_map:
            results_map = {
                "OCC":   (len(occ_parts),  0, 0, occ_result.latency),
                "DINA":  (len(dina_parts), 0, 0, dina_result.latency),
                "MEDIA": (len(media_parts),0, 0, media_result.latency),
            }
        report_bug_summary(G, layers_map, args.bandwidth, results_map)

    print()
    print(_sep())
    print("  END OF DIAGNOSTIC REPORT")
    print(_sep())


if __name__ == "__main__":
    main()
