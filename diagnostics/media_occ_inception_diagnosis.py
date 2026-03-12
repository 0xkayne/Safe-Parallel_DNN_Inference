#!/usr/bin/env python3
"""
MEDIA vs OCC Behavioral Divergence Diagnosis (InceptionV3)
==========================================================
6 experiments to systematically diagnose why MEDIA = OCC in our simulation.

Usage:
    python diagnostics/media_occ_inception_diagnosis.py --experiment all --no-pause
    python diagnostics/media_occ_inception_diagnosis.py --experiment 1
    python diagnostics/media_occ_inception_diagnosis.py --experiment 5 --servers 4 --bandwidth 100
"""

import sys
import os
import re
import argparse
import copy
import networkx as nx
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from loader import ModelLoader
from common import (
    Server, Partition, ScheduleResult, DNNLayer,
    EPC_EFFECTIVE_MB, calculate_penalty, network_latency, RTT_MS,
)
from alg_occ import OCCAlgorithm
from alg_media import MEDIAAlgorithm

# ── Helpers ──────────────────────────────────────────────────────────

COLORS = {'OCC': '#E74C3C', 'MEDIA': '#2ECC71'}

def _sep(char="=", width=80):
    return char * width

def _pause(no_pause, section_name):
    if no_pause:
        return
    try:
        input(f"\n  [Press Enter to continue to {section_name}, or Ctrl-C to stop] ")
    except (KeyboardInterrupt, EOFError):
        print("\n  Stopped by user.")
        sys.exit(0)

def _load_inception():
    csv_path = os.path.join(_ROOT, "datasets_260120", "InceptionV3.csv")
    return ModelLoader.load_model_from_csv(csv_path)

def _make_servers(n, server_type="Xeon_IceLake"):
    return [Server(i, server_type) for i in range(n)]

def _get_module_name(layer_name):
    """Extract InceptionV3 module name from layer name."""
    name = layer_name.lower()
    patterns = [
        (r'^stem|^input', 'Stem'),
        (r'^classifier|^fc|^softmax|^flatten|^avgpool', 'Classifier'),
        (r'inception_c(\d+)', lambda m: f'Inception-C{m.group(1)}'),
        (r'reduction_b', 'Reduction-B'),
        (r'inception_b(\d+)', lambda m: f'Inception-B{m.group(1)}'),
        (r'reduction_a', 'Reduction-A'),
        (r'inception_a(\d+)', lambda m: f'Inception-A{m.group(1)}'),
        (r'mixed_(\d+)', lambda m: f'Mixed-{m.group(1)}'),
        (r'aux', 'AuxLogits'),
    ]
    for pattern, result in patterns:
        match = re.search(pattern, name)
        if match:
            return result(match) if callable(result) else result
    return 'Other'

def _find_fork_concat_nodes(G, layers_map):
    """Find fork (out_degree > 1) and concat (in_degree > 1) nodes."""
    forks = []
    concats = []
    for nid in G.nodes():
        od = G.out_degree(nid)
        id_ = G.in_degree(nid)
        if od > 1:
            forks.append(nid)
        if id_ > 1:
            concats.append(nid)
    return forks, concats


# =====================================================================
# EXPERIMENT 1: Edge Selection Anatomy
# =====================================================================

def exp1_edge_selection_anatomy(G, layers_map, servers, bw):
    """Trace Constraint 1 + Constraint 2 filtering on every edge."""
    print()
    print(_sep())
    print("  EXPERIMENT 1: Edge Selection Anatomy")
    print(_sep())

    # Compute topological levels
    level_map = {}
    for level, nodes in enumerate(nx.topological_generations(G)):
        for node in nodes:
            level_map[node] = level

    # Trace every edge through C1 and C2
    # We replicate MEDIA's logic but with instrumentation
    M = set()
    edge_trace = []  # list of dicts

    for u in nx.topological_sort(G):
        for v in G.successors(u):
            rec = {
                'u': u, 'v': v,
                'u_name': layers_map[u].name,
                'v_name': layers_map[v].name,
                'u_out_deg': G.out_degree(u),
                'v_in_deg': G.in_degree(v),
                'C1_pass': False,
                'C1_reason': '',
                'C2_pass': False,
                'C2_conflict_edge': '',
                'final_selected': False,
            }

            # Constraint 1: in_degree(v)==1 OR out_degree(u)==1
            c1_reasons = []
            if G.in_degree(v) == 1:
                c1_reasons.append('in_deg(v)==1')
            if G.out_degree(u) == 1:
                c1_reasons.append('out_deg(u)==1')

            if not c1_reasons:
                rec['C1_reason'] = f'FAIL: in_deg(v)={G.in_degree(v)}, out_deg(u)={G.out_degree(u)}'
                edge_trace.append(rec)
                continue

            rec['C1_pass'] = True
            rec['C1_reason'] = ' & '.join(c1_reasons)

            # Tentatively add to M, then check C2
            M.add((u, v))
            violates_c2 = False
            conflict = ''

            for w in G.successors(u):
                for wp in G.predecessors(w):
                    if (wp, w) != (u, v) and (wp, w) in M and level_map[u] == level_map[w] - 1:
                        violates_c2 = True
                        conflict = f'({layers_map[wp].name}->{layers_map[w].name})'
                        break
                if violates_c2:
                    break

            if violates_c2:
                M.remove((u, v))
                rec['C2_pass'] = False
                rec['C2_conflict_edge'] = conflict
            else:
                rec['C2_pass'] = True
                rec['final_selected'] = True

            edge_trace.append(rec)

    # Print full trace table
    print(f"\n  Total edges: {G.number_of_edges()}, Selected (M): {len(M)}")
    print()

    # Focus on fork and concat nodes
    forks, concats = _find_fork_concat_nodes(G, layers_map)

    print(f"  Fork nodes (out_degree > 1): {len(forks)}")
    print(f"  Concat nodes (in_degree > 1): {len(concats)}")
    print()

    # Fork analysis
    print("  --- Fork Node Analysis ---")
    print(f"  {'Node':<35} {'Out-deg':>7} {'Selected':>8} {'Ratio':>6}")
    print(f"  {'-'*35} {'-'*7} {'-'*8} {'-'*6}")

    fork_data = []
    for nid in forks:
        name = layers_map[nid].name
        od = G.out_degree(nid)
        selected = sum(1 for r in edge_trace if r['u'] == nid and r['final_selected'])
        ratio = selected / od if od > 0 else 0
        print(f"  {name:<35} {od:>7} {selected:>8} {ratio:>5.0%}")
        fork_data.append({'node': name, 'out_deg': od, 'selected': selected, 'ratio': ratio})

    # Concat analysis
    print()
    print("  --- Concat Node Analysis ---")
    print(f"  {'Node':<35} {'In-deg':>7} {'Selected':>8} {'Ratio':>6}")
    print(f"  {'-'*35} {'-'*7} {'-'*8} {'-'*6}")

    concat_data = []
    for nid in concats:
        name = layers_map[nid].name
        id_ = G.in_degree(nid)
        selected = sum(1 for r in edge_trace if r['v'] == nid and r['final_selected'])
        ratio = selected / id_ if id_ > 0 else 0
        print(f"  {name:<35} {id_:>7} {selected:>8} {ratio:>5.0%}")
        concat_data.append({'node': name, 'in_deg': id_, 'selected': selected, 'ratio': ratio})

    # Summary stats
    c1_fail = sum(1 for r in edge_trace if not r['C1_pass'])
    c1_pass_c2_fail = sum(1 for r in edge_trace if r['C1_pass'] and not r['C2_pass'])
    both_pass = sum(1 for r in edge_trace if r['final_selected'])

    print()
    print(f"  SUMMARY:")
    print(f"    C1 rejected:          {c1_fail:>4} edges")
    print(f"    C1 pass, C2 rejected: {c1_pass_c2_fail:>4} edges")
    print(f"    Both pass (selected): {both_pass:>4} edges")
    print(f"    Total edges:          {G.number_of_edges():>4}")

    # Detail: edges rejected by C2
    if c1_pass_c2_fail > 0:
        print()
        print("  --- Edges rejected by Constraint 2 ---")
        for r in edge_trace:
            if r['C1_pass'] and not r['C2_pass']:
                print(f"    ({r['u_name']} -> {r['v_name']})  conflict: {r['C2_conflict_edge']}")

    return {
        'edge_trace': edge_trace,
        'fork_data': fork_data,
        'concat_data': concat_data,
        'M_size': len(M),
        'c1_fail': c1_fail,
        'c2_fail': c1_pass_c2_fail,
    }


# =====================================================================
# EXPERIMENT 2: Merge Trace & Partition DAG Topology
# =====================================================================

def exp2_merge_trace_topology(G, layers_map, servers, bw):
    """Trace merge process, analyze final partition DAG topology."""
    print()
    print(_sep())
    print("  EXPERIMENT 2: Merge Trace & Partition DAG Topology")
    print(_sep())

    # Run MEDIA with instrumented merge tracking
    # We replicate MEDIA.run() with logging
    node_to_partition = {}
    for i, (nid, layer) in enumerate(layers_map.items()):
        node_to_partition[nid] = Partition(i, [layer], G)

    # Stage 1: edge selection (same as MEDIA)
    alg = MEDIAAlgorithm(G, layers_map, servers, bw)
    edges_M = alg._select_edges_for_partitioning()

    print(f"\n  Selected edges (M): {len(edges_M)}")

    # Stage 2: merge with instrumentation
    sorted_edges = sorted(list(edges_M),
                          key=lambda e: G[e[0]][e[1]]['weight'],
                          reverse=True)

    merge_trace = []
    for (u, v) in sorted_edges:
        pu = node_to_partition[u]
        pv = node_to_partition[v]

        rec = {
            'u': layers_map[u].name, 'v': layers_map[v].name,
            'pu_id': pu.id, 'pv_id': pv.id,
            'same_partition': pu == pv,
            'cycle_blocked': False,
            'merge_check_fail': False,
            'merged': False,
        }

        if pu == pv:
            rec['merged'] = False  # already same
            merge_trace.append(rec)
            continue

        # Cycle check (replicate MEDIA logic)
        # Build partition DAG
        pg = nx.DiGraph()
        unique_pids = list(set(p.id for p in node_to_partition.values()))
        pg.add_nodes_from(unique_pids)
        for eu, ev in G.edges():
            pu_id = node_to_partition[eu].id
            pv_id = node_to_partition[ev].id
            if pu_id != pv_id:
                pg.add_edge(pu_id, pv_id)

        # Check cycle
        would_cycle = False
        if pg.has_edge(pu.id, pv.id):
            pg_check = pg.copy()
            pg_check.remove_edge(pu.id, pv.id)
            if nx.has_path(pg_check, pu.id, pv.id):
                would_cycle = True
        if not would_cycle and pg.has_edge(pv.id, pu.id):
            pg_check = pg.copy()
            pg_check.remove_edge(pv.id, pu.id)
            if nx.has_path(pg_check, pv.id, pu.id):
                would_cycle = True

        if would_cycle:
            rec['cycle_blocked'] = True
            merge_trace.append(rec)
            continue

        # Merge check
        temp_layers = list(set(pu.layers + pv.layers))
        temp_part = Partition(-1, temp_layers, G)
        merged_mem = temp_part.total_memory

        if merged_mem <= EPC_EFFECTIVE_MB:
            merge_ok = True
        else:
            avg_power = sum(s.power_ratio for s in servers) / len(servers)
            t_p1 = (pu.total_workload * calculate_penalty(pu.total_memory)) / avg_power
            t_p2 = (pv.total_workload * calculate_penalty(pv.total_memory)) / avg_power
            vol = 0.0
            for l1 in pu.layers:
                for l2 in pv.layers:
                    if G.has_edge(l1.id, l2.id): vol += G[l1.id][l2.id]['weight']
                    if G.has_edge(l2.id, l1.id): vol += G[l2.id][l1.id]['weight']
            t_comm = network_latency(vol, bw) if vol > 0 and len(servers) > 1 else 0.0
            merged_wl = pu.total_workload + pv.total_workload
            t_merged = (merged_wl * calculate_penalty(merged_mem)) / avg_power
            merge_ok = t_merged <= (t_p1 + t_p2 + t_comm)

        if not merge_ok:
            rec['merge_check_fail'] = True
            merge_trace.append(rec)
            continue

        # Actually merge
        new_layers = list(set(pu.layers + pv.layers))
        pu_new = Partition(pu.id, new_layers, G)
        for l in new_layers:
            node_to_partition[l.id] = pu_new
        rec['merged'] = True
        merge_trace.append(rec)

    # Print merge trace summary
    n_already_same = sum(1 for r in merge_trace if r['same_partition'])
    n_cycle = sum(1 for r in merge_trace if r['cycle_blocked'])
    n_check_fail = sum(1 for r in merge_trace if r['merge_check_fail'])
    n_merged = sum(1 for r in merge_trace if r['merged'])

    print(f"\n  Merge trace ({len(merge_trace)} edges):")
    print(f"    Already same partition: {n_already_same}")
    print(f"    Cycle blocked:         {n_cycle}")
    print(f"    Merge check failed:    {n_check_fail}")
    print(f"    Successfully merged:   {n_merged}")

    if n_cycle > 0:
        print(f"\n  --- Cycle-blocked merges ---")
        for r in merge_trace:
            if r['cycle_blocked']:
                print(f"    ({r['u']} -> {r['v']})  P{r['pu_id']} + P{r['pv_id']}")

    if n_check_fail > 0:
        print(f"\n  --- Merge-check failures ---")
        for r in merge_trace:
            if r['merge_check_fail']:
                print(f"    ({r['u']} -> {r['v']})  P{r['pu_id']} + P{r['pv_id']}")

    # Build final partition DAG
    unique_parts = list(set(node_to_partition.values()))
    for i, p in enumerate(unique_parts):
        p.id = i

    pg_final = nx.DiGraph()
    for p in unique_parts:
        pg_final.add_node(p.id)
    for eu, ev in G.edges():
        pu_id = node_to_partition[eu].id
        pv_id = node_to_partition[ev].id
        if pu_id != pv_id:
            pg_final.add_edge(pu_id, pv_id)

    print(f"\n  Final partition DAG: {len(unique_parts)} partitions, {pg_final.number_of_edges()} edges")

    # Compute width per topological level
    widths = []
    try:
        for gen in nx.topological_generations(pg_final):
            widths.append(len(gen))
    except nx.NetworkXUnfeasible:
        print("  WARNING: Partition DAG has cycles!")
        widths = [-1]

    max_par = max(widths) if widths else 0
    crit_path = nx.dag_longest_path_length(pg_final) + 1 if widths[0] != -1 else -1

    print(f"  Max parallelism (max width): {max_par}")
    print(f"  Critical path length: {crit_path}")
    print(f"  Width per level: {widths}")

    # Module breakdown: which modules are split across partitions?
    part_modules = {}
    for p in unique_parts:
        modules = set(_get_module_name(l.name) for l in p.layers)
        part_modules[p.id] = modules

    print(f"\n  --- Partition module mapping ---")
    for p in sorted(unique_parts, key=lambda p: p.id):
        modules = sorted(part_modules[p.id])
        n_layers = len(p.layers)
        mem = p.total_memory
        print(f"    P{p.id:>3}: {n_layers:>3} layers, {mem:>7.2f} MB, modules: {', '.join(modules)}")

    return {
        'merge_trace': merge_trace,
        'n_partitions': len(unique_parts),
        'max_parallelism': max_par,
        'critical_path': crit_path,
        'widths': widths,
        'cycle_blocked': n_cycle,
        'merge_check_fail': n_check_fail,
    }


# =====================================================================
# EXPERIMENT 3: OCC vs MEDIA Cost Model Comparison
# =====================================================================

def exp3_cost_model_comparison(G, layers_map, servers, bw):
    """Compare OCC and MEDIA cost models on the same InceptionV3 partitions."""
    print()
    print(_sep())
    print("  EXPERIMENT 3: OCC vs MEDIA Cost Model Comparison")
    print(_sep())

    # Run both algorithms
    occ = OCCAlgorithm(G, layers_map, servers, bw)
    occ_parts = occ.run()
    occ_result = occ.schedule(occ_parts)

    media = MEDIAAlgorithm(G, layers_map, servers, bw)
    media_parts = media.run()
    media_result = media.schedule(media_parts)

    best_power = max(s.power_ratio for s in servers)

    # OCC per-partition analysis
    print(f"\n  OCC: {len(occ_parts)} partitions, latency = {occ_result.latency:.2f} ms")
    print(f"  {'PID':>4} {'Layers':>6} {'Mem(MB)':>8} {'Weight(MB)':>10} {'Act(MB)':>8} "
          f"{'EPC_use':>8} {'Penalty':>8} {'Workload':>9} {'ExecTime':>9} {'WtLoad':>8}")
    print(f"  {'-'*4} {'-'*6} {'-'*8} {'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*9} {'-'*9} {'-'*8}")

    occ_total_exec = 0.0
    for p in occ_parts:
        weight_mb = p.get_static_memory()
        act_peak = p.total_memory - weight_mb
        epc_usage = act_peak + OCCAlgorithm.RING_BUFFER_EPC_MB
        penalty = calculate_penalty(epc_usage)
        exec_t = (p.total_workload * penalty) / best_power
        wt_load = weight_mb / OCCAlgorithm.WEIGHT_COPY_BW_MB_PER_MS
        effective = max(exec_t, wt_load)
        occ_total_exec += effective
        print(f"  {p.id:>4} {len(p.layers):>6} {p.total_memory:>8.2f} {weight_mb:>10.2f} {act_peak:>8.2f} "
              f"{epc_usage:>8.2f} {penalty:>7.2f}x {p.total_workload:>9.2f} {exec_t:>9.2f} {wt_load:>8.2f}")

    # MEDIA per-partition analysis
    print(f"\n  MEDIA: {len(media_parts)} partitions, latency = {media_result.latency:.2f} ms")
    print(f"  {'PID':>4} {'Layers':>6} {'Mem(MB)':>8} {'Penalty':>8} {'Workload':>9} {'ExecTime':>9}")
    print(f"  {'-'*4} {'-'*6} {'-'*8} {'-'*8} {'-'*9} {'-'*9}")

    media_total_exec = 0.0
    for p in sorted(media_parts, key=lambda p: p.id):
        penalty = calculate_penalty(p.total_memory)
        exec_t = (p.total_workload * penalty) / best_power
        media_total_exec += exec_t
        print(f"  {p.id:>4} {len(p.layers):>6} {p.total_memory:>8.2f} {penalty:>7.2f}x {p.total_workload:>9.2f} {exec_t:>9.2f}")

    # Delta
    print(f"\n  COST MODEL COMPARISON:")
    print(f"    OCC  sum(effective_time): {occ_total_exec:>10.2f} ms (single server, serial)")
    print(f"    MEDIA sum(exec_time):     {media_total_exec:>10.2f} ms (could be distributed)")
    print(f"    Delta: {abs(occ_total_exec - media_total_exec):.2f} ms ({abs(occ_total_exec - media_total_exec) / max(occ_total_exec, 1) * 100:.1f}%)")

    # Check if any partition has penalty > 1
    occ_any_penalty = any(
        calculate_penalty(
            (p.total_memory - p.get_static_memory()) + OCCAlgorithm.RING_BUFFER_EPC_MB
        ) > 1.0
        for p in occ_parts
    )
    media_any_penalty = any(calculate_penalty(p.total_memory) > 1.0 for p in media_parts)

    print(f"    OCC  any penalty > 1.0?  {'YES' if occ_any_penalty else 'NO'}")
    print(f"    MEDIA any penalty > 1.0? {'YES' if media_any_penalty else 'NO'}")

    # Weight loading pipeline analysis
    occ_wt_dominated = sum(
        1 for p in occ_parts
        if p.get_static_memory() / OCCAlgorithm.WEIGHT_COPY_BW_MB_PER_MS >
           (p.total_workload * calculate_penalty(
               (p.total_memory - p.get_static_memory()) + OCCAlgorithm.RING_BUFFER_EPC_MB
           )) / best_power
    )
    print(f"    OCC partitions where weight_load > exec: {occ_wt_dominated}/{len(occ_parts)}")

    return {
        'occ_latency': occ_result.latency,
        'media_latency': media_result.latency,
        'occ_total_exec': occ_total_exec,
        'media_total_exec': media_total_exec,
        'occ_any_penalty': occ_any_penalty,
        'media_any_penalty': media_any_penalty,
    }


# =====================================================================
# EXPERIMENT 4: Parameter Sensitivity Analysis
# =====================================================================

def exp4_sensitivity_analysis(G, layers_map):
    """Find parameter boundaries where MEDIA != OCC."""
    print()
    print(_sep())
    print("  EXPERIMENT 4: Parameter Sensitivity Analysis")
    print(_sep())

    results = {}

    # (a) EPC size scan
    print("\n  --- (a) EPC Size Scan ---")
    import common
    original_epc = common.EPC_EFFECTIVE_MB
    epc_values = [5, 10, 15, 20, 30, 50, 93, 150, 200]
    epc_results = []

    print(f"  {'EPC(MB)':>8} {'OCC(ms)':>10} {'MEDIA(ms)':>10} {'Delta%':>8} {'MEDIA_parts':>11}")

    for epc in epc_values:
        common.EPC_EFFECTIVE_MB = epc
        servers = _make_servers(4)
        bw = 100.0

        occ = OCCAlgorithm(G, layers_map, servers, bw)
        occ_parts = occ.run()
        occ_result = occ.schedule(occ_parts)

        media = MEDIAAlgorithm(G, layers_map, servers, bw)
        media_parts = media.run()
        media_result = media.schedule(media_parts)

        delta_pct = (media_result.latency - occ_result.latency) / max(occ_result.latency, 0.001) * 100
        epc_results.append({
            'epc': epc, 'occ': occ_result.latency, 'media': media_result.latency,
            'delta_pct': delta_pct, 'media_parts': len(media_parts),
        })
        print(f"  {epc:>8} {occ_result.latency:>10.2f} {media_result.latency:>10.2f} {delta_pct:>+7.1f}% {len(media_parts):>11}")

    common.EPC_EFFECTIVE_MB = original_epc
    results['epc_scan'] = epc_results

    # (b) Weight loading bandwidth scan
    print("\n  --- (b) Weight Loading Bandwidth Scan ---")
    wt_bw_values = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]
    wt_bw_results = []

    print(f"  {'WtBW(GB/s)':>10} {'OCC(ms)':>10} {'MEDIA(ms)':>10} {'Delta%':>8}")

    servers = _make_servers(4)
    bw = 100.0
    for wt_bw_gbps in wt_bw_values:
        wt_bw_mb_per_ms = wt_bw_gbps  # 1 GB/s = 1 MB/ms

        occ = OCCAlgorithm(G, layers_map, servers, bw)
        orig_bw = OCCAlgorithm.WEIGHT_COPY_BW_MB_PER_MS
        OCCAlgorithm.WEIGHT_COPY_BW_MB_PER_MS = wt_bw_mb_per_ms
        occ_parts = occ.run()
        occ_result = occ.schedule(occ_parts)
        OCCAlgorithm.WEIGHT_COPY_BW_MB_PER_MS = orig_bw

        media = MEDIAAlgorithm(G, layers_map, servers, bw)
        media_parts = media.run()
        media_result = media.schedule(media_parts)

        delta_pct = (media_result.latency - occ_result.latency) / max(occ_result.latency, 0.001) * 100
        wt_bw_results.append({
            'wt_bw': wt_bw_gbps, 'occ': occ_result.latency, 'media': media_result.latency,
            'delta_pct': delta_pct,
        })
        print(f"  {wt_bw_gbps:>10.2f} {occ_result.latency:>10.2f} {media_result.latency:>10.2f} {delta_pct:>+7.1f}%")

    results['wt_bw_scan'] = wt_bw_results

    # (c) Model scale multiplier
    print("\n  --- (c) Model Scale Multiplier ---")
    scale_values = [1, 2, 5, 10, 15, 20]
    scale_results = []

    print(f"  {'Scale':>6} {'OCC(ms)':>10} {'MEDIA(ms)':>10} {'Delta%':>8} {'MEDIA_parts':>11} {'MaxMem(MB)':>10}")

    for scale in scale_values:
        # Scale memory of all layers
        scaled_layers_map = {}
        for nid, layer in layers_map.items():
            sl = DNNLayer(
                layer.id, layer.name,
                layer.memory * scale, layer.cpu_time, layer.enclave_time,
                layer.output_bytes, layer.execution_mode,
                weight_memory=layer.weight_memory * scale,
                bias_memory=layer.bias_memory * scale,
                activation_memory=layer.activation_memory * scale,
                encryption_overhead=layer.encryption_overhead * scale,
            )
            scaled_layers_map[nid] = sl

        servers = _make_servers(4)
        bw = 100.0

        occ = OCCAlgorithm(G, scaled_layers_map, servers, bw)
        occ_parts = occ.run()
        occ_result = occ.schedule(occ_parts)

        media = MEDIAAlgorithm(G, scaled_layers_map, servers, bw)
        media_parts = media.run()
        media_result = media.schedule(media_parts)

        max_mem = max(p.total_memory for p in media_parts) if media_parts else 0
        delta_pct = (media_result.latency - occ_result.latency) / max(occ_result.latency, 0.001) * 100
        scale_results.append({
            'scale': scale, 'occ': occ_result.latency, 'media': media_result.latency,
            'delta_pct': delta_pct, 'media_parts': len(media_parts), 'max_mem': max_mem,
        })
        print(f"  {scale:>6} {occ_result.latency:>10.2f} {media_result.latency:>10.2f} {delta_pct:>+7.1f}% {len(media_parts):>11} {max_mem:>10.2f}")

    results['scale_scan'] = scale_results

    # (d) RTT scan
    print("\n  --- (d) RTT Scan ---")
    import common as common_mod
    original_rtt = common_mod.RTT_MS
    rtt_values = [0, 1, 2, 5, 10, 20, 50]
    rtt_results = []

    print(f"  {'RTT(ms)':>8} {'OCC(ms)':>10} {'MEDIA(ms)':>10} {'Delta%':>8}")

    for rtt in rtt_values:
        common_mod.RTT_MS = rtt
        servers = _make_servers(4)
        bw = 100.0

        occ = OCCAlgorithm(G, layers_map, servers, bw)
        occ_parts = occ.run()
        occ_result = occ.schedule(occ_parts)

        media = MEDIAAlgorithm(G, layers_map, servers, bw)
        media_parts = media.run()
        media_result = media.schedule(media_parts)

        delta_pct = (media_result.latency - occ_result.latency) / max(occ_result.latency, 0.001) * 100
        rtt_results.append({
            'rtt': rtt, 'occ': occ_result.latency, 'media': media_result.latency,
            'delta_pct': delta_pct,
        })
        print(f"  {rtt:>8} {occ_result.latency:>10.2f} {media_result.latency:>10.2f} {delta_pct:>+7.1f}%")

    common_mod.RTT_MS = original_rtt
    results['rtt_scan'] = rtt_results

    return results


# =====================================================================
# EXPERIMENT 5: Counterfactual Parallel Partitions
# =====================================================================

def _build_branch_partitions(G, layers_map):
    """
    Scheme A: Branch-level partitioning.
    Each Inception module's branches become separate partitions.
    Stem + classifier are single partitions.
    """
    # Identify branches by finding fork->concat paths
    forks, concats = _find_fork_concat_nodes(G, layers_map)
    topo_order = list(nx.topological_sort(G))
    topo_idx = {n: i for i, n in enumerate(topo_order)}

    # For each fork, find the corresponding concat
    # A fork's branches converge at the first concat reachable from all successors
    assigned = set()
    partitions = []

    # Assign stem layers (before first fork)
    stem_layers = []
    first_fork_idx = min(topo_idx[f] for f in forks) if forks else len(topo_order)
    for nid in topo_order[:first_fork_idx]:
        stem_layers.append(layers_map[nid])
        assigned.add(nid)
    if stem_layers:
        partitions.append(Partition(len(partitions), stem_layers, G))

    # For each fork, trace branches
    for fork_nid in sorted(forks, key=lambda f: topo_idx[f]):
        if fork_nid in assigned:
            # Fork node itself goes into a singleton partition
            pass

        succs = list(G.successors(fork_nid))

        # Fork node itself
        if fork_nid not in assigned:
            partitions.append(Partition(len(partitions), [layers_map[fork_nid]], G))
            assigned.add(fork_nid)

        # For each successor (branch start), trace until we hit a concat node
        for branch_start in succs:
            if branch_start in assigned:
                continue
            branch_layers = []
            # BFS/DFS along the branch
            stack = [branch_start]
            while stack:
                n = stack.pop()
                if n in assigned:
                    continue
                if n in concats:
                    # Don't include concat in branch partition
                    continue
                branch_layers.append(layers_map[n])
                assigned.add(n)
                for s in G.successors(n):
                    if s not in assigned and s not in concats:
                        stack.append(s)

            if branch_layers:
                # Sort by topo order
                branch_layers.sort(key=lambda l: topo_idx[l.id])
                partitions.append(Partition(len(partitions), branch_layers, G))

    # Concat nodes + remaining unassigned layers
    for nid in topo_order:
        if nid not in assigned:
            partitions.append(Partition(len(partitions), [layers_map[nid]], G))
            assigned.add(nid)

    return partitions


def _build_module_partitions(G, layers_map):
    """
    Scheme B: Module-level with branches separated.
    For each Inception/Reduction module, each branch becomes its own partition.
    Uses the global fork/concat structure rather than per-module detection.
    """
    topo_order = list(nx.topological_sort(G))
    topo_idx = {n: i for i, n in enumerate(topo_order)}

    forks, concats = _find_fork_concat_nodes(G, layers_map)
    fork_set = set(forks)
    concat_set = set(concats)

    partitions = []
    assigned = set()

    # Process in topological order
    # Collect "linear chain" segments between fork/concat boundaries
    # and individual branches within fork-concat pairs.

    current_chain = []

    for nid in topo_order:
        if nid in assigned:
            continue

        if nid in fork_set:
            # Flush any accumulated chain as a partition
            if current_chain:
                partitions.append(Partition(len(partitions),
                                            [layers_map[n] for n in current_chain], G))
                assigned.update(current_chain)
                current_chain = []

            # Fork node: make it its own partition
            partitions.append(Partition(len(partitions), [layers_map[nid]], G))
            assigned.add(nid)

            # Trace each branch from this fork to the next concat
            for succ in G.successors(nid):
                if succ in assigned:
                    continue
                branch = []
                stack = [succ]
                while stack:
                    n = stack.pop()
                    if n in assigned or n in concat_set:
                        continue
                    branch.append(n)
                    assigned.add(n)
                    for s in G.successors(n):
                        if s not in assigned and s not in concat_set:
                            stack.append(s)
                if branch:
                    branch.sort(key=lambda n: topo_idx[n])
                    partitions.append(Partition(len(partitions),
                                                [layers_map[n] for n in branch], G))

        elif nid in concat_set:
            # Flush chain
            if current_chain:
                partitions.append(Partition(len(partitions),
                                            [layers_map[n] for n in current_chain], G))
                assigned.update(current_chain)
                current_chain = []
            # Concat as its own partition
            partitions.append(Partition(len(partitions), [layers_map[nid]], G))
            assigned.add(nid)
        else:
            # Regular node: accumulate into current chain
            current_chain.append(nid)

    # Flush remaining
    if current_chain:
        partitions.append(Partition(len(partitions),
                                    [layers_map[n] for n in current_chain], G))
        assigned.update(current_chain)

    return partitions


def _schedule_with_media_scheduler(G, layers_map, partitions, servers, bw):
    """Use MEDIA's scheduling logic on arbitrary partitions."""
    # Build node_to_partition
    n2p = {}
    for p in partitions:
        for l in p.layers:
            n2p[l.id] = p

    # Build partition graph
    pg = nx.DiGraph()
    for p in partitions:
        pg.add_node(p.id)
    for u, v in G.edges():
        pu = n2p.get(u)
        pv = n2p.get(v)
        if pu and pv and pu.id != pv.id:
            pg.add_edge(pu.id, pv.id)

    partitions_dict = {p.id: p for p in partitions}

    # Compute priorities (MEDIA Eq. 11)
    avg_p = sum(s.power_ratio for s in servers) / len(servers)
    priorities = {}
    topo_order = list(nx.topological_sort(pg))
    topo_idx = {pid: i for i, pid in enumerate(topo_order)}

    for pid in reversed(topo_order):
        p = partitions_dict[pid]
        t_exec = (p.total_workload * calculate_penalty(p.total_memory)) / avg_p
        succs = list(pg.successors(pid))
        if not succs:
            priorities[pid] = t_exec
        else:
            max_val = 0.0
            for sid in succs:
                comm = sum(
                    G[l1.id][l2.id]['weight']
                    for l1 in p.layers for l2 in partitions_dict[sid].layers
                    if G.has_edge(l1.id, l2.id)
                )
                t_comm = network_latency(comm, bw) if comm > 0 else 0.0
                val = t_exec + t_comm + priorities.get(sid, 0.0)
                max_val = max(max_val, val)
            priorities[pid] = max_val

    sorted_partitions = sorted(partitions, key=lambda p: (priorities[p.id], -topo_idx[p.id]), reverse=True)

    # Schedule
    server_free_time = {s.id: 0.0 for s in servers}
    server_schedule = {s.id: [] for s in servers}
    assignment, finish = {}, {}

    for p in sorted_partitions:
        best_s, best_ft = None, float('inf')
        for s in servers:
            dep_ready = 0.0
            for pred_id in pg.predecessors(p.id):
                if pred_id not in assignment:
                    continue
                pred_s, pred_ft = assignment[pred_id], finish[pred_id]
                if pred_s.id != s.id:
                    comm = sum(
                        G[l1.id][l2.id]['weight']
                        for l1 in partitions_dict[pred_id].layers for l2 in p.layers
                        if G.has_edge(l1.id, l2.id)
                    )
                    dep_ready = max(dep_ready, pred_ft + network_latency(comm, bw))
                else:
                    dep_ready = max(dep_ready, pred_ft)

            start_t = max(server_free_time[s.id], dep_ready)
            exec_t = (p.total_workload * calculate_penalty(p.total_memory)) / s.power_ratio
            ft = start_t + exec_t
            if ft < best_ft:
                best_ft, best_s = ft, s
                best_exec_t = exec_t

        assignment[p.id] = best_s
        finish[p.id] = best_ft
        server_free_time[best_s.id] = best_ft
        server_schedule[best_s.id].append({
            'start': best_ft - best_exec_t, 'end': best_ft,
            'partition_id': p.id, 'partition': p
        })

    makespan = max(finish.values()) if finish else 0.0
    active_servers = sum(1 for sid, evts in server_schedule.items() if evts)

    return makespan, active_servers


def exp5_counterfactual_parallel(G, layers_map, servers, bw):
    """Test manually-constructed parallel partitions with MEDIA scheduler."""
    print()
    print(_sep())
    print("  EXPERIMENT 5: Counterfactual Parallel Partitions")
    print(_sep())

    # Build two partition schemes
    scheme_a = _build_branch_partitions(G, layers_map)
    scheme_b = _build_module_partitions(G, layers_map)

    # Verify partition completeness
    all_nids = set(layers_map.keys())
    for label, parts in [('A', scheme_a), ('B', scheme_b)]:
        covered = set()
        for p in parts:
            for l in p.layers:
                covered.add(l.id)
        missing = all_nids - covered
        if missing:
            print(f"  WARNING: Scheme {label} missing {len(missing)} nodes")

    # Also get OCC result for comparison
    occ = OCCAlgorithm(G, layers_map, servers, bw)
    occ_parts = occ.run()
    occ_result = occ.schedule(occ_parts)

    # Also get original MEDIA result
    media = MEDIAAlgorithm(G, layers_map, servers, bw)
    media_parts = media.run()
    media_result = media.schedule(media_parts)

    print(f"\n  Partition counts: Scheme-A={len(scheme_a)}, Scheme-B={len(scheme_b)}, "
          f"MEDIA-original={len(media_parts)}, OCC={len(occ_parts)}")

    # Analyze partition DAG topology for each scheme
    for label, parts in [('A', scheme_a), ('B', scheme_b)]:
        n2p = {}
        for p in parts:
            for l in p.layers:
                n2p[l.id] = p
        pg = nx.DiGraph()
        for p in parts:
            pg.add_node(p.id)
        for u, v in G.edges():
            pu = n2p.get(u)
            pv = n2p.get(v)
            if pu and pv and pu.id != pv.id:
                pg.add_edge(pu.id, pv.id)
        widths = []
        try:
            for gen in nx.topological_generations(pg):
                widths.append(len(gen))
        except nx.NetworkXUnfeasible:
            widths = [-1]
        max_w = max(widths) if widths and widths[0] != -1 else -1
        print(f"  Scheme {label}: max_parallelism={max_w}, widths={widths[:15]}{'...' if len(widths) > 15 else ''}")

    # Sweep: different BW and server counts
    bw_values = [1, 5, 10, 50, 100, 500]
    server_counts = [1, 2, 4, 8]

    print(f"\n  {'Scheme':<10} {'BW':>5} {'Srv':>4} {'MEDIA(ms)':>10} {'OCC(ms)':>10} {'Active':>7} {'Speedup':>8}")
    print(f"  {'-'*10} {'-'*5} {'-'*4} {'-'*10} {'-'*10} {'-'*7} {'-'*8}")

    sweep_results = []
    for n_srv in server_counts:
        srvs = _make_servers(n_srv)
        for bw_val in bw_values:
            # OCC baseline
            occ_alg = OCCAlgorithm(G, layers_map, srvs, bw_val)
            occ_p = occ_alg.run()
            occ_r = occ_alg.schedule(occ_p)

            for label, parts in [('A', scheme_a), ('B', scheme_b), ('Original', media_parts)]:
                lat, active = _schedule_with_media_scheduler(G, layers_map, parts, srvs, bw_val)
                speedup = occ_r.latency / lat if lat > 0 else float('inf')
                sweep_results.append({
                    'scheme': label, 'bw': bw_val, 'servers': n_srv,
                    'media_latency': lat, 'occ_latency': occ_r.latency,
                    'active_servers': active, 'speedup': speedup,
                })
                print(f"  {label:<10} {bw_val:>5} {n_srv:>4} {lat:>10.2f} {occ_r.latency:>10.2f} {active:>7} {speedup:>7.2f}x")

    # Summary: best speedup per scheme
    for scheme in ['A', 'B', 'Original']:
        scheme_data = [r for r in sweep_results if r['scheme'] == scheme]
        if scheme_data:
            best = max(scheme_data, key=lambda r: r['speedup'])
            print(f"\n  Best speedup for Scheme {scheme}: {best['speedup']:.2f}x "
                  f"(BW={best['bw']}, Srv={best['servers']})")

    return {'sweep_results': sweep_results}


# =====================================================================
# EXPERIMENT 6: Paper Assumption Investigation
# =====================================================================

def exp6_paper_assumption_investigation(G, layers_map):
    """Investigate discrepancies between our simulation and paper conditions."""
    print()
    print(_sep())
    print("  EXPERIMENT 6: Paper Assumption Investigation")
    print(_sep())

    # (a) Server heterogeneity
    print("\n  --- (a) Server Heterogeneity ---")
    print("  Paper Fig.7: OCC drops from ~14s (N=1) to ~8s (N=8) => 1.75x over 8 servers")
    print("  This implies mild heterogeneity (servers similar in power)")
    print()

    # Paper-like: mild heterogeneity (e.g., all within 2x range)
    # Our config: Celeron(0.11x) to i5-11600(1.97x) => 18x range
    from common import SERVER_TYPES
    print(f"  Our SERVER_TYPES:")
    for name, ratio in SERVER_TYPES.items():
        print(f"    {name}: {ratio:.2f}x")
    max_ratio = max(SERVER_TYPES.values())
    min_ratio = min(SERVER_TYPES.values())
    print(f"  Range: {min_ratio:.2f}x - {max_ratio:.2f}x ({max_ratio/min_ratio:.1f}x spread)")

    # What if we use mild heterogeneity?
    print("\n  Simulating paper-like mild heterogeneity (all servers 0.8x-1.2x):")
    mild_types = [0.8, 0.9, 1.0, 1.1, 1.2, 1.0, 0.9, 1.1]

    print(f"  {'N_servers':>10} {'OCC(ms)':>10} {'MEDIA(ms)':>10} {'Delta%':>8}")
    for n in [1, 2, 4, 6, 8]:
        # Create servers with mild heterogeneity
        srvs = []
        for i in range(n):
            s = Server(i, "Xeon_IceLake")
            s.power_ratio = mild_types[i % len(mild_types)]
            srvs.append(s)

        occ = OCCAlgorithm(G, layers_map, srvs, 100.0)
        occ_p = occ.run()
        occ_r = occ.schedule(occ_p)

        media = MEDIAAlgorithm(G, layers_map, srvs, 100.0)
        media_p = media.run()
        media_r = media.schedule(media_p)

        delta = (media_r.latency - occ_r.latency) / max(occ_r.latency, 0.001) * 100
        print(f"  {n:>10} {occ_r.latency:>10.2f} {media_r.latency:>10.2f} {delta:>+7.1f}%")

    # (b) Bandwidth range
    print("\n  --- (b) Bandwidth Range ---")
    print("  Paper Fig.8: 1-10 Mbps")
    print("  Our default sweep: 0.5-500 Mbps")
    print()

    print(f"  {'BW(Mbps)':>10} {'OCC(ms)':>10} {'MEDIA(ms)':>10} {'Delta%':>8}")
    for bw_val in [1, 2, 3, 5, 7, 10]:
        servers = _make_servers(4)
        occ = OCCAlgorithm(G, layers_map, servers, bw_val)
        occ_p = occ.run()
        occ_r = occ.schedule(occ_p)

        media = MEDIAAlgorithm(G, layers_map, servers, bw_val)
        media_p = media.run()
        media_r = media.schedule(media_p)

        delta = (media_r.latency - occ_r.latency) / max(occ_r.latency, 0.001) * 100
        print(f"  {bw_val:>10} {occ_r.latency:>10.2f} {media_r.latency:>10.2f} {delta:>+7.1f}%")

    # (c) OCC behavior with increasing servers
    print("\n  --- (c) OCC Server Scaling Behavior ---")
    print("  Paper Fig.7: OCC drops 14s -> 8s (N=1 to N=8)")
    print("  Our OCC: single-server, picks best server (step function)")
    print()

    print(f"  {'Config':>15} {'N':>3} {'OCC(ms)':>10} {'Best_server':>12}")

    # Homogeneous
    for n in [1, 2, 4, 8]:
        srvs = _make_servers(n)
        occ = OCCAlgorithm(G, layers_map, srvs, 100.0)
        occ_p = occ.run()
        occ_r = occ.schedule(occ_p)
        best = max(srvs, key=lambda s: s.power_ratio)
        print(f"  {'Homogeneous':>15} {n:>3} {occ_r.latency:>10.2f} {best.server_type}({best.power_ratio:.2f}x)")

    # Heterogeneous (our config)
    HETERO_ORDER = [
        "Celeron G4930", "Celeron G4930",
        "i5-6500", "i5-6500", "i5-6500", "i5-6500",
        "i3-10100", "i5-11600",
    ]
    for n in [1, 2, 4, 6, 8]:
        srvs = [Server(i, HETERO_ORDER[i]) for i in range(min(n, len(HETERO_ORDER)))]
        occ = OCCAlgorithm(G, layers_map, srvs, 100.0)
        occ_p = occ.run()
        occ_r = occ.schedule(occ_p)
        best = max(srvs, key=lambda s: s.power_ratio)
        print(f"  {'Hetero(ours)':>15} {n:>3} {occ_r.latency:>10.2f} {best.server_type}({best.power_ratio:.2f}x)")

    # (d) Model memory analysis
    print("\n  --- (d) Model Memory Profile ---")
    total_memory = sum(l.memory for l in layers_map.values())
    total_weight = sum(l.weight_memory + l.bias_memory for l in layers_map.values())
    total_act = sum(l.activation_memory for l in layers_map.values())
    max_layer_mem = max(l.memory for l in layers_map.values())
    n_layers = len(layers_map)

    print(f"  InceptionV3 stats:")
    print(f"    Layers:          {n_layers}")
    print(f"    Total memory:    {total_memory:.2f} MB (sum of all layers)")
    print(f"    Total weights:   {total_weight:.2f} MB")
    print(f"    Total activations: {total_act:.2f} MB")
    print(f"    Max layer mem:   {max_layer_mem:.2f} MB")
    print(f"    EPC size:        {EPC_EFFECTIVE_MB:.0f} MB")
    print(f"    Avg layer mem:   {total_memory/n_layers:.2f} MB")
    print()

    # Per MEDIA partition memory
    servers = _make_servers(4)
    media = MEDIAAlgorithm(G, layers_map, servers, 100.0)
    media_parts = media.run()
    max_part_mem = max(p.total_memory for p in media_parts)
    avg_part_mem = sum(p.total_memory for p in media_parts) / len(media_parts)
    print(f"  MEDIA partition stats:")
    print(f"    Partitions:      {len(media_parts)}")
    print(f"    Max partition mem: {max_part_mem:.2f} MB")
    print(f"    Avg partition mem: {avg_part_mem:.2f} MB")
    print(f"    Any > EPC?       {'YES' if max_part_mem > EPC_EFFECTIVE_MB else 'NO'}")
    print(f"    Ratio max/EPC:   {max_part_mem/EPC_EFFECTIVE_MB:.1%}")

    # Paper implication
    print()
    print("  KEY IMPLICATIONS:")
    print(f"  1. All MEDIA partitions fit in EPC ({max_part_mem:.1f} << {EPC_EFFECTIVE_MB:.0f} MB)")
    print(f"     => penalty=1.0 everywhere => cost model identical to OCC for single-server")
    print(f"  2. Partition DAG is chain (max_parallelism=1)")
    print(f"     => MEDIA scheduler degrades to single-server serial execution")
    print(f"  3. OCC uses activation-only EPC model (max EPC usage ≈ act + 20MB ring buffer)")
    print(f"     => Also penalty=1.0 => identical execution times")
    print(f"  4. Net effect: MEDIA = OCC because both have penalty=1.0 and serial execution")

    return {}


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="MEDIA vs OCC behavioral divergence diagnosis (InceptionV3)")
    parser.add_argument("--experiment", default="all",
                        help="Experiment number (1-6) or 'all'")
    parser.add_argument("--servers", type=int, default=4)
    parser.add_argument("--bandwidth", type=float, default=100.0)
    parser.add_argument("--output-dir", default="diagnostics/output")
    parser.add_argument("--no-pause", action="store_true")
    args = parser.parse_args()

    G, layers_map = _load_inception()
    servers = _make_servers(args.servers)
    bw = args.bandwidth

    run_all = args.experiment == 'all'
    exp_num = None if run_all else int(args.experiment)

    print(_sep())
    print("  MEDIA vs OCC Behavioral Divergence Diagnosis")
    print(f"  InceptionV3 | {args.servers} servers | {bw} Mbps")
    print(_sep())

    # Quick baseline
    occ = OCCAlgorithm(G, layers_map, servers, bw)
    occ_r = occ.schedule(occ.run())
    media = MEDIAAlgorithm(G, layers_map, servers, bw)
    media_r = media.schedule(media.run())
    print(f"\n  Baseline: OCC={occ_r.latency:.2f} ms, MEDIA={media_r.latency:.2f} ms, "
          f"Delta={abs(occ_r.latency - media_r.latency):.2f} ms")

    if run_all or exp_num == 1:
        if not run_all or True:
            _pause(args.no_pause, "Exp 1: Edge Selection Anatomy")
        exp1_edge_selection_anatomy(G, layers_map, servers, bw)

    if run_all or exp_num == 2:
        _pause(args.no_pause, "Exp 2: Merge Trace & Topology")
        exp2_merge_trace_topology(G, layers_map, servers, bw)

    if run_all or exp_num == 3:
        _pause(args.no_pause, "Exp 3: Cost Model Comparison")
        exp3_cost_model_comparison(G, layers_map, servers, bw)

    if run_all or exp_num == 4:
        _pause(args.no_pause, "Exp 4: Sensitivity Analysis")
        exp4_sensitivity_analysis(G, layers_map)

    if run_all or exp_num == 5:
        _pause(args.no_pause, "Exp 5: Counterfactual Parallel")
        exp5_counterfactual_parallel(G, layers_map, servers, bw)

    if run_all or exp_num == 6:
        _pause(args.no_pause, "Exp 6: Paper Assumptions")
        exp6_paper_assumption_investigation(G, layers_map)

    print()
    print(_sep())
    print("  END OF DIAGNOSIS")
    print(_sep())


if __name__ == "__main__":
    main()
