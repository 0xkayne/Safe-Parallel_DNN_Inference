#!/usr/bin/env python3
"""
InceptionV3 Deep Diagnostic Analysis Tool
==========================================
Focused diagnostic for InceptionV3 across all 4 algorithms (OCC, DINA, MEDIA, Ours).
Outputs 6 sections (0-5) with interactive pauses between each.

Usage:
    python diagnostics/inception_analysis.py [--servers 4] [--bandwidth 100] [--section N] [--no-pause] [--output-dir figures/inception_diagnostic]
"""

import sys
import os
import re
import argparse
import math
import networkx as nx
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from loader import ModelLoader
from common import (
    Server, EPC_EFFECTIVE_MB, calculate_penalty, network_latency,
    ScheduleResult, Partition, PAGING_BANDWIDTH_MB_PER_MS,
    PAGE_SIZE_KB, PAGE_FAULT_OVERHEAD_MS, DEFAULT_PAGING_BW_MBPS,
    ENCLAVE_ENTRY_EXIT_OVERHEAD_MS, RTT_MS,
)
from alg_occ import OCCAlgorithm
from alg_dina import DINAAlgorithm
from alg_media import MEDIAAlgorithm
from alg_ours import OursAlgorithm

# ── Color scheme (project standard) ──────────────────────────────────
COLORS = {'OCC': '#E74C3C', 'DINA': '#3498DB', 'MEDIA': '#2ECC71', 'Ours': '#9B59B6'}
ALG_ORDER = ['OCC', 'DINA', 'MEDIA', 'Ours']

# ── Helpers ──────────────────────────────────────────────────────────

def _sep(char="=", width=72):
    return char * width


def _pause(no_pause: bool, section_name: str):
    if no_pause:
        return
    try:
        input(f"\n  [Press Enter to continue to {section_name}, or Ctrl-C to stop] ")
    except (KeyboardInterrupt, EOFError):
        print("\n  Stopped by user.")
        sys.exit(0)


def build_node_to_partition(partitions):
    """Build layer_id -> Partition mapping from a partitions list (OCC/DINA)."""
    n2p = {}
    for p in partitions:
        for layer in p.layers:
            n2p[layer.id] = p
    return n2p


def build_partition_dag(partitions, G, n2p):
    """Build partition-level DAG. Returns nx.DiGraph with edge attr 'comm_mb'."""
    pg = nx.DiGraph()
    for p in partitions:
        pg.add_node(p.id)
    edge_comm = {}  # (pu_id, pv_id) -> total comm MB
    for u, v, data in G.edges(data=True):
        pu = n2p.get(u)
        pv = n2p.get(v)
        if pu is None or pv is None:
            continue
        if pu.id != pv.id:
            key = (pu.id, pv.id)
            edge_comm[key] = edge_comm.get(key, 0.0) + data.get('weight', 0.0)
    for (pu_id, pv_id), comm in edge_comm.items():
        pg.add_edge(pu_id, pv_id, comm_mb=comm)
    return pg


def get_module_name(layer_name):
    """Extract InceptionV3 module name from layer name."""
    name = layer_name.lower()
    # Remove shard/allreduce suffixes
    name = re.sub(r'_shard_\d+$', '', name)
    name = re.sub(r'_allreduce$', '', name)

    module_patterns = [
        (r'^stem', 'Stem'),
        (r'^classifier|^fc|^softmax|^flatten', 'Classifier'),
        (r'inception_c(\d+)', lambda m: f'Inception-C{m.group(1)}'),
        (r'reduction_b', 'Reduction-B'),
        (r'inception_b(\d+)', lambda m: f'Inception-B{m.group(1)}'),
        (r'reduction_a', 'Reduction-A'),
        (r'inception_a(\d+)', lambda m: f'Inception-A{m.group(1)}'),
        (r'mixed_(\d+)', lambda m: f'Mixed-{m.group(1)}'),
        (r'aux', 'AuxLogits'),
    ]
    for pattern, result in module_patterns:
        match = re.search(pattern, name)
        if match:
            if callable(result):
                return result(match)
            return result
    return 'Other'


def compute_latency_components(result, partitions, servers, G, n2p, bandwidth, alg_name):
    """Decompose makespan into {compute, paging, communication, idle}.

    Components sum to makespan (not to total work across all servers).
    We compute per-server busy time, identify the bottleneck server,
    and attribute its time to compute vs paging. Communication and idle
    come from gaps in the bottleneck server's timeline.
    """
    part_map = {p.id: p for p in partitions}
    server_power = {s.id: s.power_ratio for s in servers}
    makespan = result.latency
    n_servers = len(servers)

    # Per-server: total busy time, base compute, paging
    server_busy = {s.id: 0.0 for s in servers}
    server_base_compute = {s.id: 0.0 for s in servers}
    server_paging = {s.id: 0.0 for s in servers}

    # Collect per-event assignment for comm analysis
    pid_to_server = {}
    pid_to_times = {}  # pid -> (start, end)

    for sid, events in result.server_schedules.items():
        pwr = server_power.get(sid, 1.0)
        for ev in events:
            pid = ev['partition_id']
            part = part_map.get(pid)
            if part is None:
                continue
            duration = ev['end'] - ev['start']
            server_busy[sid] += duration
            pid_to_server[pid] = sid
            pid_to_times[pid] = (ev['start'], ev['end'])

            if alg_name in ('OCC', 'Ours'):
                activation_peak = part.total_memory - part.get_static_memory()
                epc_usage = activation_peak + 20.0
                penalty = calculate_penalty(epc_usage)
            else:
                penalty = calculate_penalty(part.total_memory)

            base = part.total_workload / pwr
            paging_add = (penalty - 1.0) * part.total_workload / pwr if penalty > 1.0 else 0.0
            server_base_compute[sid] += base
            server_paging[sid] += paging_add

    # Find bottleneck server (highest busy time)
    bottleneck_sid = max(server_busy, key=server_busy.get)

    # Communication: identify gaps in event timeline caused by cross-server deps
    total_comm_gap = 0.0
    pg = build_partition_dag(partitions, G, n2p)
    for sid, events in result.server_schedules.items():
        sorted_evs = sorted(events, key=lambda e: e['start'])
        for i, ev in enumerate(sorted_evs):
            pid = ev['partition_id']
            # Check if this event waited for a cross-server predecessor
            for pred_pid in pg.predecessors(pid):
                pred_sid = pid_to_server.get(pred_pid)
                if pred_sid is not None and pred_sid != sid:
                    pred_end = pid_to_times.get(pred_pid, (0, 0))[1]
                    # The gap between predecessor end and this event start includes comm
                    if pred_end < ev['start']:
                        # Estimate comm time for this edge
                        edge_data = pg.get_edge_data(pred_pid, pid)
                        comm_mb = edge_data.get('comm_mb', 0.0) if edge_data else 0.0
                        comm_t = network_latency(comm_mb, bandwidth) if comm_mb > 0 else 0.0
                        total_comm_gap += min(comm_t, ev['start'] - pred_end)

    # Distribute components proportionally to makespan
    total_busy_all = sum(server_busy.values())
    total_base_all = sum(server_base_compute.values())
    total_paging_all = sum(server_paging.values())

    # Use bottleneck server's ratio for the decomposition
    bn_busy = server_busy[bottleneck_sid]
    bn_base = server_base_compute[bottleneck_sid]
    bn_paging = server_paging[bottleneck_sid]

    # Scale: bottleneck server busy + its idle = makespan
    bn_idle = max(0.0, makespan - bn_busy)

    # Communication is estimated from cross-server gaps, capped
    comm = max(0.0, min(total_comm_gap, makespan - bn_busy))
    idle = max(0.0, makespan - bn_busy - comm)

    return {
        'compute': bn_base,
        'paging': bn_paging,
        'communication': comm,
        'idle': idle,
    }


# ── Section 0: Setup ────────────────────────────────────────────────

def setup(servers_count, bandwidth, server_type="Xeon_IceLake"):
    """Run all 4 algorithms on InceptionV3 and return context dict."""
    csv_path = os.path.join(_ROOT, "datasets_260120", "InceptionV3.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"InceptionV3 dataset not found: {csv_path}")

    G, layers_map = ModelLoader.load_model_from_csv(csv_path)
    servers = [Server(i, server_type) for i in range(servers_count)]

    ctx = {
        'G': G,
        'layers_map': layers_map,
        'servers': servers,
        'bandwidth': bandwidth,
        'server_type': server_type,
        'algs': {},
    }

    for alg_name, AlgClass in [('OCC', OCCAlgorithm), ('DINA', DINAAlgorithm),
                                ('MEDIA', MEDIAAlgorithm), ('Ours', OursAlgorithm)]:
        alg = AlgClass(G, layers_map, servers, bandwidth)
        parts = alg.run()
        result = alg.schedule(parts)

        # Build node_to_partition
        if alg_name in ('OCC', 'DINA'):
            n2p = build_node_to_partition(parts)
        else:
            n2p = alg.node_to_partition

        g_for_dag = alg.G_aug if hasattr(alg, 'G_aug') else G

        ctx['algs'][alg_name] = {
            'alg': alg,
            'partitions': parts,
            'result': result,
            'n2p': n2p,
            'G_dag': g_for_dag,
        }

    # Print summary
    print()
    print(_sep())
    print("  InceptionV3 Diagnostic Analysis")
    print(f"  Config: {servers_count}x {server_type}, {bandwidth} Mbps")
    print(_sep())
    print()
    print(f"  {'Algorithm':<14} {'Latency(ms)':>12} {'Partitions':>11}")
    print(f"  {'-'*14} {'-'*12} {'-'*11}")
    for name in ALG_ORDER:
        info = ctx['algs'][name]
        lat = info['result'].latency
        n_parts = len(info['partitions'])
        label = 'Ours(HPA)' if name == 'Ours' else name
        print(f"  {label:<14} {lat:>12.2f} {n_parts:>11}")
    print()

    return ctx


# ── Section 1: Partition Comparison Overview ─────────────────────────

def section_partition_comparison(ctx, output_dir):
    """Section 1: Partition overview table + grouped bar chart."""
    print(_sep())
    print("  SECTION 1: Partition Comparison Overview")
    print(_sep())

    rows = []
    for name in ALG_ORDER:
        info = ctx['algs'][name]
        parts = info['partitions']
        mems = [p.total_memory for p in parts]
        wls = [p.total_workload for p in parts]
        exceed = sum(1 for m in mems if m > EPC_EFFECTIVE_MB)
        rows.append({
            'Metric': name,
            'Partitions': len(parts),
            'Max Mem (MB)': max(mems) if mems else 0,
            'Avg Mem (MB)': np.mean(mems) if mems else 0,
            'Exceed EPC': exceed,
            'Max Workload (ms)': max(wls) if wls else 0,
            'Total Workload (ms)': sum(wls),
        })

    df = pd.DataFrame(rows).set_index('Metric')
    print()
    print(df.to_string(float_format='{:.2f}'.format))
    print()

    # Grouped bar chart
    metrics = ['Partitions', 'Avg Mem (MB)', 'Max Mem (MB)', 'Exceed EPC']
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    x = np.arange(len(ALG_ORDER))

    for ax, metric in zip(axes, metrics):
        vals = [df.loc[name, metric] for name in ALG_ORDER]
        bars = ax.bar(x, vals, color=[COLORS[n] for n in ALG_ORDER], width=0.6, edgecolor='white')
        ax.set_xticks(x)
        ax.set_xticklabels(ALG_ORDER, fontsize=10)
        ax.set_title(metric, fontsize=12, fontweight='bold')
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        # Value labels
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals)*0.02,
                    f'{val:.0f}', ha='center', va='bottom', fontsize=9)

    fig.suptitle('InceptionV3: Partition Comparison', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(output_dir, f'1_partition_comparison.{ext}'),
                    dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {output_dir}/1_partition_comparison.{{png,pdf}}")


# ── Section 2: Per-Algorithm Partition Detail ────────────────────────

def section_partition_detail(ctx, alg_name):
    """Section 2: Detailed partition table for one algorithm."""
    info = ctx['algs'][alg_name]
    parts = info['partitions']
    result = info['result']

    label = 'Ours(HPA)' if alg_name == 'Ours' else alg_name
    print()
    print(_sep("-"))
    print(f"  {label} ({len(parts)} partitions, latency={result.latency:.2f} ms)")
    print(_sep("-"))

    # Sort by partition id
    sorted_parts = sorted(parts, key=lambda p: p.id)

    header = (f"  {'PID':>4}  {'#Lay':>5}  {'Memory(MB)':>10}  {'Weight(MB)':>10}  "
              f"{'Act(MB)':>8}  {'Penalty':>8}  {'Workload(ms)':>12}  {'Module':<20}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    for p in sorted_parts:
        mem = p.total_memory
        weight = p.get_static_memory()
        act = mem - weight
        penalty = calculate_penalty(mem)
        wl = p.total_workload
        # Module: most common module among layers
        modules = [get_module_name(l.name) for l in p.layers]
        if modules:
            from collections import Counter
            module = Counter(modules).most_common(1)[0][0]
        else:
            module = '?'
        flag = " *EPC" if mem > EPC_EFFECTIVE_MB else ""
        print(f"  {p.id:>4}  {len(p.layers):>5}  {mem:>10.2f}  {weight:>10.2f}  "
              f"{act:>8.2f}  {penalty:>7.2f}x  {wl:>12.2f}  {module:<20}{flag}")

    # Summary stats
    exceed = sum(1 for p in parts if p.total_memory > EPC_EFFECTIVE_MB)
    print(f"\n  Summary: {len(parts)} partitions, {exceed} exceed EPC, "
          f"total workload={sum(p.total_workload for p in parts):.2f} ms")


# ── Section 3: Gantt Chart ──────────────────────────────────────────

def section_gantt_chart(ctx, output_dir):
    """Section 3: 2x2 Gantt chart for all 4 algorithms."""
    print()
    print(_sep())
    print("  SECTION 3: Gantt Timeline Chart")
    print(_sep())

    # Find global max latency for shared x-axis
    max_lat = max(ctx['algs'][name]['result'].latency for name in ALG_ORDER)

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    cmap = plt.cm.tab20

    for idx, alg_name in enumerate(ALG_ORDER):
        ax = axes[idx // 2][idx % 2]
        info = ctx['algs'][alg_name]
        result = info['result']
        n_servers = len(ctx['servers'])

        label = 'Ours(HPA)' if alg_name == 'Ours' else alg_name
        ax.set_title(f'{label} (latency={result.latency:.1f} ms)',
                     fontsize=12, fontweight='bold')

        # Draw events
        for sid in range(n_servers):
            events = result.server_schedules.get(sid, [])
            for ev in events:
                start = ev['start']
                duration = ev['end'] - ev['start']
                pid = ev['partition_id']
                color = cmap(pid % 20)
                ax.barh(sid, duration, left=start, height=0.6,
                        color=color, edgecolor='gray', linewidth=0.3, alpha=0.85)

            # Fill idle gaps with light gray
            if events:
                sorted_evs = sorted(events, key=lambda e: e['start'])
                prev_end = 0.0
                for ev in sorted_evs:
                    if ev['start'] > prev_end + 0.1:
                        ax.barh(sid, ev['start'] - prev_end, left=prev_end,
                                height=0.6, color='#f0f0f0', edgecolor='none')
                    prev_end = ev['end']
            else:
                # Entirely idle server
                ax.barh(sid, max_lat, left=0, height=0.6,
                        color='#f0f0f0', edgecolor='none')

        ax.set_xlim(0, max_lat * 1.05)
        ax.set_yticks(range(n_servers))
        ax.set_yticklabels([f'S{i}' for i in range(n_servers)])
        ax.set_xlabel('Time (ms)')
        ax.invert_yaxis()
        ax.grid(axis='x', linestyle='--', alpha=0.4)

    fig.suptitle('InceptionV3: Server Timelines', fontsize=14, fontweight='bold')
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(output_dir, f'3_gantt_chart.{ext}'),
                    dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {output_dir}/3_gantt_chart.{{png,pdf}}")

    # Print text summary
    for alg_name in ALG_ORDER:
        info = ctx['algs'][alg_name]
        result = info['result']
        label = 'Ours(HPA)' if alg_name == 'Ours' else alg_name
        server_counts = {}
        for sid, events in result.server_schedules.items():
            if events:
                server_counts[sid] = len(events)
        usage = "  ".join(f"S{sid}:{cnt}" for sid, cnt in sorted(server_counts.items()))
        n_used = len(server_counts)
        print(f"  {label:<14} servers_used={n_used}  events: {usage}")


# ── Section 4: Latency Breakdown ────────────────────────────────────

def section_latency_breakdown(ctx, output_dir):
    """Section 4: Stacked bar chart of latency components."""
    print()
    print(_sep())
    print("  SECTION 4: Latency Breakdown")
    print(_sep())

    components_data = {}
    for alg_name in ALG_ORDER:
        info = ctx['algs'][alg_name]
        G_dag = info['G_dag']
        comps = compute_latency_components(
            info['result'], info['partitions'], ctx['servers'],
            G_dag, info['n2p'], ctx['bandwidth'], alg_name
        )
        components_data[alg_name] = comps

    # Print table
    print(f"\n  {'Algorithm':<14} {'Compute':>10} {'Paging':>10} {'Comm':>10} {'Idle':>10} {'Total':>10}")
    print(f"  {'-'*14} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for name in ALG_ORDER:
        c = components_data[name]
        total = ctx['algs'][name]['result'].latency
        label = 'Ours(HPA)' if name == 'Ours' else name
        print(f"  {label:<14} {c['compute']:>9.1f}ms {c['paging']:>9.1f}ms "
              f"{c['communication']:>9.1f}ms {c['idle']:>9.1f}ms {total:>9.1f}ms")

    # Stacked bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(ALG_ORDER))
    width = 0.5
    comp_colors = {'compute': '#3498DB', 'paging': '#E74C3C',
                   'communication': '#F39C12', 'idle': '#BDC3C7'}
    comp_labels = {'compute': 'Base Compute', 'paging': 'Paging Penalty',
                   'communication': 'Communication', 'idle': 'Idle/Wait'}

    bottoms = np.zeros(len(ALG_ORDER))
    for comp_key in ['compute', 'paging', 'communication', 'idle']:
        vals = [components_data[name][comp_key] for name in ALG_ORDER]
        ax.bar(x, vals, width, bottom=bottoms, label=comp_labels[comp_key],
               color=comp_colors[comp_key], edgecolor='white', linewidth=0.5)
        bottoms += np.array(vals)

    ax.set_xticks(x)
    labels = ['Ours(HPA)' if n == 'Ours' else n for n in ALG_ORDER]
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel('Latency (ms)', fontsize=12, fontweight='bold')
    ax.set_title('InceptionV3: Latency Breakdown', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    # Add total latency label on top
    for i, name in enumerate(ALG_ORDER):
        total = ctx['algs'][name]['result'].latency
        ax.text(i, bottoms[i] + max(bottoms) * 0.02, f'{total:.0f}ms',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(output_dir, f'4_latency_breakdown.{ext}'),
                    dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"\n  Saved: {output_dir}/4_latency_breakdown.{{png,pdf}}")


# ── Section 5: Partition DAG Visualization ──────────────────────────

def section_partition_dag(ctx, output_dir):
    """Section 5: 2x2 partition-level DAG visualization."""
    print()
    print(_sep())
    print("  SECTION 5: Partition DAG Visualization")
    print(_sep())

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    for idx, alg_name in enumerate(ALG_ORDER):
        ax = axes[idx // 2][idx % 2]
        info = ctx['algs'][alg_name]
        parts = info['partitions']
        G_dag = info['G_dag']
        n2p = info['n2p']
        label = 'Ours(HPA)' if alg_name == 'Ours' else alg_name

        pg = build_partition_dag(parts, G_dag, n2p)

        # For Ours with many partitions, only show top-30 by workload
        show_all = True
        if len(parts) > 50:
            top_pids = sorted([p.id for p in parts],
                              key=lambda pid: next((p.total_workload for p in parts if p.id == pid), 0),
                              reverse=True)[:30]
            pg_sub = pg.subgraph(top_pids).copy()
            ax.set_title(f'{label} (top 30 of {len(parts)} partitions)',
                         fontsize=11, fontweight='bold')
            show_all = False
        else:
            pg_sub = pg
            ax.set_title(f'{label} ({len(parts)} partitions)',
                         fontsize=11, fontweight='bold')

        if len(pg_sub.nodes()) == 0:
            ax.text(0.5, 0.5, 'No partitions', transform=ax.transAxes,
                    ha='center', va='center', fontsize=14)
            ax.axis('off')
            continue

        part_map = {p.id: p for p in parts}

        # Layout
        try:
            pos = nx.nx_agraph.graphviz_layout(pg_sub, prog='dot')
        except Exception:
            try:
                pos = nx.drawing.nx_pydot.graphviz_layout(pg_sub, prog='dot')
            except Exception:
                pos = nx.spring_layout(pg_sub, seed=42, k=2.0)

        # Node properties
        node_sizes = []
        node_colors = []
        node_labels = {}
        for pid in pg_sub.nodes():
            p = part_map.get(pid)
            if p is None:
                node_sizes.append(100)
                node_colors.append('#cccccc')
                node_labels[pid] = f'P{pid}'
                continue
            wl = p.total_workload
            node_sizes.append(max(80, min(1500, wl * 3)))
            if p.total_memory > EPC_EFFECTIVE_MB:
                node_colors.append('#E67E22')  # orange = exceeds EPC
            else:
                node_colors.append('#27AE60')  # green = fits in EPC
            n_layers = len(p.layers)
            mem = p.total_memory
            node_labels[pid] = f'P{pid}\n{n_layers}L\n{mem:.0f}MB'

        nx.draw_networkx_nodes(pg_sub, pos, ax=ax, node_size=node_sizes,
                               node_color=node_colors, alpha=0.85, edgecolors='gray')
        nx.draw_networkx_labels(pg_sub, pos, ax=ax, labels=node_labels,
                                font_size=6, font_weight='bold')
        nx.draw_networkx_edges(pg_sub, pos, ax=ax, arrows=True,
                               arrowsize=8, edge_color='#888888', alpha=0.6,
                               width=0.5, connectionstyle='arc3,rad=0.1')

        # Edge labels for top-N heaviest edges
        edge_weights = {(u, v): d.get('comm_mb', 0.0) for u, v, d in pg_sub.edges(data=True)}
        if edge_weights:
            top_n = min(8, len(edge_weights))
            top_edges = sorted(edge_weights.items(), key=lambda x: -x[1])[:top_n]
            top_labels = {e: f'{w:.2f}MB' for e, w in top_edges if w > 0.001}
            if top_labels:
                nx.draw_networkx_edge_labels(pg_sub, pos, ax=ax,
                                             edge_labels=top_labels, font_size=5)

        ax.axis('off')

    fig.suptitle('InceptionV3: Partition DAGs', fontsize=14, fontweight='bold')
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(output_dir, f'5_partition_dag.{ext}'),
                    dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {output_dir}/5_partition_dag.{{png,pdf}}")

    # Print summary stats for each DAG
    for alg_name in ALG_ORDER:
        info = ctx['algs'][alg_name]
        parts = info['partitions']
        G_dag = info['G_dag']
        n2p = info['n2p']
        pg = build_partition_dag(parts, G_dag, n2p)
        n_edges = pg.number_of_edges()
        max_width = 0
        try:
            if pg.nodes():
                for gen in nx.topological_generations(pg):
                    max_width = max(max_width, len(gen))
        except nx.NetworkXUnfeasible:
            max_width = -1  # cycle detected
        label = 'Ours(HPA)' if alg_name == 'Ours' else alg_name
        cycle_note = "  (has cycles)" if max_width < 0 else ""
        print(f"  {label:<14} nodes={len(pg.nodes()):>4}  edges={n_edges:>4}  max_width={max_width}{cycle_note}")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="InceptionV3 deep diagnostic analysis for 4 algorithms")
    parser.add_argument("--servers", type=int, default=4,
                        help="Number of servers (default: 4)")
    parser.add_argument("--bandwidth", type=float, default=100.0,
                        help="Bandwidth in Mbps (default: 100)")
    parser.add_argument("--server-type", default="Xeon_IceLake",
                        help="Server type (default: Xeon_IceLake)")
    parser.add_argument("--section", type=int, default=None,
                        help="Run only section N (0-5). Default: all sections")
    parser.add_argument("--no-pause", action="store_true",
                        help="Skip interactive pauses between sections")
    parser.add_argument("--output-dir", default="figures/inception_diagnostic",
                        help="Output directory for figures")
    args = parser.parse_args()

    output_dir = args.output_dir
    run_all = args.section is None

    # Section 0: Setup (always runs)
    ctx = setup(args.servers, args.bandwidth, args.server_type)

    if run_all or args.section == 0:
        pass  # setup already printed

    # Section 1
    if run_all or args.section == 1:
        if run_all:
            _pause(args.no_pause, "Section 1: Partition Comparison")
        section_partition_comparison(ctx, output_dir)

    # Section 2
    if run_all or args.section == 2:
        if run_all:
            _pause(args.no_pause, "Section 2: Partition Details")
        print()
        print(_sep())
        print("  SECTION 2: Per-Algorithm Partition Details")
        print(_sep())
        for alg_name in ALG_ORDER:
            section_partition_detail(ctx, alg_name)

    # Section 3
    if run_all or args.section == 3:
        if run_all:
            _pause(args.no_pause, "Section 3: Gantt Chart")
        section_gantt_chart(ctx, output_dir)

    # Section 4
    if run_all or args.section == 4:
        if run_all:
            _pause(args.no_pause, "Section 4: Latency Breakdown")
        section_latency_breakdown(ctx, output_dir)

    # Section 5
    if run_all or args.section == 5:
        if run_all:
            _pause(args.no_pause, "Section 5: Partition DAG")
        section_partition_dag(ctx, output_dir)

    print()
    print(_sep())
    print("  END OF DIAGNOSTIC ANALYSIS")
    print(_sep())


if __name__ == "__main__":
    main()
