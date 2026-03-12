#!/usr/bin/env python3
"""
Server Peak Memory Diagnostic
==============================
Runs all 4 algorithms on InceptionV3, collects per-server peak memory,
and generates a 3D bar chart showing:
  1. How many servers each algorithm uses
  2. Peak memory on each server

Usage:
    python diagnostics/server_peak_memory.py
    python diagnostics/server_peak_memory.py --model InceptionV3 --servers 4 --bandwidth 100
"""

import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from loader import ModelLoader
from common import Server, EPC_EFFECTIVE_MB
from alg_occ import OCCAlgorithm
from alg_dina import DINAAlgorithm
from alg_media import MEDIAAlgorithm
from alg_ours import OursAlgorithm

MODEL_FILES = {
    "InceptionV3": "InceptionV3.csv",
    "bert_base": "bert_base.csv",
    "bert_large": "bert_large.csv",
}


def get_server_peak_memory(schedule_result, num_servers, algorithm_name=""):
    """Extract per-server EPC peak memory from a ScheduleResult.

    For OCC: weights are outside EPC, so EPC memory = peak_activation + ring_buffer.
    For all others: EPC memory = partition.total_memory (weights + activations in EPC).
    """
    server_peak = [0.0] * num_servers
    for sid, events in schedule_result.server_schedules.items():
        for event in events:
            part = event['partition']
            if algorithm_name == "OCC":
                # OCC: weights stored in unprotected DRAM, not in EPC
                mem = part._calculate_peak_activation() + OCCAlgorithm.RING_BUFFER_EPC_MB
            else:
                mem = part.total_memory
            server_peak[sid] = max(server_peak[sid], mem)
    return server_peak


def run_all_algorithms(model_name, num_servers, bandwidth):
    """Run 4 algorithms and return {alg_name: [peak_mem_per_server]}."""
    csv_file = MODEL_FILES.get(model_name, f"{model_name}.csv")
    csv_path = os.path.join(_ROOT, "datasets_260120", csv_file)
    G, layers_map = ModelLoader.load_model_from_csv(csv_path)
    servers = [Server(i) for i in range(num_servers)]

    results = {}

    for alg_name, AlgClass in [("OCC", OCCAlgorithm), ("MEDIA", MEDIAAlgorithm),
                                ("DINA", DINAAlgorithm), ("Ours", OursAlgorithm)]:
        alg = AlgClass(G, layers_map, servers, bandwidth)
        parts = alg.run()
        sched = alg.schedule(parts)
        results[alg_name] = get_server_peak_memory(sched, num_servers, alg_name)
        used = sum(1 for m in results[alg_name] if m > 0)
        print(f"  {alg_name:<6} latency={sched.latency:.1f} ms, "
              f"partitions={len(parts)}, servers used={used}")

    return results


def plot_3d_bar(results, num_servers, model_name, output_path):
    """Generate a polished 3D bar chart for per-server peak memory."""
    import matplotlib as mpl
    from matplotlib.patches import FancyBboxPatch
    import matplotlib.colors as mcolors

    mpl.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 10,
    })

    alg_names = ["OCC", "MEDIA", "DINA", "Ours"]
    # Softer, harmonious palette
    base_colors = ["#5B9BD5", "#70AD47", "#ED7D31", "#FFC000"]

    azim = -50
    elev = 25

    fig = plt.figure(figsize=(10, 7.5))
    # computed_zorder=False: we fully control draw order (painter's algorithm)
    ax = fig.add_subplot(111, projection='3d', computed_zorder=False)

    bar_w = 0.55
    bar_d = 0.55

    # ── 1. Draw EPC plane FIRST (always behind everything) ───────
    xx, yy = np.meshgrid(
        np.linspace(-0.3, num_servers + 0.1, 6),
        np.linspace(-0.3, len(alg_names) + 0.1, 6),
    )
    ax.plot_surface(
        xx, yy, np.full_like(xx, EPC_EFFECTIVE_MB),
        alpha=0.10, color='#E74C3C', zorder=0,
    )
    for y_val in [-0.3, len(alg_names) + 0.1]:
        ax.plot(
            [-0.3, num_servers + 0.1], [y_val, y_val],
            [EPC_EFFECTIVE_MB, EPC_EFFECTIVE_MB],
            color='#E74C3C', linewidth=0.8, linestyle='--', alpha=0.45,
            zorder=1,
        )
    for x_val in [-0.3, num_servers + 0.1]:
        ax.plot(
            [x_val, x_val], [-0.3, len(alg_names) + 0.1],
            [EPC_EFFECTIVE_MB, EPC_EFFECTIVE_MB],
            color='#E74C3C', linewidth=0.8, linestyle='--', alpha=0.45,
            zorder=1,
        )

    # ── 2. Collect bars and draw back-to-front ───────────────────
    bars = []
    z_max = 0.0
    for alg_idx, alg_name in enumerate(alg_names):
        peak_mem = results[alg_name]
        for srv_idx in range(num_servers):
            mem = peak_mem[srv_idx]
            if mem > 0:
                z_max = max(z_max, mem)
                bars.append((srv_idx, alg_idx, mem, alg_name))

    # Painter's algorithm: far objects drawn first, near objects drawn last
    # (last-drawn paints over earlier → near covers far).
    # With azim=-50: higher y (Ours) and higher x (S4) are farther from camera.
    cam_rad = np.radians(azim)
    bars.sort(key=lambda b: (b[0] * np.sin(cam_rad) + b[1] * np.cos(cam_rad)),
              reverse=True)

    for draw_i, (srv_idx, alg_idx, mem, alg_name) in enumerate(bars):
        base = mcolors.to_rgb(base_colors[alg_idx])
        dark = tuple(max(0, c - 0.15) for c in base)
        ax.bar3d(
            srv_idx, alg_idx, 0,
            bar_w, bar_d, mem,
            color=base_colors[alg_idx],
            edgecolor=dark,
            linewidth=0.6,
            alpha=0.92,
            zorder=2 + draw_i,
        )

    # ── 3. Text labels LAST (always on top) ──────────────────────
    for srv_idx, alg_idx, mem, alg_name in bars:
        ax.text(
            srv_idx + bar_w / 2, alg_idx + bar_d / 2, mem + 1.5,
            f"{mem:.0f}",
            ha='center', va='bottom', fontsize=7.5,
            fontweight='bold', color='#333333',
            zorder=100,
        )

    # ── Axes configuration ───────────────────────────────────────
    ax.set_xlabel('Server', fontsize=11, labelpad=12)
    ax.set_ylabel('Algorithm', fontsize=11, labelpad=12)
    ax.set_zlabel('Peak Memory (MB)', fontsize=11, labelpad=10)

    ax.set_xticks(np.arange(num_servers) + bar_w / 2)
    ax.set_xticklabels([f"$S_{{{i+1}}}$" for i in range(num_servers)], fontsize=10)
    ax.set_yticks(np.arange(len(alg_names)) + bar_d / 2)
    ax.set_yticklabels(alg_names, fontsize=10, fontweight='medium')

    ax.set_zlim(0, max(z_max * 1.2, EPC_EFFECTIVE_MB * 1.15))
    ax.set_xlim(-0.3, num_servers + 0.1)
    ax.set_ylim(-0.3, len(alg_names) + 0.1)

    ax.tick_params(axis='z', labelsize=9)

    # ── View angle ───────────────────────────────────────────────
    ax.view_init(elev=elev, azim=azim)

    # ── Pane & grid styling ──────────────────────────────────────
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('#CCCCCC')
    ax.yaxis.pane.set_edgecolor('#CCCCCC')
    ax.zaxis.pane.set_edgecolor('#CCCCCC')
    ax.xaxis._axinfo['grid'].update(color='#E0E0E0', linewidth=0.4)
    ax.yaxis._axinfo['grid'].update(color='#E0E0E0', linewidth=0.4)
    ax.zaxis._axinfo['grid'].update(color='#E0E0E0', linewidth=0.4)

    # ── Annotation box: servers used & EPC label ─────────────────
    anno_lines = []
    for alg_idx, alg_name in enumerate(alg_names):
        used = sum(1 for m in results[alg_name] if m > 0)
        max_mem = max(results[alg_name])
        anno_lines.append(f"{alg_name}: {used} srv, max {max_mem:.0f} MB")
    anno_lines.append(f"--- EPC = {EPC_EFFECTIVE_MB:.0f} MB (red plane) ---")
    anno_text = "\n".join(anno_lines)

    # Place as 2D text overlay
    fig.text(
        0.02, 0.96, anno_text,
        fontsize=8.5, fontfamily='monospace',
        verticalalignment='top',
        bbox=dict(
            boxstyle='round,pad=0.5',
            facecolor='white', edgecolor='#AAAAAA',
            alpha=0.9, linewidth=0.6,
        ),
    )

    # ── Title ────────────────────────────────────────────────────
    ax.set_title(
        f"Per-Server Peak EPC Memory — {model_name}",
        fontsize=13, fontweight='bold', pad=18, loc='center',
    )

    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.90)
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"\n  [OK] Saved: {output_path}")
    print(f"  [OK] Saved: {output_path.replace('.png', '.pdf')}")


def print_summary_table(results, num_servers):
    """Print a text summary table."""
    print(f"\n{'Algorithm':<10}", end="")
    for i in range(num_servers):
        print(f"{'Server '+str(i+1):>12}", end="")
    print(f"{'Servers Used':>14}  {'Max (MB)':>10}  {'Exceeds EPC':>12}")
    print("-" * (10 + 12 * num_servers + 14 + 10 + 12 + 6))

    for alg_name in ["OCC", "MEDIA", "DINA", "Ours"]:
        peak_mem = results[alg_name]
        used = sum(1 for m in peak_mem if m > 0)
        max_mem = max(peak_mem)
        exceeds = sum(1 for m in peak_mem if m > EPC_EFFECTIVE_MB)
        print(f"{alg_name:<10}", end="")
        for m in peak_mem:
            marker = " *" if m > EPC_EFFECTIVE_MB else ""
            print(f"{m:>10.1f}{marker}", end="")
        print(f"{used:>14}  {max_mem:>10.1f}  {exceeds:>12}")

    print(f"\n  * = exceeds EPC ({EPC_EFFECTIVE_MB:.0f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Server peak memory diagnostic")
    parser.add_argument("--model", default="InceptionV3", help="Model name")
    parser.add_argument("--servers", type=int, default=4, help="Number of servers")
    parser.add_argument("--bandwidth", type=float, default=100.0, help="Bandwidth (Mbps)")
    args = parser.parse_args()

    print(f"=== Server Peak Memory: {args.model} ({args.servers} servers, {args.bandwidth} Mbps) ===\n")

    results = run_all_algorithms(args.model, args.servers, args.bandwidth)
    print_summary_table(results, args.servers)

    out_dir = os.path.join(_ROOT, "diagnostics", "peak_memory_output")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"server_peak_mem_{args.model}.png")
    plot_3d_bar(results, args.servers, args.model, out_path)


if __name__ == "__main__":
    main()
