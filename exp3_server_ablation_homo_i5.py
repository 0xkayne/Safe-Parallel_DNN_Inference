#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
同构服务器消融实验 (8× i5-6500)
================================

在 1-8 台同构 i5-6500 服务器环境下测试 Ours 方法的扩展性，
对比异构服务器配置下的结果。

模型: InceptionV3, VGG-16, YOLOv5, ALBERT-large
配置: 1-8 台 i5-6500, 100 Mbps 带宽
算法: OCC, DINA, MEDIA, Ours
"""

import os
import sys
import glob
import math
import pandas as pd
from PIL import Image
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from loader import ModelLoader
from common import Server
from alg_occ import OCCAlgorithm
from alg_dina import DINAAlgorithm
from alg_media import MEDIAAlgorithm
from alg_ours import OursAlgorithm

# =============================================================================
# Experiment Configuration
# =============================================================================
SERVER_COUNTS = [1, 2, 3, 4, 5, 6, 7, 8]
SERVER_TYPE = "i5-6500"
BANDWIDTH = 100  # Mbps

MODELS = [
    ("InceptionV3", "InceptionV3"),
    ("VGG-16", "vgg16"),
    ("YOLOv5", "yolov5"),
    ("ALBERT-large", "albert_large"),
]

OUTPUT_CSV_DIR = "exp_results/exp3_server_ablation_homo_i5"
OUTPUT_FIG_DIR = "figures/exp3_homo_i5"

# =============================================================================
# Chart Style (matching run_all_experiments.py)
# =============================================================================
COLORS = {'OCC': '#E74C3C', 'DINA': '#3498DB', 'MEDIA': '#2ECC71', 'Ours': '#9B59B6'}
MARKERS = {'OCC': 's', 'DINA': '^', 'MEDIA': 'D', 'Ours': 'o'}
LINESTYLES = {'OCC': '-', 'DINA': (0, (5, 2)), 'MEDIA': (0, (1, 1)), 'Ours': '-'}
MARKERSIZES = {'OCC': 9, 'DINA': 12, 'MEDIA': 8, 'Ours': 10}
LINEWIDTHS = {'OCC': 2.5, 'DINA': 3.0, 'MEDIA': 2.0, 'Ours': 2.5}


# =============================================================================
# Chart Generation Functions
# =============================================================================
def create_server_chart(csv_file, model_name, output_dir):
    """Create a professional line chart for server scalability experiment."""
    df = pd.read_csv(csv_file)

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    x = df['Server number']
    for algo in ['OCC', 'DINA', 'MEDIA', 'Ours']:
        ax.plot(x, df[algo], label=algo, color=COLORS[algo], marker=MARKERS[algo],
                markersize=MARKERSIZES[algo], linewidth=LINEWIDTHS[algo],
                linestyle=LINESTYLES[algo], markeredgecolor='white',
                markeredgewidth=1.5, zorder=3 if algo in ['DINA', 'MEDIA'] else 2)

    ax.set_xlabel('Number of Servers', fontsize=14, fontweight='bold')
    ax.set_ylabel('Inference Latency (ms)', fontsize=14, fontweight='bold')
    ax.set_title(f'Inference Latency vs. Number of Servers ({model_name})',
                 fontsize=16, fontweight='bold', pad=15)

    ax.set_xticks(x)
    ax.set_xticklabels(x, fontsize=12)
    ax.tick_params(axis='y', labelsize=12)

    y_min = df[['OCC', 'DINA', 'MEDIA', 'Ours']].min().min() * 0.90
    y_max = df[['OCC', 'DINA', 'MEDIA', 'Ours']].max().max() * 1.08
    ax.set_ylim(y_min, y_max)

    ax.legend(loc='upper right', fontsize=12, frameon=True,
              fancybox=True, shadow=True, framealpha=0.95)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.annotate('Lower is better', xy=(0.02, 0.02), xycoords='axes fraction',
                fontsize=10, fontstyle='italic', color='gray')

    plt.tight_layout()

    base_name = os.path.splitext(os.path.basename(csv_file))[0]
    output_png = os.path.join(output_dir, f"{base_name}_chart.png")
    output_pdf = os.path.join(output_dir, f"{base_name}_chart.pdf")

    plt.savefig(output_png, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.savefig(output_pdf, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()

    return output_png, output_pdf


def combine_charts(image_dir, output_filename, cols=2, pattern="*_chart.png"):
    """Combine multiple charts into a single grid image."""
    image_paths = sorted(glob.glob(os.path.join(image_dir, pattern)))
    if not image_paths:
        print(f"  [WARNING] No images matching {pattern} found in {image_dir}")
        return False

    print(f"  Found {len(image_paths)} images to combine")

    images = [Image.open(f) for f in image_paths]
    w, h = images[0].size

    n_images = len(image_paths)
    rows = math.ceil(n_images / cols)

    canvas_w = w * cols
    canvas_h = h * rows
    canvas = Image.new('RGB', (canvas_w, canvas_h), 'white')

    for i, img in enumerate(images):
        r = i // cols
        c = i % cols

        if img.size != (w, h):
            img = img.resize((w, h), Image.Resampling.LANCZOS)

        canvas.paste(img, (c * w, r * h))

    canvas.save(output_filename)
    print(f"  [OK] Saved combined chart: {output_filename}")
    return True


# =============================================================================
# Main Experiment
# =============================================================================
def run_experiment():
    print("\n" + "=" * 60)
    print("Homogeneous Server Ablation Experiment (8× i5-6500)")
    print("=" * 60)

    os.makedirs(OUTPUT_CSV_DIR, exist_ok=True)
    os.makedirs(OUTPUT_FIG_DIR, exist_ok=True)

    csv_files = []

    for model_display, model_file in MODELS:
        csv_path = f"datasets_260120/{model_file}.csv"
        print(f"\n[{model_display}]")
        print(f"  Loading: {csv_path}")

        G, layers_map = ModelLoader.load_model_from_csv(csv_path)
        print(f"  DAG: {G}")

        results = []

        for n_servers in SERVER_COUNTS:
            servers = [Server(i, server_type=SERVER_TYPE) for i in range(n_servers)]
            print(f"  n={n_servers}: {[s.server_type for s in servers]}")

            occ = OCCAlgorithm(G, layers_map, servers, BANDWIDTH)
            dina = DINAAlgorithm(G, layers_map, servers, BANDWIDTH)
            media = MEDIAAlgorithm(G, layers_map, servers, BANDWIDTH)
            ours = OursAlgorithm(G, layers_map, servers, BANDWIDTH)

            results.append({
                'Server number': n_servers,
                'OCC': occ.schedule(occ.run()).latency,
                'DINA': dina.schedule(dina.run()).latency,
                'MEDIA': media.schedule(media.run()).latency,
                'Ours': ours.schedule(ours.run()).latency,
            })

        # Save CSV
        csv_filename = f"server_homo_i5_{model_display}.csv"
        csv_fullpath = os.path.join(OUTPUT_CSV_DIR, csv_filename)
        df = pd.DataFrame(results)
        df.to_csv(csv_fullpath, index=False)
        print(f"  [OK] Saved CSV: {csv_fullpath}")
        csv_files.append((csv_fullpath, model_display))

    # =============================================================================
    # Generate Charts
    # =============================================================================
    print("\n" + "=" * 60)
    print("Generating Charts")
    print("=" * 60)

    for csv_file, model_name in csv_files:
        try:
            png_file, pdf_file = create_server_chart(csv_file, model_name, OUTPUT_FIG_DIR)
            print(f"  [OK] {model_name}: {os.path.basename(png_file)}")
        except Exception as e:
            print(f"  [ERROR] {model_name}: {str(e)}")

    # Combined chart
    print("\nGenerating combined chart...")
    combined_path = os.path.join(OUTPUT_FIG_DIR, "server_ablation_combined.png")
    combine_charts(OUTPUT_FIG_DIR, combined_path, cols=2,
                   pattern="server_homo_i5_*_chart.png")

    print("\n" + "=" * 60)
    print("[OK] Experiment completed!")
    print(f"  CSV: {OUTPUT_CSV_DIR}/")
    print(f"  Figures: {OUTPUT_FIG_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    run_experiment()
