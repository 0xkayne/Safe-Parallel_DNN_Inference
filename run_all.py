#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
统一实验运行脚本
================

该脚本一次性完成以下任务：
1. 运行服务器数量消融实验
2. 运行网络带宽消融实验
3. 生成服务器消融实验图表
4. 生成网络带宽消融实验图表

使用方法：
    python run_all_experiments.py

输出：
    - results/server_*.csv, results/network_*.csv (实验数据)
    - server-chart/*.png, server-chart/*.pdf (服务器消融图表)
    - network-chart/*.png, network-chart/*.pdf (网络带宽消融图表)
"""

import os
import sys
import glob
import pandas as pd
import shutil
from datetime import datetime
from PIL import Image
import math
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from algorithms.loader import ModelLoader
from algorithms.common import Server
from algorithms.dina import DINAAlgorithm
from algorithms.media import MEDIAAlgorithm
from algorithms.ours import OursAlgorithm
from algorithms.occ import OCCAlgorithm


# ============ Configuration ============
DATASETS_DIR = 'datasets_260120'
RESULTS_DIR = 'exp_results'
SERVER_CHART_DIR = 'exp_results/exp3_server_ablation'
NETWORK_CHART_DIR = 'exp_results/exp2_network_ablation'
ABLATION_STUDY_DIR = 'exp_results/ablation_runs'
EXP1_COMPARISON_DIR = 'exp_results/exp1_fixed_comparison'

# Experiment 1: fixed comparison config
FIXED_COMPARISON_SERVERS = 4    # 4x homogeneous Xeon_IceLake
FIXED_COMPARISON_BANDWIDTH = 100  # Mbps

# Global variable to store current experiment timestamp folder
CURRENT_EXP_DIR = None

# Server ablation experiment config
SERVER_COUNTS = [1, 2, 3, 4, 5, 6, 7, 8]
BANDWIDTH_FOR_SERVER_EXP = 100  # Mbps
# Default server config for homogeneous tests (all Ice Lake)
DEFAULT_SERVER_TYPE = "Xeon_IceLake"
# Custom Heterogeneous Config for specific experiment
# Custom Heterogeneous Config for specific experiment
HETEROGENEOUS_SERVER_CONFIG = [
    "Celeron G4930", "Celeron G4930",
    "i5-6500", "i5-6500", "i5-6500", "i5-6500",
    "i3-10100",
    "i5-11600"
]
INCREMENTAL_SERVER_ORDER = HETEROGENEOUS_SERVER_CONFIG
# Adjust SERVER_COUNTS to match the length of this list (1 to 8)
INCREMENTAL_COUNTS = list(range(1, len(INCREMENTAL_SERVER_ORDER) + 1))

# Network ablation experiment config
BANDWIDTHS = [0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500]  # Mbps
SERVERS_FOR_NETWORK_EXP = 4


# ============ Model name mapping ============
# Only the 7 models used in experiments
MODEL_NAME_MAP = {
    'albert_large': 'ALBERT-large',
    'bert_large': 'BERT-large',
    'inceptionV3': 'InceptionV3',
    'InceptionV3': 'InceptionV3',
    'resnet50': 'ResNet-50',
    'vgg16': 'VGG-16',
    'vit_large': 'ViT-large',
    'yolov5': 'YOLOv5',
}

# Only run these models (basename without .csv extension)
SELECTED_MODELS = [
    'resnet50', 'vgg16', 'yolov5', 'InceptionV3',
    'bert_large', 'albert_large', 'vit_large',
]


def ensure_dirs():
    """Ensure output directories exist."""
    for d in [RESULTS_DIR, SERVER_CHART_DIR, NETWORK_CHART_DIR,
              'figures/exp2', 'figures/exp3', 'figures/exp1']:
        os.makedirs(d, exist_ok=True)


def create_experiment_folder():
    """Create timestamped folder for current experiment."""
    global CURRENT_EXP_DIR
    
    # Create main ablation study directory
    os.makedirs(ABLATION_STUDY_DIR, exist_ok=True)
    
    # Create timestamped subfolder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    CURRENT_EXP_DIR = os.path.join(ABLATION_STUDY_DIR, timestamp)
    os.makedirs(CURRENT_EXP_DIR, exist_ok=True)
    
    # Create subdirectories
    os.makedirs(os.path.join(CURRENT_EXP_DIR, 'server_charts'), exist_ok=True)
    os.makedirs(os.path.join(CURRENT_EXP_DIR, 'network_charts'), exist_ok=True)
    
    print(f"\n[INFO] Created experiment folder: {CURRENT_EXP_DIR}")
    return CURRENT_EXP_DIR

# old datasets
# def get_model_name(csv_file):
#     """Extract model name from CSV filename."""
#     basename = os.path.basename(csv_file)
#     # Format: "SafeDnnInferenceExp - ModelName.csv"
#     name = basename.replace('SafeDnnInferenceExp - ', '').replace('.csv', '')
#     return name

def get_model_name(csv_file):
    """Extract model name from CSV filename."""
    basename = os.path.basename(csv_file)
    # Format: "ModelName_size_enclave_per_head_layers.csv"
    name = basename.replace('.csv', '')
    return name


def get_selected_csv_files():
    """Return sorted list of CSV files filtered by SELECTED_MODELS."""
    all_files = glob.glob(os.path.join(DATASETS_DIR, '*.csv'))
    filtered = [f for f in all_files if get_model_name(f) in SELECTED_MODELS]
    return sorted(filtered)

def run_server_ablation():
    """Run server heterogeneous experiment with INCREMENTAL addition."""
    print("\n" + "=" * 60)
    print("Running Heterogeneous Server Experiment (Incremental Addition)")
    print("Order: [2x Celeron, 4x i5-5600, 1x i3, 1x i5-11600]")
    print("=" * 60)

    csv_files = get_selected_csv_files()
    
    # Determine output directory: use timestamped folder if exists, otherwise fallback to SERVER_CHART_DIR
    if CURRENT_EXP_DIR:
        output_dir = os.path.join(CURRENT_EXP_DIR, 'server_charts')
    else:
        output_dir = SERVER_CHART_DIR
    
    for csv_file in sorted(csv_files):
        model_name = get_model_name(csv_file)
        short_name = MODEL_NAME_MAP.get(model_name, model_name)
        
        print(f"\nProcessing: {model_name}")
        
        G, layers_map = ModelLoader.load_model_from_csv(csv_file)
        
        print(f"\nModel loaded: {G}")
         
        results = []
         
        # Loop through incremental counts (1 to 8)
        for n_servers in INCREMENTAL_COUNTS:
            # Construct cluster by taking the first n servers from the ordered list
            current_config = INCREMENTAL_SERVER_ORDER[:n_servers]
            
            servers = []
            for i, s_type in enumerate(current_config):
                servers.append(Server(i, server_type=s_type))
            
            # Print current config for verify (only once per model to avoid spam, or verbose)
            if model_name == 'ALBERT': # Print logic for first model only
                 print(f"  n={n_servers}: {[s.server_type for s in servers]}")

            dina = DINAAlgorithm(G, layers_map, servers, BANDWIDTH_FOR_SERVER_EXP)
            media = MEDIAAlgorithm(G, layers_map, servers, BANDWIDTH_FOR_SERVER_EXP)
            ours = OursAlgorithm(G, layers_map, servers, BANDWIDTH_FOR_SERVER_EXP)
            occ = OCCAlgorithm(G, layers_map, servers, BANDWIDTH_FOR_SERVER_EXP)
            
            results.append({
                'Server number': n_servers,
                'OCC': occ.schedule(occ.run()).latency,
                'DINA': dina.schedule(dina.run()).latency,
                'MEDIA': media.schedule(media.run()).latency,
                'Ours': ours.schedule(ours.run()).latency,
            })
        
        # Save to timestamped experiment folder (or fallback to original directory)
        output_file = os.path.join(output_dir, f'server_hetero_incremental_{short_name}.csv')
        df = pd.DataFrame(results)
        df.to_csv(output_file, index=False)
        print(f"  Saved: {output_file}")
        
        # Also save a copy to the original SERVER_CHART_DIR for backward compatibility
        if CURRENT_EXP_DIR:
            legacy_file = os.path.join(SERVER_CHART_DIR, f'server_hetero_incremental_{short_name}.csv')
            df.to_csv(legacy_file, index=False)
    
    print("\n[OK] Heterogeneous server experiment completed!")


def run_network_ablation():
    """Run network bandwidth ablation experiment."""
    print("\n" + "=" * 60)
    print("Running Network Bandwidth Ablation Experiment")
    print("=" * 60)

    csv_files = get_selected_csv_files()
    
    # Determine output directory: use timestamped folder if exists, otherwise fallback to NETWORK_CHART_DIR
    if CURRENT_EXP_DIR:
        output_dir = os.path.join(CURRENT_EXP_DIR, 'network_charts')
    else:
        output_dir = NETWORK_CHART_DIR
    
    for csv_file in sorted(csv_files):
        model_name = get_model_name(csv_file)
        short_name = MODEL_NAME_MAP.get(model_name, model_name)
        
        print(f"\nProcessing: {model_name}")
        
        G, layers_map = ModelLoader.load_model_from_csv(csv_file)
        
        results = []
        for bw in BANDWIDTHS:
            # Homogeneous cluster of Ice Lake (Baseline)
            servers = [Server(i, server_type=DEFAULT_SERVER_TYPE) for i in range(SERVERS_FOR_NETWORK_EXP)]
            
            dina = DINAAlgorithm(G, layers_map, servers, bw)
            media = MEDIAAlgorithm(G, layers_map, servers, bw)
            ours = OursAlgorithm(G, layers_map, servers, bw)
            occ = OCCAlgorithm(G, layers_map, servers, bw)
            
            results.append({
                'Bandwidth(Mbps)': bw,
                'OCC': occ.schedule(occ.run()).latency,
                'DINA': dina.schedule(dina.run()).latency,
                'MEDIA': media.schedule(media.run()).latency,
                'Ours': ours.schedule(ours.run()).latency,
            })
        
        # Save to timestamped experiment folder (or fallback to original directory)
        output_file = os.path.join(output_dir, f'network_{short_name}.csv')
        df = pd.DataFrame(results)
        df.to_csv(output_file, index=False)
        print(f"  Saved: {output_file}")
        
        # Also save a copy to the original NETWORK_CHART_DIR for backward compatibility
        if CURRENT_EXP_DIR:
            legacy_file = os.path.join(NETWORK_CHART_DIR, f'network_{short_name}.csv')
            df.to_csv(legacy_file, index=False)
    
    print("\n[OK] Network ablation experiment completed!")


def create_server_chart(csv_file, model_name, output_dir):
    """Create a professional line chart for server scalability experiment."""
    df = pd.read_csv(csv_file)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    
    colors = {'OCC': '#E74C3C', 'DINA': '#3498DB', 'MEDIA': '#2ECC71', 'Ours': '#9B59B6'}
    markers = {'OCC': 's', 'DINA': '^', 'MEDIA': 'D', 'Ours': 'o'}
    linestyles = {'OCC': '-', 'DINA': (0, (5, 2)), 'MEDIA': (0, (1, 1)), 'Ours': '-'}
    markersizes = {'OCC': 9, 'DINA': 12, 'MEDIA': 8, 'Ours': 10}
    linewidths = {'OCC': 2.5, 'DINA': 3.0, 'MEDIA': 2.0, 'Ours': 2.5}
    
    x = df['Server number']
    for algo in ['OCC', 'DINA', 'MEDIA', 'Ours']:
        ax.plot(x, df[algo], label=algo, color=colors[algo], marker=markers[algo],
                markersize=markersizes[algo], linewidth=linewidths[algo],
                linestyle=linestyles[algo], markeredgecolor='white',
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


def generate_server_charts():
    """Generate server ablation charts."""
    print("\n" + "=" * 60)
    print("Generating Server Ablation Charts")
    print("=" * 60)

    # Data source: always read from the authoritative exp3 directory
    data_dir = SERVER_CHART_DIR
    # Figures output: always goes to figures/exp3/
    plot_dir = 'figures/exp3'
    os.makedirs(plot_dir, exist_ok=True)

    print(f"  Reading data from: {data_dir}")
    print(f"  Saving figures to: {plot_dir}")
    
    models = [
        (f'server_hetero_incremental_{MODEL_NAME_MAP[m]}.csv', MODEL_NAME_MAP[m])
        for m in SELECTED_MODELS
    ]
    
    for csv_file, model_name in models:
        full_path = os.path.join(data_dir, csv_file)
        if os.path.exists(full_path):
            try:
                png_file, pdf_file = create_server_chart(full_path, model_name, plot_dir)
                print(f"  [OK] {model_name}: {os.path.basename(png_file)}, {os.path.basename(pdf_file)}")
            except Exception as e:
                print(f"  [ERROR] {model_name}: {str(e)}")
        else:
            print(f"  [SKIP] {model_name}: File not found - {csv_file}")

    print("  All server charts generated!")


def create_network_chart(csv_file, model_name, output_dir):
    """Create a professional line chart for network bandwidth ablation experiment."""
    import numpy as np
    
    df = pd.read_csv(csv_file)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    
    colors = {'OCC': '#E74C3C', 'DINA': '#3498DB', 'MEDIA': '#2ECC71', 'Ours': '#9B59B6'}
    markers = {'OCC': 's', 'DINA': '^', 'MEDIA': 'D', 'Ours': 'o'}
    linestyles = {'OCC': '-', 'DINA': (0, (5, 2)), 'MEDIA': (0, (1, 1)), 'Ours': '-'}
    markersizes = {'OCC': 9, 'DINA': 12, 'MEDIA': 8, 'Ours': 10}
    linewidths = {'OCC': 2.5, 'DINA': 3.0, 'MEDIA': 2.0, 'Ours': 2.5}
    
    x = df['Bandwidth(Mbps)']
    for algo in ['OCC', 'DINA', 'MEDIA', 'Ours']:
        y_data = df[algo].copy()
        ax.plot(x, y_data, label=algo, color=colors[algo], marker=markers[algo],
                markersize=markersizes[algo], linewidth=linewidths[algo],
                linestyle=linestyles[algo], markeredgecolor='white',
                markeredgewidth=1.5, zorder=3 if algo in ['DINA', 'MEDIA'] else 2)
    
    ax.set_xlabel('Network Bandwidth (Mbps)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Inference Latency (ms)', fontsize=14, fontweight='bold')
    ax.set_title(f'Inference Latency vs. Network Bandwidth ({model_name})', 
                 fontsize=16, fontweight='bold', pad=15)
    
    x_min, x_max = x.min(), x.max()
    if x_max <= 20 and (x_max - x_min) < 20:
        ax.set_xscale('linear')
        ax.set_yscale('log')
        tick_step = 1 if x_max <= 10 else 2
        ticks = np.arange(int(x_min), int(x_max) + 1, tick_step)
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(int(t)) for t in ticks], fontsize=11)
        ax.set_xlim(x_min - 0.5, x_max + 0.5)
    else:
        ax.set_xscale('log')
        ax.set_yscale('log')
        if len(x) <= 12:
            ax.set_xticks(x)
            ax.set_xticklabels([str(int(v)) for v in x], fontsize=11)
    
    ax.tick_params(axis='y', labelsize=12)
    ax.legend(loc='upper right', fontsize=12, frameon=True, 
              fancybox=True, shadow=True, framealpha=0.95)
    ax.grid(True, linestyle='--', alpha=0.7, which='both')
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


def generate_network_charts():
    """Generate network ablation charts."""
    print("\n" + "=" * 60)
    print("Generating Network Ablation Charts")
    print("=" * 60)

    # Data source: always read from the authoritative exp2 directory
    data_dir = NETWORK_CHART_DIR
    # Figures output: always goes to figures/exp2/
    plot_dir = 'figures/exp2'
    os.makedirs(plot_dir, exist_ok=True)

    print(f"  Reading data from: {data_dir}")
    print(f"  Saving figures to: {plot_dir}")

    models = [
        (f'network_{MODEL_NAME_MAP[m]}.csv', MODEL_NAME_MAP[m])
        for m in SELECTED_MODELS
    ]

    for csv_file, model_name in models:
        full_path = os.path.join(data_dir, csv_file)
        if os.path.exists(full_path):
            try:
                png_file, pdf_file = create_network_chart(full_path, model_name, plot_dir)
                print(f"  [OK] {model_name}: {os.path.basename(png_file)}, {os.path.basename(pdf_file)}")
            except Exception as e:
                print(f"  [ERROR] {model_name}: {str(e)}")
        else:
            print(f"  [SKIP] {model_name}: File not found - {csv_file}")

    print("  All network charts generated!")


def combine_charts(image_dir, output_filename, cols=2, pattern="*_chart.png"):
    """
    Combine multiple charts into a single grid image.
    
    Args:
        image_dir: Directory containing chart images
        output_filename: Output file path
        cols: Number of columns in the grid
        pattern: Glob pattern to match image files
    """
    # Find all chart images
    image_paths = sorted(glob.glob(os.path.join(image_dir, pattern)))
    if not image_paths:
        print(f"  [WARNING] No images matching {pattern} found in {image_dir}")
        return False
    
    print(f"  Found {len(image_paths)} images to combine")
    
    # Load all images
    images = [Image.open(f) for f in image_paths]
    
    # Get dimensions (use first image as reference)
    w, h = images[0].size
    
    n_images = len(images)
    rows = math.ceil(n_images / cols)
    
    # Create blank white canvas
    canvas_w = w * cols
    canvas_h = h * rows
    canvas = Image.new('RGB', (canvas_w, canvas_h), 'white')
    
    # Paste images
    for i, img in enumerate(images):
        r = i // cols
        c = i % cols
        
        # Resize if necessary
        if img.size != (w, h):
            img = img.resize((w, h), Image.Resampling.LANCZOS)
        
        canvas.paste(img, (c * w, r * h))
    
    # Save output
    canvas.save(output_filename)
    print(f"  [OK] Saved combined chart: {output_filename}")
    return True


def generate_combined_charts():
    """Generate combined overview charts from figures/exp2 and figures/exp3."""
    print("\n" + "=" * 60)
    print("Generating Combined Charts")
    print("=" * 60)

    server_fig_dir = 'figures/exp3'
    network_fig_dir = 'figures/exp2'

    # Combine server charts
    print("\nCombining server charts...")
    server_combined = 'figures/exp3/combined_server_charts.png'
    combine_charts(server_fig_dir, server_combined, cols=2,
                   pattern="server_hetero_incremental_*_chart.png")

    # Combine network charts
    print("\nCombining network charts...")
    network_combined = 'figures/exp2/combined_network_charts.png'
    combine_charts(network_fig_dir, network_combined, cols=2,
                   pattern="network_*_chart.png")

    print(f"\n[OK] Combined charts generated:")
    print(f"     - Server:  {server_combined}")
    print(f"     - Network: {network_combined}")


def run_fixed_comparison():
    """Run Experiment 1: fixed config comparison (4x Xeon_IceLake, 100 Mbps, all 12 models)."""
    print("\n" + "=" * 60)
    print("Running Experiment 1: Fixed Config Comparison")
    print(f"  Config: {FIXED_COMPARISON_SERVERS}x {DEFAULT_SERVER_TYPE}, {FIXED_COMPARISON_BANDWIDTH} Mbps")
    print("=" * 60)

    os.makedirs(EXP1_COMPARISON_DIR, exist_ok=True)
    csv_files = get_selected_csv_files()

    all_results = []

    for csv_file in csv_files:
        model_name = get_model_name(csv_file)
        short_name = MODEL_NAME_MAP.get(model_name, model_name)

        print(f"\n  Processing: {short_name}")

        G, layers_map = ModelLoader.load_model_from_csv(csv_file)
        servers = [Server(i, server_type=DEFAULT_SERVER_TYPE) for i in range(FIXED_COMPARISON_SERVERS)]

        occ   = OCCAlgorithm(G, layers_map, servers, FIXED_COMPARISON_BANDWIDTH)
        dina  = DINAAlgorithm(G, layers_map, servers, FIXED_COMPARISON_BANDWIDTH)
        media = MEDIAAlgorithm(G, layers_map, servers, FIXED_COMPARISON_BANDWIDTH)
        ours  = OursAlgorithm(G, layers_map, servers, FIXED_COMPARISON_BANDWIDTH)

        row = {
            'Model': short_name,
            'OCC':   occ.schedule(occ.run()).latency,
            'DINA':  dina.schedule(dina.run()).latency,
            'MEDIA': media.schedule(media.run()).latency,
            'Ours':  ours.schedule(ours.run()).latency,
        }
        all_results.append(row)
        print(f"    OCC={row['OCC']:.1f}  DINA={row['DINA']:.1f}  "
              f"MEDIA={row['MEDIA']:.1f}  Ours={row['Ours']:.1f} ms")

    df = pd.DataFrame(all_results)
    output_file = os.path.join(EXP1_COMPARISON_DIR, 'fixed_comparison_all_models.csv')
    df.to_csv(output_file, index=False)
    print(f"\n  [OK] Saved: {output_file}")
    print("\n[OK] Experiment 1 completed!")
    return df


def create_comparison_bar_chart(df, output_dir):
    """Create grouped bar chart for Experiment 1: all models vs all methods."""
    import numpy as np

    plt.style.use('seaborn-v0_8-whitegrid')

    models = df['Model'].tolist()
    methods = ['OCC', 'DINA', 'MEDIA', 'Ours']
    colors   = {'OCC': '#E74C3C', 'DINA': '#3498DB', 'MEDIA': '#2ECC71', 'Ours': '#9B59B6'}
    hatches  = {'OCC': '',        'DINA': '//',       'MEDIA': '\\\\',    'Ours': ''}

    n_models  = len(models)
    n_methods = len(methods)
    x         = np.arange(n_models)
    bar_width = 0.18
    offsets   = np.linspace(-(n_methods - 1) / 2, (n_methods - 1) / 2, n_methods) * bar_width

    fig, ax = plt.subplots(figsize=(max(14, n_models * 1.4), 7), dpi=150)

    for i, method in enumerate(methods):
        values = df[method].tolist()
        ax.bar(x + offsets[i], values, bar_width * 0.92,
               label=method, color=colors[method],
               hatch=hatches[method], edgecolor='white', linewidth=0.5)

    ax.set_xlabel('Model', fontsize=13, fontweight='bold')
    ax.set_ylabel('Inference Latency (ms)', fontsize=13, fontweight='bold')
    ax.set_title(
        f'End-to-End Inference Latency: {FIXED_COMPARISON_SERVERS}×{DEFAULT_SERVER_TYPE}, '
        f'{FIXED_COMPARISON_BANDWIDTH} Mbps',
        fontsize=15, fontweight='bold', pad=15
    )
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=25, ha='right', fontsize=10)
    ax.tick_params(axis='y', labelsize=11)
    ax.legend(fontsize=12, frameon=True, fancybox=True, shadow=True, framealpha=0.95)
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)
    ax.annotate('Lower is better', xy=(0.01, 0.98), xycoords='axes fraction',
                fontsize=10, fontstyle='italic', color='gray', va='top')

    plt.tight_layout()

    output_png = os.path.join(output_dir, 'fixed_comparison_bar_chart.png')
    output_pdf = os.path.join(output_dir, 'fixed_comparison_bar_chart.pdf')
    plt.savefig(output_png, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.savefig(output_pdf, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()

    print(f"  [OK] {output_png}")
    print(f"  [OK] {output_pdf}")
    return output_png, output_pdf


def generate_comparison_charts():
    """Generate Experiment 1 charts from saved CSV data."""
    print("\n" + "=" * 60)
    print("Generating Experiment 1 Charts")
    print("=" * 60)

    plot_dir = 'figures/exp1'
    os.makedirs(plot_dir, exist_ok=True)

    csv_file = os.path.join(EXP1_COMPARISON_DIR, 'fixed_comparison_all_models.csv')
    if not os.path.exists(csv_file):
        print(f"  [SKIP] Data not found: {csv_file}")
        print("         Run run_fixed_comparison() first.")
        return

    df = pd.read_csv(csv_file)
    create_comparison_bar_chart(df, plot_dir)
    print("\n[OK] Experiment 1 charts generated!")


def main():
    """Main entry point."""
    print("=" * 60)
    print("Distributed DNN Inference Simulation - Full Pipeline")
    print("=" * 60)

    # Ensure output directories exist
    ensure_dirs()

    # Exp 1: Fixed config comparison (4x Xeon_IceLake, 100 Mbps)
    run_fixed_comparison()

    # Exp 3: Server count ablation (heterogeneous incremental)
    run_server_ablation()

    # Exp 2: Network bandwidth ablation (homogeneous, variable BW)
    run_network_ablation()

    # Generate charts
    generate_comparison_charts()
    generate_server_charts()
    generate_network_charts()
    generate_combined_charts()

    print("\n" + "=" * 60)
    print("All experiments and charts completed successfully!")
    print("=" * 60)
    print(f"\nResults:")
    print(f"  Exp 1 data:    {EXP1_COMPARISON_DIR}/")
    print(f"  Exp 2 data:    {NETWORK_CHART_DIR}/")
    print(f"  Exp 3 data:    {SERVER_CHART_DIR}/")
    print(f"  Exp 1 figures: figures/exp1/")
    print(f"  Exp 2 figures: figures/exp2/")
    print(f"  Exp 3 figures: figures/exp3/")


if __name__ == "__main__":
    main()

