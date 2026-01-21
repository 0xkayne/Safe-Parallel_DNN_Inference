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

from loader import ModelLoader
from common import Server
from alg_dina import DINAAlgorithm
from alg_media import MEDIAAlgorithm
from alg_ours import OursAlgorithm
from alg_occ import OCCAlgorithm


# ============ Configuration ============
DATASETS_DIR = 'datasets_260120'
RESULTS_DIR = 'results_260120'
SERVER_CHART_DIR = 'server-chart_260120'
NETWORK_CHART_DIR = 'network-chart_260120'
ABLATION_STUDY_DIR = 'ablation_study_chart_260120'

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
BANDWIDTHS = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]  # Mbps
SERVERS_FOR_NETWORK_EXP = 4


# ============ Model name mapping ============
MODEL_NAME_MAP = {
    'albert_base': 'ALBERT-base',
    'albert_large': 'ALBERT-large',
    'bert_base': 'BERT-base',
    'bert_large': 'BERT-large',
    'distilbert_base': 'DistillBERT-base',
    'distilbert_large': 'DistillBERT-large',
    'inceptionV3': 'inceptionV3',
    'tinybert_4l': 'TinyBERT-4l',
    'tinybert_6l': 'TinyBERT-6l',
    'vit_base': 'ViT-base',
    'vit_large': 'ViT-large',
    'vit_small': 'ViT-small',
    'vit_tiny': 'ViT-tiny',
}


def ensure_dirs():
    """Ensure output directories exist."""
    for d in [RESULTS_DIR, SERVER_CHART_DIR, NETWORK_CHART_DIR]:
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

def run_server_ablation():
    """Run server heterogeneous experiment with INCREMENTAL addition."""
    print("\n" + "=" * 60)
    print("Running Heterogeneous Server Experiment (Incremental Addition)")
    print("Order: [2x Celeron, 4x i5-5600, 1x i3, 1x i5-11600]")
    print("=" * 60)
    
    csv_files = glob.glob(os.path.join(DATASETS_DIR, '*.csv'))
    
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
                'MEIDA': media.schedule(media.run()).latency,
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
    
    csv_files = glob.glob(os.path.join(DATASETS_DIR, '*.csv'))
    
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
                'MEIDA': media.schedule(media.run()).latency,
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
    
    colors = {'OCC': '#E74C3C', 'DINA': '#3498DB', 'MEIDA': '#2ECC71', 'Ours': '#9B59B6'}
    markers = {'OCC': 's', 'DINA': '^', 'MEIDA': 'D', 'Ours': 'o'}
    linestyles = {'OCC': '-', 'DINA': (0, (5, 2)), 'MEIDA': (0, (1, 1)), 'Ours': '-'}
    markersizes = {'OCC': 9, 'DINA': 12, 'MEIDA': 8, 'Ours': 10}
    linewidths = {'OCC': 2.5, 'DINA': 3.0, 'MEIDA': 2.0, 'Ours': 2.5}
    
    x = df['Server number']
    for algo in ['OCC', 'DINA', 'MEIDA', 'Ours']:
        ax.plot(x, df[algo], label=algo, color=colors[algo], marker=markers[algo],
                markersize=markersizes[algo], linewidth=linewidths[algo],
                linestyle=linestyles[algo], markeredgecolor='white',
                markeredgewidth=1.5, zorder=3 if algo in ['DINA', 'MEIDA'] else 2)
    
    ax.set_xlabel('Number of Servers', fontsize=14, fontweight='bold')
    ax.set_ylabel('Inference Latency (ms)', fontsize=14, fontweight='bold')
    ax.set_title(f'Inference Latency vs. Number of Servers ({model_name})', 
                 fontsize=16, fontweight='bold', pad=15)
    
    ax.set_xticks(x)
    ax.set_xticklabels(x, fontsize=12)
    ax.tick_params(axis='y', labelsize=12)
    
    y_min = df[['OCC', 'DINA', 'MEIDA', 'Ours']].min().min() * 0.90
    y_max = df[['OCC', 'DINA', 'MEIDA', 'Ours']].max().max() * 1.08
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
    
    if CURRENT_EXP_DIR:
        plot_dir = os.path.join(CURRENT_EXP_DIR, 'server_charts')
    else:
        plot_dir = SERVER_CHART_DIR
    
    print(f"  Using data from: {plot_dir}")
    
    models = [
        ('server_hetero_incremental_DistillBERT-base.csv', 'DistillBERT-base'),
        ('server_hetero_incremental_ALBERT-base.csv', 'ALBERT-base'),
        ('server_hetero_incremental_ALBERT-large.csv', 'ALBERT-large'),
        ('server_hetero_incremental_BERT-base.csv', 'BERT-base'),
        ('server_hetero_incremental_BERT-large.csv', 'BERT-large'),
        ('server_hetero_incremental_TinyBERT-4l.csv', 'TinyBERT-4l'),
        ('server_hetero_incremental_TinyBERT-6l.csv', 'TinyBERT-6l'),
        ('server_hetero_incremental_ViT-tiny.csv', 'ViT-tiny'),
        ('server_hetero_incremental_ViT-small.csv', 'ViT-small'),
        ('server_hetero_incremental_ViT-base.csv', 'ViT-base'),
        ('server_hetero_incremental_ViT-large.csv', 'ViT-large'),
        ('server_hetero_incremental_inceptionV3.csv', 'InceptionV3'),
    ]
    
    for csv_file, model_name in models:
        full_path = os.path.join(plot_dir, csv_file)
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
    
    colors = {'OCC': '#E74C3C', 'DINA': '#3498DB', 'MEIDA': '#2ECC71', 'Ours': '#9B59B6'}
    markers = {'OCC': 's', 'DINA': '^', 'MEIDA': 'D', 'Ours': 'o'}
    linestyles = {'OCC': '-', 'DINA': (0, (5, 2)), 'MEIDA': (0, (1, 1)), 'Ours': '-'}
    markersizes = {'OCC': 9, 'DINA': 12, 'MEIDA': 8, 'Ours': 10}
    linewidths = {'OCC': 2.5, 'DINA': 3.0, 'MEIDA': 2.0, 'Ours': 2.5}
    
    x = df['Bandwidth(Mbps)']
    for algo in ['OCC', 'DINA', 'MEIDA', 'Ours']:
        y_data = df[algo].copy()
        ax.plot(x, y_data, label=algo, color=colors[algo], marker=markers[algo],
                markersize=markersizes[algo], linewidth=linewidths[algo],
                linestyle=linestyles[algo], markeredgecolor='white',
                markeredgewidth=1.5, zorder=3 if algo in ['DINA', 'MEIDA'] else 2)
    
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
    
    if CURRENT_EXP_DIR:
        plot_dir = os.path.join(CURRENT_EXP_DIR, 'network_charts')
    else:
        plot_dir = NETWORK_CHART_DIR
    
    print(f"  Using data from: {plot_dir}")
    
    models = [
        ('network_DistillBERT-base.csv', 'DistillBERT-base'),
        ('network_ALBERT-base.csv', 'ALBERT-base'),
        ('network_ALBERT-large.csv', 'ALBERT-large'),
        ('network_BERT-base.csv', 'BERT-base'),
        ('network_BERT-large.csv', 'BERT-large'),
        ('network_TinyBERT-4l.csv', 'TinyBERT-4l'),
        ('network_TinyBERT-6l.csv', 'TinyBERT-6l'),
        ('network_ViT-tiny.csv', 'ViT-tiny'),
        ('network_ViT-small.csv', 'ViT-small'),
        ('network_ViT-base.csv', 'ViT-base'),
        ('network_ViT-large.csv', 'ViT-large'),
        ('network_inceptionV3.csv', 'InceptionV3'),
    ]
    
    for csv_file, model_name in models:
        full_path = os.path.join(plot_dir, csv_file)
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
    """Generate combined charts in the timestamped experiment folder."""
    if CURRENT_EXP_DIR is None:
        print("[ERROR] Experiment folder not created!")
        return
    
    print("\n" + "=" * 60)
    print("Generating Combined Charts")
    print("=" * 60)
    
    server_dest = os.path.join(CURRENT_EXP_DIR, 'server_charts')
    network_dest = os.path.join(CURRENT_EXP_DIR, 'network_charts')
    
    # Combine server charts
    print("\nCombining server charts...")
    server_combined = os.path.join(CURRENT_EXP_DIR, 'combined_server_charts.png')
    combine_charts(server_dest, server_combined, cols=2, 
                  pattern="server_hetero_incremental_*_chart.png")
    
    # Combine network charts
    print("\nCombining network charts...")
    network_combined = os.path.join(CURRENT_EXP_DIR, 'combined_network_charts.png')
    combine_charts(network_dest, network_combined, cols=2, 
                  pattern="network_*_chart.png")
    
    # Create summary file
    summary_file = os.path.join(CURRENT_EXP_DIR, 'experiment_info.txt')
    
    # Count charts
    server_pngs = glob.glob(os.path.join(server_dest, '*_chart.png'))
    network_pngs = glob.glob(os.path.join(network_dest, '*_chart.png'))
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("Ablation Study Experiment Results\n")
        f.write("=" * 60 + "\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Server Charts: {len(server_pngs)}\n")
        f.write(f"Network Charts: {len(network_pngs)}\n")
        f.write("\nConfiguration:\n")
        f.write(f"  Server counts: {INCREMENTAL_COUNTS}\n")
        f.write(f"  Network bandwidths: {BANDWIDTHS} Mbps\n")
        f.write(f"  Server config: {HETEROGENEOUS_SERVER_CONFIG}\n")
    
    print(f"\n[OK] All combined charts generated in: {CURRENT_EXP_DIR}")
    print(f"     - Server charts: {server_dest}/")
    print(f"     - Network charts: {network_dest}/")
    print(f"     - Combined server: combined_server_charts.png")
    print(f"     - Combined network: combined_network_charts.png")


def main():
    """Main entry point."""
    print("=" * 60)
    print("Distributed DNN Inference Simulation - Full Pipeline")
    print("=" * 60)
    
    # Ensure directories exist
    ensure_dirs()
    
    # Create timestamped experiment folder (all data will be saved here)
    exp_dir = create_experiment_folder()
    
    # Step 1: Run experiments (data saved to timestamped folder)
    run_server_ablation()
    run_network_ablation()
    
    # Step 2: Generate individual charts (using data from timestamped folder)
    generate_server_charts()
    generate_network_charts()
    
    # Step 3: Generate combined charts
    generate_combined_charts()
    
    print("\n" + "=" * 60)
    print("All experiments and charts completed successfully!")
    print("=" * 60)
    print(f"\nExperiment results saved to: {exp_dir}")
    print(f"  - Server data & charts: {os.path.join(exp_dir, 'server_charts')}/")
    print(f"  - Network data & charts: {os.path.join(exp_dir, 'network_charts')}/")
    print(f"  - Combined charts: combined_server_charts.png, combined_network_charts.png")
    print(f"\nBackward compatibility copies:")
    print(f"  - Server CSV: {SERVER_CHART_DIR}/server_*.csv")
    print(f"  - Network CSV: {NETWORK_CHART_DIR}/network_*.csv")


if __name__ == "__main__":
    main()

