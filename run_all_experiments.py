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

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from loader import ModelLoader
from common import Server
from alg_dina import DINAAlgorithm
from alg_media import MEDIAAlgorithm
from alg_ours import OursAlgorithm
from alg_occ import OCCAlgorithm


# ============ Configuration ============
DATASETS_DIR = 'datasets'
RESULTS_DIR = 'results'
SERVER_CHART_DIR = 'server-chart'
NETWORK_CHART_DIR = 'network-chart'

# Server ablation experiment config
SERVER_COUNTS = [1, 2, 4, 8, 12, 16]
BANDWIDTH_FOR_SERVER_EXP = 100  # Mbps

# Network ablation experiment config
BANDWIDTHS = [1, 10, 50, 100, 500, 1000]  # Mbps
SERVERS_FOR_NETWORK_EXP = 4


# ============ Model name mapping ============
MODEL_NAME_MAP = {
    'ALBERT': 'ALBERT',
    'BERT-base': 'BERT',
    'DistillBERT': 'DistillBERT',
    'InceptionV3': 'inceptionV3',
    'TinyBERT-4l': 'TinyBERT-4l',
    'TinyBERT-6l': 'TinyBERT-6l',
    'ViT-base': 'ViT',
}


def ensure_dirs():
    """Ensure output directories exist."""
    for d in [RESULTS_DIR, SERVER_CHART_DIR, NETWORK_CHART_DIR]:
        os.makedirs(d, exist_ok=True)


def get_model_name(csv_file):
    """Extract model name from CSV filename."""
    basename = os.path.basename(csv_file)
    # Format: "SafeDnnInferenceExp - ModelName.csv"
    name = basename.replace('SafeDnnInferenceExp - ', '').replace('.csv', '')
    return name


def run_server_ablation():
    """Run server count ablation experiment."""
    print("\n" + "=" * 60)
    print("Running Server Count Ablation Experiment")
    print("=" * 60)
    
    csv_files = glob.glob(os.path.join(DATASETS_DIR, '*.csv'))
    
    for csv_file in sorted(csv_files):
        model_name = get_model_name(csv_file)
        short_name = MODEL_NAME_MAP.get(model_name, model_name)
        
        print(f"\nProcessing: {model_name}")
        
        G, layers_map = ModelLoader.load_model_from_csv(csv_file)
        
        results = []
        for n_servers in SERVER_COUNTS:
            servers = [Server(i, 1.0) for i in range(n_servers)]
            
            dina = DINAAlgorithm(G, layers_map, servers, BANDWIDTH_FOR_SERVER_EXP)
            media = MEDIAAlgorithm(G, layers_map, servers, BANDWIDTH_FOR_SERVER_EXP)
            ours = OursAlgorithm(G, layers_map, servers, BANDWIDTH_FOR_SERVER_EXP)
            occ = OCCAlgorithm(G, layers_map, servers, BANDWIDTH_FOR_SERVER_EXP)
            
            results.append({
                'Server number': n_servers,
                'OCC': occ.schedule(occ.run()),
                'DINA': dina.schedule(dina.run()),
                'MEIDA': media.schedule(media.run()),  # Note: MEIDA (typo preserved for chart compatibility)
                'Ours': ours.schedule(ours.run()),
            })
        
        # Save to server-chart directory for plotting
        output_file = os.path.join(SERVER_CHART_DIR, f'server_{short_name}.csv')
        df = pd.DataFrame(results)
        df.to_csv(output_file, index=False)
        print(f"  Saved: {output_file}")
    
    print("\n[OK] Server ablation experiment completed!")


def run_network_ablation():
    """Run network bandwidth ablation experiment."""
    print("\n" + "=" * 60)
    print("Running Network Bandwidth Ablation Experiment")
    print("=" * 60)
    
    csv_files = glob.glob(os.path.join(DATASETS_DIR, '*.csv'))
    
    for csv_file in sorted(csv_files):
        model_name = get_model_name(csv_file)
        short_name = MODEL_NAME_MAP.get(model_name, model_name)
        
        print(f"\nProcessing: {model_name}")
        
        G, layers_map = ModelLoader.load_model_from_csv(csv_file)
        
        results = []
        for bw in BANDWIDTHS:
            servers = [Server(i, 1.0) for i in range(SERVERS_FOR_NETWORK_EXP)]
            
            dina = DINAAlgorithm(G, layers_map, servers, bw)
            media = MEDIAAlgorithm(G, layers_map, servers, bw)
            ours = OursAlgorithm(G, layers_map, servers, bw)
            occ = OCCAlgorithm(G, layers_map, servers, bw)
            
            results.append({
                'Bandwidth(Mbps)': bw,
                'OCC': occ.schedule(occ.run()),
                'DINA': dina.schedule(dina.run()),
                'MEIDA': media.schedule(media.run()),  # Note: MEIDA (typo preserved for chart compatibility)
                'Ours': ours.schedule(ours.run()),
            })
        
        # Save to network-chart directory for plotting
        output_file = os.path.join(NETWORK_CHART_DIR, f'network_{short_name}.csv')
        df = pd.DataFrame(results)
        df.to_csv(output_file, index=False)
        print(f"  Saved: {output_file}")
    
    print("\n[OK] Network ablation experiment completed!")


def generate_server_charts():
    """Generate server ablation charts."""
    print("\n" + "=" * 60)
    print("Generating Server Ablation Charts")
    print("=" * 60)
    
    import subprocess
    result = subprocess.run(
        [sys.executable, 'plot_all_server_charts.py'],
        cwd=SERVER_CHART_DIR,
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print(result.stdout)
    else:
        print(f"Error: {result.stderr}")


def generate_network_charts():
    """Generate network ablation charts."""
    print("\n" + "=" * 60)
    print("Generating Network Ablation Charts")
    print("=" * 60)
    
    import subprocess
    result = subprocess.run(
        [sys.executable, 'plot_all_network_charts.py'],
        cwd=NETWORK_CHART_DIR,
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print(result.stdout)
    else:
        print(f"Error: {result.stderr}")


def main():
    """Main entry point."""
    print("=" * 60)
    print("Distributed DNN Inference Simulation - Full Pipeline")
    print("=" * 60)
    
    # Ensure directories exist
    ensure_dirs()
    
    # Step 1: Run experiments
    run_server_ablation()
    run_network_ablation()
    
    # Step 2: Generate charts
    generate_server_charts()
    generate_network_charts()
    
    print("\n" + "=" * 60)
    print("All experiments and charts completed successfully!")
    print("=" * 60)
    print("\nOutput files:")
    print(f"  - Server CSV: {SERVER_CHART_DIR}/server_*.csv")
    print(f"  - Server Charts: {SERVER_CHART_DIR}/*.png, *.pdf")
    print(f"  - Network CSV: {NETWORK_CHART_DIR}/network_*.csv")
    print(f"  - Network Charts: {NETWORK_CHART_DIR}/*.png, *.pdf")


if __name__ == "__main__":
    main()

