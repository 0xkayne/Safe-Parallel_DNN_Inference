
import os
import sys
import math
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

from common import Server, EPC_EFFECTIVE_MB, calculate_penalty, PAGE_SIZE_KB, PAGE_FAULT_OVERHEAD_MS, ENCLAVE_ENTRY_EXIT_OVERHEAD_MS, DEFAULT_PAGING_BW_MBPS
from loader import ModelLoader
from alg_dina import DINAAlgorithm
from alg_media import MEDIAAlgorithm
from alg_ours import OursAlgorithm
from alg_occ import OCCAlgorithm

# Configuration
SERVER_TYPE_LIST = ["Celeron_G4930", "i5-6500", "i5-11600", "i5-11600"]
BANDWIDTH = 100  # Mbps
NUM_SERVERS = 4
DATASET_DIR = 'datasets_260120'
CHART_DIR = 'network-chart_260120'

def run_model_comparison(dataset_path):
    """
    Runs all 4 algorithms on a specific dataset and returns the latencies.
    """
    try:
        G, layers_map = ModelLoader.load_model_from_csv(dataset_path)
    except Exception as e:
        print(f"Error loading {dataset_path}: {e}")
        return None

    servers = [Server(i, server_type=SERVER_TYPE_LIST[i]) for i in range(NUM_SERVERS)]
    
    algorithms = [
        ("MEDIA", MEDIAAlgorithm),
        ("OCC", OCCAlgorithm),
        ("DINA", DINAAlgorithm),
        ("Ours", OursAlgorithm)
    ]
    
    results = {}
    for name, AlgoClass in algorithms:
        try:
            algo = AlgoClass(G, layers_map, servers, BANDWIDTH)
            partitions = algo.run()
            schedule_res = algo.schedule(partitions)
            results[name] = schedule_res.latency
        except Exception as e:
            print(f"  [Error] {name} failed on {dataset_path}: {str(e)}")
            results[name] = 0.0
            
    return results

def main():
    # 1. Identify models from CHART_DIR filenames
    chart_files = [f for f in os.listdir(CHART_DIR) if f.endswith('.csv')]
    # Map network_Name.csv to Name.csv (lowercase for datasets_260120)
    model_configs = []
    for f in chart_files:
        model_id = f.replace('network_', '').replace('.csv', '')
        # Special handling for case mapping and spellings
        normalized = model_id.lower().replace('-', '_')
        possible_patterns = [
            normalized,
            normalized.replace('distillbert', 'distilbert'),
            model_id + '.csv',
            model_id.replace('-', '_') + '.csv'
        ]
        
        found = False
        for p in possible_patterns:
            dataset_name = p if p.endswith('.csv') else p + '.csv'
            p_path = os.path.join(DATASET_DIR, dataset_name)
            if os.path.exists(p_path):
                model_configs.append((model_id, p_path))
                found = True
                break
        if not found:
            print(f"Warning: Could not find dataset for {model_id} in {DATASET_DIR}")

    # Limit to 12 models as per directory or 6 as per image? 
    # The user said "all models".
    all_results = {}
    for i, (model_id, path) in enumerate(model_configs):
        print(f"[{i+1}/{len(model_configs)}] Analyzing {model_id}...")
        res = run_model_comparison(path)
        if res:
            all_results[model_id] = res

    if not all_results:
        print("No results to plot.")
        return

    # 2. Plotting
    num_models = len(all_results)
    cols = 3
    rows = math.ceil(num_models / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))
    axes = axes.flatten()
    
    # Algorithm names and colors as per image
    algo_order = ["MEDIA", "OCC", "DINA", "Ours"]
    # Labels in plot: MEDIA, OCC, DNIA, DADS
    # (The image says DNIA instead of DINA)
    plot_labels = ["MEDIA", "OCC", "DINA", "Ours"]
    colors = ['#B22222', '#2E8B57', '#4682B4', '#FFA500'] # Red, Green, Blue, Orange
    
    for i, (model_id, latencies) in enumerate(all_results.items()):
        ax = axes[i]
        vals = [latencies.get(a, 0) / 1000.0 for a in algo_order] # Convert ms to seconds
        
        x = np.arange(len(plot_labels))
        bars = ax.bar(x, vals, color=colors, edgecolor='black', alpha=0.9, width=0.6)
        
        ax.set_title(f"({chr(97+i)}) {model_id}", fontsize=14, pad=10)
        ax.set_ylabel("Inference time (s)", fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(plot_labels, rotation=0, fontsize=10)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        ax.set_axisbelow(True)
        
    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
        
    plt.tight_layout()
    output_file = 'model_comparison_chart.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Chart saved to {output_file}")

if __name__ == "__main__":
    main()
