#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Algorithm Visualization Entry Point

Runs a model partitioning algorithm and visualizes the resulting partitions.
"""

import sys
import os
import argparse

# 1. Setup Path to import from parent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from loader import ModelLoader
from common import Server
from alg_dina import DINAAlgorithm
from alg_media import MEDIAAlgorithm
from alg_ours import OursAlgorithm
from alg_occ import OCCAlgorithm
from visualize_model import visualize_partitions


def run_and_visualize(model_path, alg_name, servers_count, bw, output_path=None):
    """Core logic to run an algorithm and generate visualization."""
    # 1. Setup Path to import from parent directory
    # (Already handled at module level in this file)
    
    # Load Model
    print(f"Loading model: {model_path}")
    G, layers_map = ModelLoader.load_model_from_csv(model_path)
    
    # Create Servers
    servers = [Server(i) for i in range(servers_count)]
    
    # Select Algorithm
    if alg_name == "dina":
        alg = DINAAlgorithm(G, layers_map, servers, bw)
    elif alg_name == "media":
        alg = MEDIAAlgorithm(G, layers_map, servers, bw)
    elif alg_name == "ours":
        alg = OursAlgorithm(G, layers_map, servers, bw)
    else: # occ
        alg = OCCAlgorithm(G, layers_map, servers, bw)
    
    # Run Algorithm
    print(f"Running algorithm: {alg_name.upper()}")
    partitions = alg.run()
    
    # Schedule (to get server assignments)
    try:
        schedule_res = alg.schedule(partitions)
        print(f"Latency: {schedule_res.latency:.2f} ms")
        if hasattr(schedule_res, 'partitions'):
            partitions = schedule_res.partitions
    except Exception as e:
        print(f"Warning during scheduling: {e}")

    # Determine Output Path
    if output_path is None:
        outputs_dir = os.path.join(current_dir, "outputs")
        if not os.path.exists(outputs_dir):
            os.makedirs(outputs_dir)
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        output_path = os.path.join(outputs_dir, f"{model_name}_partitioned_{alg_name}.html")

    # Visualize
    visualize_partitions(G, partitions, output_path, title=f"{alg_name.upper()} - {servers_count} Srv, {bw} Mbps")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Run and visualize model partitioning algorithms.")
    parser.add_argument("--model", "-m", required=True, help="Path to the model CSV file")
    parser.add_argument("--alg", "-a", choices=["dina", "media", "ours", "occ"], default="ours", help="Algorithm to run")
    parser.add_argument("--servers", "-s", type=int, default=4, help="Number of servers")
    parser.add_argument("--bw", "-b", type=int, default=100, help="Bandwidth in Mbps")
    parser.add_argument("--output", "-o", default=None, help="Output HTML path")

    args = parser.parse_args()
    
    run_and_visualize(args.model, args.alg, args.servers, args.bw, args.output)


if __name__ == "__main__":
    main()
