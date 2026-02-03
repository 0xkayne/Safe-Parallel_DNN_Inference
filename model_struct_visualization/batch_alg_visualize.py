#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Batch Algorithm Visualization

Automatically runs all 4 partitioning algorithms for all models in the dataset
and generates hierarchical visualization HTML files.
"""

import os
import glob
from visualize_alg import run_and_visualize

def main():
    # 1. Setup Paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    datasets_dir = os.path.join(parent_dir, "datasets_260120")
    outputs_base_dir = os.path.join(current_dir, "outputs")
    
    if not os.path.exists(datasets_dir):
        print(f"Error: Datasets directory not found at {datasets_dir}")
        return

    # 2. Find all model CSV files
    csv_files = glob.glob(os.path.join(datasets_dir, "*.csv"))
    print(f"Found {len(csv_files)} models in {datasets_dir}")

    # 3. Algorithms to run
    algorithms = ["dina", "media", "ours", "occ"]
    
    # 4. Batch Process
    for csv_path in csv_files:
        model_name = os.path.splitext(os.path.basename(csv_path))[0]
        # Clean model name if it has prefixes like "SafeDnnInferenceExp - "
        clean_model_name = model_name.replace('SafeDnnInferenceExp - ', '')
        
        model_output_dir = os.path.join(outputs_base_dir, clean_model_name, "partitions")
        
        print(f"\n" + "="*60)
        print(f"[Model] {clean_model_name}")
        print(f"="*60)
        
        if not os.path.exists(model_output_dir):
            os.makedirs(model_output_dir)
            
        for alg in algorithms:
            print(f"\n>>> Running {alg.upper()}...")
            output_file = os.path.join(model_output_dir, f"{alg}.html")
            
            try:
                run_and_visualize(
                    model_path=csv_path,
                    alg_name=alg,
                    servers_count=4,
                    bw=100,
                    output_path=output_file
                )
            except Exception as e:
                print(f"Error processing {clean_model_name} with {alg}: {e}")

    print("\n" + "="*60)
    print(f"Batch processing complete!")
    print(f"Results are organized in: {outputs_base_dir}")
    print("Structure: <model_name>/partitions/<algorithm>.html")

if __name__ == "__main__":
    main()
