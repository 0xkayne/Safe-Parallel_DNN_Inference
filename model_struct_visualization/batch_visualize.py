#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Batch DNN Model Visualizer

Iterates through all CSV files in the datasets directory and generates
interactive visualizations for each, structured into individual folders.
"""

import os
import glob
from visualize_model import visualize_model

def main():
    # 1. Setup Paths
    # This script is in model_struct_visualization/
    current_dir = os.path.dirname(os.path.abspath(__file__))
    datasets_dir = os.path.join(os.path.dirname(current_dir), "datasets_260120")
    outputs_base_dir = os.path.join(current_dir, "outputs")
    
    if not os.path.exists(datasets_dir):
        print(f"Error: Datasets directory not found at {datasets_dir}")
        return

    # 2. Find all CSV files
    csv_files = glob.glob(os.path.join(datasets_dir, "*.csv"))
    print(f"Found {len(csv_files)} models in {datasets_dir}")

    # 3. Process each model
    for csv_path in csv_files:
        model_name = os.path.splitext(os.path.basename(csv_path))[0]
        model_output_dir = os.path.join(outputs_base_dir, model_name)
        
        print(f"\n[Processing] {model_name}...")
        
        # Create dedicated folder for this model
        if not os.path.exists(model_output_dir):
            os.makedirs(model_output_dir)
            
        try:
            # Generate visualization by 'group'
            group_output = os.path.join(model_output_dir, f"{model_name}_by_group.html")
            visualize_model(
                csv_path=csv_path,
                output_path=group_output,
                color_by="group"
            )
            
            # Generate visualization by 'type'
            type_output = os.path.join(model_output_dir, f"{model_name}_by_type.html")
            visualize_model(
                csv_path=csv_path,
                output_path=type_output,
                color_by="type"
            )
            
            print(f"Successfully generated visualizations in: {model_output_dir}")
            
        except Exception as e:
            print(f"Error processing {model_name}: {e}")

    print("\n" + "="*50)
    print(f"Batch processing complete! Results are in: {outputs_base_dir}")

if __name__ == "__main__":
    main()
