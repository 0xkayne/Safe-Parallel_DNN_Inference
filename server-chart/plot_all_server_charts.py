import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

def create_server_chart(csv_file, model_name):
    """Create a professional line chart for server scalability experiment."""
    
    # Read data
    df = pd.read_csv(csv_file)
    
    # Set up the figure with a professional style
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    
    # Define colors - using a professional color palette
    colors = {
        'OCC': '#E74C3C',      # Red
        'DINA': '#3498DB',     # Blue
        'MEIDA': '#2ECC71',    # Green
        'Ours': '#9B59B6'      # Purple
    }
    
    # Define markers
    markers = {
        'OCC': 's',      # Square
        'DINA': '^',     # Triangle up
        'MEIDA': 'D',    # Diamond
        'Ours': 'o'      # Circle
    }
    
    # Define line styles - to distinguish overlapping curves
    linestyles = {
        'OCC': '-',           # Solid
        'DINA': (0, (5, 2)),  # Dashed with specific pattern (5 on, 2 off)
        'MEIDA': (0, (1, 1)), # Dense dotted (1 on, 1 off)
        'Ours': '-'           # Solid
    }
    
    # Define marker sizes - larger difference to help distinguish overlapping curves
    markersizes = {
        'OCC': 9,
        'DINA': 12,        # Much larger marker for visibility
        'MEIDA': 8,        # Smaller marker
        'Ours': 10
    }
    
    # Define line widths - different widths for overlapping lines
    linewidths = {
        'OCC': 2.5,
        'DINA': 3.0,       # Thicker line
        'MEIDA': 2.0,      # Thinner line
        'Ours': 2.5
    }
    
    # Calculate dynamic offset based on data range
    # data_range = df[['OCC', 'DINA', 'MEIDA', 'Ours']].max().max() - df[['OCC', 'DINA', 'MEIDA', 'Ours']].min().min()
    # offset_value = data_range * 0.005  # 0.5% of data range
    
    # Add small vertical offset to separate overlapping lines visually
    offsets = {
        'OCC': 0,
        'DINA': 0,       # Slight upward offset
        'MEIDA': 0,     # Slight downward offset
        'Ours': 0
    }
    
    # Plot each algorithm
    x = df['Server number']
    for algo in ['OCC', 'DINA', 'MEIDA', 'Ours']:
        y_data = df[algo] + offsets[algo]  # Apply offset
        ax.plot(x, y_data, 
                label=algo, 
                color=colors[algo], 
                marker=markers[algo],
                markersize=markersizes[algo],
                linewidth=linewidths[algo],
                linestyle=linestyles[algo],
                markeredgecolor='white',
                markeredgewidth=1.5,
                zorder=3 if algo in ['DINA', 'MEIDA'] else 2)
    
    # Customize the plot
    ax.set_xlabel('Number of Servers', fontsize=14, fontweight='bold')
    ax.set_ylabel('Inference Latency (ms)', fontsize=14, fontweight='bold')
    ax.set_title(f'Inference Latency vs. Number of Servers ({model_name})', 
                 fontsize=16, fontweight='bold', pad=15)
    
    # Set x-axis ticks to match data points
    ax.set_xticks(x)
    ax.set_xticklabels(x, fontsize=12)
    ax.tick_params(axis='y', labelsize=12)
    
    # Set y-axis range with some padding
    y_min = df[['OCC', 'DINA', 'MEIDA', 'Ours']].min().min() * 0.90
    y_max = df[['OCC', 'DINA', 'MEIDA', 'Ours']].max().max() * 1.08
    ax.set_ylim(y_min, y_max)
    
    # Add legend
    ax.legend(loc='upper right', fontsize=12, frameon=True, 
              fancybox=True, shadow=True, framealpha=0.95)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add annotation for the key insight
    ax.annotate('Lower is better', 
                xy=(0.02, 0.02), xycoords='axes fraction',
                fontsize=10, fontstyle='italic', color='gray')
    
    # Tight layout
    plt.tight_layout()
    
    # Generate output filenames
    base_name = os.path.splitext(os.path.basename(csv_file))[0]
    output_png = f"{base_name}_chart.png"
    output_pdf = f"{base_name}_chart.pdf"
    
    # Save the figure
    plt.savefig(output_png, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.savefig(output_pdf, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    plt.close()
    
    return output_png, output_pdf


# Define all models to process
models = [
    ('server_hetero_incremental_DistillBERT.csv', 'DistillBERT'),
    ('server_hetero_incremental_ALBERT.csv', 'ALBERT'),
    ('server_hetero_incremental_BERT.csv', 'BERT'),
    ('server_hetero_incremental_TinyBERT-4l.csv', 'TinyBERT-4L'),
    ('server_hetero_incremental_TinyBERT-6l.csv', 'TinyBERT-6L'),
    ('server_hetero_incremental_ViT.csv', 'ViT'),
    ('server_hetero_incremental_inceptionV3.csv', 'InceptionV3'),
]

print("Generating charts for all models...")
print("=" * 50)

for csv_file, model_name in models:
    if os.path.exists(csv_file):
        png_file, pdf_file = create_server_chart(csv_file, model_name)
        print(f"[OK] {model_name}: {png_file}, {pdf_file}")
    else:
        print(f"[SKIP] {model_name}: File not found - {csv_file}")

print("=" * 50)
print("All charts generated successfully!")
