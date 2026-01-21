import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

def create_network_chart(csv_file, model_name):
    """Create a professional line chart for network bandwidth ablation experiment."""
    
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
    
    # Calculate dynamic offset based on data range (using log scale for better visibility)
    # data_min = df[['OCC', 'DINA', 'MEIDA', 'Ours']].min().min()
    # data_max = df[['OCC', 'DINA', 'MEIDA', 'Ours']].max().max()
    
    # For network data, use multiplicative offset instead of additive (works better with log scale)
    offset_factor_up = 0.0     # 3% up
    offset_factor_down = 0.0  # 3% down
    
    # Plot each algorithm
    x = df['Bandwidth(Mbps)']
    for algo in ['OCC', 'DINA', 'MEIDA', 'Ours']:
        y_data = df[algo].copy()
        # Apply multiplicative offset for DINA and MEIDA
        if algo == 'DINA':
            y_data = y_data * offset_factor_up
        elif algo == 'MEIDA':
            y_data = y_data * offset_factor_down
            
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
    ax.set_xlabel('Network Bandwidth (Mbps)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Inference Latency (ms)', fontsize=14, fontweight='bold')
    ax.set_title(f'Inference Latency vs. Network Bandwidth ({model_name})', 
                 fontsize=16, fontweight='bold', pad=15)
    
    # Determine scale based on X-axis range
    # If range is small and linear (e.g. 1-10 Mbps), use linear scale
    # If range spans orders of magnitude (e.g. 1-1000 Mbps), use log scale
    x_min, x_max = x.min(), x.max()
    
    if x_max <= 20 and (x_max - x_min) < 20:
        # Assuming linear small range
        ax.set_xscale('linear')
        ax.set_yscale('log') # Keep Y log as latency varies widely
        
        # Set integer ticks for linear scale
        tick_step = 1 if x_max <= 10 else 2
        ticks = np.arange(int(x_min), int(x_max) + 1, tick_step)
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(int(t)) for t in ticks], fontsize=11)
        ax.set_xlim(x_min - 0.5, x_max + 0.5) # Add padding
        
    else:
        # Default to log scale for wide ranges
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        # Set x-axis ticks to match data points (if not too many)
        if len(x) <= 12:
            ax.set_xticks(x)
            ax.set_xticklabels([str(int(v)) for v in x], fontsize=11)
        
    ax.tick_params(axis='y', labelsize=12)
    
    # Add legend
    ax.legend(loc='upper right', fontsize=12, frameon=True, 
              fancybox=True, shadow=True, framealpha=0.95)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7, which='both')
    
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
    ('network_inceptionV3.csv', 'Inception V3'),
    ('network_DistillBERT.csv', 'DistillBERT'),
    ('network_ALBERT.csv', 'ALBERT'),
    ('network_BERT.csv', 'BERT'),
    ('network_TinyBERT-4l.csv', 'TinyBERT-4L'),
    ('network_TinyBERT-6l.csv', 'TinyBERT-6L'),
    ('network_ViT.csv', 'ViT'),
]

print("Generating network bandwidth charts for all models...")
print("=" * 50)

for csv_file, model_name in models:
    if os.path.exists(csv_file):
        png_file, pdf_file = create_network_chart(csv_file, model_name)
        print(f"[OK] {model_name}: {png_file}, {pdf_file}")
    else:
        print(f"[SKIP] {model_name}: File not found - {csv_file}")

print("=" * 50)
print("All network bandwidth charts generated successfully!")
