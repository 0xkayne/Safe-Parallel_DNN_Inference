import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Read data
df = pd.read_csv('server_inceptionV3.csv')

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

# Add small vertical offset to separate overlapping lines visually
# This is a common technique in scientific visualization
offsets = {
    'OCC': 0,
    'DINA': 5,         # Slight upward offset
    'MEIDA': -5,       # Slight downward offset
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
ax.set_title('Inference Latency vs. Number of Servers (Inception V3)', 
             fontsize=16, fontweight='bold', pad=15)

# Set x-axis ticks to match data points
ax.set_xticks(x)
ax.set_xticklabels(x, fontsize=12)
ax.tick_params(axis='y', labelsize=12)

# Set y-axis range with some padding
y_min = df[['OCC', 'DINA', 'MEIDA', 'Ours']].min().min() * 0.95
y_max = df[['OCC', 'DINA', 'MEIDA', 'Ours']].max().max() * 1.05
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

# Save the figure
plt.savefig('server_inceptionV3_chart.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.savefig('server_inceptionV3_chart.pdf', bbox_inches='tight', 
            facecolor='white', edgecolor='none')

print("Chart saved as 'server_inceptionV3_chart.png' and 'server_inceptionV3_chart.pdf'")

# Show the plot
plt.show()
