#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DNN Model Graph Visualizer

This script reads a DNN profiling CSV file and generates an interactive
HTML visualization of the model's layer dependency graph.

Usage:
    python visualize_model.py --input datasets_260120/bert_base.csv --output bert_viz.html
    python visualize_model.py --input datasets_260120/bert_base.csv --color-by type

Features:
    - Interactive pan/zoom with PyVis
    - Dynamic coloring by any column (group, type, partition_id, etc.)
    - Hover tooltips with layer performance metrics
    - Hierarchical layout showing data flow
"""

import argparse
import ast
import hashlib
import pandas as pd
import networkx as nx
from pyvis.network import Network


def generate_color_palette(unique_values: list) -> dict:
    """Generate a visually distinct color for each unique value."""
    # Predefined palette for common cases (up to 20 colors)
    base_colors = [
        "#FF6B6B",  # Red
        "#4ECDC4",  # Teal
        "#45B7D1",  # Blue
        "#96CEB4",  # Green
        "#FFEAA7",  # Yellow
        "#DDA0DD",  # Plum
        "#98D8C8",  # Mint
        "#F7DC6F",  # Gold
        "#BB8FCE",  # Purple
        "#85C1E9",  # Light Blue
        "#F8B500",  # Orange
        "#00CED1",  # Dark Cyan
        "#FF69B4",  # Hot Pink
        "#32CD32",  # Lime Green
        "#BA55D3",  # Medium Orchid
        "#20B2AA",  # Light Sea Green
        "#FF7F50",  # Coral
        "#6495ED",  # Cornflower Blue
        "#DC143C",  # Crimson
        "#00FA9A",  # Medium Spring Green
    ]
    
    color_map = {}
    for i, val in enumerate(unique_values):
        if i < len(base_colors):
            color_map[val] = base_colors[i]
        else:
            # Generate color from hash for overflow
            hash_val = hashlib.md5(str(val).encode()).hexdigest()[:6]
            color_map[val] = f"#{hash_val}"
    return color_map


def parse_dependencies(dep_str: str) -> list:
    """Parse the dependencies column which may be a string representation of a list, 
    comma-separated, or semicolon-separated."""
    if pd.isna(dep_str) or dep_str == "" or dep_str == "[]":
        return []
    
    dep_str = str(dep_str).strip()
    
    # Cases like "['a', 'b']"
    if dep_str.startswith('[') and dep_str.endswith(']'):
        try:
            return ast.literal_eval(dep_str)
        except (ValueError, SyntaxError):
            # Fallback: remove brackets and quotes, then split by comma
            cleaned = dep_str.strip("[]").replace("'", "").replace('"', "")
            return [s.strip() for s in cleaned.split(",") if s.strip()]
            
    # Semicolon separated (used in InceptionV3)
    if ';' in dep_str:
        return [s.strip() for s in dep_str.split(';') if s.strip()]
        
    # Single item or comma separated
    return [s.strip() for s in dep_str.split(',') if s.strip()]


def load_model_graph(csv_path: str) -> tuple[pd.DataFrame, nx.DiGraph]:
    """Load CSV and build a NetworkX directed graph."""
    df = pd.read_csv(csv_path)
    
    # Normalize column names for case-insensitive lookup
    col_map = {c.lower(): c for c in df.columns}
    
    # Map common variations
    name_col = col_map.get("name") or col_map.get("layername")
    dep_col = col_map.get("dependencies")
    
    if not name_col:
        raise ValueError(f"CSV must contain a 'name' or 'LayerName' column. Found: {list(df.columns)}")
    if not dep_col:
        raise ValueError(f"CSV must contain a 'dependencies' column. Found: {list(df.columns)}")
    
    # Standardize column names in the dataframe for internal use
    df = df.rename(columns={name_col: "name", dep_col: "dependencies"})
    
    # Handle other optional columns
    group_col = col_map.get("group")
    if group_col and group_col != "group":
        df = df.rename(columns={group_col: "group"})
        
    type_col = col_map.get("type")
    if type_col and type_col != "type":
        df = df.rename(columns={type_col: "type"})
        
    time_col = col_map.get("enclave_time_mean") or col_map.get("enclavetime_mean")
    if time_col and time_col != "enclave_time_mean":
        df = df.rename(columns={time_col: "enclave_time_mean"})

    # Parse dependencies
    df["parsed_deps"] = df["dependencies"].apply(parse_dependencies)
    
    # Build graph
    G = nx.DiGraph()
    for _, row in df.iterrows():
        node_name = row["name"]
        if pd.isna(node_name) or str(node_name).strip() == "" or str(node_name).lower() == "all":
            continue
            
        G.add_node(node_name)
        for dep in row["parsed_deps"]:
            if dep:  # Ensure dep is not empty
                G.add_edge(dep, node_name)
    
    return df, G


def build_hover_title(row: pd.Series, exclude_cols: list = None) -> str:
    """Build an HTML hover tooltip from row data."""
    exclude = exclude_cols or ["parsed_deps", "dependencies", "xfer_edges_json"]
    lines = []
    for col, val in row.items():
        if col in exclude:
            continue
        if pd.notna(val):
            # Truncate long values
            val_str = str(val)
            if len(val_str) > 50:
                val_str = val_str[:47] + "..."
            lines.append(f"<b>{col}</b>: {val_str}")
    return "<br>".join(lines)


def visualize_model(
    csv_path: str,
    output_path: str,
    color_by: str = "group",
    height: str = "900px",
    width: str = "100%",
    layout: str = "hierarchical"
):
    """
    Generate an interactive HTML visualization of the DNN model graph.
    
    Args:
        csv_path: Path to the input CSV file.
        output_path: Path to the output HTML file.
        color_by: Column name to use for node coloring.
        height: Height of the visualization canvas.
        width: Width of the visualization canvas.
        layout: Layout algorithm ('hierarchical' or 'physics').
    """
    print(f"Loading model graph from: {csv_path}")
    df, G = load_model_graph(csv_path)
    
    # Determine coloring
    if color_by not in df.columns:
        print(f"Warning: Column '{color_by}' not found. Falling back to 'group' or 'type'.")
        if "group" in df.columns:
            color_by = "group"
        elif "type" in df.columns:
            color_by = "type"
        else:
            color_by = None
    
    if color_by:
        unique_vals = df[color_by].dropna().unique().tolist()
        color_map = generate_color_palette(unique_vals)
        print(f"Coloring by '{color_by}' with {len(unique_vals)} unique values.")
    else:
        color_map = {}
        print("No coloring column available. Using default color.")
    
    # Create PyVis network
    net = Network(
        height=height,
        width=width,
        directed=True,
        bgcolor="#1a1a2e",  # Dark background
        font_color="white",
        heading=""
    )
    
    # Configure layout
    if layout == "hierarchical":
        net.set_options("""
        {
          "layout": {
            "hierarchical": {
              "enabled": true,
              "direction": "UD",
              "sortMethod": "directed",
              "levelSeparation": 80,
              "nodeSpacing": 150
            }
          },
          "physics": {
            "enabled": false
          },
          "nodes": {
            "font": {
              "size": 12,
              "color": "white"
            },
            "borderWidth": 2,
            "borderWidthSelected": 4,
            "shadow": true
          },
          "edges": {
            "color": {
              "color": "#555555",
              "highlight": "#00ff00"
            },
            "arrows": {
              "to": {
                "enabled": true,
                "scaleFactor": 0.5
              }
            },
            "smooth": {
              "type": "cubicBezier"
            }
          },
          "interaction": {
            "hover": true,
            "tooltipDelay": 100
          }
        }
        """)
    else:
        # Physics-based layout
        net.force_atlas_2based()
    
    # Add nodes with attributes
    node_data = df.set_index("name")
    for node in G.nodes():
        if node in node_data.index:
            row = node_data.loc[node]
            color = color_map.get(row.get(color_by), "#888888") if color_by else "#888888"
            title = build_hover_title(row)
            
            # Determine node size based on execution time if available
            size = 15
            if "enclave_time_mean" in row.index and pd.notna(row["enclave_time_mean"]):
                # Scale size: larger nodes for longer execution times
                time_ms = float(row["enclave_time_mean"])
                size = max(10, min(40, 10 + time_ms * 2))  # Clamp between 10-40
            
            net.add_node(
                node,
                label=node,
                color=color,
                title=title,
                size=size
            )
        else:
            # Node not in CSV (should not happen, but handle gracefully)
            net.add_node(node, label=node, color="#888888", size=10)
    
    # Add edges
    for edge in G.edges():
        net.add_edge(edge[0], edge[1])
    
    # Generate HTML
    net.save_graph(output_path)
    print(f"Visualization saved to: {output_path}")
    print(f"  - Nodes: {G.number_of_nodes()}")
    print(f"  - Edges: {G.number_of_edges()}")
    
    # Add legend to HTML
    add_legend_to_html(output_path, color_map, color_by)


def add_legend_to_html(html_path: str, color_map: dict, color_by: str):
    """Inject a CSS legend into the generated HTML file."""
    if not color_map:
        return
    
    legend_items = "".join([
        f'<div style="display:flex;align-items:center;margin:4px 0;">'
        f'<div style="width:16px;height:16px;background:{color};border-radius:3px;margin-right:8px;"></div>'
        f'<span>{label}</span></div>'
        for label, color in color_map.items()
    ])
    
    legend_html = f'''
    <div id="legend" style="
        position: fixed;
        top: 10px;
        right: 10px;
        background: rgba(26, 26, 46, 0.9);
        padding: 15px;
        border-radius: 8px;
        color: white;
        font-family: Arial, sans-serif;
        font-size: 12px;
        max-height: 400px;
        overflow-y: auto;
        z-index: 1000;
        border: 1px solid #333;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    ">
        <div style="font-weight:bold;margin-bottom:10px;border-bottom:1px solid #444;padding-bottom:5px;">
            Color by: {color_by}
        </div>
        {legend_items}
    </div>
    '''
    
    with open(html_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Insert legend before closing body tag
    content = content.replace("</body>", f"{legend_html}</body>")
    
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(content)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize DNN model layer dependencies from CSV data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic visualization
  python visualize_model.py --input datasets_260120/bert_base.csv

  # Color by operation type
  python visualize_model.py --input datasets_260120/bert_base.csv --color-by type

  # Specify output file
  python visualize_model.py --input datasets_260120/bert_base.csv --output my_graph.html

  # Future: Color by partition (after adding partition_id column)
  python visualize_model.py --input partitioned_model.csv --color-by partition_id
        """
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to the input CSV file"
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Path to the output HTML file (default: <input_name>_viz.html)"
    )
    parser.add_argument(
        "--color-by", "-c",
        default="group",
        help="Column name to use for node coloring (default: group)"
    )
    parser.add_argument(
        "--layout", "-l",
        choices=["hierarchical", "physics"],
        default="hierarchical",
        help="Layout algorithm (default: hierarchical)"
    )
    
    args = parser.parse_args()
    
    # Derive output path if not specified
    if args.output is None:
        import os
        module_dir = os.path.dirname(os.path.abspath(__file__))
        outputs_dir = os.path.join(module_dir, "outputs")
        if not os.path.exists(outputs_dir):
            os.makedirs(outputs_dir)
            
        base_name = os.path.splitext(os.path.basename(args.input))[0]
        args.output = os.path.join(outputs_dir, f"{base_name}_viz.html")
    
    visualize_model(
        csv_path=args.input,
        output_path=args.output,
        color_by=args.color_by,
        layout=args.layout
    )


if __name__ == "__main__":
    main()
