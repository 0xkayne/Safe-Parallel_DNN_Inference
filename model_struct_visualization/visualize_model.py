#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DNN Model Graph Visualizer

This script reads a DNN profiling CSV file and generates an interactive
HTML visualization of the model's layer dependency graph.

Features:
    - Interactive pan/zoom with PyVis
    - Dynamic coloring by any column (group, type, partition_id, etc.)
    - Hover tooltips with layer performance metrics
    - Hierarchical layout showing data flow
    - Support for visualizing model partitions from algorithms
"""

import argparse
import ast
import hashlib
import os
import pandas as pd
import networkx as nx
from pyvis.network import Network


def generate_color_palette(unique_values: list) -> dict:
    """Generate a visually distinct color for each unique value."""
    base_colors = [
        "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7",
        "#DDA0DD", "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E9",
        "#F8B500", "#00CED1", "#FF69B4", "#32CD32", "#BA55D3",
        "#20B2AA", "#FF7F50", "#6495ED", "#DC143C", "#00FA9A",
    ]
    
    color_map = {}
    for i, val in enumerate(unique_values):
        if i < len(base_colors):
            color_map[val] = base_colors[i]
        else:
            hash_val = hashlib.md5(str(val).encode()).hexdigest()[:6]
            color_map[val] = f"#{hash_val}"
    return color_map


def parse_dependencies(dep_str: str) -> list:
    """Parse the dependencies column which may be a string representation of a list, 
    comma-separated, or semicolon-separated."""
    if pd.isna(dep_str) or dep_str == "" or dep_str == "[]":
        return []
    
    dep_str = str(dep_str).strip()
    
    if dep_str.startswith('[') and dep_str.endswith(']'):
        try:
            return ast.literal_eval(dep_str)
        except (ValueError, SyntaxError):
            cleaned = dep_str.strip("[]").replace("'", "").replace('"', "")
            return [s.strip() for s in cleaned.split(",") if s.strip()]
            
    if ';' in dep_str:
        return [s.strip() for s in dep_str.split(';') if s.strip()]
        
    return [s.strip() for s in dep_str.split(',') if s.strip()]


def load_model_graph(csv_path: str) -> tuple[pd.DataFrame, nx.DiGraph]:
    """Load CSV and build a NetworkX directed graph."""
    df = pd.read_csv(csv_path)
    
    col_map = {c.lower(): c for c in df.columns}
    name_col = col_map.get("name") or col_map.get("layername")
    dep_col = col_map.get("dependencies")
    
    if not name_col:
        raise ValueError(f"CSV must contain a 'name' or 'LayerName' column. Found: {list(df.columns)}")
    if not dep_col:
        raise ValueError(f"CSV must contain a 'dependencies' column. Found: {list(df.columns)}")
    
    df = df.rename(columns={name_col: "name", dep_col: "dependencies"})
    
    for opt in ["group", "type"]:
        col = col_map.get(opt)
        if col and col != opt:
            df = df.rename(columns={col: opt})
            
    time_col = col_map.get("enclave_time_mean") or col_map.get("enclavetime_mean")
    if time_col and time_col != "enclave_time_mean":
        df = df.rename(columns={time_col: "enclave_time_mean"})

    df["parsed_deps"] = df["dependencies"].apply(parse_dependencies)
    
    G = nx.DiGraph()
    for _, row in df.iterrows():
        node_name = row["name"]
        if pd.isna(node_name) or str(node_name).strip() == "" or str(node_name).lower() == "all":
            continue
            
        G.add_node(node_name, **row.to_dict())
        for dep in row["parsed_deps"]:
            if dep:
                G.add_edge(dep, node_name)
    
    return df, G


def build_hover_title(data: dict, exclude_cols: list = None) -> str:
    """Build an HTML hover tooltip from node data dictionary."""
    exclude = exclude_cols or ["parsed_deps", "dependencies", "xfer_edges_json", "parsed_deps"]
    lines = []
    for col, val in data.items():
        if col in exclude:
            continue
        if pd.notna(val):
            val_str = str(val)
            if len(val_str) > 60:
                val_str = val_str[:57] + "..."
            lines.append(f"<b>{col}</b>: {val_str}")
    return "<br>".join(lines)


def _create_base_network(height="900px", width="100%", layout="hierarchical"):
    """Initialize a PyVis Network with standard options."""
    net = Network(
        height=height,
        width=width,
        directed=True,
        bgcolor="#1a1a2e",
        font_color="white",
        heading=""
    )
    
    if layout == "hierarchical":
        net.set_options("""
        {
          "layout": {
            "hierarchical": {
              "enabled": true,
              "direction": "UD",
              "sortMethod": "directed",
              "levelSeparation": 100,
              "nodeSpacing": 150
            }
          },
          "physics": {"enabled": false},
          "nodes": {
            "font": {"size": 12, "color": "white"},
            "borderWidth": 2,
            "shadow": true
          },
          "edges": {
            "color": {"color": "#555555", "highlight": "#00ff00"},
            "arrows": {"to": {"enabled": true, "scaleFactor": 0.5}},
            "smooth": {"type": "cubicBezier"}
          },
          "interaction": {"hover": true, "tooltipDelay": 100}
        }
        """)
    else:
        net.force_atlas_2based()
    return net


def visualize_model(csv_path, output_path, color_by="group", layout="hierarchical"):
    """Process a CSV and generate visualization."""
    df, G = load_model_graph(csv_path)
    
    if color_by not in df.columns:
        color_by = "group" if "group" in df.columns else ("type" if "type" in df.columns else None)
    
    unique_vals = df[color_by].dropna().unique().tolist() if color_by else []
    color_map = generate_color_palette(unique_vals)
    
    net = _create_base_network(layout=layout)
    
    for node, data in G.nodes(data=True):
        color = color_map.get(data.get(color_by), "#888888") if color_by else "#888888"
        title = build_hover_title(data)
        
        # Scaling size by workload/time
        time_val = data.get("enclave_time_mean", 10)
        try:
            size = max(10, min(40, 10 + float(time_val) * 2))
        except:
            size = 15
            
        net.add_node(node, label=node, color=color, title=title, size=size)
    
    for edge in G.edges():
        net.add_edge(edge[0], edge[1])
        
    net.save_graph(output_path)
    add_legend_to_html(output_path, color_map, color_by)


def visualize_partitions(G, partitions, output_path, title="Model Partitions"):
    """
    Visualize model graph with nodes colored by Partition ID.
    
    Args:
        G: NetworkX DiGraph (layers as nodes)
        partitions: List of Partition objects (from alg_*.py)
        output_path: Path to save HTML
    """
    # Map layers to partition IDs
    node_to_part = {}
    for p in partitions:
        for layer in p.layers:
            node_to_part[layer.id] = p.id
            
    # Also handle layer names if graph uses names as IDs
    # Algorithm uses layer.id which is an integer index.
    # Our CSV loader uses layer.name as node ID.
    # We need to bridge this. The ModelLoader usually returns a layers_map {idx: LayerObj}.
    
    # Let's check if the nodes in G are names or indices
    is_name_indexed = isinstance(list(G.nodes())[0], str)
    
    partition_ids = sorted(list({p.id for p in partitions}))
    color_map = generate_color_palette(partition_ids)
    
    net = _create_base_network(layout="hierarchical")
    
    # We need a way to relate G's nodes to partitions
    # If G comes from ModelLoader, it's indexed by layer.id (int)
    # If G comes from load_model_graph, it's indexed by layer.name (str)
    
    for node, data in G.nodes(data=True):
        # Determine layer name for lookups
        if hasattr(data, 'get') and data.get('name'):
            l_name = data['name']
            l_id = data.get('id') # may not exist
        else:
            # Fallback if graph is simple
            l_name = str(node)
            l_id = node

        # Find partition ID
        p_id = -1
        p_obj = None
        for p in partitions:
            if any(l.name == l_name or l.id == l_id for l in p.layers):
                p_id = p.id
                p_obj = p
                break
        
        color = color_map.get(p_id, "#888888")
        
        # Build tooltip with partition info
        extra_info = {"Partition": p_id}
        if p_obj:
            extra_info["Part Memory (MB)"] = f"{p_obj.total_memory:.2f}"
            extra_info["Part Workload (ms)"] = f"{p_obj.total_workload:.2f}"
            if hasattr(p_obj, 'assigned_server') and p_obj.assigned_server:
                extra_info["Server"] = p_obj.assigned_server
        
        # Merge data from graph node if it's a dict
        if isinstance(data, dict):
            full_data = {**data, **extra_info}
        else:
            full_data = extra_info
            
        tooltip = build_hover_title(full_data)
        
        net.add_node(node, label=str(node), color=color, title=tooltip, size=20)
        
    for edge in G.edges():
        net.add_edge(edge[0], edge[1])
        
    net.save_graph(output_path)
    add_legend_to_html(output_path, color_map, f"Partition ID ({title})")
    print(f"Partition visualization saved to: {output_path}")


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
    
    content = content.replace("</body>", f"{legend_html}</body>")
    
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(content)


def main():
    parser = argparse.ArgumentParser(description="Visualize DNN model graph from CSV data.")
    parser.add_argument("--input", "-i", required=True, help="Path to input CSV")
    parser.add_argument("--output", "-o", default=None, help="Output HTML path")
    parser.add_argument("--color-by", "-c", default="group", help="Column for coloring")
    parser.add_argument("--layout", "-l", choices=["hierarchical", "physics"], default="hierarchical")
    
    args = parser.parse_args()
    
    if args.output is None:
        module_dir = os.path.dirname(os.path.abspath(__file__))
        outputs_dir = os.path.join(module_dir, "outputs")
        if not os.path.exists(outputs_dir):
            os.makedirs(outputs_dir)
        base_name = os.path.splitext(os.path.basename(args.input))[0]
        args.output = os.path.join(outputs_dir, f"{base_name}_viz.html")
    
    visualize_model(args.input, args.output, args.color_by, args.layout)


if __name__ == "__main__":
    main()
