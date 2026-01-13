"""
Analyze ViT graph structure
"""
from loader import ModelLoader
import networkx as nx

G, layers_map = ModelLoader.load_model_from_csv('datasets/SafeDnnInferenceExp - ViT-base.csv')

print(f"ViT-base model structure:")
print(f"  Nodes: {G.number_of_nodes()}")
print(f"  Edges: {G.number_of_edges()}")

# Check for disconnected components
components = list(nx.weakly_connected_components(G))
print(f"  Weakly connected components: {len(components)}")

if len(components) > 1:
    for i, comp in enumerate(components[:5]):
        print(f"    Component {i}: {len(comp)} nodes")

# Check degree distribution
in_degrees = dict(G.in_degree())
out_degrees = dict(G.out_degree())

print(f"\nDegree statistics:")
print(f"  Nodes with in_degree=0: {sum(1 for d in in_degrees.values() if d == 0)}")
print(f"  Nodes with out_degree=0: {sum(1 for d in out_degrees.values() if d == 0)}")
print(f"  Nodes with in_degree=1: {sum(1 for d in in_degrees.values() if d == 1)}")
print(f"  Nodes with out_degree=1: {sum(1 for d in out_degrees.values() if d == 1)}")

# Sample node connections
print(f"\n Sample nodes and their connections:")
for node_id in list(G.nodes())[:10]:
    preds = list(G.predecessors(node_id))
    succs = list(G.successors(node_id))
    layer = layers_map[node_id]
    print(f"  Node {node_id} ({layer.name}): in={len(preds)}, out={len(succs)}")
