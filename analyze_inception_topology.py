"""
Analyze InceptionV3 topology to understand why parallelism is lost
"""
from loader import ModelLoader
import networkx as nx

G, layers_map = ModelLoader.load_model_from_csv('datasets/SafeDnnInferenceExp - inceptionV3.csv')

print(f"InceptionV3 model structure:")
print(f"  Nodes: {G.number_of_nodes()}")
print(f"  Edges: {G.number_of_edges()}")

# Check degree distribution
in_degrees = dict(G.in_degree())
out_degrees = dict(G.out_degree())

print(f"\nDegree statistics:")
print(f"  Nodes with in_degree=0: {sum(1 for d in in_degrees.values() if d == 0)}")
print(f"  Nodes with in_degree=1: {sum(1 for d in in_degrees.values() if d == 1)}")
print(f"  Nodes with in_degree>1: {sum(1 for d in in_degrees.values() if d > 1)}")
print(f"  Max in_degree: {max(in_degrees.values())}")

print(f"\n  Nodes with out_degree=0: {sum(1 for d in out_degrees.values() if d == 0)}")
print(f"  Nodes with out_degree=1: {sum(1 for d in out_degrees.values() if d == 1)}")
print(f"  Nodes with out_degree>1: {sum(1 for d in out_degrees.values() if d > 1)}")
print(f"  Max out_degree: {max(out_degrees.values())}")

# Look for fork/join patterns
forks = [n for n, d in out_degrees.items() if d > 1]
joins = [n for n, d in in_degrees.items () if d > 1]

print(f"\nParallel structure indicators:")
print(f"  Fork nodes (out_degree>1): {len(forks)}")
print(f"  Join nodes (in_degree>1): {len(joins)}")

# Sample some fork nodes
if forks:
    print(f"\n  Sample fork nodes:")
    for fork_id in forks[:3]:
        succs = list(G.successors(fork_id))
        layer = layers_map[fork_id]
        print(f"    {fork_id} ({layer.name}): {len(succs)} successors")

# Sample some join nodes
if joins:
    print(f"\n  Sample join nodes:")
    for join_id in joins[:3]:
        preds = list(G.predecessors(join_id))
        layer = layers_map[join_id]
        print(f"    {join_id} ({layer.name}): {len(preds)} predecessors")
