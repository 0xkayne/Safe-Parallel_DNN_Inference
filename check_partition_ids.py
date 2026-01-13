"""
Check if all partitions have layers
"""
from loader import ModelLoader
from common import Server
from alg_media import MEDIAAlgorithm

G, layers_map = ModelLoader.load_model_from_csv('datasets/SafeDnnInferenceExp - ViT-base.csv')
servers = [Server(i, power_ratio=1.0) for i in range(4)]

media = MEDIAAlgorithm(G, layers_map, servers, bandwidth_mbps=100)
partitions = media.run()

print(f"Total partitions: {len(partitions)}")
print(f"\nPartition IDs: {sorted([p.id for p in partitions])}")

# Check for empty or problematic partitions
for p in partitions:
    if not p.layers:
        print(f"  WARNING: Partition {p.id} has NO layers!")
    if p.total_memory == 0:
        print(f"  WARNING: Partition {p.id} has 0 memory!")

# Check graph building
partition_graph = media._build_partition_graph(partitions)
print(f"\nPartition graph nodes: {sorted(list(partition_graph.nodes()))}")

# Compare
partition_ids = set(p.id for p in partitions)
graph_nodes = set(partition_graph.nodes())

if partition_ids != graph_nodes:
    print(f"\nMISMATCH!")
    print(f"  Missing from graph: {partition_ids - graph_nodes}")
    print(f"  Extra in graph: {graph_nodes - partition_ids}")
