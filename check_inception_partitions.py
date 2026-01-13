"""
Check InceptionV3 partition graph topology
"""
from loader import ModelLoader
from common import Server
from alg_media import MEDIAAlgorithm

G, layers_map = ModelLoader.load_model_from_csv('datasets/SafeDnnInferenceExp - inceptionV3.csv')
servers = [Server(i, power_ratio=1.0) for i in range(4)]

media = MEDIAAlgorithm(G, layers_map, servers, bandwidth_mbps=100)
partitions = media.run()
partition_graph = media._build_partition_graph(partitions)

print(f"Partition Graph structure for InceptionV3:")
print(f"  Nodes: {partition_graph.number_of_nodes()}")
print(f"  Edges: {partition_graph.number_of_edges()}")

# Check topology
for p_id in partition_graph.nodes():
    preds = list(partition_graph.predecessors(p_id))
    succs = list(partition_graph.successors(p_id))
    p = partitions[p_id]
    print(f"\n  Partition {p_id}:")
    print(f"    Layers: {len(p.layers)}")
    print(f"    Memory: {p.total_memory:.2f} MB")
    print(f"    Workload: {p.total_workload:.2f} ms")
    print(f"    Predecessors: {preds}")
    print(f"    Successors: {succs}")

# Check if it's a chain
is_chain = all(
    partition_graph.in_degree(n) <= 1 and partition_graph.out_degree(n) <= 1
    for n in partition_graph.nodes()
)

print(f"\nIs linear chain? {is_chain}")

if is_chain:
    print("  -> This explains why there's no parallelism in scheduling!")
    print("  -> All partitions must execute sequentially.")
