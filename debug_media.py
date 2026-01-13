"""
Debug script to trace MEDIA partitioning and scheduling
"""
from loader import ModelLoader
from common import Server
from alg_media import MEDIAAlgorithm

def debug_media(model_path, model_name, n_servers):
    print(f"\n{'='*70}")
    print(f"DEBUG: {model_name} with {n_servers} servers")
    print(f"{'='*70}")
    
    # Load model
    G, layers_map = ModelLoader.load_model_from_csv(model_path)
    servers = [Server(i, power_ratio=1.0) for i in range(n_servers)]
    
    print(f"\nModel Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Run MEDIA
    media = MEDIAAlgorithm(G, layers_map, servers, bandwidth_mbps=100)
    
    # Check edge selection
    edges_M = media._select_edges_for_partitioning()
    print(f"\nEdge Selection (Algorithm 1):")
    print(f"  Total edges in graph: {G.number_of_edges()}")
    print(f"  Mergeable edges (M): {len(edges_M)}")
    
    # Run partitioning
    partitions = media.run()
    print(f"\nPartitioning (Algorithm 2):")
    print(f"  Total partitions: {len(partitions)}")
    for i, p in enumerate(partitions[:5]):  # Show first 5
        print(f"  Partition {p.id}: {len(p.layers)} layers, {p.total_memory:.2f} MB, {p.total_workload:.2f} ms")
    if len(partitions) > 5:
        print(f"  ... and {len(partitions) - 5} more partitions")
    
    # Build partition graph
    partition_graph = media._build_partition_graph(partitions)
    print(f"\nPartition Dependency Graph:")
    print(f"  Nodes: {partition_graph.number_of_nodes()}")
    print(f"  Edges: {partition_graph.number_of_edges()}")
    
    # Check for independent partitions (candidates for parallelization)
    in_deg_0 = [n for n in partition_graph.nodes() if partition_graph.in_degree(n) == 0]
    out_deg_0 = [n for n in partition_graph.nodes() if partition_graph.out_degree(n) == 0]
    print(f"  Root partitions (in_degree=0): {len(in_deg_0)}")
    print(f"  Leaf partitions (out_degree=0): {len(out_deg_0)}")
    
    # Run scheduling
    time_media = media.schedule(partitions)
    print(f"\nScheduling (Algorithm 3):")
    print(f"  Total inference time: {time_media:.2f} ms")
    
    print(f"{'='*70}\n")

# Test on both models
debug_media('datasets/SafeDnnInferenceExp - ViT-base.csv', 'ViT-base', 4)
debug_media('datasets/SafeDnnInferenceExp - inceptionV3.csv', 'InceptionV3', 4)
