"""
More detailed debug to find the partitioning issue
"""
from loader import ModelLoader
from common import Server
from alg_media import MEDIAAlgorithm

def debug_partitioning():
    # Load ViT-base
    G, layers_map = ModelLoader.load_model_from_csv('datasets/SafeDnnInferenceExp - ViT-base.csv')
    servers = [Server(i, power_ratio=1.0) for i in range(4)]
    
    media = MEDIAAlgorithm(G, layers_map, servers, bandwidth_mbps=100)
    
    # Run partitioning
    partitions = media.run()
    
    print(f"Total nodes in G: {G.number_of_nodes()}")
    print(f"Total partitions created: {len(partitions)}")
    print(f"node_to_partition mapping size: {len(media.node_to_partition)}")
    
    # Check if all nodes are mapped
    missing_nodes = []
    for node_id in G.nodes():
        if node_id not in media.node_to_partition:
            missing_nodes.append(node_id)
    
    if missing_nodes:
        print(f"\nWARNING: {len(missing_nodes)} nodes NOT mapped to partitions!")
        print(f"Missing nodes: {missing_nodes[:10]}")
    else:
        print("\nAll nodes are properly mapped.")
    
    # Check partition graph construction
    print("\n" + "="*60)
    print("Checking partition dependency graph construction...")
    
    partition_graph = media._build_partition_graph(partitions)
    
    # Sample some edges from original graph
    print(f"\nSampling original graph edges:")
    edge_count = 0
    for u, v in list(G.edges())[:5]:
        pu = media.node_to_partition.get(u)
        pv = media.node_to_partition.get(v)
        
        if pu and pv:
            print(f"  Edge ({u}, {v}): partition {pu.id} -> partition {pv.id}")
            if pu.id == pv.id:
                print(f"    -> Same partition (merged)")
                edge_count += 1
        else:
            print(f"  Edge ({u}, {v}): MISSING PARTITION MAPPING!")
    
    if edge_count == 5:
        print(f"\n  All sampled edges were merged into same partitions!")
        print(f"  This explains why partition graph has 0 edges.")
    
    print(f"\nPartition graph: {partition_graph.number_of_nodes()} nodes, {partition_graph.number_of_edges()} edges")

debug_partitioning()
