"""
Test exhaustive search with a tiny synthetic model
"""
import networkx as nx
from common import DNNLayer, Server, Partition
from alg_ours import OursAlgorithm

def create_tiny_model():
    """
    Create a 4-layer model: 0 -> 1 -> 2 -> 3
    Simple linear chain for validation
    """
    G = nx.DiGraph()
    G.add_edges_from([(0, 1), (1, 2), (2, 3)])
    
    # Add edge weights (communication data in MB)
    for u, v in G.edges():
        G[u][v]['weight'] = 1.0  # 1 MB per edge
    
    layers_map = {
        0: DNNLayer(0, "layer0", 30.0, 500.0, 1000.0, 1024),  # id, name, mem, cpu, enclave, out
        1: DNNLayer(1, "layer1", 50.0, 2000.0, 4000.0, 1024),
        2: DNNLayer(2, "layer2", 20.0, 2000.0, 4000.0, 1024),
        3: DNNLayer(3, "layer3", 50.0, 1500.0, 3000.0, 1024),
    }
    
    return G, layers_map

def main():
    print("="*70)
    print("Testing Ours Exhaustive Search with 4-layer Synthetic Model")
    print("="*70)
    
    # Create tiny model
    G, layers_map = create_tiny_model()
    
    # Create 2 servers with equal power
    servers = [
        Server(0, power_ratio=1.0),
        Server(1, power_ratio=1.0)
    ]
    
    print(f"\nModel: {G.number_of_nodes()} layers, {G.number_of_edges()} edges")
    print(f"Servers: {len(servers)}")
    print(f"Bandwidth: 100 Mbps")
    
    # Run exhaustive search
    print(f"\n{'='*70}")
    print("Running Exhaustive Search...")
    print(f"{'='*70}")
    
    ours = OursAlgorithm(G, layers_map, servers, bandwidth_mbps=100)
    
    # Generate partitions
    partitions = ours.run()
    print(f"\nPartitions generated: {len(ours.all_partition_schemes)} schemes")
    
    # Find optimal assignment
    optimal_time = ours.schedule(partitions)
    
    print(f"\n{'='*70}")
    print("Optimal Solution Found")
    print(f"{'='*70}")
    print(f"Optimal Time: {optimal_time:.4f} ms")
    print(f"Optimal Partitions: {len(ours.optimal_partitions)} partitions")
    
    for i, p in enumerate(ours.optimal_partitions):
        layer_ids = [l.id for l in p.layers]
        server_id = ours.optimal_server_assignment[i]
        print(f"  Partition {i}: layers {layer_ids} → Server {server_id}")
        print(f"    Memory: {p.total_memory:.1f} MB, Workload: {p.total_workload:.1f} M FLOPs")
    
    print(f"\n{'='*70}")
    print("Test PASSED - Algorithm executed successfully!")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
