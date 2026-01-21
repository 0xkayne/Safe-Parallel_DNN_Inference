import networkx as nx
import pandas as pd
from loader import ModelLoader
from alg_media import MEDIAAlgorithm
from alg_dina import DINAAlgorithm
from common import Server

# 配置
MODEL_PATH = r"c:\Users\zpwang\Desktop\0_master\毕业论文\Safe-Parallel_DNN_Inference-1\datasets_260120\bert_base.csv"
BANDWIDTH_MBPS = 10000  # 10 Gbps，给一个高带宽试试
NUM_SERVERS = 4

def diagnose_parallelism():
    print(f"=== Diagnosing Parallelism for {MODEL_PATH.split('\\')[-1]} ===")
    
    # 1. Load Model
    G, layers_map = ModelLoader.load_model_from_csv(MODEL_PATH)
    print(f"Total Layers: {len(layers_map)}")
    
    # 2. Setup Servers
    servers = [Server(i) for i in range(NUM_SERVERS)]
    
    # 3. Run MEDIA
    print("\n--- Running MEDIA Algorithm ---")
    media = MEDIAAlgorithm(G, layers_map, servers, BANDWIDTH_MBPS)
    partitions = media.run()
    result = media.schedule(partitions)
    
    print(f"Generated {len(partitions)} partitions.")
    
    # 4. Analyze Partition Structure
    p_dag = nx.DiGraph()
    node_to_pid = {}
    for p in partitions:
        for l in p.layers:
            node_to_pid[l.id] = p.id
            
    for u, v in G.edges():
        if node_to_pid[u] != node_to_pid[v]:
            p_dag.add_edge(node_to_pid[u], node_to_pid[v])
            
    # Check width of Partition DAG
    gen = list(nx.topological_generations(p_dag))
    max_width = max(len(g) for g in gen)
    print(f"Partition DAG Max Width: {max_width} (1 means strictly sequential chain)")
    
    if max_width == 1:
        print(">> FINDING: The partitioner merged all parallel branches into a serial chain.")
        print(">> REASON: Likely because communication cost > compute gain for splitting heads.")
    else:
        print(f">> FINDING: Parallel branches exist! Width = {max_width}")

    # 5. Analyze Schedule Overlap
    print("\n--- Schedule Overlap Analysis ---")
    server_intervals = []
    for s_id, tasks in result.server_schedules.items():
        for t in tasks:
            server_intervals.append((t['start'], t['end'], s_id, t['partition_id']))
            
    # Sort by start time
    server_intervals.sort(key=lambda x: x[0])
    
    overlap_found = False
    total_compute_time = 0
    
    # Check for overlaps
    for i in range(len(server_intervals)):
        start1, end1, s1, p1 = server_intervals[i]
        total_compute_time += (end1 - start1)
        for j in range(i + 1, len(server_intervals)):
            start2, end2, s2, p2 = server_intervals[j]
            
            if start2 < end1: # Overlap
                overlap_time = min(end1, end2) - start2
                if overlap_time > 0.01: # Ignore float errors
                    print(f"  [PARALLEL DETECTED] P{p1} (S{s1}) overlaps with P{p2} (S{s2}) for {overlap_time:.2f} ms")
                    overlap_found = True
            else:
                break
    
    if not overlap_found:
        print("  [NO OVERLAP] No two partitions ran simultaneously.")
        if max_width > 1:
            print("  >> REASON: Partitions are parallel in graph, but assigned to same server or waiting for data.")
    else:
        print("  >> CONFIRMED: Algorithm is scheduling tasks in parallel.")

    print(f"\nTotal End-to-End Latency: {result.latency:.2f} ms")
    print(f"Sum of Compute Times: {total_compute_time:.2f} ms")
    if result.latency < total_compute_time:
        print(">> SUCCESS: Latency < Sum(Compute), proving parallel acceleration.")
    else:
        print(">> RESULT: Latency >= Sum(Compute), strictly sequential execution.")

if __name__ == "__main__":
    diagnose_parallelism()
