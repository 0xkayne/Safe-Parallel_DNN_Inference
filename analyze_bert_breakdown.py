
import os
import sys
import math
import pandas as pd
import networkx as nx
from common import Server, EPC_EFFECTIVE_MB, calculate_penalty, PAGE_SIZE_KB, PAGE_FAULT_OVERHEAD_MS, ENCLAVE_ENTRY_EXIT_OVERHEAD_MS, DEFAULT_PAGING_BW_MBPS
from loader import ModelLoader
from alg_dina import DINAAlgorithm
from alg_media import MEDIAAlgorithm
from alg_ours import OursAlgorithm
from alg_occ import OCCAlgorithm

# Configuration
DATASET_PATH = os.path.join('datasets_260120', 'bert_large.csv')
SERVER_TYPE_LIST = ["Celeron_G4930", "i5-6500", "i5-11600", "i5-11600"]
BANDWIDTH = 500  # Mbps
NUM_SERVERS = 4

def analyze_breakdown(name, schedule_result, bandwidth_mbps):
    """
    Summarizes the latency breakdown across all servers.
    """
    total_compute_base = 0.0
    total_penalty_overhead = 0.0
    total_paging_overhead = 0.0
    total_enclave_switching = 0.0
    max_partitions = 0
    
    paging_bw_per_ms = DEFAULT_PAGING_BW_MBPS / 1000.0
    
    # We aggregate the workload processed
    for server_id, events in schedule_result.server_schedules.items():
        max_partitions += len(events)
        for event in events:
            partition = event['partition']
            mem = partition.total_memory
            workload = partition.total_workload
            
            penalty_factor = calculate_penalty(mem)
            base_time = workload
            total_time = (workload * penalty_factor)
            
            total_compute_base += base_time
            total_penalty_overhead += (total_time - base_time)
            
            swap_bytes_mb = partition.get_static_memory()
            num_pages = math.ceil(swap_bytes_mb * 1024 / PAGE_SIZE_KB)
            paging_cost = (num_pages * PAGE_FAULT_OVERHEAD_MS + swap_bytes_mb / paging_bw_per_ms)
            
            total_paging_overhead += paging_cost
            total_enclave_switching += ENCLAVE_ENTRY_EXIT_OVERHEAD_MS

    return {
        "Algorithm": name,
        "Total Latency": schedule_result.latency,
        "Base Compute": total_compute_base / NUM_SERVERS, # Rough average per server
        "EPC Penalty": total_penalty_overhead / NUM_SERVERS,
        "Paging/Swap": total_paging_overhead / NUM_SERVERS,
        "Switching": total_enclave_switching / NUM_SERVERS,
        "Partitions": max_partitions
    }

def run_analysis():
    print(f"Latency Breakdown Analysis for BERT-Base ({NUM_SERVERS} Servers)")
    print(f"=======================================================")
    
    # Load Model
    G, layers_map = ModelLoader.load_model_from_csv(DATASET_PATH)
    total_mem = sum(l.memory for l in layers_map.values())
    print(f"Model: {DATASET_PATH}")
    print(f"Total Model Memory: {total_mem:.2f} MB")
    print(f"EPC Limit: {EPC_EFFECTIVE_MB} MB")
    print("-" * 60)

    servers = [Server(i, server_type=SERVER_TYPE_LIST[i]) for i in range(NUM_SERVERS)]
    
    algorithms = [
        ("DINA", DINAAlgorithm),
        ("MEDIA", MEDIAAlgorithm),
        ("OCC", OCCAlgorithm),
        ("Ours", OursAlgorithm)
    ]
    
    stats_list = []
    
    for name, AlgoClass in algorithms:
        print(f"Running {name}...")
        try:
            algo = AlgoClass(G, layers_map, servers, BANDWIDTH)
            partitions = algo.run()
            schedule = algo.schedule(partitions)
            
            stats = analyze_breakdown(name, schedule, BANDWIDTH)
            stats_list.append(stats)
        except Exception as e:
            print(f"  [Error] {name} failed: {str(e)}")
            import traceback
            traceback.print_exc()

    # Sort and Print manually for crystal clarity
    print("\nDetailed Latency Breakdown (Average per Server in ms):")
    header = f"{'Algorithm':<12} | {'Parts':<6} | {'Compute':<10} | {'Penalty':<10} | {'Paging':<10} | {'Switching':<10} | {'Total Latency':<15}"
    print(header)
    print("-" * len(header))
    
    for s in stats_list:
        row = f"{s['Algorithm']:<12} | {s['Partitions']:<6} | {s['Base Compute']:<10.2f} | {s['EPC Penalty']:<10.2f} | {s['Paging/Swap']:<10.2f} | {s['Switching']:<10.2f} | {s['Total Latency']:<15.2f}"
        print(row)
    print("-" * len(header))
    
    # Verify Validation
    print("\nValidation Analysis (Critical Path):")
    for s in stats_list:
        print(f"  {s['Algorithm']}: Actual Latency = {s['Total Latency']:.2f}")

if __name__ == "__main__":
    run_analysis()
