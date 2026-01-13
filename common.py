import networkx as nx

# Global Constants (Defaults, can be overridden during experiment)
EPC_EFFECTIVE_MB = 93.0  # (128 - 35) MB
# Dynamic SGX Paging Penalty Model
# Based on research: Penalty is linear to the oversubscription ratio (due to encryption bandwidth bottleneck)
# Formula: Penalty = 1 + Delta + Gamma * (Ratio - 1)
# Delta (0.5): Base overhead for triggering paging mechanism (Context switch, etc.)
# Gamma (3.0): Bandwidth gap (crypto engine vs DRAM). 
def calculate_penalty(memory_mb):
    if memory_mb <= EPC_EFFECTIVE_MB:
        return 1.0
    
    ratio = memory_mb / EPC_EFFECTIVE_MB
    delta = 0.5
    gamma = 3.0
    
    penalty = 1.0 + delta + gamma * (ratio - 1.0)
    return penalty

EPC_EFFECTIVE_MB = 93.0  # (128 - 35) MB

# SGX Paging Bandwidth (EPC ↔ DRAM with AES encryption)
# Typical value: ~2 GB/s = 2000 MB/s = 2.0 MB/ms
PAGING_BANDWIDTH_MB_PER_MS = 2.0

class DNNLayer:
    def __init__(self, layer_id, name, memory, cpu_time, enclave_time, output_bytes):
        self.id = layer_id
        self.name = name
        self.memory = memory          # MB
        self.cpu_time = cpu_time      # ms
        self.enclave_time = enclave_time # ms
        self.output_bytes = output_bytes # Bytes
        
        # Workload is treated as execution time in this simulation for simplicity,
        # or we can derive FLOPs if we had server power. 
        # For this exp, we use the measured time directly.
        self.workload = enclave_time 

    def __repr__(self):
        return f"Layer({self.name}, mem={self.memory:.2f})"

class Partition:
    def __init__(self, partition_id, layers):
        self.id = partition_id
        self.layers = layers  # List of DNNLayer objects
        
        self.total_memory = sum(l.memory for l in layers)
        # Simple sum of workloads
        self.total_workload = sum(l.workload for l in layers)
        
        self.assigned_server = None
        self.start_time = 0.0
        self.finish_time = 0.0
        self.ready_time = 0.0

    def __repr__(self):
        return f"Part#{self.id}(n={len(self.layers)}, mem={self.total_memory:.1f})"

class Server:
    def __init__(self, server_id, power_ratio=1.0):
        self.id = server_id
        self.power_ratio = power_ratio # 1.0 = baseline speed, 2.0 = 2x faster, etc.
        self.schedule = []  # List of (start, end, partition_id)
        self.assigned_memory = 0.0 

    def __repr__(self):
        return f"Server#{self.id}"
