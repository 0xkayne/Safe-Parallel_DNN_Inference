
import sys
sys.path.insert(0, '.')
from loader import ModelLoader
from common import Server, hpa_cost, DNNLayer, network_latency

def analyze_bert_layer():
    # Load BERT base
    G, lm = ModelLoader.load_model_from_csv('datasets_260120/bert_base.csv')
    
    # Pick a typical FFN layer (heavy compute, large output)
    # Usually FFN output (fc2) or Attention Output
    # Let's look for "ffn_fc1" which expands dimension (Compute Heavy)
    target_layer = None
    for layer in lm.values():
        if "encoder0_ffn_fc1" in layer.name:
            target_layer = layer
            break
            
    if not target_layer:
        print("Layer encoder0_ffn_fc1 not found")
        return

    print(f"Analyzing Layer: {target_layer.name}")
    print(f"  Workload (Compute): {target_layer.workload:.2f} ms")
    print(f"  Memory: {target_layer.memory:.2f} MB")
    print(f"  Output: {target_layer.output_bytes/1024/1024:.2f} MB")
    
    bandwidth = 500 # Mbps
    print(f"  Bandwidth: {bandwidth} Mbps")
    
    print("-" * 60)
    print(f"{'k':<3} | {'Comp(ms)':<10} | {'Sync(ms)':<10} | {'Total(ms)':<10} | {'Gain?'}")
    print("-" * 60)
    
    cost_k1 = hpa_cost(target_layer, 1, bandwidth)
    
    for k in [1, 2, 4, 8]:
        # Breakdown using hpa_cost logic (replicated for display)
        efficiency_gamma = 0.9
        t_comp = target_layer.workload / (k ** efficiency_gamma)
        
        if k > 1:
             # Ring AllReduce formula from common.py
            sync_bytes = target_layer.output_bytes * 2 * (k - 1) / k
            sync_mb = sync_bytes / (1024 * 1024)
            t_sync = network_latency(sync_mb, bandwidth)
        else:
            t_sync = 0.0
            
        total = t_comp + t_sync 
        
        gain = "YES" if total < cost_k1 else "NO"
        print(f"{k:<3} | {t_comp:<10.2f} | {t_sync:<10.2f} | {total:<10.2f} | {gain}")

if __name__ == "__main__":
    analyze_bert_layer()
