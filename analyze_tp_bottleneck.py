
import sys
sys.path.insert(0, '.')
from loader import ModelLoader
from common import Server, hpa_cost, DNNLayer, network_latency, calculate_penalty

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
    print(f"  Weight Memory: {target_layer.weight_memory:.2f} MB")
    print(f"  Activation Memory: {target_layer.activation_memory:.2f} MB")
    print(f"  Total Memory: {target_layer.memory:.2f} MB")
    print(f"  Output: {target_layer.output_bytes/1024/1024:.2f} MB")
    
    bandwidth = 500 # Mbps
    print(f"  Bandwidth: {bandwidth} Mbps")
    
    # Compare OLD model (activation replicated, full sync) vs NEW model (activation split, amortized sync)
    print("\n" + "=" * 80)
    print("COMPARISON: OLD MODEL vs NEW MODEL")
    print("=" * 80)
    
    print("\n[OLD MODEL] activation_split_ratio=0.0, sync_probability=1.0 (conservative)")
    print("-" * 70)
    print(f"{'k':<3} | {'Comp(ms)':<10} | {'Paging(ms)':<12} | {'Sync(ms)':<10} | {'Total(ms)':<10} | {'vs k=1'}")
    print("-" * 70)
    
    cost_k1_old = hpa_cost(target_layer, 1, bandwidth, 
                           activation_split_ratio=0.0, sync_probability=1.0)
    
    for k in [1, 2, 4, 8]:
        cost = hpa_cost(target_layer, k, bandwidth,
                        activation_split_ratio=0.0, sync_probability=1.0)
        
        # Breakdown for display
        t_comp = target_layer.workload / (k ** 0.9)
        m_split = (target_layer.weight_memory + target_layer.bias_memory) / k + target_layer.activation_memory
        penalty = calculate_penalty(m_split)
        t_paging = (penalty - 1.0) * t_comp if penalty > 1.0 else 0.0
        
        if k > 1:
            sync_bytes = target_layer.output_bytes * 2 * (k - 1) / k
            sync_mb = sync_bytes / (1024 * 1024)
            t_sync = network_latency(sync_mb, bandwidth) * 1.0  # Full sync
        else:
            t_sync = 0.0
        
        ratio = f"{cost/cost_k1_old*100:.1f}%" if k > 1 else "baseline"
        print(f"{k:<3} | {t_comp:<10.2f} | {t_paging:<12.2f} | {t_sync:<10.2f} | {cost:<10.2f} | {ratio}")
    
    print("\n[NEW MODEL] activation_split_ratio=1.0, sync_probability=0.5 (optimized)")
    print("-" * 70)
    print(f"{'k':<3} | {'Comp(ms)':<10} | {'Paging(ms)':<12} | {'Sync(ms)':<10} | {'Total(ms)':<10} | {'vs k=1'}")
    print("-" * 70)
    
    cost_k1_new = hpa_cost(target_layer, 1, bandwidth,
                           activation_split_ratio=1.0, sync_probability=0.5)
    
    for k in [1, 2, 4, 8]:
        cost = hpa_cost(target_layer, k, bandwidth,
                        activation_split_ratio=1.0, sync_probability=0.5)
        
        # Breakdown for display (new model)
        t_comp = target_layer.workload / (k ** 0.9)
        m_activation_shard = target_layer.activation_memory / k  # Fully split
        m_split = (target_layer.weight_memory + target_layer.bias_memory) / k + m_activation_shard
        penalty = calculate_penalty(m_split)
        t_paging = (penalty - 1.0) * t_comp if penalty > 1.0 else 0.0
        
        if k > 1:
            sync_bytes = target_layer.output_bytes * 2 * (k - 1) / k
            sync_mb = sync_bytes / (1024 * 1024)
            t_sync = network_latency(sync_mb, bandwidth) * 0.5  # Amortized sync
        else:
            t_sync = 0.0
        
        ratio = f"{cost/cost_k1_new*100:.1f}%" if k > 1 else "baseline"
        print(f"{k:<3} | {t_comp:<10.2f} | {t_paging:<12.2f} | {t_sync:<10.2f} | {cost:<10.2f} | {ratio}")
    
    print("\n" + "=" * 80)
    print("CONCLUSION:")
    print(f"  - OLD model: k=2 achieves {hpa_cost(target_layer, 2, bandwidth, activation_split_ratio=0.0, sync_probability=1.0)/cost_k1_old*100:.1f}% of k=1 cost")
    print(f"  - NEW model: k=2 achieves {hpa_cost(target_layer, 2, bandwidth, activation_split_ratio=1.0, sync_probability=0.5)/cost_k1_new*100:.1f}% of k=1 cost")
    print("=" * 80)

if __name__ == "__main__":
    analyze_bert_layer()

