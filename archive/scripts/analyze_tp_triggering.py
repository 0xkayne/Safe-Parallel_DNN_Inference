
import os
import sys
import glob
sys.path.insert(0, '.')
from loader import ModelLoader
from common import Server, EPC_EFFECTIVE_MB
from alg_ours import OursAlgorithm

def analyze_models():
    dataset_dir = 'datasets_260120'
    models = glob.glob(os.path.join(dataset_dir, '*.csv'))
    
    print(f"{'Model':<20} | {'EPC':<6} | {'Cost-Ben Cand':<13} | {'Splits':<6} | {'Max Layer(MB)':<13} | {'Triggered?'}")
    print("-" * 90)
    
    servers = [Server(i) for i in range(4)]
    bandwidth = 500 # Mbps
    
    for model_path in sorted(models):
        model_name = os.path.basename(model_path).replace('.csv', '')
        
        try:
            # Load model
            G, lm = ModelLoader.load_model_from_csv(model_path)
            
            # Find max layer size
            max_mem = max(l.memory for l in lm.values())
            
            # Run Ours Analysis (Capture internal state without printing everything)
            ours = OursAlgorithm(G, lm, servers, bandwidth)
            
            # 1. Filter Candidates
            # Capture stdout to silence it
            import io
            from contextlib import redirect_stdout
            
            f = io.StringIO()
            with redirect_stdout(f):
                candidates = ours._filter_candidates_by_cost_benefit()
                cost_surface = ours._build_cost_surface(candidates)
                optimal_cfg = ours._dag_dp(cost_surface, candidates)
            
            splits = sum(1 for k in optimal_cfg.values() if k > 1)
            triggered = "YES" if splits > 0 else "NO"
            
            print(f"{model_name:<20} | {EPC_EFFECTIVE_MB:<6.1f} | {len(candidates):<13} | {splits:<6} | {max_mem:<13.2f} | {triggered}")
            
        except Exception as e:
            print(f"{model_name:<20} | ERROR: {str(e)}")

if __name__ == "__main__":
    analyze_models()
