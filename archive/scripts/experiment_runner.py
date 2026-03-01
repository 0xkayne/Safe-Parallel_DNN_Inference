import os
import glob
import pandas as pd
from loader import ModelLoader
from common import Server
from alg_dina import DINAAlgorithm
from alg_media import MEDIAAlgorithm
from alg_ours import OursAlgorithm
from alg_occ import OCCAlgorithm

# Configuration
DATASETS_DIR = 'datasets'
OUTPUT_FILE = 'results_comparison.csv'

SERVER_COUNTS = [4]
# BANDWIDTHS = [10, 100, 1000] # Mbps
BANDWIDTHS = [100]

def run_experiment():
    # Find all CSV files
    csv_files = glob.glob(os.path.join(DATASETS_DIR, '*.csv'))
    
    results = []
    
    print(f"Found {len(csv_files)} datasets.")
    
    for csv_file in csv_files:
        model_name = os.path.basename(csv_file).replace('SafeDnnInferenceExp - ', '').replace('.csv', '')
        print(f"\nProcessing Model: {model_name}")
        
        # Load Model
        G, layers_map = ModelLoader.load_model_from_csv(csv_file)
        
        for n_servers in SERVER_COUNTS:
            for bw in BANDWIDTHS:
                print(f"  Configuration: Servers={n_servers}, BW={bw} Mbps")
                
                # Create Servers (Homogeneous for now, baseline power=1.0)
                servers = [Server(i, power_ratio=1.0) for i in range(n_servers)]
                
                # Run DINA
                dina = DINAAlgorithm(G, layers_map, servers, bw)
                parts_dina = dina.run()
                time_dina = dina.schedule(parts_dina)
                
                # Run MEDIA
                media = MEDIAAlgorithm(G, layers_map, servers, bw)
                parts_media = media.run()
                time_media = media.schedule(parts_media)
                
                # Run Ours
                ours = OursAlgorithm(G, layers_map, servers, bw)
                parts_ours = ours.run()
                time_ours = ours.schedule(parts_ours)
                
                # Run OCC (single-server paging baseline)
                occ = OCCAlgorithm(G, layers_map, servers, bw)
                parts_occ = occ.run()
                time_occ = occ.schedule(parts_occ)
                
                print(f"    -> DINA: {time_dina:.2f} ms")
                print(f"    -> MEDIA: {time_media:.2f} ms")
                print(f"    -> Ours: {time_ours:.2f} ms")
                print(f"    -> OCC: {time_occ:.2f} ms")
                
                results.append({
                    'Model': model_name,
                    'Servers': n_servers,
                    'Bandwidth_Mbps': bw,
                    'DINA_Latency': time_dina,
                    'MEDIA_Latency': time_media,
                    'Ours_Latency': time_ours,
                    'OCC_Latency': time_occ
                })
    
    # Save Results
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nExperiment Completed. Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    run_experiment()
