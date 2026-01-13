"""
Test script for exhaustive search Ours algorithm
Test on small models to verify correctness
"""
import sys
import time
from loader import ModelLoader
from common import Server
from alg_ours import OursAlgorithm
from alg_dina import DINAAlgorithm
from alg_occ import OCCAlgorithm
from alg_media import MEDIAAlgorithm

def test_small_model():
    """Test on smallest model first"""
    print("="*70)
    print("Testing Exhaustive Search Ours Algorithm")
    print("="*70)
    
    # Test on DistillBERT (smallest production model)
    models = [
        ('datasets/SafeDnnInferenceExp - DistillBERT.csv', 'DistillBERT'),
        # Only add more if first succeeds
    ]
    
    for model_path, model_name in models:
        print(f"\n{'='*70}")
        print(f"Model: {model_name}, Servers: 2, Bandwidth: 100 Mbps")
        print(f"{'='*70}")
        
        # Load model
        G, layers_map = ModelLoader.load_model_from_csv(model_path)
        servers = [Server(i, power_ratio=1.0) for i in range(2)]
        
        print(f"Model info: {G.number_of_nodes()} layers, {G.number_of_edges()} edges")
        
        # Run Ours (Exhaustive)
        print(f"\n[1/4] Running Ours (Exhaustive Search)...")
        start = time.time()
        try:
            ours = OursAlgorithm(G, layers_map, servers, bandwidth_mbps=100)
            partitions_ours = ours.run()
            time_ours = ours.schedule(partitions_ours)
            elapsed_ours = time.time() - start
            print(f"  ✓ Ours: {time_ours:.2f} ms ({len(ours.optimal_partitions)} partitions)")
            print(f"  Time taken: {elapsed_ours:.2f}s")
        except Exception as e:
            print(f"  ✗ Ours failed: {e}")
            import traceback
            traceback.print_exc()
            return
        
        # Run DINA for comparison
        print(f"\n[2/4] Running DINA...")
        start = time.time()
        dina = DINAAlgorithm(G, layers_map, servers, bandwidth_mbps=100)
        partitions_dina = dina.run()
        time_dina = dina.schedule(partitions_dina)
        elapsed_dina = time.time() - start
        print(f"  ✓ DINA: {time_dina:.2f} ms ({len(partitions_dina)} partitions)")
        print(f"  Time taken: {elapsed_dina:.2f}s")
        
        # Run OCC for comparison
        print(f"\n[3/4] Running OCC...")
        start = time.time()
        occ = OCCAlgorithm(G, layers_map, servers, bandwidth_mbps=100)
        partitions_occ = occ.run()
        time_occ = occ.schedule(partitions_occ)
        elapsed_occ = time.time() - start
        print(f"  ✓ OCC: {time_occ:.2f} ms ({len(partitions_occ)} partitions)")
        print(f"  Time taken: {elapsed_occ:.2f}s")
        
        # Run MEDIA for comparison
        print(f"\n[4/4] Running MEDIA...")
        start = time.time()
        media = MEDIAAlgorithm(G, layers_map, servers, bandwidth_mbps=100)
        partitions_media = media.run()
        time_media = media.schedule(partitions_media)
        elapsed_media = time.time() - start
        print(f"  ✓ MEDIA: {time_media:.2f} ms ({len(partitions_media)} partitions)")
        print(f"  Time taken: {elapsed_media:.2f}s")
        
        # Summary
        print(f"\n{'='*70}")
        print(f"Results Summary for {model_name}")
        print(f"{'='*70}")
        print(f"  Ours (Exhaustive):  {time_ours:8.2f} ms  ({elapsed_ours:6.2f}s)")
        print(f"  DINA:               {time_dina:8.2f} ms  ({elapsed_dina:6.2f}s)")
        print(f"  OCC:                {time_occ:8.2f} ms  ({elapsed_occ:6.2f}s)")
        print(f"  MEDIA:              {time_media:8.2f} ms  ({elapsed_media:6.2f}s)")
        
        # Verification
        print(f"\n{'='*70}")
        print(f"Verification")
        print(f"{'='*70}")
        
        min_other = min(time_dina, time_occ, time_media)
        if time_ours <= min_other * 1.01:  # Allow 1% tolerance
            print(f"  ✓ PASS: Ours ({time_ours:.2f}) ≤ min of others ({min_other:.2f})")
            print(f"  Ours is optimal or near-optimal!")
        else:
            print(f"  ✗ FAIL: Ours ({time_ours:.2f}) > min of others ({min_other:.2f})")
            print(f"  Gap: {((time_ours/min_other - 1) * 100):.2f}%")
        
        print(f"\n{'='*70}\n")

if __name__ == "__main__":
    test_small_model()
