"""
Quick test script to verify MEDIA refactoring
Tests both linear (ViT) and parallel (InceptionV3) models
"""
from loader import ModelLoader
from common import Server
from alg_dina import DINAAlgorithm
from alg_media import MEDIAAlgorithm
from alg_ours import OursAlgorithm
from alg_occ import OCCAlgorithm

def test_model(model_path, model_name, n_servers, bandwidth_mbps):
    """Test a single model configuration"""
    print(f"\n{'='*60}")
    print(f"Model: {model_name}, Servers: {n_servers}, BW: {bandwidth_mbps} Mbps")
    print(f"{'='*60}")
    
    # Load model
    G, layers_map = ModelLoader.load_model_from_csv(model_path)
    servers = [Server(i, power_ratio=1.0) for i in range(n_servers)]
    
    # Run MEDIA
    media = MEDIAAlgorithm(G, layers_map, servers, bandwidth_mbps)
    parts_media = media.run()
    time_media = media.schedule(parts_media)
    
    # Run DINA
    dina = DINAAlgorithm(G, layers_map, servers, bandwidth_mbps)
    parts_dina = dina.run()
    time_dina = dina.schedule(parts_dina)
    
    # Run Ours
    ours = OursAlgorithm(G, layers_map, servers, bandwidth_mbps)
    parts_ours = ours.run()
    time_ours = ours.schedule(parts_ours)
    
    # Run OCC
    occ = OCCAlgorithm(G, layers_map, servers, bandwidth_mbps)
    parts_occ = occ.run()
    time_occ = occ.schedule(parts_occ)
    
    print(f"\nResults:")
    print(f"  MEDIA: {time_media:8.2f} ms ({len(parts_media)} partitions)")
    print(f"  DINA:  {time_dina:8.2f} ms ({len(parts_dina)} partitions)")
    print(f"  Ours:  {time_ours:8.2f} ms ({len(parts_ours)} partitions)")
    print(f"  OCC:   {time_occ:8.2f} ms ({len(parts_occ)} partitions)")
    
    return {
        'MEDIA': time_media,
        'DINA': time_dina,
        'Ours': time_ours,
        'OCC': time_occ
    }

def main():
    print("\n" + "="*60)
    print("Testing Refactored MEDIA Algorithm")
    print("="*60)
    
    # Test 1: Linear model (ViT-base)
    print("\n[Test 1] ViT-base (Linear Model)")
    vit_n1 = test_model('datasets/SafeDnnInferenceExp - ViT-base.csv', 'ViT-base', 1, 100)
    vit_n4 = test_model('datasets/SafeDnnInferenceExp - ViT-base.csv', 'ViT-base', 4, 100)
    
    # Test 2: Parallel model (InceptionV3)
    print("\n\n[Test 2] InceptionV3 (Parallel Model)")
    inc_n1 = test_model('datasets/SafeDnnInferenceExp - inceptionV3.csv', 'InceptionV3', 1, 100)
    inc_n4 = test_model('datasets/SafeDnnInferenceExp - inceptionV3.csv', 'InceptionV3', 4, 100)
    
    # Summary
    print("\n" + "="*60)
    print("Summary & Analysis")
    print("="*60)
    
    print("\n1. N=1 Validation (All should be similar to OCC):")
    print(f"   ViT-base N=1:     MEDIA={vit_n1['MEDIA']:.2f}, OCC={vit_n1['OCC']:.2f}")
    print(f"   InceptionV3 N=1:  MEDIA={inc_n1['MEDIA']:.2f}, OCC={inc_n1['OCC']:.2f}")
    
    print("\n2. Parallelism Check (N=4):")
    print(f"   ViT-base:     MEDIA={vit_n4['MEDIA']:.2f} vs Ours={vit_n4['Ours']:.2f}")
    print(f"   InceptionV3:  MEDIA={inc_n4['MEDIA']:.2f} vs Ours={inc_n4['Ours']:.2f}")
    
    print("\n3. Expected Behavior:")
    print("   • N=1: MEDIA ≈ DINA ≈ Ours ≈ OCC (single server baseline)")
    print("   • ViT N=4: MEDIA may not improve much (linear structure)")
    print("   • InceptionV3 N=4: MEDIA should now leverage parallelism better")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
