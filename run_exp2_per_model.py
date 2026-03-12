#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run Exp2 (network bandwidth ablation) per-model in separate subprocesses.
If one model segfaults, the rest continue.
"""

import os
import sys
import glob
import subprocess
import json

DATASETS_DIR = 'datasets_260120'
OUTPUT_DIR = 'exp_results/exp2_network_ablation'
BANDWIDTHS = [0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500]
SERVERS_FOR_NETWORK_EXP = 4
DEFAULT_SERVER_TYPE = "Xeon_IceLake"

MODEL_NAME_MAP = {
    'albert_base': 'ALBERT-base',
    'albert_large': 'ALBERT-large',
    'bert_base': 'BERT-base',
    'bert_large': 'BERT-large',
    'distilbert_base': 'DistillBERT-base',
    'distilbert_large': 'DistillBERT-large',
    'inceptionV3': 'InceptionV3',
    'InceptionV3': 'InceptionV3',
    'tinybert_4l': 'TinyBERT-4l',
    'tinybert_6l': 'TinyBERT-6l',
    'vit_base': 'ViT-base',
    'vit_large': 'ViT-large',
    'vit_small': 'ViT-small',
    'vit_tiny': 'ViT-tiny',
}


def run_single_model(csv_file):
    """Run exp2 for a single model (called as subprocess target)."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from loader import ModelLoader
    from common import Server
    from alg_dina import DINAAlgorithm
    from alg_media import MEDIAAlgorithm
    from alg_ours import OursAlgorithm
    from alg_occ import OCCAlgorithm
    import pandas as pd

    model_name = os.path.basename(csv_file).replace('.csv', '')
    short_name = MODEL_NAME_MAP.get(model_name, model_name)

    print(f"  Loading model: {short_name}")
    G, layers_map = ModelLoader.load_model_from_csv(csv_file)

    results = []
    for bw in BANDWIDTHS:
        servers = [Server(i, server_type=DEFAULT_SERVER_TYPE) for i in range(SERVERS_FOR_NETWORK_EXP)]

        occ = OCCAlgorithm(G, layers_map, servers, bw)
        dina = DINAAlgorithm(G, layers_map, servers, bw)
        media = MEDIAAlgorithm(G, layers_map, servers, bw)
        ours = OursAlgorithm(G, layers_map, servers, bw)

        row = {
            'Bandwidth(Mbps)': bw,
            'OCC': occ.schedule(occ.run()).latency,
            'DINA': dina.schedule(dina.run()).latency,
            'MEDIA': media.schedule(media.run()).latency,
            'Ours': ours.schedule(ours.run()).latency,
        }
        results.append(row)
        print(f"    BW={bw}: OCC={row['OCC']:.1f} DINA={row['DINA']:.1f} "
              f"MEDIA={row['MEDIA']:.1f} Ours={row['Ours']:.1f}")

    df = pd.DataFrame(results)
    output_file = os.path.join(OUTPUT_DIR, f'network_{short_name}.csv')
    df.to_csv(output_file, index=False)
    print(f"  [OK] Saved: {output_file}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_files = sorted(glob.glob(os.path.join(DATASETS_DIR, '*.csv')))

    # Allow skipping already-completed models via --skip flag
    skip_models = set()
    if '--skip' in sys.argv:
        idx = sys.argv.index('--skip')
        for arg in sys.argv[idx+1:]:
            if arg.startswith('--'):
                break
            skip_models.add(arg)

    print(f"Found {len(csv_files)} models to process")
    if skip_models:
        print(f"Skipping: {', '.join(skip_models)}")
    print("=" * 60)

    results_summary = {}

    for csv_file in csv_files:
        model_name = os.path.basename(csv_file).replace('.csv', '')
        short_name = MODEL_NAME_MAP.get(model_name, model_name)

        if short_name in skip_models or model_name in skip_models:
            print(f"\n[{short_name}] Skipped (already completed)")
            results_summary[short_name] = 'SKIPPED'
            continue

        print(f"\n[{short_name}] Running in subprocess...")

        try:
            result = subprocess.run(
                [sys.executable, __file__, '--single', csv_file],
                timeout=1800,  # 30 min timeout per model
                capture_output=False,
            )

            if result.returncode == 0:
                results_summary[short_name] = 'OK'
                print(f"[{short_name}] Completed successfully")
            elif result.returncode == -11:  # SIGSEGV
                results_summary[short_name] = 'SEGFAULT'
                print(f"[{short_name}] SEGFAULT (exit -11) — skipped")
            else:
                results_summary[short_name] = f'ERROR (exit {result.returncode})'
                print(f"[{short_name}] Failed with exit code {result.returncode}")
        except subprocess.TimeoutExpired:
            results_summary[short_name] = 'TIMEOUT'
            print(f"[{short_name}] TIMEOUT (>30 min) — skipped")

    print("\n" + "=" * 60)
    print("Summary:")
    for model, status in results_summary.items():
        print(f"  {model:20s} {status}")

    failed = [m for m, s in results_summary.items() if s != 'OK']
    if failed:
        print(f"\n[WARNING] {len(failed)} model(s) failed: {', '.join(failed)}")
    else:
        print("\n[OK] All models completed successfully!")


if __name__ == '__main__':
    if len(sys.argv) > 2 and sys.argv[1] == '--single':
        run_single_model(sys.argv[2])
    else:
        main()
