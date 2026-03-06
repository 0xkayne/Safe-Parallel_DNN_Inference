"""P_sync ablation: compare P_sync=0.5 (default) vs P_sync=1.0 (conservative)."""
import sys, io
sys.path.insert(0, '.')
import common
from loader import ModelLoader
from common import Server
from alg_ours import OursAlgorithm
from alg_occ import OCCAlgorithm
import alg_ours as _ao

orig_hpa_cost = common.hpa_cost


def hpa_cost_p1(layer, k, bw, efficiency_gamma=0.9,
                activation_split_ratio=1.0, sync_probability=0.5):
    """Force sync_probability=1.0 regardless of caller."""
    return orig_hpa_cost(layer, k, bw, efficiency_gamma, activation_split_ratio, 1.0)


class OursP1(OursAlgorithm):
    """OursAlgorithm with P_sync=1.0."""

    def _filter_candidates_by_cost_benefit(self):
        _ao.hpa_cost = hpa_cost_p1
        try:
            return super()._filter_candidates_by_cost_benefit()
        finally:
            _ao.hpa_cost = orig_hpa_cost

    def _build_cost_surface(self, candidates):
        _ao.hpa_cost = hpa_cost_p1
        try:
            return super()._build_cost_surface(candidates)
        finally:
            _ao.hpa_cost = orig_hpa_cost

    def _augment_graph(self, optimal_cfg):
        G_aug, layers_aug = super()._augment_graph(optimal_cfg)
        # super() used SYNC_PROBABILITY=0.5; double AllReduce incoming edges for P_sync=1.0
        allreduce_nodes = {nid for nid, layer in layers_aug.items()
                          if '_allreduce' in layer.name}
        for nid in allreduce_nodes:
            for pred in list(G_aug.predecessors(nid)):
                if G_aug.has_edge(pred, nid):
                    G_aug[pred][nid]['weight'] = G_aug[pred][nid].get('weight', 0) * 2.0
        return G_aug, layers_aug


def run_silent(cls, G, lm, servers, bw):
    """Run algorithm, suppress HPA prints, return (latency, k_gt1_count)."""
    buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    try:
        alg = cls(G, lm, servers, bw)
        captured_cfg = {}
        orig_dp = alg._dag_dp
        def pdp(cs, ca):
            r = orig_dp(cs, ca)
            captured_cfg.update(r)
            return r
        alg._dag_dp = pdp
        parts = alg.run()
        result = alg.schedule(parts)
    finally:
        sys.stdout = old_stdout
    k_gt1 = sum(1 for v in captured_cfg.values() if v > 1)
    return result.latency, k_gt1


models = [
    ('datasets_260120/InceptionV3.csv', 'InceptionV3'),
    ('datasets_260120/bert_large.csv',  'BERT-large'),
    ('datasets_260120/bert_base.csv',   'BERT-base'),
]
bws = [5, 10, 50, 100, 200, 500]

for csvf, mname in models:
    print(f"\n{'='*70}")
    G, lm = ModelLoader.load_model_from_csv(csvf)
    servers4 = lambda: [Server(i, 'Xeon_IceLake') for i in range(4)]
    occ_lat = OCCAlgorithm(G, lm, servers4(), 100).schedule(
                  OCCAlgorithm(G, lm, servers4(), 100).run()).latency
    print(f"{mname}  OCC_baseline={occ_lat:.0f}ms")
    print(f"{'BW':>6} | {'P0.5':>9} | {'P0.5/OCC':>8} | {'P1.0':>9} | {'P1.0/OCC':>8} | "
          f"{'k(0.5)':>6} | {'k(1.0)':>6} | diff%")
    print("-" * 80)

    for bw in bws:
        lat05, k05 = run_silent(OursAlgorithm, G, lm, servers4(), bw)
        lat10, k10 = run_silent(OursP1,        G, lm, servers4(), bw)
        diff_pct = (lat10 - lat05) / lat05 * 100 if lat05 > 0 else 0
        print(f"{bw:6.0f} | {lat05:9.0f} | {lat05/occ_lat:8.3f} | "
              f"{lat10:9.0f} | {lat10/occ_lat:8.3f} | "
              f"{k05:6d} | {k10:6d} | {diff_pct:+.1f}%")

print("\nDone.")
