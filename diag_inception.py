#!/usr/bin/env python3
"""Diagnostic script for InceptionV3: OCC vs MEDIA detailed comparison."""

import sys
sys.path.insert(0, '/home/kayne/code/Safe-Parallel_DNN_Inference')

from loader import ModelLoader
from common import (Server, EPC_EFFECTIVE_MB, calculate_penalty, network_latency)
from alg_occ import OCCAlgorithm
from alg_media import MEDIAAlgorithm

# ── Load model ──────────────────────────────────────────────────────
csv_path = "datasets_260120/InceptionV3.csv"
G, layers_map = ModelLoader.load_model_from_csv(csv_path)
print(f"=== InceptionV3 Model ===")
print(f"  Layers: {len(layers_map)}")
print(f"  Edges:  {G.number_of_edges()}")
total_workload = sum(l.workload for l in layers_map.values())
total_memory = sum(l.memory for l in layers_map.values())
total_output = sum(l.output_bytes for l in layers_map.values()) / (1024*1024)
print(f"  Total workload: {total_workload:.2f} ms")
print(f"  Total memory (sum of all layers): {total_memory:.2f} MB")
print(f"  Total output_bytes (sum): {total_output:.2f} MB")
print(f"  EPC effective: {EPC_EFFECTIVE_MB} MB")
print()

# ── Setup servers ───────────────────────────────────────────────────
servers = [Server(i, "Xeon_IceLake") for i in range(4)]
bw = 100  # Mbps

def print_partition_details(partitions, algo_name, G):
    print(f"\n{'='*70}")
    print(f"  {algo_name}: {len(partitions)} partitions")
    print(f"{'='*70}")

    for p in sorted(partitions, key=lambda x: x.id):
        # Cumulative output_bytes sum (what _calculate_peak_activation does)
        cum_output = sum(l.output_bytes / (1024*1024) for l in p.layers)
        # Persistent (weights + bias + encryption)
        persistent = sum(l.weight_memory + l.bias_memory + l.encryption_overhead for l in p.layers)
        # Penalty
        penalty = calculate_penalty(p.total_memory)

        layer_names = [l.name for l in p.layers]
        # Truncate display if too many layers
        if len(layer_names) > 8:
            display_names = ', '.join(layer_names[:4]) + f' ... ({len(layer_names)-8} more) ... ' + ', '.join(layer_names[-4:])
        else:
            display_names = ', '.join(layer_names)

        print(f"\n  Partition {p.id}:")
        print(f"    Layers: {len(p.layers)}")
        print(f"    Layer names: {display_names}")
        print(f"    total_memory (peak):  {p.total_memory:.3f} MB")
        print(f"      persistent (W+B+E): {persistent:.3f} MB")
        print(f"      cum_output_bytes:   {cum_output:.3f} MB")
        print(f"    total_workload:       {p.total_workload:.3f} ms")
        print(f"    penalty:              {penalty:.3f}")
        print(f"    penalized_workload:   {p.total_workload * penalty:.3f} ms")
        if p.assigned_server is not None:
            print(f"    assigned_server:      {p.assigned_server}")
        print(f"    [start, end]:         [{p.start_time:.3f}, {p.finish_time:.3f}]")


# ── OCC ─────────────────────────────────────────────────────────────
occ = OCCAlgorithm(G, layers_map, servers, bw)
occ_parts = occ.run()
occ_result = occ.schedule(occ_parts)

print_partition_details(occ_result.partitions, "OCC", G)
print(f"\n  >>> OCC Total Latency: {occ_result.latency:.3f} ms")

# Print OCC server schedule
print(f"\n  OCC Server Schedule:")
for sid, events in occ_result.server_schedules.items():
    if events:
        print(f"    Server {sid}:")
        for ev in events:
            print(f"      Part {ev['partition_id']}: [{ev['start']:.3f}, {ev['end']:.3f}] ({ev['end']-ev['start']:.3f} ms)")

# ── MEDIA ───────────────────────────────────────────────────────────
media = MEDIAAlgorithm(G, layers_map, servers, bw)
media_parts = media.run()
media_result = media.schedule(media_parts)

# Update partition info from schedule result
part_map = {p.id: p for p in media_result.partitions}
for sid, events in media_result.server_schedules.items():
    for ev in events:
        pid = ev['partition_id']
        if pid in part_map:
            part_map[pid].assigned_server = sid
            part_map[pid].start_time = ev['start']
            part_map[pid].finish_time = ev['end']

print_partition_details(media_result.partitions, "MEDIA", G)
print(f"\n  >>> MEDIA Total Latency: {media_result.latency:.3f} ms")

# Print MEDIA server schedule
print(f"\n  MEDIA Server Schedule:")
for sid, events in media_result.server_schedules.items():
    if events:
        print(f"    Server {sid}:")
        for ev in sorted(events, key=lambda e: e['start']):
            p = ev['partition']
            penalty = calculate_penalty(p.total_memory)
            print(f"      Part {ev['partition_id']}: [{ev['start']:.3f}, {ev['end']:.3f}] "
                  f"({ev['end']-ev['start']:.3f} ms, penalty={penalty:.3f})")

# ── MEDIA Latency Breakdown ────────────────────────────────────────
print(f"\n{'='*70}")
print(f"  MEDIA Latency Breakdown")
print(f"{'='*70}")

total_compute = 0.0
total_paging_overhead = 0.0
total_comm = 0.0
partitions_exceeding_epc = 0

for p in media_result.partitions:
    penalty = calculate_penalty(p.total_memory)
    base_compute = p.total_workload  # raw compute (no penalty, no server scaling)
    paging_extra = p.total_workload * (penalty - 1.0) if penalty > 1.0 else 0.0
    total_compute += base_compute
    total_paging_overhead += paging_extra
    if p.total_memory > EPC_EFFECTIVE_MB:
        partitions_exceeding_epc += 1

# Communication: count all cross-partition edges
for u, v in G.edges():
    pu = media.node_to_partition[u]
    pv = media.node_to_partition[v]
    if id(pu) != id(pv):
        comm_mb = G[u][v]['weight']
        total_comm += network_latency(comm_mb, bw)

print(f"  Partitions exceeding EPC: {partitions_exceeding_epc}/{len(media_result.partitions)}")
print(f"  Raw compute (sum):     {total_compute:.3f} ms")
print(f"  Paging overhead (sum): {total_paging_overhead:.3f} ms")
print(f"  Communication (sum):   {total_comm:.3f} ms")
print(f"  Actual e2e latency:    {media_result.latency:.3f} ms")
print(f"  (Note: e2e != sum because of parallelism across {len(servers)} servers)")

# Per-partition paging detail
print(f"\n  Per-Partition Paging Detail:")
for p in sorted(media_result.partitions, key=lambda x: x.id):
    penalty = calculate_penalty(p.total_memory)
    overflow = max(0, p.total_memory - EPC_EFFECTIVE_MB)
    print(f"    Part {p.id}: mem={p.total_memory:.2f} MB, overflow={overflow:.2f} MB, "
          f"penalty={penalty:.3f}, workload={p.total_workload:.3f} ms, "
          f"penalized={p.total_workload*penalty:.3f} ms")

# ── Comparison Summary ──────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"  Summary Comparison")
print(f"{'='*70}")
print(f"  OCC:   {occ_result.latency:.3f} ms  ({len(occ_result.partitions)} partitions, single-server)")
print(f"  MEDIA: {media_result.latency:.3f} ms  ({len(media_result.partitions)} partitions, {len(servers)} servers)")
ratio = occ_result.latency / media_result.latency if media_result.latency > 0 else 0
print(f"  OCC/MEDIA ratio: {ratio:.2f}x")
