"""
OursAlgorithm: HPA (Hybrid Parallel Algorithm)
==============================================
5-Stage Pipeline for distributed DNN inference on SGX TEE edge clusters:

  Stage 0  COPA        – Cost-benefit candidate filtering (which ops to split)
  Stage 1  Cost Surface – hpa_cost() for every (operator, k) pair
  Stage 2  Config Select– Greedy pick best k per operator (fast & deterministic)
  Stage 3  Graph Augment– Split operators into shards + insert sync barriers
  Stage 4  MEDIA Partition– Merge shards into EPC-friendly partitions
  Stage 5  HEFT Schedule – List-schedule partitions onto heterogeneous servers

Key Design Points
-----------------
1. Tensor parallelism is ONLY beneficial when compute reduction outweighs
   sync overhead (AllReduce/AllGather + RTT). COPA (Stage 0) enforces this.
2. Graph augmentation (Stage 3) inserts explicit zero-workload barrier nodes
   so HEFT can correctly model cross-server AllReduce latency.
3. MEDIA partitioning (Stage 4) uses degree-constraint edge selection + EPC
   merge-check. It keeps shards from the same operator separate so they can
   be distributed across servers.
4. HEFT (Stage 5) is standard list scheduling with upward-rank priorities.
   A single-server safeguard guarantees Ours never under-performs OCC.
"""

import math
from typing import List, Dict, Tuple, Set
from copy import deepcopy

import networkx as nx

from common import (
    Partition, DNNLayer, ScheduleResult,
    EPC_EFFECTIVE_MB, calculate_penalty, network_latency, hpa_cost,
    is_conv_layer,
)


def _partition_cost(partition: Partition, power_ratio: float) -> float:
    """Execution cost under weights-inside-EPC model.

    Paging penalty is charged on total_memory (weights + activations).
    Distributed TP ensures each server holds only a fraction of the
    model, keeping most partitions within EPC and avoiding the OCALL /
    HMAC overhead that single-server OCC must pay.
    """
    return (partition.total_workload
            * calculate_penalty(partition.total_memory)) / power_ratio


class OursAlgorithm:
    """Main HPA algorithm class."""

    # Maximum partition workload (ms) — prevents lightweight layers from
    # accumulating into giant partitions that collapse CSP-style parallelism.
    MAX_PART_WL = 300

    # ── Constructor ────────────────────────────────────────────────────────────
    def __init__(self, G, layers_map, servers, bandwidth_mbps):
        self.G = G                          # original DAG
        self.layers_map = layers_map        # nid -> DNNLayer
        self.servers = servers
        self.bandwidth_mbps = bandwidth_mbps
        self.bandwidth_per_ms = (bandwidth_mbps / 8.0) / 1000.0

        # parallelism degrees to evaluate: k ∈ [1, n]
        n = len(servers)
        self.K_candidates = list(range(1, n + 1))
        self.benefit_threshold = 0.95       # need >= 5% latency reduction

        # Pre-compute bottleneck power for each k-way split (§4.4.3):
        # power of the k-th fastest server — the slowest shard in a
        # k-way split determines the makespan.
        sorted_pwrs = sorted([s.power_ratio for s in servers], reverse=True)
        self._tp_power = {k: sorted_pwrs[k-1]
                          for k in range(1, n + 1)}

        # Populated during run()
        self.G_aug = None
        self.layers_aug = None
        self.node_to_partition: Dict[int, Partition] = {}

    # ═══════════════════════════════════════════════════════════════════════════
    #  Top-level: 5-stage pipeline
    # ═══════════════════════════════════════════════════════════════════════════
    def run(self) -> List[Partition]:
        """Execute Stages 0-5 for TP and no-TP, return the lower-latency result.

        Dual evaluation: the TP configuration is always compared against a
        TP-disabled baseline (all k=1).  If TP does not reduce end-to-end
        makespan — e.g. due to increased inter-partition communication from
        finer-grained shards — the baseline is selected.  This guarantees
        that adding parallelism never degrades performance.
        """
        candidates = self._filter_candidates()
        cost_surface = self._build_cost_surface(candidates)
        cfg_tp = self._select_best_k(cost_surface, candidates)
        cfg_no_tp = {nid: 1 for nid in self.layers_map}

        best_parts, best_lat = None, float('inf')
        winner_g, winner_layers, winner_tp, winner_n2p = None, None, None, None

        for cfg in (cfg_tp, cfg_no_tp):
            G_aug, layers_aug, tp_origin = self._augment_graph(cfg)
            self.G_aug, self.layers_aug = G_aug, layers_aug
            self._tp_origin = tp_origin
            parts = self._media_partition(G_aug, layers_aug)
            result = self._schedule_impl(parts)
            if result.latency < best_lat:
                best_lat = result.latency
                best_parts = parts
                winner_g, winner_layers, winner_tp = G_aug, layers_aug, tp_origin
                winner_n2p = dict(self.node_to_partition)
                self._cached_schedule = result

        # Restore winner's state so schedule() sees the correct graph + mapping
        self.G_aug, self.layers_aug = winner_g, winner_layers
        self._tp_origin = winner_tp
        self.node_to_partition = winner_n2p
        return best_parts

    # ═══════════════════════════════════════════════════════════════════════════
    #  Stage 0 : COPA – Cost-benefit candidate filtering
    # ═══════════════════════════════════════════════════════════════════════════
    def _filter_candidates(self) -> Set[int]:
        """Return set of operator IDs that benefit from tensor parallelism.

        Uses server-aware power: k-way split evaluated with average of
        k fastest servers, reflecting the bottleneck shard's speed.
        """
        candidates = set()
        for nid, layer in self.layers_map.items():
            base = hpa_cost(layer, 1, self.bandwidth_mbps,
                            avg_power=self._tp_power.get(1, 1.0))
            best = min(
                (hpa_cost(layer, k, self.bandwidth_mbps,
                          avg_power=self._tp_power.get(k, 1.0))
                 for k in self.K_candidates[1:]),
                default=base,
            )
            if best < base * self.benefit_threshold:
                candidates.add(nid)
        return candidates

    # ═══════════════════════════════════════════════════════════════════════════
    #  Stage 1 : Cost Surface
    # ═══════════════════════════════════════════════════════════════════════════
    def _build_cost_surface(self, candidates: Set[int]) -> Dict[int, Dict[int, float]]:
        """For every node store latency under each valid k.

        Server-aware: each k uses avg power of k fastest servers.
        """
        surface: Dict[int, Dict[int, float]] = {}
        for nid, layer in self.layers_map.items():
            ks = self.K_candidates if nid in candidates else [1]
            surface[nid] = {
                k: hpa_cost(layer, k, self.bandwidth_mbps,
                            avg_power=self._tp_power.get(k, 1.0))
                for k in ks
            }
        return surface

    # ═══════════════════════════════════════════════════════════════════════════
    #  Stage 2 : Select best k per operator (greedy, fast, deterministic)
    # ═══════════════════════════════════════════════════════════════════════════
    def _select_best_k(self, cost_surface: Dict[int, Dict[int, float]],
                       candidates: Set[int]) -> Dict[int, int]:
        """For each operator pick the k with lowest hpa_cost().

        Tiebreaker: when two k have costs within 1%, prefer the smaller k
        (fewer shards = less fragmentation = fewer partition boundaries).
        This prevents k-inflation when adding more same-speed servers
        does not improve the actual bottleneck power.
        """
        cfg: Dict[int, int] = {}
        for nid in self.G.nodes():
            if nid in candidates:
                costs = cost_surface[nid]
                best_k = min(costs, key=costs.get)
                best_cost = costs[best_k]
                # Among all k within 1% of best, pick the smallest
                for k in sorted(costs):
                    if costs[k] <= best_cost * 1.01 and k < best_k:
                        best_k = k
                cfg[nid] = best_k
            else:
                cfg[nid] = 1
        return cfg

    # ═══════════════════════════════════════════════════════════════════════════
    #  Stage 3 : Graph Augmentation
    # ═══════════════════════════════════════════════════════════════════════════
    def _augment_graph(self, cfg: Dict[int, int]) -> Tuple[nx.DiGraph, Dict[int, DNNLayer], Dict[int, int]]:
        """Split k>1 operators into shards and insert explicit sync barriers.

        Without barrier nodes HEFT would see no edge between shard_i and the
        successor, silently ignoring AllReduce latency.  The barrier carries
        the sync volume as its incoming edge weight so HEFT accounts for it.

        Edge-weight convention (MB, matching loader.py):
            shard_i → barrier : per-shard sync volume
            barrier → succ    : original DAG edge weight (unchanged)
        """
        G_aug = nx.DiGraph()
        layers_aug: Dict[int, DNNLayer] = {}
        nxt_id = 0

        # Book-keeping for rewiring
        node_to_shards: Dict[int, List[int]] = {}
        node_to_barrier: Dict[int, int] = {}
        SYNC_P = 0.5  # must match hpa_cost() default

        # ── 3.1 Create shards + barriers ──────────────────────────────────────
        for orig, k in cfg.items():
            orig_layer = self.layers_map[orig]
            shards = []
            for _ in range(k):
                sid = nxt_id
                nxt_id += 1
                layers_aug[sid] = self._make_shard(orig_layer, sid, k)
                G_aug.add_node(sid)
                shards.append(sid)
            node_to_shards[orig] = shards

            if k > 1:
                bid = nxt_id
                nxt_id += 1
                # sync volume depends on layer type
                if is_conv_layer(orig_layer):
                    sync_bytes = orig_layer.output_bytes * (k - 1) / k
                else:
                    sync_bytes = orig_layer.output_bytes * 2 * (k - 1) / k
                per_shard_mb = (sync_bytes / (1024 * 1024) * SYNC_P) / k

                barrier = DNNLayer(
                    layer_id=bid,
                    name=f"{orig_layer.name}_sync",
                    memory=0.0, cpu_time=0.0, enclave_time=0.0,
                    output_bytes=orig_layer.output_bytes,
                    execution_mode=orig_layer.execution_mode,
                )
                layers_aug[bid] = barrier
                G_aug.add_node(bid)
                node_to_barrier[orig] = bid
                for sid in shards:
                    G_aug.add_edge(sid, bid, weight=per_shard_mb)

        # ── 3.2 Rewire inter-operator edges ───────────────────────────────────
        for u, v in self.G.edges():
            ku, kv = cfg[u], cfg[v]
            u_src = node_to_barrier[u] if ku > 1 else node_to_shards[u][0]
            v_shards = node_to_shards[v]
            w = self.G[u][v].get("weight", 0)

            if kv == 1:
                G_aug.add_edge(u_src, v_shards[0], weight=w)
            else:
                for vs in v_shards:
                    G_aug.add_edge(u_src, vs, weight=w / kv)

        # ── 3.3 TP origin map: which original op produced each shard ──────────
        # Used by _merge_check() to prevent collapsing TP-created parallelism.
        tp_origin: Dict[int, int] = {}
        for orig, k in cfg.items():
            if k > 1:
                for sid in node_to_shards[orig]:
                    tp_origin[sid] = orig

        return G_aug, layers_aug, tp_origin

    @staticmethod
    def _make_shard(orig: DNNLayer, new_id: int, k: int) -> DNNLayer:
        """Create a single shard of an operator."""
        return DNNLayer(
            layer_id=new_id,
            name=f"{orig.name}_shard_{new_id}",
            memory=(orig.weight_memory + orig.bias_memory) / k + orig.activation_memory,
            cpu_time=orig.cpu_time / k,
            enclave_time=orig.enclave_time / k,
            output_bytes=orig.output_bytes / k,
            execution_mode=orig.execution_mode,
            weight_memory=orig.weight_memory / k,
            bias_memory=orig.bias_memory / k,
            activation_memory=orig.activation_memory,
            encryption_overhead=orig.encryption_overhead / k,
            layer_type=orig.layer_type,
        )

    # ═══════════════════════════════════════════════════════════════════════════
    #  Stage 4 : MEDIA Partitioning
    # ═══════════════════════════════════════════════════════════════════════════
    def _media_partition(self, G: nx.DiGraph, layers_map: Dict[int, DNNLayer]) -> List[Partition]:
        """MEDIA-style greedy merge with degree constraint + EPC check."""
        # 4.1 initialise: every layer is its own partition
        self.node_to_partition = {
            nid: Partition(i, [layer], G)
            for i, (nid, layer) in enumerate(layers_map.items())
        }

        # 4.2 get candidate edges (MEDIA Constraint-1 + level check)
        edges_M = self._select_edges_for_partitioning(G)

        # 4.3 greedy merge by descending communication weight
        for u, v in sorted(edges_M, key=lambda e: G[e[0]][e[1]]["weight"], reverse=True):
            pu, pv = self.node_to_partition[u], self.node_to_partition[v]
            if pu == pv:
                continue
            if not self._would_cause_cycle(pu, pv, G) and self._merge_check(pu, pv, G):
                merged = sorted({l.id: l for l in pu.layers + pv.layers}.values(), key=lambda l: l.id)
                new_part = Partition(pu.id, merged, G)
                for l in merged:
                    self.node_to_partition[l.id] = new_part

        # 4.4 post-processing: merge adjacent partitions that fit in EPC
        self._post_merge_epc(G)

        # 4.5 renumber IDs sequentially
        unique = list({id(p): p for p in self.node_to_partition.values()}.values())
        for i, p in enumerate(unique):
            p.id = i
        return unique

    # ── 4.2 Candidate edge selection ──────────────────────────────────────────
    def _select_edges_for_partitioning(self, G: nx.DiGraph) -> Set[Tuple[int, int]]:
        """MEDIA Algorithm 1: select mergeable edges.

        Constraint 1: out_deg(u)==1 OR in_deg(v)==1
        Constraint 2: level-based fork protection (prevents collapsing
        parallel branches of equal length).
        """
        M: Set[Tuple[int, int]] = set()
        # topological levels for Constraint 2
        level = {}
        for lvl, nodes in enumerate(nx.topological_generations(G)):
            for n in nodes:
                level[n] = lvl

        for u in nx.topological_sort(G):
            for v in G.successors(u):
                # Constraint 1
                if len(self.servers) > 1:
                    if not (G.out_degree(u) == 1 or G.in_degree(v) == 1):
                        continue
                else:
                    if G.in_degree(v) != 1 and G.out_degree(u) != 1:
                        continue

                M.add((u, v))

                # Constraint 2: fork protection
                bad = False
                for w in G.successors(u):
                    for wp in G.predecessors(w):
                        if (wp, w) != (u, v) and (wp, w) in M:
                            if level.get(u, -1) == level.get(w, -2) - 1:
                                bad = True
                                break
                    if bad:
                        break
                if bad:
                    M.discard((u, v))
        return M

    # ── 4.3 Cycle detection ───────────────────────────────────────────────────
    def _would_cause_cycle(self, p1: Partition, p2: Partition, G: nx.DiGraph) -> bool:
        """True if merging p1 and p2 would create a cycle in the partition DAG."""
        pg = nx.DiGraph()
        seen = {p.id for p in self.node_to_partition.values()}
        pg.add_nodes_from(seen)
        for eu, ev in G.edges():
            a, b = self.node_to_partition[eu].id, self.node_to_partition[ev].id
            if a != b:
                pg.add_edge(a, b)
        return nx.has_path(pg, p2.id, p1.id)

    # ── 4.4 Merge check (MEDIA Check()) ──────────────────────────────────────
    def _merge_check(self, p1: Partition, p2: Partition, G: nx.DiGraph) -> bool:
        """Return True if merging p1 and p2 reduces (or keeps) total latency.

        Weights-inside-EPC model: paging penalty on total_memory.
        Case A uses cumulative sum as conservative merge heuristic.
        """
        merged = list(set(p1.layers + p2.layers))

        # ── TP boundary protection ──────────────────────────────────────────
        if hasattr(self, '_tp_origin') and self._tp_origin:
            orig_p1 = {self._tp_origin.get(l.id)
                       for l in p1.layers if l.id in self._tp_origin}
            orig_p2 = {self._tp_origin.get(l.id)
                       for l in p2.layers if l.id in self._tp_origin}
            if orig_p1 & orig_p2:
                return False

        tmp = Partition(-1, merged, G)

        # Hard cap on partition workload — prevents lightweight layers
        # (e.g. YOLOv5 backbone) from accumulating into giant partitions
        # that collapse CSP-style parallel branches.
        tmp = Partition(-1, merged, G)
        if tmp.total_workload > self.MAX_PART_WL:
            return False

        # Case A: cumulative memory fits in EPC → merge
        if sum(l.memory for l in merged) <= EPC_EFFECTIVE_MB:
            return True

        # Case B: compare merged vs separate cost
        avg_pwr = sum(s.power_ratio for s in self.servers) / len(self.servers)

        def part_time(part):
            return _partition_cost(part, avg_pwr)

        # inter-partition communication volume
        vol = sum(
            G[l1.id][l2.id]["weight"]
            for l1 in p1.layers for l2 in p2.layers
            if G.has_edge(l1.id, l2.id)
        ) + sum(
            G[l2.id][l1.id]["weight"]
            for l1 in p1.layers for l2 in p2.layers
            if G.has_edge(l2.id, l1.id)
        )
        t_comm = network_latency(vol, self.bandwidth_mbps) if vol > 0 and len(self.servers) > 1 else 0.0

        t_sep = part_time(p1) + part_time(p2) + t_comm
        t_mrg = part_time(tmp)
        return t_mrg <= t_sep

    # ── 4.5 Post-merge: greedily merge neighbours that fit in EPC ─────────────
    def _post_merge_epc(self, G: nx.DiGraph):
        """Repeatedly merge adjacent partitions while total_memory <= EPC.

        Uses total_memory (including weights) as a conservative heuristic to
        prevent over-merging of parallel branches.
        """
        while True:
            parts = list({id(p): p for p in self.node_to_partition.values()}.values())
            # collect candidate pairs sorted by descending comm weight
            cand = []
            for i, a in enumerate(parts):
                for b in parts[i + 1:]:
                    comm = sum(
                        G[x.id][y.id]["weight"]
                        for x in a.layers for y in b.layers
                        if G.has_edge(x.id, y.id)
                    ) + sum(
                        G[y.id][x.id]["weight"]
                        for x in a.layers for y in b.layers
                        if G.has_edge(y.id, x.id)
                    )
                    if comm == 0:
                        continue
                    merged = sorted({l.id: l for l in a.layers + b.layers}.values(), key=lambda l: l.id)
                    tmp = Partition(-1, merged, G)
                    # Conservative: cumulative sum prevents over-merging
                    if sum(l.memory for l in merged) > EPC_EFFECTIVE_MB:
                        continue
                    # Also enforce workload cap (same as _merge_check)
                    if tmp.total_workload > self.MAX_PART_WL:
                        continue
                    cand.append((comm, a, b, merged))

            if not cand:
                break

            cand.sort(key=lambda x: -x[0])
            changed = False
            for _, a, b, merged in cand:
                if self.node_to_partition.get(a.layers[0].id) is not a:
                    continue
                if self.node_to_partition.get(b.layers[0].id) is not b:
                    continue
                if not self._would_cause_cycle(a, b, G):
                    new_p = Partition(a.id, merged, G)
                    for l in merged:
                        self.node_to_partition[l.id] = new_p
                    changed = True
                    break
            if not changed:
                break

    # ═══════════════════════════════════════════════════════════════════════════
    #  Stage 5 : HEFT Scheduling
    # ═══════════════════════════════════════════════════════════════════════════
    def schedule(self, partitions: List[Partition] = None) -> ScheduleResult:
        """Return cached result from run(), or compute fresh if needed."""
        if partitions is None and hasattr(self, '_cached_schedule'):
            return self._cached_schedule
        return self._schedule_impl(partitions)

    def _schedule_impl(self, partitions: List[Partition]) -> ScheduleResult:
        """List-schedule partitions using HEFT upward-rank priorities."""
        if not partitions:
            return ScheduleResult("Ours(HPA)", 0.0, {}, [])

        # build partition DAG
        pg = nx.DiGraph()
        for p in partitions:
            pg.add_node(p.id)
        for u, v in self.G_aug.edges():
            pu, pv = self.node_to_partition[u], self.node_to_partition[v]
            if pu.id != pv.id:
                pg.add_edge(pu.id, pv.id)

        part_list = {p.id: p for p in partitions}
        prio = self._compute_priorities(pg, part_list)

        # tie-break by reverse topological index
        try:
            topo = list(nx.topological_sort(pg))
        except nx.NetworkXUnfeasible:
            topo = list(pg.nodes())
        topo_idx = {pid: i for i, pid in enumerate(topo)}
        order = sorted(partitions, key=lambda p: (prio[p.id], -topo_idx[p.id]), reverse=True)

        # HEFT state
        free = {s.id: 0.0 for s in self.servers}
        sched = {s.id: [] for s in self.servers}
        assign, finish = {}, {}

        for p in order:
            best_s, best_ft, best_start = None, float("inf"), 0.0

            for s in self.servers:
                # data-ready = max over preds (local or remote)
                ready = 0.0
                for pred in pg.predecessors(p.id):
                    if pred not in assign:
                        continue
                    ps, pf = assign[pred], finish[pred]
                    if ps.id != s.id:
                        comm = sum(
                            self.G_aug[x.id][y.id]["weight"]
                            for x in part_list[pred].layers for y in p.layers
                            if self.G_aug.has_edge(x.id, y.id)
                        )
                        ready = max(ready, pf + network_latency(comm, self.bandwidth_mbps))
                    else:
                        ready = max(ready, pf)

                start = max(free[s.id], ready)
                exec_t = _partition_cost(p, s.power_ratio)
                ft = start + exec_t
                if ft < best_ft:
                    best_ft, best_s, best_start = ft, s, start

            assign[p.id], finish[p.id] = best_s, best_ft
            free[best_s.id] = best_ft
            sched[best_s.id].append({
                "start": best_start,
                "end": best_ft,
                "partition_id": p.id,
                "partition": p,
            })

        heft_lat = max(finish.values())

        # ── Safeguard: never worse than best single-server sequential ─────────
        # Single-server safeguard: never worse than actual OCC.
        fallback = self._occ_safeguard(heft_lat, partitions, topo, part_list)
        if fallback is not None:
            return fallback

        return ScheduleResult("Ours(HPA)", heft_lat, sched, partitions)

    # ── Helper: single-server lower bound ────────────────────────────────────
    def _occ_safeguard(self, heft_lat, partitions, topo_order, part_list):
        """If HEFT is worse than actual OCC, fall back to OCC's result.

        OCC is the single-server ground truth — our distributed method
        must never be slower than just running everything on one server.
        """
        from alg_occ import OCCAlgorithm
        occ = OCCAlgorithm(self.G, self.layers_map, self.servers, self.bandwidth_mbps)
        occ_lat = occ.schedule(occ.run()).latency
        if occ_lat < heft_lat:
            return self._build_single_server_result(partitions, topo_order, part_list, occ_lat)
        return None

    def _single_server_time(self, partitions, topo_order, part_list) -> float:
        """Estimate sequential time on the fastest server alone."""
        best_s = max(self.servers, key=lambda s: s.power_ratio)
        pw = best_s.power_ratio
        return sum(_partition_cost(part_list[pid], pw) for pid in topo_order)

    def _build_single_server_result(self, partitions, topo_order, part_list, total_time):
        """Construct ScheduleResult for the single-server fallback."""
        best_s = max(self.servers, key=lambda s: s.power_ratio)
        sched = {s.id: [] for s in self.servers}
        pw = best_s.power_ratio
        t = 0.0
        for pid in topo_order:
            p = part_list[pid]
            cost = _partition_cost(p, pw)
            sched[best_s.id].append({
                "start": t, "end": t + cost,
                "partition_id": pid, "partition": p,
            })
            t += cost
        return ScheduleResult("Ours(HPA)", total_time, sched, partitions)

    # ── Helper: upward-rank priority (HEFT) ───────────────────────────────────
    def _compute_priorities(self, pg: nx.DiGraph, part_list: Dict[int, Partition]) -> Dict[int, float]:
        """Priority = exec_time + max_succ_comm + max_succ_priority.

        Uses total_memory penalty (weights-inside-EPC model).
        """
        avg_pwr = sum(s.power_ratio for s in self.servers) / len(self.servers)
        try:
            topo = list(nx.topological_sort(pg))
        except nx.NetworkXUnfeasible:
            topo = list(pg.nodes())
        prio: Dict[int, float] = {}
        for pid in reversed(topo):
            p = part_list[pid]
            t_exec = _partition_cost(p, avg_pwr)
            succs = list(pg.successors(pid))
            if not succs:
                prio[pid] = t_exec
            else:
                max_sp = max(prio.get(s, 0.0) for s in succs)
                comm = sum(
                    self.G_aug[x.id][y.id]["weight"]
                    for s in succs
                    for x in p.layers for y in part_list[s].layers
                    if self.G_aug.has_edge(x.id, y.id)
                )
                prio[pid] = t_exec + comm / self.bandwidth_per_ms + max_sp
        return prio
