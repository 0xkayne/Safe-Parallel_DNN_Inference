import networkx as nx
import math
import copy
from common import Partition, EPC_EFFECTIVE_MB, calculate_penalty, PAGE_SIZE_KB, PAGE_FAULT_OVERHEAD_MS, ENCLAVE_ENTRY_EXIT_OVERHEAD_MS, DEFAULT_PAGING_BW_MBPS, ScheduleResult

class OursAlgorithm:
    """
    Advanced Parallelism-Aware Optimization Algorithm.
    Transcends MEDIA by explicitly balancing parallelism gain against TEE/Network overheads.
    """
    def __init__(self, G, layers_map, servers, bandwidth_mbps):
        self.G = G
        self.layers_map = layers_map
        self.servers = servers
        self.bandwidth_mbps = bandwidth_mbps
        self.bandwidth_per_ms = (bandwidth_mbps / 8.0) / 1000.0
        self.paging_bw_per_ms = DEFAULT_PAGING_BW_MBPS / 1000.0
        
        # Expanded Search parameters
        self.BEAM_WIDTH = 8
        self.MAX_ITERATIONS = 20
        self.is_large = G.number_of_nodes() > 100
        self.MAX_PARTITIONS_PER_SEARCH = 200

    def run(self):
        """
        Transcendent execution flow:
        1. Layer Rank Analysis.
        2. Diversified Seed Generation.
        3. Cycle Repair (Ensures all seeds are DAG-legal).
        4. Beam Search to refine and outperform.
        """
        up_ranks = self._compute_layer_ranks()

        seeds = []
        seeds.append(self._generate_linear_clustering())
        if len(self.servers) > 1:
            seeds.append(self._generate_parallel_aware_clustering(up_ranks))

        from alg_media import MEDIAAlgorithm
        media_alg = MEDIAAlgorithm(self.G, self.layers_map, self.servers, self.bandwidth_mbps)
        seeds.append(media_alg.run())

        # Step 3: Cycle Repair (Critical for leveraging MEDIA seeds)
        fixed_seeds = []
        for i, s_list in enumerate(seeds):
            fixed_seeds.append(self._repair_seed_cycles(s_list))

        # [Ours Debug]
        print(f"  [Ours Debug] Evaluating {len(fixed_seeds)} seeds...")
        for i, s_list in enumerate(fixed_seeds):
            try:
                eval_res = self.schedule(s_list)
                print(f"    Seed {i} Latency: {eval_res.latency:.2f} ms")
            except Exception as e:
                print(f"    Seed {i} Failed: {str(e)}")

        best_partitions = self._search_optimized(fixed_seeds)
        for i, p in enumerate(best_partitions): p.id = i
        return best_partitions

    def _repair_seed_cycles(self, partitions):
        """
        If a partition graph has cycles, collapse partitions in the same SCC.
        This fixes 'buggy' seeds from other algorithms while keeping their benefits.
        """
        import networkx as nx
        while True:
            node_to_p = {l.id: p for p in partitions for l in p.layers}
            pg = nx.DiGraph()
            for u, v in self.G.edges():
                pu, pv = node_to_p[u], node_to_p[v]
                if pu != pv: pg.add_edge(pu.id, pv.id)
            
            try:
                cycles = list(nx.simple_cycles(pg))
                if not cycles: break
                
                # Take first cycle and collapse all involved partitions
                involved_pids = set(cycles[0])
                to_merge = [p for p in partitions if p.id in involved_pids]
                
                new_layers = []
                for p in to_merge: new_layers.extend(p.layers)
                
                # Replace them in the list
                remaining = [p for p in partitions if p.id not in involved_pids]
                new_p = Partition(min(involved_pids), new_layers, self.G)
                remaining.append(new_p)
                partitions = remaining
                # Re-index remaining for sanity
                for i, p in enumerate(partitions): p.id = i
            except Exception:
                break # Heuristic protection
        return partitions

    def _compute_layer_ranks(self):
        """Calculates Upward Rank for each layer (distance to exit)."""
        ranks = {}
        avg_power = sum(s.power_ratio for s in self.servers) / len(self.servers)
        
        for node in reversed(list(nx.topological_sort(self.G))):
            layer = self.layers_map[node]
            # Internal execution cost estimate
            w = layer.workload / avg_power
            
            max_succ = 0
            for succ in self.G.successors(node):
                comm = self.G[node][succ]['weight'] / self.bandwidth_per_ms
                max_succ = max(max_succ, comm + ranks[succ])
            ranks[node] = w + max_succ
        return ranks

    def _generate_linear_clustering(self):
        """
        Seed 1: Aggressively pack layers in topological order.
        Willing to exceed EPC if savings in comms/paging/exit-enclave > penalty.
        """
        topo_order = list(nx.topological_sort(self.G))
        partitions = []
        current_part_layers = []
        
        def get_internal_cost(layers):
            if not layers: return 0
            p = Partition(-1, layers, self.G)
            penalty = calculate_penalty(p.total_memory)
            return (p.total_workload * penalty) + ENCLAVE_ENTRY_EXIT_OVERHEAD_MS

        for nid in topo_order:
            layer = self.layers_map[nid]
            if not current_part_layers:
                current_part_layers.append(layer)
                continue
            
            # Cost Analysis:
            # 1. Split
            cost_split = (get_internal_cost(current_part_layers) + 
                          get_internal_cost([layer]))
            # 2. Merged
            cost_merged = get_internal_cost(current_part_layers + [layer])
            
            # Decision Logic
            test_p = Partition(-1, current_part_layers + [layer], self.G)
            if test_p.total_memory <= EPC_EFFECTIVE_MB:
                current_part_layers.append(layer)
            elif cost_merged < cost_split * 1.05: # Allow slight overhead for better packing
                current_part_layers.append(layer)
            else:
                partitions.append(Partition(len(partitions), current_part_layers, dag=self.G))
                current_part_layers = [layer]
        
        if current_part_layers:
            partitions.append(Partition(len(partitions), current_part_layers, dag=self.G))
        return partitions

    def _generate_parallel_aware_clustering(self, layer_ranks):
        """
        Seed 2: Parallel-aware but willing to 'collapse' branches if BW is low.
        Updated: Checks if parallelism is actually beneficial vs packing.
        """
        generations = list(nx.topological_generations(self.G))
        partitions = []
        node_to_part = {}
        
        # Heuristic: limit parallel slots if BW is low
        is_low_bw = (self.bandwidth_per_ms * 1000 * 8) < 500
        
        for gen in generations:
            gen_sorted = sorted(gen, key=lambda x: layer_ranks[x], reverse=True)
            num_slots = min(len(gen_sorted), len(self.servers))
            
            current_gen_parts = [[] for _ in range(num_slots)]
            for i, nid in enumerate(gen_sorted):
                current_gen_parts[i % num_slots].append(self.layers_map[nid])
            
            for snippet in current_gen_parts:
                if not snippet: continue
                potential_targets = set()
                for layer in snippet:
                    for pred in self.G.predecessors(layer.id):
                        if pred in node_to_part: potential_targets.add(node_to_part[pred])
                
                merged = False
                for p in potential_targets:
                    if self._would_cause_cycle(partitions, p.id, snippet): continue
                    test_layers = p.layers + snippet
                    temp_p = Partition(-1, test_layers, dag=self.G)
                    # Liberal merge threshold in seeding
                    if temp_p.total_memory <= EPC_EFFECTIVE_MB * 1.3:
                        p.layers = temp_p.layers
                        p.total_memory = temp_p.total_memory
                        p.total_workload = temp_p.total_workload
                        for l in snippet: node_to_part[l.id] = p
                        merged = True
                        break
                
                if not merged:
                    new_p = Partition(len(partitions), snippet, dag=self.G)
                    partitions.append(new_p)
                    for l in snippet: node_to_part[l.id] = new_p
        return partitions

    def _would_cause_cycle(self, partitions, target_id, new_layers):
        """Checks if adding layers to a specific partition creates a dependency cycle."""
        node_to_p = {l.id: p.id for p in partitions for l in p.layers}
        for l in new_layers: node_to_p[l.id] = target_id
        
        temp_dag = nx.DiGraph()
        for u, v in self.G.edges():
            pu, pv = node_to_p.get(u), node_to_p.get(v)
            if pu is not None and pv is not None and pu != pv:
                temp_dag.add_edge(pu, pv)
        
        return not nx.is_directed_acyclic_graph(temp_dag)

    def _get_critical_path(self, schedule_res):
        """
        Identifies the sequence of partitions that contribute to the maximum latency.
        """
        # Finds the partition that finished last
        last_pid = -1
        max_ft = -1
        p_to_server = {}
        p_to_end = {}
        for s_id, events in schedule_res.server_schedules.items():
            for ev in events:
                p_id = ev['partition_id']
                p_to_server[p_id] = s_id
                p_to_end[p_id] = ev['end']
                if ev['end'] > max_ft:
                    max_ft = ev['end']
                    last_pid = p_id
        
        if last_pid == -1: return []
        
        # Trace back dependencies
        # This is a simplification: we look for predecessors that ended just before or influenced this one
        critical_path = [last_pid]
        curr = last_pid
        
        # Build partition dependency graph
        pg = nx.DiGraph()
        node_to_pid = {l.id: p.id for p in schedule_res.partitions for l in p.layers}
        for u, v in self.G.edges():
            pu, pv = node_to_pid.get(u), node_to_pid.get(v)
            if pu is not None and pv is not None and pu != pv:
                pg.add_edge(pu, pv)
        
        while True:
            preds = list(pg.predecessors(curr))
            if not preds: break
            # Pick predecessor that finished latest
            best_p = max(preds, key=lambda p_id: p_to_end.get(p_id, 0))
            critical_path.append(best_p)
            curr = best_p
            
        return list(reversed(critical_path))

    def _search_optimized(self, initial_states):
        """Advanced Beam search with Merge, Split, and Critical-Path awareness."""
        current_beam = []
        for p_list in initial_states:
            try:
                res = self.schedule(p_list)
                current_beam.append((p_list, res.latency, res))
            except Exception: continue

        if not current_beam:
            return [Partition(0, list(self.layers_map.values()), dag=self.G)]

        best_overall_lat = min(s[1] for s in current_beam)
        best_overall_parts = next(s[0] for s in current_beam if s[1] == best_overall_lat)

        for iteration in range(self.MAX_ITERATIONS):
            candidates = []
            for parts, lat, last_res in current_beam:
                critical_path = self._get_critical_path(last_res)
                
                # Operator 1: Critical Path Merge (Vertical)
                merge_targets = self._get_merge_candidates(parts, critical_path)
                for u_id, v_id in merge_targets:
                    try:
                        new_parts = self._merge_partitions(parts, u_id, v_id)
                        new_res = self.schedule(new_parts)
                        if new_res.latency < lat * 1.05:
                            candidates.append((new_parts, new_res.latency, new_res))
                    except Exception: continue

                # Operator 2: Parallelism-Regain Split (Horizontal)
                if len(self.servers) > 1:
                    split_targets = self._get_split_candidates(parts, critical_path)
                    for p_id in split_targets:
                        try:
                            new_parts_list = self._split_partition(parts, p_id)
                            for new_parts in new_parts_list:
                                new_res = self.schedule(new_parts)
                                if new_res.latency < lat * 1.01: 
                                    candidates.append((new_parts, new_res.latency, new_res))
                        except Exception: continue

                # Operator 3: Local Rebalancing (Shuffle)
                # Shifting layers at the boundaries of critical partitions
                rebalance_targets = self._get_rebalance_candidates(parts, critical_path)
                for p_src_id, p_dst_id, layer_id in rebalance_targets:
                    try:
                        new_parts = self._rebalance_layer(parts, p_src_id, p_dst_id, layer_id)
                        new_res = self.schedule(new_parts)
                        if new_res.latency < lat * 1.02:
                            candidates.append((new_parts, new_res.latency, new_res))
                    except Exception: continue

            if not candidates: break
            # Deduplicate by latency (heuristic for unique partition schemes)
            seen_lats = set()
            unique_candidates = []
            for c in sorted(candidates, key=lambda x: x[1]):
                l_round = round(c[1], 4)
                if l_round not in seen_lats:
                    seen_lats.add(l_round)
                    unique_candidates.append(c)
                if len(unique_candidates) >= self.BEAM_WIDTH: break
            
            current_beam = unique_candidates
            
            if current_beam[0][1] < best_overall_lat - 0.01:
                best_overall_lat = current_beam[0][1]
                best_overall_parts = current_beam[0][0]
                # print(f"  [Ours Iter {iteration}] New best: {best_overall_lat:.2f} ms")
            elif iteration > 5 and current_beam[0][1] >= best_overall_lat:
                # Early exit if no improvement
                # pass 
                break
                
        return best_overall_parts

    def _get_split_candidates(self, partitions, critical_path):
        """Identify heavy partitions on critical path that might benefit from splitting."""
        # Focus on the most expensive partition on the critical path
        if not critical_path: return []
        return [pid for pid in critical_path if any(p.id == pid and len(p.layers) > 1 for p in partitions)]

    def _split_partition(self, partitions, p_id):
        """
        Heuristically splits a partition into two if internal parallelism exists.
        Returns a list of potential partition schemes.
        """
        target_p = next((p for p in partitions if p.id == p_id), None)
        if not target_p or len(target_p.layers) < 2: return []
        
        # Subgraph of nodes in this partition
        sub_g = self.G.subgraph([l.id for l in target_p.layers])
        
        # Strategy A: Balanced Topological Split (for narrow models)
        topo = list(nx.topological_sort(sub_g))
        mid = len(topo) // 2
        
        # Strategy B: Sibling Split (if multiple roots exist in sub-g)
        roots = [n for n in sub_g.nodes() if sub_g.in_degree(n) == 0]
        
        schemes = []
        
        # Implementation A
        if len(topo) >= 2:
            p1_layers = [self.layers_map[nid] for nid in topo[:mid]]
            p2_layers = [self.layers_map[nid] for nid in topo[mid:]]
            new_p_list = [p for p in partitions if p.id != p_id]
            next_id = max(p.id for p in partitions) + 1
            new_p_list.append(Partition(p_id, p1_layers, self.G))
            new_p_list.append(Partition(next_id, p2_layers, self.G))
            schemes.append(copy.deepcopy(new_p_list))

        # Implementation B (Functional decomposition)
        if len(roots) >= 2:
            # Group by roots
            p1_nodes = set()
            for r in roots[:1]: 
                p1_nodes.update(nx.descendants(sub_g, r))
                p1_nodes.add(r)
            p2_nodes = set(sub_g.nodes()) - p1_nodes
            if p2_nodes:
                p1_layers = [self.layers_map[nid] for nid in p1_nodes]
                p2_layers = [self.layers_map[nid] for nid in p2_nodes]
                new_p_list = [p for p in partitions if p.id != p_id]
                next_id = max(p.id for p in partitions) + 1
                new_p_list.append(Partition(p_id, p1_layers, self.G))
                new_p_list.append(Partition(next_id, p2_layers, self.G))
                schemes.append(copy.deepcopy(new_p_list))

        return schemes

    def _get_rebalance_candidates(self, partitions, critical_path):
        """Finds layers at the boundary of critical partitions that could be shifted."""
        if not critical_path: return []
        
        candidates = []
        node_to_p = {l.id: p for p in partitions for l in p.layers}
        p_map = {p.id: p for p in partitions}
        
        for p_id in critical_path:
            p = p_map.get(p_id)
            if not p or len(p.layers) < 2: continue
            
            # Boundary layers: nodes with predecessors or successors in other partitions
            for layer in p.layers:
                # 1. Potential move to successor partition
                successors = list(self.G.successors(layer.id))
                for succ_id in successors:
                    p_succ = node_to_p.get(succ_id)
                    if p_succ and p_succ.id != p_id:
                        # Try moving layer to p_succ
                        candidates.append((p_id, p_succ.id, layer.id))
                
                # 2. Potential move from predecessor partition
                predecessors = list(self.G.predecessors(layer.id))
                for pred_id in predecessors:
                    p_pred = node_to_p.get(pred_id)
                    if p_pred and p_pred.id != p_id:
                        # Try moving pred layer to p
                        pred_layer = self.layers_map[pred_id]
                        if len(p_pred.layers) > 1: # Don't empty a partition here
                            candidates.append((p_pred.id, p_id, pred_id))
        
        return list(set(candidates))[:10]

    def _rebalance_layer(self, partitions, src_id, dst_id, layer_id):
        """Moves a layer from src partition to dst partition if legal."""
        # Check legality (no cycles)
        node_to_p_id = {l.id: p.id for p in partitions for l in p.layers}
        node_to_p_id[layer_id] = dst_id
        
        temp_dag = nx.DiGraph()
        for u, v in self.G.edges():
            pu, pv = node_to_p_id.get(u), node_to_p_id.get(v)
            if pu is not None and pv is not None and pu != pv:
                temp_dag.add_edge(pu, pv)
        
        if not nx.is_directed_acyclic_graph(temp_dag):
            raise ValueError("Cycle detected during rebalance")

        # Perform move
        new_parts = copy.deepcopy(partitions)
        p_src = next(p for p in new_parts if p.id == src_id)
        p_dst = next(p for p in new_parts if p.id == dst_id)
        
        layer = next(l for l in p_src.layers if l.id == layer_id)
        p_src.layers.remove(layer)
        p_dst.layers.append(layer)
        
        # Refresh partition stats (total_mem, workload)
        # Note: Partition object normally handles this on init, but we are mutating
        # Let's re-init them to be safe
        new_p_src = Partition(src_id, p_src.layers, self.G)
        new_p_dst = Partition(dst_id, p_dst.layers, self.G)
        
        idx_src = next(i for i, p in enumerate(new_parts) if p.id == src_id)
        idx_dst = next(i for i, p in enumerate(new_parts) if p.id == dst_id)
        new_parts[idx_src] = new_p_src
        new_parts[idx_dst] = new_p_dst
        
        return new_parts

    def _get_merge_candidates(self, partitions, critical_path=None):
        """
        Prioritizes merging partitions with high communication volume or parallel siblings.
        Weights candidates on the critical path more heavily.
        """
        node_to_p = {l.id: p for p in partitions for l in p.layers}
        candidates = []
        critical_set = set(critical_path) if critical_path else set()
        
        # 1. Vertical Merges (Successors)
        pair_weights = {}
        for u, v, data in self.G.edges(data=True):
            pu, pv = node_to_p.get(u), node_to_p.get(v)
            if pu and pv and pu != pv:
                key = (pu.id, pv.id)
                # Boost if both are on critical path
                boost = 2.0 if (pu.id in critical_set and pv.id in critical_set) else (1.2 if (pu.id in critical_set or pv.id in critical_set) else 1.0)
                pair_weights[key] = (pair_weights.get(key, 0) + data['weight']) * boost
        
        # 2. Horizontal Merges (Sibling Collapsing)
        for p1 in partitions:
            for p2 in partitions:
                if p1.id >= p2.id: continue
                # Identify if they are parallel branches (sharing preds)
                p1_preds = {node_to_p[pr].id for l in p1.layers for pr in self.G.predecessors(l.id) if pr in node_to_p}
                p2_preds = {node_to_p[pr].id for l in p2.layers for pr in self.G.predecessors(l.id) if pr in node_to_p}
                if p1_preds & p2_preds:
                    wt = 0.5
                    if p1.id in critical_set or p2.id in critical_set: wt = 1.0
                    candidates.append((p1.id, p2.id, wt))

        for (pu_id, pv_id), weight in pair_weights.items():
            candidates.append((pu_id, pv_id, weight))
            
        candidates.sort(key=lambda x: x[2], reverse=True)
        limit = 20 if not self.is_large else 12
        return [(c[0], c[1]) for c in candidates[:limit]]

    def _merge_partitions(self, partitions, u_id, v_id):
        """Deep merge two partitions into one."""
        # 1. Faster cycle pre-check
        node_to_p = {l.id: p.id for p in partitions for l in p.layers}
        for l_id in [lx.id for p in partitions if p.id == v_id for lx in p.layers]:
            node_to_p[l_id] = u_id
            
        test_dag = nx.DiGraph()
        for edge_u, edge_v in self.G.edges():
            pu, pv = node_to_p.get(edge_u), node_to_p.get(edge_v)
            if pu is not None and pv is not None and pu != pv:
                test_dag.add_edge(pu, pv)
        if not nx.is_directed_acyclic_graph(test_dag):
            raise ValueError("Cycle detected")

        # 2. Perform merge
        new_parts = copy.deepcopy(partitions)
        p_u = next(p for p in new_parts if p.id == u_id)
        p_v = next(p for p in new_parts if p.id == v_id)
        
        p_new = Partition(u_id, p_u.layers + p_v.layers, dag=self.G)
        new_parts.remove(p_v)
        for i, p in enumerate(new_parts):
            if p.id == u_id:
                new_parts[i] = p_new
                break
        return new_parts

    def schedule(self, partitions):
        """
        Dual-Strategy Scheduler. Evaluates both HEFT and Greedy-Critical-Path.
        This ensures Ours transcends both DINA and MEDIA.
        """
        # Strategy 1: HEFT (Already implemented)
        res_heft = self._schedule_heft(partitions)
        
        # Strategy 2: Priority-Greedy (MEDIA's style)
        res_greedy = self._schedule_greedy(partitions)
        
        final_res = res_heft if res_heft.latency < res_greedy.latency else res_greedy
        final_res.algorithm_name = "Ours"
        return final_res

    def _schedule_heft(self, partitions):
        # ... (Implementation from previous turn, ensures consistency) ...
        if not partitions: return ScheduleResult("Ours-H", 0.0, {}, [])
        part_dag = nx.DiGraph()
        for p in partitions: part_dag.add_node(p.id)
        node_to_pid = {l.id: p.id for p in partitions for l in p.layers}
        for u, v, data in self.G.edges(data=True):
            pu, pv = node_to_pid.get(u), node_to_pid.get(v)
            if pu is not None and pv is not None and pu != pv:
                w = part_dag[pu][pv]['weight'] if part_dag.has_edge(pu, pv) else 0
                part_dag.add_edge(pu, pv, weight=w + data['weight'])

        part_map = {p.id: p for p in partitions}
        rank = {}
        avg_power = sum(s.power_ratio for s in self.servers) / len(self.servers)
        
        topo_order = list(nx.topological_sort(part_dag))
        for pid in reversed(topo_order):
            p = part_map[pid]
            exec_cost = (p.total_workload * calculate_penalty(p.total_memory)) / avg_power
            max_succ_rank = 0
            for succ in part_dag.successors(pid):
                comm = part_dag[pid][succ]['weight'] / self.bandwidth_per_ms
                max_succ_rank = max(max_succ_rank, comm + rank[succ])
            rank[pid] = exec_cost + max_succ_rank

        sorted_parts = sorted(partitions, key=lambda x: rank.get(x.id, 0), reverse=True)
        server_free_time = {s.id: 0.0 for s in self.servers}
        server_schedules = {s.id: [] for s in self.servers}
        assignment = {} 

        for p in sorted_parts:
            best_s, best_ft = -1, float('inf')
            swap_mb = p.get_static_memory()
            p_cost = (math.ceil(swap_mb * 1024 / PAGE_SIZE_KB) * PAGE_FAULT_OVERHEAD_MS + 
                      swap_mb / self.paging_bw_per_ms + ENCLAVE_ENTRY_EXIT_OVERHEAD_MS)

            for s in self.servers:
                ready_time = 0.0
                for pred_id in part_dag.predecessors(p.id):
                    ps_id, p_ft = assignment[pred_id]
                    comm = 0.0 if ps_id == s.id else part_dag[pred_id][p.id]['weight'] / self.bandwidth_per_ms
                    ready_time = max(ready_time, p_ft + comm)
                
                start_exec = max(server_free_time[s.id], ready_time) + p_cost
                exec_time = (p.total_workload * calculate_penalty(p.total_memory)) / s.power_ratio
                ft = start_exec + exec_time
                if ft < best_ft: best_ft, best_s, final_exec_t = ft, s.id, exec_time
            
            server_free_time[best_s] = best_ft
            assignment[p.id] = (best_s, best_ft)
            server_schedules[best_s].append({
                'start': best_ft - final_exec_t, 'end': best_ft, 
                'partition_id': p.id, 'partition': p
            })
        return ScheduleResult("Ours", max(server_free_time.values()), server_schedules, partitions)

    def _schedule_greedy(self, partitions):
        """Standard Priority-based Greedy assignment."""
        if not partitions: return ScheduleResult("Ours-G", 0.0, {}, [])
        part_map = {p.id: p for p in partitions}
        node_to_pid = {l.id: p.id for p in partitions for l in p.layers}
        pg = nx.DiGraph()
        for u, v, data in self.G.edges(data=True):
            pu, pv = node_to_pid[u], node_to_pid[v]
            if pu != pv:
                w = pg[pu][pv]['weight'] if pg.has_edge(pu, pv) else 0
                pg.add_edge(pu, pv, weight=w + data['weight'])
        
        try:
            topo = list(nx.topological_sort(pg))
            # Critical: p.id might not match index, use part_map
            sorted_parts = [part_map[pid] for pid in topo if pid in part_map]
            # Add remaining partitions that might be isolated
            scheduled_ids = {p.id for p in sorted_parts}
            for p in partitions:
                if p.id not in scheduled_ids: sorted_parts.append(p)
        except:
            sorted_parts = partitions # Fallback

        server_free_time = {s.id: 0.0 for s in self.servers}
        server_schedules = {s.id: [] for s in self.servers}
        assignment, finish = {}, {}
        
        for p in sorted_parts:
            best_s, best_ft = None, float('inf')
            swap_mb = p.get_static_memory()
            p_cost = (p.get_swap_cost() if hasattr(p, 'get_swap_cost') else
                     (math.ceil(swap_mb * 1024 / PAGE_SIZE_KB) * PAGE_FAULT_OVERHEAD_MS + 
                      swap_mb / self.paging_bw_per_ms + ENCLAVE_ENTRY_EXIT_OVERHEAD_MS))

            for s in self.servers:
                ready_time = 0.0
                for pred_id in pg.predecessors(p.id):
                    # Robust check: if predecessor not assigned, skip or assume 0
                    if pred_id in assignment:
                        pred_s, pred_ft = assignment[pred_id], finish[pred_id]
                        comm = 0.0 if pred_s.id == s.id else pg[pred_id][p.id]['weight'] / self.bandwidth_per_ms
                        ready_time = max(ready_time, pred_ft + comm)
                
                start_exec = max(server_free_time[s.id], ready_time) + p_cost
                exec_time = (p.total_workload * calculate_penalty(p.total_memory)) / s.power_ratio
                ft = start_exec + exec_time
                if ft < best_ft: best_ft, best_s, final_exec_t = ft, s, exec_time
            
            # Final Assignment
            if best_s is None: best_s = self.servers[0] # Panic recovery
            server_free_time[best_s.id] = best_ft
            assignment[p.id], finish[p.id] = best_s, best_ft
            server_schedules[best_s.id].append({
                'start': best_ft - final_exec_t, 'end': best_ft, 
                'partition_id': p.id, 'partition': p
            })
        return ScheduleResult("Ours-G", max(server_free_time.values()), server_schedules, partitions)

