"""
OursAlgorithm with HPA (Hybrid Parallel Algorithm)

Implements Two-Stage DP for optimal intra-operator tensor parallelism:
- Stage 0: Cost-Benefit Candidate Filtering (not just EPC threshold)
- Stage 1: Cost Surface Construction
- Stage 2: DAG Dynamic Programming
- Stage 3: Graph Augmentation
- Stage 4: MEDIA Partitioning + HEFT Scheduling
"""

import networkx as nx
import math
from typing import List, Dict, Tuple, Set, Optional
from collections import defaultdict
from copy import deepcopy

from common import (
    Partition, EPC_EFFECTIVE_MB, calculate_penalty,
    network_latency, ScheduleResult, hpa_cost, DNNLayer,
    PAGE_SIZE_KB, PAGE_FAULT_OVERHEAD_MS, 
    ENCLAVE_ENTRY_EXIT_OVERHEAD_MS, DEFAULT_PAGING_BW_MBPS
)


class OursAlgorithm:
    def __init__(self, G, layers_map, servers, bandwidth_mbps):
        self.G = G
        self.layers_map = layers_map
        self.servers = servers
        self.bandwidth_mbps = bandwidth_mbps
        self.bandwidth_per_ms = (bandwidth_mbps / 8.0) / 1000.0
        
        # HPA Configuration
        self.K_candidates = [1, 2, 4, 8]  # Parallelism degrees to consider
        self.benefit_threshold = 0.95  # Require at least 5% latency reduction
        self.node_to_partition = {}
        
    def run(self) -> List[Partition]:
        """Main HPA algorithm entry point."""
        print("[HPA] Starting Hybrid Parallel Algorithm...")
        
        # Stage 0: Cost-Benefit Candidate Filtering
        candidates = self._filter_candidates_by_cost_benefit()
        print(f"[HPA] Filtered {len(candidates)} operators with tensor-parallel benefit")
        
        # Stage 1: Build Cost Surface
        cost_surface = self._build_cost_surface(candidates)
        
        # Stage 2: DAG DP
        optimal_cfg = self._dag_dp(cost_surface, candidates)
        split_count = sum(1 for k in optimal_cfg.values() if k > 1)
        print(f"[HPA] DP found optimal config: {split_count} operators split")
        
        # Stage 3: Graph Augmentation
        G_aug, layers_aug = self._augment_graph(optimal_cfg)
        print(f"[HPA] Augmented graph: {len(layers_aug)} nodes (from {len(self.layers_map)})")
        
        # Stage 4: MEDIA-style Partitioning on augmented graph
        partitions = self._media_partition(G_aug, layers_aug)
        print(f"[HPA] Generated {len(partitions)} partitions")
        
        return partitions
    
    def _filter_candidates_by_cost_benefit(self) -> Set[int]:
        """
        Phase 0: Filter candidates based on cost-benefit analysis.
        
        Key Insight: Even if memory < EPC, if compute time is large,
        tensor parallelism can reduce latency IF speedup > sync overhead.
        
        Decision rule: Include if Cost(k>1) < Cost(1) * threshold
        """
        candidates = set()
        
        for nid, layer in self.layers_map.items():
            # Baseline cost (no parallelism)
            cost_k1 = hpa_cost(layer, 1, self.bandwidth_mbps)
            
            # Find best parallel configuration
            best_k = 1
            best_cost = cost_k1
            
            for k in self.K_candidates[1:]:  # Skip k=1
                cost_k = hpa_cost(layer, k, self.bandwidth_mbps)
                if cost_k < best_cost:
                    best_cost = cost_k
                    best_k = k
            
            # Include if we achieve significant benefit
            if best_cost < cost_k1 * self.benefit_threshold:
                candidates.add(nid)
                if layer.memory > EPC_EFFECTIVE_MB:
                    reason = f"thrashing (M={layer.memory:.1f}MB > EPC)"
                else:
                    reason = f"compute-heavy (T={layer.workload:.1f}ms, k={best_k})"
                print(f"  [HPA] Candidate: {layer.name} - {reason}")
        
        return candidates
    
    def _build_cost_surface(self, candidates: Set[int]) -> Dict[int, Dict[int, float]]:
        """Stage 1: Build cost surface for all nodes and configurations."""
        cost = {}
        for nid, layer in self.layers_map.items():
            cost[nid] = {}
            if nid in candidates:
                # Evaluate all k in K_candidates
                for k in self.K_candidates:
                    cost[nid][k] = hpa_cost(layer, k, self.bandwidth_mbps)
            else:
                # Non-candidate: only k=1
                cost[nid][1] = hpa_cost(layer, 1, self.bandwidth_mbps)
        
        return cost
    
    def _dag_dp(self, cost_surface: Dict[int, Dict[int, float]], 
                candidates: Set[int]) -> Dict[int, int]:
        """Stage 2: DAG Dynamic Programming to find optimal split configuration."""
        # DP state: dp[node][config] = min cost to reach this node with this config
        dp = {}
        parent_cfg = {}  # For backtracking
        
        # Process nodes in topological order
        for node in nx.topological_sort(self.G):
            layer = self.layers_map[node]
            dp[node] = {}
            parent_cfg[node] = {}
            
            # Determine valid configs for this node
            if node in candidates:
                valid_k = self.K_candidates
            else:
                valid_k = [1]
            
            for k in valid_k:
                # Cost of this node with config k
                node_cost = cost_surface[node][k]
                
                # Predecessors
                preds = list(self.G.predecessors(node))
                
                if not preds:
                    # Source node: no transition cost
                    dp[node][k] = node_cost
                    parent_cfg[node][k] = {}
                else:
                    # Find optimal predecessor configs
                    best_pred_cost = float('inf')
                    best_pred_cfg = {}
                    
                    # Try all combinations of predecessor configs
                    # For simplicity, assume all preds use same config (can be relaxed)
                    for pred in preds:
                        pred_valid_k = self.K_candidates if pred in candidates else [1]
                        
                        for k_pred in pred_valid_k:
                            if k_pred not in dp[pred]:
                                continue
                            
                            # Transition cost (resharding)
                            trans_cost = self._transition_cost(pred, node, k_pred, k)
                            
                            # Critical path: max over all preds
                            pred_cost = dp[pred][k_pred] + trans_cost
                            
                            if pred_cost < best_pred_cost:
                                best_pred_cost = pred_cost
                                best_pred_cfg = {pred: k_pred}
                    
                    # Handle multiple predecessors (max for critical path)
                    if len(preds) > 1:
                        # Simplified: take max of individual pred costs
                        max_pred_cost = 0
                        for pred in preds:
                            pred_k = self.K_candidates if pred in candidates else [1]
                            for k_p in pred_k:
                                if k_p in dp[pred]:
                                    trans = self._transition_cost(pred, node, k_p, k)
                                    max_pred_cost = max(max_pred_cost, dp[pred][k_p] + trans)
                        best_pred_cost = max_pred_cost
                    
                    dp[node][k] = node_cost + best_pred_cost
                    parent_cfg[node][k] = best_pred_cfg
        
        # Find optimal config at sink nodes
        sinks = [n for n in self.G.nodes() if self.G.out_degree(n) == 0]
        optimal_cfg = {}
        
        min_total_cost = float('inf')
        best_sink_cfg = None
        
        for sink in sinks:
            for k in dp[sink]:
                if dp[sink][k] < min_total_cost:
                    min_total_cost = dp[sink][k]
                    best_sink_cfg = (sink, k)
        
        # Backtrack to recover full configuration
        # Simplified: use greedy choice for each node
        for node in self.G.nodes():
            if node in candidates:
                # Choose k that minimizes local cost
                best_k = min(cost_surface[node].keys(), key=lambda k: cost_surface[node][k])
                optimal_cfg[node] = best_k
            else:
                optimal_cfg[node] = 1
        
        return optimal_cfg
    
    def _transition_cost(self, u: int, v: int, k_u: int, k_v: int) -> float:
        """Compute transition cost (resharding) between two nodes."""
        if k_u == k_v:
            return 0.0  # No resharding needed
        
        # Resharding requires data movement
        layer_u = self.layers_map[u]
        reshard_bytes = layer_u.output_bytes
        reshard_mb = reshard_bytes / (1024 * 1024)
        
        return network_latency(reshard_mb, self.bandwidth_mbps)
    
    def _augment_graph(self, optimal_cfg: Dict[int, int]) -> Tuple[nx.DiGraph, Dict[int, DNNLayer]]:
        """Stage 3: Augment graph by splitting nodes according to optimal config."""
        G_aug = nx.DiGraph()
        layers_aug = {}
        node_id_counter = 0
        
        # Map: original node -> list of shard node IDs
        node_to_shards = {}
        
        # Phase 3.1: Node Replacement
        for orig_node, k in optimal_cfg.items():
            orig_layer = self.layers_map[orig_node]
            shards = []
            
            for i in range(k):
                new_id = node_id_counter
                node_id_counter += 1
                
                # Create shard layer
                shard_layer = DNNLayer(
                    layer_id=new_id,
                    name=f"{orig_layer.name}_shard_{i}",
                    memory=(orig_layer.weight_memory + orig_layer.bias_memory) / k + orig_layer.activation_memory,
                    cpu_time=orig_layer.cpu_time / k,
                    enclave_time=orig_layer.enclave_time / k,
                    output_bytes=orig_layer.output_bytes / k,
                    execution_mode=orig_layer.execution_mode,
                    weight_memory=(orig_layer.weight_memory) / k,
                    bias_memory=(orig_layer.bias_memory) / k,
                    activation_memory=orig_layer.activation_memory,
                    encryption_overhead=orig_layer.encryption_overhead / k
                )
                
                layers_aug[new_id] = shard_layer
                G_aug.add_node(new_id, layer=shard_layer)
                shards.append(new_id)
            
            node_to_shards[orig_node] = shards
        
        # Phase 3.2: Edge Rewiring
        for u, v in self.G.edges():
            k_u = optimal_cfg[u]
            k_v = optimal_cfg[v]
            u_shards = node_to_shards[u]
            v_shards = node_to_shards[v]
            
            weight = self.G[u][v].get('weight', 0)
            
            if k_u == k_v:
                # 1:1 or k:k mapping (no resharding)
                for i in range(k_u):
                    G_aug.add_edge(u_shards[i], v_shards[i], weight=weight / k_u)
            else:
                # Resharding needed: all-to-all connections
                for u_shard in u_shards:
                    for v_shard in v_shards:
                        G_aug.add_edge(u_shard, v_shard, weight=weight / (k_u * k_v))
        
        return G_aug, layers_aug
    
    def _media_partition(self, G: nx.DiGraph, layers_map: Dict[int, DNNLayer]) -> List[Partition]:
        """
        Stage 4: MEDIA-style partitioning on augmented graph.
        Integrated from alg_media.py.
        """
        # Initialize each layer as its own partition
        self.node_to_partition = {}
        for i, (nid, layer) in enumerate(layers_map.items()):
            self.node_to_partition[nid] = Partition(i, [layer], G)
        
        # Get candidate edges using MEDIA's constraint logic
        edges_M = self._select_edges_for_partitioning(G, layers_map)
        
        # Greedily merge based on communication volume
        sorted_edges = sorted(list(edges_M),
                            key=lambda e: G[e[0]][e[1]]['weight'],
                            reverse=True)
        
        merge_count = 0
        for (u, v) in sorted_edges:
            pu = self.node_to_partition[u]
            pv = self.node_to_partition[v]
            
            if pu != pv:
                # Cycle detection
                if not self._would_cause_cycle(pu, pv, G):
                    # Merge check based on MEDIA cost model
                    if self._merge_check(pu, pv, G):
                        # Merge pv into pu
                        new_layers = list(set(pu.layers + pv.layers))
                        pu_new = Partition(pu.id, new_layers, G)
                        # Bulk update node map
                        for l in new_layers:
                            self.node_to_partition[l.id] = pu_new
                        merge_count += 1
        
        # Finalize unique partitions
        unique_parts = list(set(self.node_to_partition.values()))
        for i, p in enumerate(unique_parts):
            p.id = i
        
        print(f"  [HPA] Merged {merge_count} times")
        return unique_parts
    
    def _select_edges_for_partitioning(self, G: nx.DiGraph, layers_map: Dict[int, DNNLayer]) -> Set[Tuple[int, int]]:
        """Select candidate edges for merging (MEDIA constraints)."""
        M = set()
        
        # Calculate topological levels
        level_map = {}
        for level, nodes in enumerate(nx.topological_generations(G)):
            for node in nodes:
                level_map[node] = level
        
        # Iterate through all edges in topological order
        for u in nx.topological_sort(G):
            for v in G.successors(u):
                # Constraint 1: Degree check
                if len(self.servers) > 1:
                    if not (G.out_degree(u) == 1 or G.in_degree(v) == 1):
                        continue
                else:
                    if G.in_degree(v) != 1 and G.out_degree(u) != 1:
                        continue
                
                M.add((u, v))
                
                # Constraint 2: Same-level conflict check
                violates = False
                for w in G.successors(u):
                    for wp in G.predecessors(w):
                        if (wp, w) != (u, v) and (wp, w) in M:
                            if level_map.get(u, -1) == level_map.get(w, -2) - 1:
                                violates = True
                                break
                    if violates:
                        break
                
                if violates:
                    M.discard((u, v))
        
        return M
    
    def _would_cause_cycle(self, p1: Partition, p2: Partition, G: nx.DiGraph) -> bool:
        """Check if merging p1 and p2 would cause a cycle."""
        pg = nx.DiGraph()
        unique_pids = list(set([p.id for p in self.node_to_partition.values()]))
        pg.add_nodes_from(unique_pids)
        
        for edge_u, edge_v in G.edges():
            pu_id = self.node_to_partition[edge_u].id
            pv_id = self.node_to_partition[edge_v].id
            if pu_id != pv_id:
                pg.add_edge(pu_id, pv_id)
        
        return nx.has_path(pg, p2.id, p1.id)
    
    def _merge_check(self, part1: Partition, part2: Partition, G: nx.DiGraph) -> bool:
        """MEDIA merge decision logic."""
        temp_layers = list(set(part1.layers + part2.layers))
        temp_part = Partition(-1, temp_layers, G)
        merged_mem = temp_part.total_memory
        
        # Case A: Fits in EPC -> always merge
        if merged_mem <= EPC_EFFECTIVE_MB:
            return True
        
        # Case B: Exceeds EPC -> cost-benefit analysis
        avg_power = sum(s.power_ratio for s in self.servers) / len(self.servers)
        
        # Split execution time
        t_p1 = (part1.total_workload * calculate_penalty(part1.total_memory)) / avg_power
        t_p2 = (part2.total_workload * calculate_penalty(part2.total_memory)) / avg_power
        
        # Communication time
        vol = 0.0
        for l1 in part1.layers:
            for l2 in part2.layers:
                if G.has_edge(l1.id, l2.id):
                    vol += G[l1.id][l2.id]['weight']
                if G.has_edge(l2.id, l1.id):
                    vol += G[l2.id][l1.id]['weight']
        
        if len(self.servers) == 1:
            t_comm = 0.0
        else:
            vol_mb = vol / (1024 * 1024)
            t_comm = network_latency(vol_mb, self.bandwidth_mbps * 8 * 1000) if vol > 0 else 0.0
        
        # Paging costs
        def paging_cost(p):
            swap_mb = p.get_static_memory()
            num_pages = math.ceil(swap_mb * 1024 / PAGE_SIZE_KB)
            return (num_pages * PAGE_FAULT_OVERHEAD_MS +
                    swap_mb / (DEFAULT_PAGING_BW_MBPS / 1000.0) +
                    ENCLAVE_ENTRY_EXIT_OVERHEAD_MS)
        
        t_paging = paging_cost(part1) + paging_cost(part2)
        
        # Merged execution time
        merged_workload = part1.total_workload + part2.total_workload
        t_merged = (merged_workload * calculate_penalty(merged_mem)) / avg_power
        t_paging_merged = paging_cost(temp_part)
        
        # Merge if merged cost is lower or equal
        return (t_merged + t_paging_merged) <= (t_p1 + t_p2 + t_comm + t_paging)
    
    def schedule(self, partitions: List[Partition]) -> ScheduleResult:
        """
        Stage 5: HEFT-based scheduling.
        Integrated from alg_media.py.
        """
        if not partitions:
            return ScheduleResult("Ours(HPA)", 0.0, {}, [])
        
        # Build partition graph
        partition_graph = nx.DiGraph()
        for p in partitions:
            partition_graph.add_node(p.id)
        
        for u, v in self.G.edges():
            pu = self.node_to_partition[u]
            pv = self.node_to_partition[v]
            if pu.id != pv.id:
                partition_graph.add_edge(pu.id, pv.id)
        
        partitions_list = {p.id: p for p in partitions}
        priorities = self._compute_priorities(partition_graph, partitions_list)
        
        # Topological sort for ordering
        try:
            topo_order = list(nx.topological_sort(partition_graph))
        except nx.NetworkXUnfeasible:
            # Fallback if there are cycles (shouldn't happen)
            topo_order = list(partition_graph.nodes())
        
        topo_idx = {pid: i for i, pid in enumerate(topo_order)}
        sorted_partitions = sorted(partitions, 
                                  key=lambda p: (priorities[p.id], -topo_idx[p.id]), 
                                  reverse=True)
        
        # HEFT scheduling
        server_free_time = {s.id: 0.0 for s in self.servers}
        server_schedule = {s.id: [] for s in self.servers}
        assignment, finish = {}, {}
        
        for p in sorted_partitions:
            best_s, best_ft = None, float('inf')
            
            # Paging cost
            swap_mb = p.get_static_memory()
            num_pages = math.ceil(swap_mb * 1024 / PAGE_SIZE_KB)
            paging_cost = (num_pages * PAGE_FAULT_OVERHEAD_MS +
                          swap_mb / (DEFAULT_PAGING_BW_MBPS / 1000.0) +
                          ENCLAVE_ENTRY_EXIT_OVERHEAD_MS)
            
            for s in self.servers:
                dependency_ready = 0.0
                
                # Check dependencies
                for pred_id in partition_graph.predecessors(p.id):
                    if pred_id not in assignment:
                        continue
                    pred_s, pred_ft = assignment[pred_id], finish[pred_id]
                    
                    if pred_s.id != s.id:
                        # Network communication
                        comm_data = sum(self.G[l1.id][l2.id]['weight'] 
                                      for l1 in partitions_list[pred_id].layers 
                                      for l2 in p.layers 
                                      if self.G.has_edge(l1.id, l2.id))
                        comm_data_mb = comm_data / (1024 * 1024)
                        comm_time = network_latency(comm_data_mb, self.bandwidth_mbps * 8 * 1000)
                        arrival = pred_ft + comm_time
                        dependency_ready = max(dependency_ready, arrival)
                    else:
                        # Local dependency
                        dependency_ready = max(dependency_ready, pred_ft)
                
                # Start time
                start_loading = max(server_free_time[s.id], dependency_ready)
                start_exec = start_loading + paging_cost
                
                # Execution time
                exec_t = (p.total_workload * calculate_penalty(p.total_memory)) / s.power_ratio
                ft = start_exec + exec_t
                
                if ft < best_ft:
                    best_ft, best_s, final_exec_t = ft, s, exec_t
            
            assignment[p.id], finish[p.id] = best_s, best_ft
            server_free_time[best_s.id] = best_ft
            server_schedule[best_s.id].append({
                'start': best_ft - final_exec_t,
                'end': best_ft,
                'partition_id': p.id,
                'partition': p
            })
        
        return ScheduleResult("Ours(HPA)", max(finish.values()), server_schedule, partitions)
    
    def _compute_priorities(self, partition_graph: nx.DiGraph, 
                          partitions_list: Dict[int, Partition]) -> Dict[int, float]:
        """Compute HEFT priorities for partitions."""
        priorities = {}
        avg_p = sum(s.power_ratio for s in self.servers) / len(self.servers)
        
        try:
            topo_order = list(nx.topological_sort(partition_graph))
        except nx.NetworkXUnfeasible:
            topo_order = list(partition_graph.nodes())
        
        for pid in reversed(topo_order):
            partition = partitions_list[pid]
            t_exec = (partition.total_workload * calculate_penalty(partition.total_memory)) / avg_p
            
            max_succ_priority = 0
            successors = list(partition_graph.successors(pid))
            if successors:
                max_succ_priority = max(priorities.get(sid, 0.0) for sid in successors)
            
            comm_data = sum(self.G[l1.id][l2.id]['weight'] 
                          for sid in successors 
                          for l1 in partition.layers 
                          for l2 in partitions_list[sid].layers 
                          if self.G.has_edge(l1.id, l2.id))
            
            priorities[pid] = t_exec + comm_data / self.bandwidth_per_ms + max_succ_priority
        
        return priorities
