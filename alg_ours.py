"""
OursAlgorithm: Optimized Partitioning for Parallel DNN Inference

Goal: Beat MEDIA with critical-path-aware merge ordering.

Key insight: MEDIA sorts merges by edge weight. We sort by critical path impact -
merging edges on the critical path reduces latency more than others.
"""

import networkx as nx
import math
from typing import List, Dict, Tuple, Set, Optional, Any
from collections import defaultdict

from common import (
    Partition, EPC_EFFECTIVE_MB, calculate_penalty,
    PAGE_SIZE_KB, PAGE_FAULT_OVERHEAD_MS, ENCLAVE_ENTRY_EXIT_OVERHEAD_MS,
    PAGING_BANDWIDTH_MB_PER_MS, ScheduleResult, network_latency,
    DEFAULT_PAGING_BW_MBPS
)


class OursAlgorithm:
    def __init__(self, G, layers_map, servers, bandwidth_mbps):
        self.G = G
        self.layers_map = layers_map
        self.servers = servers
        self.bandwidth_mbps = bandwidth_mbps
        self.bandwidth_per_ms = (bandwidth_mbps / 8.0) / 1000.0
        self.paging_bw_per_ms = PAGING_BANDWIDTH_MB_PER_MS
        self.node_to_partition = {}
        
    def run(self) -> List[Partition]:
        """Main entry point."""
        # Phase 1: Initialize each layer as its own partition
        for i, (nid, layer) in enumerate(self.layers_map.items()):
            self.node_to_partition[nid] = Partition(i, [layer], self.G)
        
        # Phase 2: Select candidate edges (MEDIA constraints)
        candidate_edges = self._select_candidate_edges()
        
        # Phase 3: Compute node criticality (distance to sink on longest path)
        node_criticality = self._compute_node_criticality()
        
        # Phase 4: Greedy merge with CRITICAL PATH AWARE ordering
        # Sort by: (edge_weight) * (max_criticality_of_endpoints)
        # This prioritizes high-communication edges on the critical path
        def edge_score(e):
            u, v = e
            weight = self.G[u][v].get('weight', 0)
            u_crit = node_criticality.get(u, 0)
            v_crit = node_criticality.get(v, 0)
            max_crit = max(u_crit, v_crit)
            # Higher criticality = more impact on latency
            return weight * (1.0 + max_crit / 1000.0)  # Normalize criticality contribution
        
        sorted_edges = sorted(list(candidate_edges), key=edge_score, reverse=True)
        
        merge_count = 0
        for u, v in sorted_edges:
            pu = self.node_to_partition[u]
            pv = self.node_to_partition[v]
            
            if pu != pv:
                if not self._would_cause_cycle(pu, pv):
                    if self._should_merge(pu, pv):
                        self._merge_partitions(pu, pv)
                        merge_count += 1
        
        # Finalize
        unique_parts = list(set(self.node_to_partition.values()))
        for i, p in enumerate(unique_parts):
            p.id = i
            
        print(f"  [Ours] Generated {len(unique_parts)} partitions ({merge_count} merges)")
        return unique_parts

    def _compute_node_criticality(self) -> Dict[int, float]:
        """
        Compute node criticality as longest path distance to any sink.
        Higher value = more critical (on longer path to completion).
        """
        criticality = {}
        
        # Reverse topological order
        for node in reversed(list(nx.topological_sort(self.G))):
            layer = self.layers_map.get(node)
            if layer:
                node_workload = layer.workload
            else:
                node_workload = 0
            
            successors = list(self.G.successors(node))
            if not successors:
                criticality[node] = node_workload
            else:
                max_succ_crit = max(criticality.get(s, 0) for s in successors)
                criticality[node] = node_workload + max_succ_crit
        
        return criticality

    def _select_candidate_edges(self) -> Set[Tuple[int, int]]:
        """MEDIA-identical edge selection with Constraint 1 and 2."""
        M = set()
        
        level_map = {}
        for level, nodes in enumerate(nx.topological_generations(self.G)):
            for node in nodes:
                level_map[node] = level
        
        for u in nx.topological_sort(self.G):
            for v in self.G.successors(u):
                # Constraint 1: Degree check
                if len(self.servers) > 1:
                    if not (self.G.out_degree(u) == 1 or self.G.in_degree(v) == 1):
                        continue
                else:
                    if self.G.in_degree(v) != 1 and self.G.out_degree(u) != 1:
                        continue
                
                M.add((u, v))
                
                # Constraint 2: Same-level conflict check
                violates = False
                for w in self.G.successors(u):
                    for wp in self.G.predecessors(w):
                        if (wp, w) != (u, v) and (wp, w) in M:
                            if level_map.get(u, -1) == level_map.get(w, -2) - 1:
                                violates = True
                                break
                    if violates:
                        break
                
                if violates:
                    M.discard((u, v))
        
        return M

    def _would_cause_cycle(self, p1: Partition, p2: Partition) -> bool:
        """Check if merging would create a cycle."""
        pg = nx.DiGraph()
        unique_pids = list(set(p.id for p in self.node_to_partition.values()))
        pg.add_nodes_from(unique_pids)
        
        for eu, ev in self.G.edges():
            pu_id = self.node_to_partition[eu].id
            pv_id = self.node_to_partition[ev].id
            if pu_id != pv_id:
                pg.add_edge(pu_id, pv_id)
        
        if p2.id not in pg.nodes() or p1.id not in pg.nodes():
            return True
        return nx.has_path(pg, p2.id, p1.id)

    def _should_merge(self, p1: Partition, p2: Partition) -> bool:
        """MEDIA-identical cost-based merge decision."""
        temp_layers = list(set(p1.layers + p2.layers))
        temp_part = Partition(-1, temp_layers, self.G)
        merged_mem = temp_part.total_memory
        
        if merged_mem <= EPC_EFFECTIVE_MB:
            return True
        
        avg_power = sum(s.power_ratio for s in self.servers) / len(self.servers)
        
        t_p1 = (p1.total_workload * calculate_penalty(p1.total_memory)) / avg_power
        t_p2 = (p2.total_workload * calculate_penalty(p2.total_memory)) / avg_power
        
        vol = 0.0
        for l1 in p1.layers:
            for l2 in p2.layers:
                if self.G.has_edge(l1.id, l2.id):
                    vol += self.G[l1.id][l2.id].get('weight', 0)
                if self.G.has_edge(l2.id, l1.id):
                    vol += self.G[l2.id][l1.id].get('weight', 0)
        
        if len(self.servers) > 1 and vol > 0:
            vol_mb = vol / (1024 * 1024)
            t_comm = network_latency(vol_mb, self.bandwidth_mbps)
        else:
            t_comm = 0.0
        
        def paging_cost(p):
            swap_mb = p.get_static_memory()
            num_pages = math.ceil(swap_mb * 1024 / PAGE_SIZE_KB)
            return (num_pages * PAGE_FAULT_OVERHEAD_MS + 
                    swap_mb / (DEFAULT_PAGING_BW_MBPS / 1000.0) +
                    ENCLAVE_ENTRY_EXIT_OVERHEAD_MS)
        
        t_paging_sep = paging_cost(p1) + paging_cost(p2)
        merged_workload = p1.total_workload + p2.total_workload
        t_merged = (merged_workload * calculate_penalty(merged_mem)) / avg_power
        t_paging_merged = paging_cost(temp_part)
        
        return (t_merged + t_paging_merged) <= (t_p1 + t_p2 + t_comm + t_paging_sep)

    def _merge_partitions(self, p1: Partition, p2: Partition):
        """Merge p2 into p1."""
        new_layers = list(set(p1.layers + p2.layers))
        new_part = Partition(p1.id, new_layers, self.G)
        for l in new_layers:
            self.node_to_partition[l.id] = new_part

    def schedule(self, partitions: List[Partition]) -> ScheduleResult:
        """Enhanced HEFT scheduling."""
        if not partitions:
            return ScheduleResult("Ours", 0.0, {}, [])
        
        part_dag = self._build_partition_dag(partitions)
        ranks = self._compute_upward_rank(partitions, part_dag)
        sorted_parts = sorted(partitions, key=lambda p: ranks.get(p.id, 0), reverse=True)
        
        server_free_time = {s.id: 0.0 for s in self.servers}
        server_schedules = {s.id: [] for s in self.servers}
        assignment = {}
        
        for p in sorted_parts:
            best_server = None
            best_finish_time = float('inf')
            best_start_time = 0.0
            best_exec_time = 0.0
            
            swap_mb = p.get_static_memory()
            num_pages = math.ceil(swap_mb * 1024 / PAGE_SIZE_KB)
            paging_overhead = (num_pages * PAGE_FAULT_OVERHEAD_MS + 
                               swap_mb / self.paging_bw_per_ms +
                               ENCLAVE_ENTRY_EXIT_OVERHEAD_MS)
            
            for s in self.servers:
                ready_time = 0.0
                for pred_id in part_dag.predecessors(p.id):
                    if pred_id in assignment:
                        pred_server_id, pred_finish = assignment[pred_id]
                        
                        if pred_server_id == s.id:
                            comm_time = 0.0
                        else:
                            comm_mb = part_dag[pred_id][p.id].get('weight', 0) / (1024 * 1024)
                            comm_time = network_latency(comm_mb, self.bandwidth_mbps)
                        
                        ready_time = max(ready_time, pred_finish + comm_time)
                
                start_time = max(server_free_time[s.id], ready_time) + paging_overhead
                penalty = calculate_penalty(p.total_memory)
                exec_time = (p.total_workload * penalty) / s.power_ratio
                finish_time = start_time + exec_time
                
                if finish_time < best_finish_time:
                    best_finish_time = finish_time
                    best_server = s
                    best_start_time = start_time
                    best_exec_time = exec_time
            
            if best_server is None:
                best_server = self.servers[0]
                best_start_time = server_free_time[best_server.id] + paging_overhead
                penalty = calculate_penalty(p.total_memory)
                best_exec_time = (p.total_workload * penalty) / best_server.power_ratio
                best_finish_time = best_start_time + best_exec_time
            
            server_free_time[best_server.id] = best_finish_time
            assignment[p.id] = (best_server.id, best_finish_time)
            
            server_schedules[best_server.id].append({
                'start': best_start_time,
                'end': best_finish_time,
                'partition_id': p.id,
                'partition': p
            })
        
        total_latency = max(server_free_time.values())
        return ScheduleResult("Ours", total_latency, server_schedules, partitions)
    
    def _build_partition_dag(self, partitions: List[Partition]) -> nx.DiGraph:
        """Build partition dependency DAG."""
        node_to_pid = {l.id: p.id for p in partitions for l in p.layers}
        
        part_dag = nx.DiGraph()
        for p in partitions:
            part_dag.add_node(p.id)
        
        for u, v, data in self.G.edges(data=True):
            pu = node_to_pid.get(u)
            pv = node_to_pid.get(v)
            
            if pu is not None and pv is not None and pu != pv:
                if part_dag.has_edge(pu, pv):
                    part_dag[pu][pv]['weight'] += data.get('weight', 0)
                else:
                    part_dag.add_edge(pu, pv, weight=data.get('weight', 0))
        
        return part_dag
    
    def _compute_upward_rank(self, partitions: List[Partition], part_dag: nx.DiGraph) -> Dict[int, float]:
        """Compute upward rank for HEFT."""
        part_map = {p.id: p for p in partitions}
        avg_power = sum(s.power_ratio for s in self.servers) / len(self.servers)
        
        ranks = {}
        
        for pid in reversed(list(nx.topological_sort(part_dag))):
            p = part_map[pid]
            penalty = calculate_penalty(p.total_memory)
            exec_cost = (p.total_workload * penalty) / avg_power
            
            max_succ = 0.0
            for succ_id in part_dag.successors(pid):
                comm_bytes = part_dag[pid][succ_id].get('weight', 0)
                comm_mb = comm_bytes / (1024 * 1024)
                comm_time = network_latency(comm_mb, self.bandwidth_mbps)
                max_succ = max(max_succ, comm_time + ranks.get(succ_id, 0))
            
            ranks[pid] = exec_cost + max_succ
        
        return ranks
