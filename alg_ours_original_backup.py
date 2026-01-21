import networkx as nx
import math
import copy
from common import Partition, EPC_EFFECTIVE_MB, calculate_penalty, PAGE_SIZE_KB, PAGE_FAULT_OVERHEAD_MS, ENCLAVE_ENTRY_EXIT_OVERHEAD_MS, DEFAULT_PAGING_BW_MBPS, ScheduleResult
from alg_media import MEDIAAlgorithm

class OursAlgorithm:
    def __init__(self, G, layers_map, servers, bandwidth_mbps):
        self.G = G
        self.layers_map = layers_map
        self.servers = servers
        self.bandwidth_per_ms = (bandwidth_mbps / 8.0) / 1000.0
        self.paging_bw_per_ms = DEFAULT_PAGING_BW_MBPS / 1000.0
        self.BEAM_WIDTH = 3 # Small beam to keep it fast

    def _get_partition_dag(self, partitions):
        part_dag = nx.DiGraph()
        node_to_pid = {}
        for p in partitions:
            part_dag.add_node(p.id)
            for l in p.layers: node_to_pid[l.id] = p.id
        for u, v in self.G.edges():
            pid_u, pid_v = node_to_pid.get(u), node_to_pid.get(v)
            if pid_u is not None and pid_v is not None and pid_u != pid_v:
                w = part_dag[pid_u][pid_v]['weight'] if part_dag.has_edge(pid_u, pid_v) else 0
                part_dag.add_edge(pid_u, pid_v, weight=w + self.G[u][v]['weight'])
        return part_dag

    def run(self):
        # 1. Use MEDIA's result as a baseline seed
        media = MEDIAAlgorithm(self.G, self.layers_map, self.servers, self.bandwidth_per_ms * 8000)
        media_partitions = media.run()
        
        # 2. Add an alternative seed: Linear Pre-merge
        linear_partitions = self._pre_process_segments()
        
        # 3. Beam Search
        initial_states = [media_partitions, linear_partitions]
        best_partitions = self._search_improvement(initial_states)
        
        # Final cleanup
        for i, p in enumerate(best_partitions): p.id = i
        return best_partitions

    def _pre_process_segments(self):
        partitions = {i: Partition(i, [self.layers_map[node_id]]) for i, node_id in enumerate(self.G.nodes())}
        node_to_pid = {node_id: i for i, node_id in enumerate(self.G.nodes())}
        topo = list(nx.topological_sort(self.G))
        for u in topo:
            if self.G.out_degree(u) == 1:
                v = list(self.G.successors(u))[0]
                if self.G.in_degree(v) == 1:
                    pu, pv = node_to_pid[u], node_to_pid[v]
                    if pu != pv and (partitions[pu].total_memory + partitions[pv].total_memory) <= EPC_EFFECTIVE_MB:
                        p1, p2 = partitions[pu], partitions[pv]
                        p1.layers.extend(p2.layers)
                        p1.total_memory += p2.total_memory
                        p1.total_workload += p2.total_workload
                        for l in p2.layers: node_to_pid[l.id] = pu
                        del partitions[pv]
        return list(partitions.values())

    def _search_improvement(self, initial_states):
        # State: (list of partitions, latency)
        current_beam = []
        for p_list in initial_states:
            current_beam.append((p_list, self.schedule(p_list).latency))
        
        # Multi-step Hill Climbing / Pruned Search
        visited_configs = set()
        for _ in range(5): # Limit steps for speed
            new_candidates = []
            for parts, lat in current_beam:
                config_key = tuple(sorted([tuple(sorted([l.id for l in p.layers])) for p in parts]))
                if config_key in visited_configs: continue
                visited_configs.add(config_key)
                
                part_dag = self._get_partition_dag(parts)
                part_map = {p.id: p for p in parts}
                
                # Explore all valid single-edge merges
                for u, v in part_dag.edges():
                    # Cycle safety
                    if any(nx.has_path(part_dag, s, v) for s in part_dag.successors(u) if s != v):
                        continue
                    
                    # Merge candidate
                    new_parts = copy.deepcopy(parts)
                    p_u = next(p for p in new_parts if p.id == u)
                    p_v = next(p for p in new_parts if p.id == v)
                    p_u.layers.extend(p_v.layers)
                    p_u.total_memory += p_v.total_memory
                    p_u.total_workload += p_v.total_workload
                    new_parts.remove(p_v)
                    
                    # Evaluation
                    new_lat = self.schedule(new_parts).latency
                    new_candidates.append((new_parts, new_lat))
            
            if not new_candidates: break
            
            # Update beam
            all_states = current_beam + new_candidates
            all_states.sort(key=lambda x: x[1])
            
            unique_beam = []
            seen_lats = set()
            for s in all_states:
                if s[1] not in seen_lats:
                    unique_beam.append(s)
                    seen_lats.add(s[1])
                if len(unique_beam) >= self.BEAM_WIDTH: break
            
            if unique_beam[0][1] >= current_beam[0][1] and len(unique_beam) == len(current_beam):
                # No improvement in best and no new variety
                pass # Continue to explore or break?
            current_beam = unique_beam
            
        return current_beam[0][0]

    def schedule(self, partitions):
        if not partitions: return ScheduleResult("Ours", 0.0, {}, [])
        for i, p in enumerate(partitions): p.id = i
        
        part_dag = nx.DiGraph()
        node_to_pid = {}
        for p in partitions:
            part_dag.add_node(p.id)
            for l in p.layers: node_to_pid[l.id] = p.id
        for u, v in self.G.edges():
            pid_u, pid_v = node_to_pid.get(u), node_to_pid.get(v)
            if pid_u is not None and pid_v is not None and pid_u != pid_v:
                w = part_dag[pid_u][pid_v]['weight'] if part_dag.has_edge(pid_u, pid_v) else 0
                part_dag.add_edge(pid_u, pid_v, weight=w + self.G[u][v]['weight'])

        part_map = {p.id: p for p in partitions}
        rank = {}
        avg_p = sum(s.power_ratio for s in self.servers) / len(self.servers)
        
        topo_sorted_pids = list(nx.topological_sort(part_dag))
        for pid in reversed(topo_sorted_pids):
            p = part_map[pid]
            w_cost = (p.total_workload * calculate_penalty(p.total_memory)) / avg_p
            max_succ_rank = 0
            for s in part_dag.successors(pid):
                comm_cost = part_dag[pid][s]['weight'] / self.bandwidth_per_ms
                max_succ_rank = max(max_succ_rank, comm_cost + rank[s])
            rank[pid] = w_cost + max_succ_rank

        sorted_parts = sorted(partitions, key=lambda x: (rank[x.id], -topo_sorted_pids.index(x.id)), reverse=True)
        
        server_free_time = {s.id: 0.0 for s in self.servers}
        server_schedule = {s.id: [] for s in self.servers}
        assignment = {} 
        
        for p in sorted_parts:
            best_s_id, min_ft = -1, float('inf')
            exec_t_best = 0.0
            for s in self.servers:
                s_avail = server_free_time[s.id]
                dep_ready = 0
                for pred_id in part_dag.predecessors(p.id):
                    pred_s_id, pred_ft = assignment[pred_id]
                    if pred_s_id != s.id:
                        comm = part_dag[pred_id][p.id]['weight'] / self.bandwidth_per_ms
                    else:
                        swap_mb = part_map[pred_id].total_memory + p.total_memory
                        num_pages = math.ceil(swap_mb * 1024 / PAGE_SIZE_KB)
                        comm = num_pages * PAGE_FAULT_OVERHEAD_MS + swap_mb / self.paging_bw_per_ms + ENCLAVE_ENTRY_EXIT_OVERHEAD_MS
                    dep_ready = max(dep_ready, pred_ft + comm)
                
                start_t = max(s_avail, dep_ready)
                exec_t = (p.total_workload * calculate_penalty(p.total_memory)) / s.power_ratio
                ft = start_t + exec_t
                if ft < min_ft: 
                    min_ft, best_s_id = ft, s.id
                    exec_t_best = exec_t
            
            server_free_time[best_s_id] = min_ft
            assignment[p.id] = (best_s_id, min_ft)
            server_schedule[best_s_id].append({'start': min_ft - exec_t_best, 'end': min_ft, 'partition_id': p.id, 'partition': p})
            
        return ScheduleResult("Ours", max([val[1] for val in assignment.values()] + [0.0]), server_schedule, partitions)
