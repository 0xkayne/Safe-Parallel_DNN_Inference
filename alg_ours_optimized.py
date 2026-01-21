import networkx as nx
import math
import copy
from common import Partition, EPC_EFFECTIVE_MB, calculate_penalty, PAGE_SIZE_KB, PAGE_FAULT_OVERHEAD_MS, ENCLAVE_ENTRY_EXIT_OVERHEAD_MS, DEFAULT_PAGING_BW_MBPS, ScheduleResult
from alg_media import MEDIAAlgorithm

class OursAlgorithmOptimized:
    """
    Optimized version of OursAlgorithm for large graphs (500+ nodes).
    
    Key optimizations:
    1. Reduced search space by limiting merge candidates
    2. Cached partition DAG to avoid repeated construction
    3. Early stopping when no improvement
    4. Limited deep copy operations
    5. Use MEDIA result directly when graph is too large
    """
    def __init__(self, G, layers_map, servers, bandwidth_mbps):
        self.G = G
        self.layers_map = layers_map
        self.servers = servers
        self.bandwidth_per_ms = (bandwidth_mbps / 8.0) / 1000.0
        self.paging_bw_per_ms = DEFAULT_PAGING_BW_MBPS / 1000.0
        self.BEAM_WIDTH = 2  # Reduced from 3 for speed
        self.MAX_SEARCH_ITERATIONS = 3  # Reduced from 5
        self.MAX_MERGE_CANDIDATES = 50  # NEW: Limit merge exploration
        
        # Adaptive behavior based on graph size
        self.is_large_graph = G.number_of_nodes() > 200
        if self.is_large_graph:
            print(f"  [INFO] Large graph detected ({G.number_of_nodes()} nodes), using simplified search")
            self.BEAM_WIDTH = 1
            self.MAX_SEARCH_ITERATIONS = 2
            self.MAX_MERGE_CANDIDATES = 20

    def _get_partition_dag(self, partitions):
        """Build partition dependency DAG (cached)"""
        part_dag = nx.DiGraph()
        node_to_pid = {}
        for p in partitions:
            part_dag.add_node(p.id)
            for l in p.layers: 
                node_to_pid[l.id] = p.id
        
        for u, v in self.G.edges():
            pid_u, pid_v = node_to_pid.get(u), node_to_pid.get(v)
            if pid_u is not None and pid_v is not None and pid_u != pid_v:
                if part_dag.has_edge(pid_u, pid_v):
                    part_dag[pid_u][pid_v]['weight'] += self.G[u][v]['weight']
                else:
                    part_dag.add_edge(pid_u, pid_v, weight=self.G[u][v]['weight'])
        return part_dag

    def run(self):
        """Main partitioning algorithm"""
        # 1. Use MEDIA's result as baseline
        media = MEDIAAlgorithm(self.G, self.layers_map, self.servers, self.bandwidth_per_ms * 8000)
        media_partitions = media.run()
        
        # For very large graphs, use MEDIA directly
        if self.is_large_graph and self.G.number_of_nodes() > 400:
            print(f"  [INFO] Very large graph ({self.G.number_of_nodes()} nodes), using MEDIA result directly")
            for i, p in enumerate(media_partitions): 
                p.id = i
            return media_partitions
        
        # 2. Try limited improvement search
        try:
            best_partitions = self._search_improvement_fast([media_partitions])
        except Exception as e:
            print(f"  [WARNING] Search failed ({str(e)}), falling back to MEDIA")
            best_partitions = media_partitions
        
        # Final cleanup
        for i, p in enumerate(best_partitions): 
            p.id = i
        return best_partitions

    def _search_improvement_fast(self, initial_states):
        """
        Fast improvement search with aggressive pruning.
        Only explores most promising merge candidates.
        """
        # Evaluate initial states
        current_beam = []
        for p_list in initial_states:
            lat = self.schedule(p_list).latency
            current_beam.append((p_list, lat))
        
        best_overall = current_beam[0]
        no_improvement_count = 0
        
        for iteration in range(self.MAX_SEARCH_ITERATIONS):
            new_candidates = []
            
            for parts, lat in current_beam:
                part_dag = self._get_partition_dag(parts)
                
                # Get edges sorted by communication weight (descending)
                # Merging high-communication edges likely reduces latency
                edges_with_weight = [(u, v, data['weight']) 
                                    for u, v, data in part_dag.edges(data=True)]
                edges_with_weight.sort(key=lambda x: x[2], reverse=True)
                
                # Only explore top candidates
                merge_candidates = 0
                for u, v, weight in edges_with_weight:
                    if merge_candidates >= self.MAX_MERGE_CANDIDATES:
                        break
                    
                    # Quick cycle check: if v has path back to u, skip
                    if nx.has_path(part_dag, v, u):
                        continue
                    
                    # Memory constraint check before deep copy
                    p_u = next((p for p in parts if p.id == u), None)
                    p_v = next((p for p in parts if p.id == v), None)
                    if p_u and p_v and (p_u.total_memory + p_v.total_memory) > EPC_EFFECTIVE_MB * 1.5:
                        continue  # Skip oversized merges
                    
                    # Perform merge
                    new_parts = copy.deepcopy(parts)
                    p_u_new = next(p for p in new_parts if p.id == u)
                    p_v_new = next(p for p in new_parts if p.id == v)
                    p_u_new.layers.extend(p_v_new.layers)
                    p_u_new.total_memory += p_v_new.total_memory
                    p_u_new.total_workload += p_v_new.total_workload
                    new_parts.remove(p_v_new)
                    
                    # Evaluate
                    new_lat = self.schedule(new_parts).latency
                    new_candidates.append((new_parts, new_lat))
                    merge_candidates += 1
            
            if not new_candidates:
                break
            
            # Update beam
            all_states = current_beam + new_candidates
            all_states.sort(key=lambda x: x[1])
            current_beam = all_states[:self.BEAM_WIDTH]
            
            # Early stopping
            if current_beam[0][1] >= best_overall[1]:
                no_improvement_count += 1
                if no_improvement_count >= 2:
                    break  # No improvement for 2 iterations, stop
            else:
                best_overall = current_beam[0]
                no_improvement_count = 0
        
        return current_beam[0][0]

    def schedule(self, partitions):
        """Schedule partitions to servers (same as original)"""
        if not partitions: 
            return ScheduleResult("Ours", 0.0, {}, [])
        
        for i, p in enumerate(partitions): 
            p.id = i
        
        # Build partition DAG
        part_dag = nx.DiGraph()
        node_to_pid = {}
        for p in partitions:
            part_dag.add_node(p.id)
            for l in p.layers: 
                node_to_pid[l.id] = p.id
        
        for u, v in self.G.edges():
            pid_u, pid_v = node_to_pid.get(u), node_to_pid.get(v)
            if pid_u is not None and pid_v is not None and pid_u != pid_v:
                if part_dag.has_edge(pid_u, pid_v):
                    part_dag[pid_u][pid_v]['weight'] += self.G[u][v]['weight']
                else:
                    part_dag.add_edge(pid_u, pid_v, weight=self.G[u][v]['weight'])

        part_map = {p.id: p for p in partitions}
        rank = {}
        avg_p = sum(s.power_ratio for s in self.servers) / len(self.servers)
        
        # Compute rank (priority) for each partition
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
            server_schedule[best_s_id].append({
                'start': min_ft - exec_t_best, 
                'end': min_ft, 
                'partition_id': p.id, 
                'partition': p
            })
            
        return ScheduleResult("Ours", max([val[1] for val in assignment.values()] + [0.0]), server_schedule, partitions)


# Alias for backward compatibility
OursAlgorithm = OursAlgorithmOptimized
