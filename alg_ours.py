import networkx as nx
from common import Partition, EPC_EFFECTIVE_MB, calculate_penalty, PAGING_BANDWIDTH_MB_PER_MS

class OursAlgorithm:
    def __init__(self, G, layers_map, servers, bandwidth_mbps):
        self.G = G
        self.layers_map = layers_map
        self.servers = servers
        self.bandwidth_per_ms = (bandwidth_mbps / 8.0) / 1000.0
        
        # Fixing Heuristic: 
        # Partitioner assumes Split -> Comm. But Scheduler often does Split -> SameServer -> NoComm.
        # We weigh the communication cost by probability of it actually happening.
        self.COMM_WEIGHT = 1.0
        self.bandwidth_mbps = bandwidth_mbps

    def run(self):
        # 1. Edge Selection (Algo 1 style from paper)
        # Select edges for potential merging. Keep linear chains, avoid branches if possible.
        # But "Ours" specifically wants to preserve parallelism.
        # So we merge u->v ONLY if u has out-degree 1 AND v has in-degree 1.
        # This naturally preserves forks and joins.
        
        edges_to_merge = []
        for u, v in self.G.edges():
            if self.G.out_degree(u) == 1 and self.G.in_degree(v) == 1:
                # Potential merge candidate
                edges_to_merge.append((u, v))
                
        # 2. Graph Partitioning (Algo 2 style)
        # form initial partitions
        partitions = []
        node_to_part = {}
        
        # Init every node as partition
        for node_id in self.G.nodes():
            p = Partition(len(partitions), [self.layers_map[node_id]])
            partitions.append(p)
            node_to_part[node_id] = p
            
        # Helper to find p info
        def get_p(node_id):
            return node_to_part[node_id]
        
        # Merge loop based on edges
        # We process linear chain edges.
        # We can iterate multiple times or just once if sorted topol?
        # A simple set union-find or greedy merge approach.
        
        # Sort edges by topological order of u to handle chains correctly (u -> v -> w)
        # Not strictly necessary if we update references properly.
        
        for u, v in edges_to_merge:
            p_u = get_p(u)
            p_v = get_p(v)
            
            if p_u == p_v: continue
            
            # Check Merge Criteria
            # 1. Memory <= EPC? Merge.
            # 2. Memory > EPC? Check Trade-off.
            
            merged_mem = p_u.total_memory + p_v.total_memory
            
            should_merge = False
            if merged_mem <= EPC_EFFECTIVE_MB:
                should_merge = True
            else:
                # Calculate Tradeoff
                avg_power = 1.0 # Assume base 1.0
                
                # Time Separate (Parallel potential is 0 here since it's a linear edge u->v)
                # T_sep = T(p_u) + T(p_v) + Comm
                # NOTE: Since u->v is a linear edge, p_u MUST finish before p_v starts. Parallelism is not relevant directly *between them*.
                
                # Comm cost
                edge_mb = self.G[u][v]['weight']
                t_comm = edge_mb / self.bandwidth_per_ms
                
                t_sep = (p_u.total_workload + p_v.total_workload) + t_comm * self.COMM_WEIGHT
                
                # Time Merged
                # Penalty based on dynamic model
                penalty_factor = calculate_penalty(merged_mem)
                t_merged = (p_u.total_workload + p_v.total_workload) * penalty_factor
                
                if t_merged <= t_sep:
                    should_merge = True
            
            if should_merge:
                # Merge p_v into p_u
                p_u.layers.extend(p_v.layers)
                p_u.total_memory += p_v.total_memory
                p_u.total_workload += p_v.total_workload
                
                # Update map
                for l in p_v.layers:
                    node_to_part[l.id] = p_u
                
                # Remove p_v
                if p_v in partitions:
                    partitions.remove(p_v)
        
        return partitions

    def schedule(self, partitions):
        # Algorithm 3: Priority-based Scheduling on Multi-Server
        # 1. Build Partition DAG
        part_dag = nx.DiGraph()
        part_map = {p.id: p for p in partitions}
        node_to_part_id = {}
        for p in partitions:
            part_dag.add_node(p.id)
            for l in p.layers:
                node_to_part_id[l.id] = p.id
                
        for u, v in self.G.edges():
            pid_u = node_to_part_id[u]
            pid_v = node_to_part_id[v]
            if pid_u != pid_v:
                # Weight logic: sum of all edges crossing these partitions
                w = 0
                if part_dag.has_edge(pid_u, pid_v):
                    w = part_dag[pid_u][pid_v]['weight']
                w += self.G[u][v]['weight']
                part_dag.add_edge(pid_u, pid_v, weight=w)

        # 2. Priority Calculation (Rank)
        # Rank_u = Work_u + max_{v in succ(u)} (Comm_{u,v} + Rank_v)
        # Recursive with memoization
        rank = {}
        
        # Avg Power for rank calc
        avg_p = sum(s.power_ratio for s in self.servers) / len(self.servers)
        
        def get_rank(pid):
            if pid in rank: return rank[pid]
            
            p = part_map[pid]
            # Basic exec time calc for priority
            penalty_factor = calculate_penalty(p.total_memory)
            w_cost = (p.total_workload * penalty_factor) / avg_p
            
            succs = list(part_dag.successors(pid))
            if not succs:
                rank[pid] = w_cost
                return w_cost
            
            max_succ_rank = 0
            for s in succs:
                comm_mb = part_dag[pid][s]['weight']
                c_cost = comm_mb / self.bandwidth_per_ms
                r = c_cost + get_rank(s)
                if r > max_succ_rank:
                    max_succ_rank = r
            
            rank[pid] = w_cost + max_succ_rank
            return rank[pid]

        for p in partitions:
            get_rank(p.id)
            
        # 3. List Scheduling
        sorted_parts = sorted(partitions, key=lambda x: rank[x.id], reverse=True)
        
        server_free_time = {s.id: 0.0 for s in self.servers}
        server_schedule = {s.id: [] for s in self.servers} # Log
        part_finish_times = {} # pid -> finish_time
        
        for p in sorted_parts:
            # Find Best Server
            best_s = None
            min_finish = float('inf')
            best_start = 0
            
            for s in self.servers:
                # Ready time = max(avail(s), max(pred_finish + comm))
                s_avail = server_free_time[s.id]
                
                pred_ready = 0
                for pred_id in part_dag.predecessors(p.id):
                    # Find where pred ran
                    # We need to store where partitions ran
                    pass 
                
                # To do this correctly, we need 'part_finish_times' to also store ServerID
                # Re-do specific tracking
                pass
            
            # --- Re-loop inside with logic ---
            pass
        
        # Let's rewrite the loop cleaner
        assignment = {} # pid -> (server_id, finish_time)
        
        for p in sorted_parts:
            best_s_id = -1
            min_ft = float('inf')
            
            for s in self.servers:
                s_avail = server_free_time[s.id]
                
                # Calc dependency time
                dep_ready = 0
                for pred_id in part_dag.predecessors(p.id):
                    if pred_id not in assignment: continue # Should not happen due to sort? 
                    # Wait, rank sort is usually good for topological, but let's verify.
                    # Rank is computed from End to Start. Sorting reverse Rank gives Start to End order. Correct.
                    
                    pred_s_id, pred_ft = assignment[pred_id]
                    
                    comm = 0
                    if pred_s_id != s.id:
                        vol = part_dag[pred_id][p.id]['weight']
                        comm = vol / self.bandwidth_per_ms
                    else:
                        # Same server: SGX context switch paging overhead
                        # We need the memory of pred partition.
                        # We can get it from partitions list or map.
                        pred_p = part_map[pred_id]
                        swap_out = min(pred_p.total_memory, EPC_EFFECTIVE_MB)
                        swap_in = min(p.total_memory, EPC_EFFECTIVE_MB)
                        comm = (swap_out + swap_in) / PAGING_BANDWIDTH_MB_PER_MS
                    
                    dep_ready = max(dep_ready, pred_ft + comm)
                
                start_t = max(s_avail, dep_ready)
                
                # Exec
                penalty_factor = calculate_penalty(p.total_memory)
                exec_t = (p.total_workload * penalty_factor) / s.power_ratio
                
                ft = start_t + exec_t
                
                if ft < min_ft:
                    min_ft = ft
                    best_s_id = s.id
            
            # Commit
            server_free_time[best_s_id] = min_ft
            assignment[p.id] = (best_s_id, min_ft)
            
        return max(val[1] for val in assignment.values())
