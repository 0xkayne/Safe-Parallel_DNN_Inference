import networkx as nx
import math
from common import Partition, EPC_EFFECTIVE_MB, calculate_penalty, PAGE_SIZE_KB, PAGE_FAULT_OVERHEAD_MS, ENCLAVE_ENTRY_EXIT_OVERHEAD_MS, DEFAULT_PAGING_BW_MBPS, ScheduleResult

class MEDIAAlgorithm:
    def __init__(self, G, layers_map, servers, bandwidth_mbps):
        self.G = G
        self.layers_map = layers_map
        self.servers = servers
        self.bandwidth_per_ms = (bandwidth_mbps / 8.0) / 1000.0
        self.node_to_partition = {}

    def _select_edges_for_partitioning(self):
        """
        Algorithm 1: Select candidate edges for merging.
        MEDIA focuses on filling partitions efficiently. We relax the branching constraints 
        to allow merging parallel nodes if it leads to better resource utilization.
        """
        M = set()

        # Step 1: Calculate topological generations for level information
        topological_gen = nx.topological_generations(self.G)
        level_map = {}
        for level, nodes in enumerate(topological_gen):
            for node in nodes:
                level_map[node] = level

        # Step 2: Iterate through all edges in topological order
        for u in nx.topological_sort(self.G):
            for v in self.G.successors(u):
                # Rule 1: To preserve parallelism and avoid cycles, 
                # do not merge if node has multiple successors or predecessors 
                # (unless only one server is present).
                if len(self.servers) > 1 and not (self.G.out_degree(u) == 1 or self.G.in_degree(v) == 1):
                    continue
                
                # Rule 2: DAG Check...
                    
        return M
    
    def _would_cause_cycle(self, p1, p2):
        """
        Rigorous cycle detection for MEDIA.
        A merge u + v causes a cycle if there's a path from v to u in current partition graph.
        """
        # Create a fresh partition DAG for the check
        pg = nx.DiGraph()
        # IDs of unique partitions
        unique_pids = list(set([p.id for p in self.node_to_partition.values()]))
        pg.add_nodes_from(unique_pids)
        
        for edge_u, edge_v in self.G.edges():
            pu_id = self.node_to_partition[edge_u].id
            pv_id = self.node_to_partition[edge_v].id
            if pu_id != pv_id:
                pg.add_edge(pu_id, pv_id)
        
        # Merge p1 and p2 conceptually. Check if a path exists from p2.id (v) back to p1.id (u)
        # However, at this stage pu and pv are neighbors u->v.
        # Check if there is already a path from the node that would be 'downstream' to 'upstream'
        if nx.has_path(pg, p2.id, p1.id):
            return True
        return False

    def _merge_check(self, part1, part2):
        """
        Case-by-case merge decision.
        MEDIA Philosophy: 
        1. Always merge if it fits in EPC.
        2. If it exceeds EPC, merge if:
           (Merged Execution with Paging Penalty) <= (Sum of Separate Execs + Network Comm + Sequential Paging Overhead)
        This allows 'oversized' partitions to exist if communication costs are the dominant bottleneck.
        """
        temp_layers = list(set(part1.layers + part2.layers))
        temp_part = Partition(-1, temp_layers, self.G)
        merged_mem = temp_part.total_memory
        
        # Case A: Fits in EPC. Always merge to reduce switching/enclave entry overhead.
        if merged_mem <= EPC_EFFECTIVE_MB:
            return True
            
        # Case B: Memory exceeds EPC.
        # We calculate the execution time with the penalty factor.
        # Note: DINA/OCC discard this case immediately. MEDIA evaluates the trade-off.
        
        avg_power = sum(s.power_ratio for s in self.servers) / len(self.servers)
        
        # 1. Calculate split execution time (including existing penalties if any)
        t_p1 = (part1.total_workload * calculate_penalty(part1.total_memory)) / avg_power
        t_p2 = (part2.total_workload * calculate_penalty(part2.total_memory)) / avg_power
        
        # 2. Calculate communication time if separate
        vol = 0.0
        for l1 in part1.layers:
            for l2 in part2.layers:
                if self.G.has_edge(l1.id, l2.id): vol += self.G[l1.id][l2.id]['weight']
                if self.G.has_edge(l2.id, l1.id): vol += self.G[l2.id][l1.id]['weight']
        
        if len(self.servers) == 1:
            t_comm = 0.0
        else:
            t_comm = vol / self.bandwidth_per_ms if vol > 0 else 0.0
            
        # 3. Calculate paging overhead for separate partitions (sequential)
        def paging_cost(p):
            swap_mb = p.get_static_memory()
            num_pages = math.ceil(swap_mb * 1024 / PAGE_SIZE_KB)
            return (num_pages * PAGE_FAULT_OVERHEAD_MS + 
                    swap_mb / (DEFAULT_PAGING_BW_MBPS / 1000.0) + 
                    ENCLAVE_ENTRY_EXIT_OVERHEAD_MS)
        
        t_paging = paging_cost(part1) + paging_cost(part2)
        
        # 4. Calculate merged execution time
        merged_workload = part1.total_workload + part2.total_workload
        t_merged = (merged_workload * calculate_penalty(merged_mem)) / avg_power
        t_paging_merged = paging_cost(temp_part)
        
        # Merge if merged cost is lower or equal
        return (t_merged + t_paging_merged) <= (t_p1 + t_p2 + t_comm + t_paging)
    
    def run(self):
        """
        Main runner for MEDIA algorithm.
        Stage 1: Select candidate edges.
        Stage 2: Greedily merge partitions based on candidate edges and cost model.
        """
        # Initialize each layer as its own partition
        self.node_to_partition = {}
        for i, (nid, layer) in enumerate(self.layers_map.items()):
            self.node_to_partition[nid] = Partition(i, [layer], self.G)
            
        # Stage 1: Get candidate edges
        edges_M = self._select_edges_for_partitioning()
        
        # Stage 2: Greedily merge
        # Sort edges by communication volume to fill partitions 'heavy' edges first
        sorted_edges = sorted(list(edges_M), 
                           key=lambda e: self.G[e[0]][e[1]]['weight'], 
                           reverse=True)

        for (u, v) in sorted_edges:
            pu = self.node_to_partition[u]
            pv = self.node_to_partition[v]
            
            if pu != pv:
                # Cycle detection: only merge if it doesn't break DAG
                if not self._would_cause_cycle(pu, pv):
                    if self._merge_check(pu, pv):
                        # Merge pv into pu
                        new_layers = list(set(pu.layers + pv.layers))
                        pu_new = Partition(pu.id, new_layers, self.G)
                        # Bulk update node map
                        for l in new_layers:
                            self.node_to_partition[l.id] = pu_new
        
        # Finalize unique partitions
        unique_parts = list(set(self.node_to_partition.values()))
        # Re-assign IDs
        for i, p in enumerate(unique_parts):
            p.id = i
        return unique_parts
    
    def schedule(self, partitions):
        if not partitions: return ScheduleResult("MEDIA", 0.0, {}, [])
        partition_graph = nx.DiGraph()
        for p in partitions: partition_graph.add_node(p.id)
        for u, v in self.G.edges():
            pu, pv = self.node_to_partition[u], self.node_to_partition[v]
            if pu.id != pv.id: partition_graph.add_edge(pu.id, pv.id)
        
        partitions_list = {p.id: p for p in partitions}
        priorities = self._compute_priorities(partition_graph, partitions_list)
        # STRICT ENFORCEMENT: No fallback for cycles.
        topo_order = list(nx.topological_sort(partition_graph))
        topo_idx = {pid: i for i, pid in enumerate(topo_order)}
            
        sorted_partitions = sorted(partitions, key=lambda p: (priorities[p.id], -topo_idx[p.id]), reverse=True)
        
        server_free_time = {s.id: 0.0 for s in self.servers}
        server_schedule = {s.id: [] for s in self.servers}
        assignment, finish = {}, {}
        
        for p in sorted_partitions:
            best_s, best_ft = None, float('inf')
            best_s, best_ft = None, float('inf')
            
            # Universal Paging Cost for Partition P (Loading Static Weights)
            swap_mb = p.get_static_memory()
            num_pages = math.ceil(swap_mb * 1024 / PAGE_SIZE_KB)
            paging_cost = (num_pages * PAGE_FAULT_OVERHEAD_MS + 
                           swap_mb / (DEFAULT_PAGING_BW_MBPS / 1000.0) + 
                           ENCLAVE_ENTRY_EXIT_OVERHEAD_MS)

            for s in self.servers:
                dependency_ready = 0.0
                
                # Check dependencies
                for pred_id in partition_graph.predecessors(p.id):
                    if pred_id not in assignment: continue
                    pred_s, pred_ft = assignment[pred_id], finish[pred_id]
                    
                    if pred_s.id != s.id:
                        # Network communication needed
                        comm_data = sum(self.G[l1.id][l2.id]['weight'] for l1 in partitions_list[pred_id].layers for l2 in p.layers if self.G.has_edge(l1.id, l2.id))
                        arrival = pred_ft + comm_data / self.bandwidth_per_ms
                        dependency_ready = max(dependency_ready, arrival)
                    else:
                        # Local dependency, data ready immediately when pred finishes
                        # (We removed the conditional local swapping here, as we apply universal swapping below)
                        dependency_ready = max(dependency_ready, pred_ft)
                
                # Scheduling:
                # 1. We can start loading when Server is free AND Data is ready (dependencies met)
                #    Actually, ensuring data is ready before loading is safer.
                # 2. Loading consumes Server CPU (ELDU).
                
                start_loading = max(server_free_time[s.id], dependency_ready)
                start_exec = start_loading + paging_cost
                
                exec_t = (p.total_workload * calculate_penalty(p.total_memory)) / s.power_ratio
                ft = start_exec + exec_t
                
                if ft < best_ft: best_ft, best_s, final_exec_t = ft, s, exec_t
            
            assignment[p.id], finish[p.id] = best_s, best_ft
            server_free_time[best_s.id] = best_ft
            # Log the event. Visualizer expects 'start' to be execution start? 
            # Or should it include loading? 
            # Usually "Latency" includes everything. 
            # Let's log the whole block [Start_Load -> Finish_Exec] as the partition event for simplicity,
            # or we can verify how other algos do it. 
            # DINA/OCC: data_ready includes paging. start_t = max(free, data_ready). 
            # So they effectively model "Paging happens BEFORE start_t" (hidden latency) OR "Paging pushes start_t back".
            # Wait, DINA: `data_ready = prev + paging`. `start = max(free, data_ready)`.
            # If free >> data_ready, Paging is hidden? No.
            # If free < data_ready, Start is delayed by Paging. 
            # This implies Paging happens in parallel or is just a delay?
            # Correct model: Paging consumes CPU.
            # My new logic `start_exec = max(free, ready) + paging` explicitly consumes CPU.
            server_schedule[best_s.id].append({'start': best_ft - final_exec_t, 'end': best_ft, 'partition_id': p.id, 'partition': p})
            # Note: The above append only highlights EXECUTION time. The "Gap" before it is Paging.
            # This matches the Breakdown script logic (Gap = Paging).
            
        return ScheduleResult("MEDIA", max(finish.values()), server_schedule, partitions)

    def _compute_priorities(self, partition_graph, partitions_list):
        priorities = {}
        avg_p = sum(s.power_ratio for s in self.servers) / len(self.servers)
        
        # Iterative Priority Calculation (Reverse Topological Order)
        try:
            topo_order = list(nx.topological_sort(partition_graph))
        except nx.NetworkXUnfeasible:
            # Cycle detected! Fallback: use simple node list (priorities will be approximate)
            # This can happen if partitioning logic constraints were relaxed too much.
            topo_order = list(partition_graph.nodes())
        
        for pid in reversed(topo_order):
            partition = partitions_list[pid]
            t_exec = (partition.total_workload * calculate_penalty(partition.total_memory)) / avg_p
            
            max_succ_priority = 0
            successors = list(partition_graph.successors(pid))
            if successors:
                # Use .get() with default 0.0 to handle potential cycles where a successor 
                # might not have been processed yet in the fallback order.
                max_succ_priority = max(priorities.get(sid, 0.0) for sid in successors)
            
            comm_data = sum(self.G[l1.id][l2.id]['weight'] for sid in successors for l1 in partition.layers for l2 in partitions_list[sid].layers if self.G.has_edge(l1.id, l2.id))
            priorities[pid] = t_exec + comm_data / self.bandwidth_per_ms + max_succ_priority
            
        return priorities
