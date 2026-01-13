import networkx as nx
from common import Partition, EPC_EFFECTIVE_MB, calculate_penalty, PAGING_BANDWIDTH_MB_PER_MS

class MEDIAAlgorithm:
    """
    MEDIA Algorithm Implementation (Paper-compliant)
    
    Implements:
    - Algorithm 1: Edge Selection (preserves parallel structures)
    - Algorithm 2: Graph Partitioning with merge check
    - Algorithm 3: Priority-based scheduling
    """
    
    def __init__(self, G, layers_map, servers, bandwidth_mbps):
        self.G = G
        self.layers_map = layers_map
        self.servers = servers
        self.bandwidth_per_ms = (bandwidth_mbps / 8.0) / 1000.0
        self.node_to_partition = {}  # Maps layer_id -> Partition object
        
    def _select_edges_for_partitioning(self):
        """
        Algorithm 1: Edge Selection
        
        Selects edges that can be safely merged without breaking parallel structures.
        Constraint 1: Only include edges where in_degree(v) == 1 OR out_degree(u) == 1
        
        Returns:
            set: Set of mergeable edges (u, v)
        """
        M = set()
        
        # Iterate through graph in topological order
        for u in nx.topological_sort(self.G):
            for v in self.G.successors(u):
                # Constraint 1: Preserve parallel structures
                # Only merge if one of the following is true:
                # - u has only one successor (no fork)
                # - v has only one predecessor (no join)
                if self.G.out_degree(u) == 1 or self.G.in_degree(v) == 1:
                    M.add((u, v))
        
        return M
    
    def _merge_check(self, part1, part2):
        """
        Algorithm 2: Merge Check Function
        
        Determines if two partitions should be merged based on:
        1. Memory constraint: merged_mem <= EPC -> always merge
        2. Time constraint: t_merged <= t_sep -> merge if beneficial
        
        Args:
            part1, part2: Partition objects to potentially merge
            
        Returns:
            bool: True if partitions should be merged
        """
        # Calculate merged metrics
        merged_mem = part1.total_memory + part2.total_memory
        merged_work = part1.total_workload + part2.total_workload
        
        # Rule 1: If within EPC, always merge (no penalty, saves communication)
        if merged_mem <= EPC_EFFECTIVE_MB:
            return True
        
        # Rule 2: Compare execution times
        # Average server power for estimation (will use actual in scheduling)
        avg_power = sum(s.power_ratio for s in self.servers) / len(self.servers)
        
        def exec_time(mem, work):
            """Calculate execution time with SGX penalty"""
            penalty = calculate_penalty(mem)
            return (work * penalty) / avg_power
        
        # Time if merged (with potential penalty)
        t_merged = exec_time(merged_mem, merged_work)
        
        # Time if separated (both execute + communication)
        t_p1 = exec_time(part1.total_memory, part1.total_workload)
        t_p2 = exec_time(part2.total_memory, part2.total_workload)
        
        # Communication time between partitions
        # Use edge weight if available, otherwise use default
        comm_data = 0.0
        for layer1 in part1.layers:
            for layer2 in part2.layers:
                if self.G.has_edge(layer1.id, layer2.id):
                    comm_data += self.G[layer1.id][layer2.id]['weight']
        
        t_comm = comm_data / self.bandwidth_per_ms if comm_data > 0 else 0.0
        t_sep = t_p1 + t_p2 + t_comm
        
        # Merge if merged execution is faster than separated
        return t_merged <= t_sep
    
    def run(self):
        """
        Algorithm 2: Graph Partitioning
        
        Creates partitions by:
        1. Selecting mergeable edges
        2. Merging adjacent layers/partitions based on merge_check
        3. Handling orphan nodes
        
        Returns:
            list: List of Partition objects
        """
        # Step 1: Get mergeable edges
        edges_M = self._select_edges_for_partitioning()
        
        partitions = []
        self.node_to_partition = {}
        
        # Step 2: Process mergeable edges
        for (u, v) in edges_M:
            pu = self.node_to_partition.get(u)
            pv = self.node_to_partition.get(v)
            
            # Case 1: Both nodes unassigned -> create new partition
            if pu is None and pv is None:
                new_part = Partition(len(partitions), [self.layers_map[u], self.layers_map[v]])
                partitions.append(new_part)
                self.node_to_partition[u] = new_part
                self.node_to_partition[v] = new_part
            
            # Case 2: Both in different partitions -> try to merge
            elif pu is not None and pv is not None and pu != pv:
                if self._merge_check(pu, pv):
                    # Merge pv into pu
                    pu.layers.extend(pv.layers)
                    pu.total_memory += pv.total_memory
                    pu.total_workload += pv.total_workload
                    
                    # Update mappings
                    for layer in pv.layers:
                        self.node_to_partition[layer.id] = pu
                    
                    # Remove pv from partitions
                    partitions.remove(pv)
            
            # Case 3: One assigned, one not -> try to add to existing
            elif pu is not None or pv is not None:
                existing = pu if pu is not None else pv
                other_id = v if pu is not None else u
                
                if other_id not in self.node_to_partition:
                    # Create temp partition for the single layer
                    temp_part = Partition(-1, [self.layers_map[other_id]])
                    
                    if self._merge_check(existing, temp_part):
                        # Add layer to existing partition
                        existing.layers.append(self.layers_map[other_id])
                        existing.total_memory += self.layers_map[other_id].memory
                        existing.total_workload += self.layers_map[other_id].workload
                        self.node_to_partition[other_id] = existing
        
        # Step 3: Handle orphan nodes (nodes not in any partition)
        for node_id in self.G.nodes():
            if node_id not in self.node_to_partition:
                orphan_part = Partition(len(partitions), [self.layers_map[node_id]])
                partitions.append(orphan_part)
                self.node_to_partition[node_id] = orphan_part
        
        # Step 4: Renumber partitions to have consecutive IDs (0, 1, 2, ...)
        # This is necessary because merging removes some partitions, leaving gaps
        for new_id, p in enumerate(partitions):
            p.id = new_id
        
        return partitions
    
    def _build_partition_graph(self, partitions):
        """
        Build partition dependency graph from original DAG
        
        Args:
            partitions: List of Partition objects
            
        Returns:
            nx.DiGraph: Graph where nodes are partition IDs and edges are dependencies
        """
        partition_graph = nx.DiGraph()
        
        # Add all partition nodes
        for p in partitions:
            partition_graph.add_node(p.id)
        
        # Add edges between partitions based on layer dependencies
        for u, v in self.G.edges():
            pu = self.node_to_partition[u]
            pv = self.node_to_partition[v]
            
            # Only add edge if partitions are different
            if pu.id != pv.id:
                partition_graph.add_edge(pu.id, pv.id)
        
        return partition_graph
    
    def _compute_partition_priority(self, partition, partition_graph, partitions_list, memo):
        """
        Algorithm 3: Compute Partition Priority (Formula 11)
        
        Priority(p) = T(p) + C(p, succ) + max(Priority(succ))
        
        Args:
            partition: Partition object
            partition_graph: Dependency graph of partitions
            partitions_list: List of all partitions (indexed by ID)
            memo: Memoization dict
            
        Returns:
            float: Priority value
        """
        # Check memo
        if partition.id in memo:
            return memo[partition.id]
        
        # Average server power for estimation
        avg_power = sum(s.power_ratio for s in self.servers) / len(self.servers)
        
        # Base execution time (with potential penalty)
        penalty = calculate_penalty(partition.total_memory)
        t_exec = (partition.total_workload * penalty) / avg_power
        
        # Get successors
        successors = list(partition_graph.successors(partition.id))
        
        # Base case: no successors
        if not successors:
            memo[partition.id] = t_exec
            return t_exec
        
        # Recursive case: max successor priority + communication
        max_succ_priority = max(
            self._compute_partition_priority(partitions_list[succ_id], partition_graph, partitions_list, memo)
            for succ_id in successors
        )
        
        # Communication time estimation (use average edge weight)
        comm_data = 0.0
        succ_count = 0
        for succ_id in successors:
            succ = partitions_list[succ_id]
            for layer1 in partition.layers:
                for layer2 in succ.layers:
                    if self.G.has_edge(layer1.id, layer2.id):
                        comm_data += self.G[layer1.id][layer2.id]['weight']
                        succ_count += 1
        
        t_comm = comm_data / self.bandwidth_per_ms if comm_data > 0 else 0.0
        
        priority = t_exec + t_comm + max_succ_priority
        memo[partition.id] = priority
        
        return priority
    
    def schedule(self, partitions):
        """
        Algorithm 3: Priority-based Scheduling
        
        Assigns partitions to servers based on:
        1. Priority (critical path)
        2. Predecessor finish times
        3. Minimum finish time heuristic
        
        Args:
            partitions: List of Partition objects
            
        Returns:
            float: Total inference time (max finish time)
        """
        if not partitions:
            return 0.0
        
        # Build partition dependency graph
        partition_graph = self._build_partition_graph(partitions)
        
        # Create indexed list for quick access
        partitions_list = {p.id: p for p in partitions}
        
        # Compute priorities for all partitions
        memo = {}
        priorities = {}
        for p in partitions:
            priorities[p.id] = self._compute_partition_priority(p, partition_graph, partitions_list, memo)
        
        # Sort partitions by priority (descending)
        sorted_partitions = sorted(partitions, key=lambda p: -priorities[p.id])
        
        # Scheduling state
        server_free_time = {s.id: 0.0 for s in self.servers}
        partition_assignment = {}  # partition_id -> server
        partition_finish = {}      # partition_id -> finish_time
        
        # Assign each partition to the server with minimum finish time
        for p in sorted_partitions:
            best_server = None
            best_finish = float('inf')
            
            for server in self.servers:
                # Calculate ready time (when all predecessors are done + communication)
                ready_time = 0.0
                
                for pred_id in partition_graph.predecessors(p.id):
                    pred_server = partition_assignment[pred_id]
                    pred_finish = partition_finish[pred_id]
                    
                    # Communication or paging overhead
                    if pred_server.id != server.id:
                        # Cross-server: network communication
                        comm_data = 0.0
                        pred = partitions_list[pred_id]
                        for layer1 in pred.layers:
                            for layer2 in p.layers:
                                if self.G.has_edge(layer1.id, layer2.id):
                                    comm_data += self.G[layer1.id][layer2.id]['weight']
                        
                        comm_time = comm_data / self.bandwidth_per_ms
                        ready_time = max(ready_time, pred_finish + comm_time)
                    else:
                        # Same server: SGX context switch paging overhead
                        pred = partitions_list[pred_id]
                        swap_out = min(pred.total_memory, EPC_EFFECTIVE_MB)
                        swap_in = min(p.total_memory, EPC_EFFECTIVE_MB)
                        paging_time = (swap_out + swap_in) / PAGING_BANDWIDTH_MB_PER_MS
                        
                        ready_time = max(ready_time, pred_finish + paging_time)
                
                # Start time is max of server free and data ready
                start_time = max(server_free_time[server.id], ready_time)
                
                # Execution time with penalty
                penalty = calculate_penalty(p.total_memory)
                exec_time = (p.total_workload * penalty) / server.power_ratio
                
                finish_time = start_time + exec_time
                
                # Track best option
                if finish_time < best_finish:
                    best_finish = finish_time
                    best_server = server
            
            # Assign partition to best server
            partition_assignment[p.id] = best_server
            partition_finish[p.id] = best_finish
            server_free_time[best_server.id] = best_finish
        
        # Return max finish time
        return max(partition_finish.values())
