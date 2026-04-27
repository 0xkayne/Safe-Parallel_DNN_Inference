import networkx as nx
from common import (Partition, EPC_EFFECTIVE_MB, calculate_penalty,
                    ScheduleResult, network_latency)


class MEDIAAlgorithm:
    def __init__(self, G, layers_map, servers, bandwidth_mbps):
        self.G = G
        self.layers_map = layers_map
        self.servers = servers
        self.bandwidth_mbps = bandwidth_mbps
        self.bandwidth_per_ms = (bandwidth_mbps / 8.0) / 1000.0
        self.node_to_partition = {}

    # ------------------------------------------------------------------
    # Memory model: paper-faithful sum of tee_total_memory
    # ------------------------------------------------------------------
    def _sum_memory(self, layers):
        """Paper m(P): sum of all layer total memory (conservative, paper-faithful).

        Unlike peak-liveness (which only counts live activations), the paper's
        memory model sums tee_total_memory for every layer in the partition.
        This produces larger estimates (~93 MB partitions on InceptionV3),
        matching Fig.6 behavior where Check() triggers Case B rejections.
        """
        return sum(l.memory for l in layers)

    # ------------------------------------------------------------------
    # Algorithm 1: Edge Selection for MEDIA Partitioning
    # ------------------------------------------------------------------
    def _select_edges_for_partitioning(self):
        """
        Algorithm 1 (paper): Select candidate edges for merging.

        Constraint 1 (line 4): Include edge (u,v) only if |Succ(u)|==1 OR |Pre(v)|==1.
        Constraint 2 (lines 6-9, fork protection): For any successor w of u,
                      if L(u)==L(w)-1 and there exists (w',w) in M, exclude (u,v).

        Traversal order (paper lines 2-3):
          - u: increasing order of level (topological generations)
          - v: following the priority on edges (descending edge weight)

        Returns an ordered list (not set) to preserve insertion order for Algorithm 2.
        """
        M_set = set()   # for O(1) membership check
        M_list = []     # preserves insertion order for Algorithm 2

        # Calculate topological generations for level information
        topological_gen = nx.topological_generations(self.G)
        level_map = {}
        level_nodes = []  # [(level, [nodes...])]
        for level, nodes in enumerate(topological_gen):
            level_nodes.append((level, list(nodes)))
            for node in nodes:
                level_map[node] = level

        # Paper line 2: "for u ∈ V following the increasing order of level"
        for _level, nodes in level_nodes:
            for u in nodes:
                # Paper line 3: "for v ∈ Succ(u) following the priority on edges"
                successors = sorted(
                    self.G.successors(u),
                    key=lambda v: self.G[u][v]['weight'],
                    reverse=True,  # higher weight = higher priority
                )
                for v in successors:
                    # Constraint 1 (line 4): |Pre(v)| == 1 OR |Succ(u)| == 1
                    if self.G.in_degree(v) != 1 and self.G.out_degree(u) != 1:
                        continue

                    M_set.add((u, v))
                    violates = False

                    # Constraint 2 (lines 6-9, fork protection)
                    for w in self.G.successors(u):
                        if level_map[u] != level_map[w] - 1:
                            continue
                        for wp in self.G.predecessors(w):
                            if (wp, w) != (u, v) and (wp, w) in M_set:
                                violates = True
                                break
                        if violates:
                            break

                    # Paper Algorithm 1 line 8: immediately remove if violates
                    if violates:
                        M_set.remove((u, v))
                    else:
                        M_list.append((u, v))
        return M_list

    # ------------------------------------------------------------------
    # Algorithm 2: Memory-aware MEDIA Partitioning
    # ------------------------------------------------------------------
    def _contract_M_edges(self, edges_M):
        """Paper Algorithm 2: Contract M edges with Check().

        Strictly follows the three cases from the paper:
        - Case 1 (line 2-6): Both vertexes not collapsed → Check({u}, {v})
        - Case 2 (line 7-12): Both vertexes collapsed → Check(P, P')
        - Case 3 (line 13-18): One vertex collapsed → Check(P, {w})
        """
        for u, v in edges_M:
            pu = self.node_to_partition[u]
            pv = self.node_to_partition[v]
            if pu == pv:
                continue

            u_collapsed = len(pu.layers) > 1
            v_collapsed = len(pv.layers) > 1

            if not u_collapsed and not v_collapsed:
                # Case 1: Both vertexes are not collapsed
                if self._merge_check(pu, pv):
                    self._merge_partitions(pu, pv)
            elif u_collapsed and v_collapsed:
                # Case 2: Both vertexes are collapsed
                if not self._would_cause_cycle(pu, pv):
                    if self._merge_check(pu, pv):
                        self._merge_partitions(pu, pv)
            else:
                # Case 3: Only one vertex is collapsed
                if not self._would_cause_cycle(pu, pv):
                    if self._merge_check(pu, pv):
                        self._merge_partitions(pu, pv)

    def _merge_partitions(self, p1, p2):
        """Merge two partitions and update node_to_partition mapping."""
        new_layers = list(set(p1.layers + p2.layers))
        new_part = Partition(p1.id, new_layers, self.G)
        for l in new_layers:
            self.node_to_partition[l.id] = new_part

    def _would_cause_cycle(self, p1, p2):
        """Check if merging p1 and p2 would create a cycle in the partition DAG."""
        pg = nx.DiGraph()
        unique_pids = list(set(id(p) for p in self.node_to_partition.values()))
        pg.add_nodes_from(unique_pids)

        for edge_u, edge_v in self.G.edges():
            pu = self.node_to_partition[edge_u]
            pv = self.node_to_partition[edge_v]
            if id(pu) != id(pv):
                pg.add_edge(id(pu), id(pv))

        # Check indirect path p1 -> ... -> p2 (excluding direct edge)
        if pg.has_edge(id(p1), id(p2)):
            pg.remove_edge(id(p1), id(p2))
        if nx.has_path(pg, id(p1), id(p2)):
            return True

        if pg.has_edge(id(p2), id(p1)):
            pg.remove_edge(id(p2), id(p1))
        if nx.has_path(pg, id(p2), id(p1)):
            return True

        return False

    def _merge_check(self, part1, part2):
        """
        Paper Check() function — case-by-case merge decision.

        Uses _sum_memory() model consistently for both partitioning and
        scheduling, matching the paper's m(P) definition.

        1. Always merge if combined memory fits in EPC (no paging).
        2. If it exceeds EPC, merge only if paging cost < communication cost saved.
        """
        merged_layers = list(set(part1.layers + part2.layers))
        merged_mem = self._sum_memory(merged_layers)

        # Case A: Fits in EPC. Always merge.
        if merged_mem <= EPC_EFFECTIVE_MB:
            return True

        # Case B: Memory exceeds EPC.
        # Paper Check(): T(merged) <= T(P1) + T(P1,P2) + T(P2)
        avg_power = sum(s.power_ratio for s in self.servers) / len(self.servers)

        mem_p1 = self._sum_memory(part1.layers)
        mem_p2 = self._sum_memory(part2.layers)
        t_p1 = (part1.total_workload * calculate_penalty(mem_p1)) / avg_power
        t_p2 = (part2.total_workload * calculate_penalty(mem_p2)) / avg_power

        # Communication time if kept separate
        vol = 0.0
        for l1 in part1.layers:
            for l2 in part2.layers:
                if self.G.has_edge(l1.id, l2.id):
                    vol += self.G[l1.id][l2.id]['weight']
                if self.G.has_edge(l2.id, l1.id):
                    vol += self.G[l2.id][l1.id]['weight']

        if len(self.servers) == 1:
            t_comm = 0.0
        else:
            t_comm = network_latency(vol, self.bandwidth_mbps) if vol > 0 else 0.0

        merged_workload = part1.total_workload + part2.total_workload
        t_merged = (merged_workload * calculate_penalty(merged_mem)) / avg_power

        return t_merged <= (t_p1 + t_p2 + t_comm)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def run(self):
        """
        Paper-faithful MEDIA partitioning:
        Algorithm 1: Select M edges (edge selection with Constraint 1 + 2)
        Algorithm 2: Contract M edges with Check() (memory-aware merging)
        No additional merge stage — Algorithm 2 is the only partitioning step.
        """
        # Initialize each layer as its own partition
        self.node_to_partition = {}
        for i, (nid, layer) in enumerate(self.layers_map.items()):
            self.node_to_partition[nid] = Partition(i, [layer], self.G)

        # Algorithm 1: Select candidate edges
        edges_M = self._select_edges_for_partitioning()
        # Algorithm 2: Contract M edges with Check()
        self._contract_M_edges(edges_M)

        # Finalize unique partitions
        unique_parts = list(set(self.node_to_partition.values()))
        for i, p in enumerate(unique_parts):
            p.id = i
        return unique_parts

    # ------------------------------------------------------------------
    # Scheduling (priority-based list scheduling, paper §5)
    # ------------------------------------------------------------------
    def schedule(self, partitions):
        if not partitions:
            return ScheduleResult("MEDIA", 0.0, {}, [])

        partition_graph = nx.DiGraph()
        for p in partitions:
            partition_graph.add_node(p.id)
        for u, v in self.G.edges():
            pu, pv = self.node_to_partition[u], self.node_to_partition[v]
            if pu.id != pv.id:
                partition_graph.add_edge(pu.id, pv.id)

        partitions_list = {p.id: p for p in partitions}
        priorities = self._compute_priorities(partition_graph, partitions_list)
        sorted_partitions = sorted(
            partitions,
            key=lambda p: priorities[p.id],
            reverse=True,
        )

        server_free_time = {s.id: 0.0 for s in self.servers}
        server_schedule = {s.id: [] for s in self.servers}
        assignment, finish = {}, {}

        for p in sorted_partitions:
            best_s, best_ft = None, float('inf')

            for s in self.servers:
                dependency_ready = 0.0
                for pred_id in partition_graph.predecessors(p.id):
                    if pred_id not in assignment:
                        continue
                    pred_s, pred_ft = assignment[pred_id], finish[pred_id]
                    if pred_s.id != s.id:
                        comm_data = sum(
                            self.G[l1.id][l2.id]['weight']
                            for l1 in partitions_list[pred_id].layers
                            for l2 in p.layers
                            if self.G.has_edge(l1.id, l2.id)
                        )
                        comm_time = network_latency(comm_data, self.bandwidth_mbps)
                        dependency_ready = max(dependency_ready, pred_ft + comm_time)
                    else:
                        dependency_ready = max(dependency_ready, pred_ft)

                start_t = max(server_free_time[s.id], dependency_ready)
                # Use the same _sum_memory model as Check() for consistency
                exec_t = (p.total_workload * calculate_penalty(self._sum_memory(p.layers))) / s.power_ratio
                ft = start_t + exec_t

                if ft < best_ft:
                    best_ft, best_s, final_exec_t = ft, s, exec_t

            assignment[p.id], finish[p.id] = best_s, best_ft
            server_free_time[best_s.id] = best_ft
            server_schedule[best_s.id].append({
                'start': best_ft - final_exec_t,
                'end': best_ft,
                'partition_id': p.id,
                'partition': p,
            })

        return ScheduleResult("MEDIA", max(finish.values()), server_schedule, partitions)

    def _compute_priorities(self, partition_graph, partitions_list):
        """
        Paper Eq. 11:
        Priority(P) = max_{P' in succ(P)} { T(P) + T(P,P') + Priority(P') }
        For leaf partitions: Priority(P) = T(P)
        """
        priorities = {}
        avg_p = sum(s.power_ratio for s in self.servers) / len(self.servers)
        topo_order = list(nx.topological_sort(partition_graph))

        for pid in reversed(topo_order):
            partition = partitions_list[pid]
            # Use the same _sum_memory model as Check() for consistency
            t_exec = (partition.total_workload * calculate_penalty(self._sum_memory(partition.layers))) / avg_p

            successors = list(partition_graph.successors(pid))
            if not successors:
                priorities[pid] = t_exec
            else:
                max_val = 0.0
                for sid in successors:
                    comm_data = sum(
                        self.G[l1.id][l2.id]['weight']
                        for l1 in partition.layers
                        for l2 in partitions_list[sid].layers
                        if self.G.has_edge(l1.id, l2.id)
                    )
                    t_comm = network_latency(comm_data, self.bandwidth_mbps) if comm_data > 0 else 0.0
                    val = t_exec + t_comm + priorities.get(sid, 0.0)
                    if val > max_val:
                        max_val = val
                priorities[pid] = max_val

        return priorities
