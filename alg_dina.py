import networkx as nx
from common import (Partition, EPC_EFFECTIVE_MB, calculate_penalty,
                    ScheduleResult, network_latency,
                    DDR_COPY_BW_MB_PER_MS, HMAC_VERIFY_BW_MB_PER_MS,
                    RING_BUFFER_EPC_MB)


class DINAAlgorithm:
    """
    DINA: Distributed Inference Acceleration with Adaptive DNN Partitioning
    and Offloading (IEEE TPDS 2024).

    Adapted from the multi-user fog scenario to single-inference SGX edge
    cluster simulation.  Two phases:

      DINA-P  (run)      – Adaptive partitioning proportional to server
                           compute capabilities, with EPC memory constraint.
      DINA-O  (schedule) – Paper-faithful direct assignment (partition_i →
                           server_i) followed by pairwise swap-matching
                           refinement for two-sided exchange stability.
    """

    def __init__(self, G, layers_map, servers, bandwidth_mbps):
        self.G = G
        self.layers_map = layers_map
        self.servers = servers
        self.bandwidth_mbps = bandwidth_mbps

    # ------------------------------------------------------------------
    # DINA-P: Workload-Proportional Partitioning
    # ------------------------------------------------------------------
    def run(self):
        """
        Partition the DNN into k = 2 sub-tasks, sizing each proportionally
        to server compute capability.

        DINA's core idea is workload-proportional partitioning (Algorithm 1,
        TPDS 2024).  We fix k = 2 as a principled balance: one partition
        captures too little of DINA's behaviour, while k = len(servers)
        creates excessive inter-partition communication in TEE edge networks.
        Two partitions with one communication hop is the simplest design
        that demonstrates DINA's distribution overhead.
        """
        return self._partition_for_k(min(2, len(self.servers)))

    def _partition_for_k(self, k):
        """
        Partition the DNN DAG into exactly *k* sub-tasks whose workload is
        proportional to the compute capability of the first *k* servers.

        Paper mapping (Algorithm 1, line 3):
            ρ_i = Σ_{j<i} (c_j / φ_{fa}) / Σ_{j=0..F} (c_j / φ_{fa})
        """
        topo_order = list(nx.topological_sort(self.G))
        servers_k = self.servers[:k]

        total_power = sum(s.power_ratio for s in servers_k)
        total_workload = sum(self.layers_map[n].workload for n in topo_order)

        targets = [total_workload * s.power_ratio / total_power for s in servers_k]

        partitions = []
        current_layers = []
        current_workload = 0.0
        slot_idx = 0

        for node_id in topo_order:
            layer = self.layers_map[node_id]
            current_layers.append(layer)
            current_workload += layer.workload

            target = targets[slot_idx] if slot_idx < len(targets) else targets[-1]
            if current_workload >= target and slot_idx < k - 1:
                partitions.append(
                    Partition(len(partitions), current_layers, self.G))
                current_layers = []
                current_workload = 0.0
                slot_idx += 1

        if current_layers:
            partitions.append(
                Partition(len(partitions), current_layers, self.G))

        return partitions

    # ------------------------------------------------------------------
    # Helper: compute the cost of running a partition on a given server
    # ------------------------------------------------------------------
    def _partition_cost(self, partition, server):
        """
        Execution cost (ms) — unified weights-outside-EPC model (§4.4.3).

        Per-layer three-thread pipeline: max(T_compute, T_load_ddr, T_hmac).
        Paging penalty on activation memory only (weights loaded via OCALL).
        """
        weight_mb = partition.get_static_memory()
        peak_act = partition.total_memory - weight_mb
        penalty = calculate_penalty(peak_act + RING_BUFFER_EPC_MB)
        total = 0.0
        for layer in partition.layers:
            t_comp = (layer.workload * penalty) / server.power_ratio
            t_load = layer.weight_memory / DDR_COPY_BW_MB_PER_MS
            t_hash = layer.weight_memory / HMAC_VERIFY_BW_MB_PER_MS
            total += max(t_comp, t_load, t_hash)
        return total

    # ------------------------------------------------------------------
    # Helper: communication cost between two partitions on different servers
    # ------------------------------------------------------------------
    def _comm_cost(self, src_partition, dst_partition, same_server):
        """
        Data transfer cost (ms) when *dst_partition* depends on layers in
        *src_partition*.  Returns 0 if both run on the same server.
        """
        if same_server:
            return 0.0

        # Sum edge weights (output data in MB) for cross-partition edges
        vol = 0.0
        for u_layer in src_partition.layers:
            for v_layer in dst_partition.layers:
                if self.G.has_edge(u_layer.id, v_layer.id):
                    vol += self.G[u_layer.id][v_layer.id]['weight']

        if vol <= 0:
            return 0.0
        return network_latency(vol, self.bandwidth_mbps)

    # ------------------------------------------------------------------
    # Helper: find which partitions a given partition depends on
    # ------------------------------------------------------------------
    def _get_predecessor_partitions(self, partitions, target_partition):
        """Return set of partition indices that have edges into target_partition."""
        target_ids = {l.id for l in target_partition.layers}
        pred_indices = set()
        for idx, p in enumerate(partitions):
            if p.id == target_partition.id:
                continue
            for layer in p.layers:
                for succ in self.G.successors(layer.id):
                    if succ in target_ids:
                        pred_indices.add(idx)
        return pred_indices

    # ------------------------------------------------------------------
    # DINA-O: Swap-Matching Offloading
    # ------------------------------------------------------------------
    def schedule(self, partitions):
        """
        Two-phase scheduling adapted from DINA-O (Algorithm 2):

        Phase 1 – Direct initial assignment (paper-faithful):
            Assign partition_i to server_i, matching the DINA-P
            partitioning intent where partition_i's workload was
            sized proportionally to server_i's compute power.

        Phase 2 – Swap-matching refinement:
            Iteratively try pairwise swaps of server assignments.  A swap is
            accepted if it reduces the overall makespan (= max server
            completion time).  Converges when no swap improves the makespan,
            yielding a two-sided exchange-stable matching (Def. 4 in paper).

        The utility concept from Eq. (5) / Eq. (8) is mapped to:
            utility  =  −makespan  (lower latency → higher utility)
        so improving utility ≡ reducing makespan.
        """
        if not partitions:
            return ScheduleResult("DINA", 0.0, {}, [])

        n_parts = len(partitions)
        n_servers = len(self.servers)

        # Build a lightweight DAG of inter-partition dependencies
        pred_map = {}  # partition_index -> set of predecessor partition indices
        for i, p in enumerate(partitions):
            pred_map[i] = self._get_predecessor_partitions(partitions, p)

        # ---- Phase 1: Direct assignment (paper-faithful DINA-O) ----
        # DINA-P sizes partition_i proportionally to server_i's power,
        # so the natural initial assignment is partition_i → server_i.
        assignment = [i % n_servers for i in range(n_parts)]

        # Build partition DAG for topological ordering
        part_dag = nx.DiGraph()
        for i in range(n_parts):
            part_dag.add_node(i)
            for pred_i in pred_map[i]:
                part_dag.add_edge(pred_i, i)
        part_topo = list(nx.topological_sort(part_dag))

        # Compute initial finish_times via simulation
        _, finish_times_list = self._evaluate_makespan(
            partitions, assignment, pred_map, part_topo, return_times=True)
        finish_times = list(finish_times_list)

        # ---- Phase 2: Swap-matching refinement ----
        # Repeatedly try all pairwise swaps; accept if makespan decreases.
        MAX_ROUNDS = 50  # safety cap

        for _ in range(MAX_ROUNDS):
            improved = False
            current_makespan = self._evaluate_makespan(
                partitions, assignment, pred_map, part_topo)

            for i in range(n_parts):
                if improved:
                    break
                for j in range(i + 1, n_parts):
                    if assignment[i] == assignment[j]:
                        continue  # already on same server, swap is no-op

                    # Try swapping
                    assignment[i], assignment[j] = assignment[j], assignment[i]
                    new_makespan = self._evaluate_makespan(
                        partitions, assignment, pred_map, part_topo)

                    if new_makespan < current_makespan - 1e-9:
                        # Accept swap
                        improved = True
                        current_makespan = new_makespan
                        break
                    else:
                        # Revert swap
                        assignment[i], assignment[j] = assignment[j], assignment[i]

            if not improved:
                break

        # ---- Build final schedule result ----
        final_makespan, finish_times = self._evaluate_makespan(
            partitions, assignment, pred_map, part_topo, return_times=True)

        server_schedule = {s.id: [] for s in self.servers}
        for pi in range(n_parts):
            si = assignment[pi]
            server = self.servers[si]
            exec_cost = self._partition_cost(partitions[pi], server)
            end_t = finish_times[pi]
            start_t = end_t - exec_cost
            server_schedule[server.id].append({
                'start': start_t,
                'end': end_t,
                'partition_id': partitions[pi].id,
                'partition': partitions[pi]
            })

        return ScheduleResult("DINA", final_makespan, server_schedule, partitions)

    # ------------------------------------------------------------------
    # Helper: evaluate makespan for a given assignment
    # ------------------------------------------------------------------
    def _evaluate_makespan(self, partitions, assignment, pred_map, part_topo,
                           return_times=False):
        """
        Simulate the schedule implied by *assignment* and return the
        makespan.  If return_times is True, also return per-partition
        finish times.

        Uses the unified weights-outside-EPC cost model.
        """
        n_parts = len(partitions)
        n_servers = len(self.servers)
        finish_times = [0.0] * n_parts
        server_available = [0.0] * n_servers

        for pi in part_topo:
            p = partitions[pi]
            si = assignment[pi]
            server = self.servers[si]

            # Data-ready time from predecessors
            data_ready = 0.0
            for pred_i in pred_map[pi]:
                pred_finish = finish_times[pred_i]
                comm = self._comm_cost(
                    partitions[pred_i], p,
                    same_server=(assignment[pred_i] == si))
                data_ready = max(data_ready, pred_finish + comm)

            start = max(server_available[si], data_ready)
            exec_cost = self._partition_cost(p, server)
            finish = start + exec_cost

            finish_times[pi] = finish
            server_available[si] = finish

        makespan = max(finish_times) if finish_times else 0.0

        if return_times:
            return makespan, finish_times
        return makespan
