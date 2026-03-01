import networkx as nx
import itertools
from typing import List, Dict, Tuple
from common import Partition, EPC_EFFECTIVE_MB, calculate_penalty, PAGING_BANDWIDTH_MB_PER_MS

class OursAlgorithm:
    """
    Ours Algorithm - Exhaustive Search Implementation
    
    Guarantees global optimal solution by:
    1. Enumerating all valid partition schemes (Bell number)
    2. For each scheme, trying all server assignments
    3. Selecting the scheme+assignment with minimum inference time
    
    WARNING: Exponential time complexity O(Bell(n) × m^p)
    Only suitable for small models (n ≤ 15 layers)
    """
    
    def __init__(self, G, layers_map, servers, bandwidth_mbps):
        self.G = G
        self.layers_map = layers_map
        self.servers = servers
        self.bandwidth_per_ms = (bandwidth_mbps / 8.0) / 1000.0
        
        # Store all partition schemes for exhaustive search in schedule()
        self.all_partition_schemes = []
        
        # Store optimal solution
        self.optimal_partitions = None
        self.optimal_server_assignment = None
        self.optimal_time = float('inf')
        
    def _generate_all_partitions(self, nodes: List[int]) -> List[List[List[int]]]:
        """
        Recursively generate all possible partition schemes (Bell number)
        
        Args:
            nodes: List of node IDs
            
        Returns:
            List of all partition schemes, each scheme is [[node1, node2], [node3], ...]
        """
        if not nodes:
            return [[]]
        
        first = nodes[0]
        rest = nodes[1:]
        
        # Recursively generate partitions for remaining nodes
        rest_partitions = self._generate_all_partitions(rest)
        
        all_partitions = []
        for p in rest_partitions:
            # Option 1: Create new partition with first node
            all_partitions.append([[first]] + p)
            
            # Option 2: Add first node to each existing partition
            for i in range(len(p)):
                new_p = p[:i] + [[first] + p[i]] + p[i + 1:]
                all_partitions.append(new_p)
        
        # Remove duplicates
        unique_partitions = []
        seen = set()
        for p in all_partitions:
            # Sort partitions for deduplication
            sorted_p = tuple(tuple(sorted(part)) for part in sorted(p))
            if sorted_p not in seen:
                seen.add(sorted_p)
                unique_partitions.append([list(part) for part in sorted_p])
        
        return unique_partitions
    
    def _is_partition_valid(self, partition_scheme: List[List[int]]) -> bool:
        """
        Check if partition scheme satisfies memory constraints
        
        Args:
            partition_scheme: [[node_ids in partition 0], [node_ids in partition 1], ...]
            
        Returns:
            True if valid (all partitions ≤ EPC limit)
        """
        for part_nodes in partition_scheme:
            total_mem = sum(self.layers_map[node_id].memory for node_id in part_nodes)
            if total_mem > EPC_EFFECTIVE_MB:
                return False
        return True
    
    def _convert_to_partition_objects(self, partition_scheme: List[List[int]]) -> List[Partition]:
        """
        Convert partition scheme (list of node IDs) to Partition objects
        
        Args:
            partition_scheme: [[node_ids], ...]
            
        Returns:
            List of Partition objects
        """
        partitions = []
        for idx, node_ids in enumerate(partition_scheme):
            layer_objs = [self.layers_map[nid] for nid in node_ids]
            p = Partition(idx, layer_objs)
            partitions.append(p)
        return partitions
    
    def _generate_server_assignments(self, num_partitions: int) -> List[Dict[int, int]]:
        """
        Generate all possible server assignment schemes
        
        Args:
            num_partitions: Number of partitions
            
        Returns:
            List of assignment dicts {partition_idx: server_id}
        """
        server_ids = [s.id for s in self.servers]
        
        # Cartesian product: each partition can go to any server
        all_assignments = list(itertools.product(server_ids, repeat=num_partitions))
        
        # Convert to dict format
        assignment_dicts = []
        for assignment in all_assignments:
            assign_dict = {part_idx: server_id 
                          for part_idx, server_id in enumerate(assignment)}
            assignment_dicts.append(assign_dict)
        
        return assignment_dicts
    
    def _calculate_inference_time(self, partitions: List[Partition], 
                                  server_assignment: Dict[int, int]) -> float:
        """
        Calculate total inference time for a specific partition+assignment scheme
        
        Args:
            partitions: List of Partition objects
            server_assignment: {partition_idx: server_id}
            
        Returns:
            Total inference time (ms)
        """
        # Build node to partition mapping
        node_to_part_idx = {}
        for part_idx, p in enumerate(partitions):
            for layer in p.layers:
                node_to_part_idx[layer.id] = part_idx
        
        # Track layer timings
        layer_start = {}
        layer_finish = {}
        
        # Track server availability
        server_max_finish = {s.id: 0.0 for s in self.servers}
        
        # Calculate server memory assignments (for penalty calculation)
        server_memory_sum = {}
        for part_idx, p in enumerate(partitions):
            srv_id = server_assignment[part_idx]
            if srv_id not in server_memory_sum:
                server_memory_sum[srv_id] = 0.0
            server_memory_sum[srv_id] += p.total_memory
        
        # Process layers in topological order
        for node_id in nx.topological_sort(self.G):
            part_idx = node_to_part_idx[node_id]
            server_id = server_assignment[part_idx]
            server = next(s for s in self.servers if s.id == server_id)
            
            # Calculate ready time based on predecessors
            pred_finish_times = []
            for pred_id in self.G.predecessors(node_id):
                pred_finish = layer_finish[pred_id]
                pred_part_idx = node_to_part_idx[pred_id]
                pred_server_id = server_assignment[pred_part_idx]
                
                # Add communication time if cross-server
                if pred_server_id != server_id:
                    edge_data = self.G[pred_id][node_id].get('weight', 1.0)
                    comm_time = edge_data / self.bandwidth_per_ms
                    pred_finish += comm_time
                else:
                    # Same server: add SGX context switch paging overhead
                    pred_part = partitions[pred_part_idx]
                    curr_part = partitions[part_idx]
                    swap_out = min(pred_part.total_memory, EPC_EFFECTIVE_MB)
                    swap_in = min(curr_part.total_memory, EPC_EFFECTIVE_MB)
                    paging_time = (swap_out + swap_in) / PAGING_BANDWIDTH_MB_PER_MS
                    pred_finish += paging_time
                
                pred_finish_times.append(pred_finish)
            
            # Start time = max(predecessor ready, server available)
            pred_ready = max(pred_finish_times) if pred_finish_times else 0.0
            server_ready = server_max_finish[server_id]
            start_time = max(pred_ready, server_ready)
            
            # Calculate execution time with SGX penalty
            layer = self.layers_map[node_id]
            total_mem = server_memory_sum[server_id]
            penalty = calculate_penalty(total_mem)
            exec_time = (layer.workload * penalty) / server.power_ratio
            
            # Update timings
            layer_start[node_id] = start_time
            layer_finish[node_id] = start_time + exec_time
            server_max_finish[server_id] = max(server_max_finish[server_id], 
                                               layer_finish[node_id])
        
        # Total time = max finish time across all layers
        return max(layer_finish.values()) if layer_finish else 0.0
    
    def run(self):
        """
        Generate all valid partition schemes
        
        Returns:
            First partition scheme (actual optimization in schedule())
        """
        # Get all nodes
        nodes = sorted(list(self.G.nodes()))
        
        print(f"\n[Ours Exhaustive Search] Starting partition generation for {len(nodes)} layers...")
        
        # Generate all possible partition schemes
        all_schemes = self._generate_all_partitions(nodes)
        print(f"[Ours] Generated {len(all_schemes)} total partition schemes (Bell number)")
        
        # Filter valid schemes (memory constraint)
        valid_schemes = []
        for scheme in all_schemes:
            if self._is_partition_valid(scheme):
                valid_schemes.append(scheme)
        
        print(f"[Ours] {len(valid_schemes)} schemes satisfy memory constraints")
        
        if not valid_schemes:
            raise ValueError("No valid partition scheme found!")
        
        # Convert to Partition objects and store
        self.all_partition_schemes = [
            self._convert_to_partition_objects(scheme) 
            for scheme in valid_schemes
        ]
        
        # Return first scheme for compatibility
        # Actual optimization happens in schedule()
        return self.all_partition_schemes[0]
    
    def schedule(self, partitions):
        """
        Exhaustively search for optimal server assignment across all partition schemes
        
        Args:
            partitions: Ignored (using stored schemes from run())
            
        Returns:
            Minimum inference time across all schemes
        """
        print(f"\n[Ours Exhaustive Search] Starting exhaustive search...")
        print(f"  Partition schemes to evaluate: {len(self.all_partition_schemes)}")
        
        total_evaluations = 0
        best_time = float('inf')
        best_scheme_idx = -1
        best_assignment = None
        
        # For each partition scheme
        for scheme_idx, partition_scheme in enumerate(self.all_partition_schemes):
            num_partitions = len(partition_scheme)
            
            # Generate all server assignments for this scheme
            assignments = self._generate_server_assignments(num_partitions)
            
            # Evaluate each assignment
            for assignment in assignments:
                total_evaluations += 1
                
                # Calculate inference time
                time = self._calculate_inference_time(partition_scheme, assignment)
                
                # Update best if better
                if time < best_time:
                    best_time = time
                    best_scheme_idx = scheme_idx
                    best_assignment = assignment
                    
                    print(f"  [New Best] Scheme {scheme_idx}, Assignment {assignment} → {time:.2f} ms")
        
        print(f"\n[Ours] Exhaustive search completed!")
        print(f"  Total evaluations: {total_evaluations}")
        print(f"  Optimal partition scheme: {best_scheme_idx}")
        print(f"  Optimal server assignment: {best_assignment}")
        print(f"  Optimal inference time: {best_time:.2f} ms")
        
        # Store optimal solution
        self.optimal_partitions = self.all_partition_schemes[best_scheme_idx]
        self.optimal_server_assignment = best_assignment
        self.optimal_time = best_time
        
        return best_time
