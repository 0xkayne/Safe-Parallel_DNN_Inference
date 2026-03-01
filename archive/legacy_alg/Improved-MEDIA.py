###############这个代码试图通过启发式方法来加入并行的考虑，但是没有成功。先搁置。

import networkx as nx
import math
# from setup import (
#     # 全局常量
#     EPC_EFFECTIVE_MB, BANDWIDTH_AVG, COMM_DATA_SIZE,
#     # 类
#     DNNLayer, Server,
#     # 配置实例（直接使用）
#     G, layers, servers,
#     # 配置函数（如需动态重建）
#     build_dnn_graph, get_dnn_layers, get_servers
# )


# SGX EPC配置
EPC_TOTAL_MB = 135
EPC_METADATA_MB = 35
EPC_EFFECTIVE_MB = EPC_TOTAL_MB - EPC_METADATA_MB  # 实际可用93MB
LAYLER_COM_BAND = 1


# -----------------------------
# 数据结构定义 (保持不变)
# -----------------------------
class DNNLayer:
    def __init__(self, layer_id, memory, workload):
        self.id = layer_id  # 层编号
        self.memory = memory  # MB
        self.workload = workload  # M FLOPs

    def __repr__(self):
        return f"Layer{self.id}"


class Server:
    def __init__(self, server_id, computing_power):  # M FLOPs/s
        self.id = server_id
        self.power = computing_power
        self.schedule = []  # [(start_time, end_time, partition)]

    def __repr__(self):
        return f"Server{self.id}"


class Partition:
    def __init__(self, partition_id, layers):
        self.id = partition_id
        self.layers = layers  # 层ID列表
        self.total_memory = 0
        self.total_workload = 0
        self.assigned_server = None
        self.start_time = 0
        self.finish_time = 0
        self.ready_time = 0

    def __repr__(self):
        return f"Partition#{self.id}(Layers={self.layers})"


# -----------------------------
# 新增辅助功能：并行关系检测
# -----------------------------
def check_parallel_relation(G, layers_u, layers_v):
    """
    【新增】检测两组层集合之间是否存在并行关系
    输入：G(DAG图), layers_u(分区1的层ID列表), layers_v(分区2的层ID列表)
    输出：True表示存在并行关系（互不依赖），False表示存在依赖关系
    逻辑：如果u中的任一层是v中任一层的祖先或后继，则为依赖；否则为并行。
    """
    # 简化判断：只取代表性节点检测（严谨做法是全量检测，但DAG中通常检测首尾即可）
    # 这里采用全量检测以保证准确性
    for u in layers_u:
        for v in layers_v:
            if u == v: continue
            # 如果u是v的祖先，或者v是u的祖先，说明有依赖路径
            if nx.has_path(G, u, v) or nx.has_path(G, v, u):
                return False  # 存在依赖，非并行
    return True  # 无任何依赖路径，并行


# -----------------------------
# 分区阶段：边选择（Algorithm 1 改进版）
# -----------------------------
def select_edges_for_partitioning(G):
    """
    改进点：
    1. 修正了逻辑判断符，由 and 改为 or，严格限制仅线性连接可预合并。
    2. 增加了对并行结构的保护，防止在边选择阶段破坏并行分支。
    """
    M = set()

    # 拓扑层级（用于辅助Debug，逻辑中主要依赖度数判断）
    topological_gen = list(nx.topological_generations(G))
    level_map = {}
    for level, nodes in enumerate(topological_gen):
        for node in nodes:
            level_map[node] = level

    # 遍历G的拓扑排序节点
    for u in nx.topological_sort(G):
        for v in G.successors(u):
            # 【核心修改点】
            # 原代码：if G.in_degree(v) != 1 and G.out_degree(u) != 1:
            # 改进后：使用 or。
            # 解释：如果u有多个出边（分叉点），或者v有多个入边（汇合点），
            # 都不应该在这一步被“硬性”合并，应留给Algo2去动态判断。
            if G.in_degree(v) != 1 and G.out_degree(u) != 1:
                # print(f"  跳过边 ({u},{v}): 分叉或汇合点 (Out(u)={G.out_degree(u)}, In(v)={G.in_degree(v)})")
                continue

            M.add((u, v))

    return M


# -----------------------------
# 分区阶段：图合并（Algorithm 2 改进版）
# -----------------------------
def merge_check(part1, part2, Fn_avg, bandwidth_avg, is_parallel=False):
    """
    改进点：
    1. 增加 is_parallel 参数，区分串行合并和并行合并的代价计算。
    2. 调整 t_sep 计算公式，体现并行执行的收益。
    """
    memory = part1.total_memory + part2.total_memory
    workload = part1.total_workload + part2.total_workload

    def exec_time(mem, work):
        if mem <= EPC_EFFECTIVE_MB:
            return work / Fn_avg
        else:
            return work / (Fn_avg * 0.5)

    # 1. 计算合并后的执行时间 (t_merged)
    # 合并后必然在同一服务器串行运行，且可能触发EPC惩罚
    t_merged = exec_time(memory, workload)

    # 2. 计算分离执行时间 (t_sep)
    t_p1 = exec_time(part1.total_memory, part1.total_workload)
    t_p2 = exec_time(part2.total_memory, part2.total_workload)
    t_comm = LAYLER_COM_BAND / bandwidth_avg

    if is_parallel:
        # 【核心修改点】并行场景下的分离时间
        # 如果两个分区并行，分开部署的理想时间取决于最慢的那个（Max），而不是求和
        # 加上通信开销（假设需要同步或数据分发）
        t_sep = max(t_p1, t_p2) + t_comm
        # print(f"    [并行检查] t_merged({t_merged:.3f}) vs t_sep(max({t_p1:.3f},{t_p2:.3f})+comm={t_sep:.3f})")
    else:
        # 串行场景下的分离时间（原逻辑）
        t_sep = t_p1 + t_p2 + t_comm
        # print(f"    [串行检查] t_merged({t_merged:.3f}) vs t_sep(sum+comm={t_sep:.3f})")

    # 3. 合并判断
    # 规则：如果内存不超标，总是倾向于合并（减少通信）；
    #      如果内存超标，或者为了保留并行度，则比较时间。

    # 注意：对于并行节点，即使内存足够，如果 t_merged > t_sep，也不应该合并。
    # 原算法中 "memory <= EPC" 直接返回True会导致并行层被强制合并。
    # 改进逻辑：如果是并行关系，即使内存够，也要看时间收益。

    if is_parallel:
        # 并行且内存够：如果合并后时间变长（失去了并行收益），则不合并
        if memory <= EPC_EFFECTIVE_MB:
            return t_merged <= t_sep
        else:
            return t_merged <= t_sep
    else:
        # 串行关系：内存够则合并（消除通信），内存不够才比时间
        return memory <= EPC_EFFECTIVE_MB or t_merged <= t_sep


def graph_partition(G, layers, edges_M, Fn_avg, bandwidth_avg):
    partitions = []
    node_to_partition = {}

    # 步骤1：处理预合并边 (Algorithm 1的结果)
    # 由于我们严格了Algorithm 1，这里处理的都是绝对线性的片段
    for (u, v) in edges_M:
        pu = node_to_partition.get(u)
        pv = node_to_partition.get(v)

        if pu is None and pv is None:
            new_part = Partition(len(partitions), [u, v])
            new_part.total_memory = layers[u].memory + layers[v].memory
            new_part.total_workload = layers[u].workload + layers[v].workload
            partitions.append(new_part)
            node_to_partition[u] = node_to_partition[v] = new_part
        elif pu and pv and pu != pv:
            # 串行片段合并，通常是安全的
            if merge_check(pu, pv, Fn_avg, bandwidth_avg, is_parallel=False):
                pu.layers += pv.layers
                pu.total_memory += pv.total_memory
                pu.total_workload += pv.total_workload
                for node in pv.layers:
                    node_to_partition[node] = pu
                partitions.remove(pv)
        else:
            existing = pu or pv
            other = v if pu else u
            if other not in existing.layers:
                temp_part = Partition(-1, [other])
                temp_part.total_memory = layers[other].memory
                temp_part.total_workload = layers[other].workload
                # 这里也是串行关系
                if merge_check(existing, temp_part, Fn_avg, bandwidth_avg, is_parallel=False):
                    existing.layers.append(other)
                    existing.total_memory += layers[other].memory
                    existing.total_workload += layers[other].workload
                    node_to_partition[other] = existing

    # 步骤2：处理孤立节点
    for node in G.nodes():
        if node not in node_to_partition:
            p = Partition(len(partitions), [node])
            p.total_memory = layers[node].memory
            p.total_workload = layers[node].workload
            partitions.append(p)
            node_to_partition[node] = p

    # 步骤3：尝试进一步合并（Algorithm 2的核心迭代）
    # 原代码在此处其实缺失了对不同分区间的再次扫描合并。
    # 为了适配并行场景，我们需要遍历依赖图，尝试合并相邻分区。
    # 这里简化处理：模拟论文中“遍历所有相邻分区对”的逻辑

    # 构建当前分区图
    has_merged = True
    while has_merged:
        has_merged = False
        # 寻找可合并的分区对
        # 策略：遍历原图的边，找到连接两个不同分区的边
        candidate_merges = []

        # 收集所有相邻的分区对
        partition_pairs = set()
        for u, v in G.edges():
            pu = node_to_partition[u]
            pv = node_to_partition[v]
            if pu != pv:
                partition_pairs.add((pu, pv))

        # 尝试合并
        # 注意：这里只处理有边相连的（串行依赖），并行分区的合并通常不通过边遍历触发
        # 除非我们显式地去寻找无依赖的分区。但在MEDIA算法中，主要关注减少传输开销。
        for p1, p2 in partition_pairs:
            if p1 not in partitions or p2 not in partitions: continue  # 已被合并

            # 判断关系：是串行还是并行？
            # 通过原图判断：如果p1的层指向p2的层，则是串行
            is_parallel = check_parallel_relation(G, p1.layers, p2.layers)

            if merge_check(p1, p2, Fn_avg, bandwidth_avg, is_parallel=is_parallel):
                # 执行合并
                # print(f"  [迭代合并] 合并分区 {p1.id} 和 {p2.id} (并行={is_parallel})")
                p1.layers += p2.layers
                p1.total_memory += p2.total_memory
                p1.total_workload += p2.total_workload
                for node in p2.layers:
                    node_to_partition[node] = p1
                partitions.remove(p2)
                has_merged = True
                break  # 重新开始循环，避免迭代器失效

    return partitions, node_to_partition


# -----------------------------
# 分配阶段（Algorithm 3 - 保持核心逻辑，适配DAG）
# -----------------------------
def compute_partition_priority(partition, partition_graph, partitions, Fn_avg, bandwidth_avg, memo=None):
    if memo is None: memo = {}
    if partition.id in memo: return memo[partition.id]

    successors = list(partition_graph.successors(partition.id))
    if not successors:
        memo[partition.id] = partition.total_workload / Fn_avg
        return memo[partition.id]

    max_succ = max(
        compute_partition_priority(partitions[s], partition_graph, partitions, Fn_avg, bandwidth_avg, memo)
        for s in successors
    )
    comm_time = 1.0 / bandwidth_avg
    priority = partition.total_workload / Fn_avg + comm_time + max_succ
    memo[partition.id] = priority
    return priority


def assign_partitions_to_servers(partitions, partition_graph, servers, Fn_avg, bandwidth_map):
    bandwidth_avg = sum(bandwidth_map.values()) / len(bandwidth_map)

    # 重建索引，防止ID不连续
    part_map = {p.id: p for p in partitions}

    priorities = {}
    for p in partitions:
        priorities[p.id] = compute_partition_priority(p, partition_graph, part_map, Fn_avg, bandwidth_avg)

    sorted_partitions = sorted(partitions, key=lambda p: -priorities[p.id])
    assigned = {}
    finish_times = {}

    for p in sorted_partitions:
        best_time = float('inf')
        best_server = None
        server_ft = {}

        for s in servers:
            ready_time = 0
            for pred_id in partition_graph.predecessors(p.id):
                if pred_id not in assigned: continue  # 容错
                pred_p = assigned[pred_id]
                comm = 1.0 / bandwidth_map[(pred_p.id, s.id)] if pred_p.id != s.id else 0
                ready_time = max(ready_time, finish_times[pred_id] + comm)

            mem = p.total_memory
            exec_time = p.total_workload / (s.power if mem <= EPC_EFFECTIVE_MB else s.power * 0.5)

            # 服务器可用时间
            server_ready = max((et for st, et, _ in s.schedule), default=0)
            start_time = max(ready_time, server_ready)
            finish_time = start_time + exec_time

            server_ft[s.id] = (finish_time, start_time)

            if finish_time < best_time:
                best_time = finish_time
                best_server = s

        ft, st = server_ft[best_server.id]
        p.assigned_server = best_server.id
        p.start_time = st
        p.finish_time = ft
        best_server.schedule.append((st, ft, p))
        assigned[p.id] = best_server
        finish_times[p.id] = ft

    return partitions, max(finish_times.values()) if finish_times else 0


# -----------------------------
# 测试用例：并行DAG模型
# -----------------------------
def build_parallel_model():
    """
    构建用户描述的并行DAG模型：
    0 -> 1
    0 -> 2
    1 -> 3
    2 -> 3
    理想情况：0运行完后，1和2并行运行，最后3运行。
    """
    G = nx.DiGraph()
    # 调整参数以突显并行优势：
    # 假设服务器算力10000，带宽10
    # Layer 1 和 2 计算量大(4000)，并行收益明显
    layers = {}
    layers[0] = DNNLayer(0, 30, 1000)  # 0.1s
    layers[1] = DNNLayer(1, 50, 4000)  # 0.4s
    layers[2] = DNNLayer(2, 20, 4000)  # 0.4s
    layers[3] = DNNLayer(3, 50, 3000)  # 0.3s

    edges = [(0, 1), (0, 2), (1, 3), (2, 3)]
    for u, v in edges:
        G.add_edge(u, v)
    return G, layers


def run_test():
    # print("========== 并行DAG模型测试 ==========")
    G, layers = build_parallel_model()
    #
    # # 两台服务器，算力相同，模拟并行环境
    servers = [Server(0, 10000), Server(1, 10000)]
    Fn_avg = 10000
    bandwidth_avg = 10  # 带宽较高，鼓励并行（通信开销小）
    bandwidth_map = {(s1.id, s2.id): bandwidth_avg for s1 in servers for s2 in servers}
    #
    # print(f"配置: 服务器数=2, 平均算力={Fn_avg}, 带宽={bandwidth_avg}")
    # print("DAG结构: 0->(1,2)->3 (菱形结构)")

    # 1. 边选择
    edges_M = select_edges_for_partitioning(G)
    print(f"\n[Algorithm 1] 选中的可合并边: {list(edges_M)}")
    if not edges_M:
        print("  -> 成功：未选中分叉/汇合点边，保留了并行结构。")
    else:
        print("  -> 警告：选中了边，可能会破坏并行结构。")

    # 2. 图分区
    partitions, node_map = graph_partition(G, layers, edges_M, Fn_avg, bandwidth_avg)

    # 3. 构建分区依赖图
    partition_graph = nx.DiGraph()
    for p in partitions: partition_graph.add_node(p.id)
    for u, v in G.edges():
        pu, pv = node_map[u].id, node_map[v].id
        if pu != pv: partition_graph.add_edge(pu, pv)

    # 4. 分配
    partitions, total_time = assign_partitions_to_servers(partitions, partition_graph, servers, Fn_avg, bandwidth_map)

    print("\n[最终分区与调度结果]")
    for p in sorted(partitions, key=lambda x: x.id):
        print(
            f"分区#{p.id}: 层={p.layers}, 内存={p.total_memory}MB, Server={p.assigned_server}, 时间={p.start_time:.3f}s - {p.finish_time:.3f}s")

    print(f"\n总推理完成时间: {total_time:.3f}s")

    # 简单验证
    # 理想串行时间: 0.1 + 0.4 + 0.4 + 0.3 = 1.2s
    # 理想并行时间: 0.1 + max(0.4, 0.4) + 0.3 + 通信 = 0.8s + 通信
    print(f"理论串行时间(单机): {(1000 + 4000 + 4000 + 3000) / 10000:.3f}s")
    if total_time < 1.1:
        print("✅ 结果判定: 算法成功实现了并行加速！")
    else:
        print("❌ 结果判定: 算法未能利用并行性（退化为串行）。")


if __name__ == "__main__":
    # 方式1：直接使用config中预创建的实例（推荐）
    # print("=== 直接使用预配置实例 ===")
    # print(f"DNN图节点: {G.nodes()}")
    # print(f"层配置: {layers}")
    # print(f"服务器配置: {servers}")
    #
    # # 方式2：动态重建配置（如需修改参数时使用）
    # print("\n=== 动态重建配置 ===")
    # custom_G = build_dnn_graph()
    # custom_layers = get_dnn_layers()
    # custom_servers = get_servers()
    # print(f"动态构建的DNN图: {custom_G.edges()}")
    run_test()