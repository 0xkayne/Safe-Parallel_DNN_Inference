import networkx as nx
import math
from setup import (
    # 全局常量
    EPC_EFFECTIVE_MB, BANDWIDTH_AVG, COMM_DATA_SIZE,SWITCH_OVERHEAD,
    # 类
    DNNLayer, Server,
    # 配置实例（直接使用）
    G, layers, servers,
    # 配置函数（如需动态重建）
    build_dnn_graph, get_dnn_layers, get_servers
)

# SGX EPC配置
# EPC_TOTAL_MB = 135
# EPC_METADATA_MB = 35
# EPC_EFFECTIVE_MB = EPC_TOTAL_MB - EPC_METADATA_MB  # 实际可用93MB
# -----------------------------
# 数据结构定义
# -----------------------------
# class DNNLayer:
#     """
#         论文中DNN层的抽象模型（对应层集合V中的单个节点v∈V）
#         每个层包含唯一标识、内存占用、计算量三个核心属性
#         """
#     def __init__(self, layer_id, memory, workload):
#         self.id = layer_id          # 层编号
#         self.memory = memory        # MB
#         self.workload = workload    # M FLOPs

# class Server:
#     """
#         论文中边缘服务器/计算节点的抽象模型（对应服务器集合S中的单个节点s∈S）
#         每个服务器包含唯一标识、算力、调度队列三个核心属性
#         """
#     def __init__(self, server_id, computing_power):  # M FLOPs/s
#         self.id = server_id
#         self.power = computing_power
#         self.schedule = []  # [(start_time, end_time, partition)]

class Partition:
    """
      论文中DNN层分区的抽象模型（对应分区集合P中的单个分区p∈P）
      每个分区是若干连续DNN层的集合，是调度和分配的基本单位
      """
    def __init__(self, partition_id, layers):
        self.id = partition_id
        self.layers = layers  # 层ID集合
        self.total_memory = 0
        self.total_workload = 0
        self.assigned_server = None
        self.start_time = 0
        self.finish_time = 0
        self.ready_time = 0

# -----------------------------
# 通用计算函数（论文公式）
# -----------------------------
def get_edge_data_size(G, u, v):
    return G.edges[u, v].get("data_size", COMM_DATA_SIZE)


def compute_fn(server, mem):
    """论文公式(1)：F_n(m)"""
    return server.power if mem <= EPC_EFFECTIVE_MB else server.power * SWITCH_OVERHEAD


def avg_compute_capacity(servers, mem):
    """论文公式(9)分母：sum_n F_n(m) / |N|"""
    return sum(compute_fn(s, mem) for s in servers) / len(servers)


def partition_exec_time(workload, mem, servers):
    return workload / avg_compute_capacity(servers, mem)


# -----------------------------
# 通用计算函数（论文公式）
# -----------------------------
def get_edge_data_size(G, u, v):
    return G.edges[u, v].get("data_size", COMM_DATA_SIZE)


def compute_fn(server, mem):
    """论文公式(1)：F_n(m)"""
    return server.power if mem <= EPC_EFFECTIVE_MB else server.power * SWITCH_OVERHEAD


def avg_compute_capacity(servers, mem):
    """论文公式(9)分母：sum_n F_n(m) / |N|"""
    return sum(compute_fn(s, mem) for s in servers) / len(servers)


def partition_exec_time(workload, mem, servers):
    return workload / avg_compute_capacity(servers, mem)


def avg_bandwidth_denominator(servers, bandwidth_map):
    """
    论文公式(10)分母：sum_{m,n} B_mn / |N|
    注意：bandwidth_map 中的值是 Mbps，需要转换为 bytes/s
    1 Mbps = 1e6 bits/s = 1e6 / 8 bytes/s = 125000 bytes/s
    """
    total = 0
    for m in servers:
        for n in servers:
            # 将 Mbps 转换为 bytes/s
            bandwidth_bytes_per_sec = bandwidth_map[(m.id, n.id)] * 1e6 / 8
            total += bandwidth_bytes_per_sec
    return total / len(servers)


def partition_comm_time(data_size, servers, bandwidth_map):
    return data_size / avg_bandwidth_denominator(servers, bandwidth_map)


def inter_partition_data(G, part_a_layers, part_b_layers):
    data_size = 0
    for u in part_a_layers:
        for v in part_b_layers:
            if G.has_edge(u, v):
                data_size += get_edge_data_size(G, u, v)
    return data_size


def would_create_cycle(G, layers1, layers2, node_to_partition):
    """
    检查合并 layers1 和 layers2 是否会在分区图中产生循环。
    通过临时构建合并后的分区图并检查是否为 DAG 来判断。
    """
    import networkx as nx
    
    merged_layers = set(layers1) | set(layers2)
    
    # 获取 layers1 和 layers2 所属的分区 ID（如果已分配）
    part1_ids = set()
    part2_ids = set()
    for layer in layers1:
        if layer in node_to_partition:
            part1_ids.add(node_to_partition[layer].id)
    for layer in layers2:
        if layer in node_to_partition:
            part2_ids.add(node_to_partition[layer].id)
    
    # 合并后的分区使用一个临时 ID（-999）
    merged_part_id = -999
    merging_part_ids = part1_ids | part2_ids
    
    # 构建临时分区图
    temp_graph = nx.DiGraph()
    
    # 添加所有当前分区的节点（除了被合并的）
    existing_partition_ids = set()
    for layer, part in node_to_partition.items():
        if part.id not in merging_part_ids:
            existing_partition_ids.add(part.id)
    
    for pid in existing_partition_ids:
        temp_graph.add_node(pid)
    temp_graph.add_node(merged_part_id)
    
    # 添加边
    for u in G.nodes():
        for v in G.successors(u):
            # 确定 u 和 v 所属的分区
            if u in merged_layers:
                u_part = merged_part_id
            elif u in node_to_partition:
                u_part = node_to_partition[u].id
            else:
                continue
            
            if v in merged_layers:
                v_part = merged_part_id
            elif v in node_to_partition:
                v_part = node_to_partition[v].id
            else:
                continue
            
            # 如果分区不同，添加边
            if u_part != v_part:
                temp_graph.add_edge(u_part, v_part)
    
    # 检查是否为 DAG
    return not nx.is_directed_acyclic_graph(temp_graph)


# -----------------------------
# 分区阶段：边选择（Algorithm 1）
# -----------------------------
def select_edges_for_partitioning(G):
    """
       论文Algorithm 1：选择满足约束的边集合M，用于后续分区合并
       输入：DNN层依赖图G=(V,E)（有向无环图）
       输出：可合并的边集合M⊆E
       核心逻辑：筛选出“入度=1且出度=1”的节点间的边，且不违反层级约束
       """
    M = set() # 初始化空的边集合M（最终返回的可合并边）
    # 步骤1：计算G的拓扑层级（对应论文中layer(v)，即节点v在拓扑排序中的层级）
    # nx.topological_generations(G)：生成G的拓扑层级迭代器，每层为一组无依赖的节点
    topological_gen = nx.topological_generations(G)#将 DAG 中的节点按 “依赖层级” 分组为拓扑代，每个迭代元素是一组 “无相互依赖、且所有前驱都已处理” 的节点集合。将图变成：{'A'}, {'B','C'}
    # 构建层级字典：key=层级编号，value=该层级的节点列表

    # ====================== DEBUG 代码 ======================
    topological_gen_list = list(topological_gen)  # 迭代器转列表，消耗迭代器
    print("=== Debug: topological_gen 拓扑层级（列表形式）===")
    print(f"类型: {type(topological_gen_list)}")
    print(f"内容: {topological_gen_list}")
    # 重新生成迭代器（因为上面转列表已消耗原迭代器，不然后续代码会无数据）
    topological_gen = nx.topological_generations(G)

    levels = {level: nodes for level, nodes in enumerate(topological_gen)}#案例：输出levels = {0: {'A'}, 1: {'B', 'C'}, 2: {'D'}}
    level_map = {}
    # 为每个节点记录其拓扑层级
    for level, nodes in levels.items():
        for node in nodes:
            level_map[node] = level
        #案例，输出level_map={'A': 0, 'B': 1, 'C': 1, 'D': 2}

        # ====================== DEBUG 代码（核心）======================
    print("=== Debug: level_map 节点-拓扑层级映射 ===")
    # 1. 打印基本信息
    print(f"1. level_map 类型: {type(level_map)}")
    print(f"2. level_map 长度（节点数量）: {len(level_map)}")
    # 2. 按节点ID排序打印键值对（可读性最优）
    print("3. level_map 键值对（按节点ID排序）:")
    for node_id in sorted(level_map.keys()):
        print(f"   节点 {node_id} → 拓扑层级 {level_map[node_id]}")
    # 3. 可选：验证所有节点都被映射（防止遗漏）
    all_nodes = list(G.nodes())
    missing_nodes = [n for n in all_nodes if n not in level_map]
    if missing_nodes:
        print(f"4. 警告：以下节点未被映射到层级 → {missing_nodes}")
    else:
        print(f"4. 验证：所有节点（{all_nodes}）均已正确映射层级")
    # ==============================================================


    # 步骤2：遍历G的拓扑排序节点（保证按层序遍历，符合DNN执行顺序）
    for u in nx.topological_sort(G):
        #遍历节点u的所有后继节点v（即边(u,v)∈E）
        for v in G.successors(u):
            # 约束1：仅考虑“u出度=1且v入度=1”的边（论文Algorithm 1第3行）
            if G.in_degree(v) != 1 and G.out_degree(u) != 1:
                print("G.in_degree(v) != 1 or G.out_degree(u) != 1")
                continue
            # 先加入候选边，再检查约束2，必要时移除
            M.add((u, v))
            violates_constraint_2 = False
            # 遍历u的所有后继节点w（防止同层级重复合并）
            for w in G.successors(u):
                for wp in G.predecessors(w):
                    # 排除当前边(u,v)自身，只检查其他已在M中的边
                    if (wp, w) != (u, v) and (wp, w) in M and level_map[u] == level_map[w] - 1:
                        violates_constraint_2 = True
                        break
                if violates_constraint_2:
                    break
            if violates_constraint_2:
                M.remove((u, v))
    return M# 返回可合并的边集合M

# -----------------------------
# 分区阶段：图合并（Algorithm 2）
# -----------------------------
def merge_check(part1, part2, G, servers, bandwidth_map):
    """
    论文Algorithm 2中的Check函数（分区合并判断逻辑）
    输入：两个待合并分区part1/part2、原始图G、服务器集合、带宽映射
    输出：布尔值（True=可合并，False=不可合并）
    核心规则：
    1. 合并后内存≤EPC有效内存 → 直接合并
    2. 合并后内存>EPC有效内存 → 仅当合并后执行时间≤分离执行+通信时间时合并
    """
    # 计算合并后的总内存/总计算量
    memory = part1.total_memory + part2.total_memory
    workload = part1.total_workload + part2.total_workload

    # 分区间通信数据量（双向累加）
    data_size = inter_partition_data(G, part1.layers, part2.layers) + inter_partition_data(G, part2.layers, part1.layers)

    # 论文公式(9)(10)
    t_merged = partition_exec_time(workload, memory, servers)
    t_part1 = partition_exec_time(part1.total_workload, part1.total_memory, servers)
    t_part2 = partition_exec_time(part2.total_workload, part2.total_memory, servers)
    t_comm = partition_comm_time(data_size, servers, bandwidth_map)
    t_sep = t_part1 + t_part2 + t_comm

    cond1 = memory <= EPC_EFFECTIVE_MB
    cond2 = t_merged <= t_sep
    return cond1 or cond2


def graph_partition(G, layers, edges_M, servers, bandwidth_map):
    """
    论文Algorithm 2：基于边集合M的图合并，生成DNN分区集合P
    输入：
        G: DNN层依赖图
        layers: DNN层字典（key=层ID，value=DNNLayer对象）
        edges_M: Algorithm 1输出的可合并边集合
        servers: 服务器集合
        bandwidth_map: 服务器间带宽映射
    输出：
        partitions: 分区集合P
        node_to_partition: 层-分区映射（key=层ID，value=Partition对象）
    """
    partitions = []# 初始化分区集合P
    node_to_partition = {} # 初始化层-分区映射（记录每层所属分区）
    next_partition_id = 0  # 分区ID计数器，确保每个分区有唯一ID

    # 打印初始状态（调试基准）
    print("=== 初始状态 ===")
    print(f"edges_M 边列表: {list(edges_M)}")
    print(f"初始分区集合: {partitions}")
    print(f"初始层-分区映射: {node_to_partition}\n")

    # 步骤1：遍历可合并边集合M，合并对应层为分区（论文Algorithm 2第1-10行）
    for (u, v) in edges_M:
        # 获取层u、v所属的分区（初始为None）
        pu = node_to_partition.get(u)
        pv = node_to_partition.get(v)

        # ====================== 增强版DEBUG pu/pv ======================
        print(f"===================== 处理第 ({u},{v}) =====================")
        # 打印u/v的基础信息
        print(f"层{u} 信息 → 内存={layers[u].memory}MB, 计算量={layers[u].workload}M FLOPs")
        print(f"层{v} 信息 → 内存={layers[v].memory}MB, 计算量={layers[v].workload}M FLOPs")
        # 打印pu的详细状态
        if pu is None:
            print(f"pu（层{u}所属分区）: None")
        else:
            print(f"pu（层{u}所属分区）:")
            print(f"  - 分区ID: {pu.id}")
            print(f"  - 包含层: {pu.layers}")
            print(f"  - 总内存: {pu.total_memory}MB")
            print(f"  - 总计算量: {pu.total_workload}M FLOPs")
        # 打印pv的详细状态
        if pv is None:
            print(f"pv（层{v}所属分区）: None")
        else:
            print(f"pv（层{v}所属分区）:")
            print(f"  - 分区ID: {pv.id}")
            print(f"  - 包含层: {pv.layers}")
            print(f"  - 总内存: {pv.total_memory}MB")
            print(f"  - 总计算量: {pv.total_workload}M FLOPs")
        # ==============================================================

        # 情况1：u和v均未分配分区 → 创建新分区
        if pu is None and pv is None:
            # 先检查合并是否会产生循环
            if would_create_cycle(G, [u], [v], node_to_partition):
                print(f"  [skip] 创建新分区失败：会产生分区循环")
            else:
                # 新建分区，先通过Check判断是否合并
                temp_part_u = Partition(-1, [u])
                temp_part_u.total_memory = layers[u].memory
                temp_part_u.total_workload = layers[u].workload
                temp_part_v = Partition(-1, [v])
                temp_part_v.total_memory = layers[v].memory
                temp_part_v.total_workload = layers[v].workload
                if merge_check(temp_part_u, temp_part_v, G, servers, bandwidth_map):
                    # 新建分区，使用唯一ID计数器
                    new_part = Partition(next_partition_id, [u, v])
                    next_partition_id += 1
                    # 计算新分区总内存（Σmem(v), v∈新分区）
                    new_part.total_memory = layers[u].memory + layers[v].memory
                    # 计算新分区总计算量（Σw(v), v∈新分区）
                    new_part.total_workload = layers[u].workload + layers[v].workload
                    partitions.append(new_part)# 将新分区加入集合P
                    # 记录u、v所属的分区
                    node_to_partition[u] = node_to_partition[v] = new_part
                    print(f"  新分区信息 → ID={new_part.id}, 层={new_part.layers}, 内存={new_part.total_memory}MB")
                else:
                    print(f"  [skip] 未通过Check：层{u}与层{v}不合并")
        # 情况2：u和v分属不同分区 → 尝试合并两个分区
        elif pu and pv and pu != pv:
            # 先检查合并是否会产生循环
            if would_create_cycle(G, pu.layers, pv.layers, node_to_partition):
                print(f"  [skip] 合并失败：会产生分区循环")
            # 调用Check函数判断是否可合并
            elif merge_check(pu, pv, G, servers, bandwidth_map):
                # 合并pv到pu：将pv的层加入pu
                pu.layers += pv.layers
                # 更新pu的总内存（累加pv的内存）
                pu.total_memory += pv.total_memory
                # 更新pu的总计算量（累加pv的计算量）
                pu.total_workload += pv.total_workload
                # 更新pv所有层的分区映射为pu
                for node in pv.layers:
                    node_to_partition[node] = pu
                # 从分区集合中移除pv（已合并）
                partitions.remove(pv)
                print(f"  合并后pu={pu.id} → 层={pu.layers}, 内存={pu.total_memory}MB")
            else:
                print(f"  [skip] 合并失败：不满足merge_check条件")
        else:
            # 情况3：仅u或v有分区 → 尝试将另一层加入现有分区
            existing = pu or pv# 已有分区（pu或pv）
            other = v if pu else u# 未分配分区的层（v或u）
            # 确保该层未在现有分区中（防止重复添加）
            if other not in existing.layers:
                # 先检查合并是否会产生循环
                if would_create_cycle(G, existing.layers, [other], node_to_partition):
                    print(f"  [skip] 添加失败：会产生分区循环")
                else:
                    # 构建临时分区（仅包含other层），用于Check判断
                    temp_part = Partition(-1, [other])
                    # 初始化临时分区的内存（对应层other的内存）
                    temp_part.total_memory = layers[other].memory
                    # 初始化临时分区的计算量（对应层other的计算量）
                    temp_part.total_workload = layers[other].workload
                    # 调用Check函数判断是否可合并
                    if merge_check(existing, temp_part, G, servers, bandwidth_map):
                        print(f"  [ok] 添加成功：层{other}加入分区{existing.id}")
                        # 将other层加入现有分区
                        existing.layers.append(other)
                        # 更新现有分区的总内存（累加other层的内存）
                        existing.total_memory += layers[other].memory
                        # 更新现有分区的总计算量（累加other层的计算量）
                        existing.total_workload += layers[other].workload
                        # 记录other层所属的分区
                        node_to_partition[other] = existing
                        print(f"  添加后分区{existing.id} → 层={existing.layers}, 内存={existing.total_memory}MB")
                    else:
                        print(f"  [skip] 添加失败：不满足merge_check条件")
            else:
                print(f"  [skip] 层{other}已在分区{existing.id}中，无需添加")
        print(f"================================================================\n")

    # 步骤2：为未合并的孤立层创建独立分区（论文Algorithm 2第11-13行）
    for node in G.nodes():
        # 若层未分配到任何分区
        if node not in node_to_partition:
            # 新建分区，包含该孤立层，使用唯一ID计数器
            p = Partition(next_partition_id, [node])
            next_partition_id += 1
            # 初始化分区内存（该层的内存）
            p.total_memory = layers[node].memory
            # 初始化分区计算量（该层的计算量）
            p.total_workload = layers[node].workload
            partitions.append(p)# 加入分区集合
            # 记录该层所属的分区
            node_to_partition[node] = p

    return partitions, node_to_partition # 返回分区集合和层-分区映射

# -----------------------------
# 分配阶段（Algorithm 3）
# -----------------------------
def compute_partition_priority(partition, partition_graph, partition_map, servers, bandwidth_map, memo=None):
    """
    论文公式11：分区优先级计算（递归定义）
    优先级Priority(p) = T(p) + C(p, succ(p)) + max(Priority(succ(p)))
    其中：
        T(p)：分区p的执行时间（公式9）
        C(p, succ(p))：p到后继分区的通信时间
        max(Priority(succ(p)))：所有后继分区的最大优先级
    输入：
        partition: 待计算优先级的分区
        partition_graph: 分区依赖图（节点=分区ID，边=分区间依赖）
        partition_map: 分区ID到分区对象的映射字典
        servers: 服务器集合
        bandwidth_map: 服务器间带宽映射
        memo: 缓存字典（避免重复递归计算）
    输出：分区的优先级值
    """
    # 初始化缓存字典（默认参数设为None，避免多次调用时缓存污染）
    if memo is None:
        memo = {}
    # 若该分区优先级已计算过，直接返回缓存值（剪枝）
    if partition.id in memo:
        return memo[partition.id]
    # 获取该分区的所有后继分区ID（分区依赖图中的边）
    successors = list(partition_graph.successors(partition.id))
    # 边界条件：无后继分区（最后一个分区）
    t_p = partition_exec_time(partition.total_workload, partition.total_memory, servers)
    if not successors:
        memo[partition.id] = t_p
        return memo[partition.id]
    # 公式11：取所有后继中的最大值
    candidate_priorities = []
    for s in successors:
        data_size = partition_graph.edges[partition.id, s].get("data_size", COMM_DATA_SIZE)
        comm_time = partition_comm_time(data_size, servers, bandwidth_map)
        succ_partition = partition_map[s]
        succ_priority = compute_partition_priority(succ_partition, partition_graph, partition_map, servers, bandwidth_map, memo)
        candidate_priorities.append(t_p + comm_time + succ_priority)
    memo[partition.id] = max(candidate_priorities)
    return memo[partition.id]

def assign_partitions_to_servers(partitions, partition_graph, servers, bandwidth_map):
    """
       论文Algorithm 3：按优先级将分区分配到服务器，计算总推理时间FT(P)
       输入：
           partitions: 分区集合P
           partition_graph: 分区依赖图
           servers: 服务器集合S
           bandwidth_map: 服务器间带宽映射（key=(s1.id, s2.id)，value=带宽值）
       输出：
           partitions: 分配后的分区（更新了服务器ID、开始/结束时间）
           total_infer_time: 总推理时间（所有分区的最晚完成时间）
       """
    # 计算服务器间平均带宽（用于优先级计算）
    # bandwidth_avg = sum(bandwidth_map.values()) / len(bandwidth_map)

    # 创建分区ID到分区对象的映射
    partition_map = {p.id: p for p in partitions}

    # 计算分区优先级
    priorities = {}  # 优先级字典：key=分区ID，value=优先级值
    for p in partitions:
        priorities[p.id] = compute_partition_priority(p, partition_graph, partition_map, servers, bandwidth_map)

    # 步骤2：按优先级降序排序分区（优先级越高，越先分配，论文Algorithm 3第3行）
    sorted_partitions = sorted(partitions, key=lambda p: -priorities[p.id])
    assigned = {} # 分区-服务器映射：key=分区ID，value=分配的Server对象
    finish_times = {}# 分区完成时间：key=分区ID，value=完成时间

    # 步骤3：遍历排序后的分区，分配到最优服务器（论文Algorithm 3第4-15行）
    for p in sorted_partitions:
        best_time = float('inf') # 初始化最优完成时间（无穷大）
        best_server = None# 初始化最优服务器
        server_ft = {}# 服务器-时间映射：key=服务器ID，value=(完成时间, 开始时间)

        # 遍历所有服务器，计算该分区在每个服务器上的完成时间
        for s in servers:
            ready_time = 0 # 该分区的就绪时间（所有前驱分区完成+通信后的最早时间）
            # 遍历该分区的所有前驱分区（保证执行顺序）
            for pred in partition_graph.predecessors(p.id):
                # 获取前驱分区分配的服务器
                pred_p = assigned[pred]
                print(f"\n[DEBUG] 前驱任务/分区信息：")
                print(f"  pred标识: {pred} | pred_p.id: {pred_p.id} | 当前服务器s.id: {s.id}")
                print(f"  pred_p完整对象信息: {pred_p}")  # 若为自定义对象，建议实现__str__方法
                # 计算前驱分区到当前服务器的通信时间
                # data_size 是 bytes，bandwidth_map 是 Mbps，需要转换
                data_size = partition_graph.edges[pred, p.id].get("data_size", COMM_DATA_SIZE)
                bandwidth_bytes_per_sec = bandwidth_map[(pred_p.id, s.id)] * 1e6 / 8
                comm = data_size / bandwidth_bytes_per_sec if pred_p.id != s.id else 0
                if pred_p.id != s.id:
                    print(f"  pred_p与当前服务器不同 | 通信时间comm: {comm}")
                else:
                    print(f"  pred_p与当前服务器相同 | 通信时间comm: {comm}")
                # 更新就绪时间（取所有前驱的最大完成+通信时间)
                ready_time = max(ready_time, finish_times[pred] + comm)

            # 公式9：计算该分区在服务器s上的执行时间（考虑EPC约束）
            mem = p.total_memory
            print(f"  目前该分区的total_memory: {mem}")
            exec_time = p.total_workload / compute_fn(s, mem)
            # 分区开始时间：取“就绪时间”和“服务器空闲时间”的最大值
            # 服务器空闲时间=调度队列中最晚的结束时间（default=0表示队列为空）
            start_time = max(ready_time, max((et for st, et, _ in s.schedule), default=0))
            # 分区完成时间=开始时间+执行时间
            finish_time = start_time + exec_time
            print(f"  分区start_time: {start_time}")
            print(f"  分区执行exec_time: {exec_time}")
            print(f"  分区执行finish_time: {finish_time}")
            # 记录该服务器上的完成时间和开始时间
            server_ft[s.id] = (finish_time, start_time)
            # 更新最优服务器（选择完成时间最小的服务器）
            if finish_time < best_time:
                best_time = finish_time
                best_server = s
                print(f"  最好服务器best_server: {best_server.id}")
                print(f"  最好服务器的结束时间finish_time: {best_time}")

        # 步骤4：将分区分配到最优服务器，更新状态
        ft, st = server_ft[best_server.id]# 获取最优服务器的完成/开始时间
        print(f"  获取最优服务器的完成时间: {ft},结束时间：{st}")
        p.assigned_server = best_server.id# 记录分区分配的服务器ID
        p.start_time = st# 记录分区开始时间
        p.finish_time = ft# 记录分区结束时间
        p.ready_time = st# 记录分区就绪时间（与开始时间一致）
        # 将该分区加入服务器的调度队列
        best_server.schedule.append((st, ft, p))
        # 记录分区分配的服务器
        assigned[p.id] = best_server
        # 记录分区的完成时间
        finish_times[p.id] = ft
        best_server.asseignedmemory+=p.total_memory

    # 返回分配后的分区集合，以及总推理时间（所有分区的最晚完成时间）
    return partitions, max(finish_times.values())

# -----------------------------
# 示例测试：简化版NiN模型（6层）
# -----------------------------
#构建一个6 节点的线性有向无环图（DNN 层依赖图）
# def build_nin_model():
#     """
#     构建一个简化的NiN模型（6层）
#     """
#     G = nx.DiGraph()# 输出：G的节点： [0, 1, 2, 3, 4, 5]
#     layers = {}
#     layers[0]=DNNLayer(0, 30,1000,)
#     layers[1] = DNNLayer(1, 50, 4000,)
#     layers[2] = DNNLayer(2, 20, 4000,)
#     layers[3] = DNNLayer(3, 50, 3000,)
#     edges = [(0,1),(0,2),(1,3),(2,3)]#DNN有4条有向边，形成线性依赖链 0→1→2→3→4→5；
#     for u,v in edges:
#         G.add_edge(u, v, data_size=1.0)  # 1MB传输，每条边1MB传输
#     return G, layers

def example_run(G, layers, servers):
    """
    论文算法完整流程测试：
    1. 构建NiN模型
    2. 初始化服务器集合
    3. 执行Algorithm 1（边选择）
    4. 执行Algorithm 2（图合并）
    5. 构建分区依赖图
    6. 执行Algorithm 3（分区分配）
    7. 输出结果
    
    参数:
        G: DNN层依赖图
        layers: DNN层字典
        servers: 服务器列表
    """
    # 构建服务器间带宽映射（所有服务器对的带宽均为平均带宽）
    bandwidth_map = {(s1.id, s2.id):BANDWIDTH_AVG for s1 in servers for s2 in servers}#说明所有服务器之间的传输带宽是多少

    # 步骤3：执行Algorithm 1：选择可合并边集合M
    edges_M = select_edges_for_partitioning(G)
    # ====================== DEBUG 代码（核心）======================
    print("=== Debug: edges_M 可合并边集合 ===")
    print(f"1. edges_M 类型: {type(edges_M)}")  # 打印类型（应为set）
    print(f"2. edges_M 长度（可合并边数量）: {len(edges_M)}")  # 打印边数量
    print(f"3. edges_M 具体内容（转换为列表）: {list(edges_M)}")  # 打印具体边
    # 可选：打印原图形的所有边，方便对比哪些边被选中
    print(f"4. 原图形G的所有边: {list(G.edges())}")
    # ==============================================================

    # 步骤4：执行Algorithm 2：基于M合并为分区
    partitions, node_map = graph_partition(G, layers, edges_M, servers, bandwidth_map)

    # 步骤5：构建分区依赖图（基于原层依赖图）
    partition_graph = nx.DiGraph()
    # 向分区依赖图中添加所有分区节点
    for p in partitions:
        partition_graph.add_node(p.id)
    # 遍历原层依赖图的边，构建分区间的依赖边
    for u, v in G.edges():
        pu = node_map[u].id # 层u所属的分区ID
        pv = node_map[v].id # 层v所属的分区ID
        if pu != pv:# 若两层分属不同分区，添加分区依赖边
            data_size = get_edge_data_size(G, u, v)
            if partition_graph.has_edge(pu, pv):
                partition_graph.edges[pu, pv]["data_size"] += data_size
            else:
                partition_graph.add_edge(pu, pv, data_size=data_size)

    # 如果分区图不是 DAG，通过移除反向边来修复
    while not nx.is_directed_acyclic_graph(partition_graph):
        try:
            cycle = nx.find_cycle(partition_graph)
            # 移除循环中权重最小的边（data_size 最小）
            min_edge = min(cycle, key=lambda e: partition_graph.edges[e[0], e[1]].get("data_size", 0))
            partition_graph.remove_edge(min_edge[0], min_edge[1])
            print(f"[warn] 移除循环边: {min_edge}")
        except nx.NetworkXNoCycle:
            break

    # 步骤6：执行Algorithm 3：将分区分配到服务器，计算总推理时间
    partitions, total_infer_time = assign_partitions_to_servers(partitions, partition_graph, servers, bandwidth_map)

    # 步骤7：输出结果（分区信息+总推理时间）
    print("===== 分区结果 =====")
    for p in partitions:
        # 输出分区ID、包含的层、总内存（标注是否超EPC）、分配服务器、开始/完成时间
        print(f"分区#{p.id}: 层={p.layers}, 总内存={p.total_memory:.1f}MB {'(over EPC)' if p.total_memory > EPC_EFFECTIVE_MB else ''}, 分配服务器={p.assigned_server}, 开始={p.start_time:.2f}s, 完成={p.finish_time:.2f}s")

    # 输出总推理时间（论文中核心优化目标）
    print(f"\n总推理时间: {total_infer_time:.2f} 秒")

if __name__ == "__main__":
    G = build_dnn_graph()
    layers = get_dnn_layers()
    servers = get_servers()
    example_run(G, layers, servers)  # 传入参数


#输入，1. 模型layer的workload和MEM。workload单位是完整的计算时间。
#setup: 1.服务器的算力：
    #Server(0, 1),  #M FLOPs/s line515
    #Server(1, 1)   #line 516