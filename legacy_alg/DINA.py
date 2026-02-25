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
LAYLER_COM_BAND = 1
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
            # 判断是否违反约束2
            print("************************************")
            # valid = True # 标记该边是否有效（可加入M）
            M.add((u, v))
            #
            # # 遍历u的所有后继节点w（防止同层级重复合并）
            #
            # # ====================== DEBUG u/w 核心代码 ======================
            # print(f"\n🔍 正在检查边 ({u}, {v}) 的约束2 → 外层节点u = {u}")
            # # 遍历u的所有后继节点w（防止同层级重复合并）
            # for w in G.successors(u):
            #     print(f"  ├─ 遍历u={u}的后继节点 → w = {w}")
            #     # 遍历w的所有前驱节点wp
            #     for wp in G.predecessors(w):
            #         print(f"  │  └─ 遍历w={w}的前驱节点 → wp = {wp}")
            #         # 约束2判断逻辑
            #         if (wp, w) in M and level_map[u] == level_map[w] - 1:
            #             valid = False
            #             print(
            #                 f"  │     ⚠️ 触发约束2：(wp,w)=({wp},{w})∈M 且 level(u)={level_map[u]} = level(w)={level_map[w]}-1")
            # # ==============================================================
            # if valid:
            #     # 若满足所有约束，将边(u,v)加入M
            #     M.add((u, v))
    return M# 返回可合并的边集合M

# -----------------------------
# 分区阶段：图合并（Algorithm 2）
# -----------------------------
def merge_check(part1, part2, Fn_avg, bandwidth_avg):
    """
    论文Algorithm 2中的Check函数（分区合并判断逻辑）
    输入：两个待合并分区part1/part2、平均算力Fn_avg、平均带宽bandwidth_avg
    输出：布尔值（True=可合并，False=不可合并）
    核心规则：
    1. 合并后内存≤EPC有效内存 → 直接合并
    2. 合并后内存>EPC有效内存 → 仅当合并后执行时间≤分离执行+通信时间时合并
    """
    # 计算合并后的总内存（对应论文中mem(p1∪p2)=mem(p1)+mem(p2)）
    memory = part1.total_memory + part2.total_memory
    if memory > EPC_EFFECTIVE_MB:
        return False
    # 计算合并后的总计算量（对应论文中w(p1∪p2)=w(p1)+w(p2)）
    workload = part1.total_workload + part2.total_workload

    # ====================== DEBUG：计算合并后内存/计算量 ======================
    print("🔧 合并后核心指标计算")
    print(f"1. 合并后总内存: {memory}MB = part1({part1.total_memory}MB) + part2({part2.total_memory}MB)")
    print(f"   → EPC约束判断: {'✅ ≤ EPC' if memory <= EPC_EFFECTIVE_MB else '❌ > EPC'} (EPC={EPC_EFFECTIVE_MB}MB)")
    print(
        f"2. 合并后总计算量: {workload}M FLOPs = part1({part1.total_workload}M FLOPs) + part2({part2.total_workload}M FLOPs)\n")


    def exec_time(mem, work):  # 论文公式9：分区执行时间计算
        """
        公式9：T(p) = w(p)/Fn(s) （内存≤EPC）；T(p)=w(p)/(0.5*Fn(s))（内存>EPC）
        此处简化为平均算力Fn_avg（后续分配阶段会替换为具体服务器算力）
        """
        print(f"   📌 计算exec_time - 内存={mem}MB, 计算量={work}M FLOPs:")
        if mem <= EPC_EFFECTIVE_MB:
            print(f"      → 内存≤EPC → 执行时间= {work} / {Fn_avg}")
            return work / Fn_avg
        else:
            print(f"      → 内存>EPC → 执行时间 = {work} / ({Fn_avg} * SWITCH_OVERHEAD)")
            return work / (Fn_avg * SWITCH_OVERHEAD)  # 超出EPC，性能下降一半

        # ====================== 第四步：计算合并执行时间t_merged ======================

    print("🔧 执行时间计算（论文公式9）")
    print(f"1. 合并后执行时间t_merged:")
    t_merged = exec_time(memory, workload)
    print(f"   → t_merged = {t_merged:.6f}s\n")
    # 计算分离执行时间+通信时间（论文公式10：T_sep = T(p1)+T(p2)+C(p1,p2)）
    # C(p1,p2)=数据量/带宽，此处简化数据量为1MB，故通信时间=1/bandwidth_avg
    # t_sep = exec_time(part1.total_memory, part1.total_workload) + \
    #         exec_time(part2.total_memory, part2.total_workload) + \
    #         1.0 / bandwidth_avg  # 公式10简化（数据量设为1MB）

    print("🔧 分离执行+通信时间计算（论文公式10：T_sep = T(p1)+T(p2)+C(p1,p2)）")
    # 计算part1执行时间
    print(f"1. part1执行时间T(p1):")
    t_part1 = exec_time(part1.total_memory, part1.total_workload)
    # 计算part2执行时间
    print(f"2. part2执行时间T(p2):")
    t_part2 = exec_time(part2.total_memory, part2.total_workload)
    # 计算通信时间C(p1,p2)（简化为1MB/带宽）
    t_comm = LAYLER_COM_BAND / bandwidth_avg
    print(f"3. 分区间通信时间C(p1,p2): {t_comm:.6f}s = 1 / {bandwidth_avg} (带宽)")
    # 总分离时间
    t_sep = t_part1 + t_part2 + t_comm
    print(f"4. 分离总时间t_sep = {t_part1:.6f} + {t_part2:.6f} + {t_comm:.6f} = {t_sep:.6f}s\n")


    # # 合并判断：满足内存约束 或 合并后时间更短 → 可合并
    # return memory <= EPC_EFFECTIVE_MB or t_merged <= t_sep
    # ====================== 第六步：最终合并判断 ======================
    print("🔧 最终合并判断逻辑")
    # 条件1：内存≤EPC
    cond1 = memory <= EPC_EFFECTIVE_MB
    # 条件2：合并时间≤分离时间（仅条件1不满足时生效）
    cond2 = t_merged <= t_sep
    # 最终结果：条件1 或 条件2
    result = cond1 or cond2

    print(f"1. 条件1（内存≤EPC）: {cond1} → {'直接合并' if cond1 else '需判断时间条件'}")
    if not cond1:  # 仅条件1不满足时打印条件2
        print(f"2. 条件2（t_merged ≤ t_sep）: {cond2} → {t_merged:.6f} ≤ {t_sep:.6f}")
    print(f"3. 最终合并结果: {'✅ 可合并' if result else '❌ 不可合并'}")
    print("=" * 80)
    return result


def graph_partition(G, layers, edges_M, Fn_avg, bandwidth_avg):
    """
    论文Algorithm 2：基于边集合M的图合并，生成DNN分区集合P
    输入：
        G: DNN层依赖图
        layers: DNN层字典（key=层ID，value=DNNLayer对象）
        edges_M: Algorithm 1输出的可合并边集合
        Fn_avg: 服务器平均算力
        bandwidth_avg: 服务器间平均带宽
    输出：
        partitions: 分区集合P
        node_to_partition: 层-分区映射（key=层ID，value=Partition对象）
    """
    partitions = []# 初始化分区集合P
    node_to_partition = {} # 初始化层-分区映射（记录每层所属分区）

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
            if layers[u].memory + layers[v].memory > EPC_EFFECTIVE_MB:
                new_part = Partition(len(partitions), [u])
                new_part.total_memory = layers[u].memory
                # 计算新分区总计算量（Σw(v), v∈新分区）
                new_part.total_workload = layers[u].workload
                partitions.append(new_part)  # 将新分区加入集合P
                # 记录u、v所属的分区
                node_to_partition[u] = new_part
                new_part = Partition(len(partitions), [v])
                new_part.total_memory = layers[v].memory
                # 计算新分区总计算量（Σw(v), v∈新分区）
                new_part.total_workload = layers[v].workload
                partitions.append(new_part)  # 将新分区加入集合P
                # 记录u、v所属的分区
                node_to_partition[v] = new_part
            else:
                # 新建分区，ID为当前分区数量（保证唯一），如果当前分区数量为0，则ID=0，该分区包含层u和v
                new_part = Partition(len(partitions), [u, v])
                # 计算新分区总内存（Σmem(v), v∈新分区）
                new_part.total_memory = layers[u].memory + layers[v].memory
                # 计算新分区总计算量（Σw(v), v∈新分区）
                new_part.total_workload = layers[u].workload + layers[v].workload
                partitions.append(new_part)# 将新分区加入集合P
                # 记录u、v所属的分区
                node_to_partition[u] = node_to_partition[v] = new_part
            print(f"  新分区信息 → ID={new_part.id}, 层={new_part.layers}, 内存={new_part.total_memory}MB")
        # 情况2：u和v分属不同分区 → 尝试合并两个分区
        elif pu and pv and pu != pv:
            # 调用Check函数判断是否可合并
            if merge_check(pu, pv, Fn_avg, bandwidth_avg):
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
                print(f"  ❌ 合并失败：不满足merge_check条件")
        else:
            # 情况3：仅u或v有分区 → 尝试将另一层加入现有分区
            existing = pu or pv# 已有分区（pu或pv）
            other = v if pu else u# 未分配分区的层（v或u）
            # 确保该层未在现有分区中（防止重复添加）
            if other not in existing.layers:
                # 构建临时分区（仅包含other层），用于Check判断
                temp_part = Partition(-1, [other])
                # 初始化临时分区的内存（对应层other的内存）
                temp_part.total_memory = layers[other].memory
                # 初始化临时分区的计算量（对应层other的计算量）
                temp_part.total_workload = layers[other].workload
                # 调用Check函数判断是否可合并
                if merge_check(existing, temp_part, Fn_avg, bandwidth_avg):
                    print(f"  ✅ 添加成功：层{other}加入分区{existing.id}")
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
                    print(f"  ❌ 添加失败：不满足merge_check条件")
            else:
                print(f"  ❌ 层{other}已在分区{existing.id}中，无需添加")
        print(f"================================================================\n")

    # 步骤2：为未合并的孤立层创建独立分区（论文Algorithm 2第11-13行）
    for node in G.nodes():
        # 若层未分配到任何分区
        if node not in node_to_partition:
            # 新建分区，包含该孤立层
            p = Partition(len(partitions), [node])
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
def compute_partition_priority(partition, partition_graph, partitions, Fn_avg, bandwidth_avg, memo=None):
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
        partitions: 所有分区的列表（通过ID索引）
        Fn_avg: 平均算力
        bandwidth_avg: 平均带宽
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
    if not successors:
        # 优先级=该分区的执行时间（公式9）
        memo[partition.id] = partition.total_workload / Fn_avg
        return memo[partition.id]
    # 递归计算所有后继分区的优先级，取最大值
    max_succ = max(
        compute_partition_priority(partitions[s], partition_graph, partitions, Fn_avg, bandwidth_avg, memo)
        for s in successors
    )
    # 计算分区到后继分区的通信时间（简化为1/平均带宽）
    comm_time = COMM_DATA_SIZE / bandwidth_avg
    # 公式11：计算当前分区的优先级
    priority = partition.total_workload / Fn_avg + comm_time + max_succ
    # 缓存优先级结果
    memo[partition.id] = priority
    return priority

def assign_partitions_to_servers(partitions, partition_graph, servers, Fn_avg, bandwidth_map):
    """
       论文Algorithm 3：按优先级将分区分配到服务器，计算总推理时间FT(P)
       输入：
           partitions: 分区集合P
           partition_graph: 分区依赖图
           servers: 服务器集合S
           Fn_avg: 平均算力（用于优先级计算）
           bandwidth_map: 服务器间带宽映射（key=(s1.id, s2.id)，value=带宽值）
       输出：
           partitions: 分配后的分区（更新了服务器ID、开始/结束时间）
           total_infer_time: 总推理时间（所有分区的最晚完成时间）
       """
    # 计算服务器间平均带宽（用于优先级计算）
    # bandwidth_avg = sum(bandwidth_map.values()) / len(bandwidth_map)

    # 计算分区优先级
    priorities = {}  # 优先级字典：key=分区ID，value=优先级值
    for p in partitions:
        priorities[p.id] = compute_partition_priority(p, partition_graph, partitions, Fn_avg, BANDWIDTH_AVG)

    # 步骤2：按优先级降序排序分区（优先级越高，越先分配，论文Algorithm 3第3行）
    sorted_partitions = sorted(partitions, key=lambda p: -priorities[p.id])
    assigned = {} # 分区-服务器映射：key=分区ID，value=分配的Server对象
    finish_times = {}# 分区完成时间：key=分区ID，value=完成时间

    # 步骤3：遍历排序后的分区，分配到最优服务器（论文Algorithm 3第4-15行）
    for p in sorted_partitions:
        print(f"____________________p.id______________________:{p.id}")
        best_time = float('inf') # 初始化最优完成时间（无穷大）
        best_server = None# 初始化最优服务器
        server_ft = {}# 服务器-时间映射：key=服务器ID，value=(完成时间, 开始时间)

        # 遍历所有服务器，计算该分区在每个服务器上的完成时间
        for s in servers:
            print("*******************")
            print(f"当前正在决策的服务器：{s.id}")
            print(f"当前分片的内存：{p.total_memory}")
            print(f"服务器已分配的内存：{s.asseignedmemory}")
            print(f"总内存和是否超过100：{p.total_memory + s.asseignedmemory}")
            if p.total_memory + s.asseignedmemory > EPC_EFFECTIVE_MB:
                continue
            ready_time = 0 # 该分区的就绪时间（所有前驱分区完成+通信后的最早时间）
            # 遍历该分区的所有前驱分区（保证执行顺序）
            for pred in partition_graph.predecessors(p.id):
                # 获取前驱分区分配的服务器
                pred_p = assigned[pred]
                print(f"\n[DEBUG] 前驱任务/分区信息：")
                print(f"  pred标识: {pred} | pred_p.id: {pred_p.id} | 当前服务器s.id: {s.id}")
                print(f"  pred_p完整对象信息: {pred_p}")  # 若为自定义对象，建议实现__str__方法
                # 计算前驱分区到当前服务器的通信时间：
                # 若前驱分区与当前服务器不同，通信时间=1/带宽；否则为0（同服务器无通信）
                comm = COMM_DATA_SIZE / bandwidth_map[(pred_p.id, s.id)] if pred_p.id != s.id else 0
                if pred_p.id != s.id:
                    print(f"  pred_p与当前服务器不同 | 通信时间comm: {comm}")
                else:
                    print(f"  pred_p与当前服务器相同 | 通信时间comm: {comm}")
                # 更新就绪时间（取所有前驱的最大完成+通信时间)
                ready_time = max(ready_time, finish_times[pred] + comm)

            # 公式9：计算该分区在服务器s上的执行时间（考虑EPC约束）
            mem = p.total_memory + s.asseignedmemory

            print(f"  目前该分区的total_memory: {mem}")
            # 内存≤EPC：执行时间=总计算量/服务器算力；否则=总计算量/(0.5*服务器算力)
            exec_time = p.total_workload / (s.power if mem <= EPC_EFFECTIVE_MB else s.power * SWITCH_OVERHEAD)
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

def example_run():
    """
    论文算法完整流程测试：
    1. 构建NiN模型
    2. 初始化服务器集合
    3. 执行Algorithm 1（边选择）
    4. 执行Algorithm 2（图合并）
    5. 构建分区依赖图
    6. 执行Algorithm 3（分区分配）
    7. 输出结果
    """
    # 步骤1：构建NiN模型的层依赖图和层字典
    # G, layers = build_nin_model()

    # # 步骤2：初始化服务器集合（2台边缘服务器，模拟异构算力）
    # servers = [
    #     Server(0, 10000),  # M FLOPs/s
    #     Server(1, 10000)
    # ]
    # bandwidth_avg = 10  # 设置服务器之间的带宽

    # 计算服务器平均算力（用于Algorithm 1/2）
    Fn_avg = sum(s.power for s in servers) / len(servers)#计算所有服务器的平均算力
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
    partitions, node_map = graph_partition(G, layers, edges_M, Fn_avg, BANDWIDTH_AVG)

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
            partition_graph.add_edge(pu, pv)

    # 步骤6：执行Algorithm 3：将分区分配到服务器，计算总推理时间
    partitions, total_infer_time = assign_partitions_to_servers(partitions, partition_graph, servers, Fn_avg, bandwidth_map)

    # 步骤7：输出结果（分区信息+总推理时间）
    print("===== 分区结果 =====")
    for p in partitions:
        # 输出分区ID、包含的层、总内存（标注是否超EPC）、分配服务器、开始/完成时间
        print(f"分区#{p.id}: 层={p.layers}, 总内存={p.total_memory:.1f}MB {'⚠️超EPC' if p.total_memory > EPC_EFFECTIVE_MB else ''}, 分配服务器={p.assigned_server}, 开始={p.start_time:.2f}s, 完成={p.finish_time:.2f}s")

    # 输出总推理时间（论文中核心优化目标）
    print(f"\n总推理时间: {total_infer_time:.2f} 秒")

if __name__ == "__main__":
    G = build_dnn_graph()
    layers = get_dnn_layers()
    servers = get_servers()
    example_run()


#输入，1. 模型layer的workload和MEM。workload单位是完整的计算时间。
#setup: 1.服务器的算力：
    #Server(0, 1),  #M FLOPs/s line515
    #Server(1, 1)   #line 516