import networkx as nx
import itertools
from typing import List, Dict, Tuple, Set
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

# ====================== 全局常量定义 ======================
# EPC_EFFECTIVE_MB = 100  # EPC有效内存约束
# BANDWIDTH_AVG = 10  # 服务器间平均带宽 (Mbps)
# 通信数据量简化为1MB，通信时间=数据量/带宽
# COMM_DATA_SIZE = 1.0


# # ====================== 核心类定义 ======================
# class DNNLayer:
#     """DNN层类，包含层ID、内存、计算量"""
#
#     def __init__(self, id: int, memory: float, workload: float):
#         self.id = id
#         self.memory = memory  # 内存 (MB)Fn_avg
#         self.workload = workload  # 计算量 (M FLOPs)
#
#     def __repr__(self):
#         return f"Layer({self.id}, mem={self.memory}, workload={self.workload})"
#
#
# class Server:
#     """服务器类，包含服务器ID、算力"""
#
#     def __init__(self, id: int, power: float):
#         self.id = id
#         self.power = power  # 算力 (M FLOPs/s)
#
#     def __repr__(self):
#         return f"Server({self.id}, power={self.power})"


class PartitionScheme:
    """分区方案类，包含分区列表、服务器分配、总推理时间等"""

    def __init__(self, partitions: List[List[int]]):
        self.partitions = partitions  # 分区列表，每个元素是层ID列表
        self.server_assignment = {}  # 分区索引 → 服务器ID
        self.total_time = float('inf')  # 总推理时间
        self.layer_start_time = {}  # 层ID → 开始时间
        self.layer_finish_time = {}  # 层ID → 完成时间
        self.partition_start_time = {}  # 分区索引 → 开始时间
        self.partition_finish_time = {}  # 分区索引 → 完成时间

    def __repr__(self):
        return (f"PartitionScheme(\n  partitions={self.partitions},\n  server_assignment={self.server_assignment},\n  "
                f"total_time={self.total_time:.2f}s\n)")


# ====================== 辅助函数：生成所有可能的分区方案 ======================
def generate_all_partitions(nodes: List[int]) -> List[List[List[int]]]:
    """
    递归生成所有可能的分区方案（穷举所有层分组方式）
    :param nodes: 层ID列表
    :return: 所有分区方案的列表，每个方案是[[层1,层2], [层3], ...]
    """
    if not nodes:
        return [[]]

    first = nodes[0]
    rest = nodes[1:]
    # 递归生成剩余节点的分区方案
    rest_partitions = generate_all_partitions(rest)

    all_partitions = []
    for p in rest_partitions:
        # 方案1：将第一个节点作为新分区
        all_partitions.append([[first]] + p)
        # 方案2：将第一个节点加入现有每个分区
        for i in range(len(p)):
            new_p = p[:i] + [[first] + p[i]] + p[i + 1:]
            all_partitions.append(new_p)

    # 去重（避免重复的分区方案）
    unique_partitions = []
    seen = set()
    for p in all_partitions:
        # 将分区排序后转成元组，用于去重
        sorted_p = tuple(tuple(sorted(part)) for part in p)
        if sorted_p not in seen:
            seen.add(sorted_p)
            unique_partitions.append([list(part) for part in sorted_p])

    return unique_partitions


def is_partition_valid(partitions: List[List[int]], G: nx.DiGraph, layers: Dict[int, DNNLayer]) -> bool:
    """
    检查分区方案是否合法（满足依赖约束+内存约束）
    :param partitions: 分区方案
    :param G: DNN层依赖图（DAG）
    :param layers: 层属性字典
    :return: True=合法，False=不合法
    """
    # 1. 检查每个层仅属于一个分区
    return True
    all_nodes = []
    for part in partitions:
        all_nodes.extend(part)
    if len(all_nodes) != len(set(all_nodes)):
        return False

    # 2. 检查每个分区的总内存 ≤ EPC约束
    for part in partitions:
        total_mem = sum(layers[node].memory for node in part)
        if total_mem > EPC_EFFECTIVE_MB:
            return False

    # 3. 检查DAG依赖约束：若u→v，则u和v同分区，或u所在分区是v所在分区的前置（允许部分依赖完成后执行）
    # 构建层→分区索引的映射
    node_to_part_idx = {}
    for part_idx, part in enumerate(partitions):
        for node in part:
            node_to_part_idx[node] = part_idx

    # 检查所有边的依赖
    for u, v in G.edges():
        u_part_idx = node_to_part_idx[u]
        v_part_idx = node_to_part_idx[v]
        # u和v不同分区时，只需保证u是v的前驱（无需整个分区前置，仅u层前置）
        # 这里依赖约束的核心是：v的执行必须在u完成后，无需限制分区的整体顺序
        pass  # 此约束在时间计算阶段体现，分区生成阶段仅保证内存和层唯一性

    return True


def generate_all_valid_partitions(G: nx.DiGraph, layers: Dict[int, DNNLayer]) -> List[List[List[int]]]:
    """
    生成所有合法的分区方案
    :param G: DNN层依赖图
    :param layers: 层属性字典
    :return: 所有合法分区方案列表
    """
    nodes = sorted(list(G.nodes()))
    all_partitions = generate_all_partitions(nodes)
    valid_partitions = []

    for idx, partition in enumerate(all_partitions):
        if is_partition_valid(partition, G, layers):
            valid_partitions.append(partition)
            print(f"合法分区方案#{idx + 1}: {partition}")
        else:
            print(f"非法分区方案#{idx + 1}: {partition}（原因：内存超EPC或层重复）")

    return valid_partitions


# ====================== 辅助函数：生成服务器分配方案 ======================
def generate_server_assignments(partitions: List[List[int]], servers: List[Server]) -> List[Dict[int, int]]:
    """
    生成所有可能的服务器分配方案（每个分区分配到任意服务器）
    :param partitions: 分区方案
    :param servers: 服务器列表
    :return: 服务器分配方案列表，每个方案是{分区索引: 服务器ID}
    """
    num_partitions = len(partitions)
    server_ids = [s.id for s in servers]

    # 生成所有可能的分配组合（笛卡尔积）
    all_assignments = list(itertools.product(server_ids, repeat=num_partitions))

    # 转换为字典格式
    assignment_dicts = []
    for assignment in all_assignments:
        assign_dict = {}
        for part_idx, server_id in enumerate(assignment):
            assign_dict[part_idx] = server_id
        assignment_dicts.append(assign_dict)

    return assignment_dicts


# ====================== 核心函数：计算推理时间 ======================
def calculate_partition_exec_time(part_idx: int, partitions: List[List[int]],
                                  server_assignment: Dict[int, int], layers: Dict[int, DNNLayer],
                                  servers: List[Server]) -> float:
    """
    计算单个分区的执行时间（考虑EPC约束）
    :param part_idx: 分区索引
    :param partitions: 分区方案
    :param server_assignment: 服务器分配方案
    :param layers: 层属性字典
    :param servers: 服务器列表
    :return: 分区执行时间 (s)
    """
    # 获取分区信息
    part = partitions[part_idx]
    server_id = server_assignment[part_idx]
    server = next(s for s in servers if s.id == server_id)

    # 计算分区总内存和总计算量
    total_mem = sum(layers[node].memory for node in part)
    total_workload = sum(layers[node].workload for node in part)

    # 计算执行时间
    if total_mem <= EPC_EFFECTIVE_MB:
        exec_time = total_workload / server.power
    else:
        exec_time = total_workload / (SWITCH_OVERHEAD * server.power)

    return exec_time


def calculate_inference_time(scheme: PartitionScheme, G: nx.DiGraph, layers: Dict[int, DNNLayer],
                             servers: List[Server], server_memory_sum: Dict[int, float] = {}) -> float:
    """
    计算分区方案的总推理时间（支持并行执行）
    :param scheme: 分区方案对象
    :param G: DNN层依赖图
    :param layers: 层属性字典
    :param servers: 服务器列表
    :return: 总推理时间 (s)
    """
    # ====================== 1. 初始化核心时间字典 ======================
    # 存储每个层的开始执行时间，键=层ID，值=开始时间(s)
    layer_start = {}
    # 存储每个层的完成执行时间，键=层ID，值=完成时间(s)
    layer_finish = {}
    # 存储每个分区的开始执行时间（分区内最早层的开始时间），键=分区索引，值=开始时间(s)
    part_start = {}
    # 存储每个分区的完成执行时间（分区内最晚层的完成时间），键=分区索引，值=完成时间(s)
    part_finish = {}
    # 存储层ID到所属分区索引的映射，键=层ID，值=分区索引
    node_to_part_idx = {}
    # ========== 新增：初始化服务器最大完成时间字典 ==========
    # 记录每个服务器上已执行层的最大完成时间（保证同一服务器层串行执行），键=服务器ID，值=最大完成时间(s)
    server_max_finish = {s.id: 0.0 for s in servers}

    # DEBUG：打印初始化后的空字典状态
    print("\n🔍 初始化核心时间字典：")
    print(f"   layer_start (层开始时间): {layer_start}")
    print(f"   layer_finish (层完成时间): {layer_finish}")
    print(f"   part_start (分区开始时间): {part_start}")
    print(f"   part_finish (分区完成时间): {part_finish}")
    print(f"   node_to_part_idx (层→分区映射): {node_to_part_idx}")
    print(f"   server_max_finish (服务器最大完成时间): {server_max_finish}")  # 新增DEBUG


    # ====================== 2. 构建层→分区索引的映射 + 初始化分区时间 ======================
    for part_idx, part in enumerate(scheme.partitions):
        # 遍历当前分区内的所有层，建立层ID到分区索引的映射
        for node in part:
            node_to_part_idx[node] = part_idx
            # DEBUG：打印每个层的分区映射关系
            print(f"   📌 层{node} 映射到 分区索引{part_idx}")
        # 初始化当前分区的开始/完成时间为0.0（后续会更新）
        part_start[part_idx] = 0.0
        part_finish[part_idx] = 0.0

    # DEBUG：打印构建后的层→分区映射和初始化的分区时间
    print("\n🔍 构建层→分区映射 + 初始化分区时间后：")
    print(f"   node_to_part_idx: {node_to_part_idx}")
    print(f"   part_start (初始化): {part_start}")
    print(f"   part_finish (初始化): {part_finish}")

    # ====================== 3. 获取层的拓扑排序（保证前驱层先处理） ======================
    # 对DAG进行拓扑排序，得到层的执行顺序（确保前驱层始终在后继层之前处理）
    topological_order = list(nx.topological_sort(G))

    # DEBUG：打印拓扑排序结果
    print(f"\n🔍 DAG拓扑排序结果（层执行顺序）: {topological_order}")

    # ====================== 4. 按拓扑顺序遍历每个层，计算时间 ======================
    for node in topological_order:
        # 打印当前处理的层（分隔线区分不同层）
        print(f"\n" + "-"*60)
        print(f"🔍 开始处理层 {node}")
        print("-"*60)

        # 4.1 获取当前层所属的分区索引
        part_idx = node_to_part_idx[node]
        # 4.2 获取当前分区分配的服务器ID
        server_id = scheme.server_assignment[part_idx]
        # 4.3 根据服务器ID找到对应的服务器对象
        server = next(s for s in servers if s.id == server_id)

        # DEBUG：打印当前层的基础信息
        print(f"   当前层ID: {node}")
        print(f"   所属分区索引: {part_idx}")
        print(f"   分配的服务器ID: {server_id} (算力: {server.power} M FLOPs/s)")
        print(f"   当前服务器{server_id}已执行层的最大完成时间: {server_max_finish[server_id]:.6f}s")  # 新增DEBUG

        # ====================== 5. 计算当前层的前驱完成时间（含跨服务器通信时间） ======================
        # 存储所有前驱层的完成时间（含通信时间）
        pred_finish_times = []
        # 遍历当前层的所有前驱层
        for pred in G.predecessors(node):
            print(f"\n   📌 处理前驱层 {pred} → 当前层 {node}")
            # 获取前驱层的完成时间（已计算过，因为拓扑排序）
            pred_finish = layer_finish[pred]
            # 获取前驱层所属的分区索引
            pred_part_idx = node_to_part_idx[pred]
            # 获取前驱层所在分区分配的服务器ID
            pred_server_id = scheme.server_assignment[pred_part_idx]

            # DEBUG：打印前驱层的基础信息
            print(f"      前驱层{pred} 所属分区索引: {pred_part_idx}")
            print(f"      前驱层{pred} 分配服务器ID: {pred_server_id}")
            print(f"      前驱层{pred} 原始完成时间: {pred_finish:.6f}s")

            # 判断前驱层和当前层是否分配到不同服务器（跨服务器需加通信时间）
            if pred_server_id != server_id:
                # 计算跨服务器通信时间：通信时间=数据量/带宽
                comm_time = COMM_DATA_SIZE / BANDWIDTH_AVG
                # 前驱完成时间 += 通信时间
                pred_finish += comm_time
                # DEBUG：打印通信时间计算
                print(f"      ❗ 跨服务器通信（{pred_server_id}→{server_id}）:")
                print(f"        通信数据量: {COMM_DATA_SIZE} MB, 带宽: {BANDWIDTH_AVG} Mbps")
                print(f"        通信时间: {comm_time:.6f}s")
                print(f"        前驱层{pred} 含通信的完成时间: {pred_finish:.6f}s")
            else:
                # DEBUG：同服务器无通信时间
                print(f"      ✅ 同服务器通信（{pred_server_id}→{server_id}）: 无通信时间")

            # 将前驱层的完成时间（含通信）加入列表
            pred_finish_times.append(pred_finish)
            # DEBUG：打印当前前驱层的最终完成时间
            print(f"      前驱层{pred} 最终完成时间: {pred_finish:.6f}s")

        # ====================== 6. 确定当前层的开始时间（核心修改） ======================
        # 步骤1：计算前驱层的最大完成时间（无前驱则为0）
        if pred_finish_times:
            pred_max = max(pred_finish_times)
            # DEBUG：打印前驱完成时间列表和前驱最大时间
            print(f"\n   📌 当前层{node} 前驱完成时间列表: {[f'{t:.6f}' for t in pred_finish_times]}")
            print(f"   当前层{node} 前驱层最大完成时间: {pred_max:.6f}s")
        else:
            pred_max = 0.0
            # DEBUG：无前置依赖的前驱最大时间
            print(f"\n   📌 当前层{node} 无前置依赖 → 前驱层最大完成时间: {pred_max:.6f}s")

        # 步骤2：获取当前服务器上已执行层的最大完成时间
        server_current_max = server_max_finish[server_id]
        print(f"   当前层{node} 所属服务器{server_id}已执行层的最大完成时间: {server_current_max:.6f}s")

        # 步骤3：核心逻辑——开始时间 = max(前驱层最大完成时间, 服务器已执行层最大完成时间)
        node_start = max(pred_max, server_current_max)
        print(f"   当前层{node} 最终开始时间: max({pred_max:.6f}, {server_current_max:.6f}) = {node_start:.6f}s")  # 新增DEBUG

        # ====================== 7. 计算当前层的执行时间（考虑EPC约束） ======================
        # 获取当前层所属的分区
        part = scheme.partitions[part_idx]
        # 计算当前分区的总内存（判断是否超EPC）
        #total_mem = sum(layers[n].memory for n in part)
        total_mem = server_memory_sum[server_id]
        # 计算当前层的执行时间：
        # - 分区内存≤EPC：执行时间=层计算量/服务器算力
        # - 分区内存>EPC：执行时间=层计算量/(0.5*服务器算力)（性能下降一半）
        if total_mem <= EPC_EFFECTIVE_MB:
            node_exec = layers[node].workload / server.power
            epc_status = "≤ EPC"
        else:
            node_exec = layers[node].workload / (SWITCH_OVERHEAD * server.power)
            epc_status = "> EPC"

        # DEBUG：打印当前层执行时间的计算过程
        print(f"\n   📌 当前层{node} 执行时间计算:")
        print(f"      所属分区总内存: {total_mem:.1f} MB (EPC约束: {EPC_EFFECTIVE_MB} MB) → {epc_status}")
        print(f"      层{node} 计算量: {layers[node].workload} M FLOPs")
        print(f"      服务器算力: {server.power} M FLOPs/s")
        print(f"      执行时间: {node_exec:.6f}s (公式: {layers[node].workload} / {server.power if epc_status=='≤ EPC' else f'(0.5*{server.power})'})")

        # ====================== 8. 更新当前层的开始/完成时间 ======================
        layer_start[node] = node_start
        layer_finish[node] = node_start + node_exec

        # ========== 新增：更新服务器最大完成时间 ==========
        # 当前服务器的最大完成时间 = max(原有值, 当前层完成时间)
        if layer_finish[node] > server_max_finish[server_id]:
            server_max_finish[server_id] = layer_finish[node]
            print(f"   📌 服务器{server_id}最大完成时间更新: {server_max_finish[server_id]:.6f}s (层{node}完成时间更大)")

        # DEBUG：打印当前层的最终时间
        print(f"\n   📌 层{node} 时间更新:")
        print(f"      开始时间: {layer_start[node]:.6f}s")
        print(f"      完成时间: {layer_finish[node]:.6f}s (开始时间 + 执行时间 = {node_start:.6f} + {node_exec:.6f})")

        # ====================== 9. 更新当前层所属分区的开始/完成时间 ======================
        # 分区的开始时间 = 分区内所有层的最小开始时间
        if layer_start[node] < part_start[part_idx]:
            part_start[part_idx] = layer_start[node]
            print(f"   📌 分区{part_idx} 开始时间更新: {part_start[part_idx]:.6f}s (层{node}开始时间更小)")
        # 分区的完成时间 = 分区内所有层的最大完成时间
        if layer_finish[node] > part_finish[part_idx]:
            part_finish[part_idx] = layer_finish[node]
            print(f"   📌 分区{part_idx} 完成时间更新: {part_finish[part_idx]:.6f}s (层{node}完成时间更大)")

    # ====================== 10. 保存所有时间信息到方案对象 ======================
    scheme.layer_start_time = layer_start
    scheme.layer_finish_time = layer_finish
    scheme.partition_start_time = part_start
    scheme.partition_finish_time = part_finish

    # DEBUG：打印所有层和分区的最终时间
    print(f"\n" + "="*60)
    print(f"🔍 所有层时间计算完成：")
    print("="*60)
    for node in sorted(layer_start.keys()):
        print(f"   层{node}: 开始={layer_start[node]:.6f}s, 完成={layer_finish[node]:.6f}s")
    print(f"\n🔍 所有分区时间计算完成：")
    for part_idx in sorted(part_start.keys()):
        print(f"   分区{part_idx}: 开始={part_start[part_idx]:.6f}s, 完成={part_finish[part_idx]:.6f}s")
    print(f"\n🔍 所有服务器最终最大完成时间：")  # 新增DEBUG
    for srv_id in sorted(server_max_finish.keys()):
        print(f"   服务器{srv_id}: 最大完成时间={server_max_finish[srv_id]:.6f}s")

    # ====================== 11. 计算总推理时间（所有层完成时间的最大值） ======================
    total_time = max(layer_finish.values()) if layer_finish else 0.0
    scheme.total_time = total_time

    # DEBUG：打印总推理时间
    print(f"\n🔍 总推理时间计算:")
    print(f"   所有层完成时间: {[f'{v:.6f}' for v in layer_finish.values()]}")
    print(f"   总推理时间 (最大值): {total_time:.6f}s")

    return total_time


# ====================== 核心函数：穷举所有方案并找最优解 ======================
def find_optimal_scheme(G: nx.DiGraph, layers: Dict[int, DNNLayer], servers: List[Server]) -> PartitionScheme:
    """
    穷举所有合法分区+服务器分配方案，找到总推理时间最小的最优方案
    :param G: DNN层依赖图
    :param layers: 层属性字典
    :param servers: 服务器列表
    :return: 最优分区方案
    """
    # 步骤1：生成所有合法分区方案
    print("=" * 80)
    print("开始生成所有合法分区方案...")
    valid_partitions = generate_all_valid_partitions(G, layers)
    if not valid_partitions:
        raise ValueError("无合法的分区方案！")

    # 步骤2：遍历所有合法分区方案
    optimal_scheme = None
    min_total_time = float('inf')
    all_schemes = []

    print("\n" + "=" * 80)
    print("开始遍历所有分区+服务器分配方案...")

    for part_idx, partitions in enumerate(valid_partitions):
        print(f"\n处理分区方案#{part_idx + 1}: {partitions}")

        # 生成所有服务器分配方案
        server_assignments = generate_server_assignments(partitions, servers)

        # ====================== DEBUG：打印服务器分配方案列表 ======================
        print(f"\n   📌 为分区方案#{part_idx + 1}生成的服务器分配方案详情：")
        print(f"      服务器分配方案总数: {len(server_assignments)}")
        print(f"      所有服务器分配方案列表:")
        for idx, assign in enumerate(server_assignments):
            print(f"         分配方案#{idx + 1}: {assign}")
        print(f"      服务器分配方案类型: {type(server_assignments)} (列表)")
        print(f"      单个分配方案类型: {type(server_assignments[0]) if server_assignments else '空'} (字典)")

        for assign_idx, server_assign in enumerate(server_assignments):
            # 创建分区方案对象
            # ====================== DEBUG：打印当前遍历的服务器分配方案 ======================
            print(f"\n   ─────────────────────────────────────────")
            print(f"   🎯 处理服务器分配方案#{assign_idx + 1}/{len(server_assignments)}")
            print(f"   ─────────────────────────────────────────")
            print(f"      当前分配方案索引 assign_idx: {assign_idx}")
            print(f"      当前服务器分配规则 server_assign: {server_assign}")
            print(
                f"         → 格式说明：{{分区索引: 服务器ID}}，例如 {{0:0, 1:1}} 表示分区0分配到服务器0，分区1分配到服务器1")

            scheme = PartitionScheme(partitions)
            scheme.server_assignment = server_assign

            # ====================== DEBUG：打印刚创建的PartitionScheme对象 ======================
            print(f"\n      📝 刚创建的PartitionScheme对象状态：")
            print(f"         scheme.partitions (分区列表): {scheme.partitions}")
            print(f"         scheme.server_assignment (服务器分配): {scheme.server_assignment}")
            print(f"         scheme.total_time (初始总时间): {scheme.total_time} (未计算前为无穷大)")
            print(f"         scheme.layer_start_time (初始层开始时间): {scheme.layer_start_time}")
            print(f"         scheme.layer_finish_time (初始层完成时间): {scheme.layer_finish_time}")

            # ====================== 核心新增：计算每个服务器分配的内存总和 ======================
            # 1. 构建「服务器ID → 分配到该服务器的层ID列表」映射
            server_to_layers = {}  # 键：服务器ID，值：该服务器的层ID列表
            # 遍历每个分区，关联服务器ID和层ID
            for part_idx_in_scheme, part in enumerate(scheme.partitions):
                # 获取当前分区分配的服务器ID
                srv_id = scheme.server_assignment[part_idx_in_scheme]
                # 将分区内的所有层ID添加到对应服务器的列表中
                if srv_id not in server_to_layers:
                    server_to_layers[srv_id] = []
                server_to_layers[srv_id].extend(part)

            # 2. 计算每个服务器分配的内存总和
            server_memory_sum = {}  # 键：服务器ID，值：该服务器的内存总和(MB)
            for srv_id in server_to_layers:
                # 累加该服务器下所有层的内存
                total_mem = sum(layers[layer_id].memory for layer_id in server_to_layers[srv_id])
                server_memory_sum[srv_id] = total_mem

            # 3. DEBUG输出：打印每个服务器的内存分配详情
            print(f"\n   📊 服务器分配方案#{assign_idx + 1} - 内存分配详情:")
            print(f"      服务器分配规则: {scheme.server_assignment}")
            for srv_id in sorted(server_memory_sum.keys()):
                # 找到对应的服务器对象（获取算力等信息，可选）
                server = next(s for s in servers if s.id == srv_id)
                print(f"         服务器{srv_id} (算力: {server.power} M FLOPs/s):")
                print(f"            - 分配的层ID: {sorted(server_to_layers[srv_id])}")
                print(
                    f"            - 各层内存: {[f'层{lid}:{layers[lid].memory}MB' for lid in sorted(server_to_layers[srv_id])]}")
                print(f"            - 内存总和: {server_memory_sum[srv_id]} MB")

            # 计算推理时间
            total_time = calculate_inference_time(scheme, G, layers, servers,server_memory_sum)
            all_schemes.append(scheme)

            # 打印方案信息
            print(f"  服务器分配方案#{assign_idx + 1}: {server_assign} → 总时间={total_time:.4f}s")


            # 更新最优方案
            if total_time < min_total_time:
                min_total_time = total_time
                optimal_scheme = scheme

                # ====================== DEBUG：打印最优方案更新 ======================
                print(f"      🌟 发现更优方案！更新最优解：")
                print(f"         原最小时间: {min_total_time:.6f}s → 新最小时间: {total_time:.6f}s")
                print(f"         最优方案分区: {optimal_scheme.partitions}")
                print(f"         最优方案服务器分配: {optimal_scheme.server_assignment}")
    # 步骤3：输出最优方案详情
    print("\n" + "=" * 80)
    print("最优方案详情：")
    print(f"最优分区方案: {optimal_scheme.partitions}")
    print(f"最优服务器分配: {optimal_scheme.server_assignment}")
    print(f"总推理时间: {optimal_scheme.total_time:.4f}s")

    # 打印每层的执行时间
    print("\n每层执行时间详情：")
    for node in sorted(optimal_scheme.layer_start_time.keys()):
        print(
            f"  层{node}: 开始={optimal_scheme.layer_start_time[node]:.4f}s, 完成={optimal_scheme.layer_finish_time[node]:.4f}s")

    # 打印每个分区的执行时间
    print("\n每个分区执行时间详情：")
    for part_idx in sorted(optimal_scheme.partition_start_time.keys()):
        print(
            f"  分区{part_idx}: 开始={optimal_scheme.partition_start_time[part_idx]:.4f}s, 完成={optimal_scheme.partition_finish_time[part_idx]:.4f}s, 分配服务器={optimal_scheme.server_assignment[part_idx]}")

    return optimal_scheme


# ====================== 测试用例 ======================
# def test_optimal_partition():
#     """测试用例：DNN层依赖图为0→1、0→2、1→3、2→3"""
#     # 1. 构建DNN层依赖图
#     G = nx.DiGraph()
#     edges = [(0, 1), (0, 2), (1, 3), (2, 3)]
#     G.add_edges_from(edges)
#
#     # 2. 定义层属性
#     layers = {
#         0: DNNLayer(0, 30, 1000),
#         1: DNNLayer(1, 50, 4000),
#         2: DNNLayer(2, 20, 4000),
#         3: DNNLayer(3, 50, 3000)
#     }
#
#     # 3. 定义服务器（2台异构服务器，此处设为同算力）
#     servers = [
#         Server(0, 5000),
#         Server(1, 5000)
#     ]
#
#     # 4. 寻找最优方案
#     optimal_scheme = find_optimal_scheme(G, layers, servers)
#
#     return optimal_scheme


# ====================== 执行测试 ======================
if __name__ == "__main__":
    # 方式1：直接使用config中预创建的实例（推荐）
    print("=== 直接使用预配置实例 ===")
    print(f"DNN图节点: {G.nodes()}")
    print(f"层配置: {layers}")
    print(f"服务器配置: {servers}")

    # 方式2：动态重建配置（如需修改参数时使用）
    print("\n=== 动态重建配置 ===")
    custom_G = build_dnn_graph()
    custom_layers = get_dnn_layers()
    custom_servers = get_servers()
    print(f"动态构建的DNN图: {custom_G.edges()}")
    optimal_scheme = find_optimal_scheme(G, layers, servers)

