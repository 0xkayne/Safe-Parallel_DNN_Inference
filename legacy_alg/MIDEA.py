###############失败的不成熟的代码··············
import networkx as nx
from typing import Dict, List, Set, Tuple
import math


class DNNLayer:
    """DNN 层（图的顶点）数据结构"""

    def __init__(self, layer_id: int, memory: float, workload: float):
        self.layer_id = layer_id  # 层ID（唯一标识）
        self.memory = memory  # 该层内存需求（MB）
        self.workload = workload  # 该层计算量（M FLOPs）


class DNNPartition:
    """DNN 分区数据结构"""

    def __init__(self, partition_id: int):
        self.partition_id = partition_id  # 分区ID
        self.layers = set()  # 该分区包含的层ID集合
        self.total_memory = 0.0  # 分区总内存需求（MB）
        self.total_workload = 0.0  # 分区总计算量（M FLOPs）


class MEDIAPartitioner:
    """MEDIA 分区算法实现（论文 Algorithm 1 + Algorithm 2）"""

    def __init__(self, epc_capacity: float = 93.0):
        """
        初始化分区器
        :param epc_capacity: SGX EPC 有效容量（MB），论文中默认93MB（128MB总容量 - 35MB元数据）
        """
        self.epc = epc_capacity  # EPC 内存上限
        self.graph = None  # DNN 图（networkx.DiGraph）
        self.layer_dict = {}  # 层ID到DNNLayer的映射：{layer_id: DNNLayer}
        self.layer_level = {}  # 层ID到拓扑层级的映射：{layer_id: int}
        self.selected_edges = set()  # 算法1选择的边集合 M
        self.partitions = []  # 最终分区结果：[DNNPartition]
        self.partition_map = {}  # 层ID到分区ID的映射：{layer_id: partition_id}

        # 模拟系统参数（论文实验设定）
        self.avg_server_flops = 100  # 边缘服务器平均计算能力（M FLOPs/s）
        self.avg_bandwidth = 10  # 服务器间平均带宽（Mbps）

    def _topological_sort(self) -> List[int]:
        """
        对DNN图进行拓扑排序，生成各层的层级（L(u)）
        :return: 拓扑排序后的层ID列表
        """
        if not nx.is_directed_acyclic_graph(self.graph):
            raise ValueError("DNN模型图必须是有向无环图（DAG）")

        # 拓扑排序（输入层在前，输出层在后）
        topo_order = list(nx.topological_sort(self.graph))
        # 分配层级（从1开始递增）
        for idx, layer_id in enumerate(topo_order):
            self.layer_level[layer_id] = idx + 1
        return topo_order

    def _check_constraint1(self, u: int, v: int) -> bool:
        """
        检查边(u, v)是否满足论文 Constraint 1：
        Succ(u) = {v}（u的后继只有v） 或 Pred(v) = {u}（v的前驱只有u）
        :param u: 边的起点层ID
        :param v: 边的终点层ID
        :return: 满足约束返回True，否则False
        """
        # 获取u的所有后继和v的所有前驱
        u_successors = set(self.graph.successors(u))
        v_predecessors = set(self.graph.predecessors(v))
        return (u_successors == {v}) or (v_predecessors == {u})

    def _check_constraint2(self, u: int, v: int) -> bool:
        """
        检查边(u, v)是否满足论文 Constraint 2：
        对于已选边集合中的任意边(w', w)，不存在 L(u) + 1 = L(w)
        避免形成环结构
        :param u: 边的起点层ID
        :param v: 边的终点层ID
        :return: 满足约束返回True，否则False
        """
        u_level = self.layer_level[u]
        for (w_prime, w) in self.selected_edges:
            if self.layer_level[w] == u_level + 1:
                return False
        return True

    def edge_selection(self) -> Set[Tuple[int, int]]:
        """
        论文 Algorithm 1：边选择算法，筛选可合并的边集合M
        :return: 满足约束的边集合M
        """
        print("开始执行边选择算法（Algorithm 1）...")
        # 1. 先对图进行拓扑排序，确定各层层级
        topo_order = self._topological_sort()

        # 2. 按层级递增顺序遍历所有顶点
        for u in topo_order:
            # 按边优先级遍历u的后继（这里简化为默认顺序）
            for v in self.graph.successors(u):
                # 检查Constraint 1：u的后继只有v 或 v的前驱只有u
                if not self._check_constraint1(u, v):
                    continue
                # 检查Constraint 2：避免与已选边形成环
                if not self._check_constraint2(u, v):
                    continue
                # 满足所有约束，加入边集合M
                self.selected_edges.add((u, v))
                print(f"选择边 ({u} -> {v})，满足约束1和约束2")

        print(f"边选择完成，共选择 {len(self.selected_edges)} 条边")
        return self.selected_edges

    def _estimate_execution_time(self, partition: DNNPartition) -> float:
        """
        估计分区的执行时间（论文公式9）
        :param partition: 待估计的分区
        :return: 执行时间（秒）
        """
        # 执行时间 = 分区总计算量 / 服务器平均计算能力
        return partition.total_workload / self.avg_server_flops

    def _estimate_comm_time(self, p1: DNNPartition, p2: DNNPartition) -> float:
        """
        估计两个依赖分区的通信时间（论文公式10）
        :param p1: 前驱分区
        :param p2: 后继分区
        :return: 通信时间（秒）
        """
        # 计算两个分区之间的总传输数据量（所有跨分区边的data_size之和）
        total_data = 0.0
        for u in p1.layers:
            for v in p2.layers:
                if self.graph.has_edge(u, v):
                    total_data += self.graph.edges[u, v]['data_size']

        # 通信时间 = 总数据量（Mb） / 带宽（Mbps）（1Byte=8bit，转换单位）
        total_data_mb = total_data * 8 / 1024  # 转换为Mb
        return total_data_mb / self.avg_bandwidth

    def _check_merge(self, p1: DNNPartition, p2: DNNPartition) -> bool:
        """
        论文 Check 函数（Algorithm 2 第21-24行）：判断两个分区是否应合并
        合并条件：
        1. 合并后总内存 ≤ EPC容量，或
        2. 合并后执行时间 ≤ 分开执行时间 + 通信时间
        :param p1: 分区1
        :param p2: 分区2
        :return: 应合并返回True，否则False
        """
        # 计算合并后的分区属性
        merged_memory = p1.total_memory + p2.total_memory
        merged_workload = p1.total_workload + p2.total_workload
        merged_partition = DNNPartition(-1)  # 临时分区，仅用于计算
        merged_partition.total_memory = merged_memory
        merged_partition.total_workload = merged_workload

        # 条件1：合并后内存不超过EPC容量
        if merged_memory <= self.epc:
            return True

        # 条件2：合并后执行时间 ≤ 分开执行时间 + 通信时间
        t_merged = self._estimate_execution_time(merged_partition)
        t_p1 = self._estimate_execution_time(p1)
        t_p2 = self._estimate_execution_time(p2)
        t_comm = self._estimate_comm_time(p1, p2)

        return t_merged <= (t_p1 + t_p2 + t_comm)

    def graph_partition(self):
        """
        论文 Algorithm 2：基于选择的边集合M进行图分区
        按三种情况合并分区，生成最终分区结果
        """
        print("\n开始执行图分区算法（Algorithm 2）...")
        # 初始化：每个层作为独立分区
        self.partitions = []
        self.partition_map = {}
        for layer_id in self.graph.nodes:
            partition = DNNPartition(partition_id=layer_id)
            partition.layers.add(layer_id)
            partition.total_memory = self.layer_dict[layer_id].memory
            partition.total_workload = self.layer_dict[layer_id].workload
            self.partitions.append(partition)
            self.partition_map[layer_id] = partition.partition_id

        # 遍历所有选中的边，合并分区
        for (u, v) in self.selected_edges:
            # 获取u和v所在的分区
            p_u_id = self.partition_map[u]
            p_v_id = self.partition_map[v]
            p_u = next(p for p in self.partitions if p.partition_id == p_u_id)
            p_v = next(p for p in self.partitions if p.partition_id == p_v_id)

            # 情况1：两边顶点都未合并（此时p_u和p_v都是单一层分区）
            if len(p_u.layers) == 1 and len(p_v.layers) == 1:
                if self._check_merge(p_u, p_v):
                    self._merge_partitions(p_u, p_v)

            # 情况2：两边顶点都已合并（p_u和p_v都是多层层分区）
            elif len(p_u.layers) > 1 and len(p_v.layers) > 1:
                if self._check_merge(p_u, p_v):
                    self._merge_partitions(p_u, p_v)

            # 情况3：只有一个顶点已合并
            else:
                # 确定已合并的分区和未合并的顶点
                if len(p_u.layers) > 1:
                    merged_part = p_u
                    unmerged_layer_id = v
                else:
                    merged_part = p_v
                    unmerged_layer_id = u

                # 构建未合并顶点的临时分区
                unmerged_part = DNNPartition(-1)
                unmerged_part.layers.add(unmerged_layer_id)
                unmerged_part.total_memory = self.layer_dict[unmerged_layer_id].memory
                unmerged_part.total_workload = self.layer_dict[unmerged_layer_id].workload

                # 检查是否可以合并
                if self._check_merge(merged_part, unmerged_part):
                    # 将未合并顶点加入已合并分区
                    merged_part.layers.add(unmerged_layer_id)
                    merged_part.total_memory += unmerged_part.total_memory
                    merged_part.total_workload += unmerged_part.total_workload
                    self.partition_map[unmerged_layer_id] = merged_part.partition_id
                    print(f"合并未合并层 {unmerged_layer_id} 到分区 {merged_part.partition_id}")

        # 清理空分区（合并后产生的冗余）
        self.partitions = [p for p in self.partitions if len(p.layers) > 0]
        print(f"图分区完成，共生成 {len(self.partitions)} 个分区")

    def _merge_partitions(self, p1: DNNPartition, p2: DNNPartition):
        """
        合并两个分区p1和p2（p1为前驱，p2为后继）
        :param p1: 前驱分区
        :param p2: 后继分区
        """
        # 将p2的所有层合并到p1
        for layer_id in p2.layers:
            p1.layers.add(layer_id)
            self.partition_map[layer_id] = p1.partition_id
        p1.total_memory += p2.total_memory
        p1.total_workload += p2.total_workload

        # 标记p2为已合并（后续清理）
        p2.layers.clear()
        print(f"合并分区 {p2.partition_id} 到分区 {p1.partition_id}")

    def run(self, dnn_graph: nx.DiGraph, layer_dict: Dict[int, DNNLayer]):
        """
        执行完整的MEDIA分区流程
        :param dnn_graph: DNN模型的有向无环图（networkx.DiGraph）
        :param layer_dict: 层ID到DNNLayer的映射
        :return: 最终分区结果列表
        """
        # 初始化输入
        self.graph = dnn_graph
        self.layer_dict = layer_dict

        # 步骤1：边选择
        self.edge_selection()

        # 步骤2：图分区
        self.graph_partition()

        # 输出分区详情
        self.print_partition_details()
        return self.partitions

    def print_partition_details(self):
        """打印分区详细信息"""
        print("\n=== 最终分区结果 ===")
        for idx, partition in enumerate(self.partitions):
            print(f"\n分区 {partition.partition_id}：")
            print(f"  包含层ID：{sorted(partition.layers)}")
            print(f"  总内存需求：{partition.total_memory:.2f} MB")
            print(f"  总计算量：{partition.total_workload:.2f} M FLOPs")
            print(f"  是否超过EPC：{'是' if partition.total_memory > self.epc else '否'}")
            print(f"  估计执行时间：{self._estimate_execution_time(partition):.2f} s")


# ------------------------------ 测试代码 ------------------------------
if __name__ == "__main__":
    # 1. 构建一个简单的DNN模型图（示例：NiN模型简化版，线性结构）
    # 创建有向无环图
    dnn_graph = nx.DiGraph()

    # 定义5个层（层ID：1-5），设置内存需求和计算量
    layer_dict = {
        1: DNNLayer(layer_id=1, memory=15.2, workload=200),  # 输入层
        2: DNNLayer(layer_id=2, memory=22.8, workload=350),  # 卷积层1
        3: DNNLayer(layer_id=3, memory=18.5, workload=300),  # 卷积层2
        4: DNNLayer(layer_id=4, memory=25.1, workload=400),  # 全连接层1
        5: DNNLayer(layer_id=5, memory=12.3, workload=150)  # 输出层
    }

    # 添加层（顶点）和依赖（边），边的data_size表示跨层传输数据量（MB）
    edges = [
        (1, 2, {'data_size': 8.5}),
        (2, 3, {'data_size': 6.2}),
        (3, 4, {'data_size': 4.8}),
        (4, 5, {'data_size': 2.1})
    ]
    dnn_graph.add_nodes_from(layer_dict.keys())
    for u, v, attr in edges:
        dnn_graph.add_edge(u, v, **attr)

    # 2. 执行MEDIA分区算法
    partitioner = MEDIAPartitioner(epc_capacity=93.0)  # EPC有效容量93MB
    partitions = partitioner.run(dnn_graph, layer_dict)