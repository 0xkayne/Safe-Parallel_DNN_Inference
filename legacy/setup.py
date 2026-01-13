# config.py
import networkx as nx

# ====================== 全局常量配置 ======================
EPC_EFFECTIVE_MB = 100  # EPC有效内存约束 (MB)
BANDWIDTH_AVG = 10  # 服务器间平均带宽 (Mbps)
COMM_DATA_SIZE = 1.0  # 跨服务器通信数据量 (MB)
SWITCH_OVERHEAD = 0.5


# ====================== 核心类定义 ======================
class DNNLayer:
    """DNN层类，包含层ID、内存、计算量"""

    def __init__(self, id: int, memory: float, workload: float):
        self.id = id
        self.memory = memory  # 内存 (MB)
        self.workload = workload  # 计算量 (M FLOPs)
        self.schedule = []  # [(start_time, end_time, partition)]

    def __repr__(self):
        return f"Layer({self.id}, mem={self.memory}, workload={self.workload})"


class Server:
    """服务器类，包含服务器ID、算力"""

    def __init__(self, id: int, power: float):
        self.id = id
        self.power = power  # 算力 (M FLOPs/s)
        self.schedule = []  # [(start_time, end_time, partition)]
        self.asseignedmemory = 0

    def __repr__(self):
        return f"Server({self.id}, power={self.power})"


# ====================== DNN模型配置 ======================
# 1. 构建DNN层依赖图（DAG）
def build_dnn_graph() -> nx.DiGraph:
    """构建并返回DNN层依赖图"""
    G = nx.DiGraph()
    edges = [(0, 1), (0, 2), (1, 3), (2, 3)]  # 层依赖关系
    G.add_edges_from(edges)
    return G


# 2. 定义层属性字典
def get_dnn_layers() -> dict:
    """返回DNN层属性配置"""
    layers = {
        0: DNNLayer(0, 60, 1000),
        1: DNNLayer(1, 51, 4000),
        2: DNNLayer(2, 20, 4000),
        3: DNNLayer(3, 50, 3000)
    }
    return layers


# ====================== 服务器配置 ======================
def get_servers() -> list:
    """返回服务器列表配置（支持异构）"""
    servers = [
        Server(0, 5000),  # 服务器0：算力5000 M FLOPs/s
        Server(1, 5000),  # 服务器1：算力5000 M FLOPs/s（可修改为异构，如4000）
        Server(2, 5000),
        Server(3, 5000)
    ]
    return servers


# ====================== 快捷导出（可选） ======================
# 直接创建配置实例，避免每次调用函数
G = build_dnn_graph()
layers = get_dnn_layers()
servers = get_servers()