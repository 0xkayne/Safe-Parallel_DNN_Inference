import pandas as pd
import ast

class LRUEPC:
    """
    模拟英特尔 SGX 的飞地页面缓存 (EPC) 行为。
    核心逻辑：维护一个 LRU (最近最少使用) 列表来跟踪驻留在 EPC 中的张量。
    当内存不足时，触发分页 (Eviction)，产生 Tier 3 时延。
    """
    def __init__(self, capacity_bytes, page_fault_time_ms, page_size_bytes=4096):
        # EPC 物理容量限制 (通常为 128MB 或 256MB)
        self.capacity = capacity_bytes
        # 当前已使用的 EPC 大小
        self.current_usage = 0
        # 单页分页开销 (Tier 3 时延，文档中约为 10-40us)
        self.page_fault_time_ms = page_fault_time_ms
        # 页面大小 (标准为 4KB)
        self.page_size = page_size_bytes
        # 驻留张量字典 {张量名: 大小}，利用 Python 字典的有序性模拟 LRU
        self.resident_tensors = {} 
        # 累计的分页置换（Eviction）时间
        self.paging_overhead_accumulated = 0.0
    
    def touch(self, name):
        """
        更新张量的访问状态 (LRU 策略)。
        如果张量被访问，将其移动到字典末尾 (表示最近使用)。
        这是 Tier 1 (驻留访问) 的模拟，时延极低，忽略不计。
        """
        if name in self.resident_tensors:
            size = self.resident_tensors.pop(name)
            self.resident_tensors[name] = size
            
    def ensure_loaded(self, name, size):
        """
        确保张量在 EPC 中。如果不在，则需要从 DRAM 加载 (ELDU 操作)。
        返回：加载产生的分页时延 (ms)。
        """
        # 情况 1: 张量已经在 EPC 中 (Tier 1)
        if name in self.resident_tensors:
            self.touch(name)
            return 0.0 # 命中缓存，无额外分页开销
        
        # 情况 2: 张量不在 EPC 中，需要加载 (Tier 3 Page In)
        # 计算需要加载的页面数量
        num_pages = (size + self.page_size - 1) // self.page_size
        # 计算加载时延：页数 * 单页开销
        load_cost = num_pages * self.page_fault_time_ms
        
        # 在 EPC 中分配空间 (如果已满，会触发驱逐)
        self._allocate_space(size, name)
        
        return load_cost

    def _allocate_space(self, size, new_tensor_name):
        """
        在 EPC 中分配内存。如果空间不足，执行 LRU 驱逐策略。
        """
        # 检查是否需要驱逐旧页面
        while self.current_usage + size > self.capacity:
            # 如果 EPC 已满且需要分配，必须驱逐 (Page Out / EWB)
            if not self.resident_tensors:
                # 极端情况：单个张量大小超过 EPC 总容量 (文档中的 "Giant Operator")
                # 此时发生剧烈抖动 (Thrashing)，由 get_thrashing_cost 处理
                break
            
            # 获取 LRU 队列中最老的张量 (字典第一个键)
            evict_name = next(iter(self.resident_tensors))
            evict_size = self.resident_tensors.pop(evict_name)
            
            # 释放空间
            self.current_usage -= evict_size
            
            # 计算驱逐开销 (Tier 3 Page Out)
            # 这里的假设是脏页回写 (EWB) 需要时间
            num_pages = (evict_size + self.page_size - 1) // self.page_size
            # 累加驱逐产生的时延
            self.paging_overhead_accumulated += num_pages * self.page_fault_time_ms
            
        # 分配新空间
        self.current_usage += size
        # 记录新张量
        self.resident_tensors[new_tensor_name] = size

    def get_thrashing_cost(self, total_needed):
        """
        计算“内存抖动”开销。
        如果单层所需的总工作集 (输入+权重+输出+激活) 超过 EPC 容量，
        系统将陷入频繁的换入换出 (Thrashing)，这是 MEDIA 算法的主要瓶颈。
        """
        if total_needed > self.capacity:
            # 超出部分的大小
            excess = total_needed - self.capacity
            num_pages = (excess + self.page_size - 1) // self.page_size
            # 惩罚系数：假设超出的内存需要频繁读写，给予 2 倍分页惩罚 (Swap Out + Swap In)
            return num_pages * self.page_fault_time_ms * 2
        return 0.0

def parse_dependencies(dep_str):
    """辅助函数：解析 CSV 中的依赖列表字符串"""
    if pd.isna(dep_str) or dep_str == '[]':
        return []
    try:
        return ast.literal_eval(dep_str)
    except:
        return []

def simulate_latency(csv_path, epc_mb=128):
    """
    主仿真函数：计算端到端推理时延。
    """
    # 读取 CSV 文件
    df = pd.read_csv(csv_path)
    
    # --- 仿真参数设定 (基于文档) ---
    # EPC 容量限制 (字节): 128 MB
    EPC_CAPACITY = epc_mb * 1024 * 1024
    # 单页分页 (Page Fault) 时延: 40 微秒 = 0.04 毫秒
    # 数据来源: 文档中提到的 "10-40us" 以及 "4500次置换导致185ms延迟" (~41us)
    PAGE_FAULT_MS = 0.04
    PAGE_SIZE = 4096
    
    # 初始化 EPC 模拟器
    epc = LRUEPC(EPC_CAPACITY, PAGE_FAULT_MS, PAGE_SIZE)
    
    # 结果统计变量
    total_compute_time = 0.0 # Tier 0/1 计算时延
    total_comm_time = 0.0    # Tier 4 通信时延
    total_paging_time = 0.0  # Tier 3 分页时延 (关键瓶颈)
    
    print(f"开始仿真，EPC 限制: {epc_mb} MB...")
    
    # 遍历神经网络的每一层 (Layer-wise Simulation)
    for index, row in df.iterrows():
        layer_name = row['name']
        
        # 1. 获取该层的内存需求 (Tier 0/1/3 相关)
        # 权重 + 偏置 (长期驻留)
        w_size = row.get('weight_bytes', 0) + row.get('bias_bytes', 0)
        # 输入数据 (来自上一层)
        i_size = row.get('input_bytes', 0)
        # 输出数据 (将传递给下一层)
        o_size = row.get('output_bytes', 0)
        # 激活值 (临时中间变量)
        a_size = row.get('activation_bytes', 0)
        
        # 解析依赖关系，确定输入张量的来源
        deps = parse_dependencies(row['dependencies'])
        
        # 2. 加载阶段 (Load Phase) - 产生 Tier 3 Page In 开销
        # 每一层计算前，必须保证权重和输入在 EPC 中
        
        # --- 处理权重 ---
        w_id = f"{layer_name}_W"
        layer_paging_cost = 0.0
        if w_size > 0:
            # 尝试加载权重，如果不在 EPC 中则产生时延
            layer_paging_cost += epc.ensure_loaded(w_id, w_size)
            
        # --- 处理输入 ---
        # 简化模型：将所有输入视为一个数据块。如果依赖层的输出还在 EPC 中，则命中。
        # 这里我们使用简单的命名规则模拟依赖：假设输入张量名为 "{layer}_input_composite"
        # 注意：准确模拟需要构建完整的张量依赖图，这里做近似处理。
        if len(deps) > 0:
            # 尝试复用上一层的输出 (如果还在 LRU 中)
            # 在单流线执行中，通常输入就是刚生成的，大概率在 EPC 中 (Hit)
            # 但如果网络很大，旧的激活可能已被驱逐
            # 我们假设输入总是需要被“确保存在”
            i_id = f"{layer_name}_input_req" 
            layer_paging_cost += epc.ensure_loaded(i_id, i_size)
        else:
            # 无依赖 (如 Embedding 层输入)，必须从外部加载
            i_id = f"{layer_name}_raw_input"
            layer_paging_cost += epc.ensure_loaded(i_id, i_size)
            
        # 3. 内存抖动检查 (Thrashing Check)
        # 如果当前层所需的 *总工作集* (Working Set) 超过了 EPC 物理上限
        # 则无论如何优化，都会在计算过程中触发频繁的分页
        total_working_set = w_size + i_size + o_size + a_size
        thrashing_cost = epc.get_thrashing_cost(total_working_set)
        layer_paging_cost += thrashing_cost
        
        # 4. 分配阶段 (Allocation Phase) - 产生 Tier 3 Page Out 开销
        # 需要为输出 (Output) 分配空间。这可能会挤出旧的张量 (如前几层的权重)
        o_id = f"{layer_name}_out"
        if o_size > 0:
            # 仅分配空间，如果有驱逐会累加到 paging_overhead_accumulated
            epc._allocate_space(o_size, o_id)
            
        # 5. 计算与通信累加
        # Tier 0/1: 飞地内计算时间 (CSV 中的 enclave_time_mean)
        compute = row['enclave_time_mean']
        # Tier 4: 跨飞地通信时间 (CSV 中的 xfer_total_mean_ms)
        comm = row['xfer_total_mean_ms']
        
        # 累加本层时延
        total_compute_time += compute
        total_comm_time += comm
        total_paging_time += layer_paging_cost
        
    # 6. 最终汇总
    # 加上所有因分配空间导致的驱逐 (Eviction) 开销
    total_paging_time += epc.paging_overhead_accumulated
    
    total_latency = total_compute_time + total_comm_time + total_paging_time
    
    return {
        "compute_ms": total_compute_time,
        "comm_ms": total_comm_time,
        "paging_ms": total_paging_time,
        "total_ms": total_latency
    }

# --- 执行入口 ---
if __name__ == "__main__":
    file_path = 'bert_base.csv' # 请确保 CSV 文件在当前目录下
    try:
        results = simulate_latency(file_path, epc_mb=128)
        
        print("-" * 30)
        print(f"STP 仿真结果 (EPC Limit = 128MB)")
        print("-" * 30)
        print(f"Tier 0/1 (计算时延): {results['compute_ms']:.2f} ms")
        print(f"Tier 4   (通信时延): {results['comm_ms']:.2f} ms")
        print(f"Tier 3   (分页时延): {results['paging_ms']:.2f} ms  <-- 性能瓶颈")
        print("-" * 30)
        print(f"端到端总时延: {results['total_ms']:.2f} ms")
        print("-" * 30)
    except FileNotFoundError:
        print(f"错误: 找不到文件 {file_path}")