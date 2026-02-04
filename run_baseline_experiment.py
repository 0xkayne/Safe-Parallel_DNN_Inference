"""
基线实验脚本
==============
测量所有模型在固定配置（4台服务器，100Mbps带宽）下使用4种算法的端到端推理时延。

使用方法:
    python run_baseline_experiment.py

输出:
    results_baseline.csv - 所有模型的性能数据
"""

import os
import glob
import pandas as pd
from datetime import datetime
from loader import ModelLoader
from common import Server
from alg_dina import DINAAlgorithm
from alg_media import MEDIAAlgorithm
from alg_ours import OursAlgorithm
from alg_occ import OCCAlgorithm

# ==================== 实验配置 ====================
DATASETS_DIR = 'datasets_260120'
OUTPUT_FILE = 'results_baseline.csv'

# 固定参数
N_SERVERS = 4
BANDWIDTH_MBPS = 10

# ==================== 主实验函数 ====================
def run_baseline_experiment():
    """运行基线实验，测试所有模型在固定配置下的性能"""
    
    print("=" * 80)
    print("基线实验 - 固定配置性能测试")
    print("=" * 80)
    print(f"配置: {N_SERVERS} 台服务器, {BANDWIDTH_MBPS} Mbps 带宽")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print()
    
    # 查找所有数据集
    csv_files = glob.glob(os.path.join(DATASETS_DIR, '*.csv'))
    
    if not csv_files:
        print(f"错误: 在 {DATASETS_DIR} 目录中未找到任何CSV文件！")
        return
    
    print(f"发现 {len(csv_files)} 个模型数据集:")
    for i, csv_file in enumerate(csv_files, 1):
        model_name = os.path.basename(csv_file).replace('SafeDnnInferenceExp - ', '').replace('.csv', '')
        print(f"  {i}. {model_name}")
    print()
    
    # 结果存储
    results = []
    
    # 遍历每个模型
    for idx, csv_file in enumerate(csv_files, 1):
        model_name = os.path.basename(csv_file).replace('SafeDnnInferenceExp - ', '').replace('.csv', '')
        
        print(f"[{idx}/{len(csv_files)}] 正在处理模型: {model_name}")
        print("-" * 80)
        
        try:
            # 加载模型
            print(f"  ⏳ 加载模型数据...")
            G, layers_map = ModelLoader.load_model_from_csv(csv_file)
            print(f"  ✓ 模型加载成功 (节点数: {len(G.nodes)}, 边数: {len(G.edges)})")
            
            # 创建服务器实例
            servers = [Server(i, server_type="Xeon_IceLake") for i in range(N_SERVERS)]
            
            # 初始化结果字典
            result = {
                'Model': model_name,
                'Servers': N_SERVERS,
                'Bandwidth_Mbps': BANDWIDTH_MBPS
            }
            
            # ========== 运行 DINA 算法 ==========
            print(f"  ⏳ 运行 DINA 算法...")
            try:
                dina = DINAAlgorithm(G, layers_map, servers, BANDWIDTH_MBPS)
                parts_dina = dina.run()
                dina_res = dina.schedule(parts_dina)
                time_dina = dina_res.latency
                result['DINA_Latency'] = round(time_dina, 2)
                result['DINA_Partitions'] = len(parts_dina)
                print(f"  ✓ DINA: {time_dina:.2f} ms ({len(parts_dina)} 个分区)")
            except Exception as e:
                print(f"  ✗ DINA 失败: {str(e)}")
                result['DINA_Latency'] = None
                result['DINA_Partitions'] = None
            
            # ========== 运行 MEDIA 算法 ==========
            print(f"  ⏳ 运行 MEDIA 算法...")
            try:
                media = MEDIAAlgorithm(G, layers_map, servers, BANDWIDTH_MBPS)
                parts_media = media.run()
                media_res = media.schedule(parts_media)
                time_media = media_res.latency
                result['MEDIA_Latency'] = round(time_media, 2)
                result['MEDIA_Partitions'] = len(parts_media)
                print(f"  ✓ MEDIA: {time_media:.2f} ms ({len(parts_media)} 个分区)")
            except Exception as e:
                print(f"  ✗ MEDIA 失败: {str(e)}")
                result['MEDIA_Latency'] = None
                result['MEDIA_Partitions'] = None
            
            # ========== 运行 Ours 算法 ==========
            print(f"  ⏳ 运行 Ours 算法...")
            try:
                ours = OursAlgorithm(G, layers_map, servers, BANDWIDTH_MBPS)
                parts_ours = ours.run()
                ours_res = ours.schedule(parts_ours)
                time_ours = ours_res.latency
                result['Ours_Latency'] = round(time_ours, 2)
                result['Ours_Partitions'] = len(parts_ours)
                print(f"  ✓ Ours: {time_ours:.2f} ms ({len(parts_ours)} 个分区)")
            except Exception as e:
                print(f"  ✗ Ours 失败: {str(e)}")
                result['Ours_Latency'] = None
                result['Ours_Partitions'] = None
            
            # ========== 运行 OCC 算法 ==========
            print(f"  ⏳ 运行 OCC 算法...")
            try:
                occ = OCCAlgorithm(G, layers_map, servers, BANDWIDTH_MBPS)
                parts_occ = occ.run()
                occ_res = occ.schedule(parts_occ)
                time_occ = occ_res.latency
                result['OCC_Latency'] = round(time_occ, 2)
                result['OCC_Partitions'] = len(parts_occ)
                print(f"  ✓ OCC: {time_occ:.2f} ms ({len(parts_occ)} 个分区)")
            except Exception as e:
                print(f"  ✗ OCC 失败: {str(e)}")
                result['OCC_Latency'] = None
                result['OCC_Partitions'] = None
            
            # 保存结果
            results.append(result)
            print(f"  ✓ {model_name} 完成")
            print()
            
        except Exception as e:
            print(f"  ✗ 模型 {model_name} 处理失败: {str(e)}")
            print()
            continue
    
    # ==================== 保存结果 ====================
    if not results:
        print("错误: 没有成功的实验结果！")
        return
    
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_FILE, index=False)
    
    print("=" * 80)
    print(f"实验完成！结果已保存到: {OUTPUT_FILE}")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print()
    
    # ==================== 结果分析 ====================
    analyze_results(df)


def analyze_results(df):
    """分析并打印实验结果摘要"""
    
    print("\n" + "=" * 80)
    print("实验结果摘要")
    print("=" * 80)
    print()
    
    # 基本统计
    print("📊 各算法平均延迟 (ms):")
    print("-" * 80)
    for alg in ['DINA', 'MEDIA', 'Ours', 'OCC']:
        col_name = f'{alg}_Latency'
        if col_name in df.columns:
            avg_latency = df[col_name].mean()
            print(f"  {alg:8s}: {avg_latency:8.2f} ms")
    print()
    
    # 找出最佳算法
    print("🏆 各模型最佳算法:")
    print("-" * 80)
    latency_cols = [col for col in df.columns if col.endswith('_Latency')]
    
    for _, row in df.iterrows():
        model = row['Model']
        latencies = {col.replace('_Latency', ''): row[col] for col in latency_cols if pd.notna(row[col])}
        
        if latencies:
            best_alg = min(latencies, key=latencies.get)
            best_time = latencies[best_alg]
            print(f"  {model:20s}: {best_alg:8s} ({best_time:.2f} ms)")
    print()
    
    # Ours vs DINA 性能提升
    if 'Ours_Latency' in df.columns and 'DINA_Latency' in df.columns:
        print("📈 Ours 相比 DINA 的性能提升:")
        print("-" * 80)
        for _, row in df.iterrows():
            if pd.notna(row['Ours_Latency']) and pd.notna(row['DINA_Latency']):
                model = row['Model']
                improvement = ((row['DINA_Latency'] - row['Ours_Latency']) / row['DINA_Latency']) * 100
                symbol = "↓" if improvement > 0 else "↑"
                print(f"  {model:20s}: {improvement:+6.2f}% {symbol}")
        print()
    
    # 分区数量统计
    print("📦 各算法平均分区数量:")
    print("-" * 80)
    for alg in ['DINA', 'MEDIA', 'Ours', 'OCC']:
        col_name = f'{alg}_Partitions'
        if col_name in df.columns:
            avg_parts = df[col_name].mean()
            print(f"  {alg:8s}: {avg_parts:6.1f} 个")
    print()


if __name__ == "__main__":
    run_baseline_experiment()
