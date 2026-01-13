"""
基线结果可视化脚本
==================
基于 results_baseline.csv 生成专业的对比图表，展示4种算法在7个模型上的性能表现。

生成图表：
1. baseline_latency_comparison.png/pdf - 端到端推理时延对比
2. baseline_partitions_comparison.png/pdf - 分区数量对比

使用方法:
    python plot_baseline_results.py
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# ==================== 配置参数 ====================

# 数据文件
CSV_FILE = 'results_baseline.csv'

# 算法列表（按图表中的顺序）
ALGORITHMS = ['DINA', 'MEDIA', 'Ours', 'OCC']

# 配色方案 - 参考示例图表
COLORS = {
    'DINA': '#4472C4',    # 蓝色
    'MEDIA': '#C55A5A',   # 红色
    'Ours': '#ED7D31',    # 橙色
    'OCC': '#70AD47'      # 绿色
}

# 子图布局
N_ROWS = 2
N_COLS = 4

# 图表尺寸
FIG_WIDTH = 16
FIG_HEIGHT = 8
DPI = 150

# ==================== 主绘图函数 ====================

def plot_latency_comparison(df):
    """绘制端到端推理时延对比图"""
    
    print("📊 生成延迟对比图...")
    
    # 创建图表
    fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=DPI)
    axes = axes.flatten()  # 展平为一维数组
    
    # 获取所有模型
    models = df['Model'].tolist()
    
    # 遍历每个模型，创建子图
    for idx, model in enumerate(models):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        row_data = df[df['Model'] == model].iloc[0]
        
        # 提取每个算法的延迟数据
        latencies = [row_data[f'{alg}_Latency'] for alg in ALGORITHMS]
        
        # 创建柱状图
        x_pos = np.arange(len(ALGORITHMS))
        bars = ax.bar(x_pos, latencies, 
                     color=[COLORS[alg] for alg in ALGORITHMS],
                     width=0.6,
                     edgecolor='black',
                     linewidth=0.8)
        
        # 设置子图标题
        label = chr(97 + idx)  # a, b, c, ...
        ax.set_title(f'({label}) {model}', fontsize=11, fontweight='bold')
        
        # 设置X轴
        ax.set_xticks(x_pos)
        ax.set_xticklabels(ALGORITHMS, fontsize=9, rotation=0)
        
        # 设置Y轴
        ax.set_ylabel('Inference time', fontsize=9)
        ax.tick_params(axis='y', labelsize=8)
        
        # 添加网格
        ax.grid(True, axis='y', linestyle='--', alpha=0.3, zorder=0)
        ax.set_axisbelow(True)
        
        # 设置Y轴范围（从0开始，留出一些顶部空间）
        y_max = max(latencies) * 1.15
        ax.set_ylim(0, y_max)
    
    # 隐藏多余的子图
    for idx in range(len(models), len(axes)):
        axes[idx].axis('off')
    
    # 调整布局
    plt.tight_layout(pad=2.0)
    
    # 保存图表
    output_png = 'baseline_latency_comparison.png'
    output_pdf = 'baseline_latency_comparison.pdf'
    
    plt.savefig(output_png, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.savefig(output_pdf, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    plt.close()
    
    print(f"  ✓ 保存: {output_png}")
    print(f"  ✓ 保存: {output_pdf}")
    
    return output_png, output_pdf


def plot_partition_comparison(df):
    """绘制分区数量对比图"""
    
    print("📦 生成分区数量对比图...")
    
    # 创建图表
    fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=DPI)
    axes = axes.flatten()
    
    # 获取所有模型
    models = df['Model'].tolist()
    
    # 遍历每个模型
    for idx, model in enumerate(models):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        row_data = df[df['Model'] == model].iloc[0]
        
        # 提取每个算法的分区数量
        partitions = [row_data[f'{alg}_Partitions'] for alg in ALGORITHMS]
        
        # 创建柱状图
        x_pos = np.arange(len(ALGORITHMS))
        bars = ax.bar(x_pos, partitions,
                     color=[COLORS[alg] for alg in ALGORITHMS],
                     width=0.6,
                     edgecolor='black',
                     linewidth=0.8)
        
        # 设置子图标题
        label = chr(97 + idx)  # a, b, c, ...
        ax.set_title(f'({label}) {model}', fontsize=11, fontweight='bold')
        
        # 设置X轴
        ax.set_xticks(x_pos)
        ax.set_xticklabels(ALGORITHMS, fontsize=9, rotation=0)
        
        # 设置Y轴
        ax.set_ylabel('Number of Partitions', fontsize=9)
        ax.tick_params(axis='y', labelsize=8)
        
        # 添加网格
        ax.grid(True, axis='y', linestyle='--', alpha=0.3, zorder=0)
        ax.set_axisbelow(True)
        
        # 设置Y轴范围（从0开始）
        y_max = max(partitions) * 1.15
        ax.set_ylim(0, y_max)
    
    # 隐藏多余的子图
    for idx in range(len(models), len(axes)):
        axes[idx].axis('off')
    
    # 调整布局
    plt.tight_layout(pad=2.0)
    
    # 保存图表
    output_png = 'baseline_partitions_comparison.png'
    output_pdf = 'baseline_partitions_comparison.pdf'
    
    plt.savefig(output_png, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(output_pdf, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    
    plt.close()
    
    print(f"  ✓ 保存: {output_png}")
    print(f"  ✓ 保存: {output_pdf}")
    
    return output_png, output_pdf


def print_data_summary(df):
    """打印数据摘要"""
    
    print("\n" + "=" * 80)
    print("数据摘要")
    print("=" * 80)
    print(f"模型数量: {len(df)}")
    print(f"模型列表: {', '.join(df['Model'].tolist())}")
    print()
    
    # 延迟统计
    print("平均延迟 (ms):")
    for alg in ALGORITHMS:
        col = f'{alg}_Latency'
        avg = df[col].mean()
        print(f"  {alg:8s}: {avg:8.2f} ms")
    print()
    
    # 分区数量统计
    print("平均分区数量:")
    for alg in ALGORITHMS:
        col = f'{alg}_Partitions'
        avg = df[col].mean()
        print(f"  {alg:8s}: {avg:6.1f} 个")
    print()


# ==================== 主程序 ====================

def main():
    print("=" * 80)
    print("基线结果可视化")
    print("=" * 80)
    print()
    
    # 检查文件是否存在
    if not os.path.exists(CSV_FILE):
        print(f"❌ 错误: 未找到数据文件 {CSV_FILE}")
        return
    
    # 读取数据
    print(f"📂 读取数据文件: {CSV_FILE}")
    df = pd.read_csv(CSV_FILE)
    print(f"  ✓ 加载成功 ({len(df)} 行数据)")
    print()
    
    # 打印数据摘要
    print_data_summary(df)
    
    # 生成图表
    print("=" * 80)
    print("生成图表")
    print("=" * 80)
    print()
    
    # 延迟对比图
    plot_latency_comparison(df)
    print()
    
    # 分区数量对比图
    plot_partition_comparison(df)
    print()
    
    print("=" * 80)
    print("✅ 所有图表生成完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()
