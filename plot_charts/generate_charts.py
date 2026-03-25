#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
图表生成脚本（纯作图，不运行实验）
============================

从 exp_results/exp2_network_ablation/ 和 exp_results/exp3_server_ablation/ 读取已有 CSV 数据，
生成中文字符图表，输出到 plot_charts/figures/。

使用方法：
    python generate_charts.py

输出：
    - figures/exp2/*.png, figures/exp2/*.pdf  (网络带宽消融图表)
    - figures/exp3/*.png, figures/exp3/*.pdf  (服务器数量消融图表)
    - figures/exp2/combined_network_charts.png   (网络图表网格合并图)
    - figures/exp3/combined_server_charts.png    (服务器图表网格合并图)
"""

import os
import glob
import math
from PIL import Image

import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ============ 中文字体配置 ============
FONT_CN = None
for font_path in [
    '/home/kayne/.local/share/fonts/NotoSansSC.ttf',   # 下载的思源黑体
    '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
    '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
    '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
    '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf',
    '/usr/share/fonts/truetype/arphic/uming.ttc',
    '/usr/share/fonts/truetype/arphic/ukai.ttc',
]:
    if os.path.exists(font_path):
        FONT_CN = fm.FontProperties(fname=font_path)
        print(f"  [字体] 使用: {font_path}")
        break

if FONT_CN is None:
    print("  [警告] 未找到中文字体，图表中文可能显示为方块")

# ============ 路径配置（所有输出放入 plot_charts/figures/） ============
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
EXP2_DATA_DIR = os.path.join(PROJECT_ROOT, 'exp_results/exp2_network_ablation')
EXP3_DATA_DIR = os.path.join(PROJECT_ROOT, 'exp_results/exp3_server_ablation')
EXP2_FIG_DIR  = os.path.join(SCRIPT_DIR, 'figures/exp2')
EXP3_FIG_DIR  = os.path.join(SCRIPT_DIR, 'figures/exp3')

# 选中的模型（与 run_all_experiments.py 保持一致）
SELECTED_MODELS = [
    'resnet50', 'vgg16', 'yolov5', 'InceptionV3',
    'bert_large', 'albert_large', 'vit_large',
]

MODEL_NAME_MAP = {
    'resnet50': 'ResNet-50',
    'vgg16': 'VGG-16',
    'yolov5': 'YOLOv5',
    'InceptionV3': 'InceptionV3',
    'bert_large': 'BERT-large',
    'albert_large': 'ALBERT-large',
    'vit_large': 'ViT-large',
}

ALGO_NAMES_CN = {
    'OCC':   'OCC',
    'DINA':  'DINA',
    'MEDIA': 'MEDIA',
    'Ours':  '本文方法',
}

# ============ 配色与样式 ============
COLORS  = {'OCC': '#E74C3C', 'DINA': '#3498DB', 'MEDIA': '#2ECC71', 'Ours': '#9B59B6'}
MARKERS = {'OCC': 's', 'DINA': '^', 'MEDIA': 'D', 'Ours': 'o'}
LSTYLES = {'OCC': '-', 'DINA': (0, (5, 2)), 'MEDIA': (0, (1, 1)), 'Ours': '-'}
MSIZES = {'OCC': 9, 'DINA': 12, 'MEDIA': 8, 'Ours': 10}
LWIDTHS = {'OCC': 2.5, 'DINA': 3.0, 'MEDIA': 2.0, 'Ours': 2.5}

ALGOS = ['OCC', 'DINA', 'MEDIA', 'Ours']


# ============ 网络带宽消融图表 ============

def create_network_chart(csv_file, model_name, output_dir):
    """创建网络带宽消融折线图（中文标签）。"""
    df = pd.read_csv(csv_file)

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    x = df['Bandwidth(Mbps)']
    for algo in ALGOS:
        ax.plot(x, df[algo],
                label=ALGO_NAMES_CN[algo],
                color=COLORS[algo],
                marker=MARKERS[algo],
                markersize=MSIZES[algo],
                linewidth=LWIDTHS[algo],
                linestyle=LSTYLES[algo],
                markeredgecolor='white',
                markeredgewidth=1.5,
                zorder=3 if algo in ['DINA', 'MEDIA'] else 2)

    ax.set_xlabel('网络带宽 (Mbps)', fontsize=14, fontweight='bold', fontproperties=FONT_CN)
    ax.set_ylabel('推理延迟 (ms)', fontsize=14, fontweight='bold', fontproperties=FONT_CN)
    ax.set_title(f'推理延迟随网络带宽变化 ({model_name})',
                 fontsize=16, fontweight='bold', fontproperties=FONT_CN, pad=15)

    x_min, x_max = x.min(), x.max()
    if x_max <= 20 and (x_max - x_min) < 20:
        ax.set_xscale('linear')
        ax.set_yscale('log')
        tick_step = 1 if x_max <= 10 else 2
        ticks = list(range(int(x_min), int(x_max) + 1, tick_step))
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(int(t)) for t in ticks], fontsize=11)
        ax.set_xlim(x_min - 0.5, x_max + 0.5)
    else:
        ax.set_xscale('log')
        ax.set_yscale('log')
        if len(x) <= 12:
            ax.set_xticks(x.values)
            ax.set_xticklabels([str(int(v)) for v in x], fontsize=11)

    ax.tick_params(axis='y', labelsize=12)
    ax.legend(loc='upper right', fontsize=12, frameon=True,
              fancybox=True, shadow=True, framealpha=0.95,
              prop=FONT_CN)
    ax.grid(True, linestyle='--', alpha=0.7, which='both')
    ax.annotate('越小越好', xy=(0.02, 0.02), xycoords='axes fraction',
                fontsize=10, fontstyle='italic', color='gray',
                fontproperties=FONT_CN)

    plt.tight_layout()

    base_name = os.path.splitext(os.path.basename(csv_file))[0]
    output_png = os.path.join(output_dir, f"{base_name}_chart.png")
    output_pdf = os.path.join(output_dir, f"{base_name}_chart.pdf")

    plt.savefig(output_png, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.savefig(output_pdf, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()

    return output_png, output_pdf


def generate_network_charts():
    """生成所有网络带宽消融图表。"""
    print("\n" + "=" * 60)
    print("生成网络带宽消融图表")
    print("=" * 60)

    os.makedirs(EXP2_FIG_DIR, exist_ok=True)
    print(f"  数据来源: {EXP2_DATA_DIR}")
    print(f"  图表输出: {EXP2_FIG_DIR}")

    for key in SELECTED_MODELS:
        model_name = MODEL_NAME_MAP.get(key, key)
        csv_file = os.path.join(EXP2_DATA_DIR, f'network_{model_name}.csv')
        if os.path.exists(csv_file):
            try:
                png, pdf = create_network_chart(csv_file, model_name, EXP2_FIG_DIR)
                print(f"  [OK] {model_name}: {os.path.basename(png)}, {os.path.basename(pdf)}")
            except Exception as e:
                print(f"  [错误] {model_name}: {e}")
        else:
            print(f"  [跳过] {model_name}: 文件不存在 — {csv_file}")

    print("  网络图表生成完成！")


# ============ 服务器数量消融图表 ============

def create_server_chart(csv_file, model_name, output_dir):
    """创建服务器数量消融折线图（中文标签）。"""
    df = pd.read_csv(csv_file)

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    x = df['Server number']
    for algo in ALGOS:
        ax.plot(x, df[algo],
                label=ALGO_NAMES_CN[algo],
                color=COLORS[algo],
                marker=MARKERS[algo],
                markersize=MSIZES[algo],
                linewidth=LWIDTHS[algo],
                linestyle=LSTYLES[algo],
                markeredgecolor='white',
                markeredgewidth=1.5,
                zorder=3 if algo in ['DINA', 'MEDIA'] else 2)

    ax.set_xlabel('服务器数量', fontsize=14, fontweight='bold', fontproperties=FONT_CN)
    ax.set_ylabel('推理延迟 (ms)', fontsize=14, fontweight='bold', fontproperties=FONT_CN)
    ax.set_title(f'推理延迟随服务器数量变化 ({model_name})',
                 fontsize=16, fontweight='bold', fontproperties=FONT_CN, pad=15)

    ax.set_yscale('log')
    ax.set_xticks(x.values)
    ax.set_xticklabels([str(int(v)) for v in x], fontsize=12)
    ax.tick_params(axis='y', labelsize=12)

    ax.legend(loc='upper right', fontsize=12, frameon=True,
              fancybox=True, shadow=True, framealpha=0.95,
              prop=FONT_CN)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.annotate('越小越好', xy=(0.02, 0.02), xycoords='axes fraction',
                fontsize=10, fontstyle='italic', color='gray',
                fontproperties=FONT_CN)

    plt.tight_layout()

    base_name = os.path.splitext(os.path.basename(csv_file))[0]
    output_png = os.path.join(output_dir, f"{base_name}_chart.png")
    output_pdf = os.path.join(output_dir, f"{base_name}_chart.pdf")

    plt.savefig(output_png, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.savefig(output_pdf, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()

    return output_png, output_pdf


def generate_server_charts():
    """生成所有服务器数量消融图表。"""
    print("\n" + "=" * 60)
    print("生成服务器数量消融图表")
    print("=" * 60)

    os.makedirs(EXP3_FIG_DIR, exist_ok=True)
    print(f"  数据来源: {EXP3_DATA_DIR}")
    print(f"  图表输出: {EXP3_FIG_DIR}")

    for key in SELECTED_MODELS:
        model_name = MODEL_NAME_MAP.get(key, key)
        csv_file = os.path.join(EXP3_DATA_DIR, f'server_hetero_incremental_{model_name}.csv')
        if os.path.exists(csv_file):
            try:
                png, pdf = create_server_chart(csv_file, model_name, EXP3_FIG_DIR)
                print(f"  [OK] {model_name}: {os.path.basename(png)}, {os.path.basename(pdf)}")
            except Exception as e:
                print(f"  [错误] {model_name}: {e}")
        else:
            print(f"  [跳过] {model_name}: 文件不存在 — {csv_file}")

    print("  服务器图表生成完成！")


# ============ 图表网格合并 ============

def combine_charts(image_dir, output_path, cols=2, pattern="*_chart.png"):
    """将多张图表合并为网格图。"""
    image_paths = sorted(glob.glob(os.path.join(image_dir, pattern)))
    if not image_paths:
        print(f"  [警告] 在 {image_dir} 中未找到匹配 {pattern} 的图片")
        return False

    print(f"  找到 {len(image_paths)} 张图片待合并")
    images = [Image.open(f) for f in image_paths]
    w, h = images[0].size

    n = len(image_paths)
    rows = math.ceil(n / cols)
    canvas_w = w * cols
    canvas_h = h * rows
    canvas = Image.new('RGB', (canvas_w, canvas_h), 'white')

    for i, img in enumerate(images):
        r, c = divmod(i, cols)
        if img.size != (w, h):
            img = img.resize((w, h), Image.Resampling.LANCZOS)
        canvas.paste(img, (c * w, r * h))

    canvas.save(output_path)
    print(f"  [OK] 已保存合并图表: {output_path}")
    return True


def generate_combined_charts():
    """生成网格合并图表。"""
    print("\n" + "=" * 60)
    print("生成合并网格图表")
    print("=" * 60)

    # 网络图表合并
    print("\n合并网络带宽消融图表...")
    network_combined = os.path.join(EXP2_FIG_DIR, 'combined_network_charts.png')
    combine_charts(EXP2_FIG_DIR, network_combined, cols=2, pattern="network_*_chart.png")

    # 服务器图表合并
    print("\n合并服务器数量消融图表...")
    server_combined = os.path.join(EXP3_FIG_DIR, 'combined_server_charts.png')
    combine_charts(EXP3_FIG_DIR, server_combined, cols=2, pattern="server_hetero_incremental_*_chart.png")

    print(f"\n[完成] 合并图表:")
    print(f"       网络: {network_combined}")
    print(f"       服务器: {server_combined}")


# ============ 主入口 ============

def main():
    print("=" * 60)
    print("图表生成脚本（仅作图，不运行实验）")
    print("=" * 60)

    os.makedirs(EXP2_FIG_DIR, exist_ok=True)
    os.makedirs(EXP3_FIG_DIR, exist_ok=True)

    generate_network_charts()
    generate_server_charts()
    generate_combined_charts()

    print("\n" + "=" * 60)
    print("全部图表生成完毕！")
    print("=" * 60)
    print(f"\n输出目录:")
    print(f"  网络图表  : {EXP2_FIG_DIR}/")
    print(f"  服务器图表: {EXP3_FIG_DIR}/")


if __name__ == '__main__':
    main()
