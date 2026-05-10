#!/usr/bin/env python3
"""
训练日志 CSV 可视化脚本
输入：training.csv（可通过命令行参数指定）
输出：多个 SVG 曲线图
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# 设置 Matplotlib 全局样式
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'lines.linewidth': 1.5,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'svg.fonttype': 'none',  # 保留文本作为字符
    'font.sans-serif': ['SimHei', 'Microsoft YaHei', 'DejaVu Sans'],  # 中文字体优先
    'axes.unicode_minus': False  # 解决负号显示为方块的问题
})

def load_and_preprocess(csv_path):
    """加载 CSV，按 Epoch 分组取最后一行（去重）"""
    df = pd.read_csv(csv_path)
    df = df.sort_values('Epoch')
    # 按 Epoch 分组，保留每组最后一条（解决可能的重复）
    df = df.groupby('Epoch', as_index=False).last()
    return df

def plot_loss_curves(df, output='loss_curve.svg'):
    """绘制 D_Loss 和 G_Loss 曲线（双 y 轴）"""
    fig, ax1 = plt.subplots(figsize=(4, 2.5))

    color_d = '#1f77b4'
    color_g = '#d62728'

    ax1.set_xlabel('训练轮数')
    ax1.set_ylabel('判别器损失', color=color_d)
    ax1.plot(df['Epoch'], df['D_Loss'], color=color_d, label='判别器损失', linewidth=1.2)
    ax1.tick_params(axis='y', labelcolor=color_d)

    ax2 = ax1.twinx()
    ax2.set_ylabel('生成器损失', color=color_g)
    ax2.plot(df['Epoch'], df['G_Loss'], color=color_g, label='生成器损失', linewidth=1.2)
    ax2.tick_params(axis='y', labelcolor=color_g)

    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.title('生成器和判别器损失 vs. 训练轮数')
    fig.tight_layout()
    fig.savefig(output, format='svg')
    plt.close()
    print(f"已保存 {output}")

def plot_val_metrics(df, output='val_metrics.svg'):
    """绘制验证集 PSNR 和 L1 损失曲线（双 y 轴）"""
    fig, ax1 = plt.subplots(figsize=(4, 2.5))

    color_psnr = '#2ca02c'
    color_l1 = '#9467bd'

    ax1.set_xlabel('训练轮数')
    ax1.set_ylabel('PSNR (dB)', color=color_psnr)
    ax1.plot(df['Epoch'], df['Val_PSNR'], color=color_psnr, label='PSNR', linewidth=1.2)
    ax1.tick_params(axis='y', labelcolor=color_psnr)

    ax2 = ax1.twinx()
    ax2.set_ylabel('验证 L1 损失', color=color_l1)
    ax2.plot(df['Epoch'], df['Val_L1'], color=color_l1, label='L1 损失', linewidth=1.2)
    ax2.tick_params(axis='y', labelcolor=color_l1)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right')

    plt.title('验证指标 vs. 训练轮数')
    fig.tight_layout()
    fig.savefig(output, format='svg')
    plt.close()
    print(f"已保存 {output}")

def plot_learning_rate(df, output='learning_rate.svg'):
    """绘制学习率曲线"""
    plt.figure(figsize=(4, 2))
    plt.plot(df['Epoch'], df['LR_G'], color='#ff7f0e', linewidth=1.5)
    plt.xlabel('训练轮数')
    plt.ylabel('学习率')
    plt.title('生成器学习率计划')
    plt.tight_layout()
    plt.savefig(output, format='svg')
    plt.close()
    print(f"已保存 {output}")

def plot_combined_loss_metrics(df, output='combined.svg'):
    """在一个图中同时显示损失和验证指标（四个子图）"""
    fig, axes = plt.subplots(2, 2, figsize=(6, 4))

    # (1) D Loss
    ax = axes[0, 0]
    ax.plot(df['Epoch'], df['D_Loss'], color='#1f77b4')
    ax.set_xlabel('训练轮数')
    ax.set_ylabel('判别器损失')
    ax.set_title('判别器损失')

    # (2) G Loss
    ax = axes[0, 1]
    ax.plot(df['Epoch'], df['G_Loss'], color='#d62728')
    ax.set_xlabel('训练轮数')
    ax.set_ylabel('生成器损失')
    ax.set_title('生成器损失')

    # (3) Validation PSNR
    ax = axes[1, 0]
    ax.plot(df['Epoch'], df['Val_PSNR'], color='#2ca02c')
    ax.set_xlabel('训练轮数')
    ax.set_ylabel('PSNR (dB)')
    ax.set_title('验证集 PSNR')

    # (4) Validation L1
    ax = axes[1, 1]
    ax.plot(df['Epoch'], df['Val_L1'], color='#9467bd')
    ax.set_xlabel('训练轮数')
    ax.set_ylabel('验证 L1 损失')
    ax.set_title('验证集 L1 损失')

    fig.suptitle('训练指标概览', fontsize=16)
    fig.tight_layout()
    fig.savefig(output, format='svg')
    plt.close()
    print(f"已保存 {output}")

def main():
    # 从命令行参数获取 CSV 路径，默认为 'training.csv'
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        csv_path = 'training.csv'

    df = load_and_preprocess(csv_path)
    print(f"已加载 {len(df)} 个轮次数据，来自 {csv_path}")

    # 生成各个曲线图
    plot_loss_curves(df, 'loss_curve.svg')
    plot_val_metrics(df, 'val_metrics.svg')
    plot_learning_rate(df, 'learning_rate.svg')
    plot_combined_loss_metrics(df, 'combined.svg')

if __name__ == '__main__':
    main()
