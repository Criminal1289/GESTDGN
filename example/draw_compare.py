import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


# ----------------- 1. 数据提取与统一 (保持原逻辑) -----------------

def load_all_data():
    results = {}

    # (1) STDGN - 每1个epoch记录一次 (Numpy)
    try:
        results['STDGN'] = {
            'loss': np.load('./compare/train_Loss_STDGN.npy'),
            'rmse': np.load('./compare/train_RMSE_all_STDGN.npy'),
            'step': 1
        }
    except FileNotFoundError:
        print("Warning: STDGN files not found.")

    # (2) ConvLSTM - 列表格式 (JSON)
    try:
        with open('./compare/history_convlstm.json', 'r') as f:
            data = json.load(f)
            results['ConvLSTM'] = {
                'loss': [h['train']['loss'] for h in data],
                'rmse': [h['train']['rmse'] for h in data],
                'step': 1
            }
    except FileNotFoundError:
        print("Warning: ConvLSTM file not found.")

    # (3) GRU - 列表格式 (JSON)
    try:
        with open('./compare/history_gru.json', 'r') as f:
            data = json.load(f)
            results['GRU'] = {
                'loss': [h['train']['loss'] for h in data],
                'rmse': [h['train']['rmse'] for h in data],
                'step': 1
            }
    except FileNotFoundError:
        print("Warning: GRU file not found.")

    # (4) STGCN - 字典格式 (Pickle)
    try:
        with open('./compare/history_stgcn.pk', 'rb') as f:
            data = pickle.load(f)
            results['STGCN'] = {
                'loss': data['train_loss'],
                # 使用 val_rmse_rain 作为对比
                'rmse': data['val_rmse_rain'],
                'step': 1
            }
    except FileNotFoundError:
        print("Warning: STGCN file not found.")

    # (5) MR-GNF - 列表格式 (JSON)
    try:
        with open('./compare/history_mrgnf.json','r') as f:
            data = json.load(f)
            results['MR-GNF'] = {
                'loss': [h['train']['loss'] for h in data],
                'rmse': [h['val']['h01_rmse_mm'] for h in data],
                'step': 1
            }
    except FileNotFoundError:
        print("Warning: MR-GNF file not found.")

    return results


# ----------------- 2. 定制“画风”绘图函数 (核心修改) -----------------

def plot_custom_style(title, ylabel, filename, data_dict, y_locator=None):
    # 创建画布
    fig, ax = plt.subplots(figsize=(10, 6), dpi=120)

    # === 核心画风定制 (匹配 image_53836f.png) ===

    # 1. 设置颜色和线条样式
    # 颜色选取匹配示例图片中的红色、深蓝色、绿色和橘色
    colors = {
        'STDGN': '#d73027',  # 红色
        'ConvLSTM': '#4575b4',  # 深蓝色
        'GRU': '#31a354',  # 绿色
        'STGCN': '#fe9929'  # 橘色
    }

    for model, (x, y) in data_dict.items():
        color = colors.get(model, 'black')
        # 如果是 STDGN，使用虚线以区分验证指标，其他模型使用实线
        linestyle = '--' if model == 'STDGN' and 'RMSE' in title else '-'
        ax.plot(x, y, label=model, color=color, linestyle=linestyle, linewidth=2.5, alpha=0.9)

    # 2. 定制坐标轴和刻度 (匹配示例)

    # 隐藏上方和右侧的边框线
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 设置左侧和下方边框线的粗细
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)

    # 刻度线定制：朝内、粗细、长度
    ax.tick_params(axis='both', which='major', direction='in', length=6, width=1.2, labelsize=11)
    ax.tick_params(axis='both', which='minor', direction='in', length=3, width=1.0)

    # 设置网格线 (匹配示例中的淡淡灰色网格)
    ax.grid(True, linestyle='-', color='#f0f0f0', linewidth=0.8, alpha=0.8)

    # 3. 定制标签和标题
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Epochs', fontsize=13, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=13, fontweight='bold')

    # 4. 定制图例 (匹配示例中的方框、透明度、位置)
    leg = ax.legend(loc='upper right', frameon=True, fontsize=11, framealpha=0.7, edgecolor='#d9d9d9')
    leg.get_frame().set_linewidth(1.0)

    # 5. 设置坐标轴刻度间隔 (如果提供了 y_locator)
    if y_locator is not None:
        ax.yaxis.set_major_locator(MultipleLocator(y_locator))

    # 设置 X 轴为 100
    ax.set_xlim(0, 105)

    # 保存并关闭
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"图片已保存为: {filename}")
    plt.close()


# ----------------- 3. 执行绘图 -----------------

# 加载数据
all_results = load_all_data()

if not all_results:
    print("Error: No data loaded. Please check file paths.")
else:
    # 准备 Loss 绘图数据
    loss_plot_data = {}
    for model, data in all_results.items():
        epochs = np.arange(1, len(data['loss']) + 1) * data['step']
        loss_plot_data[model] = (epochs, data['loss'])

    # 准备 RMSE 绘图数据
    rmse_plot_data = {}
    for model, data in all_results.items():
        epochs = np.arange(1, len(data['rmse']) + 1) * data['step']
        rmse_plot_data[model] = (epochs, data['rmse'])

    # 绘制 Loss 图 (y轴刻度间隔设为 0.5)
    plot_custom_style('Loss Comparison', 'Loss', './compare/Loss_Comparison.png', loss_plot_data, y_locator=0.5)

    # 绘制 RMSE 图 (y轴刻度间隔设为 1.0)
    plot_custom_style('RMSE Comparison', 'RMSE', './compare/RMSE_Comparison.png', rmse_plot_data, y_locator=1.0)