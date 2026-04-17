import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker  # 新增导入


def plot_rainy_session(test_y_mm, test_pred_mm, window_size=168):
    """
    调整了 y 轴密度并只保留 GE-STDGN 对比
    """
    # 1. 数据展平 (取第0个站点)
    if test_y_mm.ndim == 3:
        y_true = test_y_mm[:, :, 0].flatten()
        y_pred = test_pred_mm[:, :, 0].flatten()
    else:
        y_true = test_y_mm.flatten()
        y_pred = test_pred_mm.flatten()

    # 2. 自动寻找雨量最大的区间
    best_start = 0
    max_rain = 0
    if len(y_true) > window_size:
        for i in range(len(y_true) - window_size):
            current_sum = np.sum(y_true[i: i + window_size])
            if current_sum > max_rain:
                max_rain = current_sum
                best_start = i

    end_idx = best_start + window_size
    plot_true = y_true[best_start:end_idx]
    plot_pred = y_pred[best_start:end_idx]

    # 3. 绘图配置
    fig, ax = plt.subplots(figsize=(10, 5), facecolor='white')
    ax.set_facecolor('#f5f5f5')
    ax.grid(True, color='white', linestyle='-', linewidth=1.0, zorder=0)

    time_axis = np.arange(len(plot_true))

    # 绘制曲线
    ax.plot(time_axis, plot_true, label='real_value', color='#1f77b4', linewidth=1.5, zorder=3)
    ax.plot(time_axis, plot_pred, label='GE-STDGN', color='#2ca02c', linewidth=1.2, alpha=0.8, zorder=3)

    # --- 关键修改：调密 y 轴刻度 ---
    # 获取当前数据的最大值，动态决定间隔
    max_val = max(np.max(plot_true), np.max(plot_pred))

    # 设置主刻度间隔（比如每 1.0mm 一个大刻度）
    # 如果你的雨量很大（比如超过50），可以把 1.0 改成 5.0
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1.0))

    # 设置次刻度间隔（比如每 0.5mm 一个小标记，不显示数字，只显线条）
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.5))

    # 让 y 轴网格线同时显示在主、次刻度上，增加视觉上的“密集感”
    ax.grid(which='both', color='white', linestyle='-', linewidth=0.5)
    # ----------------------------

    ax.set_ylabel('Precipitation (mm)', fontsize=12)
    ax.set_xlabel('Time Steps', fontsize=12)
    ax.legend(loc='upper right', frameon=True, facecolor='white')

    ax.set_ylim(bottom=0, top=max_val * 1.1)  # 稍微给顶部留点空间
    ax.set_xlim(0, len(plot_true) - 1)

    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.title(f"Precipitation: Real vs GE-STDGN (Dense Y-Axis)")
    plt.tight_layout()

    plt.savefig("rain_dense_y.png", dpi=300)
    print(">>> 密集刻度图已保存至: rain_dense_y.png")
    plt.close()