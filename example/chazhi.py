import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
matplotlib.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题
# ===== 1) 改成你的 167x167 npy 路径 =====
PATH_NPY = r"../data/BJ/CP_01_05171710_00001/NWP/CAPE/CP_RRA_00001_0517-1500_CAPE.npy"

OUT_HW = 256  # 目标网格大小


# ===== 2) resize 方法 =====
def resize_nn(src, out_h, out_w):
    """最近邻：索引取整采样（你代码里用的就是这个思路）"""
    h, w = src.shape
    ys = (np.linspace(0, h - 1, out_h)).astype(int)
    xs = (np.linspace(0, w - 1, out_w)).astype(int)
    return src[np.ix_(ys, xs)]


def resize_bilinear(src, out_h, out_w):
    """纯 numpy 的双线性插值"""
    h, w = src.shape
    y = np.linspace(0, h - 1, out_h)
    x = np.linspace(0, w - 1, out_w)

    y0 = np.floor(y).astype(int)
    x0 = np.floor(x).astype(int)
    y1 = np.clip(y0 + 1, 0, h - 1)
    x1 = np.clip(x0 + 1, 0, w - 1)

    wy = y - y0
    wx = x - x0

    Ia = src[np.ix_(y0, x0)]
    Ib = src[np.ix_(y0, x1)]
    Ic = src[np.ix_(y1, x0)]
    Id = src[np.ix_(y1, x1)]

    wa = (1 - wy)[:, None] * (1 - wx)[None, :]
    wb = (1 - wy)[:, None] * (wx)[None, :]
    wc = (wy)[:, None] * (1 - wx)[None, :]
    wd = (wy)[:, None] * (wx)[None, :]

    return Ia * wa + Ib * wb + Ic * wc + Id * wd


def resize_cubic_optional(src, out_h, out_w):
    """
    可选：三次插值/样条（优先用 OpenCV，其次 SciPy）。
    你的环境若没装库，会返回 None，不影响其他图。
    """
    # 方式1：OpenCV
    try:
        import cv2
        # cv2.resize 的 size 是 (W,H)
        return cv2.resize(src, (out_w, out_h), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    except Exception:
        pass

    # 方式2：SciPy
    try:
        from scipy.ndimage import zoom
        zh = out_h / src.shape[0]
        zw = out_w / src.shape[1]
        return zoom(src, (zh, zw), order=3).astype(np.float32)
    except Exception:
        return None


# ===== 3) 读取并做 3 种 resize =====
g167 = np.load(PATH_NPY).astype(np.float32)
assert g167.shape == (167, 167), f"期望(167,167)，但读到的是 {g167.shape}"

g_nn = resize_nn(g167, OUT_HW, OUT_HW)
g_lin = resize_bilinear(g167, OUT_HW, OUT_HW)
g_cub = resize_cubic_optional(g167, OUT_HW, OUT_HW)

# 差值图
d_lin = g_lin - g_nn
d_cub = (g_cub - g_nn) if g_cub is not None else None

# 统一显示范围（让三张图可比）
vmin = float(np.nanmin(g167))
vmax = float(np.nanmax(g167))


# ===== 4) 图1：原图 + 三种 resize =====
ncols = 4 if g_cub is not None else 3
fig, axs = plt.subplots(1, ncols, figsize=(4.8 * ncols, 4.2))

axs[0].imshow(g167, vmin=vmin, vmax=vmax)
axs[0].set_title("Original (167x167)")
axs[0].axis("off")

axs[1].imshow(g_nn, vmin=vmin, vmax=vmax)
axs[1].set_title("NN -> 256x256 (current code)")
axs[1].axis("off")

axs[2].imshow(g_lin, vmin=vmin, vmax=vmax)
axs[2].set_title("Bilinear -> 256x256")
axs[2].axis("off")

if g_cub is not None:
    axs[3].imshow(g_cub, vmin=vmin, vmax=vmax)
    axs[3].set_title("Cubic/Spline -> 256x256")
    axs[3].axis("off")

plt.tight_layout()
plt.show()


# ===== 5) 图2：差值图（看“平滑/过冲”在哪些区域发生）=====
ncols2 = 2 if d_cub is not None else 1
fig, axs = plt.subplots(1, ncols2, figsize=(5.2 * ncols2, 4.2))
axs = [axs] if ncols2 == 1 else axs

axs[0].imshow(d_lin)
axs[0].set_title("Diff: Bilinear - NN")
axs[0].axis("off")

if d_cub is not None:
    axs[1].imshow(d_cub)
    axs[1].set_title("Diff: Cubic - NN")
    axs[1].axis("off")

plt.tight_layout()
plt.show()


# ===== 6) 图3：剖面线对比（看极值是否被削弱/是否过冲）=====
row = OUT_HW // 2
plt.figure(figsize=(10, 4))
plt.plot(g_nn[row], label="最近邻插值")
plt.plot(g_lin[row], label="线性插值")
if g_cub is not None:
    plt.plot(g_cub[row], label="样条插值")
plt.title(f"Profile comparison at row={row}")
plt.legend()
plt.tight_layout()
plt.show()