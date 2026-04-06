import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.interpolate import RectBivariateSpline

matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'Arial']
matplotlib.rcParams['axes.unicode_minus'] = False

# =========================================================
# 1. 改成你自己的文件路径
# =========================================================
nwp_path = r"../data/BJ/CP_01_05171710_00001/NWP/CAPE/CP_RRA_00001_0517-1500_CAPE.npy"
radar_path = r"../data/BJ/CP_01_05171710_00001/RADAR/CAP20/CP_RADA_00001_0517-1518_SA_CAP020.npy"
label_path = r"../data/BJ/CP_01_05171710_00001/LABEL/RA/CP_Label_RA_00001_0517-1518.npy"

out_dir = "sampling_region_compare"
os.makedirs(out_dir, exist_ok=True)

# =========================================================
# 2. 经纬度范围
# =========================================================
# NWP
NWP_START_LON = 113.97
NWP_START_LAT = 42.30
NWP_END_LON   = 118.98
NWP_END_LAT   = 37.29

# RADAR / LABEL
RL_START_LON = 114.9719
RL_START_LAT = 41.3089
RL_END_LON   = 117.9719
RL_END_LAT   = 38.3089

# =========================================================
# 3. 工具函数
# =========================================================
def center_crop(arr, out_h, out_w):
    h, w = arr.shape
    sy = (h - out_h) // 2
    sx = (w - out_w) // 2
    ey = sy + out_h
    ex = sx + out_w
    return arr[sy:ey, sx:ex], (sy, ey, sx, ex)

def resize_bilinear(src2d, out_h, out_w):
    h, w = src2d.shape

    y = np.linspace(0, h - 1, out_h)
    x = np.linspace(0, w - 1, out_w)

    y0 = np.floor(y).astype(int)
    x0 = np.floor(x).astype(int)
    y1 = np.clip(y0 + 1, 0, h - 1)
    x1 = np.clip(x0 + 1, 0, w - 1)

    wy = y - y0
    wx = x - x0

    Ia = src2d[np.ix_(y0, x0)]
    Ib = src2d[np.ix_(y0, x1)]
    Ic = src2d[np.ix_(y1, x0)]
    Id = src2d[np.ix_(y1, x1)]

    wa = (1 - wy)[:, None] * (1 - wx)[None, :]
    wb = (1 - wy)[:, None] * wx[None, :]
    wc = wy[:, None] * (1 - wx)[None, :]
    wd = wy[:, None] * wx[None, :]

    return (Ia * wa + Ib * wb + Ic * wc + Id * wd).astype(np.float32)

def pixel_size(start_lon, start_lat, end_lon, end_lat, w, h):
    dx = (end_lon - start_lon) / (w - 1)
    dy = (start_lat - end_lat) / (h - 1)
    return dx, dy

def crop_geo_range_from_center_crop(start_lon, start_lat, end_lon, end_lat, h, w, sy, ey, sx, ex):
    dx, dy = pixel_size(start_lon, start_lat, end_lon, end_lat, w, h)

    crop_start_lon = start_lon + sx * dx
    crop_end_lon   = start_lon + (ex - 1) * dx

    crop_start_lat = start_lat - sy * dy
    crop_end_lat   = start_lat - (ey - 1) * dy

    return crop_start_lon, crop_start_lat, crop_end_lon, crop_end_lat

def geo_to_index_range(start_lon, start_lat, end_lon, end_lat, h, w,
                       target_start_lon, target_start_lat, target_end_lon, target_end_lat):
    dx, dy = pixel_size(start_lon, start_lat, end_lon, end_lat, w, h)

    sx = int(round((target_start_lon - start_lon) / dx))
    ex = int(round((target_end_lon   - start_lon) / dx))

    sy = int(round((start_lat - target_start_lat) / dy))
    ey = int(round((start_lat - target_end_lat) / dy))

    sx = max(0, min(w - 1, sx))
    ex = max(0, min(w - 1, ex))
    sy = max(0, min(h - 1, sy))
    ey = max(0, min(h - 1, ey))

    if sx > ex:
        sx, ex = ex, sx
    if sy > ey:
        sy, ey = ey, sy

    return sy, ey + 1, sx, ex + 1

def plot_with_box(ax, arr, title, sx, sy, ex, ey, cmap="jet"):
    im = ax.imshow(arr, origin="upper", cmap=cmap)
    rect = Rectangle((sx, sy), ex - sx, ey - sy, fill=False, edgecolor="white", linewidth=2)
    ax.add_patch(rect)
    ax.set_title(title, fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

def plot_plain(ax, arr, title, cmap="jet"):
    im = ax.imshow(arr, origin="upper", cmap=cmap)
    ax.set_title(title, fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

def plot_diff(ax, diff, title):
    vmax = np.nanmax(np.abs(diff))
    if vmax == 0:
        vmax = 1e-6
    im = ax.imshow(diff, origin="upper", cmap="bwr", vmin=-vmax, vmax=vmax)
    ax.set_title(title, fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# =========================================================
# 4. 读取数据
# =========================================================
nwp = np.load(nwp_path).astype(np.float32)
radar = np.load(radar_path).astype(np.float32)
label = np.load(label_path).astype(np.float32)

print("nwp shape:", nwp.shape)
print("radar shape:", radar.shape)
print("label shape:", label.shape)

# =========================================================
# 5. 先从 RADAR/LABEL 中心裁 128×128
# =========================================================
radar_crop128, (r_sy, r_ey, r_sx, r_ex) = center_crop(radar, 128, 128)
label_crop128, (l_sy, l_ey, l_sx, l_ex) = center_crop(label, 128, 128)

# 由 radar 的中心128区域确定目标地理范围
target_start_lon, target_start_lat, target_end_lon, target_end_lat = crop_geo_range_from_center_crop(
    RL_START_LON, RL_START_LAT, RL_END_LON, RL_END_LAT,
    radar.shape[0], radar.shape[1],
    r_sy, r_ey, r_sx, r_ex
)

print("Target geo range from radar center 128x128:")
print("Lon:", target_start_lon, "~", target_end_lon)
print("Lat:", target_end_lat, "~", target_start_lat)

# =========================================================
# 6. 在 NWP 原图中找出同一地理区域
# =========================================================
n_sy, n_ey, n_sx, n_ex = geo_to_index_range(
    NWP_START_LON, NWP_START_LAT, NWP_END_LON, NWP_END_LAT,
    nwp.shape[0], nwp.shape[1],
    target_start_lon, target_start_lat, target_end_lon, target_end_lat
)

nwp_same_geo = nwp[n_sy:n_ey, n_sx:n_ex]
nwp_same_geo_to_128 = resize_bilinear(nwp_same_geo, 128, 128)
nwp_direct_center128, (nc_sy, nc_ey, nc_sx, nc_ex) = center_crop(nwp, 128, 128)

print("NWP same-geo crop index:", n_sy, n_ey, n_sx, n_ex)
print("NWP same-geo crop shape:", nwp_same_geo.shape)

# =========================================================
# 7. 画老师要的图：原图上框出采样区域
# =========================================================
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# 上排：原图 + 采样区域框
plot_with_box(
    axes[0, 0], nwp,
    f"NWP original + sampled region\nbox=({nwp_same_geo.shape[0]}x{nwp_same_geo.shape[1]})",
    n_sx, n_sy, n_ex, n_ey
)

plot_with_box(
    axes[0, 1], radar,
    "RADAR original + center crop 128x128",
    r_sx, r_sy, r_ex, r_ey
)

plot_with_box(
    axes[0, 2], label,
    "LABEL original + center crop 128x128",
    l_sx, l_sy, l_ex, l_ey
)

# 下排：采样区域 / 重采样结果 / 差值
plot_plain(
    axes[1, 0], nwp_same_geo,
    f"NWP sampled region\nsame geo range, shape={nwp_same_geo.shape}"
)

plot_plain(
    axes[1, 1], nwp_same_geo_to_128,
    "NWP sampled region -> resized to 128x128"
)

diff_nwp = nwp_same_geo_to_128 - nwp_direct_center128
plot_diff(
    axes[1, 2], diff_nwp,
    "Difference:\n(same-geo resized 128) - (direct center crop 128)"
)

plt.tight_layout()
save_path = os.path.join(out_dir, "nwp_radar_label_sampling_region_compare.png")
plt.savefig(save_path, dpi=200, bbox_inches="tight")
plt.show()

print("Saved:", save_path)

# =========================================================
# 8. 单独再画一张 radar 的“原图 -> 裁剪 / 插值”对比
# =========================================================
radar_resize128 = resize_bilinear(radar, 128, 128)
radar_diff = radar_resize128 - radar_crop128

fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))

plot_with_box(
    axes[0], radar,
    "RADAR original + center crop box",
    r_sx, r_sy, r_ex, r_ey
)

plot_plain(
    axes[1], radar_crop128,
    "RADAR center crop 128x128"
)

plot_plain(
    axes[2], radar_resize128,
    "RADAR directly resized 301->128"
)

plot_diff(
    axes[3], radar_diff,
    "Difference:\n(resize128 - crop128)"
)

plt.tight_layout()
save_path2 = os.path.join(out_dir, "radar_crop_vs_resize_compare.png")
plt.savefig(save_path2, dpi=200, bbox_inches="tight")
plt.show()

print("Saved:", save_path2)

# =========================================================
# 9. 单独再画一张 label 的“原图 -> 裁剪 / 插值”对比
# =========================================================
label_resize128 = resize_bilinear(label, 128, 128)
label_diff = label_resize128 - label_crop128

fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))

plot_with_box(
    axes[0], label,
    "LABEL original + center crop box",
    l_sx, l_sy, l_ex, l_ey
)

plot_plain(
    axes[1], label_crop128,
    "LABEL center crop 128x128"
)

plot_plain(
    axes[2], label_resize128,
    "LABEL directly resized 301->128"
)

plot_diff(
    axes[3], label_diff,
    "Difference:\n(resize128 - crop128)"
)

plt.tight_layout()
save_path3 = os.path.join(out_dir, "label_crop_vs_resize_compare.png")
plt.savefig(save_path3, dpi=200, bbox_inches="tight")
plt.show()

print("Saved:", save_path3)