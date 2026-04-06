import os
import glob
import re
import numpy as np
from datetime import datetime, timedelta
from make_knn_adj import build_knn_adj
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun", "Arial"]
matplotlib.rcParams["axes.unicode_minus"] = False

cp_root = r"../gestdgn/data/BJ"

# =========================================================
# 空间对齐方案
# radar/label: 301x301 -> 中心裁128x128 -> 取32x32
# nwp: 167x167 -> 按同地理范围裁出约43x43 -> 取32x32
# =========================================================
nwp_vars = [
    "CAPE", "DVG200", "DVG850", "DVG925", "HT0", "HT10", "HTw0", "KI",
    "LCL", "LFC", "LI300", "LI500", "muLCL", "PE", "PWAT", "Q1000", "Q700",
    "Q850", "RH1000", "RH500", "RH700", "TdSfc850", "TTdMax74", "TTdMean74",
    "WS500", "WS700", "WS925"
]

radar_vars = [
    "CR", "HBR", "ET", "VIL",
    "CAP20", "CAP30", "CAP40", "CAP50", "CAP60", "CAP70"
]

label_var = "RA"
radar_root_name = "RADAR"

# 原始网格尺寸
NWP_SRC_HW = 167
RL_SRC_HW = 301

# radar / label 中心裁剪目标尺寸
RL_CROP_HW = 128

# 最终节点数量：32x32
NODE_HW = 32
N_NODES = NODE_HW * NODE_HW

# Radar窗口
RADAR_INTERVAL = 6
WINDOW_BEFORE = 2
WINDOW_AFTER = 0

# RA未来累计
RA_INTERVAL = 6
RA_HOUR = 60

# 若遇到意外尺寸，是否允许fallback插值
ALLOW_RESIZE_FALLBACK = False

# 输出目录
OUT_PATCH_DIR = "../gestdgn/data/patch_spatial_align"
OUT_META_DIR = "../gestdgn/data"
VIS_OUT_DIR = "../gestdgn/data/patch_spatial_align/compare"

# 可视化开关
ENABLE_VIS = True
VIS_MAX_FILES = 1

# =========================================================
# 经纬度范围
# =========================================================
# NWP
NWP_START_LON = 113.97
NWP_START_LAT = 42.30
NWP_END_LON = 118.98
NWP_END_LAT = 37.29

# RADAR / LABEL
RL_START_LON = 114.9719
RL_START_LAT = 41.3089
RL_END_LON = 117.9719
RL_END_LAT = 38.3089


# =========================================================
# 时间工具
# =========================================================
def time_key(fname):
    b = os.path.basename(fname)
    m = re.search(r"_(\d{4}-\d{4})", b)
    return m.group(1) if m else None


def parse_datetime_from_key(tk):
    mmdd, hhmm = tk.split("-")
    month = int(mmdd[:2])
    day = int(mmdd[2:])
    hour = int(hhmm[:2])
    minute = int(hhmm[2:])
    return datetime(2000, month, day, hour, minute)


def hour_key(fname):
    tk = time_key(fname)
    if not tk:
        return None
    mmdd, hhmm = tk.split("-")
    hh = hhmm[:2]
    return f"{mmdd}-{hh}00"


# =========================================================
# 基础工具
# =========================================================
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


def center_crop_info(h, w, out_h, out_w):
    sy = (h - out_h) // 2
    sx = (w - out_w) // 2
    ey = sy + out_h
    ex = sx + out_w
    return sy, ey, sx, ex


def center_crop_grid(src2d, out_h, out_w):
    sy, ey, sx, ex = center_crop_info(src2d.shape[0], src2d.shape[1], out_h, out_w)
    return src2d[sy:ey, sx:ex].astype(np.float32)


def pixel_size(start_lon, start_lat, end_lon, end_lat, w, h):
    dx = (end_lon - start_lon) / (w - 1)
    dy = (start_lat - end_lat) / (h - 1)
    return dx, dy


def crop_geo_range_from_center_crop(start_lon, start_lat, end_lon, end_lat, h, w, sy, ey, sx, ex):
    dx, dy = pixel_size(start_lon, start_lat, end_lon, end_lat, w, h)

    crop_start_lon = start_lon + sx * dx
    crop_end_lon = start_lon + (ex - 1) * dx

    crop_start_lat = start_lat - sy * dy
    crop_end_lat = start_lat - (ey - 1) * dy

    return crop_start_lon, crop_start_lat, crop_end_lon, crop_end_lat


def geo_to_index_range(start_lon, start_lat, end_lon, end_lat, h, w,
                       target_start_lon, target_start_lat, target_end_lon, target_end_lat):
    dx, dy = pixel_size(start_lon, start_lat, end_lon, end_lat, w, h)

    sx = int(round((target_start_lon - start_lon) / dx))
    ex = int(round((target_end_lon - start_lon) / dx))

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


def build_node_idx_for_shape(h, w, out_hw=32):
    ys = np.linspace(0, h - 1, out_hw, dtype=int)
    xs = np.linspace(0, w - 1, out_hw, dtype=int)
    return np.array([(i, j) for i in ys for j in xs], dtype=int)


def sample_at_nodes(grid, node_idx):
    return grid[node_idx[:, 0], node_idx[:, 1]].astype(np.float32)


def preprocess_ra_grid(g):
    g = g.astype(np.float32)
    g[g < 0] = 0
    g[np.isnan(g)] = 0
    return g


def preprocess_radar_grid(g, clip_min=None, clip_max=None):
    g = g.astype(np.float32)
    g[np.isnan(g)] = 0
    if clip_min is not None:
        g[g < clip_min] = clip_min
    if clip_max is not None:
        g[g > clip_max] = clip_max
    return g


# =========================================================
# 由 radar/label 中心128区域定义统一目标地理范围
# =========================================================
RL_CROP_SY, RL_CROP_EY, RL_CROP_SX, RL_CROP_EX = center_crop_info(
    RL_SRC_HW, RL_SRC_HW, RL_CROP_HW, RL_CROP_HW
)

TARGET_START_LON, TARGET_START_LAT, TARGET_END_LON, TARGET_END_LAT = crop_geo_range_from_center_crop(
    RL_START_LON, RL_START_LAT, RL_END_LON, RL_END_LAT,
    RL_SRC_HW, RL_SRC_HW,
    RL_CROP_SY, RL_CROP_EY, RL_CROP_SX, RL_CROP_EX
)

NWP_CROP_SY, NWP_CROP_EY, NWP_CROP_SX, NWP_CROP_EX = geo_to_index_range(
    NWP_START_LON, NWP_START_LAT, NWP_END_LON, NWP_END_LAT,
    NWP_SRC_HW, NWP_SRC_HW,
    TARGET_START_LON, TARGET_START_LAT, TARGET_END_LON, TARGET_END_LAT
)

NWP_CROP_H = NWP_CROP_EY - NWP_CROP_SY
NWP_CROP_W = NWP_CROP_EX - NWP_CROP_SX

print("===== Spatial alignment setting =====")
print("Radar/Label center crop shape:", (RL_CROP_HW, RL_CROP_HW))
print("Target geo range:")
print("Lon:", TARGET_START_LON, "~", TARGET_END_LON)
print("Lat:", TARGET_END_LAT, "~", TARGET_START_LAT)
print("NWP aligned crop index:", NWP_CROP_SY, NWP_CROP_EY, NWP_CROP_SX, NWP_CROP_EX)
print("NWP aligned crop shape:", (NWP_CROP_H, NWP_CROP_W))

# 节点索引
node_idx_rl = build_node_idx_for_shape(RL_CROP_HW, RL_CROP_HW, out_hw=NODE_HW)
node_idx_nwp = build_node_idx_for_shape(NWP_CROP_H, NWP_CROP_W, out_hw=NODE_HW)

# 图结构统一用32x32规则坐标
coords_32 = np.array([(i, j) for i in range(NODE_HW) for j in range(NODE_HW)], dtype=np.float32)


# =========================================================
# 加载函数
# =========================================================
def load_radar_or_label_grid(path):
    g = np.load(path).astype(np.float32)

    if g.shape == (RL_CROP_HW, RL_CROP_HW):
        return g

    if g.shape == (RL_SRC_HW, RL_SRC_HW):
        return center_crop_grid(g, RL_CROP_HW, RL_CROP_HW)

    if ALLOW_RESIZE_FALLBACK:
        print(f"[WARN][RL] fallback resize for {path}: {g.shape} -> {(RL_CROP_HW, RL_CROP_HW)}")
        return resize_bilinear(g, RL_CROP_HW, RL_CROP_HW)

    raise ValueError(
        f"Unexpected radar/label shape {g.shape} in file: {path}. "
        f"Expected {(RL_SRC_HW, RL_SRC_HW)} or {(RL_CROP_HW, RL_CROP_HW)}."
    )


def load_nwp_aligned_grid(path):
    g = np.load(path).astype(np.float32)

    if g.shape == (NWP_SRC_HW, NWP_SRC_HW):
        return g[NWP_CROP_SY:NWP_CROP_EY, NWP_CROP_SX:NWP_CROP_EX].astype(np.float32)

    if g.shape == (NWP_CROP_H, NWP_CROP_W):
        return g.astype(np.float32)

    if ALLOW_RESIZE_FALLBACK:
        print(f"[WARN][NWP] fallback resize for {path}: {g.shape} -> {(NWP_CROP_H, NWP_CROP_W)}")
        return resize_bilinear(g, NWP_CROP_H, NWP_CROP_W)

    raise ValueError(
        f"Unexpected NWP shape {g.shape} in file: {path}. "
        f"Expected {(NWP_SRC_HW, NWP_SRC_HW)} or {(NWP_CROP_H, NWP_CROP_W)}."
    )


# =========================================================
# Radar窗口
# =========================================================
def select_radar_files(files, target_hour_key):
    target_dt = parse_datetime_from_key(target_hour_key)

    start_dt = target_dt - timedelta(minutes=WINDOW_BEFORE * RADAR_INTERVAL)
    end_dt = target_dt + timedelta(minutes=WINDOW_AFTER * RADAR_INTERVAL)

    selected = []
    for f in files:
        tk = time_key(f)
        if not tk:
            continue
        dt = parse_datetime_from_key(tk)
        if start_dt <= dt <= end_dt:
            selected.append(f)

    return sorted(selected)


def aggregate_radar_grids(files, how="max"):
    if len(files) == 0:
        return None

    grids = [load_radar_or_label_grid(f) for f in files]
    stack = np.stack(grids, axis=0)

    if how == "mean":
        return stack.mean(axis=0).astype(np.float32)
    return stack.max(axis=0).astype(np.float32)


# =========================================================
# RA未来1小时累计
# =========================================================
def select_ra_future(files, target_hour_key):
    target_dt = parse_datetime_from_key(target_hour_key)
    end_dt = target_dt + timedelta(minutes=RA_HOUR)

    selected = []
    for f in files:
        tk = time_key(f)
        if not tk:
            continue
        dt = parse_datetime_from_key(tk)

        # (当前时刻, 当前时刻+60分钟]
        if target_dt < dt <= end_dt:
            selected.append(f)

    return sorted(selected)


def aggregate_ra_sum(files):
    if len(files) == 0:
        return None

    grids = []
    for f in files:
        g = preprocess_ra_grid(load_radar_or_label_grid(f))
        grids.append(g)

    stack = np.stack(grids, axis=0)
    return stack.sum(axis=0).astype(np.float32)


# =========================================================
# 可视化
# =========================================================
def plot_img(ax, arr, title, cmap="jet", vmin=None, vmax=None):
    im = ax.imshow(arr, origin="upper", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def plot_img_with_box(ax, arr, title, sx, sy, ex, ey, cmap="jet", vmin=None, vmax=None):
    im = ax.imshow(arr, origin="upper", cmap=cmap, vmin=vmin, vmax=vmax)
    rect = Rectangle((sx, sy), ex - sx, ey - sy, fill=False, edgecolor="white", linewidth=2)
    ax.add_patch(rect)
    ax.set_title(title, fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def plot_img_with_points(ax, arr, node_idx, title, cmap="jet", vmin=None, vmax=None):
    im = ax.imshow(arr, origin="upper", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.scatter(node_idx[:, 1], node_idx[:, 0], s=6, c="black")
    ax.set_title(title, fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def plot_diff(ax, diff, title):
    vmax = np.nanmax(np.abs(diff))
    if vmax == 0:
        vmax = 1e-6
    im = ax.imshow(diff, origin="upper", cmap="bwr", vmin=-vmax, vmax=vmax)
    ax.set_title(title, fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def node_values_to_32x32(vals):
    return vals.reshape(NODE_HW, NODE_HW)


def visualize_one_case(nwp_path, radar_path, label_path, out_dir, prefix):
    os.makedirs(out_dir, exist_ok=True)

    nwp = np.load(nwp_path).astype(np.float32)
    radar = np.load(radar_path).astype(np.float32)
    label = np.load(label_path).astype(np.float32)

    # NWP
    nwp_crop = nwp[NWP_CROP_SY:NWP_CROP_EY, NWP_CROP_SX:NWP_CROP_EX]
    nwp_nodes = sample_at_nodes(nwp_crop, node_idx_nwp)
    nwp_32 = node_values_to_32x32(nwp_nodes)
    nwp_back = resize_bilinear(nwp_32, nwp_crop.shape[0], nwp_crop.shape[1])
    nwp_diff = nwp_back - nwp_crop

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    vmin = np.nanmin(nwp_crop)
    vmax = np.nanmax(nwp_crop)

    plot_img_with_box(axes[0, 0], nwp, "NWP original + aligned crop box",
                      NWP_CROP_SX, NWP_CROP_SY, NWP_CROP_EX, NWP_CROP_EY)
    plot_img(axes[0, 1], nwp_crop, f"NWP aligned crop\nshape={nwp_crop.shape}", vmin=vmin, vmax=vmax)
    plot_img_with_points(axes[0, 2], nwp_crop, node_idx_nwp,
                         "NWP crop + 32x32 sampled nodes", vmin=vmin, vmax=vmax)

    plot_img(axes[1, 0], nwp_32, "NWP sampled node values (32x32)")
    plot_img(axes[1, 1], nwp_back, f"NWP 32x32 -> back to {nwp_crop.shape[0]}x{nwp_crop.shape[1]}",
             vmin=vmin, vmax=vmax)
    plot_diff(axes[1, 2], nwp_diff, "Difference:\n(back-projected 32x32) - original crop")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_nwp.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # RADAR
    radar_crop = center_crop_grid(radar, RL_CROP_HW, RL_CROP_HW)
    radar_nodes = sample_at_nodes(radar_crop, node_idx_rl)
    radar_32 = node_values_to_32x32(radar_nodes)
    radar_back = resize_bilinear(radar_32, RL_CROP_HW, RL_CROP_HW)
    radar_diff = radar_back - radar_crop

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    vmin = np.nanmin(radar_crop)
    vmax = np.nanmax(radar_crop)

    plot_img_with_box(axes[0, 0], radar, "RADAR original + center 128x128 box",
                      RL_CROP_SX, RL_CROP_SY, RL_CROP_EX, RL_CROP_EY)
    plot_img(axes[0, 1], radar_crop, "RADAR center crop 128x128", vmin=vmin, vmax=vmax)
    plot_img_with_points(axes[0, 2], radar_crop, node_idx_rl,
                         "RADAR crop + 32x32 sampled nodes", vmin=vmin, vmax=vmax)

    plot_img(axes[1, 0], radar_32, "RADAR sampled node values (32x32)")
    plot_img(axes[1, 1], radar_back, "RADAR 32x32 -> back to 128x128", vmin=vmin, vmax=vmax)
    plot_diff(axes[1, 2], radar_diff, "Difference:\n(back-projected 32x32) - original crop")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_radar.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # LABEL
    label_crop = center_crop_grid(label, RL_CROP_HW, RL_CROP_HW)
    label_nodes = sample_at_nodes(label_crop, node_idx_rl)
    label_32 = node_values_to_32x32(label_nodes)
    label_back = resize_bilinear(label_32, RL_CROP_HW, RL_CROP_HW)
    label_diff = label_back - label_crop

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    vmin = np.nanmin(label_crop)
    vmax = np.nanmax(label_crop)

    plot_img_with_box(axes[0, 0], label, "LABEL original + center 128x128 box",
                      RL_CROP_SX, RL_CROP_SY, RL_CROP_EX, RL_CROP_EY)
    plot_img(axes[0, 1], label_crop, "LABEL center crop 128x128", vmin=vmin, vmax=vmax)
    plot_img_with_points(axes[0, 2], label_crop, node_idx_rl,
                         "LABEL crop + 32x32 sampled nodes", vmin=vmin, vmax=vmax)

    plot_img(axes[1, 0], label_32, "LABEL sampled node values (32x32)")
    plot_img(axes[1, 1], label_back, "LABEL 32x32 -> back to 128x128", vmin=vmin, vmax=vmax)
    plot_diff(axes[1, 2], label_diff, "Difference:\n(back-projected 32x32) - original crop")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_label.png"), dpi=200, bbox_inches="tight")
    plt.close()

    print(f"[VIS OK] {prefix}")


# =========================================================
# 生成图结构文件
# =========================================================
os.makedirs(OUT_META_DIR, exist_ok=True)
np.save(os.path.join(OUT_META_DIR, "coords.npy"), coords_32)

A = build_knn_adj(coords_32, k=8, mode="binary")
np.save(os.path.join(OUT_META_DIR, "adj_knn.npy"), A)

print("coords saved:", os.path.join(OUT_META_DIR, "coords.npy"))
print("adj saved   :", os.path.join(OUT_META_DIR, "adj_knn.npy"))
print("node_idx_rl shape :", node_idx_rl.shape)
print("node_idx_nwp shape:", node_idx_nwp.shape)

# =========================================================
# patch生成
# =========================================================
all_patches = []

cp_folders = sorted([p for p in glob.glob(os.path.join(cp_root, "*")) if os.path.isdir(p)])

vis_count = 0

for cp in cp_folders:
    # =====================
    # NWP
    # =====================
    nwp_map = {}
    for v in nwp_vars:
        files = glob.glob(os.path.join(cp, "NWP", v, "*.npy"))

        d = {}
        for f in files:
            hk = hour_key(f)
            if hk and hk not in d:
                d[hk] = f
        nwp_map[v] = d

    # =====================
    # Radar
    # =====================
    radar_root = os.path.join(cp, radar_root_name)
    radar_files = {}
    for rv in radar_vars:
        radar_files[rv] = glob.glob(os.path.join(radar_root, rv, "*.npy"))

    # =====================
    # RA
    # =====================
    ra_files = glob.glob(os.path.join(cp, "LABEL", label_var, "*.npy"))

    # =====================
    # 时间对齐
    # =====================
    common = None
    for v in nwp_vars:
        s = set(nwp_map[v].keys())
        common = s if common is None else (common & s)

    common = sorted(common)

    if not common:
        print("[SKIP] no common hourly keys from NWP in", cp)
        continue

    T = len(common)
    F = len(nwp_vars) + len(radar_vars) + 1

    patch = np.zeros((T, N_NODES, F), dtype=np.float32)

    nwp_offset = 0
    radar_offset = len(nwp_vars)
    label_offset = len(nwp_vars) + len(radar_vars)

    for ti, hk in enumerate(common):
        # -----------------
        # NWP：先按同地理范围裁出约43x43，再取32x32
        # -----------------
        for fi, v in enumerate(nwp_vars):
            if hk in nwp_map[v]:
                g = load_nwp_aligned_grid(nwp_map[v][hk])
                patch[ti, :, nwp_offset + fi] = sample_at_nodes(g, node_idx_nwp)

        # -----------------
        # RADAR：301 -> 中心128 -> 取32x32
        # -----------------
        for rfi, rv in enumerate(radar_vars):
            files = select_radar_files(radar_files[rv], hk)
            g = aggregate_radar_grids(files)

            if g is None:
                continue

            if rv in ["CR", "HBR", "CAP20", "CAP30", "CAP40", "CAP50", "CAP60", "CAP70"]:
                g = preprocess_radar_grid(g, 0, 600) / 600.0
            elif rv == "ET":
                g = preprocess_radar_grid(g, 0, 300) / 300.0
            elif rv == "VIL":
                g = preprocess_radar_grid(g, 0, 1000) / 1000.0
            else:
                g = preprocess_radar_grid(g)

            patch[ti, :, radar_offset + rfi] = sample_at_nodes(g, node_idx_rl)

        # -----------------
        # LABEL：301 -> 中心128 -> 取32x32
        # -----------------
        future_files = select_ra_future(ra_files, hk)
        g = aggregate_ra_sum(future_files)

        if g is not None:
            patch[ti, :, label_offset] = sample_at_nodes(g, node_idx_rl)

    all_patches.append(patch)
    print("[OK]", os.path.basename(cp), "->", patch.shape)

    # 可视化前几个样本
    if ENABLE_VIS and vis_count < VIS_MAX_FILES:
        try:
            hk0 = common[0]
            nwp_file = None
            for v in nwp_vars:
                if hk0 in nwp_map[v]:
                    nwp_file = nwp_map[v][hk0]
                    break

            radar_file = None
            if len(radar_vars) > 0 and len(radar_files[radar_vars[0]]) > 0:
                radar_file = sorted(radar_files[radar_vars[0]])[0]

            label_file = None
            cand_label = select_ra_future(ra_files, hk0)
            if len(cand_label) > 0:
                label_file = cand_label[0]
            elif len(ra_files) > 0:
                label_file = ra_files[0]

            if nwp_file is not None and radar_file is not None and label_file is not None:
                visualize_one_case(
                    nwp_file, radar_file, label_file,
                    VIS_OUT_DIR,
                    prefix=f"{os.path.basename(cp)}_{hk0}"
                )
                vis_count += 1
        except Exception as e:
            print("[VIS ERROR]", os.path.basename(cp), e)

# =========================================================
# 保存patch
# =========================================================
os.makedirs(OUT_PATCH_DIR, exist_ok=True)

for i, patch in enumerate(all_patches):
    path = os.path.join(OUT_PATCH_DIR, f"patch_{i}.npy")
    np.save(path, patch)
    print("Saved:", path, patch.shape)
