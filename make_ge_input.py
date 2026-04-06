import os
import glob
import re
import numpy as np
from datetime import datetime, timedelta
from make_knn_adj import build_knn_adj
from scipy.interpolate import RectBivariateSpline
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun", "Arial"]
matplotlib.rcParams["axes.unicode_minus"] = False

# =========================================================
# 基线方案
# 167×167 / 301×301 -> 插值到256×256 -> 取32×32
# =========================================================
cp_root = r"../gestdgn/data/BJ"

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

GRID_HW = 256
NODE_HW = 32
N_NODES = NODE_HW * NODE_HW

# Radar窗口
RADAR_INTERVAL = 6
WINDOW_BEFORE = 2
WINDOW_AFTER = 0

# RA未来累计
RA_INTERVAL = 6
RA_HOUR = 60

# 插值方式
RESIZE_METHOD = "bilinear"  # "nearest" / "bilinear" / "spline"

# 输出目录
OUT_PATCH_DIR = "../gestdgn/data/patch"
OUT_META_DIR = "../gestdgn/data"
VIS_OUT_DIR = "../gestdgn/data/patch/compare"

# 可视化开关
ENABLE_VIS = True
VIS_MAX_FILES = 1


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
# Grid工具
# =========================================================
def resize_nearest(src2d, out_h, out_w):
    h, w = src2d.shape
    ys = (np.linspace(0, h - 1, out_h)).astype(int)
    xs = (np.linspace(0, w - 1, out_w)).astype(int)
    return src2d[np.ix_(ys, xs)]


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


def resize_spline(src2d, out_h, out_w):
    h, w = src2d.shape
    y = np.arange(h)
    x = np.arange(w)
    spline = RectBivariateSpline(y, x, src2d, kx=3, ky=3)
    y_new = np.linspace(0, h - 1, out_h)
    x_new = np.linspace(0, w - 1, out_w)
    return spline(y_new, x_new).astype(np.float32)


def resize_grid(src2d, out_h, out_w, method="nearest"):
    method = method.lower()
    if method == "nearest":
        return resize_nearest(src2d, out_h, out_w)
    elif method == "bilinear":
        return resize_bilinear(src2d, out_h, out_w)
    elif method == "spline":
        return resize_spline(src2d, out_h, out_w)
    else:
        raise ValueError(f"Unsupported resize method: {method}")


def load_and_fix_grid(path):
    g = np.load(path).astype(np.float32)
    if g.shape != (GRID_HW, GRID_HW):
        g = resize_grid(g, GRID_HW, GRID_HW, method=RESIZE_METHOD)
    return g


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


def aggregate_grids(files, how="max"):
    if len(files) == 0:
        return None

    grids = [load_and_fix_grid(f) for f in files]
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
        g = preprocess_ra_grid(load_and_fix_grid(f))
        grids.append(g)

    stack = np.stack(grids, axis=0)
    return stack.sum(axis=0).astype(np.float32)


# =========================================================
# 节点生成
# =========================================================
idx = np.linspace(0, GRID_HW - 1, NODE_HW, dtype=int)
node_idx = np.array([(i, j) for i in idx for j in idx], dtype=int)

os.makedirs(OUT_META_DIR, exist_ok=True)
np.save(os.path.join(OUT_META_DIR, "coords.npy"), node_idx.astype(np.float32))

coords = np.load(os.path.join(OUT_META_DIR, "coords.npy"))
A = build_knn_adj(coords, k=8, mode="binary")
np.save(os.path.join(OUT_META_DIR, "adj_knn.npy"), A)

print("coords saved:", os.path.join(OUT_META_DIR, "coords.npy"))
print("adj saved   :", os.path.join(OUT_META_DIR, "adj_knn.npy"))


# =========================================================
# 可视化工具
# =========================================================
def node_values_to_32x32(vals):
    return vals.reshape(NODE_HW, NODE_HW)


def plot_img(ax, arr, title, cmap="jet", vmin=None, vmax=None):
    im = ax.imshow(arr, origin="upper", cmap=cmap, vmin=vmin, vmax=vmax)
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


def visualize_baseline_case(nwp_path, radar_path, label_path, out_dir, prefix):
    os.makedirs(out_dir, exist_ok=True)

    # 原始
    nwp_raw = np.load(nwp_path).astype(np.float32)
    radar_raw = np.load(radar_path).astype(np.float32)
    label_raw = np.load(label_path).astype(np.float32)

    # 插值到256
    nwp_256 = load_and_fix_grid(nwp_path)
    radar_256 = load_and_fix_grid(radar_path)
    label_256 = load_and_fix_grid(label_path)

    # 取32x32
    nwp_nodes = sample_at_nodes(nwp_256, node_idx)
    radar_nodes = sample_at_nodes(radar_256, node_idx)
    label_nodes = sample_at_nodes(label_256, node_idx)

    nwp_32 = node_values_to_32x32(nwp_nodes)
    radar_32 = node_values_to_32x32(radar_nodes)
    label_32 = node_values_to_32x32(label_nodes)

    # 回投影到256
    nwp_back = resize_bilinear(nwp_32, GRID_HW, GRID_HW)
    radar_back = resize_bilinear(radar_32, GRID_HW, GRID_HW)
    label_back = resize_bilinear(label_32, GRID_HW, GRID_HW)

    # 差值
    nwp_diff = nwp_back - nwp_256
    radar_diff = radar_back - radar_256
    label_diff = label_back - label_256

    # NWP图
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    vmin = np.nanmin(nwp_256)
    vmax = np.nanmax(nwp_256)

    plot_img(axes[0, 0], nwp_raw, f"NWP original\nshape={nwp_raw.shape}")
    plot_img(axes[0, 1], nwp_256, "NWP resized to 256x256", vmin=vmin, vmax=vmax)
    plot_img_with_points(axes[0, 2], nwp_256, node_idx,
                         "NWP 256x256 + 32x32 sampled nodes", vmin=vmin, vmax=vmax)

    plot_img(axes[1, 0], nwp_32, "NWP sampled node values (32x32)")
    plot_img(axes[1, 1], nwp_back, "NWP 32x32 -> back to 256x256", vmin=vmin, vmax=vmax)
    plot_diff(axes[1, 2], nwp_diff, "Difference:\n(back-projected 32x32) - resized 256")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_nwp.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # RADAR图
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    vmin = np.nanmin(radar_256)
    vmax = np.nanmax(radar_256)

    plot_img(axes[0, 0], radar_raw, f"RADAR original\nshape={radar_raw.shape}")
    plot_img(axes[0, 1], radar_256, "RADAR resized to 256x256", vmin=vmin, vmax=vmax)
    plot_img_with_points(axes[0, 2], radar_256, node_idx,
                         "RADAR 256x256 + 32x32 sampled nodes", vmin=vmin, vmax=vmax)

    plot_img(axes[1, 0], radar_32, "RADAR sampled node values (32x32)")
    plot_img(axes[1, 1], radar_back, "RADAR 32x32 -> back to 256x256", vmin=vmin, vmax=vmax)
    plot_diff(axes[1, 2], radar_diff, "Difference:\n(back-projected 32x32) - resized 256")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_radar.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # LABEL图
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    vmin = np.nanmin(label_256)
    vmax = np.nanmax(label_256)

    plot_img(axes[0, 0], label_raw, f"LABEL original\nshape={label_raw.shape}")
    plot_img(axes[0, 1], label_256, "LABEL resized to 256x256", vmin=vmin, vmax=vmax)
    plot_img_with_points(axes[0, 2], label_256, node_idx,
                         "LABEL 256x256 + 32x32 sampled nodes", vmin=vmin, vmax=vmax)

    plot_img(axes[1, 0], label_32, "LABEL sampled node values (32x32)")
    plot_img(axes[1, 1], label_back, "LABEL 32x32 -> back to 256x256", vmin=vmin, vmax=vmax)
    plot_diff(axes[1, 2], label_diff, "Difference:\n(back-projected 32x32) - resized 256")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{prefix}_label.png"), dpi=200, bbox_inches="tight")
    plt.close()

    print(f"[VIS OK] {prefix}")
    print("  NWP   diff abs mean/max :", np.mean(np.abs(nwp_diff)), np.max(np.abs(nwp_diff)))
    print("  RADAR diff abs mean/max :", np.mean(np.abs(radar_diff)), np.max(np.abs(radar_diff)))
    print("  LABEL diff abs mean/max :", np.mean(np.abs(label_diff)), np.max(np.abs(label_diff)))


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
        print("[SKIP] no common hourly keys from NWP")
        continue

    T = len(common)
    F = len(nwp_vars) + len(radar_vars) + 1

    patch = np.zeros((T, N_NODES, F), dtype=np.float32)

    nwp_offset = 0
    radar_offset = len(nwp_vars)
    label_offset = len(nwp_vars) + len(radar_vars)

    # =====================
    # 构建patch
    # =====================
    for ti, hk in enumerate(common):
        # NWP
        for fi, v in enumerate(nwp_vars):
            if hk in nwp_map[v]:
                g = load_and_fix_grid(nwp_map[v][hk])
                patch[ti, :, nwp_offset + fi] = sample_at_nodes(g, node_idx)

        # Radar
        for rfi, rv in enumerate(radar_vars):
            files = select_radar_files(radar_files[rv], hk)
            g = aggregate_grids(files)

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

            patch[ti, :, radar_offset + rfi] = sample_at_nodes(g, node_idx)

        # RA
        future_files = select_ra_future(ra_files, hk)
        g = aggregate_ra_sum(future_files)

        if g is not None:
            patch[ti, :, label_offset] = sample_at_nodes(g, node_idx)

    all_patches.append(patch)
    print("[OK]", os.path.basename(cp), "->", patch.shape)

    # =====================
    # 可视化前几个样本
    # =====================
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
                cand = select_radar_files(radar_files[radar_vars[0]], hk0)
                if len(cand) > 0:
                    radar_file = cand[0]
                else:
                    radar_file = sorted(radar_files[radar_vars[0]])[0]
                    print("[VIS WARN] no radar matched hk0, fallback to first radar file:", radar_file)

            label_file = None
            cand_label = select_ra_future(ra_files, hk0)
            if len(cand_label) > 0:
                label_file = cand_label[0]
            elif len(ra_files) > 0:
                label_file = sorted(ra_files)[0]
                print("[VIS WARN] no label matched hk0, fallback to first label file:", label_file)

            if nwp_file is not None and radar_file is not None and label_file is not None:
                visualize_baseline_case(
                    nwp_file, radar_file, label_file,
                    VIS_OUT_DIR,
                    prefix=f"{os.path.basename(cp)}_{hk0}"
                )
                vis_count += 1
            else:
                print("[VIS SKIP]", os.path.basename(cp), hk0)

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

print("VIS_OUT_DIR abs path:", os.path.abspath(VIS_OUT_DIR))
print("Done.")
