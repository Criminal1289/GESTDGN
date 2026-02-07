import os
import glob
import re
import numpy as np
from make_knn_adj import build_knn_adj

# ======================
# Config
# ======================
cp_root = r"../gestdgn/data/BJ"

nwp_vars = [
    "CAPE", "DVG200", "DVG850", "DVG925", "HT0", "HT10", "HTw0", "KI",
    "LCL", "LFC", "LI300", "LI500", "muLCL", "PE", "PWAT", "Q1000", "Q700",
    "Q850", "RH1000", "RH500", "RH700", "TdSfc850", "TTdMax74", "TTdMean74",
    "WS500", "WS700", "WS925"
]
label_var = "RA"

radar_vars = ["CR", "HBR", "ET", "VIL"]
radar_root_name = "RADAR"

N_NODES = 184
GRID_HW = 167


# ======================
# Utils
# ======================
def time_key(fname: str):
    """从文件名里提取时间键，例如 _0517-1710_ -> '0517-1710'"""
    b = os.path.basename(fname)
    m = re.search(r"_(\d{4}-\d{4})(?:_|\.npy)", b)
    return m.group(1) if m else None


def hour_key(fname: str):
    """
    统一按 1 小时对齐：
    _0517-1710_ -> '0517-1700'
    _0517-1700_ -> '0517-1700'
    """
    tk = time_key(fname)
    if not tk:
        return None
    mmdd, hhmm = tk.split("-")
    hh = hhmm[:2]
    return f"{mmdd}-{hh}00"


def resize_nn(src2d: np.ndarray, out_h: int, out_w: int):
    """最近邻 resize 到 out_h*out_w（避免引入插值误差）"""
    h, w = src2d.shape
    ys = (np.linspace(0, h - 1, out_h)).astype(int)
    xs = (np.linspace(0, w - 1, out_w)).astype(int)
    return src2d[np.ix_(ys, xs)]


def sample_at_nodes(grid2d: np.ndarray, node_idx: np.ndarray):
    """从 2D 网格采样到 N 个节点"""
    return grid2d[node_idx[:, 0], node_idx[:, 1]].astype(np.float32)


def preprocess_ra_grid(ra2d: np.ndarray):
    """只处理 RA：NaN->0, <0->0"""
    ra = ra2d.astype(np.float32)
    ra[np.isnan(ra)] = 0.0
    ra[ra < 0] = 0.0
    return ra


def preprocess_radar_grid(x2d: np.ndarray, clip_min=None, clip_max=None):
    """雷达通用清洗：NaN->0，可选裁剪"""
    x = x2d.astype(np.float32)
    x[np.isnan(x)] = 0.0
    if clip_min is not None:
        x[x < clip_min] = clip_min
    if clip_max is not None:
        x[x > clip_max] = clip_max
    return x


def load_and_fix_grid(path: str):
    g = np.load(path).astype(np.float32)
    if g.shape != (GRID_HW, GRID_HW):
        g = resize_nn(g, GRID_HW, GRID_HW)
    return g


def build_hourly_map(files, allow_multi=False):
    """
    返回 dict: hour_key -> filepath 或 filepath list
    allow_multi=True 时，同一小时保留多文件（用于雷达 10min -> 1h 聚合）
    """
    d = {}
    for f in files:
        hk = hour_key(f)
        if not hk:
            continue
        if not allow_multi:
            # 保留第一份即可（NWP/RA 通常每小时一份）
            if hk not in d:
                d[hk] = f
        else:
            d.setdefault(hk, []).append(f)
    return d


def aggregate_hourly(files, how="max"):
    """
    将一个小时内多张雷达图聚合成一张 2D 网格
    how: "max" / "mean"
    """
    grids = [load_and_fix_grid(fp) for fp in sorted(files)]
    stack = np.stack(grids, axis=0)  # [K,H,W]
    if how == "mean":
        return stack.mean(axis=0)
    # 默认 max：对强对流/回波类更稳
    return stack.max(axis=0)


# ======================
# Build node coords + adjacency
# ======================
idx = np.linspace(0, GRID_HW - 1, 14, dtype=int)
node_idx = np.array([(i, j) for i in idx for j in idx], dtype=int)[:N_NODES]

os.makedirs(r"../gestdgn/data", exist_ok=True)
np.save(r"../gestdgn/data/coords.npy", node_idx.astype(np.float32))

coords = np.load(r"../gestdgn/data/coords.npy")
A = build_knn_adj(coords, k=8, mode="binary")
np.save(r"../gestdgn/data/adj_knn.npy", A)


# ======================
# Generate patches
# ======================
all_patches = []

cp_folders = sorted([p for p in glob.glob(os.path.join(cp_root, "*")) if os.path.isdir(p)])
if not cp_folders:
    raise RuntimeError(f"没找到 cp 文件夹：{cp_root}")

# 雷达每个产品的 1h 聚合策略（可按需改）
radar_agg = {
    "CR": "max",
    "HBR": "max",
    "ET": "max",    # 云顶高度，用 max 更合理
    "VIL": "max",   # 想更平滑可以改 "mean"
    "CAP20": "max",
    "CAP30": "max",
    "CAP40": "max",
    "CAP50": "max",
    "CAP60": "max",
    "CAP70": "max",
}

for cp in cp_folders:
    # -------- NWP 文件映射：变量 -> {hour_key: filepath} --------
    nwp_map = {}
    for v in nwp_vars:
        files = glob.glob(os.path.join(cp, "NWP", v, "*.npy"))
        nwp_map[v] = build_hourly_map(files, allow_multi=False)

    # -------- RA label 映射：{hour_key: filepath} --------
    lab_files = glob.glob(os.path.join(cp, "LABEL", label_var, "*.npy"))
    lab_map = build_hourly_map(lab_files, allow_multi=False)

    # -------- RADAR 映射：变量 -> {hour_key: filelist} --------
    radar_map = {}
    radar_root = os.path.join(cp, radar_root_name)
    for rv in radar_vars:
        files = glob.glob(os.path.join(radar_root, rv, "*.npy"))
        radar_map[rv] = build_hourly_map(files, allow_multi=True)  # 雷达可能同一小时多张

    # -------- 对齐时间步：NWP 全变量 + RADAR 全变量 + RA 都存在（按小时）--------
    common = None
    for v in nwp_vars:
        s = set(nwp_map[v].keys())
        common = s if common is None else (common & s)

    # 叠加 radar
    for rv in radar_vars:
        s = set(radar_map[rv].keys())
        common = s if common is None else (common & s)

    # 叠加 label
    common = (common & set(lab_map.keys())) if common is not None else set()
    common = sorted(common)

    if not common:
        print(f"[SKIP] {cp} 没有可对齐时间步（请检查 {radar_root_name}/NWP/LABEL 路径与文件命名）")
        continue

    T = len(common)
    F = len(nwp_vars) + len(radar_vars) + 1  # NWP + RADAR + label(RA)
    patch = np.zeros((T, N_NODES, F), dtype=np.float32)

    # 特征列索引
    nwp_offset = 0
    radar_offset = len(nwp_vars)
    label_offset = len(nwp_vars) + len(radar_vars)

    for ti, hk in enumerate(common):
        # -------- NWP features --------
        for fi, v in enumerate(nwp_vars):
            g = load_and_fix_grid(nwp_map[v][hk])
            patch[ti, :, nwp_offset + fi] = sample_at_nodes(g, node_idx)

        # -------- RADAR features（按 1h 聚合）--------
        for rfi, rv in enumerate(radar_vars):
            filelist = radar_map[rv][hk]
            # 聚合成 1 张小时图
            g = aggregate_hourly(filelist, how=radar_agg.get(rv, "max"))

            # 简单清洗：雷达产品一般 >=0；dBZ 类可适当裁剪上限
            # 注意：你这里 unit=0.1dBZ，数值可能是 0~600（对应 0~60dBZ）
            if rv in ["CR", "HBR", "CAP20", "CAP30", "CAP40", "CAP50", "CAP60", "CAP70"]:
                # unit=0.1 dBZ, 常见范围 0~600 (对应 0~60 dBZ)
                g = preprocess_radar_grid(g, clip_min=0.0, clip_max=600.0)
                g = g / 600.0  # -> [0,1]
            elif rv == "ET":
                # unit=0.1 km, 常见范围 0~300 (对应 0~30 km)
                g = preprocess_radar_grid(g, clip_min=0.0, clip_max=300.0)
                g = g / 300.0  # -> [0,1]
            elif rv == "VIL":
                # unit=0.1 kg/m^2, 你先用 0~1000 (对应 0~100 kg/m^2) 起步
                g = preprocess_radar_grid(g, clip_min=0.0, clip_max=1000.0)
                g = g / 1000.0  # -> [0,1]
            else:
                g = preprocess_radar_grid(g)

            patch[ti, :, radar_offset + rfi] = sample_at_nodes(g, node_idx)

        # -------- RA label --------
        ra = load_and_fix_grid(lab_map[hk])
        ra = preprocess_ra_grid(ra)
        patch[ti, :, label_offset] = sample_at_nodes(ra, node_idx)

    all_patches.append(patch)
    print(f"[OK] {os.path.basename(cp)} -> patch {patch.shape}  (F={F}=NWP{len(nwp_vars)}+RADAR{len(radar_vars)}+1)")

# ======================
# Save patches
# ======================
out_dir = os.path.join("data", "patch")
os.makedirs(out_dir, exist_ok=True)

for i, patch in enumerate(all_patches):
    out_patch = os.path.join(out_dir, f"patch_{i}.npy")
    np.save(out_patch, patch)
    print(f"Saved: {out_patch} {patch.shape}")

# ======================
# Quick sanity check
# ======================
if __name__ == "__main__":
    for i in range(min(5, len(all_patches))):
        p = np.load(os.path.join(out_dir, f"patch_{i}.npy"))
        ra = p[..., -1]
        print(f"[CHECK patch_{i}] RA min/max:", float(ra.min()), float(ra.max()))
