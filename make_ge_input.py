import os, glob, re, math
import numpy as np
from make_knn_adj import *

cp_root = r"../gestdgn/data/BJ"

nwp_vars = [
    "CAPE", "DVG200", "DVG850", "DVG925", "HT0", "HT10", "HTw0", "KI",
    "LCL", "LFC", "LI300", "LI500", "muLCL", "PE", "PWAT", "Q1000", "Q700",
    "Q850", "RH1000", "RH500", "RH700", "TdSfc850", "TTdMax74", "TTdMean74",
    "WS500", "WS700", "WS925"
]
label_var = "RA"

N_NODES = 184
GRID_HW = 167

# ✅ 缺测值占位符
MISS = -9.0


def time_key(fname: str):
    b = os.path.basename(fname)
    m = re.search(r"_(\d{4}-\d{4})(?:_|\.npy)", b)
    return m.group(1) if m else None


def resize_nn(src2d: np.ndarray, out_h: int, out_w: int):
    h, w = src2d.shape
    ys = (np.linspace(0, h - 1, out_h)).astype(int)
    xs = (np.linspace(0, w - 1, out_w)).astype(int)
    return src2d[np.ix_(ys, xs)]


def sample_at_nodes(grid2d: np.ndarray, node_idx: np.ndarray):
    return grid2d[node_idx[:, 0], node_idx[:, 1]].astype(np.float32)


# ✅ 新增：对网格做缺测处理 + 返回 node 上的值与mask
def sample_with_mask_and_fill(grid2d: np.ndarray, node_idx: np.ndarray, miss=MISS):
    g = grid2d.astype(np.float32)

    # 把 miss 变成 NaN，便于统计均值
    g[g == miss] = np.nan

    # node 采样
    vals = sample_at_nodes(g, node_idx)  # 这里的 NaN 会被采样出来
    mask = (~np.isnan(vals)).astype(np.float32)  # 1=有效, 0=缺测

    # 用“该张网格的有效均值”填补 node 缺测（比填0更稳）
    fill_value = np.nanmean(g)
    if np.isnan(fill_value):
        # 极端情况：整张都是缺测，那就填 0（或你也可以填历史均值）
        fill_value = 0.0

    vals_filled = np.where(np.isnan(vals), fill_value, vals).astype(np.float32)
    return vals_filled, mask


# 184 个节点：从 167×167 均匀取 14×14=196 个，再取前 184
idx = np.linspace(0, GRID_HW - 1, 14, dtype=int)
node_idx = np.array([(i, j) for i in idx for j in idx], dtype=int)[:N_NODES]

np.save(r"../gestdgn/data/coords.npy", node_idx.astype(np.float32))
coords = np.load(r"../gestdgn/data/coords.npy")
A = build_knn_adj(coords, k=8, mode="binary")
np.save(r"../gestdgn/data/adj_knn.npy", A)

all_patches = []
all_masks = []  # ✅ 新增：保存对应mask（features+label）

cp_folders = sorted([p for p in glob.glob(os.path.join(cp_root, "*")) if os.path.isdir(p)])
if not cp_folders:
    raise RuntimeError(f"没找到 cp 文件夹：{cp_root}")

for cp in cp_folders:
    nwp_map = {}
    for v in nwp_vars:
        files = glob.glob(os.path.join(cp, "NWP", v, "*.npy"))
        d = {}
        for f in files:
            tk = time_key(f)
            if tk:
                d[tk] = f
        nwp_map[v] = d

    lab_files = glob.glob(os.path.join(cp, "LABEL", label_var, "*.npy"))
    lab_map = {}
    for f in lab_files:
        tk = time_key(f)
        if tk and tk not in lab_map:
            lab_map[tk] = f

    common = None
    for v in nwp_vars:
        s = set(nwp_map[v].keys())
        common = s if common is None else (common & s)
    common = (common & set(lab_map.keys())) if common is not None else set()
    common = sorted(common)

    if not common:
        print(f"[SKIP] {cp} 没有可对齐时间步")
        continue

    T = len(common)
    F = len(nwp_vars) + 1  # features + label
    patch = np.zeros((T, N_NODES, F), dtype=np.float32)
    maskp = np.zeros((T, N_NODES, F), dtype=np.float32)  # ✅ 同shape mask

    for ti, tk in enumerate(common):
        # features
        for fi, v in enumerate(nwp_vars):
            g = np.load(nwp_map[v][tk])
            if g.shape != (GRID_HW, GRID_HW):
                g = resize_nn(g, GRID_HW, GRID_HW)

            vals_filled, m = sample_with_mask_and_fill(g, node_idx, miss=MISS)
            patch[ti, :, fi] = vals_filled
            maskp[ti, :, fi] = m

        # label
        y = np.load(lab_map[tk])
        if y.shape != (GRID_HW, GRID_HW):
            y = resize_nn(y, GRID_HW, GRID_HW)

        y_filled, y_m = sample_with_mask_and_fill(y, node_idx, miss=MISS)
        patch[ti, :, -1] = y_filled
        maskp[ti, :, -1] = y_m

    all_patches.append(patch)
    all_masks.append(maskp)
    print(f"[OK] {os.path.basename(cp)} -> patch {patch.shape}, mask {maskp.shape}")

# ✅ 输出目录修复
out_dir = os.path.join("data", "patch")
os.makedirs(out_dir, exist_ok=True)

for i, (patch, maskp) in enumerate(zip(all_patches, all_masks)):
    out_patch = os.path.join(out_dir, f"patch_{i}.npy")
    out_mask = os.path.join(out_dir, f"patch_{i}_mask.npy")
    np.save(out_patch, patch)
    np.save(out_mask, maskp)
    print(f"Saved: {out_patch} {patch.shape}")
    print(f"Saved: {out_mask}  {maskp.shape}")

if __name__ == '__main__':
    for i in range(0, 62):
        p = np.load(f"data/patch/patch_{i}.npy")
        m = np.load(f"data/patch/patch_{i}_mask.npy")

        # print("patch min/max:", p.min(), p.max())
        # print("mask unique:", np.unique(m))
        # print("missing ratio:", 1 - m.mean())
        ra = p[..., 27]
        print("RA shape:", ra.shape)  # (T, 184)
        print("RA min/max:", ra.min(), ra.max())
        # print("RA missing ratio:", 1 - ra_mask.mean())
