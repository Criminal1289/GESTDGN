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

N_NODES = 184
GRID_HW = 167


# ======================
# Utils
# ======================
def time_key(fname: str):
    """从文件名里提取时间键，例如 _0517-1710_"""
    b = os.path.basename(fname)
    m = re.search(r"_(\d{4}-\d{4})(?:_|\.npy)", b)
    return m.group(1) if m else None


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
    """
    只处理 RA：
    - 所有 <0 的值替换为 0
    - NaN 也替换为 0（强烈建议，避免训练报错）
    """
    ra = ra2d.astype(np.float32)
    ra[np.isnan(ra)] = 0.0
    ra[ra < 0] = 0.0
    return ra


# ======================
# Build node coords + adjacency
# ======================
# 184 个节点：从 167×167 均匀取 14×14=196 个，再取前 184
idx = np.linspace(0, GRID_HW - 1, 14, dtype=int)
node_idx = np.array([(i, j) for i in idx for j in idx], dtype=int)[:N_NODES]

# 保存 coords / adj（保持你原有逻辑）
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

for cp in cp_folders:
    # NWP 文件映射：变量 -> {time_key: filepath}
    nwp_map = {}
    for v in nwp_vars:
        files = glob.glob(os.path.join(cp, "NWP", v, "*.npy"))
        d = {}
        for f in files:
            tk = time_key(f)
            if tk:
                d[tk] = f
        nwp_map[v] = d

    # RA label 映射：{time_key: filepath}
    lab_files = glob.glob(os.path.join(cp, "LABEL", label_var, "*.npy"))
    lab_map = {}
    for f in lab_files:
        tk = time_key(f)
        if tk and tk not in lab_map:
            lab_map[tk] = f

    # 对齐时间步：必须 NWP 全变量 + RA 都存在
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
    F = len(nwp_vars) + 1  # 27 features + 1 label(RA)
    patch = np.zeros((T, N_NODES, F), dtype=np.float32)

    for ti, tk in enumerate(common):
        # -------- NWP features（不做缺测处理，只读入采样）--------
        for fi, v in enumerate(nwp_vars):
            g = np.load(nwp_map[v][tk]).astype(np.float32)
            if g.shape != (GRID_HW, GRID_HW):
                g = resize_nn(g, GRID_HW, GRID_HW)
            patch[ti, :, fi] = sample_at_nodes(g, node_idx)

        # -------- RA label：只做 <0 -> 0 --------
        ra = np.load(lab_map[tk]).astype(np.float32)
        if ra.shape != (GRID_HW, GRID_HW):
            ra = resize_nn(ra, GRID_HW, GRID_HW)

        ra = preprocess_ra_grid(ra)
        patch[ti, :, -1] = sample_at_nodes(ra, node_idx)

    all_patches.append(patch)
    print(f"[OK] {os.path.basename(cp)} -> patch {patch.shape}")

# ======================
# Save patches (NO masks)
# ======================
out_dir = os.path.join("data", "patch")
os.makedirs(out_dir, exist_ok=True)

for i, patch in enumerate(all_patches):
    out_patch = os.path.join(out_dir, f"patch_{i}.npy")
    np.save(out_patch, patch)
    print(f"Saved: {out_patch} {patch.shape}")

# ======================
# Quick sanity check: RA should be >= 0
# ======================
if __name__ == "__main__":
    for i in range(min(5, len(all_patches))):
        p = np.load(os.path.join(out_dir, f"patch_{i}.npy"))
        ra = p[..., -1]
        print(f"[CHECK patch_{i}] RA min/max:", float(ra.min()), float(ra.max()))
