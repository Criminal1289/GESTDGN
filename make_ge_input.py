import os, glob, re, math
import numpy as np

# ====== 你需要改的 1 个地方：cp 文件夹们的上级目录 ======
cp_root = r"../gestdgn/data/BJ"   # 例如：r"D:\datasets\cp_all"

# 手动指定 NWP 中所有特征和标签
nwp_vars = [
    "CAPE", "DVG200", "DVG850", "DVG925", "HT0", "HT10", "HTw0", "KI",
    "LCL", "LFC", "LI300", "LI500", "muLCL", "PE", "PWAT", "Q1000", "Q700",
    "Q850", "RH1000", "RH500", "RH700", "TdSfc850", "TTdMax74", "TTdMean74",
    "WS500", "WS700", "WS925"  # RA作为标签添加
]

# 标签变量
label_var = "RA"

N_NODES = 184
GRID_HW = 167

def time_key(fname: str):
    # 兼容 ..._0517-1600_xxx.npy 或 ..._0517-1600.npy
    b = os.path.basename(fname)
    m = re.search(r"_(\d{4}-\d{4})(?:_|\.npy)", b)
    return m.group(1) if m else None

def resize_nn(src2d: np.ndarray, out_h: int, out_w: int):
    h, w = src2d.shape
    ys = (np.linspace(0, h-1, out_h)).astype(int)
    xs = (np.linspace(0, w-1, out_w)).astype(int)
    return src2d[np.ix_(ys, xs)]

def sample_at_nodes(grid2d: np.ndarray, node_idx: np.ndarray):
    return grid2d[node_idx[:, 0], node_idx[:, 1]].astype(np.float32)

# 184 个节点：从 167×167 均匀取 14×14=196 个，再取前 184
idx = np.linspace(0, GRID_HW - 1, 14, dtype=int)
node_idx = np.array([(i, j) for i in idx for j in idx], dtype=int)[:N_NODES]

# 存储所有patch数据
all_patches = []

cp_folders = sorted([p for p in glob.glob(os.path.join(cp_root, "*")) if os.path.isdir(p)])
if not cp_folders:
    raise RuntimeError(f"没找到 cp 文件夹：{cp_root}")

for cp in cp_folders:
    # 建 NWP 映射：var -> {time: path}
    nwp_map = {}
    for v in nwp_vars:
        files = glob.glob(os.path.join(cp, "NWP", v, "*.npy"))
        d = {}
        for f in files:
            tk = time_key(f)
            if tk: d[tk] = f
        nwp_map[v] = d

    # label 映射：{time: path}（若同一 time 多个，取第一个）
    lab_files = glob.glob(os.path.join(cp, "LABEL", label_var, "*.npy"))
    lab_map = {}
    for f in lab_files:
        tk = time_key(f)
        if tk and tk not in lab_map:
            lab_map[tk] = f

    # 取“所有变量都有 + label 也有”的公共时间
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
    F = len(nwp_vars) + 1  # 标签包含在内
    patch = np.zeros((T, N_NODES, F), dtype=np.float32)

    for ti, tk in enumerate(common):
        # features
        for fi, v in enumerate(nwp_vars):
            g = np.load(nwp_map[v][tk])
            if g.shape != (GRID_HW, GRID_HW):
                g = resize_nn(g, GRID_HW, GRID_HW)
            patch[ti, :, fi] = sample_at_nodes(g, node_idx)

        # label
        y = np.load(lab_map[tk])
        if y.shape != (GRID_HW, GRID_HW):
            y = resize_nn(y, GRID_HW, GRID_HW)
        patch[ti, :, -1] = sample_at_nodes(y, node_idx)

    all_patches.append(patch)
    print(f"[OK] {os.path.basename(cp)} -> {patch.shape}")

# 这里不再将所有patch数据堆叠成一个大数据集，而是每个patch独立存储
for i, patch in enumerate(all_patches):
    out_path = os.path.join("data/patch", f"patch_{i}.npy")  # 保存每个patch为独立文件
    os.makedirs("data", exist_ok=True)
    np.save(out_path, patch)
    print(f"Saved: {out_path}, Shape: {patch.shape}")
