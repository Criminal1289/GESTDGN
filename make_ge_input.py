import os, glob, re, math
import numpy as np

# ====== 你需要改的 1 个地方：cp 文件夹们的上级目录 ======
cp_root = r"../GE-STDGN-main/data/BJ"   # 例如：r"D:\datasets\cp_all"

# 8 个特征 + 1 个标签 = 9 维（要和 main.py 的 --feature=9 对齐）
nwp_vars = ["CAPE","DVG850","DVG925","HT0","HT10","KI","PWAT","RH500"]
label_var = "RA"

TARGET_T = 5848
N_NODES = 184
GRID_HW = 167

def time_key(fname: str):
    # 兼容 ..._0517-1600_xxx.npy 或 ..._0517-1600.npy
    b = os.path.basename(fname)
    m = re.search(r"_(\d{4}-\d{4})(?:_|\.npy)", b)
    return m.group(1) if m else None

def resize_nn(src2d: np.ndarray, out_h: int, out_w: int):
    h,w = src2d.shape
    ys = (np.linspace(0, h-1, out_h)).astype(int)
    xs = (np.linspace(0, w-1, out_w)).astype(int)
    return src2d[np.ix_(ys, xs)]

def sample_at_nodes(grid2d: np.ndarray, node_idx: np.ndarray):
    return grid2d[node_idx[:,0], node_idx[:,1]].astype(np.float32)

# 184 个节点：从 167×167 均匀取 14×14=196 个，再取前 184
idx = np.linspace(0, GRID_HW-1, 14, dtype=int)
node_idx = np.array([(i,j) for i in idx for j in idx], dtype=int)[:N_NODES]

all_blocks = []

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
    F = len(nwp_vars) + 1
    block = np.zeros((T, N_NODES, F), dtype=np.float32)

    for ti, tk in enumerate(common):
        # features
        for fi, v in enumerate(nwp_vars):
            g = np.load(nwp_map[v][tk])
            if g.shape != (GRID_HW, GRID_HW):
                g = resize_nn(g, GRID_HW, GRID_HW)
            block[ti, :, fi] = sample_at_nodes(g, node_idx)

        # label
        y = np.load(lab_map[tk])
        if y.shape != (GRID_HW, GRID_HW):
            y = resize_nn(y, GRID_HW, GRID_HW)
        block[ti, :, -1] = sample_at_nodes(y, node_idx)

    all_blocks.append(block)
    print(f"[OK] {os.path.basename(cp)} -> {block.shape}")

dataset = np.concatenate(all_blocks, axis=0)
if dataset.shape[0] < TARGET_T:
    rep = math.ceil(TARGET_T / dataset.shape[0])
    dataset = np.tile(dataset, (rep, 1, 1))[:TARGET_T]
else:
    dataset = dataset[:TARGET_T]

out_path = os.path.join("data", "DataSet.npy")  # 放到 GE-STDGN-main/data/
os.makedirs("data", exist_ok=True)
np.save(out_path, dataset)
print("Saved:", out_path, dataset.shape)
