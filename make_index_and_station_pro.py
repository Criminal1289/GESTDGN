import os
import numpy as np

COORDS = r"../gestdgn/data/coords.npy"  # 你的coords
OUTDIR = r"../gestdgn/data"


def generateFile(k_min, k_max):
    EPS = 1e-6

    coords = np.load(COORDS).astype(np.float32)  # (184,2)
    N = coords.shape[0]

    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((diff ** 2).sum(-1))  # (N,N)
    np.fill_diagonal(dist, np.inf)

    os.makedirs(OUTDIR, exist_ok=True)
    delete_station_path = os.path.join(OUTDIR, "station_pro.npy")
    if os.path.exists(delete_station_path):
        os.remove(delete_station_path)
    for i in range(k_min, k_max + 1):
        f_name = f"index_{i}.npy"
        delete_index_path = os.path.join(OUTDIR, f_name)
        if os.path.exists(delete_index_path):
            os.remove(delete_index_path)
    # 生成 index_k.npy：边列表 (E,2)
    for k in range(k_min, k_max + 1):
        edges = []
        for i in range(N):
            nbrs = np.argsort(dist[i])[:k]
            for j in nbrs:
                edges.append([i, int(j)])
        edges = np.array(edges, dtype=np.int64)
        np.save(os.path.join(OUTDIR, f"index_{k}.npy"), edges)
        print("saved", f"index_{k}.npy", edges.shape)

    # 生成 station_pro.npy：给最大k的边一个“优先级/权重”（用距离倒数）
    kmax_edges = np.load(os.path.join(OUTDIR, f"index_{k_max}.npy"))
    inv = 1.0 / (dist + EPS)  # dist对角是inf，无影响
    edge_w = np.array([inv[i, j] for i, j in kmax_edges], dtype=np.float32)
    np.save(os.path.join(OUTDIR, "station_pro.npy"), edge_w)
    print("saved station_pro.npy", edge_w.shape)
if __name__ == '__main__':
    generateFile(6,10)