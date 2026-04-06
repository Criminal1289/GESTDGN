import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # gestdgn/cnn
PROJECT_DIR = os.path.dirname(BASE_DIR)                 # gestdgn
CNN_DIR = os.path.join(PROJECT_DIR, "data", "cnn_spatial_align_slide")
INDEX_DIR = os.path.join(CNN_DIR, "index")

COORDS = os.path.join(CNN_DIR, "coords.npy")
ADJ = os.path.join(CNN_DIR, "adj_knn.npy")
STATION = os.path.join(CNN_DIR, "station_pro.npy")
OUTDIR = INDEX_DIR


def generateFile(k_min, k_max):
    EPS = 1e-6

    coords = np.load(COORDS).astype(np.float32)
    N = coords.shape[0]

    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((diff ** 2).sum(-1))
    np.fill_diagonal(dist, np.inf)

    os.makedirs(OUTDIR, exist_ok=True)

    if os.path.exists(STATION):
        os.remove(STATION)

    for i in range(k_min, k_max + 1):
        delete_index_path = os.path.join(OUTDIR, f"index_{i}.npy")
        if os.path.exists(delete_index_path):
            os.remove(delete_index_path)

    for k in range(k_min, k_max + 1):
        edges = []
        for i in range(N):
            nbrs = np.argsort(dist[i])[:k]
            for j in nbrs:
                edges.append([i, int(j)])
        edges = np.array(edges, dtype=np.int64)
        np.save(os.path.join(OUTDIR, f"index_{k}.npy"), edges)
        print("saved", f"index_{k}.npy", edges.shape)

    kmax_edges = np.load(os.path.join(OUTDIR, f"index_{k_max}.npy"))
    inv = 1.0 / (dist + EPS)
    edge_w = np.array([inv[i, j] for i, j in kmax_edges], dtype=np.float32)
    np.save(STATION, edge_w)
    print("saved station_pro.npy", edge_w.shape)


if __name__ == '__main__':
    generateFile(6, 15)