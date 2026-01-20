import os
import numpy as np

def build_knn_adj(coords: np.ndarray, k: int = 8, mode: str = "binary", sigma: float = 1.0) -> np.ndarray:
    """
    coords: (N, D) 节点坐标（D=2/3 都行）
    k: 每个点连接的近邻数
    mode:
      - "binary": 0/1 邻接
      - "gaussian": 用 exp(-d^2/(2*sigma^2)) 作为权重
    sigma: gaussian 权重用
    """
    N = coords.shape[0]
    # 计算距离矩阵（N,N）
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=-1))  # (N,N)

    # 对每个节点找 k 个最近邻（排除自己）
    knn_idx = np.argsort(dist, axis=1)[:, 1:k+1]  # (N,k)

    A = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        nbrs = knn_idx[i]
        if mode == "binary":
            A[i, nbrs] = 1.0
        elif mode == "gaussian":
            d = dist[i, nbrs]
            A[i, nbrs] = np.exp(-(d ** 2) / (2 * sigma ** 2)).astype(np.float32)
        else:
            raise ValueError("mode must be 'binary' or 'gaussian'")

    # 对称化（无向图）
    A = np.maximum(A, A.T)

    # 加自环（很多 GNN 更稳）
    np.fill_diagonal(A, 1.0)

    return A

if __name__ == "__main__":
    # 你需要准备一个 coords.npy: shape (N,2) 或 (N,3)
    # 比如网格点坐标 (row, col) 或 (lat, lon)
    coords_path = "data/BJ/coords.npy"       # 你改成自己的路径
    out_path = "data/BJ/adj_knn.npy"         # 输出路径
    k = 8                                    # 常用 6/8/10
    mode = "binary"                          # 推荐先 binary，稳定
    sigma = 1.0

    coords = np.load(coords_path)
    A = build_knn_adj(coords, k=k, mode=mode, sigma=sigma)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, A)
    print("Saved:", out_path, "shape:", A.shape, "k:", k, "mode:", mode)
