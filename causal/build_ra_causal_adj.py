#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
build_ra_causal_adj.py

功能：
1. 读取训练流程中保存的 32x32 RA 序列：
       data/ra_train_seq_32x32_raw.npy
2. 将 RA 从 0.1 mm 转成 mm；
3. 对每个节点只在局部邻域内做 Granger 因果检验；
4. 构建因果邻接矩阵 A_cau_ra.npy。

输入：
    ra_train_seq_32x32_raw.npy: [T_train, 1024]

输出：
    A_cau_ra.npy: [1024, 1024]

方向说明：
    A[target, source] 表示 target 节点聚合 source 节点的信息。
    如果你的图卷积是 H_new = A @ H，这个方向是合适的。
"""

import os
import argparse
import warnings

import numpy as np
from tqdm import tqdm

warnings.filterwarnings("ignore")

try:
    from statsmodels.tsa.stattools import grangercausalitytests
except ImportError:
    raise ImportError("缺少 statsmodels，请先安装：pip install statsmodels tqdm")


def get_local_neighbors(node_id, grid_h=32, grid_w=32, radius=2, include_self=False):
    r = node_id // grid_w
    c = node_id % grid_w

    neighbors = []

    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            rr = r + dr
            cc = c + dc

            if rr < 0 or rr >= grid_h or cc < 0 or cc >= grid_w:
                continue

            nid = rr * grid_w + cc

            if not include_self and nid == node_id:
                continue

            neighbors.append(nid)

    return neighbors


def safe_granger_pvalue(target, source, maxlag=4):
    """
    检验 source 是否 Granger 导致 target。

    statsmodels 中：
        data[:, 0] = target
        data[:, 1] = source

    它检验的是第二列 source 是否 Granger 导致第一列 target。
    """
    try:
        target = np.asarray(target, dtype=np.float64)
        source = np.asarray(source, dtype=np.float64)

        if len(target) != len(source):
            return None

        if len(target) < maxlag * 3 + 5:
            return None

        if np.std(target) < 1e-8 or np.std(source) < 1e-8:
            return None

        data = np.column_stack([target, source])

        result = grangercausalitytests(data, maxlag=maxlag, verbose=False)

        p_values = []
        for lag in range(1, maxlag + 1):
            p = result[lag][0]["ssr_ftest"][1]
            p_values.append(p)

        return float(np.min(p_values))

    except Exception:
        return None


def row_normalize(A):
    row_sum = A.sum(axis=1, keepdims=True)
    return A / (row_sum + 1e-8)


def build_causal_adj(
    ra_path,
    save_path,
    grid_h=32,
    grid_w=32,
    radius=2,
    maxlag=4,
    p_threshold=0.05,
    top_k=8,
    ra_unit_01mm=True,
    use_log1p=True,
):
    """
    根据训练集 32x32 RA 序列构建局部 Granger 因果邻接矩阵。
    """
    ra = np.load(ra_path).astype(np.float32)

    print("[LOAD] ra_path:", ra_path)
    print("[INFO] original ra shape:", ra.shape)

    if ra.ndim != 2:
        raise ValueError(f"输入 RA 必须是 [T, N]，当前 shape={ra.shape}")

    T, N = ra.shape

    if N != grid_h * grid_w:
        raise ValueError(
            f"N={N} 与 grid_h*grid_w={grid_h * grid_w} 不一致，"
            f"请确认你的 label 是否已经是 {grid_h}x{grid_w}"
        )

    # 缺测值处理
    ra[ra < 0] = 0

    # 0.1 mm -> mm
    if ra_unit_01mm:
        ra = ra * 0.1

    # 降水分布偏态，log1p 更稳
    if use_log1p:
        ra = np.log1p(ra)

    A = np.zeros((N, N), dtype=np.float32)

    print("[INFO] start building causal adjacency")
    print(f"[INFO] T={T}, N={N}, grid={grid_h}x{grid_w}")
    print(f"[INFO] radius={radius}, maxlag={maxlag}, p_threshold={p_threshold}, top_k={top_k}")

    for target_j in tqdm(range(N), desc="Building causal adjacency"):
        candidates = get_local_neighbors(
            node_id=target_j,
            grid_h=grid_h,
            grid_w=grid_w,
            radius=radius,
            include_self=False,
        )

        target_series = ra[:, target_j]

        edges = []

        for source_i in candidates:
            source_series = ra[:, source_i]

            p = safe_granger_pvalue(
                target=target_series,
                source=source_series,
                maxlag=maxlag,
            )

            if p is None:
                continue

            if p < p_threshold:
                weight = -np.log10(p + 1e-12)
                edges.append((source_i, weight, p))

        # 每个目标节点只保留 top_k 条最强因果边
        edges = sorted(edges, key=lambda x: x[1], reverse=True)[:top_k]

        for source_i, weight, p in edges:
            # A[target, source]
            A[target_j, source_i] = weight

    # 加自环
    A += np.eye(N, dtype=np.float32)

    # 行归一化
    A = row_normalize(A)

    # os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, A)

    print("=" * 80)
    print("[SAVE] causal adjacency:", save_path)
    print("[STAT] A shape:", A.shape)
    print("[STAT] nonzero:", np.count_nonzero(A))
    print("[STAT] nonzero ratio:", np.count_nonzero(A) / A.size)
    print("[STAT] A min/mean/max:", A.min(), A.mean(), A.max())
    print("=" * 80)

    return A


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ra_path",
        type=str,
        default="ra_train_seq_32x32_raw.npy",
        help="训练流程保存出的 32x32 RA 序列，shape=[T_train,1024]"
    )

    parser.add_argument(
        "--save_path",
        type=str,
        default="A_cau_ra.npy",
        help="输出因果邻接矩阵路径"
    )

    parser.add_argument("--grid_h", type=int, default=32)
    parser.add_argument("--grid_w", type=int, default=32)

    parser.add_argument(
        "--radius",
        type=int,
        default=2,
        help="局部邻域半径，2 表示 5x5 邻域"
    )

    parser.add_argument(
        "--maxlag",
        type=int,
        default=6,
        help="Granger 最大滞后阶数"
    )

    parser.add_argument(
        "--p_threshold",
        type=float,
        default=0.05,
        help="显著性阈值"
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=8,
        help="每个目标节点最多保留 top_k 条边"
    )

    parser.add_argument(
        "--no_log1p",
        action="store_true",
        help="默认使用 log1p；加上该参数则不使用"
    )

    parser.add_argument(
        "--ra_already_mm",
        action="store_true",
        help="如果输入 RA 已经是 mm，则加上该参数；默认认为输入是 0.1mm"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    build_causal_adj(
        ra_path=args.ra_path,
        save_path=args.save_path,
        grid_h=args.grid_h,
        grid_w=args.grid_w,
        radius=args.radius,
        maxlag=args.maxlag,
        p_threshold=args.p_threshold,
        top_k=args.top_k,
        ra_unit_01mm=not args.ra_already_mm,
        use_log1p=not args.no_log1p,
    )


if __name__ == "__main__":
    main()