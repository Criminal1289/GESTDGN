#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os
import argparse
import numpy as np
import torch
import glob
from model import STDGN
from trainer import trainer
from tester import test


def build_samples_from_patches(patch_files, his_len, pre_len):
    X_list, Y_list = [], []

    min_T = 10 ** 9
    max_T = 0
    ok_patches = 0

    for p in patch_files:
        data = np.load(p)
        shape = data.shape

        # 允许两种： (T,N,F) 或 (N,T,F) —— 自动纠正
        if data.ndim != 3:
            print(f"[BAD ndim] {p} shape={shape}")
            continue

        # 如果第二维是 184，基本就是 (T,184,F)
        # 如果第一维是 184，基本就是 (184,T,F)，需要转成 (T,184,F)
        if shape[1] == 184:
            T, N, F = shape
        elif shape[0] == 184:
            data = np.transpose(data, (1, 0, 2))  # (T,184,F)
            T, N, F = data.shape
            print(f"[TRANSPOSE] {os.path.basename(p)} {shape} -> {data.shape}")
        else:
            print(f"[UNKNOWN layout] {p} shape={shape} (no dim == 184)")
            continue

        min_T = min(min_T, T)
        max_T = max(max_T, T)

        if T < his_len + pre_len:
            print(f"[SKIP short] {os.path.basename(p)} T={T} < {his_len + pre_len}")
            continue

        cnt = 0
        for t in range(T - his_len - pre_len + 1):
            x = data[t: t + his_len, :, :27]  # (his,N,F)
            y = data[t + his_len: t + his_len + pre_len, :, -1]  # (pre,N)
            X_list.append(x)
            Y_list.append(y)
            cnt += 1

        ok_patches += 1
        print(f"[OK] {os.path.basename(p)} T={T} -> samples={cnt}")

    print(f"[STAT] patches={len(patch_files)} ok_patches={ok_patches} "
          f"samples={len(X_list)} T_range=[{min_T},{max_T}] his+pre={his_len + pre_len}")

    if len(X_list) == 0:
        raise RuntimeError("没有任何样本生成：要么所有 patch 的 T 都太短，要么 patch 维度顺序不对。")

    return np.stack(X_list), np.stack(Y_list)


def get_device(device_str: str) -> torch.device:
    """Return a torch.device; fall back to cpu if cuda not available."""
    if "cuda" in device_str:
        return torch.device(device_str if torch.cuda.is_available() else "cpu")
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser(description='PyTorch Time series forecasting (SLIM: no Graph Evolution)')

    # data & model saving
    parser.add_argument('--data', type=str, default='data/DataSet.npy', help='location of the dataset file')
    parser.add_argument('--save', type=str, default='data/model.pkl', help='path to save the final model')

    # model hyper-parameters
    parser.add_argument('--depth', type=int, default=1, help='denseGNN depth')
    parser.add_argument('--heads', type=int, default=2, help='heads of multi-head attention')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--feature', type=int, default=27, help='features used in prediction')
    parser.add_argument('--nodes', type=int, default=184, help='number of nodes')
    parser.add_argument('--his_len', type=int, default=4, help='input sequence length')
    parser.add_argument('--pre_len', type=int, default=2, help='output sequence length')
    parser.add_argument('--units', type=int, default=64, help='hidden units')

    # graph files (we ONLY load existing adj set)
    parser.add_argument('--adj_Set', type=str, default='data/AdjSet(GE).npy',
                        help='adjacency set file (K,N,N). We use the first one.')
    parser.add_argument('--adj_idx', type=int, default=0, help='which adj in adj_Set to use')

    # training hyper-parameters
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--train', type=float, default=0.8, help='ratio of training set')
    parser.add_argument('--lr', type=float, default=2e-3, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay rate')
    parser.add_argument('--epochs', type=int, default=100, help='epochs')
    parser.add_argument('--optim', type=str, default='adam')
    parser.add_argument('--device', type=str, default='cuda:0', help='cuda:0 / cpu')
    parser.add_argument('--ablation', type=str, default='knn', help='baseline,knn,ge')

    args = parser.parse_args()
    device = get_device(args.device)

    # ---- load dataset ----
    patch_files = sorted(glob.glob(r"../gestdgn/data/patch/patch_*.npy"))
    print("Found patches:", len(patch_files))
    print("Example:", patch_files[:3])

    x, y = build_samples_from_patches(
        patch_files,
        his_len=args.his_len,
        pre_len=args.pre_len
    )

    print("Samples:", x.shape, y.shape)

    # basic sanity checks
    if x.shape[2] != args.nodes:
        raise ValueError(f"nodes mismatch: DataSet N={x.shape[2]} but args.nodes={args.nodes}")
    if x.shape[3] != args.feature:
        raise ValueError(f"feature mismatch: DataSet F={x.shape[3]} but args.feature={args.feature}")

    # ---- load adjacency ----
    if args.ablation == "ge":
        adj_set = np.load(args.adj_Set)
    elif args.ablation == "knn":
        adj_set = np.load(r"../gestdgn/data/adj_knn.npy")
        adj_set = adj_set[None, :, :]
    else:
        adj_set = np.load(args.adj_Set)

    if adj_set.ndim != 3:
        raise ValueError(f"AdjSet must be 3D (K,N,N), got shape={adj_set.shape}")

    if not (0 <= args.adj_idx < adj_set.shape[0]):
        raise ValueError(f"adj_idx out of range: {args.adj_idx}, but K={adj_set.shape[0]}")

    adj_np = adj_set[args.adj_idx]
    if adj_np.shape[0] != args.nodes or adj_np.shape[1] != args.nodes:
        raise ValueError(f"adj size mismatch: adj is {adj_np.shape}, expected ({args.nodes},{args.nodes})")

    adj = torch.tensor(adj_np, dtype=torch.float32, device=device)

    # ---- build model ----
    model = STDGN(
        num_units=args.units,
        adj=adj,
        num_nodes=args.nodes,
        num_feature=args.feature,
        num_head=args.heads,
        seq_len=args.his_len,
        pre_len=args.pre_len,
        device=device,
        deep=args.depth
    ).to(device)

    # ---- train ----
    model_trainer = trainer(
        model=model,
        epoch=args.epochs,
        batch=args.batch_size,
        lr=args.lr,
        decay=args.weight_decay,
        his_step=args.his_len,
        pre_step=args.pre_len,
        x=x,
        y=y,
        train=args.train,
        valid=None,
        device=device
    )

    _, model = model_trainer.train()

    # ---- test ----
    test(model, his_step=args.his_len, pre_step=args.pre_len, x=x, y=y)


if __name__ == "__main__":
    main()
