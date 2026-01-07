#!/usr/bin/env python
# -*- coding:utf-8 -*-

import argparse
import numpy as np
import torch

from model import STDGN
from trainer import trainer
from tester import test


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
    parser.add_argument('--feature', type=int, default=9, help='features used in prediction')
    parser.add_argument('--nodes', type=int, default=184, help='number of nodes')
    parser.add_argument('--units', type=int, default=128, help='hidden units')
    parser.add_argument('--his_len', type=int, default=16, help='input sequence length')
    parser.add_argument('--pre_len', type=int, default=8, help='output sequence length')

    # graph files (we ONLY load existing adj set)
    parser.add_argument('--adj_Set', type=str, default='data/AdjSet(GE).npy',
                        help='adjacency set file (K,N,N). We use the first one.')
    parser.add_argument('--adj_idx', type=int, default=0, help='which adj in adj_Set to use')

    # training hyper-parameters
    parser.add_argument('--batch_size', type=int, default=200, help='batch size')
    parser.add_argument('--train', type=float, default=0.8, help='ratio of training set')
    parser.add_argument('--lr', type=float, default=2e-3, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay rate')
    parser.add_argument('--epochs', type=int, default=100, help='epochs')
    parser.add_argument('--optim', type=str, default='adam')
    parser.add_argument('--device', type=str, default='cuda:0', help='cuda:0 / cpu')

    args = parser.parse_args()
    device = get_device(args.device)

    # ---- load dataset ----
    dataset = np.load(args.data)
    if dataset.ndim != 3:
        raise ValueError(f"DataSet.npy must be 3D (T,N,F), got shape={dataset.shape}")

    # original code uses first 5848 time steps
    x = dataset[:5848, :, :]
    y = dataset[:5848, :, -1]

    # basic sanity checks
    if x.shape[1] != args.nodes:
        raise ValueError(f"nodes mismatch: DataSet N={x.shape[1]} but args.nodes={args.nodes}")
    if x.shape[2] != args.feature:
        raise ValueError(f"feature mismatch: DataSet F={x.shape[2]} but args.feature={args.feature}")

    # ---- load adjacency ----
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
