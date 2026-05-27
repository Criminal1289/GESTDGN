#!/usr/bin/env python
# -*- coding:utf-8 -*-
import torch
import argparse
import numpy as np
from model_cnn_tucancha_featureAttention import STDGN
from graph_evolution import Evolution, print_genetation
from trainer_cnn import trainer
from tester_cnn import test
import glob, os
from make_index_and_station_pro import generateFile

parser = argparse.ArgumentParser(description='PyTorch Time series forecasting')
parser.add_argument('--save', type=str, default='data/model.pkl',
                    help='path to save the final model')

# hyper-parameter of model
parser.add_argument('--depth', type=int, default=1, help='denseGNN depth')
parser.add_argument('--heads', type=int, default=2, help='heads of multi-head attention')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
parser.add_argument('--feature', type=int, default=37, help='features used in prediction')
parser.add_argument('--nodes', type=int, default=1024, help='number of nodes')
parser.add_argument('--his_len', type=int, default=6, help='input sequence length')
parser.add_argument('--pre_len', type=int, default=4, help='output sequence length')
parser.add_argument('--units', type=int, default=64, help='hidden units')
parser.add_argument('--adj_Set', type=str, default='data/AdjSet(GE).npy', help='best two adjs by graph evolution')

# hyper-parameter of training
parser.add_argument('--batch_size', type=int, default=8, help='batch size')
parser.add_argument('--train', type=float, default=0.8, help='ratio of training set')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0, help='weight decay rate')
parser.add_argument('--epochs', type=int, default=100, help='epochs')
parser.add_argument('--optim', type=str, default='adam')
parser.add_argument('--device', type=str, default='cuda:0', help='CUDA id')

# hyper-parameter of graph evolution
parser.add_argument('--num_CUDA', type=int, default='1', help='number of CUDA used in graph evolution')
parser.add_argument('--mutation_pro', type=float, default=0.1, help='mutation probability')
parser.add_argument('--cross_pro', type=float, default=0.8, help='crossing probability')
parser.add_argument('--k_min', type=int, default=6, help='the minimum k in KNN')
parser.add_argument('--k_max', type=int, default=15, help='the maximum k in KNN')
parser.add_argument('--dis_thres', type=int, default=3, help='the threshold of distance')
parser.add_argument('--alt_thres', type=int, default=1200, help='the threshold of altitude')
parser.add_argument('--iter', type=int, default=1, help='generation of graph evolution')
parser.add_argument('--tournament', type=int, default=5, help='number of selection of evolutionary tournament')
parser.add_argument('--pareto', type=str, default='data/Pareto(GE)', help='pareto front of no dominant')

parser.add_argument('--input_mode', type=str, default='sample', help='patch or sample(CNN)')
parser.add_argument('--sample_dir', type=str, default='../gestdgn/data/cnn_spatial_align_slide',
                    help='raw sample npz dir')
parser.add_argument('--patch_dir', type=str, default='../gestdgn/data/patch', help='patch dir')

parser.add_argument('--causal_adj',type=str,default='causal/A_cau_ra.npy',help='RA Granger causal adjacency matrix')
parser.add_argument('--causal_beta',type=float,default=0.1,help='weight of causal adjacency')

args = parser.parse_args()
device = torch.device(args.device)


def build_samples_from_patches(patch_files, his_len, pre_len, nodes, feature):
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

        # 如果第二维是 nodes，认为是 (T,N,F)
        # 如果第一维是 nodes，认为是 (N,T,F)，需要转成 (T,N,F)
        if shape[1] == nodes:
            T, N, F = shape
        elif shape[0] == nodes:
            data = np.transpose(data, (1, 0, 2))  # (T,N,F)
            T, N, F = data.shape
            print(f"[TRANSPOSE] {os.path.basename(p)} {shape} -> {data.shape}")
        else:
            print(f"[UNKNOWN layout] {p} shape={shape} (no dim == {nodes})")
            continue

        min_T = min(min_T, T)
        max_T = max(max_T, T)

        if T < his_len + pre_len:
            print(f"[SKIP short] {os.path.basename(p)} T={T} < {his_len + pre_len}")
            continue

        cnt = 0
        for t in range(T - his_len - pre_len + 1):
            x = data[t: t + his_len, :, :feature]  # (his,N,F)
            y = data[t + his_len: t + his_len + pre_len, :, -1]  # (pre,N)
            X_list.append(x)
            Y_list.append(y)
            cnt += 1

        ok_patches += 1
        print(f"[OK] {os.path.basename(p)} T={T} -> samples={cnt}")

    print(f"[STAT] patches={len(patch_files)} ok_patches={ok_patches} "
          f"samples={len(X_list)} T_range=[{min_T},{max_T}] his+pre={his_len + pre_len}")

    if len(X_list) == 0:
        raise RuntimeError("没有任何 patch 样本生成：要么所有 patch 的 T 都太短，要么 patch 维度顺序不对。")

    return np.stack(X_list), np.stack(Y_list)


def _pick_key(data_dict, candidates, file_path):
    for k in candidates:
        if k in data_dict:
            return data_dict[k]
    raise KeyError(f"{os.path.basename(file_path)} 中未找到这些 key: {candidates}，实际 keys={list(data_dict.keys())}")


def build_samples_from_raw_samples(sample_files, his_len, pre_len):
    """
    sample 模式下只做时间窗切分，不做 CNN。
    真正的 CNN 前端在 model.forward() 里执行，这样参数才能参与反传。

    约定每个 npz 至少包含:
        nwp   : [T, Nw, Cn, Hn, Wn]
        radar : [T, Cr, Hr, Wr]
        label : [T, H, W] 或 [T, 1, H, W]

    监督目标 y 直接取未来 pre_len 步的 label，
    先缩放到 32x32，再展平成 1024 个 node:
        y : [pre_len, 1024]
    """
    X_nwp_list, X_radar_list, X_label_list, Y_list = [], [], [], []

    min_T = 10 ** 9
    max_T = 0
    ok_files = 0

    for p in sample_files:
        raw = np.load(p, allow_pickle=True)

        nwp = _pick_key(raw, ['nwp', 'NWP', 'x_nwp'], p)
        radar = _pick_key(raw, ['radar', 'RADAR', 'x_radar'], p)
        label = _pick_key(raw, ['label', 'LABEL', 'y', 'rain', 'target'], p)

        if label.ndim == 4 and label.shape[1] == 1:
            label = label[:, 0, :, :]  # [T, H, W]

        if nwp.ndim != 5:
            raise ValueError(f"{os.path.basename(p)} nwp 期望 5 维 [T,Nw,C,H,W]，实际 {nwp.shape}")
        if radar.ndim != 4:
            raise ValueError(f"{os.path.basename(p)} radar 期望 4 维 [T,C,H,W]，实际 {radar.shape}")
        if label.ndim != 3:
            raise ValueError(f"{os.path.basename(p)} label 期望 3 维 [T,H,W]，实际 {label.shape}")

        T = nwp.shape[0]
        if radar.shape[0] != T or label.shape[0] != T:
            raise ValueError(
                f"{os.path.basename(p)} 时间长度不一致: "
                f"nwp={nwp.shape[0]}, radar={radar.shape[0]}, label={label.shape[0]}"
            )

        min_T = min(min_T, T)
        max_T = max(max_T, T)

        if T < his_len + pre_len:
            print(f"[SKIP short] {os.path.basename(p)} T={T} < {his_len + pre_len}")
            continue

        cnt = 0
        for t in range(T - his_len - pre_len + 1):
            x_nwp = nwp[t: t + his_len]  # [his, Nw, C, H, W]
            x_radar = radar[t: t + his_len]  # [his, C, H, W]
            y_label = label[t + his_len: t + his_len + pre_len]  # [pre, H, W]

            # 这里不做 CNN，只把 label 目标缩成 32x32 并展平为 1024
            # 纯 numpy 近邻下采样，避免在 main 引入 torch 图计算
            y_small = []
            for i in range(y_label.shape[0]):
                yi = y_label[i]
                h, w = yi.shape
                sh = h // 32
                sw = w // 32
                yi_32 = yi[:32 * sh, :32 * sw].reshape(32, sh, 32, sw).mean(axis=(1, 3))
                y_small.append(yi_32.reshape(-1))  # [1024]
            y_small = np.stack(y_small, axis=0)  # [pre, 1024]

            X_nwp_list.append(x_nwp)
            X_radar_list.append(x_radar)
            Y_list.append(y_small)
            cnt += 1

        ok_files += 1
        print(f"[OK] {os.path.basename(p)} T={T} -> samples={cnt}")

    print(f"[STAT] sample_files={len(sample_files)} ok_files={ok_files} "
          f"samples={len(X_nwp_list)} T_range=[{min_T},{max_T}] his+pre={his_len + pre_len}")

    if len(X_nwp_list) == 0:
        raise RuntimeError("没有任何 raw sample 样本生成：请检查 sample_dir、文件长度和 npz key。")

    x_nwp = np.stack(X_nwp_list)
    x_radar = np.stack(X_radar_list)
    y = np.stack(Y_list)

    return x_nwp, x_radar, y


if __name__ == "__main__":
    #generateFile(args.k_min, args.k_max)

    # -------------------- 数据入口 --------------------
    if args.input_mode == "patch":
        patch_files = sorted(glob.glob(os.path.join(args.patch_dir, "patch_*.npy")))
        if len(patch_files) == 0:
            raise RuntimeError(f"patch_dir 下没有找到 patch_*.npy: {args.patch_dir}")

        x, y = build_samples_from_patches(
            patch_files,
            his_len=args.his_len,
            pre_len=args.pre_len,
            nodes=args.nodes,
            feature=args.feature
        )

        x_radar = None


    elif args.input_mode == "sample":
        sample_files = sorted(glob.glob(os.path.join(args.sample_dir, "*.npz")))
        if len(sample_files) == 0:
            raise RuntimeError(f"sample_dir 下没有找到 *.npz: {args.sample_dir}")

        x, x_radar, y = build_samples_from_raw_samples(
            sample_files,
            his_len=args.his_len,
            pre_len=args.pre_len
        )
    else:
        raise ValueError(f"Unknown input_mode: {args.input_mode}")

    # -------------------- graph evolution --------------------
    other_adj = []
    pop_num = 2 + args.k_max - args.k_min + 1
    E = Evolution(pop=pop_num, iter=args.iter, cross_pro=args.cross_pro, mutation_pro=args.mutation_pro,
                  index_group=[args.k_min, args.k_max], his_len=args.his_len,
                  pre_len=args.pre_len)
    E.register_on_new_generation(print_genetation)
    adj_set = []
    front = E.evolve(other_adj=other_adj)
    np.save(args.pareto, E.history)
    best = E.acc_best(front)
    print('--------------------------------best--------------------------------')
    model_par = 0
    for b in best:
        adj = b.index_to_matrix() + np.eye(b.matrix_size)
        model_par += 1
        adj_set.append(adj)
    np.save(args.adj_Set, adj_set)

    # -------------------- model --------------------
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    # 控制是否使用因果图
    causal = False
    if causal:
        def row_normalize(A):
            row_sum = A.sum(axis=1, keepdims=True)
            return A / (row_sum + 1e-8)


        A_ge = np.load(args.adj_Set)[0].astype(np.float32)
        A_cau = np.load(args.causal_adj).astype(np.float32)
        if A_ge.shape != A_cau.shape:
            raise ValueError(f"A_ge shape {A_ge.shape} != A_cau shape {A_cau.shape}")
        A_ge = row_normalize(A_ge)
        A_cau = row_normalize(A_cau)
        A_final = A_ge + args.causal_beta * A_cau
        A_final = row_normalize(A_final)
        adj = torch.tensor(A_final, dtype=torch.float32)
    else:
        adj = torch.tensor(np.load(args.adj_Set)[0], dtype=torch.float32)

    model = STDGN(
        num_units=args.units,
        adj=adj,
        num_nodes=args.nodes,
        num_feature=args.feature,
        num_head=args.heads,
        seq_len=args.his_len,
        pre_len=args.pre_len,
        device=device,
        deep=args.depth,
        input_mode=args.input_mode
    )

    # -------------------- trainer / tester --------------------
    if args.input_mode == "patch":
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
        test(model, his_step=args.his_len, pre_step=args.pre_len, x=x, y=y)

    else:
        # 注意：
        # 这里只把 main 的 sample 入口改出来。
        # 前提是你的 trainer/tester 已经支持 sample 分支参数：
        # x = nwp, radar = x_radar, label = x_label, y = y
        model_trainer = trainer(
            model=model,
            epoch=args.epochs,
            batch=args.batch_size,
            lr=args.lr,
            decay=args.weight_decay,
            his_step=args.his_len,
            pre_step=args.pre_len,
            x=x,  # nwp
            radar=x_radar,  # sample 模式新增
            y=y,
            train=args.train,
            valid=None,
            device=device
        )

        _, model = model_trainer.train()
        test(
            model,
            his_step=args.his_len,
            pre_step=args.pre_len,
            x=x,
            y=y,
            radar=x_radar
        )
