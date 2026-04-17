#!/usr/bin/env python
# -*- coding:utf-8 -*-
import torch
import argparse
import numpy as np
from model_cnn_ablation import STDGN, STDGN_woa
from graph_evolution import Evolution, print_genetation
from trainer_cnn import trainer
from tester_cnn import test
import glob, os
from make_index_and_station_pro import generateFile

# -------------------- argparse --------------------
parser = argparse.ArgumentParser(description='PyTorch Time series forecasting')

parser.add_argument('--save', type=str, default='data/model_knn_ablation_complete.pkl')

# 原始超参数
parser.add_argument('--depth', type=int, default=1)
parser.add_argument('--heads', type=int, default=2)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--feature', type=int, default=37)
parser.add_argument('--nodes', type=int, default=1024)
parser.add_argument('--his_len', type=int, default=4)
parser.add_argument('--pre_len', type=int, default=3)
parser.add_argument('--units', type=int, default=64)
parser.add_argument('--adj_Set', type=str, default='data/AdjSet(GE).npy')

# 消融实验控制
parser.add_argument('--adj_mode', type=str, default='ge', choices=['knn', 'ge', 'identity'])
parser.add_argument('--knn_adj_path', type=str, default='data/cnn_spatial_align_slide/adj_knn.npy')
parser.add_argument('--use_attention', type=bool, default=True)
parser.add_argument('--graph_cell', type=str, default='gat', choices=['dgn', 'gcn', 'gat'])

# 训练超参数
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--train', type=float, default=0.8)
parser.add_argument('--lr', type=float, default=2e-3)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--optim', type=str, default='adam')
parser.add_argument('--device', type=str, default='cuda:0')

# Graph Evolution 原始超参数
parser.add_argument('--num_CUDA', type=int, default=1)
parser.add_argument('--mutation_pro', type=float, default=0.1)
parser.add_argument('--cross_pro', type=float, default=0.8)
parser.add_argument('--k_min', type=int, default=6)
parser.add_argument('--k_max', type=int, default=15)
parser.add_argument('--dis_thres', type=int, default=3)
parser.add_argument('--alt_thres', type=int, default=1200)
parser.add_argument('--iter', type=int, default=1)
parser.add_argument('--tournament', type=int, default=5)
parser.add_argument('--pareto', type=str, default='data/Pareto(GE)')

# 输入模式
parser.add_argument('--input_mode', type=str, default='sample')
parser.add_argument('--sample_dir', type=str, default='../gestdgn/data/cnn_spatial_align_slide')
parser.add_argument('--patch_dir', type=str, default='../gestdgn/data/patch')

args = parser.parse_args()
device = torch.device(args.device if torch.cuda.is_available() else "cpu")


# -------------------- 数据处理 --------------------
def build_samples_from_patches(patch_files, his_len, pre_len, nodes, feature):
    X_list, Y_list = [], []
    for p in patch_files:
        data = np.load(p)
        if data.ndim != 3: continue
        if data.shape[1] == nodes:
            T, N, F = data.shape
        elif data.shape[0] == nodes:
            data = np.transpose(data, (1, 0, 2)); T, N, F = data.shape
        else:
            continue
        if T < his_len + pre_len: continue
        for t in range(T - his_len - pre_len + 1):
            x = data[t:t + his_len, :, :feature]
            y = data[t + his_len:t + his_len + pre_len, :, -1]
            X_list.append(x)
            Y_list.append(y)
    return np.stack(X_list), np.stack(Y_list)


def _pick_key(data_dict, candidates, file_path):
    for k in candidates:
        if k in data_dict: return data_dict[k]
    raise KeyError(f"{os.path.basename(file_path)} 未找到 key: {candidates}")


def build_samples_from_raw_samples(sample_files, his_len, pre_len):
    X_nwp_list, X_radar_list, Y_list = [], [], []
    for p in sample_files:
        raw = np.load(p, allow_pickle=True)
        nwp = _pick_key(raw, ['nwp', 'NWP', 'x_nwp'], p)
        radar = _pick_key(raw, ['radar', 'RADAR', 'x_radar'], p)
        label = _pick_key(raw, ['label', 'LABEL', 'y', 'rain', 'target'], p)
        if label.ndim == 4 and label.shape[1] == 1: label = label[:, 0, :, :]
        T = nwp.shape[0]
        if T < his_len + pre_len: continue
        for t in range(T - his_len - pre_len + 1):
            x_nwp = nwp[t:t + his_len]
            x_radar = radar[t:t + his_len]
            y_label = label[t + his_len:t + his_len + pre_len]
            y_small = []
            for i in range(y_label.shape[0]):
                yi = y_label[i]
                sh, sw = yi.shape[0] // 32, yi.shape[1] // 32
                yi_32 = yi[:32 * sh, :32 * sw].reshape(32, sh, 32, sw).mean(axis=(1, 3))
                y_small.append(yi_32.reshape(-1))
            y_small = np.stack(y_small, axis=0)
            X_nwp_list.append(x_nwp)
            X_radar_list.append(x_radar)
            Y_list.append(y_small)
    return np.stack(X_nwp_list), np.stack(X_radar_list), np.stack(Y_list)


# -------------------- 邻接矩阵 --------------------
def build_adj(args):
    if args.adj_mode == 'knn':
        adj = np.load(args.knn_adj_path).astype(np.float32)
        np.fill_diagonal(adj, 1.0)
        return adj
    elif args.adj_mode == 'identity':
        return np.eye(args.nodes, dtype=np.float32)
    elif args.adj_mode == 'ge':
        other_adj = []
        pop_num = 2 + args.k_max - args.k_min + 1
        E = Evolution(pop=pop_num, iter=args.iter, cross_pro=args.cross_pro, mutation_pro=args.mutation_pro,
                      index_group=[args.k_min, args.k_max], his_len=args.his_len, pre_len=args.pre_len)
        E.register_on_new_generation(print_genetation)
        front = E.evolve(other_adj=other_adj)
        np.save(args.pareto, E.history)
        best = E.acc_best(front)
        adj_set = []
        for b in best:
            adj_set.append(b.index_to_matrix() + np.eye(b.matrix_size, dtype=np.float32))
        np.save(args.adj_Set, adj_set)
        return adj_set[0]
    else:
        raise ValueError(f"Unknown adj_mode: {args.adj_mode}")


# -------------------- 主程序 --------------------
if __name__ == "__main__":
    # 数据加载
    if args.input_mode == 'patch':
        patch_files = sorted(glob.glob(os.path.join(args.patch_dir, 'patch_*.npy')))
        x, y = build_samples_from_patches(patch_files, args.his_len, args.pre_len, args.nodes, args.feature)
        x_radar = None
    else:
        sample_files = sorted(glob.glob(os.path.join(args.sample_dir, '*.npz')))
        x, x_radar, y = build_samples_from_raw_samples(sample_files, args.his_len, args.pre_len)

    # 构建邻接矩阵
    adj_np = build_adj(args)
    adj = torch.tensor(adj_np, dtype=torch.float32)

    # -------------------- 模型选择 --------------------
    if args.graph_cell == 'dgn':
        graph_cell_type = 'dgn'
    elif args.graph_cell == 'gcn':
        graph_cell_type = 'gcn'
    elif args.graph_cell == 'gat':
        graph_cell_type = 'gat'
    else:
        raise ValueError(f"Unknown graph_cell: {args.graph_cell}")

    model = STDGN(num_units=args.units, adj=adj, num_nodes=args.nodes, num_feature=args.feature,
                  num_head=args.heads, seq_len=args.his_len, pre_len=args.pre_len,
                  device=device, deep=args.depth, input_mode=args.input_mode,
                  graph_cell=graph_cell_type, use_attention=args.use_attention)

    # -------------------- trainer / test --------------------
    if args.input_mode == 'patch':
        model_trainer = trainer(model=model, epoch=args.epochs, batch=args.batch_size, lr=args.lr,
                                decay=args.weight_decay, his_step=args.his_len, pre_step=args.pre_len,
                                x=x, y=y, train=args.train, valid=None, device=device)
        _, model = model_trainer.train()
        test(model, his_step=args.his_len, pre_step=args.pre_len, x=x, y=y)
    else:
        model_trainer = trainer(model=model, epoch=args.epochs, batch=args.batch_size, lr=args.lr,
                                decay=args.weight_decay, his_step=args.his_len, pre_step=args.pre_len,
                                x=x, radar=x_radar, y=y, train=args.train, valid=None, device=device)
        _, model = model_trainer.train()
        test(model, his_step=args.his_len, pre_step=args.pre_len, x=x, y=y, radar=x_radar)
