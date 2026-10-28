#!/usr/bin/env python
# -*- coding:utf-8 -*-
import torch
import argparse
import hashlib
import json
import shutil
import numpy as np
from model_cnn_tucancha_featureAttention import STDGN
from graph_evolution import Evolution, print_genetation
from trainer_cnn import trainer, setup_seed
from tester_cnn import test
from precip_split import grouped_split_indices, window_metadata
from precip_inputs import MOISTURE_FEATURES, select_nwp_features
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
parser.add_argument('--spatial_decoder', choices=['none', 'multiscale'], default='multiscale',
                    help='32/16/8 spatial residual decoder; none runs the original regression head')
parser.add_argument('--decoder_width', type=int, default=32, help='channels in the spatial residual decoder')
parser.add_argument('--adj_Set', type=str, default='data/AdjSet(GE).npy', help='best two adjs by graph evolution')
parser.add_argument('--regenerate_adj', action='store_true',
                    help='rerun graph evolution and overwrite adj_Set; default reuses the existing fixed graph')

# hyper-parameter of training
parser.add_argument('--batch_size', type=int, default=8, help='batch size')
parser.add_argument('--train', type=float, default=0.8,
                    help='target training window fraction, approximated with whole event/day groups')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0, help='weight decay rate')
parser.add_argument('--epochs', type=int, default=100, help='epochs')
parser.add_argument('--optim', type=str, default='adam')
parser.add_argument('--device', type=str, default='cuda:0', help='CUDA id')
parser.add_argument('--seed', type=int, default=20, help='global random seed')
parser.add_argument('--loss_mode', choices=['mse_pm', 'legacy'], default='mse_pm',
                    help='mse_pm: only physical rainfall MSE + PM; legacy: previous weighted SmoothL1 and auxiliaries')
parser.add_argument('--mm_loss_weight', type=float, default=0.0,
                    help='legacy only: auxiliary mm-space SmoothL1 weight')
parser.add_argument('--mm_loss_scale', type=float, default=10.0,
                    help='fixed rainfall scale and SmoothL1 transition in mm')
parser.add_argument('--core_loss_weight', type=float, default=0.0,
                    help='core/peak/compactness loss weight; disabled in the two-term baseline')
parser.add_argument('--coverage_loss_weight', type=float, default=0.0,
                    help='7/15/40 mm spatial coverage loss weight; 0 disables it for comparison')
parser.add_argument('--pm_loss_weight', type=float, default=1.0,
                    help='PM coefficient; mse_pm with 0 runs pure physical rainfall MSE')
parser.add_argument('--pm_loss_scale', type=float, default=10.0,
                    help='shared physical rainfall scale for MSE and PM in mse_pm mode, in mm')
parser.add_argument('--mse_lead_weights', type=float, nargs='+', default=None,
                    help='mse_pm only: positive per-lead pointwise MSE weights, normalized by their sum; default equal')
parser.add_argument('--fal_loss_weight', type=float, default=0.0,
                    help='optional Fourier amplitude auxiliary weight (disabled by default)')
parser.add_argument('--fcl_loss_weight', type=float, default=0.0,
                    help='optional real Fourier correlation auxiliary weight (disabled by default)')
parser.add_argument('--spectral_loss_scale', type=float, default=10.0,
                    help='fixed physical scale for additive FAL/FCL adaptation, in mm')
parser.add_argument('--split_mode', choices=['grouped', 'legacy'], default='grouped',
                    help='sample mode: group connected events/dates; legacy is for historical diagnostics')
parser.add_argument('--valid_ratio', type=float, default=0.1,
                    help='target validation window fraction with whole groups; train+valid must be <1')
parser.add_argument('--early_stopping_patience', type=int, default=0,
                    help='stop after this many epochs without a better selected validation score; default 0 trains the full epoch budget')
parser.add_argument('--checkpoint_metric', choices=['all','balanced'], default='balanced',
                    help='checkpoint selection: all RMSE or all RMSE + strong weight * true >=7mm RMSE')
parser.add_argument('--eval_checkpoint', choices=['last', 'best'], default='last',
                    help='weights used after training: last uses the final epoch, best restores validation-selected weights')
parser.add_argument('--strong_validation_weight', type=float, default=0.25,
                    help='fixed weight of >=7mm conditional RMSE in validation selection, not training loss')
parser.add_argument('--run_dir', type=str, default=None,
                    help='directory for this run normalization, split, checkpoint, metrics and figures')
parser.add_argument('--error_map_global_index', type=int, default=65,
                    help='fixed original raw-sample window for diagnostic maps; -1 uses the held-out test case')
parser.add_argument('--error_map_save_dir', type=str,
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fig_step_error_maps'),
                    help='overwrite STEP0_3_GT_PRED_ERROR_MAPS.png here after evaluation')

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
parser.add_argument('--input_sources', choices=['radar', 'radar_nwp', 'radar_aux'], default='radar',
                    help='sample inputs: radar only (default) or the original Radar+NWP concatenation')
parser.add_argument('--nwp_features', nargs='+', default=None,
                    help='radar_aux: NWP variable names; default PWAT Q850 RH700')
parser.add_argument('--nwp_fusion', choices=['mean', 'center'], default='mean',
                    help='mean: fuse all NWP windows; center: use window 0 only')
parser.add_argument('--sample_dir', type=str, default='../gestdgn/data/cnn_spatial_align_slide',
                    help='raw sample npz dir')
parser.add_argument('--patch_dir', type=str, default='../gestdgn/data/patch', help='patch dir')

parser.add_argument('--causal_adj',type=str,default='causal/A_cau_ra.npy',help='RA Granger causal adjacency matrix')
parser.add_argument('--causal_beta',type=float,default=0.1,help='weight of causal adjacency')

args = parser.parse_args()
if args.input_sources in ('radar', 'radar_aux') and args.input_mode != 'sample':
    parser.error('radar-only input requires --input_mode sample; patch runs use --input_sources radar_nwp')
if args.input_sources in ('radar', 'radar_aux') and args.regenerate_adj:
    parser.error('radar-only control reuses the fixed adjacency; do not regenerate it with NWP-based fitness')
if args.run_dir is None:
    name = {'radar': 'radar_only', 'radar_nwp': 'radar_nwp', 'radar_aux': 'radar_aux'}[args.input_sources]
    pm_tag = format(args.pm_loss_weight, 'g').replace('.', 'p')
    args.run_dir = f'runs/{name}_pm{pm_tag}'
if args.input_sources == 'radar_aux':
    args.nwp_features = args.nwp_features or list(MOISTURE_FEATURES)
    if len(set(args.nwp_features)) != len(args.nwp_features):
        parser.error('NWP features must be unique')
elif args.nwp_features is not None:
    parser.error('--nwp_features is only supported with --input_sources radar_aux')
if args.loss_mode == 'mse_pm' and any(getattr(args, name) != 0 for name in
        ('mm_loss_weight', 'core_loss_weight', 'coverage_loss_weight', 'fal_loss_weight', 'fcl_loss_weight')):
    parser.error('mse_pm uses only MSE + PM; auxiliary weights must be 0 (use --loss_mode legacy for the previous objective)')
setup_seed(args.seed)
print(f"[SEED] global seed fixed to {args.seed}")
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


def _clean_nwp_array(nwp, file_path):
    nwp = nwp.astype(np.float32, copy=True)
    invalid = (~np.isfinite(nwp)) | (nwp <= -9999.0)
    invalid_count = int(invalid.sum())
    if invalid_count > 0:
        # Keep missing values distinguishable until train-only normalization is fitted.
        nwp[invalid] = np.nan
        print(f"[NWP missing] {os.path.basename(file_path)} invalid={invalid_count}")
    return nwp


def build_samples_from_raw_samples(sample_files, his_len, pre_len, return_metadata=False):
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
    metadata = []

    min_T = 10 ** 9
    max_T = 0
    ok_files = 0

    for p in sample_files:
        raw = np.load(p, allow_pickle=True)

        radar = _pick_key(raw, ['radar', 'RADAR', 'x_radar'], p)
        label = _pick_key(raw, ['label', 'LABEL', 'y', 'rain', 'target'], p)
        radar_only = args.input_sources == 'radar'
        # Retain the trainer's two-input calling convention with a tiny dummy.
        # No NWP array is read, cleaned or transferred to the GPU in radar-only mode.
        nwp = (np.zeros((radar.shape[0], 1), dtype=np.float32) if radar_only else
               _clean_nwp_array(select_nwp_features(raw, args.nwp_features), p))
        if not radar_only and args.nwp_fusion == 'center':
            if 'slide_offsets' not in raw or not np.array_equal(raw['slide_offsets'][0], [0, 0]):
                raise ValueError(f'{p}: center fusion requires slide_offsets[0] == [0, 0]')

        if label.ndim == 4 and label.shape[1] == 1:
            label = label[:, 0, :, :]  # [T, H, W]

        if not radar_only and nwp.ndim != 5:
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

        # label[k] accumulates precipitation over (time[k], time[k] + 1 h],
        # so the label at the final history time is the first forecast hour.
        needed_T = his_len + pre_len - 1
        if T < needed_T:
            print(f"[SKIP short] {os.path.basename(p)} T={T} < {needed_T}")
            raw.close()
            continue

        if return_metadata:
            if 'cp_name' not in raw or 'time_keys' not in raw:
                raise ValueError(f'{p}: grouped split requires cp_name and time_keys metadata')
            event = raw['cp_name'].item()
            times = raw['time_keys'].tolist()
            if len(times) != T:
                raise ValueError(f'{p}: time_keys length must match the sample frames')

        cnt = 0
        for t in range(T - needed_T + 1):
            x_nwp = nwp[t: t + his_len]  # [his, Nw, C, H, W]
            x_radar = radar[t: t + his_len]  # [his, C, H, W]
            target_start = t + his_len - 1
            y_label = label[target_start: target_start + pre_len]  # [pre, H, W]

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
            if return_metadata:
                metadata.append(window_metadata(event, times, p, t, his_len, pre_len))
            cnt += 1

        ok_files += 1
        print(f"[OK] {os.path.basename(p)} T={T} -> samples={cnt}")
        raw.close()

    print(f"[STAT] sample_files={len(sample_files)} ok_files={ok_files} "
          f"samples={len(X_nwp_list)} T_range=[{min_T},{max_T}] needed_T={his_len + pre_len - 1}")

    if len(X_nwp_list) == 0:
        raise RuntimeError("没有任何 raw sample 样本生成：请检查 sample_dir、文件长度和 npz key。")

    x_nwp = np.stack(X_nwp_list)
    x_radar = np.stack(X_radar_list)
    y = np.stack(Y_list)

    if return_metadata:
        return x_nwp, x_radar, y, metadata
    return x_nwp, x_radar, y


def fixed_error_map_data(model_trainer, global_index, sample_metadata, run_dir):
    """Keep the original display case separate from held-out test evaluation."""
    if global_index == -1 or not model_trainer._is_sample_mode():
        return None
    if not 0 <= global_index < len(model_trainer.y):
        raise ValueError(f"error_map_global_index={global_index} is outside the loaded sample windows")
    partition = next(name for name in ('train', 'valid', 'test')
                     if global_index in model_trainer.split_indices[name])
    info = dict(global_index=int(global_index), partition=partition,
                purpose='fixed_case_diagnostic', included_in_test_scores=(partition == 'test'))
    if sample_metadata is not None:
        info.update(sample_metadata[global_index])
    with open(os.path.join(run_dir, 'training_summary.json'), encoding='utf-8') as stream:
        summary = json.load(stream)
    checkpoint = summary.get('evaluation_checkpoint', 'best' if summary.get('restored_best', True) else 'last')
    info['checkpoint'] = checkpoint
    info['input_sources'] = getattr(getattr(model_trainer, 'model', None), 'input_sources', 'radar_nwp')
    info['nwp_features'] = getattr(getattr(model_trainer, 'model', None), 'nwp_features', None)
    info['checkpoint_epoch'] = summary.get('evaluation_epoch',
        summary['best_epoch'] if checkpoint == 'best' else summary['epochs_run'])
    info['checkpoint_path'] = os.path.abspath(os.path.join(run_dir, 'checkpoints', checkpoint + '.pt'))
    print(f"[ERROR MAP] fixed global index={global_index}, partition={partition}; "
          "held-out scores are computed separately")
    return dict(x=model_trainer.x[global_index:global_index + 1],
                y=model_trainer.y[global_index:global_index + 1],
                radar=model_trainer.radar[global_index:global_index + 1], sample_info=info)


def publish_error_map(run_dir, save_dir):
    """Overwrite the requested display PNG while keeping run artifacts isolated."""
    filename = 'STEP0_3_GT_PRED_ERROR_MAPS.png'
    source = os.path.abspath(os.path.join(run_dir, 'fig_step_error_maps', filename))
    destination = os.path.abspath(os.path.join(save_dir, filename))
    if not os.path.isfile(source):
        raise FileNotFoundError('Evaluation did not generate the error map: ' + source)
    os.makedirs(save_dir, exist_ok=True)
    if os.path.normcase(source) != os.path.normcase(destination):
        shutil.copy2(source, destination)
    print('[ERROR MAP] updated: ' + destination)


if __name__ == "__main__":
    #generateFile(args.k_min, args.k_max)

    sample_metadata = None
    split_indices = None
    split_mode = args.split_mode
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
        split_mode = 'legacy'
        print('[SPLIT] patch files have no event/time metadata; using legacy window split')


    elif args.input_mode == "sample":
        sample_files = sorted(glob.glob(os.path.join(args.sample_dir, "*.npz")))
        if len(sample_files) == 0:
            raise RuntimeError(f"sample_dir 下没有找到 *.npz: {args.sample_dir}")

        if split_mode == 'grouped':
            x, x_radar, y, sample_metadata = build_samples_from_raw_samples(
                sample_files, his_len=args.his_len, pre_len=args.pre_len, return_metadata=True
            )
            split_indices = grouped_split_indices(
                sample_metadata, train=args.train, valid=args.valid_ratio, seed=args.seed
            )
        else:
            x, x_radar, y = build_samples_from_raw_samples(
                sample_files, his_len=args.his_len, pre_len=args.pre_len
            )
            print('[SPLIT] legacy diagnostic mode: independent event/day separation is not enforced')
    else:
        raise ValueError(f"Unknown input_mode: {args.input_mode}")

    if args.input_mode == 'sample' and args.error_map_global_index != -1:
        if not 0 <= args.error_map_global_index < len(y):
            raise ValueError(f"error_map_global_index={args.error_map_global_index} must be -1 or within [0, {len(y)})")

    # -------------------- fixed graph / graph evolution --------------------
    if args.regenerate_adj or not os.path.isfile(args.adj_Set):
        if args.input_sources in ('radar', 'radar_aux'):
            raise FileNotFoundError('Radar-only control requires the existing fixed adjacency: ' + args.adj_Set)
        reason = "requested" if args.regenerate_adj else "fixed graph file not found"
        print(f"[ADJ] running graph evolution: {reason}")
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
        for b in best:
            adj_set.append(b.index_to_matrix() + np.eye(b.matrix_size))
        np.save(args.adj_Set, np.asarray(adj_set))
    else:
        print(f"[ADJ] reusing fixed graph: {args.adj_Set}")

    fixed_adj_set = np.load(args.adj_Set)
    if fixed_adj_set.ndim == 2:
        fixed_adj_set = fixed_adj_set[None, ...]
    expected_shape = (args.nodes, args.nodes)
    if fixed_adj_set.ndim != 3 or fixed_adj_set.shape[1:] != expected_shape:
        raise ValueError(
            f"Invalid fixed adjacency shape {fixed_adj_set.shape}; expected [K, {args.nodes}, {args.nodes}]"
        )
    adj_fingerprint = hashlib.sha256(np.ascontiguousarray(fixed_adj_set).tobytes()).hexdigest()[:16]
    print(f"[ADJ] shape={fixed_adj_set.shape}, fingerprint={adj_fingerprint}")

    # -------------------- model --------------------
    # Decouple model initialization from random numbers consumed by graph evolution.
    setup_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    # 控制是否使用因果图
    causal = False
    if causal:
        def row_normalize(A):
            row_sum = A.sum(axis=1, keepdims=True)
            return A / (row_sum + 1e-8)


        A_ge = fixed_adj_set[0].astype(np.float32)
        A_cau = np.load(args.causal_adj).astype(np.float32)
        if A_ge.shape != A_cau.shape:
            raise ValueError(f"A_ge shape {A_ge.shape} != A_cau shape {A_cau.shape}")
        A_ge = row_normalize(A_ge)
        A_cau = row_normalize(A_cau)
        A_final = A_ge + args.causal_beta * A_cau
        A_final = row_normalize(A_final)
        adj = torch.tensor(A_final, dtype=torch.float32)
    else:
        adj = torch.tensor(fixed_adj_set[0], dtype=torch.float32)

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
        input_mode=args.input_mode,
        nwp_fusion=args.nwp_fusion,
        spatial_decoder=args.spatial_decoder,
        decoder_width=args.decoder_width,
        input_sources=args.input_sources, nwp_features=args.nwp_features
    )
    decoder_params = sum(p.numel() for p in model.spatial_decoder.parameters()) if model.spatial_decoder is not None else 0
    print(f"[MODEL] input_sources={args.input_sources}, spatial_decoder={args.spatial_decoder}, width={args.decoder_width}, "
          f"decoder_params={decoder_params}, total_params={sum(p.numel() for p in model.parameters())}")
    if args.input_sources == 'radar_aux':
        print(f'[INPUT] auxiliary NWP features={args.nwp_features}, radar width={model.frontend.out_dim}')

    os.makedirs(args.run_dir, exist_ok=True)
    with open(os.path.join(args.run_dir, 'run_config.json'), 'w', encoding='utf-8') as stream:
        json.dump(dict(vars(args), effective_split_mode=split_mode, adj_fingerprint=adj_fingerprint),
                  stream, ensure_ascii=False, indent=2)
    training_options = dict(
        loss_mode=args.loss_mode,
        mse_lead_weights=args.mse_lead_weights,
        pm_loss_weight=args.pm_loss_weight, pm_loss_scale=args.pm_loss_scale,
        fal_loss_weight=args.fal_loss_weight, fcl_loss_weight=args.fcl_loss_weight,
        spectral_loss_scale=args.spectral_loss_scale,
        split_indices=split_indices, sample_metadata=sample_metadata, split_mode=split_mode,
        valid_ratio=args.valid_ratio, log_dir=args.run_dir,
        checkpoint_dir=os.path.join(args.run_dir, 'checkpoints'),
        early_stopping_patience=args.early_stopping_patience, restore_best=(args.eval_checkpoint == 'best'),
        checkpoint_metric=args.checkpoint_metric, strong_validation_weight=args.strong_validation_weight
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
            device=device,
            seed=args.seed,
            mm_loss_weight=args.mm_loss_weight,
            mm_loss_scale=args.mm_loss_scale,
            coverage_loss_weight=args.coverage_loss_weight,
            core_loss_weight=args.core_loss_weight,
            **training_options
        )

        _, model = model_trainer.train()
        test(model, his_step=args.his_len, pre_step=args.pre_len,
             norm_dir=args.run_dir, output_dir=args.run_dir, **model_trainer.get_test_data())

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
            device=device,
            seed=args.seed,
            mm_loss_weight=args.mm_loss_weight,
            mm_loss_scale=args.mm_loss_scale,
            coverage_loss_weight=args.coverage_loss_weight,
            core_loss_weight=args.core_loss_weight,
            **training_options
        )

        _, model = model_trainer.train()
        test(
            model,
            his_step=args.his_len,
            pre_step=args.pre_len,
            norm_dir=args.run_dir,
            output_dir=args.run_dir,
            sample_indices=model_trainer.test_indices,
            sample_metadata=sample_metadata,
            error_map_data=fixed_error_map_data(model_trainer, args.error_map_global_index,
                                                sample_metadata, args.run_dir),
            **model_trainer.get_test_data()
        )

    publish_error_map(args.run_dir, args.error_map_save_dir)
