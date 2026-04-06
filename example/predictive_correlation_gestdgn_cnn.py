#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import glob
import random
from typing import Dict, List

import numpy as np
import torch

from model_cnn import STDGN
from trainer_cnn import trainer
from tester_cnn import test


# =========================================================
# 0. 配置
# =========================================================
SAMPLE_DIR = r"C:/Users/17878/Desktop/gestdgn/data/cnn_spatial_align_slide"
ADJ_PATH = r"C:/Users/17878/Desktop/gestdgn/data/AdjSet(GE).npy"
SAVE_DIR = r"./feature_corr_stdgn_cnn"

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# 与 main_cnn.py / trainer_cnn.py 一致的主要超参数
DEPTH = 1
HEADS = 2
FEATURE = 37
NODES = 1024
HIS_LEN = 4
PRE_LEN = 2
UNITS = 64

BATCH_SIZE = 8
TRAIN_RATIO = 0.8
LR = 2e-3
WEIGHT_DECAY = 0.0
EPOCHS = 100

RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

NWP_FEATURES = [
    "CAPE", "DVG200", "DVG850", "DVG925", "HT0", "HT10", "HTw0", "KI",
    "LCL", "LFC", "LI300", "LI500", "muLCL", "PE", "PWAT", "Q1000", "Q700",
    "Q850", "RH1000", "RH500", "RH700", "TdSfc850", "TTdMax74", "TTdMean74",
    "WS500", "WS700", "WS925"
]

RADAR_FEATURES = [
    "CR", "HBR", "ET", "VIL",
    "CAP20", "CAP30", "CAP40", "CAP50", "CAP60", "CAP70"
]

ALL_FEATURES = NWP_FEATURES + RADAR_FEATURES


# =========================================================
# 1. 工具
# =========================================================
def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def list_sample_files(sample_dir: str) -> List[str]:
    files = sorted(glob.glob(os.path.join(sample_dir, "*.npz")))
    if len(files) == 0:
        raise RuntimeError(f"sample_dir 下没有找到 *.npz: {sample_dir}")
    return files


def _pick_key(data_dict, candidates, file_path):
    for k in candidates:
        if k in data_dict:
            return data_dict[k]
    raise KeyError(
        f"{os.path.basename(file_path)} 中未找到这些 key: {candidates}，"
        f"实际 keys={list(data_dict.keys())}"
    )


# =========================================================
# 2. 按 main_cnn.py 风格构造 sample 样本
# =========================================================
def build_samples_from_raw_samples(sample_files, his_len, pre_len):
    """
    读取每个 sample_xxx.npz，切成时间窗。

    返回:
        x_nwp   : [B, his, Nw, Cn, H, W]
        x_radar : [B, his, Cr, Hr, Wr]
        y       : [B, pre, 1024]

    其中 y 会按 main_cnn.py 的思路从 128x128 下采样到 32x32，再展平成 1024 节点。
    """
    X_nwp_list, X_radar_list, Y_list = [], [], []

    min_T = 10**9
    max_T = 0
    ok_files = 0

    for p in sample_files:
        raw = np.load(p, allow_pickle=True)

        nwp = _pick_key(raw, ['nwp', 'NWP', 'x_nwp'], p)
        radar = _pick_key(raw, ['radar', 'RADAR', 'x_radar'], p)
        label = _pick_key(raw, ['label', 'LABEL', 'y', 'rain', 'target'], p)

        # label: [T,1,H,W] -> [T,H,W]
        if label.ndim == 4 and label.shape[1] == 1:
            label = label[:, 0, :, :]

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
            x_nwp = nwp[t:t + his_len]                     # [his, Nw, Cn, H, W]
            x_radar = radar[t:t + his_len]                # [his, Cr, H, W]
            y_label = label[t + his_len:t + his_len + pre_len]   # [pre, H, W]

            y_small = []
            for i in range(y_label.shape[0]):
                yi = y_label[i]
                h, w = yi.shape
                sh = h // 32
                sw = w // 32

                if sh == 0 or sw == 0:
                    raise ValueError(f"标签尺寸过小，无法下采样到32x32: {yi.shape}")

                yi_32 = yi[:32 * sh, :32 * sw].reshape(32, sh, 32, sw).mean(axis=(1, 3))
                y_small.append(yi_32.reshape(-1))         # [1024]

            y_small = np.stack(y_small, axis=0)           # [pre, 1024]

            X_nwp_list.append(x_nwp.astype(np.float32))
            X_radar_list.append(x_radar.astype(np.float32))
            Y_list.append(y_small.astype(np.float32))
            cnt += 1

        ok_files += 1
        print(f"[OK] {os.path.basename(p)} T={T} -> samples={cnt}")

    print(
        f"[STAT] sample_files={len(sample_files)} ok_files={ok_files} "
        f"samples={len(X_nwp_list)} T_range=[{min_T},{max_T}] his+pre={his_len + pre_len}"
    )

    if len(X_nwp_list) == 0:
        raise RuntimeError("没有任何 raw sample 样本生成：请检查 sample_dir、文件长度和 npz key。")

    x_nwp = np.stack(X_nwp_list).astype(np.float32)
    x_radar = np.stack(X_radar_list).astype(np.float32)
    y = np.stack(Y_list).astype(np.float32)

    return x_nwp, x_radar, y


# =========================================================
# 3. LOFO: 不删除通道，只置零屏蔽
# =========================================================
def mask_one_feature_from_sample_input(x_nwp: np.ndarray, x_radar: np.ndarray, feat_idx: int):
    """
    不删除通道，只把对应特征通道置零。
    这样可以兼容 cnn_frontend 的固定输入通道数。

    x_nwp   : [B, his, Nw, Cn, H, W]
    x_radar : [B, his, Cr, H, W]
    """
    x_nwp_new = x_nwp.copy()
    x_radar_new = x_radar.copy()

    if feat_idx < len(NWP_FEATURES):
        x_nwp_new[:, :, :, feat_idx, :, :] = 0.0
    else:
        ridx = feat_idx - len(NWP_FEATURES)
        x_radar_new[:, :, ridx, :, :] = 0.0

    return x_nwp_new, x_radar_new


# =========================================================
# 4. 建模 / 训练 / 测试
# =========================================================
def load_adj(adj_path: str, device: str):
    arr = np.load(adj_path)
    if arr.ndim == 3:
        arr = arr[0]
    elif arr.ndim != 2:
        raise ValueError(f"邻接矩阵维度不对: {arr.shape}")

    return torch.tensor(arr, dtype=torch.float32, device=torch.device(device))


def build_model(adj: torch.Tensor):
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")

    model = STDGN(
        num_units=UNITS,
        adj=adj,
        num_nodes=NODES,
        num_feature=FEATURE,   # 注意：这里保持固定 37，不随屏蔽变化
        num_head=HEADS,
        seq_len=HIS_LEN,
        pre_len=PRE_LEN,
        device=device,
        deep=DEPTH,
        input_mode="sample"
    )
    return model


def train_and_eval_once(x_nwp: np.ndarray, x_radar: np.ndarray, y: np.ndarray):
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    adj = load_adj(ADJ_PATH, DEVICE)

    model = build_model(adj=adj)

    model_trainer = trainer(
        model=model,
        epoch=EPOCHS,
        batch=BATCH_SIZE,
        lr=LR,
        decay=WEIGHT_DECAY,
        his_step=HIS_LEN,
        pre_step=PRE_LEN,
        x=x_nwp,
        radar=x_radar,
        y=y,
        train=TRAIN_RATIO,
        valid=None,
        device=device
    )

    # trainer_cnn.py 返回 rmse_reg_rain, model
    rmse_rain, model = model_trainer.train()

    print("\n[TEST DETAIL]")
    test(
        model,
        his_step=HIS_LEN,
        pre_step=PRE_LEN,
        x=x_nwp,
        y=y,
        radar=x_radar
    )

    return float(rmse_rain)


# =========================================================
# 5. 预测相关性分析
# =========================================================
def predictive_correlation_analysis():
    ensure_dir(SAVE_DIR)

    print("[INFO] loading raw sample npz files ...")
    sample_files = list_sample_files(SAMPLE_DIR)

    x_nwp, x_radar, y = build_samples_from_raw_samples(
        sample_files,
        his_len=HIS_LEN,
        pre_len=PRE_LEN
    )

    print(f"[INFO] x_nwp shape   = {x_nwp.shape}")
    print(f"[INFO] x_radar shape = {x_radar.shape}")
    print(f"[INFO] y shape       = {y.shape}")

    # baseline
    print("\n================ BASELINE: all features ================\n")
    baseline_rmse_rain = train_and_eval_once(
        x_nwp=x_nwp,
        x_radar=x_radar,
        y=y
    )

    results: Dict[str, Dict] = {}

    for i, feat in enumerate(ALL_FEATURES):
        print(f"\n================ LOFO: mask {feat} ================\n")

        x_nwp_mask, x_radar_mask = mask_one_feature_from_sample_input(x_nwp, x_radar, i)

        rmse_rain = train_and_eval_once(
            x_nwp=x_nwp_mask,
            x_radar=x_radar_mask,
            y=y
        )

        d_rmse_rain = rmse_rain - baseline_rmse_rain

        results[feat] = {
            "rmse_rain_without_feature": float(rmse_rain),
            "delta_RMSE_rain": float(d_rmse_rain),
            "PC_score": float(d_rmse_rain)
        }

        print(
            f"[RESULT] {feat:10s} | "
            f"RMSE_rain_wo={rmse_rain:.6f} | "
            f"dRMSE_rain={d_rmse_rain:.6f}"
        )

        save_predictive_results(
            baseline_rmse_rain=baseline_rmse_rain,
            results=results,
            out_path=os.path.join(SAVE_DIR, "predictive_correlation_results_stdgn_cnn.csv")
        )

    return baseline_rmse_rain, results


# =========================================================
# 6. 输出
# =========================================================
def print_predictive_ranking(results: Dict):
    print("\n================ Predictive Correlation Ranking (GESTDGN-CNN) ================")
    sorted_items = sorted(results.items(), key=lambda x: x[1]["PC_score"], reverse=True)
    for i, (feat, info) in enumerate(sorted_items, 1):
        print(
            f"{i:02d}. {feat:10s} | "
            f"PC={info['PC_score']:.6f} | "
            f"dRMSE_rain={info['delta_RMSE_rain']:.6f} | "
            f"RMSE_rain_wo={info['rmse_rain_without_feature']:.6f}"
        )


def save_predictive_results(baseline_rmse_rain: float, results: Dict, out_path: str):
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"baseline_rmse_rain,{baseline_rmse_rain:.8f}\n")
        f.write("feature,PC_score,delta_RMSE_rain,rmse_rain_without_feature\n")
        sorted_items = sorted(results.items(), key=lambda x: x[1]["PC_score"], reverse=True)
        for feat, info in sorted_items:
            f.write(
                f"{feat},"
                f"{info['PC_score']:.8f},"
                f"{info['delta_RMSE_rain']:.8f},"
                f"{info['rmse_rain_without_feature']:.8f}\n"
            )

    print(f"[INFO] predictive results saved to: {out_path}")


# =========================================================
# 7. 主函数
# =========================================================
def main():
    baseline_rmse_rain, results = predictive_correlation_analysis()
    print(f"\n[BASELINE RMSE_rain] {baseline_rmse_rain:.6f}")
    print_predictive_ranking(results)


if __name__ == "__main__":
    main()
