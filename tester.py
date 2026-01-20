#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import glob, os
import torch
import torch.nn as nn
import numpy as np
import time
from trainer import trainer, Divide, TrainValidTest
from model import STDGN_woa
import measure
from torch.utils.data import TensorDataset, Dataset, DataLoader
from torch.autograd import Variable

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
PATCH_GLOB = r"data/patch/patch_*.npy"  # 你当前patch目录（按实际改）


def build_samples_from_patches(patch_files, his_len, pre_len):
    X_list, Y_list = [], []
    for p in patch_files:
        data = np.load(p)  # (T,184,28)
        if data.ndim != 3:
            continue
        T, N, F = data.shape
        if T < his_len + pre_len:
            continue

        for t in range(T - his_len - pre_len + 1):
            x = data[t:t + his_len, :, :27]  # (his,184,27)
            y = data[t + his_len:t + his_len + pre_len, :, -1]  # (pre,184)
            X_list.append(x)
            Y_list.append(y)

    if len(X_list) == 0:
        raise RuntimeError("GE fitness_cal: no samples generated (his+pre too long or patches too short).")

    return np.stack(X_list), np.stack(Y_list)


def fitness_cal(adjmat, device="cuda:0", his_len=None, pre_len=None):
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    patch_files = sorted(glob.glob(PATCH_GLOB))
    x, y = build_samples_from_patches(patch_files, his_len, pre_len)

    adj = torch.tensor(adjmat, dtype=torch.float32, device=device)

    # 图进化评估
    model = STDGN_woa(
        num_units=64,
        adj=adj,
        num_nodes=x.shape[2],  # 184
        num_feature=x.shape[3],  # 27
        seq_len=his_len,
        pre_len=pre_len,
        device=device,
        deep=1
    ).to(device)

    model_train = trainer(
        model=model,
        epoch=2,  # GE评估用很小epoch即可（否则太慢）
        batch=64,
        lr=1e-3,
        decay=0,
        his_step=his_len,
        pre_step=pre_len,
        x=x,
        y=y,
        train=0.8,
        valid=None,
        device=device
    )
    rmse, _ = model_train.train()
    return rmse, model


def test(model, his_step, pre_step, x, y):
    model = model.to(device)
    model.eval()

    # ===============================
    # NEW: 兼容两种数据范式
    # ===============================
    if x.ndim == 4 and y.ndim == 3:
        # 新范式：x=(B,his,N,F), y=(B,pre,N)
        hist_x = torch.tensor(x, dtype=torch.float32).to(device)
        fore_y = torch.tensor(y, dtype=torch.float32)  # (B, pre, N)

        # ✅ 从训练阶段加载 y_min / y_max
        y_log_min = float(np.load("y_log_min.npy"))
        y_log_max = float(np.load("y_log_max.npy"))
        y_log_range = y_log_max - y_log_min + 1e-8
        print("loaded y_log_min/y_log_max:", y_log_min, y_log_max)


    else:
        # 旧范式：x=(T,N,F), y=(T,N)
        feature_num = x.shape[-1]
        nodes = x.shape[1]

        x_ones, x_min, x_max = measure.Norm(
            np.array(torch.tensor(x).view(-1, feature_num), dtype=float),
            dim=2
        )
        y_ones, y_min, y_max = measure.Norm(
            np.array(torch.tensor(y).view(-1), dtype=float),
            dim=1
        )

        source_x = torch.tensor(x_ones).view(-1, nodes, feature_num)
        source_y = torch.tensor(y_ones).view(-1, nodes)

        hist_x, fore_y = Divide(source_x, source_y, his_step, pre_step)
        hist_x = torch.tensor(hist_x, dtype=torch.float32).to(device)
        fore_y = torch.tensor(fore_y, dtype=torch.float32)

    with torch.no_grad():
        # 模型输出：(B, N, pre)
        outputs = model(hist_x)
        outputs = outputs.permute(0, 2, 1).contiguous().cpu()
        # 反归一化
        outputs = torch.expm1(outputs * y_log_range + y_log_min)
        outputs = torch.clamp(outputs, min=0.0)
        fore_y = fore_y.cpu()
        mask = fore_y > -8.5

        # ========== 整体指标 ==========
        rmse = measure.GetRMSE(outputs[mask], fore_y[mask])
        mae = measure.GetMAP(outputs[mask], fore_y[mask])
        print("整体RMSE:", rmse)
        print("整体MAE:", mae)

        fy = fore_y[mask]
        po = outputs[mask]

        for thr in [0.1, 1.0, 5.0, 10.0, 50.0]:
            y_event = (fy >= thr).float().mean().item()
            p_event = (po >= thr).float().mean().item()
            print(f"thr={thr}  y_event_ratio={y_event:.6f}  pred_event_ratio={p_event:.6f}")

        # ========== 分类指标 ==========
        for thr in [0.1, 1.0, 5.0, 10.0, 50.0]:
            pod = measure.GetPOD(fy, po, thr=thr)
            far = measure.GetFAR(fy, po, thr=thr)
            csi = measure.GetCSI(fy, po, thr=thr)
            print(f"[thr={thr}] POD={pod:.4f} FAR={far:.4f} CSI={csi:.4f}")

        # ========== 分步评估 ==========
        for i in range(pre_step):
            print('step ', i)
            to = outputs[:, i, :]
            ty = fore_y[:, i, :]
            m = ty > -8.5

            rmse = measure.GetRMSE(to[m], ty[m])
            mae = measure.GetMAP(to[m], ty[m])
            mape = measure.GetMAPE(to[m], ty[m])
            r2 = measure.R_square(ty[m].cpu(), to[m].cpu())  # R_square 用 numpy，确保在 cpu

            print("RMSE:", rmse)
            print("MAE:", mae)
            print("MAPE:", mape)
            print("R2:", r2)

            for thr in [0.1, 1.0, 5.0]:
                pod = measure.GetPOD(ty[m], to[m], thr=thr)
                far = measure.GetFAR(ty[m], to[m], thr=thr)
                csi = measure.GetCSI(ty[m], to[m], thr=thr)
                print(f"[thr={thr}] POD={pod:.4f} FAR={far:.4f} CSI={csi:.4f}")
