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

p_thr = 0.5


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
            x = data[t:t + his_len, :, :-1]  # (his,1024,f_feature-1)
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
        num_feature=x.shape[3],  # feature-1
        seq_len=his_len,
        pre_len=pre_len,
        device=device,
        deep=1
    ).to(device)

    model_train = trainer(
        model=model,
        epoch=10,
        batch=8,
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
        fore_y_raw = torch.tensor(y, dtype=torch.float32)  # 原始：0.1mm单位
        fore_y_mm = (fore_y_raw * 0.1).clamp_min(0.0)  # ✅ 真值：mm

        # 可选：如果你后面还需要“真值norm空间”，再算一份（不强制）
        y_log_min = float(np.load("y_log_min.npy"))
        y_log_max = float(np.load("y_log_max.npy"))
        y_log_range = y_log_max - y_log_min + 1e-8
        print("[DEBUG] y raw stats:", float(fore_y_raw.min()), float(fore_y_raw.mean()), float(fore_y_raw.max()))
        print("[DEBUG] y mm(stats after *0.1):", float(fore_y_mm.min()), float(fore_y_mm.mean()),
              float(fore_y_mm.max()))
        print("[DEBUG] check max should be ~1.99mm if raw max is 19.9 (0.1mm unit).")

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

    with torch.no_grad():
        # 模型输出：(B, N, pre)
        y_hat, logit_rain = model(hist_x)  # (B,N,pre)

        # 统一转成 (B,pre,N) 并搬到 CPU，避免设备混乱
        y_hat = y_hat.permute(0, 2, 1).contiguous().cpu()
        logit_rain = logit_rain.permute(0, 2, 1).contiguous().cpu()

        # 回归输出从 norm 空间反归一化到 mm
        y_hat_mm = torch.expm1(y_hat * y_log_range + y_log_min)
        y_hat_mm = torch.clamp(y_hat_mm, min=0.0)

        # 分类头得到事件预测
        p_rain = torch.sigmoid(logit_rain)
        # 最终降水（门控）
        event = (p_rain >= p_thr).float()
        outputs = y_hat_mm * event

        fore_y_mm = fore_y_mm.cpu()
        mask = torch.ones_like(fore_y_mm, dtype=torch.bool)
        # 或者：mask = (fore_y >= 0)

        # ========== 整体指标 ==========
        rmse = measure.GetRMSE(outputs[mask], fore_y_mm[mask])
        mae = measure.GetMAP(outputs[mask], fore_y_mm[mask])
        print("整体RMSE:", rmse)
        print("整体MAE:", mae)

        event_true = (fore_y_mm >= 0.1)  # 真值是否下雨（mm口径）

        print("pred min/mean/max:", outputs.min().item(), outputs.mean().item(), outputs.max().item())
        print("true min/mean/max:", fore_y_mm.min().item(), fore_y_mm.mean().item(), fore_y_mm.max().item())

        print("=== CLS metrics (from p_rain only) ===")
        for p_thr_try in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]:
            event_pred = (p_rain >= p_thr_try)

            TP = (event_pred & event_true).sum().item()
            FP = (event_pred & (~event_true)).sum().item()
            FN = ((~event_pred) & event_true).sum().item()

            POD = TP / (TP + FN + 1e-6)
            FAR = FP / (TP + FP + 1e-6)
            CSI = TP / (TP + FP + FN + 1e-6)

            print(
                f"[CLS thr=0.1] p_thr={p_thr_try:.2f} "
                f"POD={POD:.4f} FAR={FAR:.4f} CSI={CSI:.4f}"
            )

        # ========== 分步评估 ==========
        for i in range(pre_step):
            print('step ', i)
            to = outputs[:, i, :]
            to_reg = y_hat_mm[:, i, :]
            print("to stats:", float(to.min()), float(to.mean()), float(to.max()))
            print("to_reg stats:", float(to_reg.min()), float(to_reg.mean()), float(to_reg.max()))
            ty = fore_y_mm[:, i, :]
            m = ty >= 0

            rmse = measure.GetRMSE(to[m], ty[m])
            mae = measure.GetMAP(to[m], ty[m])
            mape = measure.GetMAPE(to[m], ty[m])
            r2 = measure.R_square(ty[m].cpu(), to[m].cpu())  # R_square 用 numpy，确保在 cpu

            print("RMSE:", rmse)
            print("MAE:", mae)
            print("MAPE:", mape)
            print("R2:", r2)

            p_step = p_rain[:, i, :]

            # 放在 for thr 循环外，step i 内（ty/ m 已经有了）
            valid_ty = ty[m]
            print("ty stats:", float(valid_ty.min()), float(valid_ty.mean()), float(valid_ty.max()))
            print("count 0.1<=ty<1.0:", int(((valid_ty >= 0.1) & (valid_ty < 1.0)).sum().item()))
            print("count ty>=0.1:", int((valid_ty >= 0.1).sum().item()))
            print("count ty>=1.0:", int((valid_ty >= 1.0).sum().item()))
            print("count ty>=5.0:", int((valid_ty >= 5.0).sum().item()))

            for thr in [0.1, 1.0, 5.0]:
                y_event = (ty >= thr)
                pred_event = (to >= thr)
                # 只在 m 的有效位置统计
                yy = y_event[m]
                pp = pred_event[m]

                TP = (pp & yy).sum().item()
                FP = (pp & (~yy)).sum().item()
                FN = ((~pp) & yy).sum().item()

                pod = TP / (TP + FN + 1e-9)
                far = FP / (TP + FP + 1e-9)
                csi = TP / (TP + FP + FN + 1e-9)

                print(f"[thr={thr}] p_thr={p_thr:.2f} POD={pod:.4f} FAR={far:.4f} CSI={csi:.4f}")
