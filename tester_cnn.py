#!/usr/bin/env python
# -*- coding:utf-8 -*-
import glob
import os
import torch
import numpy as np

from trainer_cnn import trainer, Divide, TrainValidTest, TrainValidTestSample
from model_cnn import STDGN_woa
import measure

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
PATCH_GLOB = r"data/patch/patch_*.npy"

p_thr = 0.5


def build_samples_from_patches(patch_files, his_len, pre_len):
    X_list, Y_list = [], []
    for p in patch_files:
        data = np.load(p)
        if data.ndim != 3:
            continue

        T, N, F = data.shape
        if T < his_len + pre_len:
            continue

        for t in range(T - his_len - pre_len + 1):
            x = data[t:t + his_len, :, :-1]
            y = data[t + his_len:t + his_len + pre_len, :, -1]
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

    model = STDGN_woa(
        num_units=64,
        adj=adj,
        num_nodes=x.shape[2],
        num_feature=x.shape[3],
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


def _is_sample_mode(model):
    return getattr(model, "input_mode", "patch") == "sample"


def _load_y_norm_params():
    y_log_min = float(np.load("y_log_min.npy"))
    y_log_max = float(np.load("y_log_max.npy"))
    y_log_range = y_log_max - y_log_min + 1e-8
    return y_log_min, y_log_max, y_log_range


def _inv_y(y_norm, y_log_min, y_log_range):
    y_log = y_norm * y_log_range + y_log_min
    y_mm = torch.expm1(y_log)
    return torch.clamp(y_mm, min=0.0)


# def _safe_print_cls_metrics(p_rain, event_true):
#     print("=== CLS metrics (from p_rain only) ===")
#     for p_thr_try in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]:
#         event_pred = (p_rain >= p_thr_try)
#
#         TP = (event_pred & event_true).sum().item()
#         FP = (event_pred & (~event_true)).sum().item()
#         FN = ((~event_pred) & event_true).sum().item()
#
#         POD = TP / (TP + FN + 1e-6)
#         FAR = FP / (TP + FP + 1e-6)
#         CSI = TP / (TP + FP + FN + 1e-6)
#
#         print(
#             f"[CLS thr=0.1] p_thr={p_thr_try:.2f} "
#             f"POD={POD:.4f} FAR={FAR:.4f} CSI={CSI:.4f}"
#         )
def _safe_print_cls_metrics(p_rain, fore_y_mm):
    print("=== CLS metrics ===")

    # 降雨量阈值：0.1~5 中选几个典型值
    rain_thr_list = [0.1, 0.5, 1.0, 2.0, 5.0]

    # 固定概率阈值
    p_thr_list = [0.4, 0.5, 0.6]

    for rain_thr in rain_thr_list:
        event_true = (fore_y_mm >= rain_thr)

        print(f"\n--- Rainfall threshold = {rain_thr:.1f} mm ---")
        for p_thr_try in p_thr_list:
            event_pred = (p_rain >= p_thr_try)

            TP = (event_pred & event_true).sum().item()
            FP = (event_pred & (~event_true)).sum().item()
            FN = ((~event_pred) & event_true).sum().item()

            POD = TP / (TP + FN + 1e-6)
            FAR = FP / (TP + FP + 1e-6)
            CSI = TP / (TP + FP + FN + 1e-6)

            print(
                f"[rain_thr={rain_thr:.1f}mm, p_thr={p_thr_try:.1f}] "
                f"POD={POD:.4f} FAR={FAR:.4f} CSI={CSI:.4f}"
            )

def _eval_and_print(outputs, y_hat_mm, fore_y_mm, p_rain, pre_step):
    mask = torch.ones_like(fore_y_mm, dtype=torch.bool)

    rmse = measure.GetRMSE(outputs[mask], fore_y_mm[mask])
    mae = measure.GetMAP(outputs[mask], fore_y_mm[mask])
    print("整体RMSE:", rmse)
    print("整体MAE:", mae)

    # event_true = (fore_y_mm >= 0.1)

    print("pred min/mean/max:", outputs.min().item(), outputs.mean().item(), outputs.max().item())
    print("true min/mean/max:", fore_y_mm.min().item(), fore_y_mm.mean().item(), fore_y_mm.max().item())

    _safe_print_cls_metrics(p_rain, fore_y_mm)

    for i in range(pre_step):
        print('step ', i)

        to = outputs[:, i, :]
        to_reg = y_hat_mm[:, i, :]
        ty = fore_y_mm[:, i, :]
        m = ty >= 0

        print("to stats:", float(to.min()), float(to.mean()), float(to.max()))
        print("to_reg stats:", float(to_reg.min()), float(to_reg.mean()), float(to_reg.max()))

        rmse = measure.GetRMSE(to[m], ty[m])
        mae = measure.GetMAP(to[m], ty[m])
        mape = measure.GetMAPE(to[m], ty[m])
        r2 = measure.R_square(ty[m].cpu(), to[m].cpu())

        print("RMSE:", rmse)
        print("MAE:", mae)
        print("MAPE:", mape)
        print("R2:", r2)

        valid_ty = ty[m]
        print("ty stats:", float(valid_ty.min()), float(valid_ty.mean()), float(valid_ty.max()))
        print("count 0.1<=ty<1.0:", int(((valid_ty >= 0.1) & (valid_ty < 1.0)).sum().item()))
        print("count ty>=0.1:", int((valid_ty >= 0.1).sum().item()))
        print("count ty>=1.0:", int((valid_ty >= 1.0).sum().item()))
        print("count ty>=5.0:", int((valid_ty >= 5.0).sum().item()))

        for thr in [0.1, 1.0, 5.0]:
            y_event = (ty >= thr)
            pred_event = (to >= thr)

            yy = y_event[m]
            pp = pred_event[m]

            TP = (pp & yy).sum().item()
            FP = (pp & (~yy)).sum().item()
            FN = ((~pp) & yy).sum().item()

            pod = TP / (TP + FN + 1e-9)
            far = FP / (TP + FP + 1e-9)
            csi = TP / (TP + FP + FN + 1e-9)

            print(f"[thr={thr}] p_thr={p_thr:.2f} POD={pod:.4f} FAR={far:.4f} CSI={csi:.4f}")


def test(model, his_step, pre_step, x, y, radar=None):
    model = model.to(device)
    model.eval()

    sample_mode = _is_sample_mode(model)
    y_log_min, _, y_log_range = _load_y_norm_params()

    with torch.no_grad():
        if not sample_mode:
            # ===============================
            # patch 模式
            # ===============================
            if x.ndim == 4 and y.ndim == 3:
                hist_x = torch.tensor(x, dtype=torch.float32).to(device)

                # 原始真值单位: 0.1mm
                fore_y_raw = torch.tensor(y, dtype=torch.float32)
                fore_y_mm = (fore_y_raw * 0.1).clamp_min(0.0)

                print("[DEBUG] y raw stats:",
                      float(fore_y_raw.min()), float(fore_y_raw.mean()), float(fore_y_raw.max()))
                print("[DEBUG] y mm(stats after *0.1):",
                      float(fore_y_mm.min()), float(fore_y_mm.mean()), float(fore_y_mm.max()))
            else:
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

                # 旧范式下直接按训练保存的反归一化参数回到 mm
                fore_y = torch.tensor(fore_y, dtype=torch.float32)
                fore_y_mm = _inv_y(fore_y, y_log_min, y_log_range)

            y_hat, logit_rain = model(hist_x)

        else:
            # ===============================
            # sample 模式
            # ===============================
            if radar is None:
                raise ValueError("sample 模式下 test 需要额外传入 radar")

            hist_x = torch.tensor(x, dtype=torch.float32).to(device)          # nwp
            hist_radar = torch.tensor(radar, dtype=torch.float32).to(device)

            # y 已经是 [B, pre, 1024]，单位仍按 0.1mm 处理
            fore_y_raw = torch.tensor(y, dtype=torch.float32)
            fore_y_mm = (fore_y_raw * 0.1).clamp_min(0.0)

            print("[DEBUG] sample y raw stats:",
                  float(fore_y_raw.min()), float(fore_y_raw.mean()), float(fore_y_raw.max()))
            print("[DEBUG] sample y mm(stats after *0.1):",
                  float(fore_y_mm.min()), float(fore_y_mm.mean()), float(fore_y_mm.max()))

            y_hat, logit_rain = model(hist_x, hist_radar)

        # 模型输出统一为 (B, N, pre)
        y_hat = y_hat.permute(0, 2, 1).contiguous().cpu()           # -> (B, pre, N)
        logit_rain = logit_rain.permute(0, 2, 1).contiguous().cpu() # -> (B, pre, N)

        y_hat_mm = _inv_y(y_hat, y_log_min, y_log_range)

        p_rain = torch.sigmoid(logit_rain)
        event = (p_rain >= p_thr).float()
        outputs = y_hat_mm * event

        fore_y_mm = fore_y_mm.cpu()

        _eval_and_print(outputs, y_hat_mm, fore_y_mm, p_rain, pre_step)