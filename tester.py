#!/usr/bin/env python 
# -*- coding:utf-8 -*-
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


def fitness_cal(adjmat, device=device):
    adjmat = torch.tensor(adjmat, dtype=torch.float32)
    start = time.time()
    dataset = np.load("../GE-STDGN-main/data/DataSet.npy")
    x = dataset[:5848, :, :]
    y = dataset[:5848, :, -1]
    nodes = x.shape[1]
    features = x.shape[-1]
    model = STDGN_woa(num_units=64, adj=adjmat, num_nodes=nodes, num_feature=features,
                      seq_len=16, pre_len=8, device=device, deep=1)

    model_train = trainer(model, epoch=10, batch=200, lr=1e-3, decay=0, his_step=16, pre_step=8, x=x, y=y, train=0.8,
                          valid='yes', device=device)
    RMSE, model = model_train.train()

    end = time.time()
    print("time cost: ", end - start)
    return RMSE, model


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
        y_min = float(np.load("y_min.npy"))
        y_max = float(np.load("y_max.npy"))

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

        # -> (B, pre, N)，和标签一致
        outputs = outputs.permute(0, 2, 1).contiguous().cpu()

        # 反归一化
        if y_min is not None:
            outputs = outputs * (y_max - y_min) + y_min

        # ========== 整体指标 ==========
        rmse = measure.GetRMSE(outputs, fore_y)
        mae = measure.GetMAP(outputs, fore_y)
        skill = measure.Skill(outputs, fore_y)

        print("RMSE:", rmse)
        print("MAE:", mae)
        print("SKILL:", skill)

        for thr in [0.1, 1.0, 5.0]:
            y_event = (fore_y >= thr).float().mean().item()
            p_event = (outputs >= thr).float().mean().item()
            print(f"thr={thr}  y_event_ratio={y_event:.6f}  pred_event_ratio={p_event:.6f}")

        # ========== 分类指标 ==========
        for thr in [0.1, 1.0, 5.0]:
            pod = measure.GetPOD(fore_y, outputs, thr=thr)
            far = measure.GetFAR(fore_y, outputs, thr=thr)
            csi = measure.GetCSI(fore_y, outputs, thr=thr)
            print(f"[thr={thr}] POD={pod:.4f} FAR={far:.4f} CSI={csi:.4f}")

        # ========== 分步评估 ==========
        for i in range(pre_step):
            print('step ', i)
            to = outputs[:, i, :]
            ty = fore_y[:, i, :]

            rmse = measure.GetRMSE(ty, to)
            mae = measure.GetMAP(ty, to)
            mape = measure.GetMAPE(ty, to)
            r2 = measure.R_square(ty, to)

            print("RMSE:", rmse)
            print("MAE:", mae)
            print("MAPE:", mape)
            print("R2:", r2)

            for thr in [0.1, 1.0, 5.0]:
                pod = measure.GetPOD(ty, to, thr=thr)
                far = measure.GetFAR(ty, to, thr=thr)
                csi = measure.GetCSI(ty, to, thr=thr)
                print(f"[thr={thr}] POD={pod:.4f} FAR={far:.4f} CSI={csi:.4f}")
