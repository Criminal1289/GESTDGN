#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import measure

from torch.utils.data import TensorDataset, Dataset, DataLoader
from torch.autograd import Variable

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

cuda_gpu = torch.cuda.is_available()

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    #random.seed(seed)
    torch.backends.cudnn.deterministic = True

def TrainValidTest(x, y, train=0.8, valid=None, shuffle=True):

    train_len = int(x.shape[0] * train)
    if valid is None:
        train_x = torch.tensor(x[:train_len], dtype=torch.float32)
        train_y = torch.tensor(y[:train_len], dtype=torch.float32)
        test_x = torch.tensor(x[train_len:], dtype=torch.float32)
        test_y = torch.tensor(y[train_len:], dtype=torch.float32)
        return train_x, train_y, test_x, test_y
    else:
        valid_len = int(x.shape[0] * valid)
        train_x = x[:train_len].clone().detach().float()
        train_y = y[:train_len].clone().detach().float()
        _x = x[train_len:]
        _y = y[train_len:]

        if shuffle:
            if shuffle:
                setup_seed(20)
                idx = torch.randperm(_x.shape[0])
                _x = _x[idx]
                _y = _y[idx]
        valid_x = _x[:valid_len].clone().detach().float()
        valid_y = _y[:valid_len].clone().detach().float()
        test_x = _x[valid_len:].clone().detach().float()
        test_y = _y[valid_len:].clone().detach().float()
        return train_x, train_y, valid_x, valid_y, test_x, test_y

def Divide(x, y, his_step, pre_step):
    x_shape = len(x.shape)
    if x_shape == 3:
        hist_x = np.zeros((x.shape[0] - his_step - pre_step + 1, x.shape[1], his_step, x.shape[2]))
    else:
        hist_x = np.zeros((x.shape[0] - his_step - pre_step + 1, x.shape[1], his_step))
    fore_y = np.zeros((y.shape[0] - his_step - pre_step + 1, y.shape[1], pre_step))
    for i in range(x.shape[0] - his_step - pre_step + 1):
        for j in range(his_step):
            if x_shape == 3:
                hist_x[i, :, j, :] = x[i+j, :, :]
            else:
                hist_x[i, :, j] = x[i + j, :]
        for j in range(pre_step):
            fore_y[i, :, j] = y[i+his_step+j, :]
    print(hist_x.shape)
    print(fore_y.shape)
    return hist_x, fore_y


class trainer():
    def __init__(self, model, epoch, batch, lr, decay, his_step, pre_step, x, y, train=0.8, valid=None, device=None):
        self.epoch = epoch
        self.batch = batch
        self.model = model
        self.lr = lr
        self.decay = decay
        self.x = x
        self.y = y
        self.his = his_step
        self.pre = pre_step
        self.valid = valid
        self.trainset = train
        self.device = device if device is not None else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    def train(self):
        train_loss = []
        valid_rsme = []

        nodes = self.x.shape[1]
        feature_num = self.x.shape[-1]
        criterion = nn.MSELoss().to(self.device)

        self.model = self.model.to(self.device)
        self.model.train()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.decay)

        # ===============================
        # NEW: 支持两种数据范式
        # ===============================
        if self.x.ndim == 4:
            # 新范式：x=(B,his,N,F), y=(B,pre,N)
            x_np = self.x
            y_np = self.y

            B, his, N, F = x_np.shape
            pre = y_np.shape[1]

            # normalization（按 feature 维）
            x_flat = x_np.reshape(-1, F)
            x_ones, x_min, x_max = measure.Norm(x_flat, dim=2)
            x_ones = x_ones.reshape(B, his, N, F)

            NULL_VAL = -9.0

            y_flat = y_np.reshape(-1).astype(np.float32)
            valid = (y_flat != NULL_VAL)

            # 对有效降水做裁剪：负值当 0（降水物理意义）
            y_valid = np.clip(y_flat[valid], 0.0, None)

            # log1p 变换（增强尾部）
            y_log_valid = np.log1p(y_valid)

            # 用有效值计算 min/max（在 log 空间）
            y_log_min = float(y_log_valid.min())
            y_log_max = float(y_log_valid.max())
            y_log_range = y_log_max - y_log_min + 1e-8

            # 归一化到 [0,1]（仍在 log 空间）
            y_norm = np.zeros_like(y_flat, dtype=np.float32)
            y_norm[valid] = (y_log_valid - y_log_min) / y_log_range

            # 缺测位置给 0（不会参与 loss/metric，因为我们会 mask）
            y_norm[~valid] = 0.0

            y_ones = y_norm.reshape(B, pre, N)

            # 保存可逆参数（注意：这是 log 空间的 min/max）
            np.save("y_log_min.npy", np.array(y_log_min, dtype=np.float32))
            np.save("y_log_max.npy", np.array(y_log_max, dtype=np.float32))
            np.save("y_null_val.npy", np.array(NULL_VAL, dtype=np.float32))

            hist_x = torch.tensor(x_ones, dtype=torch.float32)

            y_mask = valid.reshape(B, pre, N).astype(np.float32)  # 1=有效,0=缺测
            y_mask = torch.tensor(y_mask, dtype=torch.float32)

            fore_y_val = torch.tensor(y_ones, dtype=torch.float32)  # (B, pre, N)
            fore_y = torch.stack([fore_y_val, y_mask], dim=-1)  # (B, pre, N, 2)


        else:
            # 旧范式：x=(T,N,F), y=(T,N)
            x_ones, x_min, x_max = measure.Norm(
                np.array(torch.tensor(self.x).view(-1, feature_num), dtype=float),
                dim=2
            )
            y_ones, y_min, y_max = measure.Norm(
                np.array(torch.tensor(self.y).view(-1), dtype=float),
                dim=1
            )

            source_x = torch.tensor(x_ones).view(-1, nodes, feature_num)
            source_y = torch.tensor(y_ones).view(-1, nodes)

            hist_x, fore_y = Divide(source_x, source_y, self.his, self.pre)

        train_x, train_y, valid_x, valid_y, test_x, test_y = TrainValidTest(hist_x, fore_y, train=self.trainset, valid=0.05)

        # ===== log1p inverse 参数（只加载一次）=====
        y_log_min = float(np.load("y_log_min.npy"))
        y_log_max = float(np.load("y_log_max.npy"))
        y_log_range = y_log_max - y_log_min + 1e-8

        def inv_y(y_norm):
            """
            y_norm: torch.Tensor, log-space normalized in [0,1]
            return: mm
            """
            y_log = y_norm * y_log_range + y_log_min
            return torch.expm1(y_log)

        setup_seed(20)
        train_loader = DataLoader(dataset=TensorDataset(train_x, train_y), batch_size=self.batch, shuffle=True)

        valid_y_val = valid_y[..., 0] if (valid_y.ndim == 4 and valid_y.shape[-1] == 2) else valid_y
        Valid_y = inv_y(valid_y_val.cpu())

        for e in range(self.epoch):
            for i, data in enumerate(train_loader):
                inputs, target_pack = data
                inputs = inputs.to(self.device)
                target_pack = target_pack.to(self.device)  # (B, pre, N, 2)

                target = target_pack[..., 0]  # y_norm
                mask = target_pack[..., 1]  # 1 valid

                target = target.permute(0, 2, 1).contiguous()  # (B, N, pre)
                mask = mask.permute(0, 2, 1).contiguous()

                y_hat = self.model(inputs)

                diff2 = (y_hat - target) ** 2

                with torch.no_grad():
                    # 把 target 从 log-norm 还原到 mm，用来判阈值
                    target_mm = torch.expm1(target * y_log_range + y_log_min)

                # 权重：小雨=1，中雨/大雨逐级放大
                w = torch.ones_like(target_mm)
                w = w + 1.0 * (target_mm >= 1.0).float()
                w = w + 3.0 * (target_mm >= 5.0).float()
                w = w + 6.0 * (target_mm >= 10.0).float()

                loss = (diff2 * w * mask).sum() / ((w * mask).sum() + 1e-8)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if (e + 1) % 10 == 0:
                # print(device)
                print('Epoch [{}/{}], Loss:{:.4f}'.format(e + 1, self.epoch, loss.item()))
                train_loss.append(loss.item())
                with torch.no_grad():
                    valid_x = valid_x.to(self.device)

                    self.model.eval()
                    valid_outputs = self.model(valid_x)
                    valid_outputs = valid_outputs.permute(0, 2, 1).contiguous()

                    valid_outputs = valid_outputs.detach().cpu()
                    valid_outputs_mm = inv_y(valid_outputs)
                    rmse = measure.GetRMSE(valid_outputs_mm, Valid_y)

                    valid_rsme.append(rmse)
                    print("valid RMSE:", rmse)
                    self.model.train()
        model_name = self.model.__class__.__name__
        np.save("valid_RSME_" + model_name+'.npy', np.array(valid_rsme))
        np.save("train_Loss_" + model_name+'.npy', np.array(train_loss))

        if self.valid is not None:
            print('valid result: ', rmse)
            return rmse, self.model.cpu()

        with torch.no_grad():
            self.model.eval()
            test_x = test_x.to(self.device)

            test_outputs = self.model(test_x)
            test_outputs = test_outputs.permute(0, 2, 1).contiguous()
            test_outputs = test_outputs.detach().cpu()
            test_outputs = test_outputs.cpu()
            test_y_val = test_y[..., 0] if (test_y.ndim == 4 and test_y.shape[-1] == 2) else test_y
            test_y_mm = inv_y(test_y_val.cpu())

            test_outputs_mm = inv_y(test_outputs.cpu())
            rmse = measure.GetRMSE(test_y_mm, test_outputs_mm)
            print("RMSE:", rmse)

        return rmse, self.model.cpu()
