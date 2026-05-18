import os.path

import torch
import torch.nn as nn
import numpy as np
import measure
import torch.nn.functional as Fu

from torch.utils.data import TensorDataset, DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

cuda_gpu = torch.cuda.is_available()
p_thr = 0.5
save_dir = "./log"


def focal_bce_with_logits(logits, targets, mask=None, alpha=0.25, gamma=2.0):
    prob = torch.sigmoid(logits)
    ce = Fu.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce * ((1 - p_t) ** gamma)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = alpha_t * loss
    if mask is not None:
        return (loss * mask).sum() / (mask.sum() + 1e-6)
    return loss.mean()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
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
            setup_seed(20)
            idx = torch.randperm(_x.shape[0])
            _x = _x[idx]
            _y = _y[idx]

        valid_x = _x[:valid_len].clone().detach().float()
        valid_y = _y[:valid_len].clone().detach().float()
        test_x = _x[valid_len:].clone().detach().float()
        test_y = _y[valid_len:].clone().detach().float()
        return train_x, train_y, valid_x, valid_y, test_x, test_y


def TrainValidTestSample(x, radar, y, train=0.8, valid=None, shuffle=True):
    train_len = int(x.shape[0] * train)
    if valid is None:
        train_x = torch.tensor(x[:train_len], dtype=torch.float32)
        train_radar = torch.tensor(radar[:train_len], dtype=torch.float32)
        train_y = torch.tensor(y[:train_len], dtype=torch.float32)

        test_x = torch.tensor(x[train_len:], dtype=torch.float32)
        test_radar = torch.tensor(radar[train_len:], dtype=torch.float32)
        test_y = torch.tensor(y[train_len:], dtype=torch.float32)

        return train_x, train_radar, train_y, test_x, test_radar, test_y
    else:
        valid_len = int(x.shape[0] * valid)

        x = torch.tensor(x, dtype=torch.float32)
        radar = torch.tensor(radar, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        train_x = x[:train_len].clone().detach().float()
        train_radar = radar[:train_len].clone().detach().float()
        train_y = y[:train_len].clone().detach().float()

        _x = x[train_len:]
        _radar = radar[train_len:]
        _y = y[train_len:]

        if shuffle:
            setup_seed(20)
            idx = torch.randperm(_x.shape[0])
            _x = _x[idx]
            _radar = _radar[idx]
            _y = _y[idx]

        valid_x = _x[:valid_len].clone().detach().float()
        valid_radar = _radar[:valid_len].clone().detach().float()
        valid_y = _y[:valid_len].clone().detach().float()

        test_x = _x[valid_len:].clone().detach().float()
        test_radar = _radar[valid_len:].clone().detach().float()
        test_y = _y[valid_len:].clone().detach().float()

        return (
            train_x, train_radar, train_y,
            valid_x, valid_radar, valid_y,
            test_x, test_radar, test_y
        )


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
                hist_x[i, :, j, :] = x[i + j, :, :]
            else:
                hist_x[i, :, j] = x[i + j, :]
        for j in range(pre_step):
            fore_y[i, :, j] = y[i + his_step + j, :]
    print(hist_x.shape)
    print(fore_y.shape)
    return hist_x, fore_y


class trainer():
    def __init__(self, model, epoch, batch, lr, decay, his_step, pre_step,
                 x, y, train=0.8, valid=None, device=None, radar=None, label=None):
        self.epoch = epoch
        self.batch = batch
        self.model = model
        self.lr = lr
        self.decay = decay
        self.x = x
        self.radar = radar
        self.label = label
        self.y = y
        self.his = his_step
        self.pre = pre_step
        self.valid = valid
        self.trainset = train
        self.device = device if device is not None else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def _is_sample_mode(self):
        return getattr(self.model, "input_mode", "patch") == "sample"

    def _prepare_patch_data(self):
        nodes = self.x.shape[1]
        feature_num = self.x.shape[-1]

        # 兼容两种 patch 数据范式
        if self.x.ndim == 4:
            # x=(B,his,N,F), y=(B,pre,N)
            x_np = self.x
            y_np = self.y
            B, his, N, F = x_np.shape
            pre = y_np.shape[1]

            x_flat = x_np.reshape(-1, F)
            x_ones, x_min, x_max = measure.Norm(x_flat, dim=2)
            x_ones = x_ones.reshape(B, his, N, F)

            y_flat = (y_np.reshape(-1) * 0.1).astype(np.float32)  # 0.1mm -> mm
            y_log = np.log1p(y_flat)
            y_log_min = float(y_log.min())
            y_log_max = float(y_log.max())
            y_log_range = y_log_max - y_log_min + 1e-8
            y_norm = (y_log - y_log_min) / y_log_range
            y_ones = y_norm.reshape(B, pre, N)

            np.save("log/y_log_min.npy", np.array(y_log_min, dtype=np.float32))
            np.save("log/y_log_max.npy", np.array(y_log_max, dtype=np.float32))

            hist_x = torch.tensor(x_ones, dtype=torch.float32)
            fore_y = torch.tensor(y_ones, dtype=torch.float32)
        else:
            # 原版旧范式：x=(T,N,F), y=(T,N)
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

            # 为了统一后面的 inv_y 流程，这里也存一份
            np.save("log/y_log_min.npy", np.array(0.0, dtype=np.float32))
            np.save("log/y_log_max.npy", np.array(1.0, dtype=np.float32))

        train_x, train_y, valid_x, valid_y, test_x, test_y = TrainValidTest(
            hist_x, fore_y, train=self.trainset, valid=0.05
        )

        return train_x, train_y, valid_x, valid_y, test_x, test_y

    def _prepare_sample_data(self):
        if self.radar is None:
            raise ValueError("sample 模式下 trainer 需要额外传入 radar")

        # sample 输入不在 trainer 里做 CNN；只处理监督 y
        x_np = self.x
        radar_np = self.radar
        y_np = self.y  # [B, pre, 1024]

        y_flat = (y_np.reshape(-1) * 0.1).astype(np.float32)  # 0.1mm -> mm
        y_log = np.log1p(y_flat)
        y_log_min = float(y_log.min())
        y_log_max = float(y_log.max())
        y_log_range = y_log_max - y_log_min + 1e-8
        y_norm = (y_log - y_log_min) / y_log_range
        y_ones = y_norm.reshape(y_np.shape[0], y_np.shape[1], y_np.shape[2])

        np.save("log/y_log_min.npy", np.array(y_log_min, dtype=np.float32))
        np.save("log/y_log_max.npy", np.array(y_log_max, dtype=np.float32))

        (
            train_x, train_radar, train_y,
            valid_x, valid_radar, valid_y,
            test_x, test_radar, test_y
        ) = TrainValidTestSample(
            x_np, radar_np, y_ones,
            train=self.trainset, valid=0.05
        )

        return (
            train_x, train_radar, train_y,
            valid_x, valid_radar, valid_y,
            test_x, test_radar, test_y
        )

    def train(self):
        train_loss = []

        train_rmse_all_list = []
        train_rmse_rain_list = []

        valid_rmse_all_list = []
        valid_rmse_rain_list = []

        self.model = self.model.to(self.device)
        self.model.train()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.decay)

        sample_mode = self._is_sample_mode()

        if not sample_mode:
            train_x, train_y, valid_x, valid_y, test_x, test_y = self._prepare_patch_data()
        else:
            (
                train_x, train_radar, train_y,
                valid_x, valid_radar, valid_y,
                test_x, test_radar, test_y
            ) = self._prepare_sample_data()

        y_log_min = float(np.load("log/y_log_min.npy"))
        y_log_max = float(np.load("log/y_log_max.npy"))
        y_log_range = y_log_max - y_log_min + 1e-8

        def inv_y(y_norm):
            y_log = y_norm * y_log_range + y_log_min
            return torch.clamp(torch.expm1(y_log), min=0.0)

        def safe_rmse(pred, true, mask):
            if mask.sum().item() == 0:
                return float("nan")
            return measure.GetRMSE(pred[mask], true[mask])

        with torch.no_grad():
            train_y_mm = inv_y(train_y.to(self.device))
            # train_y 统一按 (S, pre, N)
            if train_y_mm.dim() == 3:
                rain_label_all = (train_y_mm >= 0.1).float()
            else:
                raise ValueError(f"Unexpected train_y shape: {train_y_mm.shape}")

            pos = rain_label_all.sum()
            neg = rain_label_all.numel() - pos
            pos_weight_global = (neg / (pos + 1e-6)).clamp(1.0, 10.0).detach()

        print(f"[POS_WEIGHT_GLOBAL] pos={pos.item():.0f}, "
              f"neg={neg.item():.0f}, ratio={pos_weight_global.item():.4f}")

        bce = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_global)

        setup_seed(20)

        if not sample_mode:
            train_loader = DataLoader(
                dataset=TensorDataset(train_x, train_y),
                batch_size=self.batch,
                shuffle=True
            )
        else:
            train_loader = DataLoader(
                dataset=TensorDataset(train_x, train_radar, train_y),
                batch_size=self.batch,
                shuffle=True
            )

        Valid_y = inv_y(valid_y.cpu())

        for e in range(self.epoch):
            epoch_loss = 0.0
            batch_count = 0
            train_sse_all = 0.0
            train_cnt_all = 0

            train_sse_rain = 0.0
            train_cnt_rain = 0
            for i, data in enumerate(train_loader):
                if not sample_mode:
                    inputs, target = data
                    inputs = inputs.to(self.device)
                    target = target.to(self.device)  # (B, pre, N)

                    target = target.permute(0, 2, 1).contiguous()  # -> (B, N, pre)
                    y_hat, logit_rain = self.model(inputs)
                else:
                    inputs, radar, target = data
                    inputs = inputs.to(self.device)   # nwp
                    radar = radar.to(self.device)
                    target = target.to(self.device)   # (B, pre, N)

                    target = target.permute(0, 2, 1).contiguous()  # -> (B, N, pre)
                    y_hat, logit_rain = self.model(inputs, radar)

                with torch.no_grad():
                    target_mm = inv_y(target)
                    # 分类头概率
                    p_train = torch.sigmoid(logit_rain)

                    if p_train.dim() == 2:
                        p_train = p_train.unsqueeze(-1).expand_as(y_hat)
                    elif p_train.shape[-1] == 1 and y_hat.shape[-1] > 1:
                        p_train = p_train.expand_as(y_hat)

                    # 分类头门控：判断无雨的位置置 0
                    event_train = (p_train >= p_thr).float()

                    # 回归头反归一化到 mm
                    pred_train_mm_reg = inv_y(y_hat.detach())

                    # 门控后的训练预测值
                    pred_train_mm = pred_train_mm_reg * event_train

                    # 计算训练集 all RMSE 的 SSE
                    diff_train = pred_train_mm - target_mm

                    train_sse_all += torch.sum(diff_train ** 2).item()
                    train_cnt_all += diff_train.numel()

                    # 计算训练集 rain RMSE 的 SSE
                    train_mask_rain = (target_mm >= 0.1)

                    if train_mask_rain.sum().item() > 0:
                        train_sse_rain += torch.sum(diff_train[train_mask_rain] ** 2).item()
                        train_cnt_rain += train_mask_rain.sum().item()

                thr = 0.1
                rain_label = (target_mm >= thr).float()

                loss_cls = bce(logit_rain, rain_label)

                rain_mask = (target_mm >= 0.1).float()
                per = Fu.smooth_l1_loss(y_hat, target, reduction="none")
                w = (
                        1.0
                        + 1.0 * (target_mm >= 0.1).float()
                        + 2.0 * (target_mm >= 2.6).float()
                        + 5.0 * (target_mm >= 8.1).float()
                        + 10.0 * (target_mm >= 16.0).float()
                )
                loss_reg = (per * w * rain_mask).sum() / (rain_mask.sum() + 1e-6)

                lambda_reg = min(1.0, (e + 1) / 5.0)
                loss = loss_cls + lambda_reg * loss_reg

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                batch_count += 1

            avg_train_loss = epoch_loss / max(batch_count, 1)

            train_rmse_all = np.sqrt(train_sse_all / max(train_cnt_all, 1))

            if train_cnt_rain > 0:
                train_rmse_rain = np.sqrt(train_sse_rain / train_cnt_rain)
            else:
                train_rmse_rain = np.nan

            print(
                'Epoch [{}/{}], Loss:{:.4f}, Train RMSE_all:{:.4f}, Train RMSE_rain:{:.4f}'.format(
                    e + 1, self.epoch, avg_train_loss, train_rmse_all, train_rmse_rain
                )
            )

            train_loss.append(avg_train_loss)
            train_rmse_all_list.append(train_rmse_all)
            train_rmse_rain_list.append(train_rmse_rain)

            with torch.no_grad():
                self.model.eval()

                if not sample_mode:
                    valid_x = valid_x.to(self.device)
                    y_hat, logit_rain = self.model(valid_x)
                else:
                    valid_x = valid_x.to(self.device)
                    valid_radar = valid_radar.to(self.device)
                    y_hat, logit_rain = self.model(valid_x, valid_radar)

                # 1. 回归头输出: (B, N, pre) -> (B, pre, N)
                valid_outputs = y_hat.permute(0, 2, 1).contiguous()

                # 2. 先反归一化到 mm
                valid_outputs_mm_reg = inv_y(valid_outputs.detach().cpu())

                # 3. 分类头输出概率: (B, N, pre) -> (B, pre, N)
                p = torch.sigmoid(logit_rain)

                if p.dim() == 2:
                    p = p.unsqueeze(-1).expand_as(y_hat)
                elif p.shape[-1] == 1 and y_hat.shape[-1] > 1:
                    p = p.expand_as(y_hat)

                event = (p >= p_thr).float()
                event = event.permute(0, 2, 1).contiguous().cpu()

                # 4. 分类头判断无雨的位置，回归结果强制置 0
                valid_outputs_mm = valid_outputs_mm_reg * event

                print("pred_event_ratio =", event.mean().item())

                mask_all = torch.ones_like(Valid_y, dtype=torch.bool)
                mask_rain = (Valid_y >= 0.1)

                rmse_all = safe_rmse(valid_outputs_mm, Valid_y, mask_all)
                rmse_rain = safe_rmse(valid_outputs_mm, Valid_y, mask_rain)

                valid_rmse_all_list.append(rmse_all)
                valid_rmse_rain_list.append(rmse_rain)
                print("valid RMSE_all :", rmse_all)
                print("valid RMSE_rain:", rmse_rain)

                self.model.train()

        model_name = self.model.__class__.__name__
        if model_name != 'STDGN_woa':
            np.save(os.path.join(save_dir, "train_Loss_" + model_name + '.npy'),
                    np.array(train_loss))

            np.save(os.path.join(save_dir, "train_RMSE_all_" + model_name + '.npy'),
                    np.array(train_rmse_all_list))

            np.save(os.path.join(save_dir, "train_RMSE_rain_" + model_name + '.npy'),
                    np.array(train_rmse_rain_list))

            np.save(os.path.join(save_dir, "valid_RMSE_all_" + model_name + '.npy'),
                    np.array(valid_rmse_all_list))

            np.save(os.path.join(save_dir, "valid_RMSE_rain_" + model_name + '.npy'),
                    np.array(valid_rmse_rain_list))

        if self.valid is not None:
            print('valid result (all): ', rmse_all)
            print('valid result (rain):', rmse_rain)
            return rmse_all, self.model.cpu()

        with torch.no_grad():
            self.model.eval()

            if not sample_mode:
                test_x = test_x.to(self.device)
                y_hat, logit_rain = self.model(test_x)
            else:
                test_x = test_x.to(self.device)
                test_radar = test_radar.to(self.device)
                y_hat, logit_rain = self.model(test_x, test_radar)

            p = torch.sigmoid(logit_rain)
            if p.dim() == 2:
                p = p.unsqueeze(-1).expand_as(y_hat)
            elif p.shape[-1] == 1 and y_hat.shape[-1] > 1:
                p = p.expand_as(y_hat)

            test_outputs_reg = y_hat.permute(0, 2, 1).contiguous()   # (B, pre, N)
            test_outputs_reg_mm = inv_y(test_outputs_reg.detach().cpu())

            test_y_mm = inv_y(test_y.detach().cpu())
            if test_y_mm.shape != test_outputs_reg_mm.shape:
                test_y_mm = test_y_mm.permute(0, 2, 1).contiguous()

            mask_all = torch.ones_like(test_y_mm, dtype=torch.bool)
            mask_rain = (test_y_mm >= 0.1)

            rmse_reg_all = safe_rmse(test_outputs_reg_mm, test_y_mm, mask_all)
            rmse_reg_rain = safe_rmse(test_outputs_reg_mm, test_y_mm, mask_rain)

            print("RMSE(REG all):", rmse_reg_all)
            print("RMSE(REG rain):", rmse_reg_rain)

            event = (p >= p_thr).float()
            event = event.permute(0, 2, 1).contiguous().cpu()

            # 先反归一化，再置零，保证无雨位置真的是 0 mm
            test_outputs_g_mm = test_outputs_reg_mm * event

            rmse_g_all = safe_rmse(test_outputs_g_mm, test_y_mm, mask_all)
            rmse_g_rain = safe_rmse(test_outputs_g_mm, test_y_mm, mask_rain)

            print("RMSE(GATED all):", rmse_g_all)
            print("RMSE(GATED rain):", rmse_g_rain)

            p_stats = p.detach().cpu()
            print("p stats(min/mean/max) =",
                  float(p_stats.min()), float(p_stats.mean()), float(p_stats.max()))

        return rmse_g_rain, self.model.cpu()