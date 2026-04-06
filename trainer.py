import torch
import torch.nn as nn
import numpy as np
import measure
import torch.nn.functional as Fu


from torch.utils.data import TensorDataset, Dataset, DataLoader
from torch.autograd import Variable

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

cuda_gpu = torch.cuda.is_available()
p_thr = 0.5


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
    # random.seed(seed)
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
        valid_rmse_all_list = []
        valid_rmse_rain_list = []

        nodes = self.x.shape[1]
        feature_num = self.x.shape[-1]

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
            # ---------- X normalization ----------
            x_flat = x_np.reshape(-1, F)
            x_ones, x_min, x_max = measure.Norm(x_flat, dim=2)
            x_ones = x_ones.reshape(B, his, N, F)
            # ---------- Y: log1p + norm ----------
            y_flat = (y_np.reshape(-1) * 0.1).astype(np.float32)  # 0.1mm -> mm
            y_log = np.log1p(y_flat)
            y_log_min = float(y_log.min())
            y_log_max = float(y_log.max())
            y_log_range = y_log_max - y_log_min + 1e-8
            y_norm = (y_log - y_log_min) / y_log_range
            y_ones = y_norm.reshape(B, pre, N)
            # 保存反归一化参数
            np.save("y_log_min.npy", np.array(y_log_min, dtype=np.float32))
            np.save("y_log_max.npy", np.array(y_log_max, dtype=np.float32))
            hist_x = torch.tensor(x_ones, dtype=torch.float32)
            fore_y = torch.tensor(y_ones, dtype=torch.float32)
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

        train_x, train_y, valid_x, valid_y, test_x, test_y = TrainValidTest(hist_x, fore_y, train=self.trainset,
                                                                            valid=0.05)

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
            return torch.clamp(torch.expm1(y_log), min=0.0)

        def safe_rmse(pred, true, mask):
            if mask.sum().item() == 0:
                return float("nan")
            return measure.GetRMSE(pred[mask], true[mask])

        with torch.no_grad():
            # train_y: (S, N, pre)
            train_y_mm = inv_y(train_y.to(self.device))  # 转回 mm
            rain_label_all = (train_y_mm >= 0.1).float()

            pos = rain_label_all.sum()
            neg = rain_label_all.numel() - pos

            pos_weight_global = (neg / (pos + 1e-6)).clamp(1.0, 10.0).detach()

        print(f"[POS_WEIGHT_GLOBAL] pos={pos.item():.0f}, "
              f"neg={neg.item():.0f}, ratio={pos_weight_global.item():.4f}")

        # 固定分类loss
        bce = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_global)

        setup_seed(20)
        train_loader = DataLoader(dataset=TensorDataset(train_x, train_y), batch_size=self.batch, shuffle=True)

        Valid_y = inv_y(valid_y.cpu())

        for e in range(self.epoch):
            for i, data in enumerate(train_loader):
                inputs, target = data
                inputs = inputs.to(self.device)
                target = target.to(self.device)  # target: (B, pre, N)

                # 统一成模型输出的形状 (B, N, pre)
                target = target.permute(0, 2, 1).contiguous()  # (B, N, pre)

                y_hat, logit_rain = self.model(inputs)  # 期望 y_hat/logit_rain: (B, N, pre)

                # --- 把 y_norm 还原到 mm，用于是否下雨标签 ---
                with torch.no_grad():
                    target_mm = inv_y(target)  # (B, N, pre)

                thr = 0.1
                rain_label = (target_mm >= thr).float()  # (B, N, pre)

                # ===== 分类：BCE + pos_weight（全局固定）=====
                loss_cls = bce(logit_rain, rain_label)

                rain_mask = (target_mm >= 0.1).float()
                per = Fu.smooth_l1_loss(y_hat, target, reduction="none")
                # 我国规定如下：24小时降水量小于10毫米为小雨；大于等于10毫米，小于25毫米为中雨；大于等于25毫米，小于50毫米为大雨
                # 由于本研究预测目标为逐小时降水量，因此损失函数中的权重阈值采用小时降水等级划分（0.1、2.5、8 mm）
                # 而不是直接使用24小时降水等级标准，以保证权重设置与模型预测时间尺度一致。
                w = (
                        1.0
                        + 2.0 * (target_mm >= 0.1).float()  # 有降水
                        + 3.0 * (target_mm >= 2.5).float()  # 中雨
                        + 10.0 * (target_mm >= 8.0).float()  # 大雨
                )
                loss_reg = (per * w * rain_mask).sum() / (rain_mask.sum() + 1e-6)

                # ===== 总损失 =====
                lambda_reg = min(1.0, (e + 1) / 5.0)
                loss = loss_cls + lambda_reg * loss_reg

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
                    y_hat, logit_rain = self.model(valid_x)
                    # ===== 回归 RMSE：不门控 =====
                    valid_outputs = y_hat.permute(0, 2, 1).contiguous()
                    valid_outputs_mm = inv_y(valid_outputs.detach().cpu())

                    # 两类RMSE
                    mask_all = torch.ones_like(Valid_y, dtype=torch.bool)
                    mask_rain = (Valid_y >= 0.1)

                    rmse_all = safe_rmse(valid_outputs_mm, Valid_y, mask_all)
                    rmse_rain = safe_rmse(valid_outputs_mm, Valid_y, mask_rain)

                    # ===== 事件评估：才用分类头 =====
                    p = torch.sigmoid(logit_rain)
                    event = (p >= p_thr).float()
                    print("pred_event_ratio =", event.mean().item())

                    valid_rmse_all_list.append(rmse_all)
                    valid_rmse_rain_list.append(rmse_rain)
                    print("valid RMSE_all :", rmse_all)
                    print("valid RMSE_rain:", rmse_rain)

                    self.model.train()
        model_name = self.model.__class__.__name__
        np.save("valid_RSME_" + model_name + '.npy', np.array(valid_rmse_all_list))
        np.save("train_Loss_" + model_name + '.npy', np.array(train_loss))

        if self.valid is not None:
            print('valid result (all): ', rmse_all)
            print('valid result (rain):', rmse_rain)
            return rmse_all, self.model.cpu()

        with torch.no_grad():
            self.model.eval()
            test_x = test_x.to(self.device)
            y_hat, logit_rain = self.model(test_x)  # 期望 y_hat: (B, N, pre)

            # --- 分类概率 ---
            p = torch.sigmoid(logit_rain)  # (B, N, pre) 或 (B, N, 1)/(B,N)
            # 统一成 (B, N, pre)
            if p.dim() == 2:
                p = p.unsqueeze(-1).expand_as(y_hat)
            elif p.shape[-1] == 1 and y_hat.shape[-1] > 1:
                p = p.expand_as(y_hat)

            # ========= 1) 纯回归输出（科研评估用）=========
            # y_hat: (B, N, pre) -> (B, pre, N)
            test_outputs_reg = y_hat.permute(0, 2, 1).contiguous()
            test_outputs_reg_mm = inv_y(test_outputs_reg.detach().cpu())

            test_y_mm = inv_y(test_y.detach().cpu())  # test_y 期望 (B, pre, N) 或 (B, N, pre)
            if test_y_mm.shape != test_outputs_reg_mm.shape:
                # 如果 test_y 是 (B, N, pre)，转成 (B, pre, N)
                test_y_mm = test_y_mm.permute(0, 2, 1).contiguous()

            mask_all = torch.ones_like(test_y_mm, dtype=torch.bool)
            mask_rain = (test_y_mm >= 0.1)

            rmse_reg_all = safe_rmse(test_outputs_reg_mm, test_y_mm, mask_all)
            rmse_reg_rain = safe_rmse(test_outputs_reg_mm, test_y_mm, mask_rain)

            print("RMSE(REG all):", rmse_reg_all)
            print("RMSE(REG rain):", rmse_reg_rain)

            # ========= 2) 门控输出（业务参考，可选）=========
            event = (p >= p_thr).float()  # (B, N, pre)
            y_pred_norm = y_hat * event  # (B, N, pre)

            test_outputs_g = y_pred_norm.permute(0, 2, 1).contiguous()
            test_outputs_g_mm = inv_y(test_outputs_g.detach().cpu())

            rmse_g_all = safe_rmse(test_outputs_g_mm, test_y_mm, mask_all)
            rmse_g_rain = safe_rmse(test_outputs_g_mm, test_y_mm, mask_rain)

            print("RMSE(GATED all):", rmse_g_all)
            print("RMSE(GATED rain):", rmse_g_rain)

            # --- 额外打印 p 的统计，保持你原来的日志习惯 ---
            p_stats = p.detach().cpu()
            print("p stats(min/mean/max) =",
                  float(p_stats.min()), float(p_stats.mean()), float(p_stats.max()))

        return rmse_reg_rain, self.model.cpu()
