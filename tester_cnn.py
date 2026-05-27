#!/usr/bin/env python
# -*- coding:utf-8 -*-
import glob
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import numpy as np
import matplotlib.pyplot as plt
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
    y_log_min = float(np.load("log/y_log_min.npy"))
    y_log_max = float(np.load("log/y_log_max.npy"))
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
    rain_thr_list = [0.1, 2.6, 8.1, 16.0]

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


def plot_step_rmse_proof(outputs, y_hat_mm, fore_y_mm, pre_step, save_dir="fig_step_proof"):
    """
    画图证明：
    1. step0-3 的 RMSE 下降；
    2. 同时真实降雨均值、雨区数量、强降雨数量也在下降；
    3. 因此整体 RMSE 下降主要受真实降雨分布变化影响，而不能简单解释为预测越远越准。

    outputs: 门控后的预测结果, shape = [B, pre_step, N]
    y_hat_mm: 纯回归头预测结果, shape = [B, pre_step, N]
    fore_y_mm: 真实降雨, shape = [B, pre_step, N]
    """

    os.makedirs(save_dir, exist_ok=True)

    steps = []
    rmse_all_list = []
    rmse_rain_list = []
    rmse_heavy_list = []
    rmse_reg_all_list = []
    true_mean_list = []
    true_max_list = []
    pred_mean_list = []
    rain_count_list = []
    heavy_count_list = []
    light_count_list = []
    zero_count_list = []

    for i in range(pre_step):
        to = outputs[:, i, :].detach().cpu()
        to_reg = y_hat_mm[:, i, :].detach().cpu()
        ty = fore_y_mm[:, i, :].detach().cpu()

        # 有效值，正常情况下 ty>=0
        m_all = ty >= 0
        m_rain = ty >= 0.1
        m_heavy = ty >= 5.0
        m_light = (ty >= 0.1) & (ty < 1.0)
        m_zero = ty < 0.1

        def safe_rmse(pred, true, mask):
            if mask.sum().item() == 0:
                return np.nan
            return torch.sqrt(torch.mean((pred[mask] - true[mask]) ** 2)).item()

        rmse_all = safe_rmse(to, ty, m_all)
        rmse_rain = safe_rmse(to, ty, m_rain)
        rmse_heavy = safe_rmse(to, ty, m_heavy)
        rmse_reg_all = safe_rmse(to_reg, ty, m_all)

        steps.append(i)
        rmse_all_list.append(rmse_all)
        rmse_rain_list.append(rmse_rain)
        rmse_heavy_list.append(rmse_heavy)
        rmse_reg_all_list.append(rmse_reg_all)

        true_mean_list.append(ty[m_all].mean().item())
        true_max_list.append(ty[m_all].max().item())
        pred_mean_list.append(to[m_all].mean().item())

        rain_count_list.append(m_rain.sum().item())
        heavy_count_list.append(m_heavy.sum().item())
        light_count_list.append(m_light.sum().item())
        zero_count_list.append(m_zero.sum().item())

    # ==============================
    # 图1：RMSE 与真实降雨均值对照
    # ==============================
    fig, ax1 = plt.subplots(figsize=(7, 5))

    ax1.plot(steps, rmse_all_list, marker='o', linewidth=2, label='RMSE_all')
    ax1.set_xlabel('Forecast step')
    ax1.set_ylabel('RMSE_all / mm')
    ax1.set_xticks(steps)

    ax2 = ax1.twinx()
    ax2.plot(steps, true_mean_list, marker='s', linewidth=2, linestyle='--', label='True mean rainfall')
    ax2.set_ylabel('True mean rainfall / mm')

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')

    plt.title('RMSE decreases with the decrease of true rainfall intensity')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "RMSE_VS_TRUE_MEAN.png"), dpi=300)
    plt.close()

    # ==============================
    # 图2：不同 step 的雨区数量变化
    # ==============================
    x = np.arange(len(steps))
    width = 0.25

    plt.figure(figsize=(8, 5))
    plt.bar(x - width, light_count_list, width=width, label='0.1 <= true < 1.0 mm')
    plt.bar(x, rain_count_list, width=width, label='true >= 0.1 mm')
    plt.bar(x + width, heavy_count_list, width=width, label='true >= 5.0 mm')

    plt.xlabel('Forecast step')
    plt.ylabel('Pixel count')
    plt.xticks(x, [f'step{i}' for i in steps])
    plt.title('Rainfall pixel counts decrease with forecast step')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "RAIN_PIXEL_COUNT_BY_STEP.png"), dpi=300)
    plt.close()

    # ==============================
    # 图3：整体 RMSE、雨区 RMSE、强降雨 RMSE 对比
    # ==============================
    plt.figure(figsize=(8, 5))
    plt.plot(steps, rmse_all_list, marker='o', linewidth=2, label='RMSE_all')
    plt.plot(steps, rmse_rain_list, marker='s', linewidth=2, label='RMSE_rain: true >= 0.1 mm')
    plt.plot(steps, rmse_heavy_list, marker='^', linewidth=2, label='RMSE_heavy: true >= 5.0 mm')
    plt.plot(steps, rmse_reg_all_list, marker='d', linewidth=2, linestyle='--', label='RMSE_reg_only')

    plt.xlabel('Forecast step')
    plt.ylabel('RMSE / mm')
    plt.xticks(steps)
    plt.title('RMSE comparison under different evaluation masks')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "RMSE_ALL_RAIN_HEAVY_COMPARE.png"), dpi=300)
    plt.close()

    # ==============================
    # 图4：真实降雨分布箱线图
    # 用 log1p 是为了避免强降雨极端值把图拉得太扁
    # ==============================
    box_data = []
    labels = []

    for i in range(pre_step):
        ty = fore_y_mm[:, i, :].detach().cpu()
        valid_ty = ty[ty >= 0].numpy()

        # 用 log1p 压缩尺度：log(1 + rain)
        box_data.append(np.log1p(valid_ty))
        labels.append(f'step{i}')

    plt.figure(figsize=(8, 5))
    plt.boxplot(box_data, labels=labels, showfliers=False)
    plt.xlabel('Forecast step')
    plt.ylabel('log(1 + true rainfall)')
    plt.title('Distribution of true rainfall by forecast step')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "TRUE_RAINFALL_DISTRIBUTION_BOXPLOT.png"), dpi=300)
    plt.close()

    # ==============================
    # 图5：保存统计表，论文可以直接整理成表格
    # ==============================
    csv_path = os.path.join(save_dir, "step_distribution_rmse_summary.csv")

    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(
            "step,true_mean,true_max,pred_mean,"
            "rain_count_ge_0.1,heavy_count_ge_5.0,light_count_0.1_1.0,zero_count_lt_0.1,"
            "rmse_all,rmse_rain,rmse_heavy,rmse_reg_only\n"
        )

        for i in range(pre_step):
            f.write(
                f"{steps[i]},"
                f"{true_mean_list[i]:.6f},"
                f"{true_max_list[i]:.6f},"
                f"{pred_mean_list[i]:.6f},"
                f"{rain_count_list[i]},"
                f"{heavy_count_list[i]},"
                f"{light_count_list[i]},"
                f"{zero_count_list[i]},"
                f"{rmse_all_list[i]:.6f},"
                f"{rmse_rain_list[i]:.6f},"
                f"{rmse_heavy_list[i]:.6f},"
                f"{rmse_reg_all_list[i]:.6f}\n"
            )

    print(f"[Saved] step proof figures and csv saved to: {save_dir}")

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

        p_step = p_rain[:, i, :]

        for thr in [0.1, 2.0, 4.0, 8.0]:
            y_event = (ty >= thr)

            if thr == 0.1:
                # 有雨/无雨用分类头
                pred_event = (p_step >= p_thr)
            else:
                # 中雨、强降水用分类头 + 回归量联合判断
                pred_event = (p_step >= p_thr) & (to >= thr)

            yy = y_event[m]
            pp = pred_event[m]

            TP = (pp & yy).sum().item()
            FP = (pp & (~yy)).sum().item()
            FN = ((~pp) & yy).sum().item()

            pod = TP / (TP + FN + 1e-9)
            far = FP / (TP + FP + 1e-9)
            csi = TP / (TP + FP + FN + 1e-9)

            print(f"[thr={thr}] p_thr={p_thr:.2f} POD={pod:.4f} FAR={far:.4f} CSI={csi:.4f}")

    # 新增：画图证明 step RMSE 下降的原因
    plot_step_rmse_proof(
        outputs=outputs,
        y_hat_mm=y_hat_mm,
        fore_y_mm=fore_y_mm,
        pre_step=pre_step,
        save_dir="fig_step_proof"
    )

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

        outputs = y_hat_mm

        fore_y_mm = fore_y_mm.cpu()

        _eval_and_print(outputs, y_hat_mm, fore_y_mm, p_rain, pre_step)
