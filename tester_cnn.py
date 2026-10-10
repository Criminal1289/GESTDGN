#!/usr/bin/env python
# -*- coding:utf-8 -*-
import glob
import csv
import json
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import numpy as np
import matplotlib.pyplot as plt
from trainer_cnn import trainer, Divide, TrainValidTest, TrainValidTestSample
from model_cnn_tucancha_featureAttention import STDGN
import measure

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
SAMPLE_GLOB = r"data/cnn_spatial_align_slide/*.npz"
_GE_SAMPLE_CACHE = {}

p_thr = 0.5


def _sample_value(raw, candidates, path):
    for key in candidates:
        if key in raw:
            return raw[key]
    raise KeyError(f"{os.path.basename(path)} missing keys {candidates}; actual={list(raw.keys())}")


def _clean_sample_nwp(nwp):
    nwp = nwp.astype(np.float32, copy=True)
    invalid = (~np.isfinite(nwp)) | (nwp <= -9999.0)
    nwp[invalid] = np.nan
    return nwp


def build_ge_samples_from_raw(sample_files, his_len, pre_len):
    """Build the same NWP/Radar targets used by main_cnn sample training."""
    nwp_list, radar_list, target_list = [], [], []
    needed_t = his_len + pre_len - 1

    for path in sample_files:
        with np.load(path, allow_pickle=True) as raw:
            nwp = _clean_sample_nwp(_sample_value(raw, ['nwp', 'NWP', 'x_nwp'], path))
            radar = _sample_value(raw, ['radar', 'RADAR', 'x_radar'], path).astype(np.float32)
            label = _sample_value(raw, ['label', 'LABEL', 'y', 'rain', 'target'], path).astype(np.float32)

        if label.ndim == 4 and label.shape[1] == 1:
            label = label[:, 0]
        if nwp.ndim != 5 or radar.ndim != 4 or label.ndim != 3:
            raise ValueError(
                f"Unexpected sample shapes in {os.path.basename(path)}: "
                f"nwp={nwp.shape}, radar={radar.shape}, label={label.shape}"
            )
        if not (nwp.shape[0] == radar.shape[0] == label.shape[0]):
            raise ValueError(
                f"Time length mismatch in {os.path.basename(path)}: "
                f"nwp={nwp.shape[0]}, radar={radar.shape[0]}, label={label.shape[0]}"
            )

        total_t = nwp.shape[0]
        if total_t < needed_t:
            continue

        for start in range(total_t - needed_t + 1):
            target_start = start + his_len - 1
            y_label = label[target_start:target_start + pre_len]

            y_small = []
            for field in y_label:
                height, width = field.shape
                stride_h, stride_w = height // 32, width // 32
                field_32 = field[:32 * stride_h, :32 * stride_w].reshape(
                    32, stride_h, 32, stride_w
                ).mean(axis=(1, 3))
                y_small.append(field_32.reshape(-1))

            nwp_list.append(nwp[start:start + his_len])
            radar_list.append(radar[start:start + his_len])
            target_list.append(np.stack(y_small, axis=0))

    if not nwp_list:
        raise RuntimeError(
            "GE fitness_cal: no sample windows generated; check sample length and his/pre settings."
        )

    return np.stack(nwp_list), np.stack(radar_list), np.stack(target_list)


def _get_ge_sample_data(his_len, pre_len):
    cache_key = (int(his_len), int(pre_len))
    if cache_key not in _GE_SAMPLE_CACHE:
        sample_files = sorted(glob.glob(SAMPLE_GLOB))
        if not sample_files:
            raise RuntimeError(f"GE fitness_cal: no sample files matched {SAMPLE_GLOB}")
        _GE_SAMPLE_CACHE[cache_key] = build_ge_samples_from_raw(sample_files, his_len, pre_len)
        print(f"[GE sample] files={len(sample_files)}, windows={_GE_SAMPLE_CACHE[cache_key][0].shape[0]}")
    return _GE_SAMPLE_CACHE[cache_key]


def fitness_cal(adjmat, device="cuda:0", his_len=None, pre_len=None):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    x_nwp, x_radar, y = _get_ge_sample_data(his_len, pre_len)

    adj = torch.tensor(adjmat, dtype=torch.float32, device=device)

    model = STDGN(
        num_units=64,
        adj=adj,
        num_nodes=y.shape[2],
        num_feature=37,
        num_head=2,
        seq_len=his_len,
        pre_len=pre_len,
        device=device,
        deep=1,
        input_mode="sample"
    ).to(device)

    model_train = trainer(
        model=model,
        epoch=10,
        batch=8,
        lr=1e-3,
        decay=0,
        his_step=his_len,
        pre_step=pre_len,
        x=x_nwp,
        radar=x_radar,
        y=y,
        train=0.8,
        valid=None,
        device=device,
        seed=20
    )
    rmse, _ = model_train.train()
    return rmse, model


def _is_sample_mode(model):
    return getattr(model, "input_mode", "patch") == "sample"


def _load_y_norm_params(norm_dir="log"):
    y_log_min = float(np.load(os.path.join(norm_dir, "y_log_min.npy")))
    y_log_max = float(np.load(os.path.join(norm_dir, "y_log_max.npy")))
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
def rainfall_event_metrics(pred_mm, true_mm, threshold):
    """Regression-only threshold scores; no untrained classification gate."""
    if pred_mm.shape != true_mm.shape:
        raise ValueError("Event metrics require matching rainfall fields")
    valid = torch.isfinite(true_mm) & (true_mm >= 0)
    pp = (pred_mm[valid] >= threshold)
    yy = (true_mm[valid] >= threshold)
    tp = int((pp & yy).sum().item())
    fp = int((pp & ~yy).sum().item())
    fn = int((~pp & yy).sum().item())
    true_count, pred_count = tp + fn, tp + fp

    def ratio(numerator, denominator):
        return numerator / denominator if denominator else float("nan")

    return dict(tp=tp, fp=fp, fn=fn, true_count=true_count, pred_count=pred_count,
                pod=ratio(tp, true_count), far=ratio(fp, pred_count),
                csi=ratio(tp, tp + fp + fn), area_ratio=ratio(pred_count, true_count))


def _print_regression_event_metrics(pred_mm, true_mm):
    print("=== Regression rainfall event metrics (all steps) ===")
    for threshold in (0.1, 1.5, 7.0, 15.0, 40.0):
        metrics = rainfall_event_metrics(pred_mm, true_mm, threshold)
        print(f"[thr={threshold:g} mm] POD={metrics['pod']:.4f} FAR={metrics['far']:.4f} "
              f"CSI={metrics['csi']:.4f} area_ratio={metrics['area_ratio']:.4f}")


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

def plot_step_error_maps(outputs, fore_y_mm, pre_step, save_dir="fig_step_error_maps", sample_idx=None, grid_shape=None, max_steps=4, sample_info=None):
    """
    绘制 step0-step3 的空间误差图：
    第一行：GT
    第二行：Pred
    第三行：Error = Pred - GT

    outputs: 预测结果，shape = [B, pre_step, N]，单位 mm
    fore_y_mm: 真实降水，shape = [B, pre_step, N]，单位 mm
    """

    os.makedirs(save_dir, exist_ok=True)

    outputs = outputs.detach().cpu()
    fore_y_mm = fore_y_mm.detach().cpu()

    B, P, N = fore_y_mm.shape
    steps_to_plot = min(pre_step, max_steps, P)

    # 自动推断网格大小，例如 N=1024 -> 32x32
    if grid_shape is None:
        side = int(np.sqrt(N))
        if side * side != N:
            raise ValueError(
                f"N={N} 不能自动 reshape 成正方形网格，请手动设置 grid_shape，例如 grid_shape=(32, 32)"
            )
        grid_shape = (side, side)

    # 默认选择降雨最明显的样本，避免随机挑到几乎无雨样本
    selection = "auto_max_total_rain" if sample_idx is None else "fixed_index"
    if sample_idx is None:
        score = fore_y_mm[:, :steps_to_plot, :].sum(dim=(1, 2))
        sample_idx = int(torch.argmax(score).item())
    if not 0 <= sample_idx < B:
        raise ValueError(f"Error-map sample index {sample_idx} is outside [0, {B})")

    # 统一 GT 和 Pred 的色标，方便 step0-3 横向对比
    gt_sel = fore_y_mm[sample_idx, :steps_to_plot, :]
    pred_sel = outputs[sample_idx, :steps_to_plot, :]

    rain_vmax = max(
        float(gt_sel.max().item()),
        float(pred_sel.max().item()),
        1.0
    )

    err_sel = pred_sel - gt_sel
    err_abs_max = max(float(torch.abs(err_sel).max().item()), 1.0)

    fig, axes = plt.subplots(
        3, steps_to_plot,
        figsize=(4.2 * steps_to_plot, 9.5)
    )

    if steps_to_plot == 1:
        axes = axes.reshape(3, 1)

    rain_im = None
    err_im = None

    step_metrics = []
    for i in range(steps_to_plot):
        gt = fore_y_mm[sample_idx, i, :].reshape(grid_shape).numpy()
        pred = outputs[sample_idx, i, :].reshape(grid_shape).numpy()
        err = pred - gt

        mae = np.mean(np.abs(err))
        rmse = np.sqrt(np.mean(err ** 2))
        bias = np.mean(err)
        step_metrics.append(dict(step=i, true_max_mm=float(gt.max()), pred_max_mm=float(pred.max()),
                                 mae_mm=float(mae), rmse_mm=float(rmse), bias_mm=float(bias)))

        # =====================
        # GT
        # =====================
        rain_im = axes[0, i].imshow(
            gt,
            cmap="jet",
            vmin=0,
            vmax=rain_vmax
        )
        axes[0, i].set_title(
            f"GT Step {i}\nmax={gt.max():.1f} mm",
            fontsize=10
        )
        axes[0, i].axis("off")

        # =====================
        # Pred
        # =====================
        axes[1, i].imshow(
            pred,
            cmap="jet",
            vmin=0,
            vmax=rain_vmax
        )
        axes[1, i].set_title(
            f"Pred Step {i}\nmax={pred.max():.1f} mm",
            fontsize=10
        )
        axes[1, i].axis("off")

        # =====================
        # Error = Pred - GT
        # =====================
        err_im = axes[2, i].imshow(
            err,
            cmap="RdBu_r",
            vmin=-err_abs_max,
            vmax=err_abs_max
        )
        axes[2, i].set_title(
            f"Error Step {i}\nMAE={mae:.2f}, RMSE={rmse:.2f}, Bias={bias:.2f}",
            fontsize=9
        )
        axes[2, i].axis("off")

    cbar1 = fig.colorbar(
        rain_im,
        ax=axes[0:2, :].ravel().tolist(),
        shrink=0.75,
        pad=0.02
    )
    cbar1.set_label("Precipitation / mm")

    cbar2 = fig.colorbar(
        err_im,
        ax=axes[2, :].ravel().tolist(),
        shrink=0.75,
        pad=0.02
    )
    cbar2.set_label("Error: Pred - GT / mm")

    identity = dict(sample_info or {})
    title_suffix = ""
    if identity:
        parts = [f"global index {identity['global_index']}"] if 'global_index' in identity else []
        if identity.get('partition'):
            parts.append(str(identity['partition']))
        if identity.get('checkpoint_epoch') is not None:
            parts.append(f"epoch {identity['checkpoint_epoch']}")
        if identity.get('issue_time'):
            # Source files can omit the year; do not display the parser's anchor year.
            parts.append(str(identity['issue_time'])[5:16].replace('T', ' '))
        title_suffix = "\n" + " | ".join(parts)
    fig.suptitle(
        f"Step 0-{steps_to_plot - 1} precipitation prediction error maps{title_suffix}",
        fontsize=14
    )

    plt.savefig(
        os.path.join(save_dir, "STEP0_3_GT_PRED_ERROR_MAPS.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()
    with open(os.path.join(save_dir, "sample_metadata.json"), "w", encoding="utf-8") as stream:
        json.dump(dict(identity, local_index=int(sample_idx), selection=selection, steps=step_metrics),
                  stream, ensure_ascii=False, indent=2)

    print(f"[Saved] step0-3 error maps saved to: {save_dir}")

def _eval_and_print(outputs, y_hat_mm, fore_y_mm, p_rain, pre_step, output_dir=".",
                    error_map_save_dir=None, sample_indices=None, sample_metadata=None,
                    checkpoint_info=None):
    mask = torch.ones_like(fore_y_mm, dtype=torch.bool)

    rmse = measure.GetRMSE(outputs[mask], fore_y_mm[mask])
    mae = measure.GetMAP(outputs[mask], fore_y_mm[mask])
    print("整体RMSE:", rmse)
    print("整体MAE:", mae)

    # event_true = (fore_y_mm >= 0.1)

    print("pred min/mean/max:", outputs.min().item(), outputs.mean().item(), outputs.max().item())
    print("true min/mean/max:", fore_y_mm.min().item(), fore_y_mm.mean().item(), fore_y_mm.max().item())

    _print_regression_event_metrics(outputs, fore_y_mm)
    event_rows = []

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

        for thr in (0.1, 1.5, 7.0, 15.0, 40.0):
            metrics = rainfall_event_metrics(to, ty, thr)
            event_rows.append(dict(step=i, threshold_mm=thr, **metrics))
            print(f"[thr={thr:g} mm] POD={metrics['pod']:.4f} FAR={metrics['far']:.4f} "
                  f"CSI={metrics['csi']:.4f} area_ratio={metrics['area_ratio']:.4f}")

    proof_dir = os.path.join(output_dir, "fig_step_proof")
    os.makedirs(proof_dir, exist_ok=True)
    with open(os.path.join(proof_dir, "step_event_metrics_summary.csv"), "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(event_rows[0]))
        writer.writeheader()
        writer.writerows(event_rows)

    # 新增：画 step0-step3 的 GT / Pred / Error 空间误差图
    case_info = dict(checkpoint_info or {})
    if sample_indices is not None:
        local_index = int(fore_y_mm[:, :min(pre_step, 4), :].sum(dim=(1, 2)).argmax().item())
        global_index = int(sample_indices[local_index])
        case_info = dict(global_index=global_index, partition="test")
        if sample_metadata is not None:
            case_info.update(sample_metadata[global_index])
        case_info.update(checkpoint_info or {})
    plot_step_error_maps(
            outputs=outputs,
            fore_y_mm=fore_y_mm,
            pre_step=pre_step,
            save_dir=error_map_save_dir or os.path.join(output_dir, "fig_step_error_maps"),
            sample_idx=None,
            grid_shape=(32, 32),
            max_steps=4,
            sample_info=case_info
    )
    # 新增：画图证明 step RMSE 下降的原因
    plot_step_rmse_proof(
        outputs=outputs,
        y_hat_mm=y_hat_mm,
        fore_y_mm=fore_y_mm,
        pre_step=pre_step,
        save_dir=proof_dir
    )

def save_fixed_case_error_map(model, pre_step, x, y, radar, sample_info,
                             norm_dir="log", output_dir="."):
    """Draw one explicit diagnostic case without including it in test scores."""
    if not _is_sample_mode(model):
        raise ValueError("Fixed raw-sample diagnostics require sample mode")
    if len(x) != 1 or len(y) != 1 or len(radar) != 1:
        raise ValueError("Fixed-case diagnostics must contain exactly one window")
    model = model.to(device)
    model.eval()
    y_log_min, _, y_log_range = _load_y_norm_params(norm_dir)
    with torch.inference_mode():
        y_hat, _ = model(torch.as_tensor(x, dtype=torch.float32, device=device),
                         torch.as_tensor(radar, dtype=torch.float32, device=device))
        pred_mm = _inv_y(y_hat.permute(0, 2, 1).cpu(), y_log_min, y_log_range)
    target_mm = (torch.as_tensor(y, dtype=torch.float32) * 0.1).clamp_min(0.0)
    plot_step_error_maps(pred_mm, target_mm, pre_step,
                         save_dir=os.path.join(output_dir, "fig_step_error_maps"),
                         sample_idx=0, grid_shape=(32, 32), sample_info=sample_info)


def test(model, his_step, pre_step, x, y, radar=None, norm_dir="log", output_dir=".",
         sample_indices=None, sample_metadata=None, error_map_data=None):
    model = model.to(device)
    model.eval()

    sample_mode = _is_sample_mode(model)
    y_log_min, _, y_log_range = _load_y_norm_params(norm_dir)
    checkpoint_info = {}
    summary_path = os.path.join(norm_dir, 'training_summary.json')
    if os.path.isfile(summary_path):
        with open(summary_path, encoding='utf-8') as stream:
            summary = json.load(stream)
        checkpoint = summary.get('evaluation_checkpoint', 'best' if summary.get('restored_best', True) else 'last')
        checkpoint_info = dict(checkpoint=checkpoint, checkpoint_epoch=summary.get('evaluation_epoch',
            summary['best_epoch'] if checkpoint == 'best' else summary['epochs_run']))

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

        _eval_and_print(outputs, y_hat_mm, fore_y_mm, p_rain, pre_step, output_dir=output_dir,
                        error_map_save_dir=(os.path.join(output_dir, "fig_test_error_maps")
                                            if error_map_data is not None else None),
                        sample_indices=sample_indices, sample_metadata=sample_metadata,
                        checkpoint_info=checkpoint_info)

    if error_map_data is not None:
        save_fixed_case_error_map(model, pre_step, norm_dir=norm_dir,
                                 output_dir=output_dir, **error_map_data)
