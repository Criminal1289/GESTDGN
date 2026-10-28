import os
import os.path
import random
import math
import json
import csv

import torch
import torch.nn as nn
import numpy as np
import measure
import torch.nn.functional as Fu

from torch.utils.data import TensorDataset, DataLoader
from precip_losses import probability_matching_loss, fourier_amplitude_correlation_loss, mse_pm_loss
from precip_split import validate_split_indices, assert_metadata_disjoint, save_split_manifest
from precip_metrics import precipitation_metrics, balanced_validation_score

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

cuda_gpu = torch.cuda.is_available()
p_thr = 0.5
save_dir = "./log"


def weighted_mm_smooth_l1_loss(pred_mm, target_mm, weights, scale_mm=10.0):
    """Direct rainfall error, with a SmoothL1 transition at scale_mm mm.

    Dividing both fields by a fixed physical scale keeps the auxiliary loss
    dimensionless without compressing rainfall through log1p.
    """
    if not math.isfinite(scale_mm) or scale_mm <= 0:
        raise ValueError("scale_mm must be finite and positive")
    per_pixel = Fu.smooth_l1_loss(
        pred_mm / scale_mm, target_mm / scale_mm, reduction="none", beta=1.0
    )
    return (per_pixel * weights).sum() / (weights.sum() + 1e-6)


def rain_coverage_loss(pred_mm, target_mm, grid_h=32, grid_w=32):
    """Soft coverage error at 7/15/40 mm, averaged per sample and lead.

    The same soft threshold is applied to both fields, so an exact prediction
    has zero loss. Local pooling includes only real grid cells at the border.
    """
    if pred_mm.ndim != 3 or pred_mm.shape != target_mm.shape:
        raise ValueError("Coverage loss expects matching [B, N, pre] rainfall fields")
    b, n, pre = pred_mm.shape
    if n != grid_h * grid_w:
        raise ValueError(f"Coverage loss expects {grid_h * grid_w} row-major nodes, got {n}")
    pred = pred_mm.permute(0, 2, 1).reshape(b * pre, 1, grid_h, grid_w)
    target = target_mm.permute(0, 2, 1).reshape(b * pre, 1, grid_h, grid_w)
    losses = []
    for threshold in (7.0, 15.0, 40.0):
        p = torch.sigmoid((pred - threshold) / 2.0)
        t = torch.sigmoid((target - threshold) / 2.0)
        for size in (1, 3):
            p_area = Fu.avg_pool2d(p, size, stride=1, padding=size // 2, count_include_pad=False)
            t_area = Fu.avg_pool2d(t, size, stride=1, padding=size // 2, count_include_pad=False)
            numerator = (p_area - t_area).square().mean(dim=(1, 2, 3))
            denominator = (p_area.square() + t_area.square()).mean(dim=(1, 2, 3))
            losses.append((numerator / denominator.clamp_min(1e-6)).mean())
    return torch.stack(losses).mean()


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


def regional_spectral_amse_loss(pred_mm, target_mm, grid_h=32, grid_w=32, eps=1e-6):
    """
    Regional PSD/coherence AMSE auxiliary loss for precipitation fields.
    This adapts the AMSE idea to a local 2D grid with FFT radial wave-number
    bins instead of spherical harmonics.
    """
    b, n, pre = pred_mm.shape
    if n != grid_h * grid_w:
        raise ValueError(f"regional_spectral_amse_loss expects {grid_h * grid_w} nodes, got {n}")

    pred = torch.clamp(pred_mm, min=0.0).permute(0, 2, 1).reshape(b * pre, grid_h, grid_w)
    target = torch.clamp(target_mm, min=0.0).permute(0, 2, 1).reshape(b * pre, grid_h, grid_w)

    pred = torch.log1p(pred)
    target = torch.log1p(target)
    pred = pred - pred.mean(dim=(-2, -1), keepdim=True)
    target = target - target.mean(dim=(-2, -1), keepdim=True)

    pred_fft = torch.fft.rfft2(pred, norm="ortho")
    target_fft = torch.fft.rfft2(target, norm="ortho")

    psd_pred = pred_fft.real.square() + pred_fft.imag.square()
    psd_target = target_fft.real.square() + target_fft.imag.square()
    cross = pred_fft * torch.conj(target_fft)

    fy = torch.fft.fftfreq(grid_h, device=pred_mm.device).view(grid_h, 1)
    fx = torch.fft.rfftfreq(grid_w, device=pred_mm.device).view(1, grid_w // 2 + 1)
    radial_bin = torch.round(torch.sqrt(fy.square() + fx.square()) * min(grid_h, grid_w)).long()
    radial_bin = radial_bin.clamp(max=min(grid_h, grid_w) // 2)
    max_bin = int(radial_bin.max().item())

    losses = []
    for k in range(1, max_bin + 1):
        mask = radial_bin == k
        if not torch.any(mask):
            continue

        p = psd_pred[:, mask].mean(dim=1)
        t = psd_target[:, mask].mean(dim=1)
        c = cross[:, mask].mean(dim=1)
        coh = c.abs() / torch.sqrt(p * t + eps)
        coh = coh.clamp(0.0, 1.0)

        amp_loss = (torch.sqrt(p + eps) - torch.sqrt(t + eps)).square()
        coh_loss = 2.0 * torch.maximum(p, t) * (1.0 - coh)
        norm = t.detach().mean().clamp(min=eps)
        losses.append(((amp_loss + coh_loss) / norm).mean())

    if len(losses) == 0:
        return pred.new_tensor(0.0)
    return torch.stack(losses).mean()


def core_intensity_loss(pred_mm, target_mm, grid_h=32, grid_w=32, topk=8):
    """Original core/peak/compactness supervision, with equal forecast-step weights."""
    b, n, pre = pred_mm.shape
    if n != grid_h * grid_w:
        raise ValueError(f"core_intensity_loss expects {grid_h * grid_w} nodes, got {n}")

    pred = torch.clamp(pred_mm, min=0.0).permute(0, 2, 1)
    target = torch.clamp(target_mm, min=0.0).permute(0, 2, 1)

    k = min(topk, n)
    target_top, idx = torch.topk(target, k=k, dim=2)
    pred_at_core = pred.gather(2, idx)
    loss_core = Fu.smooth_l1_loss(
        torch.log1p(pred_at_core), torch.log1p(target_top), reduction="none"
    ).mean(dim=2)

    # Compare at the observed peak location, not two independent maxima.
    target_peak, target_peak_idx = target.max(dim=2, keepdim=True)
    pred_at_target_peak = pred.gather(2, target_peak_idx)
    loss_peak = Fu.smooth_l1_loss(
        torch.log1p(pred_at_target_peak.squeeze(2)),
        torch.log1p(target_peak.squeeze(2)),
        reduction="none",
    )

    # Original light penalty on moderate-rain spillover in dry/light-rain pixels.
    false_region = (target < 1.5).to(pred.dtype)
    false_excess = Fu.relu(torch.log1p(pred) - math.log1p(7.0))
    loss_compact = (false_excess.square() * false_region).sum(dim=2) / (
        false_region.sum(dim=2) + 1e-6
    )
    loss_by_step = loss_core + 0.5 * loss_peak + 0.10 * loss_compact
    return loss_by_step.mean()


def soft_rain_area_loss(pred_mm, target_mm, threshold=5.0, sharpness=2.0):
    """
    Differentiable rain-area overlap loss on the regression output.
    It discourages broad false rain areas while allowing smooth boundaries.
    """
    pred_prob = torch.sigmoid((torch.clamp(pred_mm, min=0.0) - threshold) / sharpness)
    target_prob = torch.sigmoid((torch.clamp(target_mm, min=0.0) - threshold) / sharpness)

    reduce_dims = (1, 2)
    inter = (pred_prob * target_prob).sum(dim=reduce_dims)
    union = pred_prob.sum(dim=reduce_dims) + target_prob.sum(dim=reduce_dims)
    dice = (2.0 * inter + 1e-6) / (union + 1e-6)
    return 1.0 - dice.mean()


def false_moderate_rain_loss(pred_mm, target_mm, pred_threshold=5.0, true_threshold=2.0):
    """
    Penalize moderate/heavy false alarms where the observation is nearly dry.
    This targets broad positive-bias rain areas without suppressing true cores.
    """
    false_mask = (target_mm < true_threshold).float()
    excess = Fu.relu(pred_mm - pred_threshold)
    return (excess * false_mask).sum() / (false_mask.sum() + 1e-6)


def positive_bias_loss(pred_mm, target_mm, margin=2.0):
    """
    Penalize only obvious sample-wise positive step bias.
    A small margin avoids over-correcting acceptable forecast uncertainty.
    """
    step_bias = (pred_mm - target_mm).mean(dim=1)
    return Fu.relu(step_bias - margin).pow(2).mean()


def soft_false_alarm_loss(pred_mm, target_mm, pred_threshold=7.0, true_threshold=1.5, temperature=2.0):
    """
    Differentiable false-alarm penalty for moderate-rain expansion.
    It only acts where prediction is close to/above moderate rain but the
    observation remains below light rain, so weak rain bands are not erased.
    """
    pred_mask = torch.sigmoid((torch.clamp(pred_mm, min=0.0) - pred_threshold) / temperature)
    dry_target = torch.sigmoid((true_threshold - torch.clamp(target_mm, min=0.0)) / temperature)
    return (pred_mask * dry_target).mean()


def heavy_underestimation_loss(pred_mm, target_mm, threshold=15.0):
    """
    Protect heavy-rain cores from being suppressed by bias/false-alarm losses.
    Only underestimation at observed heavy-rain pixels is penalized.
    """
    heavy_mask = (target_mm >= threshold).float()
    under = Fu.relu(torch.log1p(torch.clamp(target_mm, min=0.0)) - torch.log1p(torch.clamp(pred_mm, min=0.0)))
    return (under.pow(2) * heavy_mask).sum() / (heavy_mask.sum() + 1e-6)


def nwp_physical_support(nwp, grid_h=32, grid_w=32, eps=1e-6):
    """
    Build NWP physical support from known sample channels.

    nwp_vars order:
    2=DVG850, 3=DVG925, 14=PWAT, 17=Q850, 18=RH1000, 26=WS925.
    Negative divergence is treated as convergence support.
    """
    x = nwp[:, -1].float()  # [B, Nw, C, H, W]
    valid = torch.isfinite(x) & (x > -9999.0)
    x = torch.where(valid, x, torch.zeros_like(x))

    count = valid.float().sum(dim=(-2, -1), keepdim=True).clamp(min=1.0)
    mean = x.sum(dim=(-2, -1), keepdim=True) / count
    var = ((x - mean).pow(2) * valid.float()).sum(dim=(-2, -1), keepdim=True) / count
    z = torch.where(valid, (x - mean) / torch.sqrt(var + eps), torch.zeros_like(x))

    z = z.mean(dim=1)  # [B, C, H, W]

    if z.shape[1] < 27:
        support = z.abs().mean(dim=1, keepdim=True)
    else:
        moisture = (
                0.40 * z[:, 14:15]      # PWAT
                + 0.35 * z[:, 17:18]    # Q850
                + 0.25 * z[:, 18:19]    # RH1000
        )
        convergence = (
                0.55 * (-z[:, 3:4])     # -DVG925
                + 0.45 * (-z[:, 2:3])   # -DVG850
        )
        wind = z[:, 26:27]              # WS925

        activity = z.abs().mean(dim=1, keepdim=True)
        moisture_gate = torch.sigmoid(moisture)
        conv_support = torch.sigmoid(convergence)
        wind_gate = torch.sigmoid(wind)
        support = 0.90 * moisture_gate * conv_support + 0.05 * wind_gate + 0.05 * activity

    support = Fu.interpolate(support, size=(grid_h, grid_w), mode="bilinear", align_corners=False)

    support_min = support.amin(dim=(-2, -1), keepdim=True)
    support_max = support.amax(dim=(-2, -1), keepdim=True)
    support = (support - support_min) / (support_max - support_min + eps)
    return support.reshape(nwp.shape[0], grid_h * grid_w)


def nwp_physical_consistency_loss(pred_mm, target_mm, nwp, grid_h=32, grid_w=32,
                                  pred_threshold=1.5, target_threshold=1.5,
                                  temperature=2.0):
    """
    Suppress false precipitation where both the observed rain and NWP-derived
    physical support are weak.
    """
    support = nwp_physical_support(nwp, grid_h=grid_h, grid_w=grid_w).unsqueeze(-1)
    support = support.expand(-1, -1, pred_mm.shape[-1])

    pred_rain = torch.sigmoid((torch.clamp(pred_mm, min=0.0) - pred_threshold) / temperature)
    weak_obs = torch.sigmoid((target_threshold - torch.clamp(target_mm, min=0.0)) / temperature)
    return (pred_rain * weak_obs * (1.0 - support)).mean()


def setup_seed(seed):
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def TrainValidTest(x, y, train=0.8, valid=None, shuffle=True, seed=20):
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
            setup_seed(seed)
            idx = torch.randperm(_x.shape[0])
            _x = _x[idx]
            _y = _y[idx]

        valid_x = _x[:valid_len].clone().detach().float()
        valid_y = _y[:valid_len].clone().detach().float()
        test_x = _x[valid_len:].clone().detach().float()
        test_y = _y[valid_len:].clone().detach().float()
        return train_x, train_y, valid_x, valid_y, test_x, test_y


def TrainValidTestSample(x, radar, y, train=0.8, valid=None, shuffle=True, seed=20):
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
            setup_seed(seed)
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
                 x, y, train=0.8, valid=None, device=None, radar=None, label=None,
                 seed=20, checkpoint_dir=None, mm_loss_weight=0.0, mm_loss_scale=10.0,
                 coverage_loss_weight=0.0, core_loss_weight=0.0,
                 pm_loss_weight=1.0, pm_loss_scale=10.0,
                 fal_loss_weight=0.0, fcl_loss_weight=0.0, spectral_loss_scale=10.0,
                 split_indices=None, sample_metadata=None, split_mode="legacy", valid_ratio=0.05,
                 log_dir=None, early_stopping_patience=0, restore_best=True,
                 checkpoint_metric="all", strong_validation_weight=0.25, loss_mode="mse_pm",
                 mse_lead_weights=None):
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
        self.seed = int(seed)
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir if log_dir is not None else save_dir
        self.split_indices = split_indices
        self.sample_metadata = sample_metadata
        self.split_mode = split_mode
        self.valid_ratio = float(valid_ratio)
        self.early_stopping_patience = int(early_stopping_patience)
        self.restore_best = bool(restore_best)
        if checkpoint_metric not in ('all', 'balanced'):
            raise ValueError('checkpoint_metric must be all or balanced')
        self.checkpoint_metric = checkpoint_metric
        self.strong_validation_weight = float(strong_validation_weight)
        if not math.isfinite(self.strong_validation_weight) or self.strong_validation_weight <= 0:
            raise ValueError('strong_validation_weight must be finite and positive')
        self.pm_loss_weight = float(pm_loss_weight)
        self.pm_loss_scale = float(pm_loss_scale)
        self.fal_loss_weight = float(fal_loss_weight)
        self.fcl_loss_weight = float(fcl_loss_weight)
        self.spectral_loss_scale = float(spectral_loss_scale)
        for name in ("pm_loss_weight", "fal_loss_weight", "fcl_loss_weight"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in ("pm_loss_scale", "spectral_loss_scale"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.epoch <= 0 or self.batch <= 0 or self.early_stopping_patience < 0:
            raise ValueError("epoch/batch must be positive and patience nonnegative")
        self.mm_loss_weight = float(mm_loss_weight)
        self.mm_loss_scale = float(mm_loss_scale)
        self.coverage_loss_weight = float(coverage_loss_weight)
        self.core_loss_weight = float(core_loss_weight)
        if loss_mode not in ('mse_pm', 'legacy'):
            raise ValueError('loss_mode must be mse_pm or legacy')
        self.loss_mode = loss_mode
        self.mse_lead_weights = None if mse_lead_weights is None else [float(w) for w in mse_lead_weights]
        if self.mse_lead_weights is not None:
            if (len(self.mse_lead_weights) != self.pre
                    or any(not math.isfinite(w) or w <= 0 for w in self.mse_lead_weights)):
                raise ValueError('mse_lead_weights must contain one finite positive weight per lead')
            if loss_mode != 'mse_pm':
                raise ValueError('mse_lead_weights is supported only for mse_pm')
        if loss_mode == 'mse_pm' and any(getattr(self, name) != 0 for name in
                ('mm_loss_weight', 'core_loss_weight', 'coverage_loss_weight', 'fal_loss_weight', 'fcl_loss_weight')):
            raise ValueError('mse_pm uses only MSE + PM; auxiliary weights must be 0 (use loss_mode="legacy" for the previous objective)')
        if not math.isfinite(self.core_loss_weight) or self.core_loss_weight < 0:
            raise ValueError("core_loss_weight must be finite and nonnegative")
        if not math.isfinite(self.coverage_loss_weight) or self.coverage_loss_weight < 0:
            raise ValueError("coverage_loss_weight must be finite and nonnegative")
        if not math.isfinite(self.mm_loss_weight) or self.mm_loss_weight < 0:
            raise ValueError("mm_loss_weight must be finite and nonnegative")
        if not math.isfinite(self.mm_loss_scale) or self.mm_loss_scale <= 0:
            raise ValueError("mm_loss_scale must be finite and positive")
        self.device = device if device is not None else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def _is_sample_mode(self):
        return getattr(self.model, "input_mode", "patch") == "sample"

    def _window_split_indices(self, count):
        if self.split_mode not in ("grouped", "legacy"):
            raise ValueError("split_mode must be grouped or legacy")
        if self.split_indices is None:
            if self.split_mode == "grouped":
                raise ValueError("Grouped splitting requires explicit indices and sample metadata")
            if not (0 < self.trainset < 1 and 0 < self.valid_ratio < 1
                    and self.trainset + self.valid_ratio < 1):
                raise ValueError("train/valid ratios must be positive and sum to less than 1")
            train_end = int(count * self.trainset)
            valid_count = max(1, int(count * self.valid_ratio))
            generator = torch.Generator().manual_seed(self.seed)
            rest = torch.randperm(count - train_end, generator=generator).numpy() + train_end
            self.split_indices = dict(train=np.arange(train_end), valid=rest[:valid_count],
                                      test=rest[valid_count:])
        self.split_indices = validate_split_indices(self.split_indices, count)
        if self.split_mode == "grouped":
            if self.sample_metadata is None or len(self.sample_metadata) != count:
                raise ValueError("Grouped split requires metadata for every window")
            assert_metadata_disjoint(self.sample_metadata, self.split_indices)
        self.test_indices = self.split_indices["test"]
        save_split_manifest(os.path.join(self.log_dir, "split_manifest.json"), self.split_indices,
                            self.sample_metadata, mode=self.split_mode, seed=self.seed)
        print(f"[SPLIT] mode={self.split_mode}, "
              + ", ".join(f"{name}={len(values)}" for name, values in self.split_indices.items()))
        return self.split_indices

    def _normalize_window_labels(self, y_np, train_indices):
        """Fit the rainfall transform on training labels only; retain held-out extremes."""
        y_mm = np.asarray(y_np, dtype=np.float32) * 0.1
        if not np.isfinite(y_mm).all() or (y_mm < 0).any():
            raise ValueError("Rainfall labels must be finite and nonnegative")
        train_mm = y_mm[train_indices]
        self.y_log_min = 0.0
        self.y_log_max = max(float(np.log1p(train_mm).max()), float(np.log1p(1.0)))
        np.save(os.path.join(self.log_dir, "y_log_min.npy"), np.array(self.y_log_min, dtype=np.float32))
        np.save(os.path.join(self.log_dir, "y_log_max.npy"), np.array(self.y_log_max, dtype=np.float32))
        self.label_stats = dict(fitted_on="train", y_log_min=self.y_log_min, y_log_max=self.y_log_max,
                                train_max_mm=float(train_mm.max()),
                                train_99th_percentile_mm=float(np.percentile(train_mm, 99)))
        with open(os.path.join(self.log_dir, "label_normalization.json"), "w", encoding="utf-8") as stream:
            json.dump(self.label_stats, stream, indent=2)
        # Do not clip targets >1: validation/test observations may exceed the train maximum.
        return np.log1p(y_mm) / (self.y_log_max - self.y_log_min + 1e-8)

    def get_test_data(self):
        """Return held-out windows in the layout/units expected by tester_cnn.test."""
        if not hasattr(self, "test_indices"):
            raise RuntimeError("Prepare/train the model before requesting test windows")
        if self._is_sample_mode():
            return dict(x=self.x[self.test_indices], y=self.y[self.test_indices],
                        radar=self.radar[self.test_indices])
        if self.x.ndim == 4:
            return dict(x=self.prepared_test_x.cpu().numpy(), y=self.y[self.test_indices])
        raise ValueError("External held-out testing requires prebuilt patch or sample windows")

    def _prepare_patch_data(self):
        os.makedirs(self.log_dir, exist_ok=True)
        nodes = self.x.shape[1]
        feature_num = self.x.shape[-1]

        # 兼容两种 patch 数据范式
        if self.x.ndim == 4:
            # x=(B,his,N,F), y=(B,pre,N)
            x_np = self.x
            y_np = self.y
            B, his, N, F = x_np.shape
            pre = y_np.shape[1]

            indices = self._window_split_indices(B)
            x_train = x_np[indices["train"]].reshape(-1, F)
            x_min, x_max = x_train.min(axis=0), x_train.max(axis=0)
            x_ones = (x_np - x_min) / (x_max - x_min + 1e-8)
            np.savez(os.path.join(self.log_dir, "patch_input_normalization.npz"),
                     x_min=x_min, x_max=x_max)
            y_ones = self._normalize_window_labels(y_np, indices["train"])
            hist_x = torch.tensor(x_ones, dtype=torch.float32)
            fore_y = torch.tensor(y_ones, dtype=torch.float32)
            self.prepared_test_x = hist_x[indices["test"]]
            return tuple(value for name in ("train", "valid", "test")
                         for value in (hist_x[indices[name]], fore_y[indices[name]]))
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
            np.save(os.path.join(self.log_dir, "y_log_min.npy"), np.array(0.0, dtype=np.float32))
            np.save(os.path.join(self.log_dir, "y_log_max.npy"), np.array(1.0, dtype=np.float32))

        train_x, train_y, valid_x, valid_y, test_x, test_y = TrainValidTest(
            hist_x, fore_y, train=self.trainset, valid=0.05, seed=self.seed
        )

        return train_x, train_y, valid_x, valid_y, test_x, test_y

    def _prepare_sample_data(self):
        if self.radar is None:
            raise ValueError("sample 模式下 trainer 需要额外传入 radar")

        # sample 输入不在 trainer 里做 CNN；只处理监督 y
        x_np = self.x
        radar_np = self.radar
        y_np = self.y  # [B, pre, 1024]

        os.makedirs(self.log_dir, exist_ok=True)
        indices = self._window_split_indices(y_np.shape[0])
        y_ones = self._normalize_window_labels(y_np, indices["train"])
        return tuple(torch.tensor(array[indices[name]], dtype=torch.float32)
                     for name in ("train", "valid", "test") for array in (x_np, radar_np, y_ones))

    def train(self):
        train_loss = []
        loss_components = {key: [] for key in
                           ("mse", "log_reg", "mm_reg", "weighted_mm_reg", "core", "weighted_core",
                            "coverage", "weighted_coverage", "pm", "weighted_pm",
                            "fal", "weighted_fal", "fcl", "weighted_fcl", "warmup")}
        print(f"[LOSS CONFIG] mode={self.loss_mode}, mse_pm_scale={self.pm_loss_scale:g} mm, "
              f"mm_weight={self.mm_loss_weight:g}, "
              f"mm_scale={self.mm_loss_scale:g} mm, core_weight={self.core_loss_weight:g}, "
              f"coverage_weight={self.coverage_loss_weight:g}, pm_weight={self.pm_loss_weight:g}, "
              f"pm_scale={self.pm_loss_scale:g} mm, fal_weight={self.fal_loss_weight:g}, "
              f"fcl_weight={self.fcl_loss_weight:g}, spectral_scale={self.spectral_loss_scale:g} mm")

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

            frontend = getattr(self.model, "frontend", None)
            if frontend is None or not hasattr(frontend, "fit_normalization"):
                raise RuntimeError("Sample mode requires a CNN frontend with fixed normalization support")
            frontend.fit_normalization(train_x, train_radar)
            np.savez(
                os.path.join(self.log_dir, "input_normalization_stats.npz"),
                **({name: getattr(frontend, name).detach().cpu().numpy()
                    for name in ('nwp_mean', 'nwp_std')} if hasattr(frontend, 'nwp_mean') else {}),
                radar_mean=frontend.radar_mean.detach().cpu().numpy(),
                radar_std=frontend.radar_std.detach().cpu().numpy(),
            )
            print(f"[NORM] fitted on {train_x.shape[0]} training samples, saved to {self.log_dir}")

        y_log_min = float(np.load(os.path.join(self.log_dir, "y_log_min.npy")))
        y_log_max = float(np.load(os.path.join(self.log_dir, "y_log_max.npy")))
        y_log_range = y_log_max - y_log_min + 1e-8

        def inv_y(y_norm):
            y_log = y_norm * y_log_range + y_log_min
            return torch.clamp(torch.expm1(y_log), min=0.0)

        def safe_rmse(pred, true, mask):
            if mask.sum().item() == 0:
                return float("nan")
            return measure.GetRMSE(pred[mask], true[mask])

            # train_y 统一按 (S, pre, N)
        setup_seed(self.seed)
        loader_generator = torch.Generator()
        loader_generator.manual_seed(self.seed)

        if not sample_mode:
            train_loader = DataLoader(
                dataset=TensorDataset(train_x, train_y),
                batch_size=self.batch,
                shuffle=True,
                generator=loader_generator
            )
        else:
            train_loader = DataLoader(
                dataset=TensorDataset(train_x, train_radar, train_y),
                batch_size=self.batch,
                shuffle=True,
                generator=loader_generator
            )

        Valid_y = inv_y(valid_y.cpu())
        best_valid_rmse = float('inf')
        best_valid_rain = float('nan')
        best_state = None
        best_selection_score = float('inf')
        best_all_score, best_strong_score = float('inf'), float('inf')
        best_all_epoch, best_strong_epoch = None, None
        validation_history, validation_lead_history = [], []
        checkpoint_template = None
        self.best_epoch = None
        epochs_without_improvement = 0
        if self.checkpoint_dir is not None:
            os.makedirs(self.checkpoint_dir, exist_ok=True)

        for e in range(self.epoch):
            epoch_loss = 0.0
            epoch_reg_loss = 0.0
            epoch_mse_loss = 0.0
            epoch_mm_loss = 0.0
            epoch_core_loss = 0.0
            epoch_coverage_loss = 0.0
            epoch_pm_loss = 0.0
            epoch_fal_loss = 0.0
            epoch_fcl_loss = 0.0
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

                    # 回归头反归一化到 mm
                    pred_train_mm_reg = inv_y(y_hat.detach())

                    # HARD-GATED 之前的逻辑：训练 RMSE 使用纯回归头
                    pred_train_mm = pred_train_mm_reg

                    # 计算训练集 all RMSE 的 SSE
                    diff_train = pred_train_mm - target_mm

                    train_sse_all += torch.sum(diff_train ** 2).item()
                    train_cnt_all += diff_train.numel()

                    # 计算训练集 rain RMSE 的 SSE
                    train_mask_rain = (target_mm >= 0.1)

                    if train_mask_rain.sum().item() > 0:
                        train_sse_rain += torch.sum(diff_train[train_mask_rain] ** 2).item()
                        train_cnt_rain += train_mask_rain.sum().item()

                pred_mm = inv_y(y_hat)
                # Both new losses supervise physical rainfall through inv_y.
                # Do not detach pred_mm: gradients must reach the regression head.
                zero = pred_mm.new_tensor(0.0)
                loss_mse = loss_reg = loss_mm = loss_core = loss_cover = loss_fal = loss_fcl = zero
                if self.loss_mode == 'mse_pm':
                    objective, loss_mse, loss_pm = mse_pm_loss(
                        pred_mm, target_mm, pm_weight=self.pm_loss_weight,
                        scale_mm=self.pm_loss_scale, mse_lead_weights=self.mse_lead_weights
                    )
                else:
                    per = Fu.smooth_l1_loss(y_hat, target, reduction="none")
                    w = torch.ones_like(target_mm)
                    w = torch.where(target_mm >= 0.1, torch.full_like(w, 1.2), w)
                    w = torch.where(target_mm >= 1.5, torch.full_like(w, 1.8), w)
                    w = torch.where(target_mm >= 7.0, torch.full_like(w, 3.0), w)
                    w = torch.where(target_mm >= 15.0, torch.full_like(w, 4.0), w)
                    w = torch.where(target_mm >= 40.0, torch.full_like(w, 5.0), w)
                    loss_reg = (per * w).sum() / (w.sum() + 1e-6)
                    loss_mm = weighted_mm_smooth_l1_loss(
                        pred_mm, target_mm, w, scale_mm=self.mm_loss_scale
                    )
                    if self.core_loss_weight > 0:
                        loss_core = core_intensity_loss(pred_mm, target_mm)
                    if self.coverage_loss_weight > 0:
                        loss_cover = rain_coverage_loss(pred_mm, target_mm)
                    loss_pm = (probability_matching_loss(pred_mm, target_mm, scale_mm=self.pm_loss_scale)
                               if self.pm_loss_weight > 0 else zero)
                    if self.fal_loss_weight > 0 or self.fcl_loss_weight > 0:
                        loss_fal, loss_fcl = fourier_amplitude_correlation_loss(
                            pred_mm, target_mm, scale_mm=self.spectral_loss_scale
                        )
                    objective = (loss_reg + self.mm_loss_weight * loss_mm
                                 + self.core_loss_weight * loss_core
                                 + self.coverage_loss_weight * loss_cover
                                 + self.pm_loss_weight * loss_pm
                                 + self.fal_loss_weight * loss_fal
                                 + self.fcl_loss_weight * loss_fcl)

                lambda_warmup = min(1.0, (e + 1) / 5.0)
                lambda_core = self.core_loss_weight
                loss = lambda_warmup * objective

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                epoch_mse_loss += loss_mse.item()
                epoch_reg_loss += loss_reg.item()
                epoch_mm_loss += loss_mm.item()
                epoch_core_loss += loss_core.item()
                epoch_coverage_loss += loss_cover.item()
                epoch_pm_loss += loss_pm.item()
                epoch_fal_loss += loss_fal.item()
                epoch_fcl_loss += loss_fcl.item()
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

            print(
                f"[LOSS] MSE={epoch_mse_loss / max(batch_count, 1):.6f}, "
                f"LogReg={epoch_reg_loss / max(batch_count, 1):.6f}, "
                f"MMReg={epoch_mm_loss / max(batch_count, 1):.6f}, "
                f"weighted_MMReg={self.mm_loss_weight * epoch_mm_loss / max(batch_count, 1):.6f}, "
                f"Core={epoch_core_loss / max(batch_count, 1):.6f}, "
                f"weighted_Core={lambda_core * epoch_core_loss / max(batch_count, 1):.6f}, "
                f"Coverage={epoch_coverage_loss / max(batch_count, 1):.6f}, "
                f"weighted_Coverage={self.coverage_loss_weight * epoch_coverage_loss / max(batch_count, 1):.6f}, "
                f"PM={epoch_pm_loss / max(batch_count, 1):.6f}, "
                f"weighted_PM={self.pm_loss_weight * epoch_pm_loss / max(batch_count, 1):.6f}, "
                f"FAL={epoch_fal_loss / max(batch_count, 1):.6f}, "
                f"weighted_FAL={self.fal_loss_weight * epoch_fal_loss / max(batch_count, 1):.6f}, "
                f"FCL={epoch_fcl_loss / max(batch_count, 1):.6f}, "
                f"weighted_FCL={self.fcl_loss_weight * epoch_fcl_loss / max(batch_count, 1):.6f} "
                "(components before shared warmup)"
            )
            train_loss.append(avg_train_loss)
            loss_components["mse"].append(epoch_mse_loss / max(batch_count, 1))
            loss_components["log_reg"].append(epoch_reg_loss / max(batch_count, 1))
            loss_components["mm_reg"].append(epoch_mm_loss / max(batch_count, 1))
            loss_components["weighted_mm_reg"].append(self.mm_loss_weight * epoch_mm_loss / max(batch_count, 1))
            loss_components["core"].append(epoch_core_loss / max(batch_count, 1))
            loss_components["weighted_core"].append(lambda_core * epoch_core_loss / max(batch_count, 1))
            loss_components["coverage"].append(epoch_coverage_loss / max(batch_count, 1))
            loss_components["weighted_coverage"].append(self.coverage_loss_weight * epoch_coverage_loss / max(batch_count, 1))
            for name, total, weight in (("pm", epoch_pm_loss, self.pm_loss_weight),
                                        ("fal", epoch_fal_loss, self.fal_loss_weight),
                                        ("fcl", epoch_fcl_loss, self.fcl_loss_weight)):
                loss_components[name].append(total / max(batch_count, 1))
                loss_components["weighted_" + name].append(weight * total / max(batch_count, 1))
            loss_components["warmup"].append(lambda_warmup)
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

                # 分类概率转成 (B, pre, N)，只用于观察分类头，不参与 RMSE
                p_valid = p.permute(0, 2, 1).contiguous().cpu()
                event = (p_valid >= p_thr).float()

                print("pred_event_ratio =", event.mean().item())
                print("p_valid min/mean/max =",
                      float(p_valid.min()), float(p_valid.mean()), float(p_valid.max()))

                valid_outputs_mm = valid_outputs_mm_reg

                mask_all = torch.ones_like(Valid_y, dtype=torch.bool)
                mask_rain = (Valid_y >= 0.1)

                rmse_all = safe_rmse(valid_outputs_mm, Valid_y, mask_all)
                rmse_rain = safe_rmse(valid_outputs_mm, Valid_y, mask_rain)

                valid_rmse_all_list.append(rmse_all)
                valid_rmse_rain_list.append(rmse_rain)

                print("valid RMSE_all :", rmse_all)
                print("valid RMSE_rain:", rmse_rain)
                metrics = precipitation_metrics(valid_outputs_mm, Valid_y)
                # Preserve the existing all/rain RMSE implementation and compatibility.
                metrics.update(rmse_all=float(rmse_all), rmse_rain=float(rmse_rain) if math.isfinite(rmse_rain) else None)
                balanced_score = balanced_validation_score(metrics, self.strong_validation_weight)
                selection_score = (balanced_score if self.checkpoint_metric == 'balanced' and balanced_score is not None
                                   else float(rmse_all))
                metrics.update(balanced_score=balanced_score, selection_score=selection_score)
                validation_history.append(dict(epoch=e+1, **metrics))
                for step in range(valid_outputs_mm.shape[1]):
                    validation_lead_history.append(dict(epoch=e+1, step=step,
                        **precipitation_metrics(valid_outputs_mm[:,step], Valid_y[:,step])))
                print(f"[VALID STRONG] RMSE_ge7={metrics['rmse_ge7']}, "
                      f"CSI_ge7={metrics['ge7_csi']}, POD_ge15={metrics['ge15_pod']}, "
                      f"area_ge01={metrics['ge01_area_ratio']}, score={selection_score:.6f}")
                improved_selected = math.isfinite(selection_score) and selection_score < best_selection_score
                improved_all = math.isfinite(rmse_all) and rmse_all < best_all_score
                improved_strong = balanced_score is not None and balanced_score < best_strong_score
                payload = None
                if improved_selected or improved_all or improved_strong:
                    state = {key:value.detach().cpu().clone() for key,value in self.model.state_dict().items()}
                    payload = dict(model_state_dict=state, epoch=e+1,
                        input_sources=getattr(self.model, 'input_sources', 'radar_nwp'),
                        nwp_features=getattr(self.model, 'nwp_features', None),
                        valid_rmse_all=float(rmse_all), valid_rmse_rain=float(rmse_rain),
                        validation_metrics=metrics, checkpoint_metric=self.checkpoint_metric,
                        strong_validation_weight=self.strong_validation_weight,
                        y_log_min=y_log_min, y_log_max=y_log_max,
                        loss_config={name:getattr(self,name) for name in
                            ('loss_mode','mse_lead_weights','mm_loss_weight','mm_loss_scale','core_loss_weight','coverage_loss_weight',
                             'pm_loss_weight','pm_loss_scale','fal_loss_weight','fcl_loss_weight','spectral_loss_scale')},
                        split_mode=self.split_mode, seed=self.seed,
                        split_indices=({k:v.tolist() for k,v in self.split_indices.items()}
                                       if self.split_indices is not None else None))
                    checkpoint_template = payload
                if improved_all:
                    best_all_score, best_all_epoch = float(rmse_all), e+1
                    if self.checkpoint_dir is not None:
                        torch.save(payload, os.path.join(self.checkpoint_dir, 'best_all.pt'))
                if improved_strong:
                    best_strong_score, best_strong_epoch = float(balanced_score), e+1
                    if self.checkpoint_dir is not None:
                        torch.save(payload, os.path.join(self.checkpoint_dir, 'best_strong.pt'))
                if improved_selected:
                    best_selection_score = float(selection_score)
                    best_valid_rmse = float(rmse_all)
                    best_valid_rain = float(rmse_rain)
                    self.best_epoch = e + 1
                    epochs_without_improvement = 0
                    best_state = payload['model_state_dict']
                    if self.checkpoint_dir is not None:
                        torch.save(payload, os.path.join(self.checkpoint_dir, 'best.pt'))
                else:
                    epochs_without_improvement += 1

                self.model.train()

            if self.early_stopping_patience and epochs_without_improvement >= self.early_stopping_patience:
                print(f"[EARLY STOP] epoch={e + 1}, best_epoch={self.best_epoch}, "
                      f"patience={self.early_stopping_patience}")
                break

        if self.checkpoint_dir is not None and validation_history and checkpoint_template is not None:
            # Save the actual final epoch before restoring a validation-selected model.
            final_payload = dict(checkpoint_template)
            final_payload.update(model_state_dict={k:v.detach().cpu().clone() for k,v in self.model.state_dict().items()},
                                 epoch=len(train_loss), validation_metrics=validation_history[-1],
                                 valid_rmse_all=float(rmse_all), valid_rmse_rain=float(rmse_rain))
            torch.save(final_payload, os.path.join(self.checkpoint_dir, 'last.pt'))
        for filename, rows in (('validation_metrics.csv',validation_history),
                               ('validation_lead_metrics.csv',validation_lead_history)):
            if rows:
                with open(os.path.join(self.log_dir,filename),'w',newline='',encoding='utf-8') as stream:
                    writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
        if best_state is None:
            raise RuntimeError("No finite validation score; refusing to evaluate an unselected model")
        self.best_valid_rmse = best_valid_rmse
        if self.restore_best:
            self.model.load_state_dict(best_state)
            print(f"[RESTORE] best epoch={self.best_epoch}, valid_RMSE_all={best_valid_rmse:.6f}")
        self.model.eval()
        with open(os.path.join(self.log_dir, "training_summary.json"), "w", encoding="utf-8") as stream:
            json.dump(dict(epochs_run=len(train_loss), best_epoch=self.best_epoch,
                           best_valid_rmse_all=best_valid_rmse,
                           best_valid_rmse_rain=best_valid_rain if math.isfinite(best_valid_rain) else None,
                           restored_best=self.restore_best, split_mode=self.split_mode,
                           evaluation_checkpoint=('best' if self.restore_best else 'last'),
                           evaluation_epoch=(self.best_epoch if self.restore_best else len(train_loss)),
                           loss_mode=self.loss_mode, mse_lead_weights=self.mse_lead_weights,
                           input_sources=getattr(self.model, 'input_sources', 'radar_nwp'),
                           nwp_features=getattr(self.model, 'nwp_features', None),
                           pm_loss_weight=self.pm_loss_weight,
                           pm_loss_scale=self.pm_loss_scale,
                           early_stopping_patience=self.early_stopping_patience,
                           checkpoint_metric=self.checkpoint_metric,
                           strong_validation_weight=self.strong_validation_weight,
                           best_selection_score=best_selection_score,
                           best_all_epoch=best_all_epoch, best_all_valid_rmse=best_all_score,
                           best_strong_epoch=best_strong_epoch,
                           best_strong_score=best_strong_score if best_strong_epoch is not None else None,
                           effective_checkpoint_metric=('balanced' if self.checkpoint_metric=='balanced' and best_strong_epoch is not None else 'all')),
                      stream, indent=2, allow_nan=False)

        model_name = self.model.__class__.__name__
        if model_name != 'STDGN_woa':
            np.savez(
                os.path.join(self.log_dir, "train_Loss_components_" + model_name + ".npz"),
                **{key: np.asarray(values) for key, values in loss_components.items()},
                loss_mode=np.asarray(self.loss_mode),
                mse_lead_weights=np.asarray(self.mse_lead_weights if self.mse_lead_weights is not None
                                            else [1.0] * self.pre),
                mm_loss_weight=np.asarray(self.mm_loss_weight),
                mm_loss_scale=np.asarray(self.mm_loss_scale),
                coverage_loss_weight=np.asarray(self.coverage_loss_weight),
                core_loss_weight=np.asarray(self.core_loss_weight),
                pm_loss_weight=np.asarray(self.pm_loss_weight),
                pm_loss_scale=np.asarray(self.pm_loss_scale),
                fal_loss_weight=np.asarray(self.fal_loss_weight),
                fcl_loss_weight=np.asarray(self.fcl_loss_weight),
                spectral_loss_scale=np.asarray(self.spectral_loss_scale),
            )
            np.save(os.path.join(self.log_dir, "train_Loss_" + model_name + '.npy'),
                    np.array(train_loss))

            np.save(os.path.join(self.log_dir, "train_RMSE_all_" + model_name + '.npy'),
                    np.array(train_rmse_all_list))

            np.save(os.path.join(self.log_dir, "train_RMSE_rain_" + model_name + '.npy'),
                    np.array(train_rmse_rain_list))

            np.save(os.path.join(self.log_dir, "valid_RMSE_all_" + model_name + '.npy'),
                    np.array(valid_rmse_all_list))

            np.save(os.path.join(self.log_dir, "valid_RMSE_rain_" + model_name + '.npy'),
                    np.array(valid_rmse_rain_list))

        if self.valid is not None:
            selected_rmse = best_valid_rmse if self.restore_best else rmse_all
            print('valid result (all): ', selected_rmse)
            return selected_rmse, self.model.cpu()

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

            np.savez(os.path.join(self.log_dir, "test_predictions.npz"),
                     pred_mm=test_outputs_reg_mm.numpy(), target_mm=test_y_mm.numpy(),
                     indices=(self.test_indices if hasattr(self, "test_indices")
                              else np.arange(test_y_mm.shape[0])),
                     checkpoint=np.asarray('best' if self.restore_best else 'last'),
                     epoch=np.asarray(self.best_epoch if self.restore_best else len(train_loss)))

            print("RMSE(REG all):", rmse_reg_all)
            print("RMSE(REG rain):", rmse_reg_rain)

            p_stats = p.detach().cpu()
            print("p stats(min/mean/max) =",
                  float(p_stats.min()), float(p_stats.mean()), float(p_stats.max()))

        return rmse_reg_rain, self.model.cpu()
