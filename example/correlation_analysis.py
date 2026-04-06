import os
import glob
import random
from typing import Dict, List, Tuple

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# =========================================================
# 0. 配置
# =========================================================
DATA_DIR = r"C:/Users/17878/Desktop/gestdgn/data/cnn_spatial_align_slide"   # sample_*.npz 所在目录
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 时序设置
SEQ_LEN = 4
PRED_LEN = 2

# 阈值（你的 RA 若是 0.1mm 单位，这里可改成 1.0 或 0.1）
RAIN_THR = 1.0

# 训练设置
BATCH_SIZE = 2
EPOCHS = 3
LR = 1e-3
RANDOM_SEED = 42

# 内部相关性抽样点数
MAX_POINTS_INTERNAL = 10000

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


# =========================================================
# 1. 数据读取
# =========================================================
def load_one_npz(npz_path: str):
    data = np.load(npz_path, allow_pickle=True)
    nwp = data["nwp"].astype(np.float32)       # [T, Nw, Cn, Hn, Wn]
    radar = data["radar"].astype(np.float32)   # [T, Cr, 128, 128]
    label = data["label"].astype(np.float32)   # [T, 1, 128, 128]

    nwp_vars = [str(x) for x in data["nwp_vars"].tolist()]
    radar_vars = [str(x) for x in data["radar_vars"].tolist()]
    cp_name = str(data["cp_name"].tolist()) if np.ndim(data["cp_name"]) == 0 else str(data["cp_name"])

    return {
        "nwp": nwp,
        "radar": radar,
        "label": label,
        "nwp_vars": nwp_vars,
        "radar_vars": radar_vars,
        "cp_name": cp_name,
    }


def load_all_npz(data_dir: str):
    files = sorted(glob.glob(os.path.join(data_dir, "sample_*.npz")))
    if not files:
        raise RuntimeError(f"在目录 {data_dir} 下没有找到 sample_*.npz")

    all_data = [load_one_npz(f) for f in files]

    # 默认所有样本的变量顺序一致
    nwp_vars = all_data[0]["nwp_vars"]
    radar_vars = all_data[0]["radar_vars"]
    feature_names = nwp_vars + radar_vars

    return all_data, feature_names


# =========================================================
# 2. 特征聚合
# =========================================================
def aggregate_nwp_features(nwp_t: np.ndarray) -> np.ndarray:
    """
    nwp_t: [Nw, Cn, Hn, Wn]
    输出: [Cn]
    做法: 对窗口和空间全部取均值，得到每个NWP变量的一个标量
    """
    return nwp_t.mean(axis=(0, 2, 3)).astype(np.float32)


def aggregate_radar_features(radar_t: np.ndarray) -> np.ndarray:
    """
    radar_t: [Cr, Hr, Wr]
    输出: [Cr]
    做法: 对空间取均值，得到每个雷达变量的一个标量
    """
    return radar_t.mean(axis=(1, 2)).astype(np.float32)


def aggregate_label_value(label_t: np.ndarray) -> float:
    """
    label_t: [1, H, W]
    输出: 标量
    做法: 对整幅降水场取均值
    """
    return float(label_t[0].mean())


# =========================================================
# 3. 构造时序样本
# =========================================================
def build_sequence_samples(all_data: List[Dict], seq_len=4, pred_len=1):
    """
    每个样本构造成：
      X: [seq_len, F]
      y: [pred_len]
    其中 F = 27 + 10 = 37
    """
    samples = []

    for item in all_data:
        nwp = item["nwp"]
        radar = item["radar"]
        label = item["label"]
        cp_name = item["cp_name"]

        T = nwp.shape[0]
        for end_idx in range(seq_len - 1, T - pred_len):
            hist_idx = list(range(end_idx - seq_len + 1, end_idx + 1))
            fut_idx = list(range(end_idx + 1, end_idx + 1 + pred_len))

            x_seq = []
            for ti in hist_idx:
                nwp_feat = aggregate_nwp_features(nwp[ti])
                radar_feat = aggregate_radar_features(radar[ti])
                feat = np.concatenate([nwp_feat, radar_feat], axis=0)  # [F]
                x_seq.append(feat)

            y_seq = []
            for ti in fut_idx:
                y_seq.append(aggregate_label_value(label[ti]))

            x_seq = np.stack(x_seq, axis=0).astype(np.float32)   # [seq_len, F]
            y_seq = np.asarray(y_seq, dtype=np.float32)          # [pred_len]
            samples.append((x_seq, y_seq, cp_name))

    return samples


# =========================================================
# 4. 数据集
# =========================================================
class SequenceDataset(Dataset):
    def __init__(self, samples, feature_indices=None):
        self.samples = samples
        self.feature_indices = feature_indices

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y, _ = self.samples[idx]
        if self.feature_indices is not None:
            x = x[:, self.feature_indices]
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )


# =========================================================
# 5. 轻量预测模型
# =========================================================
class FeatureGRU(nn.Module):
    """
    输入: [B, T, F]
    输出: [B, P]
    """
    def __init__(self, in_dim, hidden_dim=64, num_layers=1, pred_len=1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=in_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_dim, pred_len)

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.head(out)
        return out


# =========================================================
# 6. 指标
# =========================================================
def compute_metrics(pred: np.ndarray, true: np.ndarray, rain_thr=1.0):
    pred = np.asarray(pred).reshape(-1)
    true = np.asarray(true).reshape(-1)

    rmse = np.sqrt(np.mean((pred - true) ** 2))
    mae = np.mean(np.abs(pred - true))

    pred_rain = pred >= rain_thr
    true_rain = true >= rain_thr

    tp = np.logical_and(pred_rain, true_rain).sum()
    fp = np.logical_and(pred_rain, ~true_rain).sum()
    fn = np.logical_and(~pred_rain, true_rain).sum()
    csi = tp / (tp + fp + fn + 1e-8)

    return {
        "RMSE": float(rmse),
        "MAE": float(mae),
        "CSI": float(csi),
    }


# =========================================================
# 7. 训练与验证
# =========================================================
def train_one_model(train_loader, val_loader, in_dim, pred_len=1, epochs=3, lr=1e-3, device="cpu"):
    model = FeatureGRU(in_dim=in_dim, pred_len=pred_len).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_state = None
    best_val = 1e18

    for ep in range(epochs):
        model.train()
        train_loss = 0.0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x.size(0)

        train_loss /= max(len(train_loader.dataset), 1)

        model.eval()
        val_loss = 0.0
        preds_all, trues_all = [], []

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)

                pred = model(x)
                loss = criterion(pred, y)

                val_loss += loss.item() * x.size(0)
                preds_all.append(pred.cpu().numpy())
                trues_all.append(y.cpu().numpy())

        val_loss /= max(len(val_loader.dataset), 1)
        preds_all = np.concatenate(preds_all, axis=0)
        trues_all = np.concatenate(trues_all, axis=0)
        metrics = compute_metrics(preds_all, trues_all, rain_thr=RAIN_THR)

        print(
            f"[Epoch {ep + 1:02d}] "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"RMSE={metrics['RMSE']:.4f} "
            f"MAE={metrics['MAE']:.4f} "
            f"CSI={metrics['CSI']:.4f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model


def evaluate_model(model, loader, device="cpu"):
    model.eval()
    preds_all, trues_all = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            pred = model(x)

            preds_all.append(pred.cpu().numpy())
            trues_all.append(y.numpy())

    preds_all = np.concatenate(preds_all, axis=0)
    trues_all = np.concatenate(trues_all, axis=0)
    return compute_metrics(preds_all, trues_all, rain_thr=RAIN_THR)


# =========================================================
# 8. 内部相关性分析
# =========================================================
def build_internal_feature_matrix(samples, max_points=10000):
    rows = []
    for x, _, _ in samples:
        rows.append(x)  # [T, F]
    Xmat = np.concatenate(rows, axis=0)  # [num_points, F]

    if len(Xmat) > max_points:
        idx = np.random.choice(len(Xmat), size=max_points, replace=False)
        Xmat = Xmat[idx]

    return Xmat


def internal_correlation_analysis(samples, feature_names):
    Xmat = build_internal_feature_matrix(samples, max_points=MAX_POINTS_INTERNAL)
    results = {}

    print(f"[INFO] internal matrix shape = {Xmat.shape}")

    for i, feat in enumerate(feature_names):
        y = Xmat[:, i]
        X = np.delete(Xmat, i, axis=1)

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_SEED
        )

        rf = RandomForestRegressor(
            n_estimators=50,
            max_depth=10,
            random_state=RANDOM_SEED,
            n_jobs=1,
        )
        rf.fit(X_train, y_train)

        r2 = rf.score(X_val, y_val)

        perm = permutation_importance(
            rf,
            X_val,
            y_val,
            n_repeats=3,
            random_state=RANDOM_SEED,
            n_jobs=1,
        )

        other_feats = feature_names[:i] + feature_names[i + 1:]
        importance_map = {f: float(v) for f, v in zip(other_feats, perm.importances_mean)}

        results[feat] = {
            "r2_from_others": float(r2),
            "importance_from_others": importance_map,
        }

        print(f"[Internal] target={feat:10s}  R2={r2:.4f}")

    return results


# =========================================================
# 9. 预测相关性分析
# =========================================================
def predictive_correlation_analysis(samples, feature_names):
    train_samples, test_samples = train_test_split(
        samples, test_size=0.2, random_state=RANDOM_SEED
    )
    train_samples, val_samples = train_test_split(
        train_samples, test_size=0.2, random_state=RANDOM_SEED
    )

    full_indices = list(range(len(feature_names)))

    train_ds = SequenceDataset(train_samples, feature_indices=full_indices)
    val_ds = SequenceDataset(val_samples, feature_indices=full_indices)
    test_ds = SequenceDataset(test_samples, feature_indices=full_indices)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    print("\n[Baseline] train with all features")
    baseline_model = train_one_model(
        train_loader, val_loader,
        in_dim=len(full_indices),
        pred_len=PRED_LEN,
        epochs=EPOCHS,
        lr=LR,
        device=DEVICE,
    )

    baseline_metrics = evaluate_model(baseline_model, test_loader, device=DEVICE)
    print("[Baseline metrics]", baseline_metrics)

    rmse0 = baseline_metrics["RMSE"]
    mae0 = baseline_metrics["MAE"]
    csi0 = baseline_metrics["CSI"]

    results = {}

    for i, feat in enumerate(feature_names):
        keep_indices = [j for j in range(len(feature_names)) if j != i]

        train_ds = SequenceDataset(train_samples, feature_indices=keep_indices)
        val_ds = SequenceDataset(val_samples, feature_indices=keep_indices)
        test_ds = SequenceDataset(test_samples, feature_indices=keep_indices)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

        print(f"\n[LOFO] remove feature: {feat}")
        model = train_one_model(
            train_loader, val_loader,
            in_dim=len(keep_indices),
            pred_len=PRED_LEN,
            epochs=EPOCHS,
            lr=LR,
            device=DEVICE,
        )

        metrics = evaluate_model(model, test_loader, device=DEVICE)

        d_rmse = metrics["RMSE"] - rmse0
        d_mae = metrics["MAE"] - mae0
        d_csi = csi0 - metrics["CSI"]
        pc_score = d_rmse + d_mae + d_csi

        results[feat] = {
            "metrics_without_feature": metrics,
            "delta_RMSE": float(d_rmse),
            "delta_MAE": float(d_mae),
            "delta_CSI_drop": float(d_csi),
            "PC_score": float(pc_score),
        }

        print(
            f"[Result] {feat:10s} "
            f"dRMSE={d_rmse:.4f} "
            f"dMAE={d_mae:.4f} "
            f"dCSI_drop={d_csi:.4f} "
            f"PC={pc_score:.4f}"
        )

    return baseline_metrics, results


# =========================================================
# 10. 输出与保存
# =========================================================
def print_internal_summary(results: Dict):
    print("\n================ Internal Correlation Summary ================")
    sorted_items = sorted(results.items(), key=lambda x: x[1]["r2_from_others"], reverse=True)
    for i, (feat, info) in enumerate(sorted_items, 1):
        print(f"{i:02d}. {feat:10s} | R2_from_others={info['r2_from_others']:.6f}")


def print_predictive_ranking(results: Dict):
    print("\n================ Predictive Correlation Ranking ================")
    sorted_items = sorted(results.items(), key=lambda x: x[1]["PC_score"], reverse=True)
    for i, (feat, info) in enumerate(sorted_items, 1):
        print(
            f"{i:02d}. {feat:10s} | "
            f"PC={info['PC_score']:.6f} | "
            f"dRMSE={info['delta_RMSE']:.6f} | "
            f"dMAE={info['delta_MAE']:.6f} | "
            f"dCSI_drop={info['delta_CSI_drop']:.6f}"
        )


def save_internal_results(results: Dict, out_path: str):
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("feature,r2_from_others\n")
        sorted_items = sorted(results.items(), key=lambda x: x[1]["r2_from_others"], reverse=True)
        for feat, info in sorted_items:
            f.write(f"{feat},{info['r2_from_others']:.8f}\n")
    print(f"[INFO] internal results saved to: {out_path}")


def save_predictive_results(results: Dict, out_path: str):
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("feature,PC_score,delta_RMSE,delta_MAE,delta_CSI_drop\n")
        sorted_items = sorted(results.items(), key=lambda x: x[1]["PC_score"], reverse=True)
        for feat, info in sorted_items:
            f.write(
                f"{feat},"
                f"{info['PC_score']:.8f},"
                f"{info['delta_RMSE']:.8f},"
                f"{info['delta_MAE']:.8f},"
                f"{info['delta_CSI_drop']:.8f}\n"
            )
    print(f"[INFO] predictive results saved to: {out_path}")


# =========================================================
# 11. 主函数
# =========================================================
def main():
    print("[INFO] loading npz files ...")
    all_data, feature_names = load_all_npz(DATA_DIR)

    print(f"[INFO] file num = {len(all_data)}")
    print(f"[INFO] feature num = {len(feature_names)}")
    print(f"[INFO] features = {feature_names}")

    print("[INFO] building sequence samples ...")
    samples = build_sequence_samples(all_data, seq_len=SEQ_LEN, pred_len=PRED_LEN)
    if len(samples) == 0:
        raise RuntimeError("没有构造出样本，请检查 npz 数据。")

    x0, y0, cp0 = samples[0]
    print(f"[INFO] total samples = {len(samples)}")
    print(f"[INFO] sample from cp = {cp0}")
    print(f"[INFO] sample X shape = {x0.shape}, Y shape = {y0.shape}")

    internal_results = internal_correlation_analysis(samples, feature_names)
    print_internal_summary(internal_results)
    save_internal_results(internal_results, os.path.join(DATA_DIR, "internal_correlation_results.csv"))

    baseline_metrics, predictive_results = predictive_correlation_analysis(samples, feature_names)
    print("\nBaseline metrics:", baseline_metrics)
    print_predictive_ranking(predictive_results)
    save_predictive_results(predictive_results, os.path.join(DATA_DIR, "predictive_correlation_results.csv"))


if __name__ == "__main__":
    main()
