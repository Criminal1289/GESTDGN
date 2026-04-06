import os
import glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# =========================
# 1. 路径配置
# =========================
DATA_DIR = r"C:/Users/17878/Desktop/gestdgn/data/cnn_spatial_align_slide"
SAVE_DIR = r"./corr_figs"
os.makedirs(SAVE_DIR, exist_ok=True)

# =========================
# 2. 特征名
# =========================
NWP_FEATURES = [
    "CAPE", "DVG200", "DVG850", "DVG925", "HT0", "HT10", "HTw0", "KI",
    "LCL", "LFC", "LI300", "LI500", "muLCL", "PE", "PWAT", "Q1000", "Q700",
    "Q850", "RH1000", "RH500", "RH700", "TdSfc850", "TTdMax74", "TTdMean74",
    "WS500", "WS700", "WS925"
]

RADAR_FEATURES = [
    "CR", "HBR", "ET", "VIL",
    "CAP20", "CAP30", "CAP40", "CAP50", "CAP60", "CAP70"
]

ALL_FEATURES = NWP_FEATURES + RADAR_FEATURES
SKIP_FEATURES = {
    "HT0", "HTw0",
    "CAP20", "CAP30", "CAP50", "CAP70",
    "PE", "Q1000", "Q700"
}

# =========================
# 3. 读取 npz 并构造特征表
# =========================
def list_npz_files(data_dir):
    files = sorted(glob.glob(os.path.join(data_dir, "sample_*.npz")))
    if len(files) == 0:
        raise RuntimeError(f"没有找到 sample_*.npz: {data_dir}")
    return files


def load_feature_rows_from_npz(npz_path):
    d = np.load(npz_path, allow_pickle=True)

    nwp = d["nwp"].astype(np.float32)       # [T, Nw, Cn, H, W]
    radar = d["radar"].astype(np.float32)   # [T, Cr, H, W]

    # NWP -> [T, 27]
    nwp_feat = nwp.mean(axis=(1, 3, 4))     # [T, 27]

    # Radar -> [T, 10]
    radar_feat = radar.mean(axis=(2, 3))    # [T, 10]

    # 拼接 -> [T, 37]
    feat = np.concatenate([nwp_feat, radar_feat], axis=1)

    if feat.shape[1] != len(ALL_FEATURES):
        raise ValueError(f"{os.path.basename(npz_path)} 特征维度不对: {feat.shape}")

    # 按 SKIP_FEATURES 删除列
    keep_indices = [i for i, f in enumerate(ALL_FEATURES) if f not in SKIP_FEATURES]
    feat = feat[:, keep_indices]

    return feat


def build_feature_dataframe(data_dir):
    all_rows = []

    for f in list_npz_files(data_dir):
        feat = load_feature_rows_from_npz(f)
        all_rows.append(feat)

    X = np.concatenate(all_rows, axis=0)

    kept_features = [f for f in ALL_FEATURES if f not in SKIP_FEATURES]
    df = pd.DataFrame(X, columns=kept_features)
    return df

# =========================
# 4. 计算 Pearson 相关矩阵
# =========================
df = build_feature_dataframe(DATA_DIR)
corr_matrix = df.corr(method="pearson")

# 保存数值
csv_path = os.path.join(SAVE_DIR, "pearson_corr_matrix.csv")
corr_matrix.to_csv(csv_path, encoding="utf-8-sig")
print(f"[INFO] Pearson相关矩阵已保存: {csv_path}")

# =========================
# 5. 绘制热力图
# =========================
plt.figure(figsize=(18, 15))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # 只显示下三角

sns.heatmap(
    corr_matrix,
    mask=mask,
    cmap="RdBu_r",
    vmin=-1,
    vmax=1,
    center=0,
    annot=True,
    fmt=".2f",
    square=True,
    linewidths=0.5,
    cbar_kws={"shrink": 0.8}
)

plt.xticks(rotation=45, ha="right", fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.tight_layout()

fig_path = os.path.join(SAVE_DIR, "pearson_corr_heatmap.png")
plt.savefig(fig_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"[INFO] 热力图已保存: {fig_path}")