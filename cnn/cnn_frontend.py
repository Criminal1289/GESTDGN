import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from make_knn_adj import build_knn_adj

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
cnn_path = os.path.join(PROJECT_DIR, "data", "cnn_spatial_align_slide")
coords_path = os.path.join(PROJECT_DIR, "data", "cnn_spatial_align_slide", "coords.npy")


class ConvBlock2D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, dropout=0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),

            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class NWPEncoder(nn.Module):
    """
    输入:
        [B*T*Nw, Cn, Hn, Wn]
    输出:
        [B*T*Nw, out_dim, 32, 32]
    """

    def __init__(self, in_ch, out_dim=32, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock2D(in_ch, 32, stride=1, dropout=dropout),
            ConvBlock2D(32, 64, stride=1, dropout=dropout),
            ConvBlock2D(64, out_dim, stride=1, dropout=dropout),
        )
        self.pool = nn.AdaptiveAvgPool2d((32, 32))

    def forward(self, x):
        x = self.net(x)
        x = self.pool(x)
        return x


class RadarEncoder(nn.Module):
    """
    输入:
        [B*T, Cr, 128, 128]
    输出:
        [B*T, out_dim, 32, 32]
    """

    def __init__(self, in_ch, out_dim=32, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock2D(in_ch, 32, stride=2, dropout=dropout),  # 128 -> 64
            ConvBlock2D(32, 64, stride=2, dropout=dropout),  # 64 -> 32
            ConvBlock2D(64, out_dim, stride=1, dropout=dropout),
        )
        self.pool = nn.AdaptiveAvgPool2d((32, 32))

    def forward(self, x):
        x = self.net(x)
        x = self.pool(x)
        return x


class WindowFusion(nn.Module):
    """
    NWP多窗口融合

    输入:
        [B, T, Nw, D, 32, 32]
    输出:
        [B, T, D, 32, 32]
    """

    def __init__(self, method="mean"):
        super().__init__()
        self.method = method

    def forward(self, x):
        if self.method == "mean":
            return x.mean(dim=2)
        else:
            raise ValueError(f"Unsupported fusion method: {self.method}")


class CNNFrontEnd(nn.Module):
    def __init__(self,
                 nwp_dim=32,
                 radar_dim=32,
                 fusion="mean",
                 dropout=0.1):
        super().__init__()

        nwp_in_ch = 27
        radar_in_ch = 10

        self.nwp_encoder = NWPEncoder(nwp_in_ch, out_dim=nwp_dim, dropout=dropout)
        self.radar_encoder = RadarEncoder(radar_in_ch, out_dim=radar_dim, dropout=dropout)
        self.fusion = WindowFusion(method=fusion)

        self.out_dim = nwp_dim + radar_dim
        self.has_saved_patch = False

    def forward(self, nwp, radar):
        """
        nwp   : [B, T, Nw, Cn, Hn, Wn]
        radar : [B, T, Cr, Hr, Wr]
        label : [B, T, 301, 301] or [B, T, 1, 301, 301]

        return:
            node_x : [B, 1024, T, D_total(+1)]
        """
        B, T, Nw, Cn, Hn, Wn = nwp.shape
        _, _, Cr, Hr, Wr = radar.shape

        # NWP
        nwp = nwp.reshape(B * T * Nw, Cn, Hn, Wn)
        nwp_feat = self.nwp_encoder(nwp)  # [B*T*Nw, Dn, 32, 32]
        Dn = nwp_feat.shape[1]
        nwp_feat = nwp_feat.reshape(B, T, Nw, Dn, 32, 32)  # [B, T, Nw, Dn, 32, 32]
        nwp_feat = self.fusion(nwp_feat)  # [B, T, Dn, 32, 32]

        # Radar
        radar = radar.reshape(B * T, Cr, Hr, Wr)
        radar_feat = self.radar_encoder(radar)  # [B*T, Dr, 32, 32]
        Dr = radar_feat.shape[1]
        radar_feat = radar_feat.reshape(B, T, Dr, 32, 32)  # [B, T, Dr, 32, 32]

        # 先拼 nwp/radar
        feat = torch.cat([nwp_feat, radar_feat], dim=2)  # [B, T, D, 32, 32]

        B, T, D, H32, W32 = feat.shape

        # 生成coords文件
        if self.has_saved_patch == False:
            ensure_cnn_coords(coords_path=coords_path, h=32, w=32)
            self.has_saved_patch = True
        # [B, T, D, 32, 32] -> [B, 1024, T, D]
        feat = feat.permute(0, 1, 3, 4, 2).reshape(B, T, H32 * W32, D)
        node_x = feat.permute(0, 2, 1, 3).contiguous()

        return node_x


class CNNRegressor(nn.Module):
    """
    一个最简单的CNN后接MLP回归头示例
    不依赖 STDGN，只是为了给 sample 方案一个最小可跑 baseline

    输出:
        y_hat: [B, pre_len, 1024]
    """

    def __init__(self,
                 nwp_in_ch,
                 radar_in_ch,
                 seq_len,
                 pre_len,
                 nwp_dim=32,
                 radar_dim=32,
                 fusion="mean",
                 dropout=0.1):
        super().__init__()

        self.frontend = CNNFrontEnd(
            nwp_in_ch=nwp_in_ch,
            radar_in_ch=radar_in_ch,
            nwp_dim=nwp_dim,
            radar_dim=radar_dim,
            fusion=fusion,
            dropout=dropout
        )

        in_dim = (nwp_dim + radar_dim) * seq_len

        self.head = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, pre_len)
        )

    def forward(self, nwp, radar):
        """
        nwp   : [B, T, Nw, Cn, Hn, Wn]
        radar : [B, T, Cr, 128, 128]

        返回:
            y_hat: [B, pre_len, 1024]
        """
        node_x = self.frontend(nwp, radar)  # [B, 1024, T, D]
        B, N, T, D = node_x.shape

        x = node_x.reshape(B, N, T * D)  # [B, 1024, T*D]
        y = self.head(x)  # [B, 1024, pre_len]
        y = y.permute(0, 2, 1).contiguous()  # [B, pre_len, 1024]
        return y


def ensure_cnn_coords(coords_path, h=32, w=32):
    os.makedirs(os.path.dirname(coords_path), exist_ok=True)

    if os.path.exists(coords_path):
        coords = np.load(coords_path)
        if coords.shape == (h * w, 2):
            print(f"[SKIP] coords already exists: {coords_path}, shape={coords.shape}")
            return
        else:
            print(f"[REBUILD] coords shape mismatch: {coords.shape}, rebuild -> {(h * w, 2)}")

    coords = []
    for i in range(h):
        for j in range(w):
            coords.append([i, j])

    coords = np.array(coords, dtype=np.float32)
    np.save(coords_path, coords)
    print(f"[OK] save cnn coords: {coords_path}, shape={coords.shape}")
    coords_file = np.load(os.path.join(cnn_path, "coords.npy"))
    A = build_knn_adj(coords_file, k=8, mode="binary")
    np.save(os.path.join(cnn_path, "adj_knn.npy"), A)
