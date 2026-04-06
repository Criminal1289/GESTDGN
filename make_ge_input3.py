import os
import glob
import re
import numpy as np
from datetime import datetime, timedelta

# =========================================================
# radar / label 固定主窗口
# nwp 主窗口 + 滑动辅助窗口
# 不再取32x32，直接保留二维网格给CNN
# =========================================================

cp_root = r"../gestdgn/data/BJ"

nwp_vars = [
    "CAPE", "DVG200", "DVG850", "DVG925", "HT0", "HT10", "HTw0", "KI",
    "LCL", "LFC", "LI300", "LI500", "muLCL", "PE", "PWAT", "Q1000", "Q700",
    "Q850", "RH1000", "RH500", "RH700", "TdSfc850", "TTdMax74", "TTdMean74",
    "WS500", "WS700", "WS925"
]

radar_vars = [
    "CR", "HBR", "ET", "VIL",
    "CAP20", "CAP30", "CAP40", "CAP50", "CAP60", "CAP70"
]

label_var = "RA"
radar_root_name = "RADAR"

# =========================================================
# 原始尺寸
# =========================================================
NWP_SRC_HW = 167
RL_SRC_HW = 301
RL_CROP_HW = 128

# =========================================================
# 经纬度范围
# =========================================================
# NWP
NWP_START_LON = 113.97
NWP_START_LAT = 42.30
NWP_END_LON = 118.98
NWP_END_LAT = 37.29

# RADAR / LABEL
RL_START_LON = 114.9719
RL_START_LAT = 41.3089
RL_END_LON = 117.9719
RL_END_LAT = 38.3089

# =========================================================
# 时间配置
# =========================================================
RADAR_INTERVAL = 6
WINDOW_BEFORE = 2
WINDOW_AFTER = 0
RA_HOUR = 60

# =========================================================
# NWP滑动窗口配置
# 中心主窗口 + 周围8个辅助窗口
# 单位：NWP网格偏移量
# =========================================================
SLIDE_OFFSETS = [
    (0, 0),
    (-2, 0), (2, 0), (0, -2), (0, 2),
    (-2, -2), (-2, 2), (2, -2), (2, 2),
]
N_NWP_WINDOWS = len(SLIDE_OFFSETS)

# =========================================================
# 输出目录
# =========================================================
OUT_DIR = "../gestdgn/data/cnn_spatial_align_slide"
os.makedirs(OUT_DIR, exist_ok=True)


# =========================================================
# 时间工具
# =========================================================
def time_key(fname):
    b = os.path.basename(fname)
    m = re.search(r"_(\d{4}-\d{4})", b)
    return m.group(1) if m else None


def parse_datetime_from_key(tk):
    mmdd, hhmm = tk.split("-")
    month = int(mmdd[:2])
    day = int(mmdd[2:])
    hour = int(hhmm[:2])
    minute = int(hhmm[2:])
    return datetime(2000, month, day, hour, minute)


def hour_key(fname):
    tk = time_key(fname)
    if not tk:
        return None
    mmdd, hhmm = tk.split("-")
    hh = hhmm[:2]
    return f"{mmdd}-{hh}00"


# =========================================================
# 基础工具
# =========================================================
def center_crop_info(h, w, out_h, out_w):
    sy = (h - out_h) // 2
    sx = (w - out_w) // 2
    ey = sy + out_h
    ex = sx + out_w
    return sy, ey, sx, ex


def center_crop_grid(src2d, out_h, out_w):
    sy, ey, sx, ex = center_crop_info(src2d.shape[0], src2d.shape[1], out_h, out_w)
    return src2d[sy:ey, sx:ex].astype(np.float32)


def pixel_size(start_lon, start_lat, end_lon, end_lat, w, h):
    dx = (end_lon - start_lon) / (w - 1)
    dy = (start_lat - end_lat) / (h - 1)
    return dx, dy


def crop_geo_range_from_center_crop(start_lon, start_lat, end_lon, end_lat, h, w, sy, ey, sx, ex):
    dx, dy = pixel_size(start_lon, start_lat, end_lon, end_lat, w, h)

    crop_start_lon = start_lon + sx * dx
    crop_end_lon = start_lon + (ex - 1) * dx
    crop_start_lat = start_lat - sy * dy
    crop_end_lat = start_lat - (ey - 1) * dy

    return crop_start_lon, crop_start_lat, crop_end_lon, crop_end_lat


def geo_to_index_range(start_lon, start_lat, end_lon, end_lat, h, w,
                       target_start_lon, target_start_lat, target_end_lon, target_end_lat):
    dx, dy = pixel_size(start_lon, start_lat, end_lon, end_lat, w, h)

    sx = int(round((target_start_lon - start_lon) / dx))
    ex = int(round((target_end_lon - start_lon) / dx))
    sy = int(round((start_lat - target_start_lat) / dy))
    ey = int(round((start_lat - target_end_lat) / dy))

    sx = max(0, min(w - 1, sx))
    ex = max(0, min(w - 1, ex))
    sy = max(0, min(h - 1, sy))
    ey = max(0, min(h - 1, ey))

    if sx > ex:
        sx, ex = ex, sx
    if sy > ey:
        sy, ey = ey, sy

    return sy, ey + 1, sx, ex + 1


def shift_window(sy, ey, sx, ex, dy, dx, H, W):
    h = ey - sy
    w = ex - sx

    new_sy = sy + dy
    new_ey = ey + dy
    new_sx = sx + dx
    new_ex = ex + dx

    if new_sy < 0:
        new_ey += -new_sy
        new_sy = 0
    if new_ey > H:
        new_sy -= (new_ey - H)
        new_ey = H
    if new_sx < 0:
        new_ex += -new_sx
        new_sx = 0
    if new_ex > W:
        new_sx -= (new_ex - W)
        new_ex = W

    new_sy = max(0, new_sy)
    new_sx = max(0, new_sx)
    new_ey = min(H, new_ey)
    new_ex = min(W, new_ex)

    if (new_ey - new_sy) != h or (new_ex - new_sx) != w:
        raise ValueError("Sliding window size changed unexpectedly.")

    return new_sy, new_ey, new_sx, new_ex


def preprocess_ra_grid(g):
    g = g.astype(np.float32)
    g[g < 0] = 0
    g[np.isnan(g)] = 0
    return g


def preprocess_radar_grid(g, clip_min=None, clip_max=None):
    g = g.astype(np.float32)
    g[np.isnan(g)] = 0
    if clip_min is not None:
        g[g < clip_min] = clip_min
    if clip_max is not None:
        g[g > clip_max] = clip_max
    return g


# =========================================================
# 由 radar/label 中心128区域定义主对齐地理范围
# =========================================================
RL_CROP_SY, RL_CROP_EY, RL_CROP_SX, RL_CROP_EX = center_crop_info(
    RL_SRC_HW, RL_SRC_HW, RL_CROP_HW, RL_CROP_HW
)

TARGET_START_LON, TARGET_START_LAT, TARGET_END_LON, TARGET_END_LAT = crop_geo_range_from_center_crop(
    RL_START_LON, RL_START_LAT, RL_END_LON, RL_END_LAT,
    RL_SRC_HW, RL_SRC_HW,
    RL_CROP_SY, RL_CROP_EY, RL_CROP_SX, RL_CROP_EX
)

# NWP主对齐窗口
NWP_MAIN_SY, NWP_MAIN_EY, NWP_MAIN_SX, NWP_MAIN_EX = geo_to_index_range(
    NWP_START_LON, NWP_START_LAT, NWP_END_LON, NWP_END_LAT,
    NWP_SRC_HW, NWP_SRC_HW,
    TARGET_START_LON, TARGET_START_LAT, TARGET_END_LON, TARGET_END_LAT
)

NWP_MAIN_H = NWP_MAIN_EY - NWP_MAIN_SY
NWP_MAIN_W = NWP_MAIN_EX - NWP_MAIN_SX

print("===== CNN preprocessing with sliding windows =====")
print("Radar/Label crop:", (RL_CROP_HW, RL_CROP_HW))
print("NWP main crop index:", NWP_MAIN_SY, NWP_MAIN_EY, NWP_MAIN_SX, NWP_MAIN_EX)
print("NWP main crop shape:", (NWP_MAIN_H, NWP_MAIN_W))
print("NWP windows:", N_NWP_WINDOWS)

# 预先算好所有滑动窗口
NWP_WINDOWS = []
for off_y, off_x in SLIDE_OFFSETS:
    sy, ey, sx, ex = shift_window(
        NWP_MAIN_SY, NWP_MAIN_EY, NWP_MAIN_SX, NWP_MAIN_EX,
        off_y, off_x,
        NWP_SRC_HW, NWP_SRC_HW
    )
    NWP_WINDOWS.append((sy, ey, sx, ex))

print("Sliding windows:")
for i, (sy, ey, sx, ex) in enumerate(NWP_WINDOWS):
    print(i, (sy, ey, sx, ex), "shape=", (ey - sy, ex - sx))


# =========================================================
# 加载函数
# =========================================================
def load_rl_grid(path):
    g = np.load(path).astype(np.float32)

    if g.shape == (RL_SRC_HW, RL_SRC_HW):
        return center_crop_grid(g, RL_CROP_HW, RL_CROP_HW)

    if g.shape == (RL_CROP_HW, RL_CROP_HW):
        return g.astype(np.float32)

    raise ValueError(
        f"Unexpected radar/label shape {g.shape} in file: {path}. "
        f"Expected {(RL_SRC_HW, RL_SRC_HW)} or {(RL_CROP_HW, RL_CROP_HW)}."
    )


def load_nwp_all_windows(path):
    g = np.load(path).astype(np.float32)

    if g.shape != (NWP_SRC_HW, NWP_SRC_HW):
        raise ValueError(
            f"Unexpected NWP shape {g.shape} in file: {path}. "
            f"Expected {(NWP_SRC_HW, NWP_SRC_HW)}."
        )

    windows = []
    for sy, ey, sx, ex in NWP_WINDOWS:
        windows.append(g[sy:ey, sx:ex].astype(np.float32))

    # [N_windows, H, W]
    return np.stack(windows, axis=0)


# =========================================================
# Radar窗口
# =========================================================
def select_radar_files(files, target_hour_key):
    target_dt = parse_datetime_from_key(target_hour_key)

    start_dt = target_dt - timedelta(minutes=WINDOW_BEFORE * RADAR_INTERVAL)
    end_dt = target_dt + timedelta(minutes=WINDOW_AFTER * RADAR_INTERVAL)

    selected = []
    for f in files:
        tk = time_key(f)
        if not tk:
            continue
        dt = parse_datetime_from_key(tk)
        if start_dt <= dt <= end_dt:
            selected.append(f)

    return sorted(selected)


def aggregate_radar_grids(files, how="max"):
    if len(files) == 0:
        return None

    grids = [load_rl_grid(f) for f in files]
    stack = np.stack(grids, axis=0)  # [T_radar, 128, 128]

    if how == "mean":
        return stack.mean(axis=0).astype(np.float32)
    return stack.max(axis=0).astype(np.float32)


# =========================================================
# RA未来1小时累计
# =========================================================
def select_ra_future(files, target_hour_key):
    target_dt = parse_datetime_from_key(target_hour_key)
    end_dt = target_dt + timedelta(minutes=RA_HOUR)

    selected = []
    for f in files:
        tk = time_key(f)
        if not tk:
            continue
        dt = parse_datetime_from_key(tk)
        if target_dt < dt <= end_dt:
            selected.append(f)

    return sorted(selected)


def aggregate_ra_sum(files):
    if len(files) == 0:
        return None

    grids = []
    for f in files:
        g = preprocess_ra_grid(load_rl_grid(f))
        grids.append(g)

    stack = np.stack(grids, axis=0)
    return stack.sum(axis=0).astype(np.float32)


# =========================================================
# 主流程：生成给CNN的时序样本
# =========================================================
cp_folders = sorted([p for p in glob.glob(os.path.join(cp_root, "*")) if os.path.isdir(p)])

for cp_idx, cp in enumerate(cp_folders):
    # ---------------------
    # NWP
    # ---------------------
    nwp_map = {}
    for v in nwp_vars:
        files = glob.glob(os.path.join(cp, "NWP", v, "*.npy"))
        d = {}
        for f in files:
            hk = hour_key(f)
            if hk and hk not in d:
                d[hk] = f
        nwp_map[v] = d

    # ---------------------
    # Radar
    # ---------------------
    radar_root = os.path.join(cp, radar_root_name)
    radar_files = {}
    for rv in radar_vars:
        radar_files[rv] = glob.glob(os.path.join(radar_root, rv, "*.npy"))

    # ---------------------
    # Label
    # ---------------------
    ra_files = glob.glob(os.path.join(cp, "LABEL", label_var, "*.npy"))

    # ---------------------
    # 时间对齐：以 NWP 公共时次为主
    # ---------------------
    common = None
    for v in nwp_vars:
        s = set(nwp_map[v].keys())
        common = s if common is None else (common & s)

    common = sorted(common)

    if not common:
        print("[SKIP] no common hourly keys from NWP in", cp)
        continue

    T = len(common)

    # nwp: [T, N_windows, C_nwp, H, W]
    nwp_tensor = np.zeros(
        (T, N_NWP_WINDOWS, len(nwp_vars), NWP_MAIN_H, NWP_MAIN_W),
        dtype=np.float32
    )

    # radar: [T, C_radar, 128, 128]
    radar_tensor = np.zeros(
        (T, len(radar_vars), RL_CROP_HW, RL_CROP_HW),
        dtype=np.float32
    )

    # label: [T, 1, 128, 128]
    label_tensor = np.zeros(
        (T, 1, RL_CROP_HW, RL_CROP_HW),
        dtype=np.float32
    )

    # ---------------------
    # 构造张量
    # ---------------------
    for ti, hk in enumerate(common):
        # NWP
        for fi, v in enumerate(nwp_vars):
            if hk in nwp_map[v]:
                # [N_windows, H, W]
                g_multi = load_nwp_all_windows(nwp_map[v][hk])
                nwp_tensor[ti, :, fi, :, :] = g_multi

        # Radar
        for rfi, rv in enumerate(radar_vars):
            files = select_radar_files(radar_files[rv], hk)
            g = aggregate_radar_grids(files)

            if g is None:
                continue

            if rv in ["CR", "HBR", "CAP20", "CAP30", "CAP40", "CAP50", "CAP60", "CAP70"]:
                g = preprocess_radar_grid(g, 0, 600) / 600.0
            elif rv == "ET":
                g = preprocess_radar_grid(g, 0, 300) / 300.0
            elif rv == "VIL":
                g = preprocess_radar_grid(g, 0, 1000) / 1000.0
            else:
                g = preprocess_radar_grid(g)

            radar_tensor[ti, rfi] = g

        # Label
        future_files = select_ra_future(ra_files, hk)
        g = aggregate_ra_sum(future_files)
        if g is not None:
            label_tensor[ti, 0] = g

    # ---------------------
    # 保存
    # ---------------------
    out_path = os.path.join(OUT_DIR, f"sample_{cp_idx:03d}.npz")
    np.savez_compressed(
        out_path,
        nwp=nwp_tensor,
        radar=radar_tensor,
        label=label_tensor,
        time_keys=np.array(common),
        nwp_windows=np.array(NWP_WINDOWS, dtype=np.int32),
        slide_offsets=np.array(SLIDE_OFFSETS, dtype=np.int32),
        nwp_vars=np.array(nwp_vars),
        radar_vars=np.array(radar_vars),
        cp_name=os.path.basename(cp),
    )

    print("[OK]", os.path.basename(cp))

print("Done")
