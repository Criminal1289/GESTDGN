import os
import math
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.patches import Rectangle
from PIL import Image
import requests


# =====================================================
# 1. 路径设置：只需要改这里
# =====================================================
root_dir = r"../../data/BJ"  # 你的 BJ 文件夹路径，里面放很多 CP_01_xxx 文件夹

label_format_path = r"../../data/BJ/Label_Format_00001.txt"
radar_format_path = r"../../data/BJ/RADA_Format_00001.txt"
nwp_format_path = r"../../data/BJ/NWP_Format_00001.txt"

output_png = r"../../example/extreme_events/matched_extreme_events_map.png"
output_csv = r"../../example/extreme_events/matched_extreme_events_summary.csv"


# =====================================================
# 1.1 卫星底图设置
# =====================================================
# 这个版本不再使用 contextily / rasterio / pyproj，避免 Windows 下 GDAL 安装问题。
# 只依赖：requests、Pillow、numpy、matplotlib、pandas。
USE_SATELLITE_BACKGROUND = True

# Esri World Imagery 在线卫星瓦片
SATELLITE_TILE_URL = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
)

# "auto"：自动选择瓦片数量不太多的 zoom；也可以手动设置成 8、9、10、11。
SATELLITE_ZOOM = "auto"

# 自动 zoom 时最多下载多少张瓦片。越大越清晰，但越慢。
MAX_TILE_COUNT = 80

# 瓦片缓存目录：第一次会下载，之后复用本地缓存。
SATELLITE_CACHE_DIR = Path(output_png).parent / "satellite_tile_cache"

# 是否显示经纬度刻度
SHOW_LONLAT_TICKS = True


# =====================================================
# 2. 读取 txt 格式说明文件
# =====================================================
def read_format_txt(txt_path):
    info = {}

    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("["):
                continue

            if "=" in line:
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()

                try:
                    info[key] = float(value)
                except ValueError:
                    info[key] = value

    return info


label_info = read_format_txt(label_format_path)
radar_info = read_format_txt(radar_format_path)
nwp_info = read_format_txt(nwp_format_path)


# =====================================================
# 3. 构建 Label / Radar 经纬度网格
# =====================================================
def build_lon_lat_grid(info):
    nx = int(info["nx"])
    ny = int(info["ny"])

    lon = np.linspace(info["StartLon"], info["EndLon"], nx)
    lat = np.linspace(info["StartLat"], info["EndLat"], ny)

    lon2d, lat2d = np.meshgrid(lon, lat)

    return lon, lat, lon2d, lat2d


label_lon, label_lat, label_lon2d, label_lat2d = build_lon_lat_grid(label_info)


# =====================================================
# 4. 找到所有 CP 事件文件夹
# =====================================================
def find_cp_event_folders(root_dir):
    root_dir = Path(root_dir)

    event_folders = []

    for item in root_dir.iterdir():
        if item.is_dir() and item.name.startswith("CP_"):
            event_folders.append(item)

    event_folders = sorted(event_folders, key=lambda x: x.name)

    return event_folders


# =====================================================
# 5. 找到单个事件里的 LABEL/RA/*.npy 文件
# =====================================================
def find_ra_files(event_folder):
    event_folder = Path(event_folder)

    ra_dir = event_folder / "LABEL" / "RA"

    if not ra_dir.exists():
        return []

    ra_files = sorted(list(ra_dir.glob("*.npy")))

    return ra_files


# =====================================================
# 6. 处理单个 CP 事件，提取最大降水位置
# =====================================================
def process_one_event(event_folder, label_info, lon2d, lat2d):
    event_folder = Path(event_folder)

    ra_files = find_ra_files(event_folder)

    if len(ra_files) == 0:
        print(f"[跳过] {event_folder.name} 中没有找到 LABEL/RA/*.npy")
        return None

    fill_value = label_info["FillValue"]

    # RA 的单位是 0.1 mm，所以需要乘 0.1 转成 mm
    unit_scale = 0.1

    max_rain = -np.inf
    max_row = None
    max_col = None
    max_file = None

    for ra_file in ra_files:
        try:
            ra = np.load(ra_file).astype(float)
        except Exception as e:
            print(f"[跳过文件] {ra_file} 读取失败: {e}")
            continue

        # 缺测值处理
        ra[ra == fill_value] = np.nan

        # 单位转换：0.1 mm -> mm
        ra_mm = ra * unit_scale

        if np.all(np.isnan(ra_mm)):
            continue

        current_max = np.nanmax(ra_mm)

        if current_max > max_rain:
            max_rain = current_max
            max_row, max_col = np.unravel_index(np.nanargmax(ra_mm), ra_mm.shape)
            max_file = ra_file.name

    if max_file is None:
        print(f"[跳过] {event_folder.name} 中没有有效降水数据")
        return None

    max_lon = lon2d[max_row, max_col]
    max_lat = lat2d[max_row, max_col]

    return {
        "event_name": event_folder.name,
        "max_rainfall_mm": float(max_rain),
        "max_lon": float(max_lon),
        "max_lat": float(max_lat),
        "max_file": max_file,
        "ra_file_count": len(ra_files)
    }


# =====================================================
# 7. 卫星底图辅助函数：不依赖 contextily / pyproj
# =====================================================
def get_map_extent(radar_info, nwp_info, margin=0.1):
    """返回绘图经纬度范围：xmin, xmax, ymin, ymax。"""
    xmin = min(radar_info["StartLon"], nwp_info["StartLon"]) - margin
    xmax = max(radar_info["EndLon"], nwp_info["EndLon"]) + margin
    ymin = min(radar_info["EndLat"], nwp_info["EndLat"]) - margin
    ymax = max(radar_info["StartLat"], nwp_info["StartLat"]) + margin
    return xmin, xmax, ymin, ymax


def lonlat_to_world_pixel(lon, lat, zoom):
    """
    经纬度 -> Web Mercator 世界像素坐标。

    注意：这里的 y 是瓦片坐标方向，向下增大。
    绘图时会使用 -y，让地图北方向朝上。
    """
    lat = np.clip(lat, -85.05112878, 85.05112878)

    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)

    n = 2 ** zoom
    x = (lon + 180.0) / 360.0 * n * 256.0

    lat_rad = np.deg2rad(lat)
    y = (
        1.0
        - np.log(np.tan(lat_rad) + 1.0 / np.cos(lat_rad)) / np.pi
    ) / 2.0 * n * 256.0

    return x, y


def lonlat_to_plot_xy(lon, lat, zoom):
    """经纬度 -> 绘图坐标。"""
    x, y = lonlat_to_world_pixel(lon, lat, zoom)
    return x, -y


def tile_range_for_extent(xmin, xmax, ymin, ymax, zoom):
    """计算某个经纬度范围在指定 zoom 下覆盖的瓦片范围。"""
    x_left, y_bottom = lonlat_to_world_pixel(xmin, ymin, zoom)
    x_right, y_top = lonlat_to_world_pixel(xmax, ymax, zoom)

    tile_x_min = int(math.floor(min(x_left, x_right) / 256.0))
    tile_x_max = int(math.floor(max(x_left, x_right) / 256.0))
    tile_y_min = int(math.floor(min(y_top, y_bottom) / 256.0))
    tile_y_max = int(math.floor(max(y_top, y_bottom) / 256.0))

    return tile_x_min, tile_x_max, tile_y_min, tile_y_max


def count_tiles_for_extent(xmin, xmax, ymin, ymax, zoom):
    tx0, tx1, ty0, ty1 = tile_range_for_extent(xmin, xmax, ymin, ymax, zoom)
    return (tx1 - tx0 + 1) * (ty1 - ty0 + 1)


def choose_auto_zoom(xmin, xmax, ymin, ymax, max_tile_count=80):
    """自动选择一个清晰但不会下载过多瓦片的 zoom。"""
    for zoom in range(13, 4, -1):
        tile_count = count_tiles_for_extent(xmin, xmax, ymin, ymax, zoom)
        if tile_count <= max_tile_count:
            return zoom
    return 5


def download_one_tile(z, x, y, cache_dir):
    """下载单张卫星瓦片，失败时返回灰色空瓦片。"""
    cache_dir = Path(cache_dir)
    tile_path = cache_dir / str(z) / str(x) / f"{y}.jpg"
    tile_path.parent.mkdir(parents=True, exist_ok=True)

    if tile_path.exists():
        try:
            return Image.open(tile_path).convert("RGB"), True
        except Exception:
            tile_path.unlink(missing_ok=True)

    url = SATELLITE_TILE_URL.format(z=z, x=x, y=y)

    try:
        response = requests.get(
            url,
            timeout=15,
            headers={"User-Agent": "Mozilla/5.0"}
        )
        response.raise_for_status()

        img = Image.open(io.BytesIO(response.content)).convert("RGB")
        img.save(tile_path)
        return img, True

    except Exception as e:
        print(f"[警告] 瓦片下载失败 z={z}, x={x}, y={y}: {e}")
        blank = Image.new("RGB", (256, 256), (235, 235, 235))
        return blank, False


def build_satellite_mosaic(xmin, xmax, ymin, ymax, zoom, cache_dir):
    """下载并拼接卫星瓦片，返回 mosaic 和绘图 extent。"""
    tx0, tx1, ty0, ty1 = tile_range_for_extent(xmin, xmax, ymin, ymax, zoom)

    nx = tx1 - tx0 + 1
    ny = ty1 - ty0 + 1
    tile_count = nx * ny

    print(f"[卫星底图] zoom={zoom}, 瓦片数量={tile_count} ({nx} x {ny})")

    mosaic = Image.new("RGB", (nx * 256, ny * 256))
    success_count = 0

    for ix, tx in enumerate(range(tx0, tx1 + 1)):
        for iy, ty in enumerate(range(ty0, ty1 + 1)):
            tile, ok = download_one_tile(zoom, tx, ty, cache_dir)
            if ok:
                success_count += 1
            mosaic.paste(tile, (ix * 256, iy * 256))

    if success_count == 0:
        raise RuntimeError("所有卫星瓦片都下载失败，请检查网络或更换瓦片源。")

    # imshow 的 extent：x 向右增大，y 取负后向北增大。
    left = tx0 * 256.0
    right = (tx1 + 1) * 256.0
    bottom = -(ty1 + 1) * 256.0
    top = -ty0 * 256.0

    return mosaic, (left, right, bottom, top)


def add_lonlat_ticks(ax, xmin, xmax, ymin, ymax, zoom, n_ticks=5):
    """在 Web Mercator 绘图坐标上显示经纬度刻度。"""
    lon_ticks = np.linspace(xmin, xmax, n_ticks)
    lat_ticks = np.linspace(ymin, ymax, n_ticks)

    x_ticks, _ = lonlat_to_plot_xy(lon_ticks, np.full_like(lon_ticks, ymin), zoom)
    _, y_ticks = lonlat_to_plot_xy(np.full_like(lat_ticks, xmin), lat_ticks, zoom)

    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.set_xticklabels([f"{lon:.2f}°E" for lon in lon_ticks], fontsize=9)
    ax.set_yticklabels([f"{lat:.2f}°N" for lat in lat_ticks], fontsize=9)


def add_projected_domain_rect(ax, info, zoom, **kwargs):
    """把经纬度矩形区域转成绘图坐标后绘制。"""
    x0, y0 = lonlat_to_plot_xy(info["StartLon"], info["EndLat"], zoom)
    x1, y1 = lonlat_to_plot_xy(info["EndLon"], info["StartLat"], zoom)

    rect = Rectangle(
        (x0, y0),
        x1 - x0,
        y1 - y0,
        fill=False,
        **kwargs
    )
    ax.add_patch(rect)
    return rect


def plot_plain_map(df, radar_info, nwp_info, output_png):
    """普通白底图版本。"""
    fig, ax = plt.subplots(figsize=(9, 8))

    radar_rect = Rectangle(
        (radar_info["StartLon"], radar_info["EndLat"]),
        radar_info["EndLon"] - radar_info["StartLon"],
        radar_info["StartLat"] - radar_info["EndLat"],
        fill=False,
        linewidth=2,
        edgecolor="blue",
        label="Radar / Label domain"
    )
    ax.add_patch(radar_rect)

    nwp_rect = Rectangle(
        (nwp_info["StartLon"], nwp_info["EndLat"]),
        nwp_info["EndLon"] - nwp_info["StartLon"],
        nwp_info["StartLat"] - nwp_info["EndLat"],
        fill=False,
        linewidth=2,
        linestyle="--",
        edgecolor="green",
        label="NWP domain"
    )
    ax.add_patch(nwp_rect)

    scatter = ax.scatter(
        df["max_lon"],
        df["max_lat"],
        c=df["max_rainfall_mm"],
        s=65,
        cmap="Reds",
        edgecolors="black",
        linewidths=0.5,
        label="Matched extreme events",
        zorder=5
    )

    cbar = plt.colorbar(scatter, ax=ax, shrink=0.85)
    cbar.set_label("Maximum rainfall (mm)")

    xmin, xmax, ymin, ymax = get_map_extent(radar_info, nwp_info)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Spatial Distribution of Matched Extreme Rainfall Events (n={len(df)})")

    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    plt.show()


def plot_satellite_map(df, radar_info, nwp_info, output_png):
    """卫星底图版本：直接下载并拼接在线瓦片。"""
    xmin, xmax, ymin, ymax = get_map_extent(radar_info, nwp_info)

    if SATELLITE_ZOOM == "auto":
        zoom = choose_auto_zoom(xmin, xmax, ymin, ymax, MAX_TILE_COUNT)
    else:
        zoom = int(SATELLITE_ZOOM)

    try:
        mosaic, extent = build_satellite_mosaic(
            xmin, xmax, ymin, ymax, zoom, SATELLITE_CACHE_DIR
        )
    except Exception as e:
        print("[警告] 卫星底图生成失败，改用普通白底图。")
        print(f"[错误信息] {e}")
        plot_plain_map(df, radar_info, nwp_info, output_png)
        return

    fig, ax = plt.subplots(figsize=(9, 8))

    ax.imshow(mosaic, extent=extent, origin="upper", zorder=0)

    # 设置精确显示范围
    plot_xmin, plot_ymin = lonlat_to_plot_xy(xmin, ymin, zoom)
    plot_xmax, plot_ymax = lonlat_to_plot_xy(xmax, ymax, zoom)
    ax.set_xlim(plot_xmin, plot_xmax)
    ax.set_ylim(plot_ymin, plot_ymax)

    # Radar / Label 区域范围
    add_projected_domain_rect(
        ax,
        radar_info,
        zoom,
        linewidth=2,
        edgecolor="deepskyblue",
        label="Radar / Label domain",
        zorder=4
    )

    # NWP 区域范围
    add_projected_domain_rect(
        ax,
        nwp_info,
        zoom,
        linewidth=2,
        linestyle="--",
        edgecolor="lime",
        label="NWP domain",
        zorder=4
    )

    # 事件点
    event_x, event_y = lonlat_to_plot_xy(
        df["max_lon"].to_numpy(),
        df["max_lat"].to_numpy(),
        zoom
    )

    scatter = ax.scatter(
        event_x,
        event_y,
        c=df["max_rainfall_mm"],
        s=70,
        cmap="Reds",
        edgecolors="white",
        linewidths=0.8,
        label="Matched extreme events",
        zorder=5
    )

    cbar = plt.colorbar(scatter, ax=ax, shrink=0.85)
    cbar.set_label("Maximum rainfall (mm)")

    if SHOW_LONLAT_TICKS:
        add_lonlat_ticks(ax, xmin, xmax, ymin, ymax, zoom)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
    else:
        ax.set_axis_off()

    ax.set_title(f"Spatial Distribution of Matched Extreme Rainfall Events (n={len(df)})")
    ax.grid(True, linestyle="--", alpha=0.25, color="white")
    ax.legend(loc="upper right")

    # 底图来源标注
    ax.text(
        0.99,
        0.01,
        "Satellite: Esri World Imagery",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7,
        color="white",
        bbox=dict(facecolor="black", alpha=0.35, edgecolor="none", pad=2),
        zorder=10
    )

    plt.tight_layout()
    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.show()


# =====================================================
# 8. 批量处理所有 CP 事件
# =====================================================
event_folders = find_cp_event_folders(root_dir)

print(f"共找到 {len(event_folders)} 个 CP 事件文件夹")

results = []

for idx, event_folder in enumerate(event_folders, 1):
    print(f"[{idx}/{len(event_folders)}] 正在处理: {event_folder.name}")

    result = process_one_event(
        event_folder,
        label_info,
        label_lon2d,
        label_lat2d
    )

    if result is not None:
        results.append(result)

if len(results) == 0:
    raise RuntimeError("没有成功读取任何 CP 事件，请检查 root_dir 或 LABEL/RA 文件夹结构。")

df = pd.DataFrame(results)
df = df.sort_values(by="max_rainfall_mm", ascending=False).reset_index(drop=True)

df.to_csv(output_csv, index=False, encoding="utf-8-sig")

print("\n成功读取事件数量:", len(df))
print(df.head())


# =====================================================
# 9. 绘制所有极端事件空间分布图
# =====================================================
if USE_SATELLITE_BACKGROUND:
    plot_satellite_map(df, radar_info, nwp_info, output_png)
else:
    plot_plain_map(df, radar_info, nwp_info, output_png)

print(f"\n图像已保存到: {output_png}")
print(f"事件统计表已保存到: {output_csv}")
