"""
曲面切片渲染。

计算侧系数点是 WGS84。前端高德底图是 GCJ-02，栅格无法逐点纠偏，
因此本服务把 XYZ / WMS 的经纬度解释为 GCJ-02：像素 lon/lat → 反解 WGS84 → IDW 采样。
响应必须带 tile_crs='GCJ02'（如实声明）。
WMS TIME 维度：当前不支持（time_dimension 仅回放点层时间轴）。
"""
from __future__ import annotations

import io
import json
import math
from functools import lru_cache

import numpy as np
from PIL import Image

from app.core.config import settings
from app.services.crs_transform import gcj02_to_wgs84, wgs84_to_gcj02
from app.services.storage import storage

_VIRIDIS = [
    (68, 1, 84), (72, 40, 120), (62, 74, 137), (49, 104, 142),
    (38, 130, 142), (31, 158, 137), (53, 183, 121), (109, 205, 89),
    (180, 222, 44), (253, 231, 37),
]


def _colormap(t: float) -> tuple[int, int, int, int]:
    t = min(1.0, max(0.0, t))
    x = t * (len(_VIRIDIS) - 1)
    i = int(x)
    f = x - i
    if i >= len(_VIRIDIS) - 1:
        r, g, b = _VIRIDIS[-1]
    else:
        a, c = _VIRIDIS[i], _VIRIDIS[i + 1]
        r = int(a[0] + (c[0] - a[0]) * f)
        g = int(a[1] + (c[1] - a[1]) * f)
        b = int(a[2] + (c[2] - a[2]) * f)
    return r, g, b, 200


def _load_points(task_id: str) -> dict:
    key = f"results/{task_id}/coefficients.json"
    raw = json.loads(storage.get_bytes(key).decode("utf-8"))
    pts = raw.get("points") or []
    lons = np.array([p["lon"] for p in pts], float)
    lats = np.array([p["lat"] for p in pts], float)
    cols = raw.get("columns") or (list(pts[0]["coef"].keys()) if pts else [])
    primary = cols[0] if cols else "intercept"
    vals = np.array([p.get("coef", {}).get(primary, 0.0) for p in pts], float)
    return {
        "lons": lons, "lats": lats, "vals": vals, "primary": primary,
        "times": raw.get("times") or [], "temporal": bool(raw.get("temporal")),
        "n": len(pts),
    }


@lru_cache(maxsize=32)
def load_points_cached(task_id: str) -> dict:
    d = _load_points(task_id)
    # ndarray 不能进 lru 返回后被原地改，这里转 tuple 不够；每次调用重新 load 也可。
    # 为简单起见缓存文件解析结果的 python 列表。
    return {
        "lons": d["lons"].tolist(),
        "lats": d["lats"].tolist(),
        "vals": d["vals"].tolist(),
        "primary": d["primary"],
        "times": d["times"],
        "temporal": d["temporal"],
        "n": d["n"],
    }


def field_bbox_gcj02(task_id: str) -> list[float]:
    d = load_points_cached(task_id)
    if not d["n"]:
        return [73.0, 18.0, 135.0, 54.0]
    xs, ys = [], []
    for lon, lat in zip(d["lons"], d["lats"]):
        g = wgs84_to_gcj02(float(lon), float(lat))
        xs.append(g[0]); ys.append(g[1])
    pad = 0.02
    return [min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad]


def _idw(lons, lats, vals, qlon, qlat, k=12, power=2.0) -> float:
    dx = lons - qlon
    dy = lats - qlat
    dist = np.sqrt(dx * dx + dy * dy)
    if dist.min() < 1e-9:
        return float(vals[int(dist.argmin())])
    if len(vals) > k:
        idx = np.argpartition(dist, k)[:k]
        dist, vals = dist[idx], vals[idx]
    w = 1.0 / np.power(dist + 1e-12, power)
    return float(np.sum(w * vals) / np.sum(w))


def render_bbox_png(task_id: str, min_lon: float, min_lat: float,
                    max_lon: float, max_lat: float, width=256, height=256,
                    tile_crs: str = "GCJ02") -> bytes:
    d = load_points_cached(task_id)
    lons = np.asarray(d["lons"], float)
    lats = np.asarray(d["lats"], float)
    vals = np.asarray(d["vals"], float)
    if d["n"] == 0:
        img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        buf = io.BytesIO(); img.save(buf, format="PNG")
        return buf.getvalue()

    vmin, vmax = float(np.nanpercentile(vals, 5)), float(np.nanpercentile(vals, 95))
    if abs(vmax - vmin) < 1e-9:
        vmax = vmin + 1e-6

    xs = np.linspace(min_lon, max_lon, width)
    ys = np.linspace(max_lat, min_lat, height)  # 图像 y 向下
    pixels = np.zeros((height, width, 4), dtype=np.uint8)
    # 降采样网格加速：先 64 再双线性放大对演示够用；小图直接算
    step = 2 if width >= 256 else 1
    grid = np.zeros((math.ceil(height / step), math.ceil(width / step)))
    for iy, y in enumerate(ys[::step]):
        for ix, x in enumerate(xs[::step]):
            if tile_crs.upper() in ("GCJ02", "GCJ-02"):
                wlon, wlat = gcj02_to_wgs84(float(x), float(y))
            else:
                wlon, wlat = float(x), float(y)
            grid[iy, ix] = _idw(lons, lats, vals, wlon, wlat)

    from PIL import Image as PImage
    gimg = PImage.fromarray(
        ((np.clip(grid, vmin, vmax) - vmin) / (vmax - vmin) * 255).astype(np.uint8), mode="L"
    )
    gimg = gimg.resize((width, height), PImage.BILINEAR)
    t = np.asarray(gimg, float) / 255.0
    for iy in range(height):
        for ix in range(width):
            pixels[iy, ix] = _colormap(float(t[iy, ix]))
    img = Image.fromarray(pixels, "RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def xyz_to_bbox_lonlat(z: int, x: int, y: int) -> tuple[float, float, float, float]:
    """Web Mercator 瓦片 → 经纬度 bbox（后续按 tile_crs 解释）。"""
    n = 2 ** z
    min_lon = x / n * 360.0 - 180.0
    max_lon = (x + 1) / n * 360.0 - 180.0

    def lat(y_):
        rad = math.pi * (1 - 2 * y_ / n)
        return math.degrees(math.atan(math.sinh(rad)))

    max_lat = lat(y)
    min_lat = lat(y + 1)
    return min_lon, min_lat, max_lon, max_lat


def render_xyz(task_id: str, z: int, x: int, y: int) -> bytes:
    min_lon, min_lat, max_lon, max_lat = xyz_to_bbox_lonlat(z, x, y)
    return render_bbox_png(
        task_id, min_lon, min_lat, max_lon, max_lat,
        tile_crs=settings.SURFACE_TILE_CRS,
    )


def legend_png(task_id: str, width=24, height=128) -> bytes:
    d = load_points_cached(task_id)
    vals = np.asarray(d["vals"], float)
    pixels = np.zeros((height, width, 4), dtype=np.uint8)
    for iy in range(height):
        t = 1.0 - iy / max(height - 1, 1)
        pixels[iy, :] = _colormap(t)
    img = Image.fromarray(pixels, "RGBA")
    buf = io.BytesIO(); img.save(buf, format="PNG")
    return buf.getvalue()
