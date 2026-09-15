"""
坐标系转换（唯一入口）。
计算 / 入库几何一律 WGS84；渲染给高德时再纠偏为 GCJ-02。
CGCS2000 在本平台精度下与 WGS84 视为等价。
"""
from __future__ import annotations

import math

_A = 6378245.0
_EE = 0.00669342162296594323
_PI = math.pi


def out_of_china(lon: float, lat: float) -> bool:
    return not (72.004 <= lon <= 137.8347 and 0.8293 <= lat <= 55.8271)


def _transform_lat(x: float, y: float) -> float:
    ret = (-100.0 + 2.0 * x + 3.0 * y + 0.2 * y * y + 0.1 * x * y
           + 0.2 * math.sqrt(abs(x)))
    ret += (20.0 * math.sin(6.0 * x * _PI) + 20.0 * math.sin(2.0 * x * _PI)) * 2.0 / 3.0
    ret += (20.0 * math.sin(y * _PI) + 40.0 * math.sin(y / 3.0 * _PI)) * 2.0 / 3.0
    ret += (160.0 * math.sin(y / 12.0 * _PI) + 320 * math.sin(y * _PI / 30.0)) * 2.0 / 3.0
    return ret


def _transform_lon(x: float, y: float) -> float:
    ret = (300.0 + x + 2.0 * y + 0.1 * x * x + 0.1 * x * y
           + 0.1 * math.sqrt(abs(x)))
    ret += (20.0 * math.sin(6.0 * x * _PI) + 20.0 * math.sin(2.0 * x * _PI)) * 2.0 / 3.0
    ret += (20.0 * math.sin(x * _PI) + 40.0 * math.sin(x / 3.0 * _PI)) * 2.0 / 3.0
    ret += (150.0 * math.sin(x / 12.0 * _PI) + 300.0 * math.sin(x / 30.0 * _PI)) * 2.0 / 3.0
    return ret


def wgs84_to_gcj02(lon: float, lat: float) -> tuple[float, float]:
    if out_of_china(lon, lat):
        return lon, lat
    dlat = _transform_lat(lon - 105.0, lat - 35.0)
    dlon = _transform_lon(lon - 105.0, lat - 35.0)
    radlat = lat / 180.0 * _PI
    magic = math.sin(radlat)
    magic = 1 - _EE * magic * magic
    sqrtmagic = math.sqrt(magic)
    dlat = (dlat * 180.0) / ((_A * (1 - _EE)) / (magic * sqrtmagic) * _PI)
    dlon = (dlon * 180.0) / (_A / sqrtmagic * math.cos(radlat) * _PI)
    return lon + dlon, lat + dlat


def gcj02_to_wgs84(lon: float, lat: float, rounds: int = 3) -> tuple[float, float]:
    """迭代反解，东部沿海可达亚米级。"""
    if out_of_china(lon, lat):
        return lon, lat
    wlon, wlat = lon, lat
    for _ in range(rounds):
        glon, glat = wgs84_to_gcj02(wlon, wlat)
        wlon += lon - glon
        wlat += lat - glat
    return wlon, wlat


def normalize_crs(crs: str | None) -> str:
    if not crs:
        return "WGS84"
    c = crs.strip().upper().replace("_", "-")
    if c in ("WGS84", "WGS-84", "EPSG:4326", "4326"):
        return "WGS84"
    if c in ("CGCS2000", "CGCS-2000", "EPSG:4490", "4490"):
        return "WGS84"
    if c in ("GCJ-02", "GCJ02", "AMAP", "GAODE"):
        return "GCJ02"
    return "WGS84"


def transform(lon: float, lat: float, src: str, dst: str) -> tuple[float, float]:
    src = normalize_crs(src)
    dst = normalize_crs(dst)
    if src == dst:
        return lon, lat
    if src == "GCJ02":
        lon, lat = gcj02_to_wgs84(lon, lat)
    if dst == "GCJ02":
        return wgs84_to_gcj02(lon, lat)
    return lon, lat


def offshore_ratio(points: list[tuple[float, float]], sample: int = 200) -> float:
    if not points:
        return 0.0
    step = max(1, len(points) // sample)
    checked = bad = 0
    for i in range(0, len(points), step):
        lon, lat = points[i]
        checked += 1
        if out_of_china(lon, lat):
            bad += 1
    return bad / checked if checked else 0.0
