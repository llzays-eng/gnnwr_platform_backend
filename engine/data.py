"""
合成数据生成器（离线可复现，用于跑通闭环与看板演示）。

两个场景都刻意构造"系数随空间/时空变化"的真值，这样才能验证
GNNWR/GTNNWR 能否复现空间非平稳性——这是选用该模型的根本理由。
真实上线时把这里替换为 RESDC/环境监测总站/住建网签等真实数据即可，
下游引擎与看板不需要改动。

约定的统一输入形式（大纲 2.3）：
    一个 (经度 lon, 纬度 lat[, 时间 t]) + 若干属性列 X + 一个目标 y
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---- 场景二：城市住宅价格（GNNWR·纯空间） ----
HOUSING_X = ["area", "age", "floor_ratio", "green_ratio",
             "dist_subway", "dist_cbd", "school_score", "poi_density"]

# ---- 场景一：大气污染物 PM2.5（GTNNWR·时空） ----
AIR_X = ["aod", "temp", "humidity", "wind", "blh", "dem", "ntl"]


def _standardize(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        mu, sd = out[c].mean(), out[c].std()
        out[c] = (out[c] - mu) / (sd if sd > 1e-9 else 1.0)
    return out


def make_housing(n: int = 340, seed: int = 42) -> tuple[pd.DataFrame, dict]:
    """
    模拟某城市二手房单价，刻意构造**跨空间变号**的非平稳系数——这类关系
    全局 OLS 根本无法表达（全局系数会把正负效应平均抵消），只有空间变系数
    模型才能复现，因此是检验 GNNWR 的最佳场景：

      · 容积率 floor_ratio：市中心密度=便利(正) → 郊区密度=拥挤(负)，沿半径变号
      · 绿化率 green_ratio：东城稀缺被追捧(正) → 西城本就多不加价(负)，沿东西向变号
      · 学区 school_score / 地铁邻近性：市中心权重高，郊区趋弱

    并有意**解耦协变量与半径**（dist_cbd、poi_density 加大独立噪声），
    使空间效应"不可被全局回归吸收"。返回 (标准化后 DataFrame, 真值元数据)。
    """
    rng = np.random.default_rng(seed)
    lon0, lat0 = 114.06, 22.54          # 近似深圳市中心
    lon = lon0 + rng.uniform(-0.28, 0.28, n)
    lat = lat0 + rng.uniform(-0.22, 0.22, n)
    r = np.clip(np.sqrt(((lon - lon0) / 0.28) ** 2 + ((lat - lat0) / 0.22) ** 2), 0, 1.2)
    east = (lon - lon0) / 0.28          # 东西向坐标 [-1,1]

    area = rng.uniform(45, 145, n)
    age = rng.uniform(1, 35, n)
    floor_ratio = rng.uniform(1.5, 6.5, n)
    green_ratio = rng.uniform(0.15, 0.45, n)
    dist_subway = np.abs(rng.normal(1.0, 0.7, n)) + 0.05          # km
    dist_cbd = np.clip(r * 12 + rng.normal(6, 5, n), 0.3, None)   # 大独立噪声→与 r 解耦
    school_score = rng.uniform(0, 100, n)
    poi_density = np.clip(rng.normal(55, 22, n), 1, None)         # 不再乘 (1.2-r)

    X = pd.DataFrame(dict(area=area, age=age, floor_ratio=floor_ratio,
                          green_ratio=green_ratio, dist_subway=dist_subway,
                          dist_cbd=dist_cbd, school_score=school_score,
                          poi_density=poi_density))

    # —— 空间非平稳真系数场（标准化尺度）——
    b_area = 0.45 + 0.6 * r                      # 郊区面积更值钱
    b_age = -0.55 - 0.25 * (1 - r)               # 处处贬值，市中心更敏感
    b_floor = 0.9 * (0.5 - r)                    # 变号：中心正、郊区负
    b_green = 0.7 * east                         # 变号：东城正、西城负
    b_subway = -(1.2 - 0.9 * r)                  # 距地铁越远越便宜，中心权重大
    b_cbd = -0.55 * np.ones(n)                   # 近似全局，不给 OLS "免费"空间信号
    b_school = 1.15 * (1 - r) - 0.15             # 中心强正，远郊转弱/微负
    b_poi = 0.15 + 0.55 * (1 - r)

    Xs = _standardize(X, HOUSING_X)
    noise = rng.normal(0, 0.32, n)
    y_std = (b_area * Xs.area + b_age * Xs.age + b_floor * Xs.floor_ratio
             + b_green * Xs.green_ratio + b_subway * Xs.dist_subway
             + b_cbd * Xs.dist_cbd + b_school * Xs.school_score
             + b_poi * Xs.poi_density + noise)
    price = 55000 + 12000 * y_std               # 映射到元/㎡量级（仅为可读）

    df = Xs.copy()
    df.insert(0, "lon", lon)
    df.insert(1, "lat", lat)
    df["price"] = price
    df["price_std"] = (price - price.mean()) / price.std()

    meta = {
        "scenario": "housing_price", "model": "GNNWR", "temporal": False,
        "x_columns": HOUSING_X, "y_column": "price_std", "y_display": "price",
        "spatial_columns": ["lon", "lat"], "temporal_column": None,
        "center": [lon0, lat0], "n": n,
        "true_coef_note": "容积率沿半径变号、绿化率沿东西向变号，"
                          "学区/地铁邻近性在市中心权重更高——全局回归无法表达",
    }
    return df, meta


def make_air_quality(n_sites: int = 90, n_days: int = 14, seed: int = 7
                     ) -> tuple[pd.DataFrame, dict]:
    """
    模拟京津冀区域地面 PM2.5 与 AOD/气象协变量。AOD→PM2.5 的转换系数
    既随空间变化（城市 vs 郊野），也随时间变化（如冷锋过境后系数下降），
    构成时空非平稳——对应 GTNNWR 的用武之地。
    """
    rng = np.random.default_rng(seed)
    lon0, lat0 = 116.4, 39.9            # 近似北京
    site_lon = lon0 + rng.uniform(-2.2, 2.2, n_sites)
    site_lat = lat0 + rng.uniform(-1.8, 1.8, n_sites)
    urban = (np.sqrt(((site_lon - lon0) / 2.2) ** 2
                     + ((site_lat - lat0) / 1.8) ** 2) < 0.55).astype(float)

    rows = []
    for d in range(n_days):
        # 逐日天气态：一次"冷锋过境"让 t>=8 的转换效率整体下降
        front = 1.0 if d < 8 else 0.6
        for s in range(n_sites):
            aod = np.clip(rng.normal(0.6 + 0.3 * urban[s], 0.25), 0.05, None)
            temp = rng.normal(8 - 0.3 * d, 4)
            humidity = np.clip(rng.normal(55 + 2 * d, 12), 5, 100)
            wind = np.clip(rng.normal(2.5 + (0.0 if d < 8 else 1.6), 1.0), 0.1, None)
            blh = np.clip(rng.normal(700 + 40 * wind, 150), 100, None)
            dem = np.clip(rng.normal(50 + 200 * (1 - urban[s]), 60), 0, None)
            ntl = np.clip(rng.normal(45 * urban[s] + 5, 10), 0, None)

            # 时空非平稳真系数：AOD 城市系数更高、冷锋后整体下降
            k_aod = (2.2 + 1.3 * urban[s]) * front
            pm = (18
                  + k_aod * aod * 30
                  - 0.15 * wind * 6
                  - 0.02 * blh * 0.1
                  + 0.08 * humidity * 0.3
                  + 0.01 * dem * 0.2
                  + 0.05 * ntl
                  + rng.normal(0, 6))
            pm = float(np.clip(pm, 3, 500))
            rows.append(dict(lon=site_lon[s], lat=site_lat[s], t=d,
                             aod=aod, temp=temp, humidity=humidity, wind=wind,
                             blh=blh, dem=dem, ntl=ntl, pm25=pm))

    df = pd.DataFrame(rows)
    dfx = _standardize(df, AIR_X)
    dfx["lon"], dfx["lat"], dfx["t"] = df["lon"], df["lat"], df["t"]
    dfx["pm25"] = df["pm25"]
    dfx["pm25_std"] = (df["pm25"] - df["pm25"].mean()) / df["pm25"].std()

    meta = {
        "scenario": "air_quality", "model": "GTNNWR", "temporal": True,
        "x_columns": AIR_X, "y_column": "pm25_std", "y_display": "pm25",
        "spatial_columns": ["lon", "lat"], "temporal_column": "t",
        "center": [lon0, lat0], "n": len(df), "n_days": n_days, "n_sites": n_sites,
        "true_coef_note": "AOD→PM2.5 转换系数城市高于郊野，且冷锋过境(第8天)后整体下降",
    }
    return dfx, meta
