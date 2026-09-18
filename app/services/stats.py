"""系数摘要 / 残差直方图 / 简易 Moran's I。"""
from __future__ import annotations

import math

import numpy as np


def summarize_variable(name: str, values: np.ndarray) -> dict:
    v = np.asarray(values, float).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {
            "variable": name, "mean": 0, "std": 0, "min": 0, "max": 0,
            "q05": 0, "q25": 0, "q50": 0, "q75": 0, "q95": 0, "positive_ratio": 0,
        }
    qs = np.quantile(v, [0.05, 0.25, 0.5, 0.75, 0.95])
    return {
        "variable": name,
        "mean": round(float(v.mean()), 6),
        "std": round(float(v.std()), 6),
        "min": round(float(v.min()), 6),
        "max": round(float(v.max()), 6),
        "q05": round(float(qs[0]), 6),
        "q25": round(float(qs[1]), 6),
        "q50": round(float(qs[2]), 6),
        "q75": round(float(qs[3]), 6),
        "q95": round(float(qs[4]), 6),
        "positive_ratio": round(float((v > 0).mean()), 4),
    }


def residual_summary(residuals: np.ndarray, lons: np.ndarray | None = None,
                     lats: np.ndarray | None = None) -> dict:
    r = np.asarray(residuals, float).ravel()
    r = r[np.isfinite(r)]
    if r.size == 0:
        return {"mean": 0, "std": 0, "min": 0, "max": 0, "histogram": [], "morans_i": None}
    lo, hi = float(r.min()), float(r.max())
    bins = 24
    step = (hi - lo) / bins or 1.0
    hist = []
    for i in range(bins):
        a = lo + i * step
        b = a + step
        hist.append([round(a, 6), int(((r >= a) & (r < b if i < bins - 1 else r <= b)).sum())])
    moran = None
    if lons is not None and lats is not None and len(lons) == len(residuals) and len(lons) >= 8:
        try:
            moran = _morans_i(np.asarray(residuals, float),
                              np.asarray(lons, float), np.asarray(lats, float))
        except Exception:
            moran = None
    return {
        "mean": round(float(r.mean()), 6),
        "std": round(float(r.std()), 6),
        "min": round(lo, 6),
        "max": round(hi, 6),
        "histogram": hist,
        "morans_i": None if moran is None else round(float(moran), 4),
    }


def _morans_i(values: np.ndarray, lon: np.ndarray, lat: np.ndarray, k: int = 8) -> float:
    """k 近邻二进制权重的 Moran's I（量级参考，非严格推断）。"""
    n = len(values)
    if n > 2500:
        idx = np.linspace(0, n - 1, 2500).astype(int)
        values, lon, lat = values[idx], lon[idx], lat[idx]
        n = len(values)
    z = values - values.mean()
    coords = np.column_stack([lon, lat])
    w = np.zeros((n, n))
    for i in range(n):
        d = np.sqrt(((coords - coords[i]) ** 2).sum(1))
        d[i] = np.inf
        nn = np.argpartition(d, min(k, n - 1))[:k]
        w[i, nn] = 1.0
    w = 0.5 * (w + w.T)
    s0 = w.sum()
    if s0 <= 0:
        return 0.0
    num = float(z @ w @ z)
    den = float(z @ z)
    if den <= 1e-12:
        return 0.0
    return (n / s0) * (num / den)


def global_collapsed(summary: list[dict]) -> list[dict]:
    """把空间变系数摘要塌成一根线，给 OLS / RF 对照。"""
    out = []
    for c in summary:
        m = c["mean"]
        out.append({**c, "min": m, "max": m, "q05": m, "q25": m, "q50": m, "q75": m, "q95": m, "std": 0.0})
    return out


def best_by_metric(entries: list[dict], target: str) -> dict[str, str]:
    best = {}
    for key, higher in (("r2", True), ("rmse", False), ("mae", False), ("aicc", False)):
        ranked = [e for e in entries if e.get(key) is not None and not _nan(e.get(key))]
        if not ranked:
            best[key] = target
            continue
        ranked.sort(key=lambda e: e[key], reverse=higher)
        best[key] = ranked[0]["model"]
    return best


def _nan(v) -> bool:
    try:
        return v is None or (isinstance(v, float) and math.isnan(v))
    except Exception:
        return True
