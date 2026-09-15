"""
回归评价指标。

对应大纲第十章"验证指标"：R²、RMSE、MAE、AICc。
其中 AICc 需要"有效参数个数"，GWR/GNNWR 一类局部模型的有效参数
不是简单的自变量个数，而是帽子矩阵的迹 tr(S)——本模块允许显式传入
effective_params，供局部加权模型使用；不传时退化为普通线性模型口径。
"""
from __future__ import annotations

import numpy as np


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, float).ravel()
    y_pred = np.asarray(y_pred, float).ravel()
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot <= 1e-12:
        return 0.0
    return 1.0 - ss_res / ss_tot


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, float).ravel()
    y_pred = np.asarray(y_pred, float).ravel()
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, float).ravel()
    y_pred = np.asarray(y_pred, float).ravel()
    return float(np.mean(np.abs(y_true - y_pred)))


def aicc(y_true: np.ndarray, y_pred: np.ndarray, effective_params: float) -> float:
    """
    修正赤池信息量。n 较小时相比 AIC 更稳健。
    effective_params：局部模型传 tr(S)（帽子矩阵迹）；全局线性模型传 k+1。
    """
    y_true = np.asarray(y_true, float).ravel()
    y_pred = np.asarray(y_pred, float).ravel()
    n = y_true.size
    rss = float(np.sum((y_true - y_pred) ** 2))
    if rss <= 1e-12:
        rss = 1e-12
    sigma2 = rss / n
    k = float(effective_params)
    # 高斯似然下的 AICc
    aic = n * np.log(2 * np.pi * sigma2) + n
    denom = max(n - k - 1.0, 1e-6)
    return float(aic + 2 * k + (2 * k * (k + 1)) / denom)


def all_metrics(y_true, y_pred, effective_params: float | None = None) -> dict:
    """一次性给出四项指标；effective_params 缺省时按 2 个参数近似。"""
    ep = effective_params if effective_params is not None else 2.0
    return {
        "r2": round(r2_score(y_true, y_pred), 4),
        "rmse": round(rmse(y_true, y_pred), 4),
        "mae": round(mae(y_true, y_pred), 4),
        "aicc": round(aicc(y_true, y_pred, ep), 2),
    }
