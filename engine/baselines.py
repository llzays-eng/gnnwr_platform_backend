"""
基线模型：OLS / GWR / GTWR / RandomForest。

用于大纲要求的"精度对比报告"，回答一个关键问题：把 GNNWR 工具化后，
是否还优于传统全局回归（OLS）与经典地理加权（GWR/GTWR）。

- OLS：全局线性回归，空间平稳假设的下限参照。
- GWR：经典地理加权回归，用固定核+带宽做局部 WLS（空间非平稳基线）。
- GTWR：在 GWR 基础上把时间并入距离（时空非平稳基线）。
- RandomForest：非线性黑箱上限参照（精度可能高，但无逐点可解释系数）。
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from .metrics import all_metrics


def _add_intercept(X: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(X)), X])


def fit_ols(Xtr, ytr, Xte, yte) -> dict:
    A = _add_intercept(Xtr)
    beta, *_ = np.linalg.lstsq(A, ytr, rcond=None)
    pred = _add_intercept(Xte) @ beta
    m = all_metrics(yte, pred, effective_params=A.shape[1])
    m["method"] = "OLS"
    return m


def _weighted_local_fit(coord_tr, X_tr, y_tr, coord_q, X_q, knn=None, bandwidth=None):
    """
    对每个查询点用高斯核加权最小二乘估计局部系数，返回预测值。
    knn 给定时使用自适应带宽（=到第 k 近训练点的距离，GWR 标准做法）；
    否则使用固定 bandwidth。coord_* 含时间维时 d>2。
    """
    A_tr = _add_intercept(X_tr)
    preds = np.empty(len(X_q))
    for i in range(len(X_q)):
        d = np.sqrt(np.sum((coord_tr - coord_q[i]) ** 2, axis=1))
        if knn is not None:
            bw = np.partition(d, min(knn, len(d) - 1))[min(knn, len(d) - 1)]
            bw = bw if bw > 1e-9 else 1e-6
        else:
            bw = bandwidth
        w = np.exp(-(d ** 2) / (2 * bw ** 2)) + 1e-6
        AtWA = A_tr.T @ (A_tr * w[:, None])
        AtWy = A_tr.T @ (w * y_tr)
        try:
            beta = np.linalg.solve(AtWA + 1e-6 * np.eye(AtWA.shape[0]), AtWy)
        except np.linalg.LinAlgError:
            beta, *_ = np.linalg.lstsq(AtWA, AtWy, rcond=None)
        preds[i] = _add_intercept(X_q[i:i + 1])[0] @ beta
    return preds


def fit_gwr(coord_tr, Xtr, ytr, coord_te, Xte, yte, knn=None) -> dict:
    # 自适应带宽：默认取约 30% 训练点为局部邻域（跨样本量稳定）
    if knn is None:
        knn = max(12, int(0.3 * len(coord_tr)))
    pred = _weighted_local_fit(coord_tr, Xtr, ytr, coord_te, Xte, knn=knn)
    m = all_metrics(yte, pred)
    m["method"] = "GWR"
    return m


def fit_gtwr(coord_tr, Xtr, ytr, coord_te, Xte, yte, st_ratio=0.15) -> dict:
    """GTWR：把时间列按 st_ratio 缩放后并入坐标，再做 GWR。coord 最后一列为时间。"""
    ct = coord_tr.copy().astype(float)
    ce = coord_te.copy().astype(float)
    ct[:, -1] *= st_ratio
    ce[:, -1] *= st_ratio
    res = fit_gwr(ct, Xtr, ytr, ce, Xte, yte)
    res["method"] = "GTWR"
    return res


def fit_random_forest(Xtr, ytr, Xte, yte, seed=0) -> dict:
    rf = RandomForestRegressor(n_estimators=250, max_depth=None,
                               n_jobs=-1, random_state=seed)
    rf.fit(Xtr, ytr)
    pred = rf.predict(Xte)
    m = all_metrics(yte, pred)
    m["method"] = "RandomForest"
    return m
