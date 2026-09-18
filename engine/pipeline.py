"""
端到端分析管线（对应大纲第八章流程图的 ML 部分）。

职责：给定"统一输入形式"的数据 + 字段映射 + 超参数，完成
    划分 → 训练 GNNWR/GTNNWR → 逐点系数 → 残差 → 基线对比 → 汇总结果
并输出一个 JSON 友好的结果字典（后端持久化 / 看板直接消费）。

后端 Celery 任务 train_task 直接调用 run_pipeline；离线独立看板的数据
也由本管线导出，两者格式一致。
"""
from __future__ import annotations

import numpy as np

from . import baselines
from .gnnwr_lite import GNNWRLite
from .metrics import all_metrics


def _split(n, test_ratio, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_te = max(1, int(n * test_ratio))
    return idx[n_te:], idx[:n_te]        # train_idx, test_idx


def run_pipeline(df, x_columns, y_column, spatial_columns,
                 temporal_column=None, hyperparams=None,
                 test_ratio=0.2, valid_ratio=0.15,
                 run_baselines=True, seed=42, epoch_cb=None) -> dict:
    hp = dict(hidden=(32, 16), lr=0.01, max_epoch=200,
              patience=25, dropout=0.0, batch_size=None, l2_coef=1e-3)
    if hyperparams:
        hp.update({k: v for k, v in hyperparams.items() if v is not None})

    temporal = temporal_column is not None
    coord_cols = list(spatial_columns) + ([temporal_column] if temporal else [])

    X = df[x_columns].to_numpy(float)
    y = df[y_column].to_numpy(float)
    coords_full = df[coord_cols].to_numpy(float)          # 含时间维（若有）
    space_full = df[spatial_columns].to_numpy(float)      # 仅经纬度，供作图
    n = len(df)

    tr, te = _split(n, test_ratio, seed)

    # —— 主模型：GNNWR / GTNNWR（lite）——
    model = GNNWRLite(hidden=hp["hidden"], lr=hp["lr"], max_epoch=hp["max_epoch"],
                      patience=hp["patience"], dropout=hp["dropout"],
                      l2_coef=hp["l2_coef"], batch_size=hp["batch_size"], seed=seed)
    model.fit(coords_full[tr], X[tr], y[tr], valid_ratio=valid_ratio, epoch_cb=epoch_cb)

    test_metrics = model.evaluate(coords_full[te], X[te], y[te])
    train_metrics = model.evaluate(coords_full[tr], X[tr], y[tr])

    # 全样本逐点系数与残差（供空间热力图 / 残差诊断）
    beta_all = model.local_coefficients(coords_full)      # (n, p+1)
    pred_all = model.predict(coords_full, X)
    resid_all = y - pred_all

    coef_columns = ["intercept"] + list(x_columns)
    times = sorted(set(df[temporal_column].tolist())) if temporal else []

    points = []
    for i in range(n):
        rec = {
            "lon": float(space_full[i, 0]),
            "lat": float(space_full[i, 1]),
            "coef": {coef_columns[k]: round(float(beta_all[i, k]), 4)
                     for k in range(len(coef_columns))},
            "residual": round(float(resid_all[i]), 4),
        }
        if temporal:
            rec["time"] = float(df[temporal_column].iloc[i])
        points.append(rec)

    # —— 基线对比 ——
    comparisons = []
    if run_baselines:
        comparisons.append(baselines.fit_ols(X[tr], y[tr], X[te], y[te]))
        comparisons.append(baselines.fit_gwr(space_full[tr], X[tr], y[tr],
                                              space_full[te], X[te], y[te]))
        if temporal:
            comparisons.append(baselines.fit_gtwr(coords_full[tr], X[tr], y[tr],
                                                   coords_full[te], X[te], y[te]))
        comparisons.append(baselines.fit_random_forest(X[tr], y[tr], X[te], y[te], seed))

    # 全局 OLS 系数（给"局部 vs 全局"对照）
    Aall = np.column_stack([np.ones(len(tr)), X[tr]])
    beta_ols, *_ = np.linalg.lstsq(Aall, y[tr], rcond=None)
    beta_ols_map = {coef_columns[k]: round(float(beta_ols[k]), 4)
                    for k in range(len(coef_columns))}

    return {
        "model_type": "GTNNWR" if temporal else "GNNWR",
        "temporal": temporal,
        "coef_columns": coef_columns,
        "x_columns": list(x_columns),
        "metrics": {"test": test_metrics, "train": train_metrics},
        "history": model.history,
        "coefficients": {
            "columns": coef_columns,
            "points": points,
            "temporal": temporal,
            "times": times,
        },
        "beta_ols": beta_ols_map,
        "baselines": comparisons,
        "n_train": int(len(tr)),
        "n_test": int(len(te)),
    }


# ============================================================
# 兼容层：保留 v1 公开接口（backend/app/tasks/training.py 与
# engine/test_engine.py 依赖），内部委托给调优后的 run_pipeline。
# ============================================================
from dataclasses import dataclass, field


@dataclass
class FieldMapping:
    """字段映射（大纲 2.3「字段可配置」的载体）。"""
    y: str
    x: list = field(default_factory=list)
    spatial: list = field(default_factory=list)   # [lon_col, lat_col]
    temporal: str | None = None


def run_analysis(df, mapping: FieldMapping, hyperparams=None,
                 progress_cb=None, test_ratio=0.2, valid_ratio=0.15,
                 seed=42) -> dict:
    """
    v1 接口：返回扁平 metrics（测试集）+ 逐 epoch 记录列表，
    供 Celery 任务直接落库、经 WebSocket 推送训练进度。
    """
    res = run_pipeline(
        df,
        x_columns=list(mapping.x),
        y_column=mapping.y,
        spatial_columns=list(mapping.spatial),
        temporal_column=mapping.temporal,
        hyperparams=hyperparams,
        test_ratio=test_ratio,
        valid_ratio=valid_ratio,
        seed=seed,
        epoch_cb=progress_cb,
    )
    hist = res.get("history") or {}
    records = [
        {"epoch": i + 1, "train_loss": t, "val_loss": v}
        for i, (t, v) in enumerate(zip(hist.get("train_loss", []),
                                       hist.get("val_loss", [])))
    ]
    out = dict(res)
    out["metrics"] = res["metrics"]["test"]        # 扁平化：r2/rmse/mae/aicc
    out["train_metrics"] = res["metrics"]["train"]
    out["history"] = records
    return out
