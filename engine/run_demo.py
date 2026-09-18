"""
跑通两个核心场景，导出：
  1) demo/housing_result.json / air_result.json —— 完整结果（后端结果表口径）
  2) demo/dashboard_data.json                   —— 独立看板消费的精简数据
控制台打印精度对比，验证 GNNWR/GTNNWR 相对基线是否有增益。
"""
from __future__ import annotations

import json
import os
import time

import numpy as np

from engine import data
from engine.pipeline import run_pipeline

OUT = os.path.join(os.path.dirname(__file__), "..", "demo")
os.makedirs(OUT, exist_ok=True)


def _print_compare(title, res):
    print(f"\n===== {title}（{res['model_type']}, "
          f"train={res['n_train']} / test={res['n_test']}）=====")
    tm = res["metrics"]["test"]
    print(f"{res['model_type']:>14s} | R²={tm['r2']:.4f}  RMSE={tm['rmse']:.4f}  "
          f"MAE={tm['mae']:.4f}  AICc={tm['aicc']:.1f}")
    for b in res["baselines"]:
        print(f"{b['method']:>14s} | R²={b['r2']:.4f}  RMSE={b['rmse']:.4f}  "
              f"MAE={b['mae']:.4f}")
    print(f"  收敛轮数={len(res['history']['train_loss'])}  "
          f"末轮 val_loss={res['history']['val_loss'][-1]:.5f}")


def main():
    t0 = time.time()

    # ---- 场景二：住宅价格（GNNWR·纯空间）----
    df_h, meta_h = data.make_housing(n=340, seed=42)
    res_h = run_pipeline(
        df_h, x_columns=meta_h["x_columns"], y_column=meta_h["y_column"],
        spatial_columns=meta_h["spatial_columns"], temporal_column=None,
        hyperparams=dict(hidden=(64, 32), lr=0.02, max_epoch=400,
                         patience=40, l2_coef=2e-4),
    )
    res_h["meta"] = meta_h
    _print_compare("场景二·城市住宅价格", res_h)

    # ---- 场景一：大气 PM2.5（GTNNWR·时空）----
    df_a, meta_a = data.make_air_quality(n_sites=90, n_days=14, seed=7)
    res_a = run_pipeline(
        df_a, x_columns=meta_a["x_columns"], y_column=meta_a["y_column"],
        spatial_columns=meta_a["spatial_columns"],
        temporal_column=meta_a["temporal_column"],
        hyperparams=dict(hidden=(48, 24), lr=0.02, max_epoch=300, patience=30),
    )
    res_a["meta"] = meta_a
    _print_compare("场景一·大气污染物 PM2.5", res_a)

    # ---- 落盘：完整结果 ----
    with open(os.path.join(OUT, "housing_result.json"), "w", encoding="utf-8") as f:
        json.dump(res_h, f, ensure_ascii=False, indent=2)
    with open(os.path.join(OUT, "air_result.json"), "w", encoding="utf-8") as f:
        json.dump(res_a, f, ensure_ascii=False, indent=2)

    # ---- 落盘：看板精简数据（两个场景合并）----
    def pack(res):
        return {
            "model_type": res["model_type"],
            "temporal": res["temporal"],
            "columns": res["coefficients"]["columns"],
            "points": res["coefficients"]["points"],
            "times": res["coefficients"]["times"],
            "metrics": res["metrics"]["test"],
            "history": res["history"],
            "beta_ols": res["beta_ols"],
            "baselines": res["baselines"],
            "meta": {"scenario": res["meta"]["scenario"],
                     "center": res["meta"]["center"],
                     "y_display": res["meta"]["y_display"],
                     "note": res["meta"]["true_coef_note"]},
        }

    dashboard = {"housing_price": pack(res_h), "air_quality": pack(res_a)}
    with open(os.path.join(OUT, "dashboard_data.json"), "w", encoding="utf-8") as f:
        json.dump(dashboard, f, ensure_ascii=False, indent=2)

    print(f"\n已导出 demo/*.json，用时 {time.time() - t0:.1f}s")
    return dashboard


if __name__ == "__main__":
    main()
