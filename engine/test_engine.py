"""
引擎回归测试（pytest 或直接 python 运行皆可）。
锁住两条核心保证（大纲 7.4 / 十）：
  1) 在内置空间/时空非平稳的数据上，GNNWR/GTNNWR 优于全局 OLS（时空场景亦优于 GWR）；
  2) 局部系数确实随空间变化、并能复现跨空间"变号"——这是选用该模型的意义所在。
同时锁住 v1 接口契约（FieldMapping / run_analysis），保证 Celery 任务不被引擎重构破坏。
"""
import numpy as np

from engine import data
from engine.metrics import r2_score, rmse
from engine.pipeline import FieldMapping, run_analysis

HP_H = dict(hidden=(64, 32), lr=0.02, max_epoch=400, patience=40, l2_coef=2e-4)
HP_A = dict(hidden=(48, 24), lr=0.02, max_epoch=300, patience=30)


def _run_housing():
    df, meta = data.make_housing(n=340, seed=42)
    m = FieldMapping(y=meta["y_column"], x=meta["x_columns"],
                     spatial=meta["spatial_columns"], temporal=None)
    return run_analysis(df, m, hyperparams=HP_H, seed=42)


def test_metrics_sane():
    y = np.array([1.0, 2, 3, 4, 5])
    assert abs(rmse(y, y)) < 1e-9
    assert abs(r2_score(y, y) - 1.0) < 1e-9


def test_gnnwr_beats_ols_housing():
    r = _run_housing()
    ols = next(b for b in r["baselines"] if b["method"] == "OLS")
    assert r["metrics"]["r2"] > ols["r2"], "GNNWR 应优于全局 OLS"
    assert r["metrics"]["r2"] > 0.85


def test_gtnnwr_beats_gwr_air():
    df, meta = data.make_air_quality(n_sites=90, n_days=14, seed=7)
    m = FieldMapping(y=meta["y_column"], x=meta["x_columns"],
                     spatial=meta["spatial_columns"], temporal=meta["temporal_column"])
    r = run_analysis(df, m, hyperparams=HP_A, seed=42)
    gwr = next(b for b in r["baselines"] if b["method"] == "GWR")
    assert r["metrics"]["r2"] > gwr["r2"], "GTNNWR（含时间）应优于忽略时间的 GWR"
    assert r["model_type"] == "GTNNWR" and r["coefficients"]["temporal"]


def test_local_coefficients_flip_sign():
    r = _run_housing()
    g = np.array([p["coef"]["green_ratio"] for p in r["coefficients"]["points"]])
    assert g.std() > 1e-3, "局部系数应随空间变化"
    assert (g > 0).any() and (g < 0).any(), "green_ratio 系数应跨空间变号（全局模型无法表达）"


def test_v1_contract_shape():
    r = _run_housing()
    for k in ("r2", "rmse", "mae", "aicc"):
        assert k in r["metrics"], f"扁平 metrics 缺 {k}"
    assert isinstance(r["history"], list) and {"epoch", "train_loss", "val_loss"} <= set(r["history"][0])
    assert set(r["coefficients"]) >= {"columns", "points", "temporal", "times"}


if __name__ == "__main__":
    for fn in [test_metrics_sane, test_gnnwr_beats_ols_housing,
               test_gtnnwr_beats_gwr_air, test_local_coefficients_flip_sign,
               test_v1_contract_shape]:
        fn()
        print(f"✓ {fn.__name__}")
    print("全部通过")
