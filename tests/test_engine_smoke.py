"""引擎契约与精度冒烟（小网络、少 epoch，CI 友好）。完整回归见 engine/test_engine.py。"""
from __future__ import annotations

import numpy as np

from engine import data
from engine.metrics import r2_score, rmse
from engine.pipeline import FieldMapping, run_analysis
from engine.gnnwr_lite import GNNWRLite


HP = dict(hidden=(8, 4), lr=0.05, max_epoch=25, patience=10, l2_coef=1e-3)


def test_metrics_sane():
    y = np.array([1.0, 2, 3, 4, 5])
    assert abs(rmse(y, y)) < 1e-9
    assert abs(r2_score(y, y) - 1.0) < 1e-9


def test_lite_fits_toy():
    rng = np.random.default_rng(0)
    n, p = 80, 3
    coords = rng.normal(size=(n, 2))
    X = rng.normal(size=(n, p))
    y = X[:, 0] * 0.5 + coords[:, 0] * 0.2 + rng.normal(scale=0.05, size=n)
    m = GNNWRLite(hidden=(6, 4), lr=0.05, max_epoch=20, patience=8, seed=0)
    m.fit(coords, X, y, valid_ratio=0.2)
    pred = m.predict(coords, X)
    assert pred.shape == (n,)
    assert np.isfinite(pred).all()


def test_v1_contract_shape_housing_smoke():
    df, meta = data.make_housing(n=80, seed=1)
    mapping = FieldMapping(
        y=meta["y_column"], x=meta["x_columns"][:4],
        spatial=meta["spatial_columns"], temporal=None,
    )
    r = run_analysis(df, mapping, hyperparams=HP, seed=1)
    for k in ("r2", "rmse", "mae", "aicc"):
        assert k in r["metrics"]
    assert isinstance(r["history"], list) and r["history"]
    assert {"epoch", "train_loss", "val_loss"} <= set(r["history"][0])
    assert set(r["coefficients"]) >= {"columns", "points", "temporal", "times"}
    assert r["coefficients"]["points"]
    assert "coef" in r["coefficients"]["points"][0]
