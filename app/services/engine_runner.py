"""
引擎切换点。

默认 gnnwr_lite（纯 numpy，本仓库 engine/）。
设置 ENGINE_BACKEND=gnnwr_torch 且安装官方 gnnwr+torch 后，走 engine/gnnwr_torch.py。
对外契约由 engine.test_engine.test_v1_contract_shape / tests/test_engine_smoke.py 锁定。
"""
from __future__ import annotations

import logging

from app.core.config import settings

log = logging.getLogger(__name__)


def run_analysis(df, mapping, hyperparams=None, progress_cb=None,
                 test_ratio=0.2, valid_ratio=0.15, seed=42) -> dict:
    backend = (settings.ENGINE_BACKEND or "gnnwr_lite").strip().lower()
    if backend in ("gnnwr_torch", "gnnwr", "torch"):
        try:
            from engine.gnnwr_torch import run_analysis as impl
            log.info("engine backend = gnnwr_torch")
            return impl(df, mapping, hyperparams=hyperparams, progress_cb=progress_cb,
                        test_ratio=test_ratio, valid_ratio=valid_ratio, seed=seed)
        except Exception as exc:  # noqa: BLE001
            log.warning("gnnwr_torch 不可用（%s），回退 gnnwr_lite", exc)
    from engine.pipeline import run_analysis as impl
    return impl(df, mapping, hyperparams=hyperparams, progress_cb=progress_cb,
                test_ratio=test_ratio, valid_ratio=valid_ratio, seed=seed)
