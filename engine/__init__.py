"""GNNWR/GTNNWR 离线分析引擎（numpy 实现，无需 GPU/torch）。"""
from .gnnwr_lite import GNNWRLite
from .pipeline import run_pipeline
from . import data, baselines, metrics

__all__ = ["GNNWRLite", "run_pipeline", "data", "baselines", "metrics"]
