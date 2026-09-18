"""
官方 gnnwr + PyTorch 的替换点。

生产环境：
  1. pip install gnnwr torch
  2. ENGINE_BACKEND=gnnwr_torch

本模块把官方包的输出整理成与 gnnwr_lite.run_analysis 相同的字典，
后端任务层 / 看板无需改动。当前仓库默认不安装 torch。
"""
from __future__ import annotations


def run_analysis(df, mapping, hyperparams=None, progress_cb=None,
                 test_ratio=0.2, valid_ratio=0.15, seed=42):
    try:
        import gnnwr  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "ENGINE_BACKEND=gnnwr_torch 但未安装官方 gnnwr 包。"
            "请 pip install gnnwr torch，或改回 ENGINE_BACKEND=gnnwr_lite。"
        ) from exc
    # 官方包适配应在接入真实环境时按 gnnwr 文档补全；在此明确失败以免 silently 跑错模型。
    raise RuntimeError(
        "gnnwr_torch 适配器尚未绑定具体训练 API。"
        "请在 engine/gnnwr_torch.py 中实现，并保持返回值满足 test_v1_contract_shape。"
    )
