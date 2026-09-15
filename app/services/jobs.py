"""把任务丢给 Celery；broker 不可用时同步执行，保证本地/文档环境可跑通。"""
from __future__ import annotations

import logging
from typing import Callable

log = logging.getLogger(__name__)


def enqueue(delay_fn: Callable, *args, fallback: Callable | None = None, **kwargs):
    """
    delay_fn: celery task .delay
    fallback: 同步可调用，签名与任务一致
    返回 (celery_id | None, ran_inline: bool)
    """
    try:
        async_res = delay_fn(*args, **kwargs)
        return getattr(async_res, "id", None), False
    except Exception as exc:  # noqa: BLE001
        log.warning("Celery 不可用（%s），回退同步执行", exc)
        if fallback is None:
            raise
        fallback(*args, **kwargs)
        return None, True
