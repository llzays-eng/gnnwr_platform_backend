"""把任务丢给 Celery。默认同步回退关闭；仅 ALLOW_INLINE_JOBS=true 时在 API 进程内执行。"""
from __future__ import annotations

import logging
from typing import Callable

from app.core.config import settings

log = logging.getLogger(__name__)


class BrokerUnavailable(Exception):
    """Celery broker 不可达，且未允许 inline 回退。"""


def enqueue(delay_fn: Callable, *args, fallback: Callable | None = None, **kwargs):
    """
    delay_fn: celery task .delay
    fallback: 仅当 settings.ALLOW_INLINE_JOBS 时调用
    返回 (celery_id | None, ran_inline: bool)
    """
    try:
        async_res = delay_fn(*args, **kwargs)
        return getattr(async_res, "id", None), False
    except Exception as exc:  # noqa: BLE001
        if settings.ALLOW_INLINE_JOBS and fallback is not None:
            log.warning("Celery 不可用（%s），ALLOW_INLINE_JOBS=true，改为进程内执行", exc)
            fallback(*args, **kwargs)
            return None, True
        log.error("Celery 不可用（%s），拒绝 inline 回退", exc)
        raise BrokerUnavailable(
            "任务队列不可用。请确认 Redis/Celery worker 已启动，"
            "或在开发环境设置 ALLOW_INLINE_JOBS=true。"
        ) from exc
