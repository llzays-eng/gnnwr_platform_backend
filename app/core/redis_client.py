"""Redis 连接小封装：进度频道、WS ticket、取消标记、任务日志、token 黑名单。"""
from __future__ import annotations

from functools import lru_cache

import redis

from app.core.config import settings


@lru_cache
def get_sync_redis() -> redis.Redis:
    return redis.from_url(settings.redis_uri, decode_responses=True)


def redis_available() -> bool:
    try:
        get_sync_redis().ping()
        return True
    except Exception:
        return False


def blacklist_jti(jti: str, ttl_seconds: int) -> None:
    if not jti or ttl_seconds <= 0:
        return
    try:
        get_sync_redis().setex(f"auth:deny:{jti}", ttl_seconds, "1")
    except Exception:
        pass


def is_jti_denied(jti: str | None) -> bool:
    if not jti:
        return False
    try:
        return bool(get_sync_redis().get(f"auth:deny:{jti}"))
    except Exception:
        return False
