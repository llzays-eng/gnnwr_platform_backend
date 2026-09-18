"""WebSocket 一次性 ticket（Redis GETDEL）。"""
from __future__ import annotations

import json
import secrets
from typing import Any

from app.core.redis_client import get_sync_redis

TICKET_TTL_S = 120


def issue_ws_ticket(user_id: str, task_id: str | None = None, client: Any = None,
                    ttl: int = TICKET_TTL_S) -> str:
    r = client or get_sync_redis()
    ticket = secrets.token_urlsafe(24)
    payload = json.dumps({"user_id": user_id, "task_id": task_id})
    r.setex(f"ws:ticket:{ticket}", ttl, payload)
    return ticket


def consume_ws_ticket(ticket: str, task_id: str | None = None, client: Any = None) -> str | None:
    """
    一次性消费。成功返回 user_id；无效/过期/已用/绑定任务不匹配返回 None。
    兼容旧格式（值为纯 user_id 字符串）。
    """
    if not ticket:
        return None
    r = client or get_sync_redis()
    key = f"ws:ticket:{ticket}"
    raw = _getdel(r, key)
    if not raw:
        return None
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    user_id, bound = _parse(raw)
    if not user_id:
        return None
    if bound and task_id and bound != task_id:
        return None
    return user_id


def _getdel(r: Any, key: str) -> str | None:
    if hasattr(r, "getdel"):
        try:
            return r.getdel(key)
        except Exception:
            pass
    val = r.get(key)
    if val is None:
        return None
    try:
        r.delete(key)
    except Exception:
        pass
    return val


def _parse(raw: str) -> tuple[str | None, str | None]:
    raw = raw.strip()
    if raw.startswith("{"):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return None, None
        return data.get("user_id"), data.get("task_id")
    return raw, None
