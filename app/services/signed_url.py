"""地图瓦片短时签名：浏览器 <img>/WMS 无法带 Authorization。"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import jwt
from fastapi import HTTPException, status

from app.core.config import settings
from app.core.security import decode_token

def _cred() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="瓦片签名无效或已过期",
    )


def create_tile_token(user_id: str, task_id: str, expires_minutes: int | None = None) -> str:
    minutes = expires_minutes or settings.TILE_TOKEN_EXPIRE_MINUTES
    now = datetime.now(timezone.utc)
    payload = {
        "sub": user_id,
        "tid": task_id,
        "typ": "tile",
        "iat": now,
        "exp": now + timedelta(minutes=minutes),
    }
    return jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


def verify_tile_token(token: str | None, task_id: str) -> str:
    """校验 query `sig`，返回 user_id。失败抛 401。"""
    if not token:
        raise _cred()
    try:
        payload = decode_token(token)
    except jwt.PyJWTError:
        raise _cred()
    if payload.get("typ") != "tile" or payload.get("tid") != task_id:
        raise _cred()
    user_id = payload.get("sub")
    if not user_id:
        raise _cred()
    return str(user_id)


def append_query(url: str, **params: str) -> str:
    parts = urlsplit(url)
    q = dict(parse_qsl(parts.query, keep_blank_values=True))
    q.update({k: v for k, v in params.items() if v is not None})
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(q), parts.fragment))
