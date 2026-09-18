"""FastAPI 依赖：JWT 当前用户、角色守卫。"""
from __future__ import annotations

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.redis_client import is_jti_denied
from app.core.security import Role, decode_token
from app.models.user import User

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")

_CRED_EXC = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="无效或过期的凭证",
    headers={"WWW-Authenticate": "Bearer"},
)


def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
    try:
        payload = decode_token(token)
        if payload.get("typ") not in (None, "access"):
            raise _CRED_EXC
        if is_jti_denied(payload.get("jti")):
            raise _CRED_EXC
        user_id = payload.get("sub")
        if user_id is None:
            raise _CRED_EXC
    except jwt.PyJWTError:
        raise _CRED_EXC
    user = db.get(User, user_id)
    if user is None:
        raise _CRED_EXC
    return user


def require_admin(user: User = Depends(get_current_user)) -> User:
    if user.role != Role.admin.value:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="需要管理员权限")
    return user
