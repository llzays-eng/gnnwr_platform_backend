"""认证：register / login（JSON + OAuth2 表单）/ me / refresh / logout / ws-ticket。"""
from __future__ import annotations

from datetime import datetime, timezone

import jwt
from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.deps import get_current_user
from app.core.redis_client import blacklist_jti, redis_available
from app.core.security import (
    Role, create_access_token, create_refresh_token, decode_token,
    hash_password, verify_password,
)
from app.models.user import User
from app.schemas.auth import LoginResponse, RefreshRequest, UserCreate, UserProfile, WsTicketOut
from app.services.serialize import login_response, user_profile
from app.services.tickets import TICKET_TTL_S, issue_ws_ticket

router = APIRouter(prefix="/auth", tags=["auth"])


async def _extract_credentials(request: Request) -> tuple[str, str]:
    ct = (request.headers.get("content-type") or "").lower()
    if "application/json" in ct:
        data = await request.json()
        username = data.get("username") or data.get("email")
        password = data.get("password")
        if not username or not password:
            raise HTTPException(status_code=422, detail="需要 username 与 password")
        return str(username), str(password)
    form = await request.form()
    username = form.get("username")
    password = form.get("password")
    if not username or not password:
        raise HTTPException(status_code=422, detail="需要 username 与 password")
    return str(username), str(password)


@router.post("/register", response_model=UserProfile, status_code=201)
def register(payload: UserCreate, db: Session = Depends(get_db)):
    exists = db.scalar(select(User).where(
        or_(User.email == payload.email, User.username == payload.username)
    ))
    if exists:
        raise HTTPException(status_code=400, detail="邮箱或用户名已注册")
    user = User(
        email=payload.email,
        username=payload.username,
        display_name=payload.display_name or payload.username,
        hashed_password=hash_password(payload.password),
        role=Role.user.value,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user_profile(user)


@router.post("/login", response_model=LoginResponse)
async def login(request: Request, db: Session = Depends(get_db)):
    username, password = await _extract_credentials(request)
    user = db.scalar(select(User).where(or_(User.email == username, User.username == username)))
    if not user or not verify_password(password, user.hashed_password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="邮箱或密码错误")
    access = create_access_token(subject=user.id, role=user.role)
    refresh = create_refresh_token(subject=user.id, role=user.role)
    return login_response(user, access, refresh)


@router.get("/me", response_model=UserProfile)
def me(user: User = Depends(get_current_user)):
    return user_profile(user)


@router.post("/refresh", response_model=LoginResponse)
def refresh(payload: RefreshRequest, db: Session = Depends(get_db)):
    try:
        data = decode_token(payload.refresh_token)
        if data.get("typ") != "refresh":
            raise ValueError("not refresh")
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="无效的 refresh_token")
    user = db.get(User, data.get("sub"))
    if user is None:
        raise HTTPException(status_code=401, detail="用户不存在")
    access = create_access_token(subject=user.id, role=user.role)
    refresh_tok = create_refresh_token(subject=user.id, role=user.role)
    return login_response(user, access, refresh_tok)


@router.post("/logout", status_code=204)
def logout(request: Request, user: User = Depends(get_current_user)):
    auth = request.headers.get("authorization") or ""
    token = auth.split(" ", 1)[-1] if auth else ""
    try:
        payload = decode_token(token)
        exp = payload.get("exp")
        ttl = 60
        if isinstance(exp, (int, float)):
            ttl = max(1, int(exp - datetime.now(timezone.utc).timestamp()))
        blacklist_jti(payload.get("jti"), ttl)
    except Exception:
        pass
    return None


@router.post("/ws-ticket", response_model=WsTicketOut)
def ws_ticket(task_id: str | None = None, user: User = Depends(get_current_user)):
    """一次性短票。可选绑定 task_id；连接成功后即失效（GETDEL）。"""
    if not redis_available():
        raise HTTPException(status_code=503, detail="Redis 不可用，请改用 WebSocket ?token=")
    ticket = issue_ws_ticket(user.id, task_id=task_id)
    return WsTicketOut(ticket=ticket, expires_in=TICKET_TTL_S)
