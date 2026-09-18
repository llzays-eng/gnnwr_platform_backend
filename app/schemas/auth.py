"""认证契约：对齐独立前端 LoginResponse / UserProfile，同时兼容 OAuth2 表单登录。"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, EmailStr, Field


class UserCreate(BaseModel):
    email: EmailStr
    username: str
    password: str = Field(min_length=6)
    display_name: str | None = None


class UserProfile(BaseModel):
    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "examples": [
                {
                    "id": "11111111-1111-1111-1111-111111111111",
                    "username": "demo",
                    "display_name": "Demo",
                    "role": "user",
                    "created_at": "2026-09-15T00:00:00",
                }
            ]
        },
    )

    id: str
    username: str
    display_name: str
    role: str
    created_at: str | None = None
    email: str | None = None


class LoginJSON(BaseModel):
    """独立前端 POST JSON：{username, password}，username 可为邮箱。"""

    username: str
    password: str


class LoginResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                    "token_type": "Bearer",
                    "expires_in": 86400,
                    "user": {
                        "id": "11111111-1111-1111-1111-111111111111",
                        "username": "demo",
                        "display_name": "Demo",
                        "role": "user",
                        "created_at": "2026-09-15T00:00:00",
                    },
                    "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                }
            ]
        }
    )

    access_token: str
    token_type: str = "Bearer"
    expires_in: int
    user: UserProfile
    refresh_token: str | None = None
    role: str | None = None  # 兼容旧客户端


class Token(LoginResponse):
    """OAuth2 文档别名。"""


class RefreshRequest(BaseModel):
    refresh_token: str


class WsTicketOut(BaseModel):
    ticket: str
    expires_in: int = 120
