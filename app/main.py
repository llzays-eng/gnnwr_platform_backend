"""
FastAPI 应用入口（空间服务层）。
挂载 /api/v1 与 /ws/models/tasks/{task_id}；CORS；启动时尝试建表。
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse

from app.api.v1 import api_router
from app.core.config import settings
from app.services.jobs import BrokerUnavailable
from app.ws.progress import router as ws_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    from app.core.startup import assert_secure_startup
    assert_secure_startup()
    try:
        from app.core.database import init_db
        init_db()
    except Exception as exc:  # noqa: BLE001
        print(f"[启动] 建表跳过（数据库暂不可用）：{exc}")
    yield


tags_metadata = [
    {"name": "health", "description": "存活与依赖探针"},
    {"name": "auth", "description": "注册 / 登录 / 当前用户 / refresh / logout"},
    {"name": "projects", "description": "分析项目"},
    {"name": "datasets", "description": "上传、预览、预处理"},
    {"name": "models", "description": "训练任务与结果"},
    {"name": "spatial", "description": "视野切片与曲面 WMS"},
    {"name": "reports", "description": "报告导出"},
]

app = FastAPI(
    title=settings.APP_NAME,
    version="1.1.0",
    description=(
        "GNNWR/GTNNWR 时空智能分析云平台 · 后端服务。"
        "WebSocket 进度通道：`/ws/models/tasks/{task_id}?token=`；"
        "曲面响应含 `tile_crs`（默认 GCJ02，由本服务 XYZ/WMS 如实生成）。"
    ),
    lifespan=lifespan,
    openapi_tags=tags_metadata,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    rid = request.headers.get("x-request-id") or uuid4().hex
    response = await call_next(request)
    response.headers["X-Request-Id"] = rid
    return response


app.include_router(api_router, prefix=settings.API_V1_PREFIX)
app.include_router(ws_router)


@app.exception_handler(BrokerUnavailable)
async def broker_unavailable_handler(_request: Request, exc: BrokerUnavailable):
    return JSONResponse(status_code=503, content={"detail": str(exc)})


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
        tags=tags_metadata,
    )
    schema.setdefault("paths", {})
    schema["paths"]["/ws/models/tasks/{task_id}"] = {
        "get": {
            "tags": ["models"],
            "summary": "WebSocket 训练进度（浏览器升级）",
            "description": (
                "冻结路径。连接：`ws(s)://{host}/ws/models/tasks/{task_id}?token=<JWT>` "
                "或一次性 `?ticket=`（`POST /api/v1/auth/ws-ticket`，连接成功即 GETDEL 失效）。"
                "消息：`{type:progress,status,progress:{epoch,total_epochs,train_loss,val_loss,elapsed_s,eta_s},coef_summary?}`。"
                "status 仅为 PENDING|RUNNING|SUCCESS|FAILED（取消为 FAILED + TASK_CANCELLED）。"
                "兼容旧路径 `/api/v1/models/tasks/{task_id}/ws`。"
            ),
            "parameters": [
                {
                    "name": "task_id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string", "format": "uuid"},
                },
                {
                    "name": "token",
                    "in": "query",
                    "required": False,
                    "schema": {"type": "string"},
                    "description": "JWT access token（浏览器 WebSocket 无法自定义 Header）",
                },
                {
                    "name": "ticket",
                    "in": "query",
                    "required": False,
                    "schema": {"type": "string"},
                },
            ],
            "responses": {"101": {"description": "Switching Protocols"}},
        }
    }
    login = schema["paths"].get("/api/v1/auth/login", {}).get("post", {})
    login["requestBody"] = {
        "required": True,
        "content": {
            "application/json": {
                "schema": {
                    "type": "object",
                    "required": ["username", "password"],
                    "properties": {
                        "username": {"type": "string", "description": "邮箱或用户名"},
                        "password": {"type": "string"},
                    },
                    "example": {"username": "demo@example.com", "password": "secret12"},
                }
            },
            "application/x-www-form-urlencoded": {
                "schema": {
                    "type": "object",
                    "required": ["username", "password"],
                    "properties": {
                        "username": {"type": "string"},
                        "password": {"type": "string"},
                    },
                }
            },
        },
    }
    schema["paths"]["/api/v1/auth/login"]["post"] = login
    app.openapi_schema = schema
    return schema


app.openapi = custom_openapi


@app.get("/", tags=["health"])
def root():
    return {
        "app": settings.APP_NAME,
        "docs": "/docs",
        "api": settings.API_V1_PREFIX,
        "ws_progress": "/ws/models/tasks/{task_id}",
        "tile_crs": settings.SURFACE_TILE_CRS,
    }


@app.get("/health", tags=["health"])
def health():
    status = {"api": "ok"}
    try:
        from sqlalchemy import text
        from app.core.database import engine
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        status["postgres"] = "ok"
    except Exception:
        status["postgres"] = "unavailable"
    try:
        import redis
        redis.from_url(settings.redis_uri).ping()
        status["redis"] = "ok"
    except Exception:
        status["redis"] = "unavailable"
    from app.services.storage import storage
    status["storage"] = storage.backend
    status["engine"] = settings.ENGINE_BACKEND
    return status
