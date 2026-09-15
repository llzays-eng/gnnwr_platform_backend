"""
FastAPI 应用入口（空间服务层）。
挂载 /api/v1 与 /ws/models/tasks/{task_id}；CORS；启动时尝试建表。
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1 import api_router
from app.core.config import settings
from app.ws.progress import router as ws_router


@asynccontextmanager
async def lifespan(app: FastAPI):
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
