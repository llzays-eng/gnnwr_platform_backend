"""WebSocket 训练进度：路径 /ws/models/tasks/{task_id}，query token 或 ticket 鉴权。"""
from __future__ import annotations

import asyncio
import json
import logging

import jwt
import redis.asyncio as aioredis
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.core.config import settings
from app.core.database import SessionLocal
from app.core.redis_client import is_jti_denied
from app.core.security import decode_token
from app.models.model_task import ModelTask
from app.models.project import Project
from app.services.status import public_task_status
from app.services.tickets import consume_ws_ticket
from app.tasks.progress import build_progress_message, build_status_message, channel

log = logging.getLogger(__name__)
router = APIRouter()


async def _resolve_user_id(websocket: WebSocket, task_id: str) -> str | None:
    token = websocket.query_params.get("token")
    ticket = websocket.query_params.get("ticket")
    if ticket:
        try:
            uid = consume_ws_ticket(ticket, task_id=task_id)
            if uid:
                return uid
        except Exception:
            pass
    if token:
        try:
            payload = decode_token(token)
            if payload.get("typ") in (None, "access") and not is_jti_denied(payload.get("jti")):
                return payload.get("sub")
        except jwt.PyJWTError:
            return None
    return None


def _can_access(task_id: str, user_id: str | None) -> ModelTask | None:
    db = SessionLocal()
    try:
        task = db.get(ModelTask, task_id)
        if not task:
            return None
        if user_id:
            proj = db.get(Project, task.project_id)
            if not proj or proj.user_id != user_id:
                return None
        return task
    finally:
        db.close()


@router.websocket("/ws/models/tasks/{task_id}")
async def ws_progress(websocket: WebSocket, task_id: str):
    user_id = await _resolve_user_id(websocket, task_id)
    if settings.ws_must_auth and not user_id:
        await websocket.close(code=4401, reason="unauthorized")
        return

    task = _can_access(task_id, user_id)
    if task is None:
        await websocket.close(code=4404, reason="not found")
        return

    await websocket.accept()

    async def send_snapshot():
        db = SessionLocal()
        try:
            t = db.get(ModelTask, task_id)
            if not t:
                return
            detail = t.progress_detail or {}
            hp = t.hyperparams or {}
            total = int(detail.get("total_epochs") or hp.get("max_epochs") or hp.get("max_epoch") or 0)
            status = public_task_status(t.status)
            if t.status == "RUNNING" or detail:
                await websocket.send_json(build_progress_message(
                    task_id, status,
                    int(detail.get("epoch") or 0), total,
                    detail.get("train_loss"), detail.get("val_loss"),
                    float(detail.get("elapsed_s") or 0), detail.get("eta_s"),
                ))
            await websocket.send_json(build_status_message(
                task_id, status, t.error, code=t.error_code,
            ))
        finally:
            db.close()

    await send_snapshot()

    r = aioredis.from_url(settings.redis_uri)
    pubsub = r.pubsub()
    await pubsub.subscribe(channel(task_id))
    try:
        while True:
            try:
                msg = await asyncio.wait_for(
                    pubsub.get_message(ignore_subscribe_messages=True, timeout=10),
                    timeout=12,
                )
            except asyncio.TimeoutError:
                msg = None
            if msg and msg.get("type") == "message":
                data = msg["data"]
                if isinstance(data, bytes):
                    data = data.decode("utf-8")
                payload = json.loads(data)
                await websocket.send_json(payload)
                st = payload.get("status")
                if payload.get("type") == "status" and st in ("SUCCESS", "FAILED"):
                    break
                if st in ("SUCCESS", "FAILED") and payload.get("type") == "progress":
                    # 终态 progress 后再等 status；若只有 progress 也结束
                    if payload.get("progress", {}).get("eta_s") == 0:
                        break
            else:
                # 心跳，避免前端 20s 假死检测
                await websocket.send_json({"type": "pong"})
            try:
                incoming = await asyncio.wait_for(websocket.receive_text(), timeout=0.05)
                if incoming == "ping":
                    await websocket.send_json({"type": "pong"})
            except (asyncio.TimeoutError, WebSocketDisconnect):
                pass
    except WebSocketDisconnect:
        pass
    except Exception as exc:  # noqa: BLE001
        log.info("ws closed: %s", exc)
    finally:
        await pubsub.unsubscribe(channel(task_id))
        await r.close()


# 主仓旧路径兼容
@router.websocket("/api/v1/models/tasks/{task_id}/ws")
async def ws_progress_legacy(websocket: WebSocket, task_id: str):
    await ws_progress(websocket, task_id)
