"""训练进度广播：worker publish，FastAPI WebSocket subscribe。"""
from __future__ import annotations

import json
import time
from typing import Any

import redis

from app.core.config import settings
from app.core.redis_client import get_sync_redis


def channel(task_id: str) -> str:
    return f"train:progress:{task_id}"


def logs_key(task_id: str) -> str:
    return f"train:logs:{task_id}"


def cancel_key(task_id: str) -> str:
    return f"train:cancel:{task_id}"


def started_key(task_id: str) -> str:
    return f"train:started:{task_id}"


def publish_progress(task_id: str, payload: dict) -> None:
    r = redis.from_url(settings.redis_uri)
    try:
        r.publish(channel(task_id), json.dumps(payload, ensure_ascii=False))
    finally:
        r.close()


def append_log(task_id: str, message: str, level: str = "info") -> None:
    line = {"at": int(time.time() * 1000), "level": level, "message": message}
    try:
        r = get_sync_redis()
        r.rpush(logs_key(task_id), json.dumps(line, ensure_ascii=False))
        r.expire(logs_key(task_id), 7 * 24 * 3600)
        publish_progress(task_id, {
            "type": "log", "task_id": task_id,
            "level": level, "message": message, "at": line["at"],
        })
    except Exception:
        pass


def read_logs(task_id: str, since: int | None = None) -> list[dict]:
    try:
        raw = get_sync_redis().lrange(logs_key(task_id), 0, -1)
    except Exception:
        return []
    out = []
    for item in raw:
        try:
            rec = json.loads(item)
        except Exception:
            continue
        if since and rec.get("at", 0) < since:
            continue
        out.append(rec)
    return out


def mark_cancelled(task_id: str) -> None:
    try:
        get_sync_redis().setex(cancel_key(task_id), 24 * 3600, "1")
    except Exception:
        pass


def is_cancelled(task_id: str) -> bool:
    try:
        return bool(get_sync_redis().get(cancel_key(task_id)))
    except Exception:
        return False


def mark_started(task_id: str) -> None:
    try:
        get_sync_redis().set(started_key(task_id), str(time.time()))
    except Exception:
        pass


def started_at(task_id: str) -> float | None:
    try:
        v = get_sync_redis().get(started_key(task_id))
        return float(v) if v else None
    except Exception:
        return None


def build_progress_message(
    task_id: str,
    status: str,
    epoch: int,
    total_epochs: int,
    train_loss: float | None,
    val_loss: float | None,
    elapsed_s: float,
    eta_s: float | None,
    coef_summary: list[dict] | None = None,
) -> dict[str, Any]:
    """对齐独立前端 TaskSocketMessage + 确认清单建议字段。"""
    msg: dict[str, Any] = {
        "type": "progress",
        "task_id": task_id,
        "status": status,
        "progress": {
            "epoch": int(epoch),
            "total_epochs": int(total_epochs),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "elapsed_s": round(float(elapsed_s), 2),
            "eta_s": None if eta_s is None else round(float(eta_s), 2),
        },
    }
    if coef_summary:
        msg["coef_summary"] = coef_summary
    return msg


def build_status_message(task_id: str, status: str, error: str | None = None,
                         code: str | None = None) -> dict:
    from app.services.status import public_task_status
    status = public_task_status(status)
    msg = {"type": "status", "task_id": task_id, "status": status}
    if error:
        err_code = code or (
            "TASK_CANCELLED" if "取消" in error else "TRAIN_FAILED"
        )
        msg["error"] = error
        msg["type"] = "error"
        msg["code"] = err_code
        msg["message"] = error
        if err_code == "TASK_CANCELLED":
            msg["retryable"] = False
    return msg
