"""
Celery 应用。
- broker / result backend 均用 Redis
- cpu_queue：清洗入库；gpu_queue：训练（可在 CPU 上回落跑 gnnwr_lite）
"""
from __future__ import annotations

from celery import Celery
from kombu import Queue

from app.core.config import settings

celery_app = Celery(
    "gnnwr",
    broker=settings.redis_uri,
    backend=settings.redis_uri,
    include=["app.tasks.training", "app.tasks.ingest_task"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="Asia/Shanghai",
    enable_utc=True,
    task_track_started=True,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    task_queues=(Queue("cpu_queue"), Queue("gpu_queue")),
    task_default_queue="cpu_queue",
    task_routes={
        "app.tasks.training.*": {"queue": "gpu_queue"},
        "app.tasks.ingest_task.*": {"queue": "cpu_queue"},
    },
)
