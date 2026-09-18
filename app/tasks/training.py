"""
建模训练任务。
读取清洗后 CSV → engine.run_analysis（lite / 官方包切换点）→ Redis 进度 → 落库。
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from app.core.database import SessionLocal  # noqa: E402
from app.models.model_task import BaselineComparison, ModelResult, ModelTask  # noqa: E402
from app.services.engine_runner import run_analysis  # noqa: E402
from app.services.stats import global_collapsed, residual_summary, summarize_variable  # noqa: E402
from app.services.storage import storage  # noqa: E402
from app.tasks.celery_app import celery_app  # noqa: E402
from app.tasks.progress import (  # noqa: E402
    append_log, build_progress_message, build_status_message,
    is_cancelled, mark_started, publish_progress, started_at,
)
from engine.pipeline import FieldMapping  # noqa: E402


class TaskCancelled(Exception):
    pass


def _set_status(db, task: ModelTask, status: str, **fields):
    task.status = status
    for k, v in fields.items():
        setattr(task, k, v)
    db.commit()


def run_train(task_id: str, csv_key: str, celery_id: str | None = None) -> dict:
    db = SessionLocal()
    try:
        task = db.get(ModelTask, task_id)
        if task is None:
            return {"task_id": task_id, "error": "task not found"}

        _set_status(
            db, task, "RUNNING",
            celery_task_id=celery_id or task.celery_task_id,
            progress=0.0,
            started_at=datetime.now(timezone.utc),
            error=None, error_code=None, error_retryable=None,
        )
        mark_started(task_id)
        append_log(task_id, f"开始训练 {task.model_type}，读取 {csv_key}")
        publish_progress(task_id, build_status_message(task_id, "RUNNING"))

        local_csv = storage.local_path(csv_key)
        df = pd.read_csv(local_csv)
        append_log(task_id, f"数据加载完成, {len(df)} 条样本")

        mapping = FieldMapping(
            y=task.y_column, x=list(task.x_columns),
            spatial=list(task.spatial_columns), temporal=task.temporal_column,
        )
        hp = dict(task.hyperparams or {})
        max_epoch = int(hp.get("max_epoch") or hp.get("max_epochs") or 200)
        test_ratio = float(hp.get("test_ratio", 0.2))
        valid_ratio = float(hp.get("valid_ratio", 0.15))
        seed = int(hp.get("seed") or hp.get("random_seed") or 42)
        t0 = time.time()

        def cb(rec: dict):
            if is_cancelled(task_id):
                raise TaskCancelled("cancelled")
            epoch = int(rec["epoch"])
            elapsed = time.time() - t0
            eta = (elapsed / epoch) * (max_epoch - epoch) if epoch else None
            pct = min(epoch / max_epoch, 1.0) * 100.0
            detail = {
                "epoch": epoch,
                "total_epochs": max_epoch,
                "train_loss": rec.get("train_loss"),
                "val_loss": rec.get("val_loss"),
                "elapsed_s": elapsed,
                "eta_s": eta,
            }
            task.progress = pct
            task.progress_detail = detail
            db.commit()
            coef_summary = None
            # 每 10 epoch 给监控页一条系数带摘要（可选；没有则前端隐藏）
            if epoch == 1 or epoch % 10 == 0:
                coef_summary = None  # 训练中完整系数较贵，终态再给；此处留扩展点
            publish_progress(task_id, build_progress_message(
                task_id, "RUNNING", epoch, max_epoch,
                rec.get("train_loss"), rec.get("val_loss"), elapsed, eta, coef_summary,
            ))

        result = run_analysis(
            df, mapping, hyperparams=hp, progress_cb=cb,
            test_ratio=test_ratio, valid_ratio=valid_ratio, seed=seed,
        )

        if is_cancelled(task_id):
            raise TaskCancelled("cancelled")

        coef_key = f"results/{task_id}/coefficients.json"
        storage.put_bytes(
            coef_key,
            json.dumps(result["coefficients"], ensure_ascii=False).encode("utf-8"),
            content_type="application/json",
        )

        points = result["coefficients"]["points"]
        cols = result["coefficients"]["columns"]
        summaries = []
        for col in cols:
            vals = np.array([p["coef"].get(col, 0.0) for p in points], float)
            summaries.append(summarize_variable(col, vals))

        resid = np.array([p.get("residual", 0.0) for p in points], float)
        lons = np.array([p["lon"] for p in points], float)
        lats = np.array([p["lat"] for p in points], float)
        rsum = residual_summary(resid, lons, lats)

        hist = result.get("history") or []
        if hist and isinstance(hist[0], dict):
            loss_history = [[h["epoch"], h["train_loss"], h["val_loss"]] for h in hist]
        else:
            loss_history = []

        res = ModelResult(
            task_id=task_id,
            r2=result["metrics"]["r2"], rmse=result["metrics"]["rmse"],
            mae=result["metrics"]["mae"], aicc=result["metrics"]["aicc"],
            model_weight_path=None,
            coefficients_summary={
                "columns": cols,
                "beta_ols": result["beta_ols"],
                "coefficients_key": coef_key,
                "variables": summaries,
                "history_len": len(hist),
            },
            residuals_path=coef_key,
            sample_count=len(points),
            loss_history=loss_history,
            residual_summary=rsum,
        )
        db.add(res)

        for b in result["baselines"]:
            method = b["method"]
            coefs = None if method in ("OLS", "RandomForest") else (
                global_collapsed(summaries) if method == "OLS" else summaries
            )
            if method in ("OLS", "RandomForest"):
                coefs = None if method == "RandomForest" else global_collapsed(summaries)
            db.add(BaselineComparison(
                task_id=task_id, method=method,
                r2=b.get("r2"), rmse=b.get("rmse"), mae=b.get("mae"), aicc=b.get("aicc"),
                coefficients_summary=coefs,
            ))

        elapsed = time.time() - (started_at(task_id) or t0)
        _set_status(db, task, "SUCCESS", progress=100.0,
                    finished_at=datetime.now(timezone.utc),
                    progress_detail={
                        "epoch": max_epoch, "total_epochs": max_epoch,
                        "train_loss": (hist[-1]["train_loss"] if hist else None),
                        "val_loss": (hist[-1]["val_loss"] if hist else None),
                        "elapsed_s": elapsed, "eta_s": 0,
                    })
        append_log(task_id, f"训练完成 R²={result['metrics']['r2']}")
        publish_progress(task_id, build_progress_message(
            task_id, "SUCCESS", max_epoch, max_epoch,
            (hist[-1]["train_loss"] if hist else None),
            (hist[-1]["val_loss"] if hist else None),
            elapsed, 0, summaries,
        ))
        publish_progress(task_id, build_status_message(task_id, "SUCCESS"))
        return {"task_id": task_id, "metrics": result["metrics"]}

    except TaskCancelled:
        task = db.get(ModelTask, task_id)
        if task is not None:
            _set_status(db, task, "FAILED", error="任务已取消",
                        error_code="TASK_CANCELLED", error_retryable=False,
                        finished_at=datetime.now(timezone.utc))
        append_log(task_id, "任务已取消", "warn")
        publish_progress(
            task_id,
            build_status_message(task_id, "FAILED", "任务已取消", code="TASK_CANCELLED"),
        )
        return {"task_id": task_id, "status": "FAILED", "error_code": "TASK_CANCELLED"}

    except Exception as exc:  # noqa: BLE001
        db.rollback()
        task = db.get(ModelTask, task_id)
        msg = str(exc)[:480]
        retryable = "timeout" in msg.lower() or "memory" in msg.lower()
        if task is not None:
            _set_status(db, task, "FAILED", error=msg,
                        error_code="TRAIN_FAILED", error_retryable=retryable,
                        finished_at=datetime.now(timezone.utc))
        append_log(task_id, msg, "error")
        publish_progress(task_id, build_status_message(task_id, "FAILED", msg))
        raise
    finally:
        db.close()


@celery_app.task(bind=True, name="app.tasks.training.train_task", queue="gpu_queue")
def train_task(self, task_id: str, csv_key: str):
    return run_train(task_id, csv_key, celery_id=self.request.id)
