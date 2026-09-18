"""建模：训练 / 任务列表 / 状态 / 结果 / 对比 / 系数 / 取消 / 日志。"""
from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.database import get_db
from app.core.deps import get_current_user
from app.models.model_task import ModelTask
from app.models.project import Dataset, Project
from app.models.user import User
from app.schemas.common import Paginated, page_args
from app.schemas.model import (
    BaselineComparisonOut, BaselineEntry, CancelOut, CoefficientPage, CoefficientPoint,
    GeoJsonPoint, ModelResultOut, ModelTaskOut, TaskLogsOut, TrainRequest,
)
from app.services.jobs import BrokerUnavailable, enqueue
from app.services.serialize import result_out, task_out
from app.services.status import public_task_status
from app.services.stats import best_by_metric
from app.services.storage import storage
from app.tasks.progress import mark_cancelled, read_logs
from app.tasks.training import run_train, train_task

router = APIRouter(prefix="/models", tags=["models"])


def _check_owner(db: Session, project_id: str, user: User) -> Project:
    proj = db.get(Project, project_id)
    if not proj or proj.user_id != user.id:
        raise HTTPException(status_code=404, detail="项目不存在")
    return proj


def _get_task(db: Session, task_id: str, user: User) -> ModelTask:
    task = db.get(ModelTask, task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    _check_owner(db, task.project_id, user)
    return task


@router.post("/train", response_model=ModelTaskOut, status_code=202)
def train(req: TrainRequest, db: Session = Depends(get_db),
          user: User = Depends(get_current_user)):
    _check_owner(db, req.project_id, user)
    ds = db.get(Dataset, req.dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail="数据集不存在")
    csv_key = f"datasets/{req.dataset_id}/cleaned.csv"
    if not storage.exists(csv_key):
        raise HTTPException(status_code=409, detail="请先完成数据预处理（缺少 cleaned.csv）")

    hp = req.hyperparams.normalized()
    if req.test_ratio is not None:
        hp["test_ratio"] = req.test_ratio
    if req.valid_ratio is not None:
        hp["valid_ratio"] = req.valid_ratio

    task = ModelTask(
        project_id=req.project_id, dataset_id=req.dataset_id,
        model_type=req.model_type, x_columns=req.x_columns, y_column=req.y_column,
        spatial_columns=req.spatial_columns, temporal_column=req.temporal_column,
        hyperparams=hp, status="PENDING",
    )
    db.add(task)
    db.commit()
    db.refresh(task)

    try:
        celery_id, _inline = enqueue(
            train_task.delay, task.id, csv_key,
            fallback=lambda tid, key: run_train(tid, key),
        )
    except BrokerUnavailable as exc:
        db.delete(task)
        db.commit()
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if celery_id:
        task.celery_task_id = celery_id
        db.commit()
        db.refresh(task)
    else:
        db.refresh(task)
    return task_out(task)


@router.get("/tasks", response_model=Paginated[ModelTaskOut])
def list_tasks(project_id: str = Query(..., description="必填"),
               page: int = Query(1, ge=1), page_size: int = Query(50, ge=1, le=200),
               db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    _check_owner(db, project_id, user)
    page, page_size, offset = page_args(page, page_size)
    q = select(ModelTask).where(ModelTask.project_id == project_id)
    total = db.scalar(select(func.count()).select_from(q.subquery())) or 0
    items = db.scalars(q.order_by(ModelTask.created_at.desc()).offset(offset).limit(page_size)).all()
    return Paginated(items=[task_out(t) for t in items], total=total, page=page, page_size=page_size)


@router.get("/tasks/{task_id}/status", response_model=ModelTaskOut)
def status(task_id: str, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    return task_out(_get_task(db, task_id, user))


@router.get("/tasks/{task_id}/result", response_model=ModelResultOut)
def result(task_id: str, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    task = _get_task(db, task_id, user)
    if task.status != "SUCCESS" or task.result is None:
        raise HTTPException(status_code=409, detail=f"任务尚未完成（当前 {task.status}）")
    return result_out(task)


@router.get("/tasks/{task_id}/compare", response_model=BaselineComparisonOut)
def compare(task_id: str, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    task = _get_task(db, task_id, user)
    if task.result is None:
        raise HTTPException(status_code=409, detail="任务尚未完成")
    target = task.model_type
    main = task.result
    target_coefs = (main.coefficients_summary or {}).get("variables") or []
    entries = [
        BaselineEntry(
            model=target, is_target=True,
            r2=main.r2, rmse=main.rmse, mae=main.mae, aicc=main.aicc,
            coefficients_summary=target_coefs,
        ).model_dump()
    ]
    for b in task.baselines:
        entries.append(BaselineEntry(
            model=b.method, is_target=False,
            r2=b.r2, rmse=b.rmse, mae=b.mae, aicc=b.aicc,
            coefficients_summary=b.coefficients_summary,
        ).model_dump())
    return BaselineComparisonOut(
        task_id=task_id,
        entries=entries,
        best_by_metric=best_by_metric(entries, target),
        model_type=target,
        main={"r2": main.r2, "rmse": main.rmse, "mae": main.mae, "aicc": main.aicc},
        baselines=[{"method": b.method, "r2": b.r2, "rmse": b.rmse, "mae": b.mae} for b in task.baselines],
    )


@router.get("/tasks/{task_id}/coefficients", response_model=CoefficientPage)
def coefficients(
    task_id: str,
    bbox: str | None = Query(None, description="minLon,minLat,maxLon,maxLat（默认按 WGS84 解释）"),
    vars: str | None = Query(None, description="逗号分隔变量名"),
    time: str | None = None,
    time_start: float | None = None,
    time_end: float | None = None,
    cursor: str | None = None,
    limit: int = Query(0, ge=0, le=50000, description="0 表示整包"),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """默认整包；可选 bbox / cursor / limit / vars。坐标为 WGS84（前端 toRenderCRS）。"""
    task = _get_task(db, task_id, user)
    if task.result is None:
        raise HTTPException(status_code=404, detail="结果不存在")
    key = (task.result.coefficients_summary or {}).get("coefficients_key")
    if not key:
        raise HTTPException(status_code=404, detail="系数文件不存在")
    raw = json.loads(storage.get_bytes(key).decode("utf-8"))
    pts = raw.get("points") or []
    columns = raw.get("columns") or []
    var_filter = [v.strip() for v in vars.split(",")] if vars else None

    bbox_t = None
    if bbox:
        try:
            a, b, c, d = (float(x) for x in bbox.split(","))
            bbox_t = (a, b, c, d)
        except ValueError:
            raise HTTPException(status_code=400, detail="bbox 格式应为 minLon,minLat,maxLon,maxLat")

    filtered = []
    for i, p in enumerate(pts):
        lon, lat = float(p["lon"]), float(p["lat"])
        if bbox_t and not (bbox_t[0] <= lon <= bbox_t[2] and bbox_t[1] <= lat <= bbox_t[3]):
            continue
        if time is not None and p.get("time") is not None and str(p.get("time")) != str(time):
            continue
        coef = dict(p.get("coef") or {})
        if var_filter:
            coef = {k: v for k, v in coef.items() if k in var_filter}
        filtered.append((i, p, lon, lat, coef))

    start = int(cursor or 0)
    cap = limit if limit and limit > 0 else len(filtered)
    page = filtered[start:start + cap]
    next_cursor = str(start + cap) if start + cap < len(filtered) else None

    points = [
        CoefficientPoint(
            feature_id=str(i),
            geom=GeoJsonPoint(type="Point", coordinates=[round(lon, 6), round(lat, 6)]),
            coefficients=coef,
            residual=p.get("residual"),
            observed=p.get("observed"),
            predicted=p.get("predicted"),
            time=p.get("time"),
        )
        for i, p, lon, lat, coef in page
    ]
    return CoefficientPage(
        points=points,
        next_cursor=next_cursor,
        total=len(filtered),
        crs=settings.VECTOR_OUTPUT_CRS,
        truncated=bool(next_cursor),
        columns=columns,
        temporal=raw.get("temporal"),
        times=raw.get("times"),
    )


@router.post("/tasks/{task_id}/cancel", response_model=CancelOut)
def cancel(task_id: str, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    task = _get_task(db, task_id, user)
    if task.status in ("SUCCESS", "FAILED", "CANCELLED"):
        return CancelOut(
            task_id=task.id, status=public_task_status(task.status), revoked=False,
        )
    mark_cancelled(task.id)
    revoked = False
    if task.celery_task_id:
        try:
            from app.tasks.celery_app import celery_app
            celery_app.control.revoke(task.celery_task_id, terminate=True)
            revoked = True
        except Exception:
            revoked = False
    task.status = "FAILED"
    task.error = "任务已取消"
    task.error_code = "TASK_CANCELLED"
    task.error_retryable = False
    db.commit()
    return CancelOut(task_id=task.id, status="FAILED", revoked=revoked)


@router.get("/tasks/{task_id}/logs", response_model=TaskLogsOut)
def logs(task_id: str, since: int | None = None,
         db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    _get_task(db, task_id, user)
    return TaskLogsOut(lines=read_logs(task_id, since=since))
