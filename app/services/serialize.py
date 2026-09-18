"""ORM → 前端冻结/待冻结契约。"""
from __future__ import annotations

from app.core.config import settings
from app.models.model_task import ModelTask
from app.models.project import Dataset, Project
from app.models.user import User
from app.schemas.auth import LoginResponse, UserProfile
from app.schemas.common import iso
from app.schemas.dataset import DatasetOut
from app.schemas.model import ModelResultOut, ModelTaskOut, TaskError, TaskProgress
from app.schemas.project import ProjectOut
from app.services.status import public_task_status


def user_profile(user: User) -> UserProfile:
    return UserProfile(
        id=user.id,
        username=user.username,
        display_name=user.display_name or user.username,
        role=user.role,
        created_at=iso(user.created_at),
        email=user.email,
    )


def login_response(user: User, access: str, refresh: str | None = None) -> LoginResponse:
    return LoginResponse(
        access_token=access,
        token_type="Bearer",
        expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        user=user_profile(user),
        refresh_token=refresh,
        role=user.role,
    )


def project_out(p: Project) -> ProjectOut:
    return ProjectOut(
        id=p.id,
        name=p.name,
        description=p.description or "",
        scenario_type=p.scenario_type,
        owner_id=p.user_id,
        created_at=iso(p.created_at) or "",
        updated_at=iso(p.updated_at) or iso(p.created_at) or "",
        is_demo=bool(p.is_demo),
    )


def dataset_out(ds: Dataset) -> DatasetOut:
    filename = ds.filename or ds.name or ""
    fmt = ds.format or _format_from_type(ds.file_type)
    return DatasetOut(
        id=ds.id,
        project_id=ds.project_id,
        filename=filename,
        format=fmt,
        size_bytes=ds.size_bytes or 0,
        row_count=ds.row_count or 0,
        status=ds.status,
        status_detail=ds.status_detail,
        source_crs=ds.source_crs,
        schema=ds.schema_json or [],
        column_guess=ds.column_guess or {},
        created_at=iso(ds.created_at) or "",
        name=ds.name or filename,
        file_type=ds.file_type,
    )


def _format_from_type(ft: str | None) -> str:
    t = (ft or "csv").lower()
    if t in ("xlsx", "xls"):
        return "excel"
    if t in ("geojson", "json"):
        return "geojson"
    if t in ("zip", "shp"):
        return "shapefile"
    return "csv"


def task_out(task: ModelTask) -> ModelTaskOut:
    progress = None
    detail = task.progress_detail or {}
    if task.status in ("RUNNING", "SUCCESS") or detail:
        hp = task.hyperparams or {}
        progress = TaskProgress(
            epoch=int(detail.get("epoch") or 0),
            total_epochs=int(detail.get("total_epochs") or hp.get("max_epochs") or hp.get("max_epoch") or 0),
            train_loss=detail.get("train_loss"),
            val_loss=detail.get("val_loss"),
            elapsed_s=float(detail.get("elapsed_s") or 0),
            eta_s=detail.get("eta_s"),
        )
    status = public_task_status(task.status)
    cancelled = (
        task.status == "CANCELLED"
        or task.error_code == "TASK_CANCELLED"
        or (task.error is not None and "取消" in task.error)
    )
    err = None
    if cancelled:
        err = TaskError(code="TASK_CANCELLED", message=task.error or "任务已取消", retryable=False)
        status = "FAILED"
    elif task.error:
        err = TaskError(
            code=task.error_code or "TRAIN_FAILED",
            message=task.error,
            retryable=bool(task.error_retryable) if task.error_retryable is not None else False,
        )
    return ModelTaskOut(
        id=task.id,
        task_id=task.id,
        project_id=task.project_id,
        dataset_id=task.dataset_id,
        model_type=task.model_type,
        y_column=task.y_column,
        x_columns=list(task.x_columns or []),
        spatial_columns=list(task.spatial_columns or []),
        temporal_column=task.temporal_column,
        hyperparams=task.hyperparams or {},
        status=status,
        progress=progress,
        error=err,
        created_at=iso(task.created_at) or "",
        started_at=iso(task.started_at),
        finished_at=iso(task.finished_at),
    )


def result_out(task: ModelTask) -> ModelResultOut:
    r = task.result
    summary = r.coefficients_summary or {}
    coef_list = summary.get("variables") or summary.get("list") or []
    return ModelResultOut(
        task_id=task.id,
        r2=r.r2, rmse=r.rmse, mae=r.mae, aicc=r.aicc,
        coefficients_summary=coef_list,
        residual_summary=r.residual_summary or {},
        sample_count=r.sample_count or 0,
        loss_history=r.loss_history or [],
        status=task.status,
        model_type=task.model_type,
        metrics={"r2": r.r2, "rmse": r.rmse, "mae": r.mae, "aicc": r.aicc},
        beta_ols=summary.get("beta_ols") or {},
        coefficients_key=summary.get("coefficients_key"),
    )
