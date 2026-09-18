"""数据集：单次上传 / 分片 / 预签名 / 预览 / 预处理 / 列表。"""
from __future__ import annotations

import json
import uuid

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.database import get_db
from app.core.deps import get_current_user
from app.core.redis_client import get_sync_redis, redis_available
from app.models.project import Dataset, Project
from app.models.user import User
from app.schemas.common import Paginated, page_args
from app.schemas.dataset import (
    ChunkInitRequest, ChunkInitResponse, DatasetOut, PreprocessRequest,
    PresignResponse, PreviewResponse,
)
from app.services import ingest
from app.services.crs_transform import normalize_crs, offshore_ratio, wgs84_to_gcj02
from app.services.field_mapper import infer_field_schema, suggest_mapping
from app.services.jobs import BrokerUnavailable, enqueue
from app.services.serialize import dataset_out
from app.services.storage import storage
from app.tasks.ingest_task import preprocess_task, run_preprocess

router = APIRouter(prefix="/datasets", tags=["datasets"])

_ALLOWED = {"csv", "tsv", "xlsx", "xls", "geojson", "json"}
_MAX = settings.MAX_UPLOAD_MB * 1024 * 1024


def _owns_project(db: Session, project_id: str, user: User) -> Project:
    proj = db.get(Project, project_id)
    if not proj or proj.user_id != user.id:
        raise HTTPException(status_code=404, detail="项目不存在")
    return proj


def _owns_dataset(db: Session, dataset_id: str, user: User) -> Dataset:
    ds = db.get(Dataset, dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail="数据集不存在")
    _owns_project(db, ds.project_id, user)
    return ds


def _ext_of(name: str) -> str:
    return (name or "").rsplit(".", 1)[-1].lower() if "." in (name or "") else "csv"


def _format_of(ext: str) -> str:
    if ext in ("xlsx", "xls"):
        return "excel"
    if ext in ("geojson", "json"):
        return "geojson"
    if ext in ("zip", "shp"):
        return "shapefile"
    return "csv"


def _create_dataset_record(db: Session, project_id: str, filename: str, key: str,
                           ext: str, size: int, source_crs: str | None,
                           schema_json: list, guess: dict) -> Dataset:
    ds = Dataset(
        project_id=project_id,
        name=filename,
        filename=filename,
        storage_path=key,
        file_type=ext,
        format=_format_of(ext),
        source_crs=source_crs,
        size_bytes=size,
        status="uploaded",
        schema_json=schema_json,
        column_guess=guess,
    )
    db.add(ds)
    db.commit()
    db.refresh(ds)
    return ds


def _inspect_bytes(data: bytes, ext: str) -> tuple[list, dict, int]:
    df = ingest.read_tabular(data, ext)
    sug = suggest_mapping(df)
    return infer_field_schema(df), sug.as_column_guess(), len(df)


@router.post("/upload", response_model=DatasetOut, status_code=201)
async def upload(project_id: str = Form(...), file: UploadFile = File(...),
                 source_crs: str | None = Form(None),
                 db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    _owns_project(db, project_id, user)
    ext = _ext_of(file.filename or "dataset.csv")
    if ext not in _ALLOWED:
        raise HTTPException(status_code=400, detail=f"不支持的文件类型：.{ext}")
    data = await file.read()
    if len(data) > _MAX:
        raise HTTPException(status_code=413, detail=f"文件超过 {settings.MAX_UPLOAD_MB}MB，请走分片上传")

    ds_id = str(uuid.uuid4())
    key = f"datasets/{ds_id}/raw.{ext}"
    storage.put_bytes(key, data)
    try:
        schema_json, guess, n = _inspect_bytes(data, ext)
    except Exception:
        schema_json, guess, n = [], {}, 0

    ds = Dataset(
        id=ds_id, project_id=project_id,
        name=file.filename or "dataset", filename=file.filename or "dataset",
        storage_path=key, file_type=ext, format=_format_of(ext),
        source_crs=source_crs, size_bytes=len(data), row_count=n,
        status="uploaded", schema_json=schema_json, column_guess=guess,
    )
    db.add(ds)
    db.commit()
    db.refresh(ds)
    return dataset_out(ds)


@router.post("/upload/init", response_model=ChunkInitResponse)
def upload_init(payload: ChunkInitRequest, db: Session = Depends(get_db),
                user: User = Depends(get_current_user)):
    _owns_project(db, payload.project_id, user)
    if payload.size > 2 * 1024 ** 3:
        raise HTTPException(status_code=413, detail="单文件上限 2GB")
    upload_id = str(uuid.uuid4())
    meta = {
        "project_id": payload.project_id, "filename": payload.filename,
        "size": payload.size, "chunk_size": payload.chunk_size,
        "source_crs": payload.source_crs, "user_id": user.id,
        "chunks": [],
    }
    if redis_available():
        get_sync_redis().setex(f"upload:{upload_id}", 86400, json.dumps(meta))
    else:
        storage.put_bytes(f"uploads/{upload_id}/meta.json", json.dumps(meta).encode())
    return ChunkInitResponse(upload_id=upload_id, uploaded_chunks=[], chunk_size=payload.chunk_size)


def _load_upload_meta(upload_id: str) -> dict:
    if redis_available():
        raw = get_sync_redis().get(f"upload:{upload_id}")
        if not raw:
            raise HTTPException(status_code=404, detail="upload_id 不存在或已过期")
        return json.loads(raw)
    try:
        return json.loads(storage.get_bytes(f"uploads/{upload_id}/meta.json"))
    except Exception:
        raise HTTPException(status_code=404, detail="upload_id 不存在或已过期")


def _save_upload_meta(upload_id: str, meta: dict) -> None:
    if redis_available():
        get_sync_redis().setex(f"upload:{upload_id}", 86400, json.dumps(meta))
    else:
        storage.put_bytes(f"uploads/{upload_id}/meta.json", json.dumps(meta).encode())


@router.put("/upload/{upload_id}/chunk/{n}")
async def upload_chunk(upload_id: str, n: int, request: Request,
                       user: User = Depends(get_current_user)):
    meta = _load_upload_meta(upload_id)
    if meta.get("user_id") != user.id:
        raise HTTPException(status_code=404, detail="upload_id 不存在或已过期")
    data = await request.body()
    storage.put_bytes(f"uploads/{upload_id}/chunk_{n}", data)
    chunks = set(meta.get("chunks") or [])
    chunks.add(n)
    meta["chunks"] = sorted(chunks)
    _save_upload_meta(upload_id, meta)
    return {"ok": True, "index": n}


@router.post("/upload/{upload_id}/complete", response_model=DatasetOut)
def upload_complete(upload_id: str, db: Session = Depends(get_db),
                    user: User = Depends(get_current_user)):
    meta = _load_upload_meta(upload_id)
    if meta.get("user_id") != user.id:
        raise HTTPException(status_code=404, detail="upload_id 不存在或已过期")
    chunk_size = int(meta["chunk_size"])
    total = (int(meta["size"]) + chunk_size - 1) // chunk_size
    parts = []
    for i in range(total):
        parts.append(storage.get_bytes(f"uploads/{upload_id}/chunk_{i}"))
    data = b"".join(parts)
    ext = _ext_of(meta["filename"])
    ds_id = str(uuid.uuid4())
    key = f"datasets/{ds_id}/raw.{ext}"
    storage.put_bytes(key, data)
    try:
        schema_json, guess, n = _inspect_bytes(data, ext)
    except Exception:
        schema_json, guess, n = [], {}, 0
    ds = Dataset(
        id=ds_id, project_id=meta["project_id"],
        name=meta["filename"], filename=meta["filename"],
        storage_path=key, file_type=ext, format=_format_of(ext),
        source_crs=meta.get("source_crs"), size_bytes=len(data), row_count=n,
        status="uploaded", schema_json=schema_json, column_guess=guess,
    )
    db.add(ds)
    db.commit()
    db.refresh(ds)
    return dataset_out(ds)


@router.post("/upload/presign", response_model=PresignResponse)
def upload_presign(payload: ChunkInitRequest, db: Session = Depends(get_db),
                   user: User = Depends(get_current_user)):
    _owns_project(db, payload.project_id, user)
    upload_id = str(uuid.uuid4())
    ext = _ext_of(payload.filename)
    key = f"uploads/{upload_id}/raw.{ext}"
    url = storage.presign_put(key)
    if not url:
        raise HTTPException(status_code=501, detail="当前存储后端不支持预签名，请走分片或单次上传")
    meta = {**payload.model_dump(), "user_id": user.id, "object_key": key}
    if redis_available():
        get_sync_redis().setex(f"upload:{upload_id}", 3600, json.dumps(meta))
    return PresignResponse(upload_id=upload_id, url=url, object_key=key, expires_in=3600)


@router.get("/{dataset_id}/preview", response_model=PreviewResponse)
def preview(dataset_id: str, rows: int = Query(20, alias="rows"), n: int | None = None,
            db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    ds = _owns_dataset(db, dataset_id, user)
    limit = n or rows
    raw = storage.get_bytes(ds.storage_path)
    df = ingest.read_tabular(raw, ds.file_type)
    sug = suggest_mapping(df)

    spatial_sample: list[list[float]] = []
    points = []
    src = ds.source_crs or "WGS84"
    if sug.lon and sug.lat:
        sample = df[[sug.lon, sug.lat]].dropna()
        step = max(1, len(sample) // 800)
        for i, (_, r) in enumerate(sample.iterrows()):
            try:
                lon, lat = float(r[sug.lon]), float(r[sug.lat])
            except (ValueError, TypeError):
                continue
            spatial_sample.append([lon, lat])
            if i % step == 0:
                # preview `points` are GCJ-02 for Amap; WGS84/CGCS2000 are shifted once.
                if normalize_crs(src) == "GCJ02":
                    glon, glat = lon, lat
                else:
                    glon, glat = wgs84_to_gcj02(lon, lat)
                points.append({"lon": round(glon, 6), "lat": round(glat, 6)})
            if len(spatial_sample) >= 2000:
                break

    head = df.head(limit).where(df.head(limit).notna(), None)
    return PreviewResponse(
        dataset_id=ds.id,
        total_rows=len(df),
        n_rows_total=len(df),
        columns=list(df.columns),
        rows=head.to_dict(orient="records"),
        spatial_sample=spatial_sample[:800],
        offshore_ratio=round(offshore_ratio([(p[0], p[1]) for p in spatial_sample]), 4),
        mapping_suggestion=sug.to_dict(),
        points=points[:2000],
    )


@router.post("/{dataset_id}/preprocess", response_model=DatasetOut, status_code=202)
def preprocess(dataset_id: str, payload: PreprocessRequest,
               db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    ds = _owns_dataset(db, dataset_id, user)
    raw = storage.get_bytes(ds.storage_path)
    df = ingest.read_tabular(raw, ds.file_type)
    sug = suggest_mapping(df)
    lon = payload.lon_col() or sug.lon
    lat = payload.lat_col() or sug.lat
    if not lon or not lat:
        raise HTTPException(status_code=422, detail="未指定经纬度列")
    mapping = {
        "y": payload.y or sug.y,
        "x": payload.x if payload.x is not None else sug.x,
        "lon": lon, "lat": lat,
        "temporal": payload.time_col() or sug.temporal,
        "source_crs": payload.source_crs or ds.source_crs or "WGS84",
        "drop_invalid_rows": payload.drop_invalid_rows,
        "longitude_column": lon, "latitude_column": lat,
        "temporal_column": payload.time_col() or sug.temporal,
    }
    ds.status = "cleaning"
    ds.source_crs = mapping["source_crs"]
    ds.mapping = mapping
    db.commit()

    try:
        celery_id, inline = enqueue(
            preprocess_task.delay, dataset_id, mapping, fallback=run_preprocess,
        )
    except BrokerUnavailable as exc:
        ds.status = "uploaded"
        ds.status_detail = {"error": str(exc), "code": "QUEUE_UNAVAILABLE"}
        db.commit()
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    db.refresh(ds)
    if inline:
        db.refresh(ds)
    _ = celery_id
    return dataset_out(ds)


@router.get("", response_model=Paginated[DatasetOut])
def list_datasets(project_id: str = Query(..., description="必填"),
                  page: int = Query(1, ge=1), page_size: int = Query(50, ge=1, le=200),
                  db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    _owns_project(db, project_id, user)
    page, page_size, offset = page_args(page, page_size)
    q = select(Dataset).where(Dataset.project_id == project_id)
    total = db.scalar(select(func.count()).select_from(q.subquery())) or 0
    items = db.scalars(q.order_by(Dataset.created_at.desc()).offset(offset).limit(page_size)).all()
    return Paginated(items=[dataset_out(d) for d in items], total=total, page=page, page_size=page_size)
