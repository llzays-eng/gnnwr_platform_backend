"""数据预处理异步任务：清洗 + 坐标转换 + PostGIS 入库。"""
from __future__ import annotations

from app.core.database import SessionLocal
from app.models.project import Dataset
from app.services import ingest
from app.services.storage import storage
from app.tasks.celery_app import celery_app


def run_preprocess(dataset_id: str, mapping: dict) -> dict:
    db = SessionLocal()
    try:
        ds = db.get(Dataset, dataset_id)
        if ds is None:
            return {"dataset_id": dataset_id, "error": "dataset not found"}

        ds.status = "cleaning"
        db.commit()

        raw = storage.get_bytes(ds.storage_path)
        df = ingest.read_tabular(raw, ds.file_type)

        lon = mapping.get("lon") or mapping.get("longitude_column")
        lat = mapping.get("lat") or mapping.get("latitude_column")
        temporal = mapping.get("temporal") or mapping.get("temporal_column")
        y = mapping.get("y")
        x = mapping.get("x") or []
        source_crs = mapping.get("source_crs") or ds.source_crs or "WGS84"

        cleaned, report = ingest.clean(
            df, lon=lon, lat=lat, y=y, x=x, temporal=temporal,
            drop_invalid_rows=mapping.get("drop_invalid_rows", True),
        )

        cleaned_key = f"datasets/{dataset_id}/cleaned.csv"
        storage.put_bytes(
            cleaned_key, cleaned.to_csv(index=False).encode("utf-8"), content_type="text/csv"
        )
        ds.status = "cleaned"
        db.commit()

        n = ingest.ingest_to_postgis(
            db, dataset_id, cleaned, lon=lon, lat=lat, temporal=temporal, source_crs=source_crs,
        )
        ds.row_count = n
        ds.status = "ingested"
        ds.source_crs = source_crs
        ds.mapping = {**mapping, "cleaned_key": cleaned_key, "report": report}
        ds.status_detail = None
        db.commit()
        return {"dataset_id": dataset_id, "ingested": n, "cleaned_key": cleaned_key, "report": report}

    except Exception as exc:  # noqa: BLE001
        db.rollback()
        ds = db.get(Dataset, dataset_id)
        if ds is not None:
            ds.status = "failed"
            ds.status_detail = {
                "code": "PREPROCESS_FAILED",
                "message": str(exc)[:400],
                "next_actions": [
                    {"kind": "remap_columns"},
                    {"kind": "change_crs", "suggested": "WGS84"},
                    {"kind": "reupload"},
                ],
            }
            db.commit()
        raise
    finally:
        db.close()


@celery_app.task(bind=True, name="app.tasks.ingest_task.preprocess_task", queue="cpu_queue")
def preprocess_task(self, dataset_id: str, mapping: dict):
    return run_preprocess(dataset_id, mapping)
