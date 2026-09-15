"""
数据接入：读文件 → 识别/转换坐标系 → 缺失/异常清洗 → 写入 spatial_features。
"""
from __future__ import annotations

import json
from io import BytesIO

import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.services.crs_transform import transform


def read_tabular(data: bytes, file_type: str) -> pd.DataFrame:
    ft = file_type.lower().lstrip(".")
    if ft in ("csv", "tsv"):
        sep = "\t" if ft == "tsv" else ","
        return pd.read_csv(BytesIO(data), sep=sep)
    if ft in ("xlsx", "xls", "excel"):
        return pd.read_excel(BytesIO(data))
    if ft in ("geojson", "json"):
        gj = json.loads(data.decode("utf-8"))
        rows = []
        for feat in gj.get("features", []):
            geom = feat.get("geometry") or {}
            props = dict(feat.get("properties") or {})
            if geom.get("type") == "Point":
                props["lon"], props["lat"] = geom["coordinates"][:2]
            rows.append(props)
        return pd.DataFrame(rows)
    if ft in ("zip", "shp", "shapefile"):
        raise ValueError("Shapefile 请先转为 CSV/GeoJSON 再上传（zip 解析将在后续版本提供）")
    raise ValueError(f"暂不支持的文件类型：{file_type}")


def clean(df: pd.DataFrame, lon: str, lat: str,
          y: str | None = None, x: list[str] | None = None,
          temporal: str | None = None, drop_invalid_rows: bool = True
          ) -> tuple[pd.DataFrame, dict]:
    involved = [c for c in ([lon, lat] + ([y] if y else []) + (x or []) +
                            ([temporal] if temporal else [])) if c in df.columns]
    before = len(df)
    work = df.copy()
    for c in involved:
        if c != temporal:
            work[c] = pd.to_numeric(work[c], errors="coerce")

    dropped_na = int(work[involved].isna().any(axis=1).sum()) if involved else 0
    if drop_invalid_rows and involved:
        work = work.dropna(subset=involved)

    outliers = 0
    for c in ([y] if y else []) + (x or []):
        if c in work.columns and pd.api.types.is_numeric_dtype(work[c]):
            q1, q3 = work[c].quantile(0.25), work[c].quantile(0.75)
            iqr = q3 - q1
            lo, hi = q1 - 3 * iqr, q3 + 3 * iqr
            mask = (work[c] < lo) | (work[c] > hi)
            outliers += int(mask.sum())
            work = work[~mask]

    report = {
        "rows_before": before,
        "rows_after": len(work),
        "dropped_missing": dropped_na,
        "dropped_outliers": outliers,
    }
    return work.reset_index(drop=True), report


def ingest_to_postgis(db: Session, dataset_id: str, df: pd.DataFrame, lon: str, lat: str,
                      temporal: str | None, source_crs: str = "WGS84") -> int:
    """坐标先统一到 WGS84（4326）再入库。"""
    from sqlalchemy import text as sql_text

    db.execute(sql_text("DELETE FROM spatial_features WHERE dataset_id = :did"), {"did": dataset_id})

    prop_cols = [c for c in df.columns if c not in (lon, lat, temporal)]
    inserted = 0
    stmt = text("""
        INSERT INTO spatial_features (dataset_id, geom, observed_time, properties)
        VALUES (:did, ST_SetSRID(ST_MakePoint(:lon, :lat), 4326), :obs, CAST(:props AS jsonb))
    """)
    for _, r in df.iterrows():
        x, y = transform(float(r[lon]), float(r[lat]), source_crs, "WGS84")
        props = {c: (None if pd.isna(r[c]) else _py(r[c])) for c in prop_cols}
        obs = _to_ts(r[temporal]) if temporal else None
        db.execute(stmt, {
            "did": dataset_id, "lon": x, "lat": y, "obs": obs,
            "props": json.dumps(props, ensure_ascii=False, default=str),
        })
        inserted += 1
    db.commit()
    return inserted


def _py(v):
    try:
        return v.item()
    except AttributeError:
        if hasattr(v, "isoformat"):
            return v.isoformat()
        return v


def _to_ts(v):
    try:
        return pd.to_datetime(v).to_pydatetime()
    except Exception:
        return None
