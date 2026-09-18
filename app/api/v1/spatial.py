"""空间查询：视野增量 tiles + 曲面 WMS/XYZ（tile_crs 必填）。"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.database import get_db
from app.core.deps import get_current_user
from app.models.model_task import ModelTask
from app.models.project import Dataset, Project
from app.models.user import User
from app.schemas.spatial import SpatialFeatureOut, SurfaceLayerInfo, TilesResponse
from app.services.crs_transform import gcj02_to_wgs84, normalize_crs, wgs84_to_gcj02
from app.services.geoserver import geoserver
from app.services.signed_url import append_query, create_tile_token, verify_tile_token
from app.services.surface import (
    field_bbox_gcj02, legend_png, load_points_cached, render_bbox_png, render_xyz,
)

router = APIRouter(prefix="/spatial", tags=["spatial"])


def _parse_bbox(bbox: str) -> tuple[float, float, float, float]:
    try:
        a, b, c, d = (float(v) for v in bbox.split(","))
        return a, b, c, d
    except ValueError:
        raise HTTPException(status_code=400, detail="bbox 格式应为 minLon,minLat,maxLon,maxLat")


def _owns_dataset(db: Session, dataset_id: str, user: User) -> Dataset:
    ds = db.get(Dataset, dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail="数据集不存在")
    proj = db.get(Project, ds.project_id)
    if not proj or proj.user_id != user.id:
        raise HTTPException(status_code=404, detail="数据集不存在")
    return ds


def _owns_task(db: Session, task_id: str, user: User) -> ModelTask:
    task = db.get(ModelTask, task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    proj = db.get(Project, task.project_id)
    if not proj or proj.user_id != user.id:
        raise HTTPException(status_code=404, detail="任务不存在")
    return task


def _bbox_to_wgs84(min_lon: float, min_lat: float, max_lon: float, max_lat: float,
                   bbox_crs: str) -> tuple[float, float, float, float]:
    """bbox_crs=WGS84（默认，对齐前端 BBox）时原样查询；GCJ02 时四角反解再取包络。"""
    crs = normalize_crs(bbox_crs)
    if crs != "GCJ02":
        return min_lon, min_lat, max_lon, max_lat
    corners = [
        gcj02_to_wgs84(min_lon, min_lat),
        gcj02_to_wgs84(min_lon, max_lat),
        gcj02_to_wgs84(max_lon, min_lat),
        gcj02_to_wgs84(max_lon, max_lat),
    ]
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    return min(xs), min(ys), max(xs), max(ys)


def _wms_crs_element() -> str:
    """与 SURFACE_TILE_CRS / 响应 tile_crs 一致。GCJ02 不是 CRS:84。"""
    c = (settings.SURFACE_TILE_CRS or "GCJ02").upper().replace("-", "")
    if c == "WGS84":
        return "CRS:84"
    if c == "GCJ02":
        return "GCJ02"
    return c


def _public() -> str:
    return settings.PUBLIC_BASE_URL.rstrip("/")


@router.get("/tiles", response_model=TilesResponse)
def tiles(
    dataset_id: str,
    bbox: str = Query(..., description="minLon,minLat,maxLon,maxLat"),
    bbox_crs: str = Query("WGS84", description="bbox 坐标系：WGS84（默认，对齐前端）或 GCJ02"),
    time_from: str | None = None,
    time_to: str | None = None,
    time_start: float | None = None,
    time_end: float | None = None,
    fields: str | None = Query(None, description="逗号分隔属性列"),
    limit: int = 5000,
    cursor: str | None = None,
    output_crs: str | None = Query(None, description="WGS84（默认，对齐前端）或 GCJ02"),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """
    库内几何为 WGS84。默认出库 WGS84，独立前端 toRenderCRS() 负责纠偏到高德。
    传 output_crs=GCJ02 时服务端纠偏（须阅读响应 crs 字段，避免二次转换）。
    bbox 默认按 WGS84 解释；仅 bbox_crs=GCJ02 时转换四角。
    """
    _owns_dataset(db, dataset_id, user)
    min_lon, min_lat, max_lon, max_lat = _parse_bbox(bbox)
    qmin_lon, qmin_lat, qmax_lon, qmax_lat = _bbox_to_wgs84(
        min_lon, min_lat, max_lon, max_lat, bbox_crs,
    )
    out_crs = (output_crs or settings.VECTOR_OUTPUT_CRS).upper().replace("-", "")

    offset = int(cursor or 0)
    sql = """
        SELECT id, ST_X(geom) AS lon, ST_Y(geom) AS lat, observed_time, properties
        FROM spatial_features
        WHERE dataset_id = :did
          AND ST_Intersects(geom, ST_MakeEnvelope(:min_lon, :min_lat, :max_lon, :max_lat, 4326))
    """
    params = {
        "did": dataset_id, "min_lon": qmin_lon, "min_lat": qmin_lat,
        "max_lon": qmax_lon, "max_lat": qmax_lat, "limit": limit + 1, "offset": offset,
    }
    if time_from:
        sql += " AND observed_time >= :tf"; params["tf"] = time_from
    if time_to:
        sql += " AND observed_time <= :tt"; params["tt"] = time_to
    if time_start is not None:
        sql += " AND (observed_time IS NULL OR EXTRACT(EPOCH FROM observed_time)*1000 >= :ts)"
        params["ts"] = time_start
    if time_end is not None:
        sql += " AND (observed_time IS NULL OR EXTRACT(EPOCH FROM observed_time)*1000 <= :te)"
        params["te"] = time_end
    sql += " ORDER BY id LIMIT :limit OFFSET :offset"

    field_set = {f.strip() for f in fields.split(",")} if fields else None
    rows = db.execute(text(sql), params).mappings().all()
    more = len(rows) > limit
    rows = rows[:limit]
    feats = []
    for r in rows:
        lon, lat = float(r["lon"]), float(r["lat"])
        if out_crs == "GCJ02":
            lon, lat = wgs84_to_gcj02(lon, lat)
        props = dict(r["properties"] or {})
        if field_set is not None:
            props = {k: v for k, v in props.items() if k in field_set}
        feats.append(SpatialFeatureOut(
            id=str(r["id"]),
            dataset_id=dataset_id,
            geom={"type": "Point", "coordinates": [round(lon, 6), round(lat, 6)]},
            observed_time=r["observed_time"].isoformat() if r["observed_time"] else None,
            properties=props,
        ))
    return TilesResponse(
        features=feats,
        crs="GCJ02" if out_crs == "GCJ02" else "WGS84",
        next_cursor=str(offset + limit) if more else None,
        total=None,
        count=len(feats),
    )


@router.get("/surface/{task_id}", response_model=SurfaceLayerInfo)
def surface(task_id: str, db: Session = Depends(get_db),
            user: User = Depends(get_current_user)):
    """
    栅格曲面。tile_crs 必填且如实。
    返回的 WMS/XYZ/legend URL 已带短时签名 query `sig`（地图 <img> 无法带 Authorization）。
    UUID 本身不是授权凭据。
    """
    _owns_task(db, task_id, user)
    sig = create_tile_token(user.id, task_id)
    layer = f"{geoserver.ws}:surface_{task_id}"
    base = append_query(f"{_public()}/api/v1/spatial/wms/{task_id}", sig=sig)
    tile_crs = settings.SURFACE_TILE_CRS.upper().replace("-", "")
    if tile_crs not in ("GCJ02", "WGS84"):
        tile_crs = "GCJ02"
    try:
        bbox = field_bbox_gcj02(task_id) if tile_crs == "GCJ02" else [73.0, 18.0, 135.0, 54.0]
        times = load_points_cached(task_id).get("times") or []
    except Exception:
        bbox = [73.0, 18.0, 135.0, 54.0]
        times = []
    time_dim = [str(t) for t in times] if times else None
    xyz = append_query(
        f"{_public()}/api/v1/spatial/surface/{task_id}/xyz/{{z}}/{{x}}/{{y}}.png", sig=sig,
    )
    legend = append_query(
        f"{_public()}/api/v1/spatial/surface/{task_id}/legend.png", sig=sig,
    )
    return SurfaceLayerInfo(
        task_id=task_id,
        service="WMS",
        base_url=base,
        layer_name=layer,
        tile_crs=tile_crs,
        bbox=bbox,
        time_dimension=time_dim,
        legend_url=legend,
        xyz_url_template=xyz,
        wms_url=base,
        layer=layer,
        wms_time_supported=False,
        note=(
            f"瓦片经纬度按 tile_crs={tile_crs} 解释。"
            "XYZ/WMS/legend 必须带 query sig（短时 JWT，绑定 task_id+用户）。"
            "WMS TIME 维未实现。"
        ),
    )


@router.get("/surface/{task_id}/xyz/{z}/{x}/{y}.png")
def surface_xyz(task_id: str, z: int, x: int, y: int, sig: str | None = None,
                db: Session = Depends(get_db)):
    verify_tile_token(sig, task_id)
    if not db.get(ModelTask, task_id):
        raise HTTPException(status_code=404, detail="任务不存在")
    png = render_xyz(task_id, z, x, y)
    return Response(content=png, media_type="image/png")


@router.get("/surface/{task_id}/legend.png")
def surface_legend(task_id: str, sig: str | None = None, db: Session = Depends(get_db)):
    verify_tile_token(sig, task_id)
    if not db.get(ModelTask, task_id):
        raise HTTPException(status_code=404, detail="任务不存在")
    return Response(content=legend_png(task_id), media_type="image/png")


@router.get("/wms/{task_id}")
def wms_getmap(
    task_id: str,
    sig: str | None = None,
    BBOX: str | None = None,
    WIDTH: int = 256,
    HEIGHT: int = 256,
    REQUEST: str = "GetMap",
    db: Session = Depends(get_db),
):
    """简易 WMS。BBOX 按 tile_crs 解释。GetCapabilities 的 CRS 与 tile_crs 一致。"""
    verify_tile_token(sig, task_id)
    task = db.get(ModelTask, task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    tile_crs = (settings.SURFACE_TILE_CRS or "GCJ02").upper().replace("-", "")
    if REQUEST.lower() == "getcapabilities":
        crs = _wms_crs_element()
        xml = f"""<?xml version="1.0"?>
<WMS_Capabilities>
  <Service><Title>GNNWR surface</Title></Service>
  <Capability><Layer>
    <Name>surface_{task_id}</Name>
    <CRS>{crs}</CRS>
    <Abstract>tile_crs={tile_crs}. GCJ02 is not CRS:84/EPSG:4326.</Abstract>
  </Layer></Capability>
</WMS_Capabilities>"""
        return Response(content=xml, media_type="text/xml")
    if not BBOX:
        raise HTTPException(status_code=400, detail="GetMap 需要 BBOX")
    minx, miny, maxx, maxy = _parse_bbox(BBOX)
    png = render_bbox_png(task_id, minx, miny, maxx, maxy, WIDTH, HEIGHT,
                          tile_crs=settings.SURFACE_TILE_CRS)
    return Response(content=png, media_type="image/png")
