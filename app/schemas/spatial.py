from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class GeoJsonPoint(BaseModel):
    type: str = "Point"
    coordinates: list[float]


class SpatialFeatureOut(BaseModel):
    id: str
    dataset_id: str
    geom: GeoJsonPoint
    observed_time: str | None = None
    properties: dict[str, Any] = Field(default_factory=dict)


class TilesResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "features": [
                        {
                            "id": "1",
                            "dataset_id": "33333333-3333-3333-3333-333333333333",
                            "geom": {"type": "Point", "coordinates": [114.06, 22.54]},
                            "observed_time": None,
                            "properties": {"price": 55000},
                        }
                    ],
                    "crs": "WGS84",
                    "next_cursor": None,
                    "total": 1,
                    "count": 1,
                }
            ]
        }
    )

    features: list[SpatialFeatureOut]
    crs: str = "WGS84"
    next_cursor: str | None = None
    total: int | None = None
    count: int | None = None


class SurfaceLayerInfo(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "task_id": "44444444-4444-4444-4444-444444444444",
                    "service": "WMS",
                    "base_url": "http://localhost:8000/api/v1/spatial/wms/44444444-4444-4444-4444-444444444444",
                    "layer_name": "gnnwr:surface_44444444-4444-4444-4444-444444444444",
                    "tile_crs": "GCJ02",
                    "bbox": [113.7, 22.3, 114.4, 22.8],
                    "time_dimension": None,
                    "legend_url": "http://localhost:8000/api/v1/spatial/surface/44444444-4444-4444-4444-444444444444/legend.png",
                    "xyz_url_template": "http://localhost:8000/api/v1/spatial/surface/44444444-4444-4444-4444-444444444444/xyz/{z}/{x}/{y}.png",
                    "wms_time_supported": False,
                }
            ]
        }
    )

    task_id: str
    service: str = "WMS"
    base_url: str
    layer_name: str
    tile_crs: str
    bbox: list[float]
    time_dimension: list[str] | None = None
    legend_url: str | None = None
    xyz_url_template: str | None = None
    wms_url: str | None = None  # 主仓兼容
    layer: str | None = None
    wms_time_supported: bool = False
    note: str | None = None
