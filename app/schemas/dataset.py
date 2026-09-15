from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class FieldSchema(BaseModel):
    name: str
    kind: str = "unknown"
    missing_count: int = 0
    missing_ratio: float = 0.0
    distinct_count: int = 0
    sample_values: list[Any] = Field(default_factory=list)
    stats: dict | None = None


class ColumnGuess(BaseModel):
    longitude: str | None = None
    latitude: str | None = None
    temporal: str | None = None
    confidence: float = 0.0
    reason: str = ""
    y: str | None = None
    x: list[str] = Field(default_factory=list)
    candidates: dict = Field(default_factory=dict)


class DatasetStatusDetail(BaseModel):
    code: str
    message: str
    next_actions: list[dict] = Field(default_factory=list)


class DatasetOut(BaseModel):
    model_config = ConfigDict(
        from_attributes=True,
        populate_by_name=True,
        json_schema_extra={
            "examples": [
                {
                    "id": "33333333-3333-3333-3333-333333333333",
                    "project_id": "22222222-2222-2222-2222-222222222222",
                    "filename": "housing.csv",
                    "format": "csv",
                    "size_bytes": 12000,
                    "row_count": 340,
                    "status": "uploaded",
                    "status_detail": None,
                    "source_crs": "WGS84",
                    "schema": [],
                    "column_guess": {
                        "longitude": "lon",
                        "latitude": "lat",
                        "temporal": None,
                        "confidence": 0.9,
                        "reason": "列名命中 lon/lat",
                    },
                    "created_at": "2026-09-15T00:00:00",
                    "name": "housing.csv",
                    "file_type": "csv",
                }
            ]
        },
    )

    id: str
    project_id: str
    filename: str
    format: str
    size_bytes: int = 0
    row_count: int = 0
    status: str
    status_detail: dict | None = None
    source_crs: str | None = None
    schema_: list[dict] = Field(default_factory=list, alias="schema")
    column_guess: dict = Field(default_factory=dict)
    created_at: str
    # 兼容主仓 DatasetOut
    name: str | None = None
    file_type: str | None = None


class PreprocessRequest(BaseModel):
    """同时接受独立前端字段名与主仓 {y,x,lon,lat} 别名。"""

    source_crs: str = "WGS84"
    longitude_column: str | None = None
    latitude_column: str | None = None
    temporal_column: str | None = None
    drop_invalid_rows: bool = True
    y: str | None = None
    x: list[str] | None = None
    lon: str | None = None
    lat: str | None = None
    temporal: str | None = None

    def lon_col(self) -> str | None:
        return self.longitude_column or self.lon

    def lat_col(self) -> str | None:
        return self.latitude_column or self.lat

    def time_col(self) -> str | None:
        return self.temporal_column or self.temporal


class PreviewResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "dataset_id": "33333333-3333-3333-3333-333333333333",
                    "total_rows": 340,
                    "columns": ["lon", "lat", "price"],
                    "rows": [{"lon": 114.06, "lat": 22.54, "price": 55000}],
                    "spatial_sample": [[114.06, 22.54]],
                    "offshore_ratio": 0.0,
                    "mapping_suggestion": {"lon": "lon", "lat": "lat", "y": "price", "x": []},
                    "points": [{"lon": 114.067, "lat": 22.543}],
                    "n_rows_total": 340,
                }
            ]
        }
    )

    dataset_id: str
    total_rows: int
    columns: list[str]
    rows: list[dict]
    spatial_sample: list[list[float]]
    offshore_ratio: float = 0.0
    mapping_suggestion: dict = Field(default_factory=dict)
    points: list[dict] = Field(default_factory=list)  # GCJ-02，给高德直接打点
    n_rows_total: int | None = None


class ChunkInitRequest(BaseModel):
    project_id: str
    filename: str
    size: int
    chunk_size: int = 8 * 1024 * 1024
    source_crs: str | None = None


class ChunkInitResponse(BaseModel):
    upload_id: str
    uploaded_chunks: list[int] = Field(default_factory=list)
    chunk_size: int
    expires_in: int = 86400


class PresignResponse(BaseModel):
    upload_id: str
    url: str
    method: str = "PUT"
    headers: dict = Field(default_factory=dict)
    object_key: str
    expires_in: int = 3600
