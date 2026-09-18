"""OpenAPI 契约冒烟：应用可导入，P0 路径齐全。"""
from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app

REQUIRED = [
    "/api/v1/auth/login",
    "/api/v1/auth/register",
    "/api/v1/auth/me",
    "/api/v1/auth/refresh",
    "/api/v1/auth/logout",
    "/api/v1/projects",
    "/api/v1/datasets",
    "/api/v1/datasets/upload",
    "/api/v1/datasets/{dataset_id}/preview",
    "/api/v1/datasets/{dataset_id}/preprocess",
    "/api/v1/models/train",
    "/api/v1/models/tasks",
    "/api/v1/models/tasks/{task_id}/status",
    "/api/v1/models/tasks/{task_id}/result",
    "/api/v1/models/tasks/{task_id}/compare",
    "/api/v1/models/tasks/{task_id}/coefficients",
    "/api/v1/spatial/tiles",
    "/api/v1/spatial/surface/{task_id}",
    "/api/v1/reports/{task_id}/export",
]


def test_health_and_root():
    c = TestClient(app)
    r = c.get("/")
    assert r.status_code == 200
    body = r.json()
    assert body["ws_progress"] == "/ws/models/tasks/{task_id}"
    assert body["tile_crs"]
    h = c.get("/health")
    assert h.status_code == 200
    assert h.json()["api"] == "ok"
    assert "X-Request-Id" in h.headers


def test_openapi_contains_p0_paths():
    c = TestClient(app)
    spec = c.get("/openapi.json").json()
    paths = spec["paths"]
    for p in REQUIRED:
        assert p in paths, f"missing {p}"
    surface = spec["paths"]["/api/v1/spatial/surface/{task_id}"]["get"]
    # 响应模型含 tile_crs（在 schema 组件里）
    assert "tile_crs" in spec["components"]["schemas"]["SurfaceLayerInfo"]["properties"]
    assert "/ws/models/tasks/{task_id}" in paths


def test_docs_up():
    c = TestClient(app)
    assert c.get("/docs").status_code == 200


def test_business_routes_require_auth():
    c = TestClient(app)
    assert c.get("/api/v1/projects").status_code == 401
    assert c.get("/api/v1/datasets", params={"project_id": "x"}).status_code == 401
    assert c.get("/api/v1/models/tasks", params={"project_id": "x"}).status_code == 401
