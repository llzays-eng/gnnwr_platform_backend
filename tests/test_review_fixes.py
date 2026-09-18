"""Review-fix coverage: tiles ownership, tile sig, no inline train, one-shot WS tickets."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.api.v1.spatial import _bbox_to_wgs84, _owns_dataset, _wms_crs_element
from app.core.config import settings
from app.core.database import get_db
from app.core.deps import get_current_user
from app.core.startup import InsecureConfiguration, assert_secure_startup
from app.main import app
from app.models.project import Dataset, Project
from app.services.jobs import BrokerUnavailable, enqueue
from app.services.signed_url import create_tile_token, verify_tile_token
from app.services.status import public_task_status
from app.services.tickets import consume_ws_ticket, issue_ws_ticket
from app.tasks.progress import build_status_message


class FakeRedis:
    def __init__(self):
        self.store: dict[str, str] = {}

    def setex(self, key, _ttl, value):
        self.store[key] = value

    def getdel(self, key):
        return self.store.pop(key, None)

    def get(self, key):
        return self.store.get(key)

    def delete(self, key):
        self.store.pop(key, None)


class FakeDB:
    def __init__(self, objects: dict):
        self.objects = objects
        self.executed = False

    def get(self, model, key):
        return self.objects.get((model, key))

    def execute(self, *_a, **_k):
        self.executed = True
        raise AssertionError("must not query spatial_features before ownership")

    def close(self):
        pass


def test_tiles_ownership_404_wrong_owner():
    user = SimpleNamespace(id="user-a")
    ds = SimpleNamespace(id="ds-1", project_id="proj-1")
    proj = SimpleNamespace(id="proj-1", user_id="user-b")
    db = FakeDB({(Dataset, "ds-1"): ds, (Project, "proj-1"): proj})
    with pytest.raises(Exception) as ei:
        _owns_dataset(db, "ds-1", user)
    assert ei.value.status_code == 404
    assert db.executed is False


def test_tiles_ownership_404_missing_dataset():
    user = SimpleNamespace(id="user-a")
    db = FakeDB({})
    with pytest.raises(Exception) as ei:
        _owns_dataset(db, "missing", user)
    assert ei.value.status_code == 404


def test_tiles_route_404_does_not_query_features():
    user = SimpleNamespace(id="user-a")
    ds = SimpleNamespace(id="ds-1", project_id="proj-1")
    proj = SimpleNamespace(id="proj-1", user_id="someone-else")
    db = FakeDB({(Dataset, "ds-1"): ds, (Project, "proj-1"): proj})

    def override_user():
        return user

    def override_db():
        yield db

    app.dependency_overrides[get_current_user] = override_user
    app.dependency_overrides[get_db] = override_db
    try:
        c = TestClient(app)
        r = c.get("/api/v1/spatial/tiles", params={
            "dataset_id": "ds-1", "bbox": "114,22,115,23",
        })
        assert r.status_code == 404
        assert db.executed is False
    finally:
        app.dependency_overrides.clear()


def test_surface_xyz_rejects_missing_sig():
    c = TestClient(app)
    r = c.get("/api/v1/spatial/surface/aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa/xyz/1/0/0.png")
    assert r.status_code == 401


def test_surface_legend_and_wms_reject_missing_sig():
    c = TestClient(app)
    tid = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    assert c.get(f"/api/v1/spatial/surface/{tid}/legend.png").status_code == 401
    assert c.get(f"/api/v1/spatial/wms/{tid}").status_code == 401


def test_surface_xyz_rejects_wrong_task_sig():
    sig = create_tile_token("user-1", "task-aaa")
    c = TestClient(app)
    r = c.get(
        "/api/v1/spatial/surface/task-bbb/xyz/1/0/0.png",
        params={"sig": sig},
    )
    assert r.status_code == 401


def test_verify_tile_token_ok_and_expired():
    sig = create_tile_token("user-1", "task-1")
    assert verify_tile_token(sig, "task-1") == "user-1"
    with pytest.raises(Exception) as ei:
        verify_tile_token(None, "task-1")
    assert ei.value.status_code == 401
    expired = create_tile_token("user-1", "task-1", expires_minutes=-1)
    with pytest.raises(Exception) as ei2:
        verify_tile_token(expired, "task-1")
    assert ei2.value.status_code == 401


def test_enqueue_no_inline_without_flag(monkeypatch):
    monkeypatch.setattr(settings, "ALLOW_INLINE_JOBS", False)
    ran: list[int] = []

    def boom(*_a, **_k):
        raise ConnectionError("broker down")

    def fallback(*_a, **_k):
        ran.append(1)

    with pytest.raises(BrokerUnavailable):
        enqueue(boom, "tid", fallback=fallback)
    assert ran == []


def test_enqueue_inline_only_when_flag(monkeypatch):
    monkeypatch.setattr(settings, "ALLOW_INLINE_JOBS", True)

    def boom(*_a, **_k):
        raise ConnectionError("broker down")

    seen = []

    def fallback(tid, key):
        seen.append((tid, key))

    celery_id, inline = enqueue(boom, "t1", "csv", fallback=fallback)
    assert celery_id is None and inline is True
    assert seen == [("t1", "csv")]


def test_ticket_single_use_and_task_bind():
    r = FakeRedis()
    ticket = issue_ws_ticket("user-1", task_id="task-1", client=r)
    assert consume_ws_ticket(ticket, task_id="task-other", client=r) is None
    ticket2 = issue_ws_ticket("user-1", task_id="task-1", client=r)
    assert consume_ws_ticket(ticket2, task_id="task-1", client=r) == "user-1"
    assert consume_ws_ticket(ticket2, task_id="task-1", client=r) is None


def test_bbox_crs_wgs84_passthrough_gcj_transforms():
    box = (114.0, 22.5, 114.2, 22.7)
    assert _bbox_to_wgs84(*box, "WGS84") == box
    transformed = _bbox_to_wgs84(*box, "GCJ02")
    assert transformed != box


def test_wms_crs_matches_surface_tile_crs(monkeypatch):
    monkeypatch.setattr(settings, "SURFACE_TILE_CRS", "GCJ02")
    assert _wms_crs_element() == "GCJ02"
    monkeypatch.setattr(settings, "SURFACE_TILE_CRS", "WGS84")
    assert _wms_crs_element() == "CRS:84"


def test_cancel_public_status_and_ws_payload():
    assert public_task_status("CANCELLED") == "FAILED"
    msg = build_status_message("t1", "CANCELLED", "任务已取消", code="TASK_CANCELLED")
    assert msg["status"] == "FAILED"
    assert msg["code"] == "TASK_CANCELLED"
    assert msg["retryable"] is False


def test_placeholder_secret_refused_when_not_debug(monkeypatch):
    monkeypatch.setattr(settings, "SECRET_KEY", "CHANGE_ME_IN_PRODUCTION_please_use_openssl_rand_hex_32")
    monkeypatch.setattr(settings, "DEBUG", False)
    with pytest.raises(InsecureConfiguration):
        assert_secure_startup()
    monkeypatch.setattr(settings, "DEBUG", True)
    assert_secure_startup()
