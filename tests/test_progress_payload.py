from app.tasks.progress import build_progress_message, build_status_message


def test_progress_payload_shape():
    msg = build_progress_message(
        "t1", "RUNNING", 12, 100, 0.12, 0.15, 34.0, 210.0,
        coef_summary=[{"variable": "intercept", "mean": 1.0}],
    )
    assert msg["type"] == "progress"
    assert msg["task_id"] == "t1"
    assert msg["status"] == "RUNNING"
    p = msg["progress"]
    assert p["epoch"] == 12
    assert p["total_epochs"] == 100
    assert p["train_loss"] == 0.12
    assert p["val_loss"] == 0.15
    assert p["elapsed_s"] == 34.0
    assert p["eta_s"] == 210.0
    assert msg["coef_summary"][0]["variable"] == "intercept"


def test_progress_without_coef_summary():
    msg = build_progress_message("t1", "RUNNING", 1, 10, 0.2, 0.3, 1.0, None)
    assert "coef_summary" not in msg
    assert msg["progress"]["eta_s"] is None


def test_status_payload():
    ok = build_status_message("t1", "SUCCESS")
    assert ok == {"type": "status", "task_id": "t1", "status": "SUCCESS"}
    bad = build_status_message("t1", "FAILED", "boom")
    assert bad["type"] == "error"
    assert bad["message"] == "boom"
