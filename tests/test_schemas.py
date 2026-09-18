import pandas as pd

from app.services.field_mapper import infer_field_schema, suggest_mapping, validate_mapping
from app.schemas.model import ModelHyperparams, TrainRequest


def test_suggest_lon_lat_y():
    df = pd.DataFrame({
        "lon": [114.0, 114.1, 114.2],
        "lat": [22.5, 22.6, 22.4],
        "area": [70, 80, 90],
        "price": [50000, 52000, 48000],
    })
    sug = suggest_mapping(df)
    assert sug.lon == "lon" and sug.lat == "lat"
    assert sug.y == "price"
    assert sug.confidence >= 0.6
    schema = infer_field_schema(df)
    names = {s["name"] for s in schema}
    assert names == set(df.columns)


def test_validate_gtnnwr_requires_time():
    df = pd.DataFrame({"lon": [1], "lat": [2], "x1": [3], "y": [4]})
    r = validate_mapping(df, y="y", x=["x1"], lon="lon", lat="lat", model_type="GTNNWR")
    assert not r.ok


def test_hyperparams_aliases():
    hp = ModelHyperparams(hidden=[32, 16], max_epoch=50, lr=0.01, patience=5)
    n = hp.normalized()
    assert n["hidden"] == [32, 16]
    assert n["max_epoch"] == 50
    assert n["lr"] == 0.01
    assert n["patience"] == 5


def test_train_request_gtnnwr_and_gnnwr_rules():
    base = dict(
        project_id="p", dataset_id="d", y_column="y",
        x_columns=["a"], spatial_columns=["lon", "lat"],
    )
    TrainRequest(model_type="GNNWR", **base)
    try:
        TrainRequest(model_type="GTNNWR", **base)
        raise AssertionError("expected validation error")
    except Exception:
        pass
    TrainRequest(model_type="GTNNWR", temporal_column="t", **base)
