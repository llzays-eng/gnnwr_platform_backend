from app.services.crs_transform import (
    gcj02_to_wgs84, normalize_crs, offshore_ratio, transform, wgs84_to_gcj02,
)


def test_roundtrip_shanghai():
    lon, lat = 121.4737, 31.2304
    g = wgs84_to_gcj02(lon, lat)
    assert g != (lon, lat)
    back = gcj02_to_wgs84(*g)
    assert abs(back[0] - lon) < 1e-5
    assert abs(back[1] - lat) < 1e-5


def test_out_of_china_identity():
    lon, lat = 2.35, 48.85
    assert wgs84_to_gcj02(lon, lat) == (lon, lat)


def test_normalize_and_transform():
    assert normalize_crs("epsg:4326") == "WGS84"
    assert normalize_crs("GCJ-02") == "GCJ02"
    lon, lat = transform(121.5, 31.2, "WGS84", "GCJ02")
    assert lon != 121.5


def test_offshore_ratio():
    pts = [(121.5, 31.2), (114.0, 22.5), (0.0, 0.0)]
    r = offshore_ratio(pts, sample=10)
    assert 0 < r < 1
