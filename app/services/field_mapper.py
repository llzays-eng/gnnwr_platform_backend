"""
字段映射器：对任意上传表格自动识别经纬度 / 时间 / Y / X 候选。
场景类型只影响默认文案，不写死列名。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

_LON_KEYS = ["lon", "lng", "longitude", "经度", "x坐标", "x_coord", "经"]
_LAT_KEYS = ["lat", "latitude", "纬度", "y坐标", "y_coord", "纬"]
_TIME_KEYS = ["time", "date", "datetime", "timestamp", "day", "month", "year",
              "时间", "日期", "年月", "观测时间", "网签时间", "挂牌时间"]
_Y_HINT = ["price", "pm25", "pm2.5", "no2", "concentration", "value", "target",
           "单价", "价格", "浓度", "房价"]


def _match_any(name: str, keys: list[str]) -> bool:
    n = name.strip().lower()
    return any(k in n for k in keys)


def _looks_like_lon(s: pd.Series) -> bool:
    if not pd.api.types.is_numeric_dtype(s):
        return False
    v = s.dropna()
    return len(v) > 0 and float(v.min()) >= -180 and float(v.max()) <= 180 and float(v.std() or 1) > 1e-6


def _looks_like_lat(s: pd.Series) -> bool:
    if not pd.api.types.is_numeric_dtype(s):
        return False
    v = s.dropna()
    return len(v) > 0 and float(v.min()) >= -90 and float(v.max()) <= 90 and float(v.std() or 1) > 1e-6


def _looks_like_time(s: pd.Series) -> bool:
    if pd.api.types.is_integer_dtype(s) or pd.api.types.is_datetime64_any_dtype(s):
        return True
    try:
        pd.to_datetime(s.dropna().head(20), errors="raise")
        return True
    except Exception:
        return False


@dataclass
class MappingSuggestion:
    lon: str | None = None
    lat: str | None = None
    temporal: str | None = None
    y: str | None = None
    x: list[str] = field(default_factory=list)
    candidates: dict = field(default_factory=dict)
    confidence: float = 0.0
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "lon": self.lon,
            "lat": self.lat,
            "temporal": self.temporal,
            "y": self.y,
            "x": self.x,
            "candidates": self.candidates,
            "longitude": self.lon,
            "latitude": self.lat,
            "confidence": self.confidence,
            "reason": self.reason,
        }

    def as_column_guess(self) -> dict:
        return {
            "longitude": self.lon,
            "latitude": self.lat,
            "temporal": self.temporal,
            "confidence": self.confidence,
            "reason": self.reason,
            "y": self.y,
            "x": self.x,
            "candidates": self.candidates,
        }


def suggest_mapping(df: pd.DataFrame) -> MappingSuggestion:
    cols = list(df.columns)
    lon_c, lat_c, time_c = [], [], []

    for c in cols:
        s = df[c]
        name_lon = _match_any(c, _LON_KEYS)
        name_lat = _match_any(c, _LAT_KEYS)
        if name_lon and _looks_like_lon(s):
            lon_c.append(c)
        elif name_lat and _looks_like_lat(s):
            lat_c.append(c)
        elif _match_any(c, _TIME_KEYS) and _looks_like_time(s):
            time_c.append(c)

    if not lon_c:
        lon_c = [c for c in cols if _looks_like_lon(df[c]) and c not in lat_c][:2]
    if not lat_c:
        lat_c = [c for c in cols if _looks_like_lat(df[c]) and c not in lon_c][:2]

    sug = MappingSuggestion()
    sug.lon = lon_c[0] if lon_c else None
    sug.lat = lat_c[0] if lat_c else None
    sug.temporal = time_c[0] if time_c else None

    used = {sug.lon, sug.lat, sug.temporal}
    numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c]) and c not in used]
    y_hits = [c for c in numeric_cols if _match_any(c, _Y_HINT)]
    sug.y = y_hits[0] if y_hits else (numeric_cols[-1] if numeric_cols else None)
    sug.x = [c for c in numeric_cols if c != sug.y]
    sug.candidates = {
        "lon": lon_c or [c for c in cols if _looks_like_lon(df[c])],
        "lat": lat_c or [c for c in cols if _looks_like_lat(df[c])],
        "temporal": time_c,
        "numeric": numeric_cols + ([sug.y] if sug.y and sug.y not in numeric_cols else []),
    }

    if sug.lon and sug.lat and _match_any(sug.lon, _LON_KEYS) and _match_any(sug.lat, _LAT_KEYS):
        sug.confidence = 0.92
        sug.reason = f"列名命中 {sug.lon}/{sug.lat}"
    elif sug.lon and sug.lat:
        sug.confidence = 0.62
        sug.reason = "按数值范围推断经纬度，请人工确认"
    else:
        sug.confidence = 0.2
        sug.reason = "未能可靠识别经纬度列"
    return sug


@dataclass
class ValidationResult:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def validate_mapping(df: pd.DataFrame, y: str, x: list[str], lon: str, lat: str,
                     temporal: str | None = None, model_type: str = "GNNWR") -> ValidationResult:
    errors: list[str] = []
    warnings: list[str] = []
    cols = set(df.columns)

    for role, val in [("目标变量Y", y), ("经度", lon), ("纬度", lat)]:
        if not val or val not in cols:
            errors.append(f"{role} 未指定或不存在于数据列中")
    if not x:
        errors.append("至少需要一个自变量 X")
    for c in x:
        if c not in cols:
            errors.append(f"自变量 {c} 不存在于数据列中")

    if model_type.upper() == "GTNNWR":
        if not temporal or temporal not in cols:
            errors.append("GTNNWR 时空模式必须指定有效的时间字段")
    if temporal and temporal in (x + [y]):
        errors.append("时间字段不能同时作为 X 或 Y")
    if y in x:
        errors.append("目标变量 Y 不能同时作为自变量 X")
    if lon == lat:
        errors.append("经度与纬度不能是同一列")

    if not errors:
        involved = [c for c in ([y, lon, lat] + x + ([temporal] if temporal else [])) if c in cols]
        miss = df[involved].isna().sum()
        for c, n in miss.items():
            if n > 0:
                warnings.append(f"列 {c} 有 {int(n)} 个缺失值，将在清洗阶段处理")
        if lon in cols and not _looks_like_lon(df[lon]):
            warnings.append(f"{lon} 的数值范围不像经度，请确认坐标系")
        if lat in cols and not _looks_like_lat(df[lat]):
            warnings.append(f"{lat} 的数值范围不像纬度，请确认坐标系")

    return ValidationResult(ok=len(errors) == 0, errors=errors, warnings=warnings)


_INT_RE = re.compile(r"^-?\d+$")


def infer_field_schema(df: pd.DataFrame, sample_n: int = 5) -> list[dict]:
    out = []
    n = max(len(df), 1)
    for c in df.columns:
        s = df[c]
        missing = int(s.isna().sum())
        kind = "unknown"
        stats = None
        if pd.api.types.is_bool_dtype(s):
            kind = "boolean"
        elif pd.api.types.is_datetime64_any_dtype(s):
            kind = "datetime"
        elif pd.api.types.is_integer_dtype(s):
            kind = "integer"
        elif pd.api.types.is_numeric_dtype(s):
            kind = "numeric"
        else:
            as_str = s.dropna().astype(str)
            if len(as_str) and as_str.str.match(_INT_RE).mean() > 0.9:
                kind = "integer"
            else:
                kind = "text"
        if kind in ("numeric", "integer") and pd.api.types.is_numeric_dtype(s):
            v = pd.to_numeric(s, errors="coerce").dropna()
            if len(v):
                stats = {
                    "min": float(v.min()), "max": float(v.max()),
                    "mean": float(v.mean()), "std": float(v.std() or 0),
                    "q25": float(v.quantile(0.25)), "q50": float(v.quantile(0.5)),
                    "q75": float(v.quantile(0.75)),
                }
        samples = s.dropna().head(sample_n).tolist()
        samples = [_py(x) for x in samples]
        out.append({
            "name": str(c),
            "kind": kind,
            "missing_count": missing,
            "missing_ratio": round(missing / n, 4),
            "distinct_count": int(s.nunique(dropna=True)),
            "sample_values": samples,
            "stats": stats,
        })
    return out


def _py(v):
    try:
        return v.item()
    except AttributeError:
        if hasattr(v, "isoformat"):
            return v.isoformat()
        return v
