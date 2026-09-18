from __future__ import annotations

from datetime import datetime

from geoalchemy2 import Geometry
from sqlalchemy import BigInteger, ForeignKey, Index
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class SpatialFeature(Base):
    """
    清洗后统一入库的标准化地理要素：
    一个（空间点，[时间]），若干属性字段，一个目标值。
    几何一律 WGS84 / EPSG:4326。
    """

    __tablename__ = "spatial_features"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    dataset_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), ForeignKey("datasets.id"), index=True
    )
    geom: Mapped[object] = mapped_column(Geometry(geometry_type="POINT", srid=4326))
    observed_time: Mapped[datetime | None] = mapped_column(nullable=True)
    properties: Mapped[dict] = mapped_column(JSONB, default=dict)

    __table_args__ = (
        Index("idx_spatial_features_geom", "geom", postgresql_using="gist"),
    )
