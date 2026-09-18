from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Boolean, ForeignKey, Integer, String, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class Project(Base):
    """分析项目：绑定应用场景（air_quality / housing_price / custom）。"""

    __tablename__ = "projects"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    user_id: Mapped[str] = mapped_column(UUID(as_uuid=False), ForeignKey("users.id"))
    name: Mapped[str] = mapped_column(String(200))
    description: Mapped[str] = mapped_column(String(500), default="")
    scenario_type: Mapped[str] = mapped_column(String(50), default="custom")
    is_demo: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(server_default=func.now(), onupdate=func.now())

    datasets: Mapped[list["Dataset"]] = relationship(
        back_populates="project", cascade="all, delete-orphan"
    )


class Dataset(Base):
    """数据集元数据：文件在对象存储，这里只存路径、schema 与状态机。"""

    __tablename__ = "datasets"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    project_id: Mapped[str] = mapped_column(UUID(as_uuid=False), ForeignKey("projects.id"))
    name: Mapped[str] = mapped_column(String(200), default="")
    filename: Mapped[str] = mapped_column(String(200), default="")
    storage_path: Mapped[str] = mapped_column(String(400))
    file_type: Mapped[str] = mapped_column(String(20))
    format: Mapped[str] = mapped_column(String(20), default="csv")
    source_crs: Mapped[str | None] = mapped_column(String(30), nullable=True)
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    row_count: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(20), default="uploaded")
    status_detail: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    schema_json: Mapped[list] = mapped_column("schema", JSONB, default=list)
    column_guess: Mapped[dict] = mapped_column(JSONB, default=dict)
    mapping: Mapped[dict] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    project: Mapped["Project"] = relationship(back_populates="datasets")
