"""通用分页信封（对齐独立前端 Paginated<T>）。"""
from __future__ import annotations

from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T")


class Paginated(BaseModel, Generic[T]):
    model_config = ConfigDict(from_attributes=True)

    items: list[T]
    total: int
    page: int = 1
    page_size: int = 50


class ORMModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


def iso(dt) -> str | None:
    if dt is None:
        return None
    try:
        return dt.isoformat()
    except Exception:
        return str(dt)


def page_args(page: int = 1, page_size: int = 50) -> tuple[int, int, int]:
    page = max(1, page)
    page_size = min(max(1, page_size), 200)
    offset = (page - 1) * page_size
    return page, page_size, offset
