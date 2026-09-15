from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ProjectCreate(BaseModel):
    name: str
    description: str = ""
    scenario_type: str = Field(
        default="custom",
        description="air_quality | housing_price | custom",
    )


class ProjectOut(BaseModel):
    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "examples": [
                {
                    "id": "22222222-2222-2222-2222-222222222222",
                    "name": "杭州市二手房",
                    "description": "",
                    "scenario_type": "housing_price",
                    "owner_id": "11111111-1111-1111-1111-111111111111",
                    "created_at": "2026-09-15T00:00:00",
                    "updated_at": "2026-09-15T00:00:00",
                    "is_demo": False,
                }
            ]
        },
    )

    id: str
    name: str
    description: str = ""
    scenario_type: str
    owner_id: str
    created_at: str
    updated_at: str
    is_demo: bool = False
