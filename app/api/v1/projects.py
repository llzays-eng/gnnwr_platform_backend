"""项目：创建 / 分页列表 / 详情 / 删除。列表返回 Paginated，对齐独立前端。"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.deps import get_current_user
from app.models.project import Project
from app.models.user import User
from app.schemas.common import Paginated, page_args
from app.schemas.project import ProjectCreate, ProjectOut
from app.services.serialize import project_out

router = APIRouter(prefix="/projects", tags=["projects"])


@router.post("", response_model=ProjectOut, status_code=201)
def create_project(payload: ProjectCreate, db: Session = Depends(get_db),
                   user: User = Depends(get_current_user)):
    proj = Project(
        user_id=user.id, name=payload.name,
        description=payload.description or "",
        scenario_type=payload.scenario_type,
    )
    db.add(proj)
    db.commit()
    db.refresh(proj)
    return project_out(proj)


@router.get("", response_model=Paginated[ProjectOut])
def list_projects(page: int = Query(1, ge=1), page_size: int = Query(50, ge=1, le=200),
                  db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    page, page_size, offset = page_args(page, page_size)
    q = select(Project).where(Project.user_id == user.id)
    total = db.scalar(select(func.count()).select_from(q.subquery())) or 0
    items = db.scalars(q.order_by(Project.created_at.desc()).offset(offset).limit(page_size)).all()
    return Paginated(items=[project_out(p) for p in items], total=total, page=page, page_size=page_size)


@router.get("/{project_id}", response_model=ProjectOut)
def get_project(project_id: str, db: Session = Depends(get_db),
                user: User = Depends(get_current_user)):
    proj = db.get(Project, project_id)
    if not proj or proj.user_id != user.id:
        raise HTTPException(status_code=404, detail="项目不存在")
    return project_out(proj)


@router.delete("/{project_id}", status_code=204)
def delete_project(project_id: str, db: Session = Depends(get_db),
                   user: User = Depends(get_current_user)):
    proj = db.get(Project, project_id)
    if not proj or proj.user_id != user.id:
        raise HTTPException(status_code=404, detail="项目不存在")
    db.delete(proj)
    db.commit()
