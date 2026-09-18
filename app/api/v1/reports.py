"""报告导出：默认同步返回 PDF（无 weasyprint 则 HTML）。前端按 blob/json 探测兼容。"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.deps import get_current_user
from app.models.model_task import ModelTask
from app.models.project import Project
from app.models.user import User
from app.services.reports import render_html

router = APIRouter(prefix="/reports", tags=["reports"])


def _owned_task(db: Session, task_id: str, user: User) -> ModelTask:
    task = db.get(ModelTask, task_id)
    if not task or task.result is None:
        raise HTTPException(status_code=404, detail="结果不存在")
    proj = db.get(Project, task.project_id)
    if not proj or proj.user_id != user.id:
        raise HTTPException(status_code=404, detail="结果不存在")
    return task


@router.get("/{task_id}/export")
def export_report(task_id: str, db: Session = Depends(get_db),
                  user: User = Depends(get_current_user)):
    task = _owned_task(db, task_id, user)
    html = render_html(task)
    try:
        from weasyprint import HTML
        pdf = HTML(string=html).write_pdf()
        return Response(
            content=pdf, media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="report_{task_id}.pdf"'},
        )
    except Exception:
        return Response(content=html, media_type="text/html; charset=utf-8")
