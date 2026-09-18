"""任务对外状态：前端仅有 PENDING | RUNNING | SUCCESS | FAILED。"""


def public_task_status(status: str | None) -> str:
    if status == "CANCELLED":
        return "FAILED"
    return status or "PENDING"
