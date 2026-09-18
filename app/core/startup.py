"""启动安全检查：占位 SECRET_KEY 在非 DEBUG 下拒绝启动。"""
from __future__ import annotations

import logging

from app.core.config import settings
from app.core.secrets import is_placeholder_secret

log = logging.getLogger("gnnwr.startup")


class InsecureConfiguration(RuntimeError):
    pass


def assert_secure_startup() -> None:
    if not is_placeholder_secret(settings.SECRET_KEY):
        return
    msg = (
        "SECRET_KEY 仍是占位符（CHANGE_ME…）。"
        "生产环境必须设置随机密钥（例如 openssl rand -hex 32）。"
    )
    if not settings.DEBUG:
        raise InsecureConfiguration(msg)
    log.warning("*** %s 当前 DEBUG=true，仅允许本地开发。***", msg)
