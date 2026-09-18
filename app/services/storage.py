"""
对象存储：优先 MinIO，不可用则回退本地磁盘。api 与 worker 需共享同一后端。
"""
from __future__ import annotations

import io
import logging
import shutil
from datetime import timedelta
from pathlib import Path

from app.core.config import settings

log = logging.getLogger(__name__)


class StorageService:
    def __init__(self) -> None:
        self._minio = None
        self._local_dir = Path(settings.LOCAL_STORAGE_DIR)
        self._local_dir.mkdir(parents=True, exist_ok=True)
        prefer = settings.STORAGE_BACKEND.lower()
        if prefer != "local":
            self._try_minio()
        if prefer == "minio" and self._minio is None:
            log.warning("STORAGE_BACKEND=minio 但 MinIO 不可用，回退本地磁盘")

    def _try_minio(self) -> None:
        try:
            from minio import Minio

            client = Minio(
                settings.MINIO_ENDPOINT,
                access_key=settings.MINIO_ACCESS_KEY,
                secret_key=settings.MINIO_SECRET_KEY,
                secure=settings.MINIO_SECURE,
            )
            if not client.bucket_exists(settings.MINIO_BUCKET):
                client.make_bucket(settings.MINIO_BUCKET)
            self._minio = client
        except Exception as exc:  # noqa: BLE001
            log.info("MinIO 未启用（%s），使用本地存储 %s", exc, self._local_dir)
            self._minio = None

    @property
    def backend(self) -> str:
        return "minio" if self._minio else "local"

    def put_bytes(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> str:
        if self._minio:
            self._minio.put_object(
                settings.MINIO_BUCKET, key, io.BytesIO(data),
                length=len(data), content_type=content_type,
            )
        else:
            path = self._local_dir / key
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        return key

    def put_file(self, key: str, file_path: str) -> str:
        if self._minio:
            self._minio.fput_object(settings.MINIO_BUCKET, key, file_path)
        else:
            dst = self._local_dir / key
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(file_path, dst)
        return key

    def get_bytes(self, key: str) -> bytes:
        if self._minio:
            resp = self._minio.get_object(settings.MINIO_BUCKET, key)
            try:
                return resp.read()
            finally:
                resp.close()
                resp.release_conn()
        return (self._local_dir / key).read_bytes()

    def exists(self, key: str) -> bool:
        if self._minio:
            try:
                self._minio.stat_object(settings.MINIO_BUCKET, key)
                return True
            except Exception:
                return False
        return (self._local_dir / key).exists()

    def local_path(self, key: str) -> str:
        if self._minio:
            tmp = Path("/tmp/gnnwr-cache") / key
            tmp.parent.mkdir(parents=True, exist_ok=True)
            self._minio.fget_object(settings.MINIO_BUCKET, key, str(tmp))
            return str(tmp)
        return str(self._local_dir / key)

    def presign_put(self, key: str, expires: int = 3600) -> str | None:
        if not self._minio:
            return None
        return self._minio.presigned_put_object(
            settings.MINIO_BUCKET, key, expires=timedelta(seconds=expires)
        )

    def list_prefix(self, prefix: str) -> list[str]:
        if self._minio:
            return [o.object_name for o in self._minio.list_objects(
                settings.MINIO_BUCKET, prefix=prefix, recursive=True
            )]
        root = self._local_dir / prefix
        if not root.exists():
            return []
        if root.is_file():
            return [prefix]
        return [str(p.relative_to(self._local_dir)) for p in root.rglob("*") if p.is_file()]


storage = StorageService()
