"""GeoServer REST 小封装。一期曲面默认走本服务 XYZ/WMS，本模块为可选增强。"""
from __future__ import annotations

import logging

import requests

from app.core.config import settings

log = logging.getLogger(__name__)


class GeoServerService:
    def __init__(self) -> None:
        self.base = settings.GEOSERVER_URL.rstrip("/")
        self.auth = (settings.GEOSERVER_USER, settings.GEOSERVER_PASSWORD)
        self.ws = settings.GEOSERVER_WORKSPACE

    def reachable(self) -> bool:
        try:
            r = requests.get(f"{self.base}/web/", auth=self.auth, timeout=2)
            return r.status_code < 500
        except Exception:
            return False

    def ensure_workspace(self) -> None:
        url = f"{self.base}/rest/workspaces"
        r = requests.get(f"{url}/{self.ws}", auth=self.auth, timeout=10)
        if r.status_code == 404:
            requests.post(url, json={"workspace": {"name": self.ws}}, auth=self.auth, timeout=10)

    def publish_geotiff(self, store: str, geotiff_path: str) -> str:
        self.ensure_workspace()
        url = f"{self.base}/rest/workspaces/{self.ws}/coveragestores/{store}/file.geotiff"
        with open(geotiff_path, "rb") as f:
            requests.put(url, data=f, headers={"Content-type": "image/tiff"},
                         auth=self.auth, timeout=60)
        return f"{self.ws}:{store}"

    def wms_url(self, layer: str) -> str:
        return f"{self.base}/{self.ws}/wms"


geoserver = GeoServerService()
