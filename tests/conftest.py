"""Set a non-placeholder SECRET_KEY before app.core.config is imported."""
from __future__ import annotations

import os

os.environ["SECRET_KEY"] = "pytest-secret-key-not-a-placeholder-value"
os.environ.setdefault("ALLOW_INLINE_JOBS", "false")
