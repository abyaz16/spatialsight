"""Runtime configuration, read from the environment with safe defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _path(env: str, default: str) -> Path:
    raw = os.environ.get(env, default)
    p = Path(raw)
    return p if p.is_absolute() else REPO_ROOT / p


@dataclass(frozen=True)
class Config:
    api_key: str | None
    api_base: str
    sam_model: str
    spark_model: str
    db_path: Path
    cache_dir: Path

    @property
    def has_key(self) -> bool:
        return bool(self.api_key)


def load() -> Config:
    return Config(
        api_key=os.environ.get("MODEL_API_KEY") or None,
        api_base=os.environ.get("META_API_BASE", "https://api.meta.ai/v1"),
        sam_model=os.environ.get("SAM_MODEL", "sam-3.1"),
        spark_model=os.environ.get("SPARK_MODEL", "muse-spark-1.3"),
        db_path=_path("SS_DB", "data/db/spatialsight.sqlite"),
        cache_dir=_path("SS_CACHE", "data/cache"),
    )
