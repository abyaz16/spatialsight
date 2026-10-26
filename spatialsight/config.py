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
    # The text model. Defaults to Muse Spark on the Meta Model API, but any
    # OpenAI-compatible endpoint works: a local llama.cpp server running a
    # Meta open-weights model, or a hosted free tier.
    llm_base_url: str = "https://api.meta.ai/v1"
    llm_model: str = "muse-spark-1.3"
    llm_api_key: str | None = None
    llm_api_style: str = "responses"
    llm_explicit: bool = False

    @property
    def has_key(self) -> bool:
        """Whether the paid Meta Model API is usable (SAM 3.1)."""
        return bool(self.api_key)

    @property
    def has_llm(self) -> bool:
        """Whether natural-language questions can be answered.

        A local server needs no key, so naming an endpoint is enough.
        """
        return bool(self.llm_api_key) or self.llm_explicit


def load() -> Config:
    api_key = os.environ.get("MODEL_API_KEY") or None
    api_base = os.environ.get("META_API_BASE", "https://api.meta.ai/v1")
    spark_model = os.environ.get("SPARK_MODEL", "muse-spark-1.3")

    explicit_base = os.environ.get("LLM_BASE_URL") or None
    llm_base = explicit_base or api_base
    # The Meta Model API documents the Responses API; llama.cpp and most
    # free tiers speak chat completions. Pick by endpoint unless told.
    default_style = "responses" if "api.meta.ai" in llm_base else "chat"

    return Config(
        api_key=api_key,
        api_base=api_base,
        sam_model=os.environ.get("SAM_MODEL", "sam-3.1"),
        spark_model=spark_model,
        db_path=_path("SS_DB", "data/db/spatialsight.sqlite"),
        cache_dir=_path("SS_CACHE", "data/cache"),
        llm_base_url=llm_base,
        llm_model=os.environ.get("LLM_MODEL") or spark_model,
        llm_api_key=os.environ.get("LLM_API_KEY") or api_key,
        llm_api_style=os.environ.get("LLM_API_STYLE") or default_style,
        llm_explicit=explicit_base is not None,
    )
