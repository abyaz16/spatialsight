"""A disk cache for segmentation responses, keyed by what was actually asked.

SAM 3.1 is the only step in this pipeline that costs money, and the shape of
hackathon work is to run the same clip dozens of times while the code
downstream changes. So the raw response text is cached verbatim, keyed by the
clip's content hash plus every request parameter that could change the answer.
Re-running the pipeline after an edit to the tracker costs nothing.

The cache stores the model's **output text deltas**, not parsed detections.
That is the deliberate choice: the official parser stays in the loop on every
replay, so a cached run exercises the same decoding path as a live one, and a
parser upgrade takes effect without re-spending a cent.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CacheKey:
    video_sha256: str
    model: str
    prompt: str
    mask_encoding: str
    extra: str = ""

    def digest(self) -> str:
        blob = "|".join(
            [self.video_sha256, self.model, self.prompt, self.mask_encoding, self.extra]
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:40]


class ResponseCache:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: CacheKey) -> Path:
        return self.root / f"{key.digest()}.json"

    def get(self, key: CacheKey) -> list[str] | None:
        path = self._path(key)
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        return list(payload["deltas"])

    def put(self, key: CacheKey, deltas: list[str], *, meta: dict | None = None) -> Path:
        path = self._path(key)
        path.write_text(
            json.dumps(
                {
                    "key": {
                        "video_sha256": key.video_sha256,
                        "model": key.model,
                        "prompt": key.prompt,
                        "mask_encoding": key.mask_encoding,
                        "extra": key.extra,
                    },
                    "meta": meta or {},
                    "deltas": deltas,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return path

    def stats(self) -> dict[str, int]:
        files = list(self.root.glob("*.json"))
        return {
            "entries": len(files),
            "bytes": sum(f.stat().st_size for f in files),
        }
