"""Render the simulated scene into genuine SAM 3.1 wire format.

This is the piece that makes the project testable without a key, and it is
worth being precise about what it buys. The synthetic backend short-circuits
segmentation entirely: useful, but it never exercises the parts most likely
to break on contact with the real API -- the response grammar, the base85
mask payload, the frame headers, the object-id namespacing, and the cache
replay path.

So the scene is encoded here into the same text SAM returns, using the
official parser's own mask encoder, and written into the response cache as if
it had come off the wire. `CachedSamBackend` then reads it back through the
real parser. Everything from the cache onwards is the production path.

`_encode_raster` is a private function of `meta_sam_parser`. Depending on it
is a deliberate, contained bet: it is used only to build fixtures, never at
run time, and if a future version moves it only the test fixtures regenerate.
"""

from __future__ import annotations

import json
from pathlib import Path

from meta_sam_parser._mask_codec import _encode_raster

from .backends.synthetic import SyntheticBackend, SyntheticScene
from .cache import CacheKey, ResponseCache
from .types import Detection, VideoMeta

# The noun phrase SAM is asked for, per entity class.
CONCEPTS = {"person": "person", "vehicle": "forklift"}


def _record(det: Detection, object_id: str, meta: VideoMeta) -> str | None:
    """One `<objid><|box|><|mask|>` record, or None if it cannot be represented.

    SAM only ever returns boxes that lie inside the frame, so a detection
    clipped away at the edge is dropped rather than encoded -- exactly what
    the real API would do with an object leaving the shot.
    """
    if det.mask is None:
        return None
    left = max(0, int(round(det.box.left)))
    top = max(0, int(round(det.box.top)))
    right = min(meta.width - 1, int(round(det.box.right)))
    bottom = min(meta.height - 1, int(round(det.box.bottom)))
    if right <= left or bottom <= top:
        return None

    rows, cols = det.mask.rows, det.mask.cols
    payload = _encode_raster(list(det.mask.data), cols, rows)
    box = (
        f"<|box;x1={left};y1={top};x2={right};y2={bottom}"
        f";w={meta.width};h={meta.height}"
        + (f";c={det.confidence}" if det.confidence is not None else "")
        + "|>"
    )
    mask = (
        "<|mask;x=0;y=0"
        + (f";c={det.confidence}" if det.confidence is not None else "")
        + f";data={rows},{cols},{payload}|>"
    )
    return f"{object_id}{box}{mask}"


def encode_scene(
    detections: list[Detection], meta: VideoMeta, label: str
) -> list[str]:
    """Group a class's detections into one output line per frame.

    Object ids are renumbered from 1 per concept, because that is what SAM
    does: ids are scoped to a request, not to the clip.
    """
    wanted = [d for d in detections if d.label == label]
    renumber: dict[str, str] = {}
    by_frame: dict[int, list[str]] = {}
    for det in sorted(wanted, key=lambda d: (d.frame, d.object_id)):
        if det.object_id not in renumber:
            renumber[det.object_id] = str(len(renumber) + 1)
        encoded = _record(det, renumber[det.object_id], meta)
        if encoded is not None:
            by_frame.setdefault(det.frame, []).append(encoded)

    return [
        f"<{frame}f>" + ",".join(records) + "\n"
        for frame, records in sorted(by_frame.items())
        if records
    ]


def build_fixture(
    clip: str = "synthetic-01",
    scene: SyntheticScene | None = None,
    cache_dir: str | Path = "data/cache",
    manifest_path: str | Path = "data/db/clips.json",
    model: str = "sam-3.1",
) -> dict[str, object]:
    """Write a complete cached SAM response for the simulated clip."""
    backend = SyntheticBackend(clip, scene)
    meta = backend.meta()
    detections = backend.detect()
    cache = ResponseCache(cache_dir)

    written: dict[str, int] = {}
    for label, concept in CONCEPTS.items():
        deltas = encode_scene(detections, meta, label)
        if not deltas:
            continue
        key = CacheKey(
            video_sha256=meta.sha256,
            model=model,
            prompt=concept,
            mask_encoding="one_bit",
        )
        cache.put(
            key,
            deltas,
            meta={
                "clip": clip,
                "label": label,
                "concept": concept,
                "frames": meta.frame_count,
                "source": "fixture (simulated scene, real wire format)",
                "est_usd": 0.0,
            },
        )
        written[label] = len(deltas)

    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else {}
    )
    manifest[clip] = {
        "path": meta.path,
        "width": meta.width,
        "height": meta.height,
        "fps": meta.fps,
        "frame_count": meta.frame_count,
        "duration_s": meta.duration_s,
        "sha256": meta.sha256,
        "backend": "fixture",
        "model": model,
        "mask_encoding": "one_bit",
        "concepts": CONCEPTS,
        "zones_path": f"zones/{clip}.json",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return {"clip": clip, "lines_per_label": written, "cache": cache.stats()}
