"""The SAM 3.1 backend: the only step in the pipeline that spends money.

Three things here are worth knowing before reading the code.

**One request per class.** SAM 3.1 concept prompts take a single short noun
phrase; one concept per request is the documented contract, and a list or a
conjunction degrades the result. Two entity classes therefore means two
requests over the same clip.

**Responses are cached as raw text.** See `cache.py`. A cold run costs money;
every run after it is free and still goes through the official parser.

**Object ids are namespaced per class.** SAM numbers objects within a
request, so `person` and `vehicle` would both return an object `1`. Prefixing
the class keeps two different entities from being fused by the tracker.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Sequence

from meta_sam_parser import (
    SegmentationBoxRecord,
    SegmentationMaskRecord,
    decode_mask_to_raster,
    parse_responses_stream,
    video_segmentation_format,
)

from ..cache import CacheKey, ResponseCache
from ..probe import probe_video
from ..types import Box, Detection, Mask, VideoMeta

# Documented SAM 3.1 video limits, enforced locally so a bad request fails
# before it is billed rather than after.
MAX_VIDEO_FRAMES = 15_000
MIN_BILLED_FRAMES = 50
MAX_TRACKED_OBJECTS = 16

PRICE_PER_1K_FRAMES_USD = 0.20
PRICE_PER_1K_IMAGES_USD = 2.50


class SamBackend:
    """Segments a local video file with SAM 3.1 via the Meta Model API."""

    name = "sam-3.1"

    def __init__(
        self,
        video_path: str | Path,
        *,
        api_key: str,
        api_base: str = "https://api.meta.ai/v1",
        model: str = "sam-3.1",
        cache_dir: str | Path = "data/cache",
        mask_encoding: str = "one_bit",
        concept_for: dict[str, str] | None = None,
        allow_spend: bool = False,
    ) -> None:
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(self.video_path)
        self.api_key = api_key
        self.api_base = api_base
        self.model = model
        self.mask_encoding = mask_encoding
        self.cache = ResponseCache(cache_dir)
        self.allow_spend = allow_spend
        # The noun phrase sent for each class. Short and concrete, per the
        # concept-prompt guidance: commands and lists are unreliable.
        self.concept_for = concept_for or {
            "person": "person",
            "vehicle": "forklift",
        }
        self._meta: VideoMeta | None = None
        self._file_id: str | None = None
        self.last_run: dict[str, object] = {}

    # ----- metadata -------------------------------------------------------

    def meta(self) -> VideoMeta:
        if self._meta is None:
            self._meta = probe_video(self.video_path)
        return self._meta

    def estimate_cost_usd(self) -> float:
        frames = max(MIN_BILLED_FRAMES, self.meta().frame_count)
        per_request = frames / 1000.0 * PRICE_PER_1K_FRAMES_USD
        return round(per_request * len(self.concept_for), 4)

    # ----- the paid path --------------------------------------------------

    def detect(self, labels: Sequence[str] = ("person", "vehicle")) -> list[Detection]:
        meta = self.meta()
        if meta.frame_count > MAX_VIDEO_FRAMES:
            raise ValueError(
                f"{self.video_path.name} has {meta.frame_count} frames; SAM 3.1 accepts "
                f"at most {MAX_VIDEO_FRAMES} per request. Trim or resample the clip "
                f"(see `ss trim`)."
            )

        detections: list[Detection] = []
        spent_requests = 0
        for label in labels:
            concept = self.concept_for.get(label, label)
            key = CacheKey(
                video_sha256=meta.sha256,
                model=self.model,
                prompt=concept,
                mask_encoding=self.mask_encoding,
            )
            deltas = self.cache.get(key)
            if deltas is None:
                if not self.allow_spend:
                    raise RuntimeError(
                        f"no cached response for concept {concept!r} on "
                        f"{self.video_path.name}, and spending is disabled. "
                        f"Re-run with --allow-spend (estimated "
                        f"${self.estimate_cost_usd():.2f} for this clip)."
                    )
                deltas = asyncio.run(self._request(concept))
                self.cache.put(
                    key,
                    deltas,
                    meta={
                        "clip": self.video_path.name,
                        "frames": meta.frame_count,
                        "label": label,
                        "concept": concept,
                        "est_usd": round(
                            max(MIN_BILLED_FRAMES, meta.frame_count)
                            / 1000.0
                            * PRICE_PER_1K_FRAMES_USD,
                            4,
                        ),
                    },
                )
                spent_requests += 1
            detections.extend(self._parse(deltas, label))

        self.last_run = {
            "requests_billed": spent_requests,
            "cache": self.cache.stats(),
            "est_usd_this_run": round(
                spent_requests
                * max(MIN_BILLED_FRAMES, meta.frame_count)
                / 1000.0
                * PRICE_PER_1K_FRAMES_USD,
                4,
            ),
        }
        return detections

    async def _upload(self, client) -> str:
        """Upload the clip once; SAM then references it by file id."""
        if self._file_id is None:
            with self.video_path.open("rb") as fh:
                uploaded = await client.files.create(file=fh, purpose="user_data")
            self._file_id = uploaded.id
        return self._file_id

    async def _request(self, concept: str) -> list[str]:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(base_url=self.api_base, api_key=self.api_key)
        try:
            file_id = await self._upload(client)
            deltas: list[str] = []
            async with client.responses.stream(
                model=self.model,
                input=[
                    {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": concept},
                            {"type": "input_video", "file_id": file_id},
                        ],
                    }
                ],
                metadata={
                    "mask_encoding": self.mask_encoding,
                    "include_confidence": "true",
                },
            ) as stream:
                async for event in stream:
                    if getattr(event, "type", "") == "response.output_text.delta":
                        deltas.append(event.delta)
            return deltas
        finally:
            await client.close()

    # ----- parsing --------------------------------------------------------

    def _parse(self, deltas: list[str], label: str) -> list[Detection]:
        return asyncio.run(_parse_deltas(deltas, label))


async def _replay(deltas: list[str], item_id: str = "cached"):
    """Feed cached text back through the official parser as Responses events.

    The parser tracks which output item a delta belongs to, so the lane
    identity (`item_id`, `output_index`, `content_index`) has to be present
    and consistent even on a replay; without it the stream is rejected as
    ambiguous.
    """
    lane = {"item_id": item_id, "output_index": 0, "content_index": 0}
    for delta in deltas:
        yield {"type": "response.output_text.delta", "delta": delta, **lane}
    # The lane has to be closed before the response completes, or the parser
    # rejects the stream as truncated.
    yield {"type": "response.output_text.done", "text": "".join(deltas), **lane}
    yield {"type": "response.completed"}


async def _parse_deltas(deltas: list[str], label: str) -> list[Detection]:
    """Turn SAM output text into `Detection`s, pairing each box with its mask."""
    parsed = parse_responses_stream(_replay(deltas), video_segmentation_format())
    async for _ in parsed:  # drive the stream; snapshots are cumulative
        pass
    result = await parsed.final_result()

    boxes: dict[tuple[int, str], SegmentationBoxRecord] = {}
    masks: dict[tuple[int, str], SegmentationMaskRecord] = {}
    for record in result.records:
        frame = record.frame.frame_index if record.frame is not None else 0
        if isinstance(record, SegmentationBoxRecord):
            boxes[(frame, record.object_id)] = record
        elif isinstance(record, SegmentationMaskRecord):
            masks[(frame, record.object_id)] = record

    detections: list[Detection] = []
    for (frame, object_id), box_record in sorted(boxes.items()):
        box = Box(
            left=box_record.left,
            top=box_record.top,
            right=box_record.right,
            bottom=box_record.bottom,
        )
        mask: Mask | None = None
        mask_record = masks.get((frame, object_id))
        if mask_record is not None:
            raster = decode_mask_to_raster(mask_record.mask)
            bounds = mask_record.bounds
            mask = Mask(
                rows=mask_record.mask.height,
                cols=mask_record.mask.width,
                data=bytes(raster),
                box=Box(bounds.left, bounds.top, bounds.right, bounds.bottom),
            )
        detections.append(
            Detection(
                frame=frame,
                # SAM numbers objects per request, so two concepts both return
                # an object "1". Namespacing by class keeps them distinct.
                object_id=f"{label}:{object_id}",
                label=label,
                box=box,
                mask=mask,
                confidence=box_record.confidence,
            )
        )
    return detections


class CachedSamBackend:
    """Replays cached SAM responses with no network and no key.

    This is what the submission runs in CI, and what a judge can run without
    credentials: the cache directory is checked in for the eval clips, so the
    numbers in the write-up are reproducible by anyone.
    """

    name = "sam-3.1 (cached)"

    def __init__(
        self,
        manifest_path: str | Path,
        clip: str,
        cache_dir: str | Path = "data/cache",
    ) -> None:
        self.clip = clip
        self.cache = ResponseCache(cache_dir)
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        if clip not in manifest:
            raise KeyError(f"clip {clip!r} is not in {manifest_path}")
        self.entry = manifest[clip]

    def meta(self) -> VideoMeta:
        e = self.entry
        return VideoMeta(
            path=e["path"],
            width=e["width"],
            height=e["height"],
            fps=e["fps"],
            frame_count=e["frame_count"],
            duration_s=e["duration_s"],
            sha256=e["sha256"],
        )

    def detect(self, labels: Sequence[str] = ("person", "vehicle")) -> list[Detection]:
        meta = self.meta()
        concepts: dict[str, str] = self.entry.get("concepts", {})
        model = self.entry.get("model", "sam-3.1")
        encoding = self.entry.get("mask_encoding", "one_bit")
        out: list[Detection] = []
        for label in labels:
            concept = concepts.get(label, label)
            key = CacheKey(meta.sha256, model, concept, encoding)
            deltas = self.cache.get(key)
            if deltas is None:
                raise RuntimeError(
                    f"cache miss for {self.clip}/{label}: expected an entry for "
                    f"concept {concept!r}. Run the live pipeline once to populate it."
                )
            out.extend(asyncio.run(_parse_deltas(deltas, label)))
        return out
