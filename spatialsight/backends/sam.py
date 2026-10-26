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
import logging
from dataclasses import dataclass, field
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

from .._retry import RetryPolicy, RetryableError, call_with_retries

log = logging.getLogger("spatialsight.sam")


class SamError(RuntimeError):
    """Base class for failures of the segmentation call."""


class IncompleteResponseError(SamError, RetryableError):
    """The stream ended before the model said it was finished.

    The most dangerous failure this client can hit, and the reason it has its
    own type. A dropped connection leaves us holding bytes that are perfectly
    valid and silently partial: the records parse, the masks decode, the
    events compile, and the only symptom is that the second half of the clip
    appears to contain nobody. Caching that would make the corruption
    permanent and free to reproduce. So a response is used only when the API
    has said `response.completed`.
    """


class RefusedError(SamError):
    """The model declined the request. A successful HTTP call, not a crash."""


class BudgetExceededError(SamError):
    """The run would spend more than it was allowed to."""


@dataclass
class RunRecord:
    """What a run actually did, for the cost line in the submission."""

    requests_billed: int = 0
    requests_served_from_cache: int = 0
    retries: int = 0
    frames: int = 0
    est_usd: float = 0.0
    concepts: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "requests_billed": self.requests_billed,
            "requests_served_from_cache": self.requests_served_from_cache,
            "retries": self.retries,
            "frames": self.frames,
            "est_usd_this_run": round(self.est_usd, 4),
            "concepts": list(self.concepts),
        }


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
        budget_usd: float | None = 5.0,
        timeout_s: float = 600.0,
        retry: RetryPolicy | None = None,
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
        # A hard ceiling on what one construction of this object may spend.
        # `--allow-spend` says spending is permitted, not that it is
        # unlimited; a loop over clips with no cap is how a grant evaporates.
        self.budget_usd = budget_usd
        self.timeout_s = timeout_s
        self.retry = retry or RetryPolicy()
        self._meta: VideoMeta | None = None
        self._file_id: str | None = None
        self.record = RunRecord()
        self.last_run: dict[str, object] = {}

    # ----- metadata -------------------------------------------------------

    def meta(self) -> VideoMeta:
        if self._meta is None:
            self._meta = probe_video(self.video_path)
        return self._meta

    def cost_per_request_usd(self) -> float:
        frames = max(MIN_BILLED_FRAMES, self.meta().frame_count)
        return frames / 1000.0 * PRICE_PER_1K_FRAMES_USD

    def estimate_cost_usd(self) -> float:
        return round(self.cost_per_request_usd() * len(self.concept_for), 4)

    def _check_budget(self, concept: str) -> None:
        if self.budget_usd is None:
            return
        cost = self.cost_per_request_usd()
        if self.record.est_usd + cost > self.budget_usd:
            raise BudgetExceededError(
                f"segmenting {concept!r} would bring this run to "
                f"${self.record.est_usd + cost:.4f}, over the ${self.budget_usd:.2f} "
                f"budget. Raise budget_usd, or resample the clip to fewer frames "
                f"(`ss resample`)."
            )

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
        self.record = RunRecord(frames=meta.frame_count)

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
                    raise SamError(
                        f"no cached response for concept {concept!r} on "
                        f"{self.video_path.name}, and spending is disabled. "
                        f"Re-run with --allow-spend (estimated "
                        f"${self.estimate_cost_usd():.2f} for this clip)."
                    )
                self._check_budget(concept)
                deltas = self._request_with_retries(concept)
                # Only a response the API confirmed complete reaches the
                # cache; `_request` raises otherwise.
                self.cache.put(
                    key,
                    deltas,
                    meta={
                        "clip": self.video_path.name,
                        "frames": meta.frame_count,
                        "label": label,
                        "concept": concept,
                        "est_usd": round(self.cost_per_request_usd(), 4),
                    },
                )
                self.record.requests_billed += 1
                self.record.est_usd += self.cost_per_request_usd()
            else:
                self.record.requests_served_from_cache += 1
            self.record.concepts.append(concept)
            detections.extend(self._parse(deltas, label))

        self.last_run = {**self.record.as_dict(), "cache": self.cache.stats()}
        return detections

    def _request_with_retries(self, concept: str) -> list[str]:
        def note(exc: Exception, attempt: int, delay: float) -> None:
            self.record.retries += 1
            log.warning(
                "segmenting %r failed (%s: %s); attempt %d of %d, retrying in %.1fs",
                concept, type(exc).__name__, exc, attempt,
                self.retry.max_attempts, delay,
            )

        return call_with_retries(
            lambda: asyncio.run(self._request(concept)),
            policy=self.retry,
            on_retry=note,
            what=f"segmenting {concept!r}",
        )

    async def _upload(self, client) -> str:
        """Upload the clip once; SAM then references it by file id."""
        if self._file_id is None:
            with self.video_path.open("rb") as fh:
                uploaded = await client.files.create(file=fh, purpose="user_data")
            self._file_id = uploaded.id
        return self._file_id

    async def _request(self, concept: str) -> list[str]:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(
            base_url=self.api_base,
            api_key=self.api_key,
            timeout=self.timeout_s,
            max_retries=0,  # retries are ours, so they can be counted and logged
        )
        try:
            file_id = await self._upload(client)
            deltas: list[str] = []
            completed = False
            refusal: list[str] = []

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
                    kind = getattr(event, "type", "")
                    if kind == "response.output_text.delta":
                        deltas.append(event.delta)
                    elif kind == "response.refusal.delta":
                        refusal.append(getattr(event, "delta", ""))
                    elif kind == "response.completed":
                        completed = True
                    elif kind in ("response.failed", "response.incomplete"):
                        raise IncompleteResponseError(
                            f"the API reported {kind} for concept {concept!r}"
                        )

            if refusal:
                raise RefusedError(
                    f"the model refused concept {concept!r}: {''.join(refusal).strip()}"
                )
            if not completed:
                # Valid but partial bytes. Refusing them here is the whole
                # point: accepted, they would be cached and silently wrong.
                raise IncompleteResponseError(
                    f"the stream for concept {concept!r} ended after "
                    f"{len(deltas)} frame record(s) without response.completed; "
                    f"treating it as truncated rather than caching a partial clip"
                )
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
