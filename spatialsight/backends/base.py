"""The one interface the rest of the pipeline is allowed to know about."""

from __future__ import annotations

from typing import Protocol, Sequence

from ..types import Detection, VideoMeta


class SegmentationBackend(Protocol):
    """Produces per-frame detections for a clip.

    Implemented twice: by `SamBackend`, which calls SAM 3.1 and costs money,
    and by `SyntheticBackend`, which simulates a scene and costs nothing. The
    deterministic layers are tested entirely against the second one.
    """

    name: str

    def meta(self) -> VideoMeta: ...

    def detect(self, labels: Sequence[str]) -> list[Detection]: ...
