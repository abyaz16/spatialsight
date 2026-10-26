"""The contracts every stage of the pipeline speaks.

`Detection` is the seam between the segmentation backend and everything
downstream. The real SAM 3.1 client and the offline fixture backend both emit
`Detection`s, so the deterministic layers -- tracking, zone occupancy, event
compilation -- never know which one produced them, and can be tested without
an API key or a cent of spend.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Box:
    """Axis-aligned box in frame pixel coordinates. Right/bottom are inclusive."""

    left: float
    top: float
    right: float
    bottom: float

    @property
    def width(self) -> float:
        return max(0.0, self.right - self.left)

    @property
    def height(self) -> float:
        return max(0.0, self.bottom - self.top)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def centroid(self) -> tuple[float, float]:
        return ((self.left + self.right) / 2.0, (self.top + self.bottom) / 2.0)

    def iou(self, other: Box) -> float:
        ix = max(0.0, min(self.right, other.right) - max(self.left, other.left))
        iy = max(0.0, min(self.bottom, other.bottom) - max(self.top, other.top))
        inter = ix * iy
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0


@dataclass(frozen=True)
class Mask:
    """A binary mask covering `box`, stored row-major as one byte per pixel.

    SAM returns masks smaller than the frame plus the bounds they occupy, so a
    mask is only meaningful together with its placement. `rows * cols` must
    equal `len(data)`.
    """

    rows: int
    cols: int
    data: bytes
    box: Box

    def __post_init__(self) -> None:
        if len(self.data) != self.rows * self.cols:
            raise ValueError(
                f"mask data is {len(self.data)} bytes, expected {self.rows * self.cols}"
            )

    @property
    def pixel_count(self) -> int:
        return sum(self.data)


@dataclass(frozen=True)
class Detection:
    """One object seen in one frame."""

    frame: int
    object_id: str
    label: str
    box: Box
    mask: Mask | None = None
    confidence: float | None = None


@dataclass(frozen=True)
class VideoMeta:
    """Everything downstream needs to turn frame indices into wall-clock time."""

    path: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration_s: float
    sha256: str

    def to_seconds(self, frame: int) -> float:
        return frame / self.fps if self.fps > 0 else 0.0


@dataclass
class Track:
    """A single entity's life across the clip, after ID reconciliation."""

    entity_id: int
    label: str
    detections: list[Detection] = field(default_factory=list)
    source_object_ids: list[str] = field(default_factory=list)

    @property
    def first_frame(self) -> int:
        return min(d.frame for d in self.detections)

    @property
    def last_frame(self) -> int:
        return max(d.frame for d in self.detections)
