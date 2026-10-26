"""Compile tracks x zones into enter / exit / dwell intervals.

The one design decision that matters here is **hysteresis**. A single
occupancy threshold makes an entity standing on a zone boundary flicker in and
out, and each flicker invents an enter and an exit. The table then reports
forty events where a human watching the footage counts one. So entry and exit
use two different thresholds: an entity is in a zone once more than
`enter_at` of its pixels are inside, and only leaves once fewer than
`exit_at` are. Between the two it keeps whatever state it had.

`min_frames` then drops intervals too short to be real, and `max_gap_frames`
bridges momentary dropouts inside one visit rather than splitting it in two.
"""

from __future__ import annotations

from dataclasses import dataclass

from .types import Track, VideoMeta
from .zones import ZoneRaster


@dataclass(frozen=True)
class EventParams:
    enter_at: float = 0.50
    """Fraction of an entity's pixels inside a zone that starts a visit."""

    exit_at: float = 0.30
    """Fraction below which the visit ends. Must be <= enter_at."""

    min_frames: int = 2
    """Visits shorter than this are dropped as noise."""

    max_gap_frames: int = 6
    """Frames an entity may be absent mid-visit without ending the visit."""

    def __post_init__(self) -> None:
        if not 0.0 < self.exit_at <= self.enter_at <= 1.0:
            raise ValueError("need 0 < exit_at <= enter_at <= 1")


@dataclass(frozen=True)
class ZoneEvent:
    entity_id: int
    label: str
    zone: str
    frame_start: int
    frame_end: int
    entered_at: float
    exited_at: float
    dwell_s: float
    peak_overlap: float
    closed: bool
    """False when the visit was still open at the last frame of the clip."""


def _intervals(
    frames: list[int], overlaps: list[float], params: EventParams
) -> list[tuple[int, int, float]]:
    """Hysteresis state machine -> (frame_start, frame_end, peak_overlap)."""
    out: list[tuple[int, int, float]] = []
    inside = False
    start = 0
    peak = 0.0
    last_in_frame = 0

    for frame, ov in zip(frames, overlaps):
        if not inside:
            if ov > params.enter_at:
                inside, start, peak, last_in_frame = True, frame, ov, frame
        else:
            if ov >= params.exit_at:
                peak = max(peak, ov)
                last_in_frame = frame
            elif frame - last_in_frame > params.max_gap_frames:
                out.append((start, last_in_frame, peak))
                inside, peak = False, 0.0
    if inside:
        out.append((start, last_in_frame, peak))
    return out


def compile_events(
    tracks: list[Track],
    zone_raster: ZoneRaster,
    meta: VideoMeta,
    params: EventParams | None = None,
) -> list[ZoneEvent]:
    params = params or EventParams()
    last_frame = max((t.last_frame for t in tracks), default=0)
    events: list[ZoneEvent] = []

    for track in tracks:
        frames = [d.frame for d in track.detections]
        # Per-frame overlap for every zone at once; masks are decoded once.
        per_zone: dict[str, list[float]] = {n: [] for n in zone_raster.masks}
        for d in track.detections:
            if d.mask is None:
                for n in per_zone:
                    per_zone[n].append(0.0)
                continue
            ov = zone_raster.overlap(d.mask)
            for n in per_zone:
                per_zone[n].append(ov.get(n, 0.0))

        for zone, overlaps in per_zone.items():
            for f0, f1, peak in _intervals(frames, overlaps, params):
                if f1 - f0 + 1 < params.min_frames:
                    continue
                t0 = meta.to_seconds(f0)
                t1 = meta.to_seconds(f1)
                events.append(
                    ZoneEvent(
                        entity_id=track.entity_id,
                        label=track.label,
                        zone=zone,
                        frame_start=f0,
                        frame_end=f1,
                        entered_at=round(t0, 3),
                        exited_at=round(t1, 3),
                        dwell_s=round(t1 - t0, 3),
                        peak_overlap=round(peak, 4),
                        closed=f1 < last_frame,
                    )
                )

    events.sort(key=lambda e: (e.entered_at, e.entity_id, e.zone))
    return events
