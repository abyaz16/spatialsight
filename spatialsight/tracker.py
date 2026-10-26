"""ID reconciliation over SAM's per-frame object ids.

SAM 3.1 tracks objects through a video and keeps an identity per object, but a
sampled, real-world clip still produces two failures that corrupt any
downstream count: an entity that leaves and returns gets a fresh id (a *split*
identity), and two entities that cross can swap ids (an *ID switch*).

Neither is fixable by a bigger model call. Both are fixable with geometry, so
this layer is deliberately our own deterministic code:

  * **stitch** -- join two ids when one ends and the other begins within
    `max_gap_frames`, and the boxes agree in position and size. This is what
    turns "11 forklifts" back into "3 forklifts that came and went".
  * **split** -- cut one id into two entities when its box teleports: a jump
    far larger than its own size, with no overlap. Keeping them joined would
    invent a dwell that never happened.

Every decision is recorded on the `Track`, so the eval can report how many
repairs were made and what they cost.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import hypot

from .types import Box, Detection, Track


@dataclass(frozen=True)
class TrackerParams:
    max_gap_frames: int = 12
    """How long an entity may be unseen and still count as the same entity."""

    stitch_min_iou: float = 0.10
    """Box overlap across the gap below which a stitch needs the distance test."""

    stitch_slack: float = 0.75
    """Allowed error around the predicted position, in box diagonals."""

    stitch_max_size_ratio: float = 2.5
    """Reject a stitch when the two boxes differ in area by more than this."""

    split_min_jump: float = 3.0
    """Centroid jump, in box diagonals, that marks an ID switch."""


@dataclass
class Repair:
    kind: str  # "stitch" | "split"
    detail: str


@dataclass
class Reconciliation:
    tracks: list[Track]
    repairs: list[Repair] = field(default_factory=list)

    @property
    def stitches(self) -> int:
        return sum(1 for r in self.repairs if r.kind == "stitch")

    @property
    def splits(self) -> int:
        return sum(1 for r in self.repairs if r.kind == "split")


def _diag(box: Box) -> float:
    return max(1.0, hypot(box.width, box.height))


def _segment(dets: list[Detection], params: TrackerParams) -> list[list[Detection]]:
    """Cut one object id wherever its box teleports."""
    dets = sorted(dets, key=lambda d: d.frame)
    segments: list[list[Detection]] = [[dets[0]]]
    for prev, cur in zip(dets, dets[1:]):
        px, py = prev.box.centroid
        cx, cy = cur.box.centroid
        jump = hypot(cx - px, cy - py) / _diag(prev.box)
        if jump > params.split_min_jump and prev.box.iou(cur.box) == 0.0:
            segments.append([cur])
        else:
            segments[-1].append(cur)
    return segments


def _velocity(segment: list[Detection], window: int = 5) -> tuple[float, float]:
    """Average pixels-per-frame over the tail of a segment."""
    recent = segment[-window:]
    if len(recent) < 2:
        return 0.0, 0.0
    (x0, y0), (x1, y1) = recent[0].box.centroid, recent[-1].box.centroid
    span = recent[-1].frame - recent[0].frame
    if span <= 0:
        return 0.0, 0.0
    return (x1 - x0) / span, (y1 - y0) / span


def _can_stitch(
    tail_segment: list[Detection], head: Detection, params: TrackerParams
) -> bool:
    """Would `head` plausibly continue `tail_segment`?

    The test is a prediction, not a distance. Extrapolating the tail's own
    motion across the gap and asking whether the candidate landed near where
    the entity was heading is what separates "the same forklift, reacquired"
    from "a different person who happened to appear 5 seconds later across
    the room". A plain drift radius cannot tell those apart: it accepts any
    candidate inside a circle, which at a 10-frame gap is most of the frame.
    """
    tail = tail_segment[-1]
    gap = head.frame - tail.frame
    if gap <= 0 or gap > params.max_gap_frames:
        return False
    if tail.label != head.label:
        return False

    a, b = tail.box.area, head.box.area
    if a > 0 and b > 0 and max(a, b) / min(a, b) > params.stitch_max_size_ratio:
        return False

    # An overlapping box across a short gap is already conclusive.
    if tail.box.iou(head.box) >= params.stitch_min_iou:
        return True

    vx, vy = _velocity(tail_segment)
    tx, ty = tail.box.centroid
    px, py = tx + vx * gap, ty + vy * gap
    hx, hy = head.box.centroid
    residual = hypot(hx - px, hy - py)
    # Tolerance grows with the gap, because a prediction degrades over time.
    allowance = params.stitch_slack * _diag(tail.box) * (1.0 + 0.1 * gap)
    return residual <= allowance


def reconcile(
    detections: list[Detection], params: TrackerParams | None = None
) -> Reconciliation:
    """Turn per-frame detections into entity tracks, repairing splits and switches."""
    params = params or TrackerParams()
    repairs: list[Repair] = []

    by_id: dict[str, list[Detection]] = {}
    for d in detections:
        by_id.setdefault(d.object_id, []).append(d)

    # 1. Cut ID switches.
    segments: list[tuple[str, list[Detection]]] = []
    for object_id, dets in by_id.items():
        parts = _segment(dets, params)
        if len(parts) > 1:
            repairs.append(
                Repair("split", f"object {object_id} cut into {len(parts)} at a teleport")
            )
        for part in parts:
            segments.append((object_id, part))

    # 2. Greedily stitch segments that continue one another, earliest first.
    segments.sort(key=lambda s: s[1][0].frame)
    chains: list[list[tuple[str, list[Detection]]]] = []
    used: set[int] = set()
    for i, seg in enumerate(segments):
        if i in used:
            continue
        chain = [seg]
        used.add(i)
        while True:
            tail_segment = chain[-1][1]
            tail = tail_segment[-1]
            best: tuple[float, int] | None = None
            for j, cand in enumerate(segments):
                if j in used:
                    continue
                head = cand[1][0]
                if not _can_stitch(tail_segment, head, params):
                    continue
                # Among the candidates that pass the gate, take the one
                # closest to where the entity was actually heading.
                vx, vy = _velocity(tail_segment)
                gap = head.frame - tail.frame
                px, py = tail.box.centroid
                hx, hy = head.box.centroid
                cost = hypot(hx - (px + vx * gap), hy - (py + vy * gap)) + gap
                if best is None or cost < best[0]:
                    best = (cost, j)
            if best is None:
                break
            j = best[1]
            used.add(j)
            repairs.append(
                Repair(
                    "stitch",
                    f"object {segments[j][0]} joined to {chain[-1][0]} "
                    f"across frames {tail.frame}->{segments[j][1][0].frame}",
                )
            )
            chain.append(segments[j])
        chains.append(chain)

    tracks: list[Track] = []
    for entity_id, chain in enumerate(chains, start=1):
        dets = sorted((d for _, part in chain for d in part), key=lambda d: d.frame)
        tracks.append(
            Track(
                entity_id=entity_id,
                label=dets[0].label,
                detections=dets,
                source_object_ids=sorted({oid for oid, _ in chain}),
            )
        )
    tracks.sort(key=lambda t: (t.first_frame, t.entity_id))
    for new_id, t in enumerate(tracks, start=1):
        t.entity_id = new_id
    return Reconciliation(tracks=tracks, repairs=repairs)
