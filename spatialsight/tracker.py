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
    """How long an entity may be unseen and still count as the same entity.

    In frames, so it only means a fixed duration once paired with a frame
    rate: 12 frames is 6 seconds at 2 fps and 0.4 seconds at 30. Callers
    should derive it from the clip (see `for_fps`) rather than take this
    default, which suits the 2 fps simulated clip.
    """

    @classmethod
    def for_fps(cls, fps: float, seconds: float = 1.0, **overrides) -> "TrackerParams":
        """Parameters whose gap window is a duration rather than a frame count."""
        return cls(max_gap_frames=max(1, int(round(fps * seconds))), **overrides)

    stitch_min_iou: float = 0.50
    """Box overlap that is conclusive on its own, bypassing the motion test.

    Measured on MOT17: at 0.10 this shortcut alone caused 21 wrong merges,
    because two different people standing near each other overlap 10% of
    their boxes without difficulty. Raising it to 0.50 removed a third of all
    merges. Above 0.50 nothing changes, so almost no genuine reconnection
    depends on partial overlap -- it is either decisive or irrelevant.
    """

    stitch_slack: float = 0.10
    """Base allowance around the prediction, in box diagonals.

    Covers detection jitter on a stationary target. Tuned on MOT17, where
    0.35 produced nearly twice the merges of 0.10 for a 3% gain in
    fragmentation recovery -- a bad trade, because a merge invents a history
    and a fragment only duplicates one.
    """

    stitch_slack_motion: float = 1.0
    """Extra allowance per pixel the target was predicted to travel.

    The tolerance has to scale with **motion**, not with box size. Scaling it
    by the box put the radius near 280px on 1080p pedestrians, which in a
    crowd is wider than the spacing between people, so the nearest candidate
    was routinely the wrong person. A target drifting 5px per frame across a
    10-frame gap could plausibly be 50px off its prediction, not 280.
    """

    stitch_slack_cap: float = 2.0
    """Ceiling on the allowance, in box diagonals.

    Without a ceiling the allowance grows linearly with the gap, and on a
    1920x1080 sequence a 60-frame gap put the search radius at 1,476 px --
    most of the scene. Past a point a prediction is not degraded but
    worthless, and the honest response is to refuse the stitch rather than
    accept whoever is nearest.
    """

    ambiguity_ratio: float = 2.5
    """How much better the best candidate must be than the runner-up.

    In a crowd several people fit a prediction. Picking the closest is a coin
    flip that produces a merge -- two real people fused into one history,
    which nothing downstream can detect. When two candidates are within this
    ratio the reconnection is refused, leaving two entities instead. That is
    the safe error: it over-counts rather than inventing a history.
    """

    stitch_max_size_ratio: float = 2.5
    """Reject a stitch when the two boxes differ in area by more than this."""

    split_min_jump: float = 3.0
    """Centroid jump, in box diagonals, that marks an ID switch."""

    min_detections: int = 3
    """Frames an entity must be seen in before it counts as an entity.

    A detector run over real footage emits brief false positives -- a shadow,
    a bag, half a reflection -- that live for a frame or two. Each one
    otherwise becomes an entity with its own row, and on a real clip that
    inflated the count from roughly 35 to 58 while contributing 94 of 5,771
    detections. Three sightings with consistent geometry is the least that
    distinguishes an object from noise; raise it for a noisier detector.
    """


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

    @property
    def dropped(self) -> int:
        return sum(1 for r in self.repairs if r.kind == "dropped")

    @property
    def ambiguous(self) -> int:
        return sum(1 for r in self.repairs if r.kind == "ambiguous")


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

    return _residual(tail_segment, head) <= _allowance(tail_segment, head, params)


def _residual(tail_segment: list[Detection], head: Detection) -> float:
    """How far the candidate landed from where the tail was heading."""
    tail = tail_segment[-1]
    gap = head.frame - tail.frame
    vx, vy = _velocity(tail_segment)
    tx, ty = tail.box.centroid
    hx, hy = head.box.centroid
    return hypot(hx - (tx + vx * gap), hy - (ty + vy * gap))


def _allowance(
    tail_segment: list[Detection], head: Detection, params: TrackerParams
) -> float:
    """How far from the prediction a candidate may land, in pixels.

    Widens with the gap, because a prediction degrades, but is capped: past
    the ceiling the prediction carries no information and a wider radius only
    buys false matches.
    """
    tail = tail_segment[-1]
    gap = head.frame - tail.frame
    vx, vy = _velocity(tail_segment)
    travelled = hypot(vx, vy) * gap
    diagonal = _diag(tail.box)
    allowance = params.stitch_slack * diagonal + params.stitch_slack_motion * travelled
    return min(allowance, params.stitch_slack_cap * diagonal)


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
            # Score every candidate that passes the gate, then insist the
            # winner is clearly better than the runner-up.
            scored: list[tuple[float, int]] = []
            for j, cand in enumerate(segments):
                if j in used:
                    continue
                head = cand[1][0]
                if not _can_stitch(tail_segment, head, params):
                    continue
                scored.append((_residual(tail_segment, head), j))
            if not scored:
                break
            scored.sort()
            if len(scored) > 1:
                best_cost, runner_up = scored[0][0], scored[1][0]
                # Guard the degenerate case where the prediction lands exactly
                # on two candidates: a zero-distance tie is maximally ambiguous.
                if best_cost <= 0.0 and runner_up <= 0.0:
                    break
                if best_cost * params.ambiguity_ratio > runner_up:
                    repairs.append(
                        Repair(
                            "ambiguous",
                            f"refused to reconnect {chain[-1][0]} at frame "
                            f"{tail.frame}: two candidates fit the prediction "
                            f"({best_cost:.0f}px vs {runner_up:.0f}px)",
                        )
                    )
                    break
            j = scored[0][1]
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
        if len(dets) < params.min_detections:
            repairs.append(
                Repair(
                    "dropped",
                    f"object {sorted({oid for oid, _ in chain})} seen in only "
                    f"{len(dets)} frame(s), below the {params.min_detections}-frame "
                    f"minimum",
                )
            )
            continue
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
