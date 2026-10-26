"""Measuring the pipeline against ground truth, and separating the two errors.

A single accuracy number for a system like this is close to meaningless,
because two independent things can be wrong: the model may fail to see an
object, and our own code may mis-assemble what the model did see. They have
different fixes, so they are reported separately.

* **Tracking** is scored by matching predicted tracks to true actors frame by
  frame on box overlap, then counting the three failures that matter to a
  downstream count: missed detections (recall), spurious detections
  (precision), and identity errors -- an ID switch, where one predicted track
  changes which actor it follows, and a fragmentation, where one actor is
  split across several predicted tracks.

* **Zone events** are scored as caught / missed / invented against the events
  derived from the actors' true paths, with a match requiring the same zone,
  the same class, and a temporal overlap of at least `min_temporal_iou`.
  Counting events rather than frames is the honest unit, because an event is
  what the product actually reports to a user.

On the simulated clip the truth is exact, so these numbers measure our code
alone. On a real clip the truth is hand-labelled and the same functions are
used, so the two runs are directly comparable, and the gap between them is
what the model contributes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .backends.synthetic import SyntheticScene, ellipse_mask
from .events import EventParams, ZoneEvent, compile_events
from .tracker import reconcile
from .types import Detection, Track, VideoMeta
from .zones import ZoneRaster


# ----- ground truth from the simulated scene --------------------------------


def truth_detections(scene: SyntheticScene) -> list[Detection]:
    """The scene as it really is: one object id per actor, no injected faults."""
    out: list[Detection] = []
    for frame in range(scene.frame_count):
        for actor in scene.actors:
            pos = actor.position(frame)
            if pos is None:
                continue
            cx, cy = pos
            w, h = actor.size
            mask = ellipse_mask(cx, cy, w, h)
            out.append(
                Detection(
                    frame=frame,
                    object_id=f"truth:{actor.truth_id}",
                    label=actor.label,
                    box=mask.box,
                    mask=mask,
                )
            )
    return out


def truth_tracks(scene: SyntheticScene) -> list[Track]:
    """One track per actor, by construction rather than by reconciliation."""
    grouped: dict[int, list[Detection]] = {}
    for det in truth_detections(scene):
        truth_id = int(det.object_id.split(":")[1])
        grouped.setdefault(truth_id, []).append(det)
    return [
        Track(
            entity_id=truth_id,
            label=dets[0].label,
            detections=sorted(dets, key=lambda d: d.frame),
            source_object_ids=[f"truth:{truth_id}"],
        )
        for truth_id, dets in sorted(grouped.items())
    ]


def truth_events(
    scene: SyntheticScene,
    raster: ZoneRaster,
    meta: VideoMeta,
    params: EventParams | None = None,
) -> list[ZoneEvent]:
    return compile_events(truth_tracks(scene), raster, meta, params)


# ----- tracking metrics -----------------------------------------------------


@dataclass
class TrackingMetrics:
    true_detections: int
    predicted_detections: int
    matched: int
    missed: int
    spurious: int
    id_switches: int
    fragmentations: int
    true_entities: int
    predicted_entities: int
    iou_threshold: float

    @property
    def recall(self) -> float:
        return self.matched / self.true_detections if self.true_detections else 0.0

    @property
    def precision(self) -> float:
        return (
            self.matched / self.predicted_detections if self.predicted_detections else 0.0
        )

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def lines(self) -> list[str]:
        return [
            f"  entities        {self.predicted_entities} predicted vs "
            f"{self.true_entities} true",
            f"  detections      {self.predicted_detections} predicted vs "
            f"{self.true_detections} true",
            f"  precision       {self.precision:.3f}",
            f"  recall          {self.recall:.3f}",
            f"  F1              {self.f1:.3f}",
            f"  missed          {self.missed}",
            f"  spurious        {self.spurious}",
            f"  ID switches     {self.id_switches}",
            f"  fragmentations  {self.fragmentations}",
        ]

    def as_dict(self) -> dict:
        return {
            "true_detections": self.true_detections,
            "predicted_detections": self.predicted_detections,
            "matched": self.matched,
            "missed": self.missed,
            "spurious": self.spurious,
            "id_switches": self.id_switches,
            "fragmentations": self.fragmentations,
            "true_entities": self.true_entities,
            "predicted_entities": self.predicted_entities,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "iou_threshold": self.iou_threshold,
        }


def score_tracking(
    predicted: list[Track],
    truth: list[Track],
    *,
    iou_threshold: float = 0.3,
) -> TrackingMetrics:
    """Greedy per-frame matching on box IoU, then identity bookkeeping."""
    pred_by_frame: dict[int, list[tuple[int, Detection]]] = {}
    for track in predicted:
        for det in track.detections:
            pred_by_frame.setdefault(det.frame, []).append((track.entity_id, det))

    truth_by_frame: dict[int, list[tuple[int, Detection]]] = {}
    for track in truth:
        for det in track.detections:
            truth_by_frame.setdefault(det.frame, []).append((track.entity_id, det))

    matched = missed = spurious = 0
    # Which true actor each predicted track followed, in frame order.
    assignment: dict[int, list[int]] = {}
    # Which predicted tracks covered each true actor.
    coverage: dict[int, set[int]] = {}

    for frame in sorted(set(pred_by_frame) | set(truth_by_frame)):
        preds = list(pred_by_frame.get(frame, []))
        truths = list(truth_by_frame.get(frame, []))

        pairs: list[tuple[float, int, int]] = []
        for pi, (_, pd) in enumerate(preds):
            for ti, (_, td) in enumerate(truths):
                if pd.label != td.label:
                    continue
                iou = pd.box.iou(td.box)
                if iou >= iou_threshold:
                    pairs.append((iou, pi, ti))
        pairs.sort(key=lambda p: (-p[0], p[1], p[2]))

        used_p: set[int] = set()
        used_t: set[int] = set()
        for _iou, pi, ti in pairs:
            if pi in used_p or ti in used_t:
                continue
            used_p.add(pi)
            used_t.add(ti)
            matched += 1
            pred_id, truth_id = preds[pi][0], truths[ti][0]
            assignment.setdefault(pred_id, []).append(truth_id)
            coverage.setdefault(truth_id, set()).add(pred_id)

        missed += len(truths) - len(used_t)
        spurious += len(preds) - len(used_p)

    id_switches = sum(
        sum(1 for a, b in zip(seq, seq[1:]) if a != b) for seq in assignment.values()
    )
    fragmentations = sum(max(0, len(ids) - 1) for ids in coverage.values())

    return TrackingMetrics(
        true_detections=sum(len(v) for v in truth_by_frame.values()),
        predicted_detections=sum(len(v) for v in pred_by_frame.values()),
        matched=matched,
        missed=missed,
        spurious=spurious,
        id_switches=id_switches,
        fragmentations=fragmentations,
        true_entities=len(truth),
        predicted_entities=len(predicted),
        iou_threshold=iou_threshold,
    )


# ----- event metrics --------------------------------------------------------


@dataclass
class EventMetrics:
    caught: int
    missed: int
    invented: int
    true_total: int
    predicted_total: int
    dwell_errors: list[float] = field(default_factory=list)
    min_temporal_iou: float = 0.5

    @property
    def recall(self) -> float:
        return self.caught / self.true_total if self.true_total else 0.0

    @property
    def precision(self) -> float:
        return self.caught / self.predicted_total if self.predicted_total else 0.0

    @property
    def mean_abs_dwell_error_s(self) -> float:
        return sum(self.dwell_errors) / len(self.dwell_errors) if self.dwell_errors else 0.0

    @property
    def max_abs_dwell_error_s(self) -> float:
        return max(self.dwell_errors) if self.dwell_errors else 0.0

    def lines(self) -> list[str]:
        return [
            f"  events          {self.predicted_total} reported vs "
            f"{self.true_total} true",
            f"  caught          {self.caught}",
            f"  missed          {self.missed}",
            f"  invented        {self.invented}",
            f"  precision       {self.precision:.3f}",
            f"  recall          {self.recall:.3f}",
            f"  dwell error     {self.mean_abs_dwell_error_s:.2f}s mean, "
            f"{self.max_abs_dwell_error_s:.2f}s worst",
        ]

    def as_dict(self) -> dict:
        return {
            "caught": self.caught,
            "missed": self.missed,
            "invented": self.invented,
            "true_total": self.true_total,
            "predicted_total": self.predicted_total,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "mean_abs_dwell_error_s": round(self.mean_abs_dwell_error_s, 3),
            "max_abs_dwell_error_s": round(self.max_abs_dwell_error_s, 3),
            "min_temporal_iou": self.min_temporal_iou,
        }


def _temporal_iou(a: ZoneEvent, b: ZoneEvent) -> float:
    inter = max(0.0, min(a.exited_at, b.exited_at) - max(a.entered_at, b.entered_at))
    union = max(a.exited_at, b.exited_at) - min(a.entered_at, b.entered_at)
    if union <= 0:
        # Two instantaneous events at the same moment.
        return 1.0 if a.entered_at == b.entered_at else 0.0
    return inter / union


def score_events(
    predicted: list[ZoneEvent],
    truth: list[ZoneEvent],
    *,
    min_temporal_iou: float = 0.5,
) -> EventMetrics:
    """Match on zone, class and temporal overlap, ignoring entity numbering.

    Entity ids are deliberately not required to agree. A clip whose entity 3
    is the truth's entity 4 is not wrong about the operation: the zone, the
    time and the duration are what anyone acts on. Identity accuracy is
    reported separately, by `score_tracking`.
    """
    remaining = list(truth)
    caught = 0
    dwell_errors: list[float] = []

    for pred in sorted(predicted, key=lambda e: e.entered_at):
        best: tuple[float, int] | None = None
        for i, candidate in enumerate(remaining):
            if candidate.zone != pred.zone or candidate.label != pred.label:
                continue
            score = _temporal_iou(pred, candidate)
            if score >= min_temporal_iou and (best is None or score > best[0]):
                best = (score, i)
        if best is not None:
            matched = remaining.pop(best[1])
            caught += 1
            dwell_errors.append(abs(pred.dwell_s - matched.dwell_s))

    return EventMetrics(
        caught=caught,
        missed=len(remaining),
        invented=len(predicted) - caught,
        true_total=len(truth),
        predicted_total=len(predicted),
        dwell_errors=dwell_errors,
        min_temporal_iou=min_temporal_iou,
    )


# ----- compression, as one line not a feature -------------------------------


def compression_ratio(meta: VideoMeta, db_bytes: int) -> dict[str, float]:
    """How much smaller the derived event table is than the footage.

    Reported as a single measured figure rather than a capability. The source
    size is the file on disk, or an estimate from a 4 Mbps stream when the
    clip is simulated and has no file.
    """
    source_bytes = 0.0
    if not meta.path.startswith("synthetic://"):
        from pathlib import Path

        p = Path(meta.path)
        if p.exists():
            source_bytes = float(p.stat().st_size)
    if source_bytes <= 0:
        source_bytes = meta.duration_s * 4_000_000 / 8  # 4 Mbps reference stream
    return {
        "source_bytes": source_bytes,
        "derived_bytes": float(db_bytes),
        "ratio": source_bytes / db_bytes if db_bytes else 0.0,
    }


# ----- the whole report -----------------------------------------------------


@dataclass
class EvalReport:
    clip: str
    tracking: TrackingMetrics
    events: EventMetrics
    compression: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def lines(self) -> list[str]:
        out = [f"eval: {self.clip}", "", "tracking"]
        out += self.tracking.lines()
        out += ["", "zone events"]
        out += self.events.lines()
        if self.compression:
            ratio = self.compression.get("ratio", 0.0)
            out += [
                "",
                "derived data size",
                f"  source          {self.compression['source_bytes']:,.0f} bytes",
                f"  event table     {self.compression['derived_bytes']:,.0f} bytes",
                f"  ratio           {ratio:,.0f}x smaller",
            ]
        if self.notes:
            out += [""] + [f"note: {n}" for n in self.notes]
        return out

    def as_dict(self) -> dict:
        return {
            "clip": self.clip,
            "tracking": self.tracking.as_dict(),
            "events": self.events.as_dict(),
            "compression": {k: round(v, 2) for k, v in self.compression.items()},
            "notes": self.notes,
        }


def evaluate_synthetic(
    detections: list[Detection],
    scene: SyntheticScene,
    raster: ZoneRaster,
    meta: VideoMeta,
    clip: str,
    *,
    event_params: EventParams | None = None,
    db_bytes: int = 0,
) -> EvalReport:
    """Score a run of the simulated clip end to end."""
    recon = reconcile(detections)
    predicted_events = compile_events(recon.tracks, raster, meta, event_params)
    truth_ev = truth_events(scene, raster, meta, event_params)

    notes: list[str] = []
    if scene.id_switch_at is not None:
        frame, a, b = scene.id_switch_at
        notes.append(
            f"the scene swaps object ids for actors {a} and {b} at frame {frame}, "
            f"while they are adjacent; an ID switch between two objects that are "
            f"touching produces no geometric discontinuity, so the tracker cannot "
            f"see it and does not claim to"
        )

    return EvalReport(
        clip=clip,
        tracking=score_tracking(recon.tracks, truth_tracks(scene)),
        events=score_events(predicted_events, truth_ev),
        compression=compression_ratio(meta, db_bytes) if db_bytes else {},
        notes=notes,
    )
