"""MOT17 ground truth, for measuring our code on real human motion.

The simulated clip proves the deterministic layers are self-consistent. It
cannot prove they survive contact with how people actually move: two
pedestrians passing at the same depth, a target lost behind a pillar for two
seconds, a group merging and separating. MOT17 is 30 frames per second of
exactly that, hand-annotated, with every identity known.

**Why this measures our code and not a detector.** The boxes here are human
annotations, so detection error is exactly zero by construction. Anything the
pipeline gets wrong on this input is ours. That is the whole point: on a real
clip the detector's mistakes and our mistakes arrive mixed together, and the
only way to size ours is to remove the detector.

**How the input is made realistic.** Feeding perfect, perfectly-identified
tracks in would measure nothing, because there would be nothing to repair.
Instead the fragmentation an upstream tracker produces is reproduced from the
annotations themselves: wherever a true identity disappears for longer than
the tracker's patience -- which is what an occlusion looks like -- it is given
a fresh observed id on return, exactly as SAM 3.1 or ByteTrack would. The
fragmentation is therefore derived from real occlusions in the footage rather
than injected at random.

What we then measure is how much of the true identity our reconciliation gets
back, and -- the number that matters more -- how often it wrongly welds two
different people together.
"""

from __future__ import annotations

import configparser
from dataclasses import dataclass
from pathlib import Path

from ..backends.synthetic import ellipse_mask
from ..types import Box, Detection, Track, VideoMeta

# gt.txt columns: frame, id, bb_left, bb_top, bb_width, bb_height, conf, class, visibility
PEDESTRIAN_CLASS = 1


@dataclass(frozen=True)
class GtBox:
    frame: int  # 0-indexed; the file is 1-indexed
    track_id: int
    box: Box
    visibility: float


@dataclass
class MotSequence:
    name: str
    meta: VideoMeta
    boxes: list[GtBox]

    @property
    def identities(self) -> int:
        return len({b.track_id for b in self.boxes})


def load_sequence(
    directory: str | Path, *, min_visibility: float = 0.0
) -> MotSequence:
    """Read one MOT17 sequence's annotations.

    Only rows flagged active (`conf == 1`) and of the pedestrian class are
    kept, which is what the benchmark's own evaluation counts. Boxes below
    `min_visibility` can be dropped: a target annotated at 10% visible is
    real, but no detector would find it, so including it measures the
    annotation rather than the pipeline.
    """
    directory = Path(directory)
    info = configparser.ConfigParser()
    info.read(directory / "seqinfo.ini")
    section = info["Sequence"]

    boxes: list[GtBox] = []
    with (directory / "gt" / "gt.txt").open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split(",")
            if len(parts) < 9:
                continue
            frame, track_id = int(parts[0]), int(parts[1])
            left, top, width, height = (float(p) for p in parts[2:6])
            conf, cls, visibility = int(parts[6]), int(parts[7]), float(parts[8])
            if conf != 1 or cls != PEDESTRIAN_CLASS or visibility < min_visibility:
                continue
            boxes.append(
                GtBox(
                    frame=frame - 1,  # the file is 1-indexed, our pipeline is not
                    track_id=track_id,
                    box=Box(left, top, left + width, top + height),
                    visibility=visibility,
                )
            )

    fps = float(section.get("frameRate", 30))
    length = int(section.get("seqLength", 0)) or (
        max((b.frame for b in boxes), default=0) + 1
    )
    meta = VideoMeta(
        path=str(directory),
        width=int(section.get("imWidth", 1920)),
        height=int(section.get("imHeight", 1080)),
        fps=fps,
        frame_count=length,
        duration_s=length / fps if fps else 0.0,
        sha256=f"mot:{directory.name}",
    )
    boxes.sort(key=lambda b: (b.frame, b.track_id))
    return MotSequence(name=section.get("name", directory.name), meta=meta, boxes=boxes)


def _detection(gt: GtBox, object_id: str, with_mask: bool) -> Detection:
    mask = None
    if with_mask:
        # MOT annotates boxes, not silhouettes. An inscribed ellipse is a
        # deliberately crude stand-in for a person's pixels -- good enough to
        # exercise the occupancy maths, and not good enough to quote zone
        # accuracy from. Mask-level claims belong to the SAM 3.1 run.
        cx = (gt.box.left + gt.box.right) / 2.0
        cy = (gt.box.top + gt.box.bottom) / 2.0
        mask = ellipse_mask(cx, cy, gt.box.width, gt.box.height)
    return Detection(
        frame=gt.frame,
        object_id=object_id,
        label="person",
        box=gt.box,
        mask=mask,
        confidence=gt.visibility,
    )


def true_tracks(sequence: MotSequence, *, with_mask: bool = False) -> list[Track]:
    """One track per annotated identity: the answer we are scored against."""
    grouped: dict[int, list[Detection]] = {}
    for gt in sequence.boxes:
        grouped.setdefault(gt.track_id, []).append(
            _detection(gt, f"gt:{gt.track_id}", with_mask)
        )
    tracks = [
        Track(
            entity_id=track_id,
            label="person",
            detections=sorted(dets, key=lambda d: d.frame),
            source_object_ids=[f"gt:{track_id}"],
        )
        for track_id, dets in sorted(grouped.items())
    ]
    tracks.sort(key=lambda t: (t.first_frame, t.entity_id))
    for new_id, track in enumerate(tracks, start=1):
        track.entity_id = new_id
    return tracks


def fragment(
    sequence: MotSequence, *, patience: int = 0, with_mask: bool = False
) -> tuple[list[Detection], int]:
    """Reproduce what an upstream tracker would hand us, from real occlusions.

    `patience` is how many frames the upstream tracker will keep a target
    alive while it is unseen. A gap longer than that ends the track, and the
    target gets a fresh id when it reappears -- the identity split this
    project exists to repair. `patience=0` is the harshest reading, where any
    missing frame breaks the track.

    Returns the detections and the number of observed ids, which is the
    entity count a pipeline with no reconciliation would report.
    """
    by_identity: dict[int, list[GtBox]] = {}
    for gt in sequence.boxes:
        by_identity.setdefault(gt.track_id, []).append(gt)

    detections: list[Detection] = []
    next_observed = 0
    for track_id, boxes in sorted(by_identity.items()):
        boxes.sort(key=lambda b: b.frame)
        next_observed += 1
        observed = next_observed
        previous = boxes[0].frame
        for gt in boxes:
            if gt.frame - previous > patience + 1:
                next_observed += 1
                observed = next_observed
            previous = gt.frame
            detections.append(_detection(gt, f"obs:{observed}", with_mask))

    detections.sort(key=lambda d: (d.frame, d.object_id))
    return detections, next_observed


def identity_errors(predicted: list[Track], sequence: MotSequence) -> dict[str, int]:
    """Count the two ways reconciliation can be wrong about who is who.

    A **merge** is two annotated people inside one predicted entity: the
    dangerous error, because it invents a continuous history that never
    happened and no later stage can detect it. A **fragment** is one annotated
    person spread across several predicted entities: the safe error, which
    only over-counts.
    """
    truth_of: dict[tuple[int, str], int] = {}
    for gt in sequence.boxes:
        truth_of[(gt.frame, f"{gt.box.left:.1f},{gt.box.top:.1f}")] = gt.track_id

    merged = 0
    covered: dict[int, set[int]] = {}
    for track in predicted:
        identities = set()
        for det in track.detections:
            key = (det.frame, f"{det.box.left:.1f},{det.box.top:.1f}")
            truth = truth_of.get(key)
            if truth is not None:
                identities.add(truth)
                covered.setdefault(truth, set()).add(track.entity_id)
        if len(identities) > 1:
            merged += len(identities) - 1

    fragmented = sum(max(0, len(ids) - 1) for ids in covered.values())
    return {
        "merges": merged,
        "fragments": fragmented,
        "true_identities": sequence.identities,
        "predicted_entities": len(predicted),
    }
