"""A simulated warehouse camera, with ground truth we know exactly.

This exists for three reasons, in order of importance:

1. **The deterministic layers can be validated in isolation.** When the
   tracker and the event compiler run against a scene whose true entity count
   and true zone visits are known by construction, any disagreement is our
   bug, not the model's. Measuring those two sources of error separately is
   the difference between "our numbers" and "SAM's numbers".
2. **The pipeline runs with no API key and no spend.** Every stage downstream
   of segmentation is exercised today.
3. **It injects the failure modes on purpose.** A real clip may or may not
   contain an identity split, an ID switch, or an entity loitering exactly on
   a zone boundary. This one always does, so the repair logic is never
   untested.

Masks are rendered as filled ellipses. The shape is not the point; the point
is that they are real pixel masks placed in frame coordinates, so the
occupancy maths downstream is the same maths the real clip will use.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field

from ..types import Box, Detection, Mask, VideoMeta


@dataclass
class Actor:
    """One true entity and the path it walks, in frame pixels."""

    truth_id: int
    label: str
    waypoints: list[tuple[int, float, float]]  # (frame, cx, cy)
    size: tuple[float, float]  # (width, height) in pixels
    visible: list[tuple[int, int]] = field(default_factory=list)  # inclusive spans
    rgb: tuple[int, int, int] = (128, 128, 128)  # the colour it is painted
    colour: str = "grey"  # and the name that colour must be read back as

    def position(self, frame: int) -> tuple[float, float] | None:
        if not any(a <= frame <= b for a, b in self.visible):
            return None
        pts = self.waypoints
        if frame <= pts[0][0]:
            return pts[0][1], pts[0][2]
        if frame >= pts[-1][0]:
            return pts[-1][1], pts[-1][2]
        for (f0, x0, y0), (f1, x1, y1) in zip(pts, pts[1:]):
            if f0 <= frame <= f1:
                t = 0.0 if f1 == f0 else (frame - f0) / (f1 - f0)
                return x0 + t * (x1 - x0), y0 + t * (y1 - y0)
        return pts[-1][1], pts[-1][2]


def ellipse_mask(cx: float, cy: float, w: float, h: float, res: int = 24) -> Mask:
    """A filled ellipse rendered at `res` resolution, placed at its bounds."""
    left, top = cx - w / 2.0, cy - h / 2.0
    box = Box(left, top, left + w, top + h)
    rows = cols = res
    data = bytearray(rows * cols)
    for r in range(rows):
        ny = (r + 0.5) / rows * 2.0 - 1.0
        for c in range(cols):
            nx = (c + 0.5) / cols * 2.0 - 1.0
            if nx * nx + ny * ny <= 1.0:
                data[r * cols + c] = 1
    return Mask(rows=rows, cols=cols, data=bytes(data), box=box)


@dataclass
class SyntheticScene:
    """A scripted scene: actors, plus the observation faults to inject."""

    width: int = 1280
    height: int = 720
    fps: float = 2.0
    frame_count: int = 240  # 120 s at 2 fps
    actors: list[Actor] = field(default_factory=list)
    id_switch_at: tuple[int, int, int] | None = None
    """(frame, truth_id_a, truth_id_b): swap the two object ids from here on."""


def default_scene() -> SyntheticScene:
    """A warehouse aisle with a loading dock, a pick aisle and a packing bench.

    Deliberately includes, in order: a clean long dwell; an entity that leaves
    the frame and returns (identity split); two entities that cross (ID
    switch); and one that loiters on the dock boundary (hysteresis).
    """
    actors = [
        # 1. Clean long dwell in the loading dock, then drives out.
        Actor(
            truth_id=1,
            label="vehicle",
            waypoints=[(0, 120, 180), (20, 240, 200), (90, 260, 210), (120, 900, 240)],
            size=(180.0, 120.0),
            visible=[(0, 120)],
            rgb=(196, 32, 32),
            colour="red",
        ),
        # 2. Steps out of frame at 60 and back at 68 -> a new object id for
        #    the same person. This is the identity split the stitcher exists
        #    for: 4 seconds at 2 fps, which is what "walked behind a rack"
        #    looks like in sampled footage.
        Actor(
            truth_id=2,
            label="person",
            waypoints=[(0, 700, 500), (60, 1270, 520), (68, 1270, 520), (160, 640, 480)],
            size=(55.0, 130.0),
            visible=[(0, 60), (68, 160)],
            rgb=(32, 72, 200),
            colour="blue",
        ),
        # 3 & 4. Cross at frame ~130 in the pick aisle -> ids swap.
        Actor(
            truth_id=3,
            label="person",
            waypoints=[(100, 300, 620), (160, 1000, 640)],
            size=(55.0, 130.0),
            visible=[(100, 160)],
            rgb=(36, 150, 60),
            colour="green",
        ),
        Actor(
            truth_id=4,
            label="person",
            waypoints=[(100, 1000, 660), (160, 300, 600)],
            size=(55.0, 130.0),
            visible=[(100, 160)],
            rgb=(226, 196, 40),
            colour="yellow",
        ),
        # 5. Loiters on the dock lower boundary -> occupancy hovers near 0.5.
        Actor(
            truth_id=5,
            label="person",
            waypoints=[(170, 300, 360), (190, 310, 372), (215, 300, 364), (239, 305, 368)],
            size=(55.0, 130.0),
            visible=[(170, 239)],
            rgb=(238, 240, 244),
            colour="white",
        ),
    ]
    return SyntheticScene(actors=actors, id_switch_at=(130, 3, 4))


class SyntheticBackend:
    """Renders a `SyntheticScene` into detections, with faults applied."""

    name = "synthetic"

    def __init__(self, clip: str = "synthetic-01", scene: SyntheticScene | None = None):
        self.clip = clip
        self.scene = scene or default_scene()

    def meta(self) -> VideoMeta:
        s = self.scene
        digest = hashlib.sha256(
            f"{self.clip}|{s.width}x{s.height}|{s.fps}|{s.frame_count}".encode()
        ).hexdigest()
        return VideoMeta(
            path=f"synthetic://{self.clip}",
            width=s.width,
            height=s.height,
            fps=s.fps,
            frame_count=s.frame_count,
            duration_s=s.frame_count / s.fps,
            sha256=digest,
        )

    def detect(self, labels=None) -> list[Detection]:
        s = self.scene
        wanted = set(labels) if labels else None
        # Observed object id per actor. A re-appearance gets a fresh one, the
        # way a tracker does when an object leaves the frame and comes back.
        next_obj = 1
        obj_of: dict[int, str] = {}
        seen_last: dict[int, int] = {}
        swap: dict[int, int] = {}
        if s.id_switch_at:
            _, a, b = s.id_switch_at
            swap = {a: b, b: a}

        dets: list[Detection] = []
        for frame in range(s.frame_count):
            for actor in s.actors:
                if wanted is not None and actor.label not in wanted:
                    continue
                pos = actor.position(frame)
                if pos is None:
                    continue
                tid = actor.truth_id
                if tid not in obj_of or frame - seen_last.get(tid, -99) > 1:
                    obj_of[tid] = str(next_obj)
                    next_obj += 1
                seen_last[tid] = frame

                id_source = swap[tid] if (
                    s.id_switch_at and frame >= s.id_switch_at[0] and tid in swap
                ) else tid
                object_id = obj_of.get(id_source, obj_of[tid])

                cx, cy = pos
                w, h = actor.size
                mask = ellipse_mask(cx, cy, w, h)
                conf = 0.80 + 0.15 * math.sin(frame * 0.3 + tid)
                dets.append(
                    Detection(
                        frame=frame,
                        object_id=object_id,
                        label=actor.label,
                        box=mask.box,
                        mask=mask,
                        confidence=round(min(0.99, max(0.5, conf)), 4),
                    )
                )
        return dets

    def truth_actors(self) -> list[Actor]:
        return list(self.scene.actors)

    def frame_reader(self) -> "SyntheticFrameReader":
        """Pixels for the scene, so colour is exercised without a video file.

        The painted actors are the ground truth the colour reader is scored
        against: each sits in the middle of its hue band, on a background
        chosen to be a colour no actor is, so a reader that leaked background
        pixels into the average would be caught rather than merely be a
        little off.
        """
        return SyntheticFrameReader(self.scene)


# A mid grey-brown the actors are all far from in hue. If the mask were
# leaking, every entity would drift toward this instead of its own colour.
BACKGROUND_RGB = (104, 96, 88)


class SyntheticFrameReader:
    """Renders one frame of a `SyntheticScene` as an RGB array."""

    def __init__(self, scene: SyntheticScene) -> None:
        self.scene = scene

    def frame(self, index: int):
        import numpy as np

        s = self.scene
        if not 0 <= index < s.frame_count:
            return None
        canvas = np.empty((s.height, s.width, 3), dtype=np.uint8)
        canvas[:, :] = BACKGROUND_RGB
        for actor in s.actors:
            pos = actor.position(index)
            if pos is None:
                continue
            cx, cy = pos
            w, h = actor.size
            # The same ellipse the mask describes, filled with the actor's
            # colour, so mask and pixels agree by construction.
            y0, y1 = int(cy - h / 2), int(cy + h / 2)
            x0, x1 = int(cx - w / 2), int(cx + w / 2)
            ys = np.arange(max(0, y0), min(s.height, y1 + 1))
            xs = np.arange(max(0, x0), min(s.width, x1 + 1))
            if not len(ys) or not len(xs):
                continue
            ny = ((ys + 0.5 - cy) / (h / 2.0))[:, None]
            nx = ((xs + 0.5 - cx) / (w / 2.0))[None, :]
            inside = (nx * nx + ny * ny) <= 1.0
            region = canvas[ys[0] : ys[-1] + 1, xs[0] : xs[-1] + 1]
            region[inside] = actor.rgb
        return canvas

    def close(self) -> None:
        return None
