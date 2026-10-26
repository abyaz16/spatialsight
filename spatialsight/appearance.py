"""What an entity looks like, read from the pixels its own mask covers.

This is the one attribute that a box-based tracker cannot honestly produce.
A box around a person on a grey floor is perhaps half floor; averaging it
gives you the colour of the room. A mask is the object and nothing else, so
the average inside it is the object's own colour. Pixel-level segmentation
earns its cost here as directly as it does in zone occupancy.

Three decisions worth defending:

**The median across frames, not the mean.** A single frame can catch an
entity half-shadowed, blown out by a light, or mid-occlusion behind
something the mask does not know about. Averaging folds those in; taking the
median of several per-frame readings discards them.

**The hex value is stored next to the name.** Colour naming is a lossy
judgement with hard cases -- maroon is red or brown depending on the light,
and red under sodium lighting is not red to a camera. A query that wants to
be careful can work on the numbers; one that wants to be readable uses the
name. Storing only the name would throw away the evidence for it.

**A colourless reading is reported as grey, not guessed.** Naming a hue
from a nearly colourless pixel average is inventing data. Dark, washed-out
and genuinely grey objects all come back on the neutral axis, and the caller
can tell because the name says so.
"""

from __future__ import annotations

import colorsys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol, Sequence

import numpy as np

from .types import Detection, Mask, Track

# How many frames of an entity's life to read. Enough that one bad frame
# cannot carry the median; few enough that a 900-frame clip stays quick.
SAMPLES_PER_TRACK = 7

# Below this value everything reads as black whatever its hue.
MIN_VALUE = 0.16
# Above this value, with no colour in it, it is white rather than grey.
WHITE_VALUE = 0.80

# The hue gate is absolute chroma -- max(r,g,b) - min(r,g,b) on the 0..255
# scale -- and not saturation, which was the first thing tried and was wrong.
# Saturation is a ratio, so it stays large on near-black pixels where the
# channels differ by two or three levels of sensor noise. On real footage
# that named #393440 "purple" and #2c2a30 "grey" on alternate frames, and the
# flicker then showed up as 15 of 41 entities apparently changing colour
# mid-life. Chroma is the absolute amount of colour present, so a dark grey
# is a dark grey however the noise falls.
MIN_CHROMA = 36.0

# Hue is in degrees. Upper bound of each band, walking the wheel from red.
HUE_BANDS: tuple[tuple[float, str], ...] = (
    (15.0, "red"),
    (45.0, "orange"),
    (70.0, "yellow"),
    (170.0, "green"),
    (195.0, "cyan"),
    (255.0, "blue"),
    (290.0, "purple"),
    (345.0, "pink"),
    (360.0, "red"),
)


class FrameReader(Protocol):
    """Random access to a clip's pixels, as RGB uint8 arrays."""

    def frame(self, index: int) -> np.ndarray | None: ...

    def close(self) -> None: ...


# Below this agreement between an entity's own frames, its colour changed
# during its life -- which a single object's colour does not do.
STABLE_ABOVE = 0.75

# Black, grey and white are one family for the purpose of judging stability.
# A dark coat reads black in shade and grey in sun, and on real footage that
# lightness wobble accounted for most of the entities first flagged as
# changing colour -- 15 of 41, almost all of them flipping black/grey and
# nothing else. Only a change of hue is evidence that the pixels belong to a
# different object; a change of brightness is evidence about the weather.
NEUTRAL = frozenset({"black", "grey", "white"})


def colour_family(name: str) -> str:
    return "neutral" if name in NEUTRAL else name


@dataclass(frozen=True)
class Appearance:
    """One entity's colour, as a name and as the evidence for it."""

    colour: str
    rgb: tuple[int, int, int]
    chroma: float
    samples: int
    stability: float = 1.0
    """Fraction of this entity's sampled frames that agreed on the colour.

    This is the useful part, and it was a surprise. The synthetic scene
    contains two actors who swap identities while adjacent, and the eval has
    always reported those two ID switches as undetectable: two touching
    objects exchanging ids produce no jump in position or size, so geometry
    has nothing to notice. Colour does. An entity that is green for the first
    half of its life and yellow for the second is not one entity, and this
    number says so without the tracker needing to understand why.
    """

    @property
    def hex(self) -> str:
        return "#{:02x}{:02x}{:02x}".format(*self.rgb)

    @property
    def certain(self) -> bool:
        """Whether the name rests on a real hue rather than the neutral axis."""
        return self.chroma >= MIN_CHROMA

    @property
    def stable(self) -> bool:
        """False when this entity's own frames disagree about its colour."""
        return self.stability >= STABLE_ABOVE


def name_colour(rgb: Sequence[float]) -> tuple[str, float]:
    """An (approximate) colour name, and the chroma it was judged on."""
    channels = [max(0.0, min(255.0, float(c))) for c in rgb]
    chroma = max(channels) - min(channels)
    r, g, b = (c / 255.0 for c in channels)
    hue, _saturation, value = colorsys.rgb_to_hsv(r, g, b)
    if value < MIN_VALUE:
        return "black", chroma
    if chroma < MIN_CHROMA:
        return ("white" if value >= WHITE_VALUE else "grey"), chroma
    degrees = hue * 360.0
    for upper, name in HUE_BANDS:
        if degrees < upper:
            # A dark but genuinely coloured object is that colour, not black.
            return name, chroma
    return "red", chroma


def mask_pixels(frame: np.ndarray, mask: Mask) -> np.ndarray:
    """The frame pixels this mask marks, as an (n, 3) array.

    The mask is stored at its own resolution over its own box, so each frame
    pixel inside the box is tested against the mask cell it falls in. Nearest
    neighbour, not interpolation: a mask cell is in or out, and softening the
    edge would pull in exactly the background pixels this function exists to
    exclude.
    """
    height, width = frame.shape[:2]
    left = max(0, int(np.floor(mask.box.left)))
    top = max(0, int(np.floor(mask.box.top)))
    right = min(width - 1, int(np.ceil(mask.box.right)))
    bottom = min(height - 1, int(np.ceil(mask.box.bottom)))
    if right < left or bottom < top:
        return np.empty((0, 3), dtype=np.uint8)

    patch = frame[top : bottom + 1, left : right + 1]
    if patch.size == 0:
        return np.empty((0, 3), dtype=np.uint8)

    box_w = max(1e-6, mask.box.right - mask.box.left)
    box_h = max(1e-6, mask.box.bottom - mask.box.top)
    ys = np.arange(top, bottom + 1, dtype=np.float64) + 0.5
    xs = np.arange(left, right + 1, dtype=np.float64) + 0.5
    rows = np.clip(
        ((ys - mask.box.top) / box_h * mask.rows).astype(np.int64), 0, mask.rows - 1
    )
    cols = np.clip(
        ((xs - mask.box.left) / box_w * mask.cols).astype(np.int64), 0, mask.cols - 1
    )

    grid = np.frombuffer(mask.data, dtype=np.uint8).reshape(mask.rows, mask.cols)
    keep = grid[np.ix_(rows, cols)].astype(bool)
    if not keep.any():
        return np.empty((0, 3), dtype=np.uint8)
    return patch[keep].reshape(-1, patch.shape[2])[:, :3]


def _spread(detections: Sequence[Detection], count: int) -> list[Detection]:
    """Up to `count` detections spread evenly over the entity's life."""
    usable = [d for d in detections if d.mask is not None]
    if len(usable) <= count:
        return usable
    step = (len(usable) - 1) / (count - 1)
    return [usable[int(round(i * step))] for i in range(count)]


def describe_track(
    track: Track,
    reader: FrameReader,
    *,
    samples: int = SAMPLES_PER_TRACK,
    min_pixels: int = 12,
) -> Appearance | None:
    """One entity's colour, or None when its pixels could not be read.

    None is a real answer and is stored as NULL rather than guessed at. A
    clip with no pixels available -- a cached run with the video gone, a
    synthetic scene -- leaves every entity uncoloured, and a query for a red
    box then returns nothing instead of something wrong.
    """
    readings: list[np.ndarray] = []
    for detection in _spread(track.detections, samples):
        frame = reader.frame(detection.frame)
        if frame is None:
            continue
        pixels = mask_pixels(frame, detection.mask)  # type: ignore[arg-type]
        if len(pixels) >= min_pixels:
            readings.append(np.median(pixels.astype(np.float64), axis=0))

    if not readings:
        return None
    rgb = np.median(np.vstack(readings), axis=0)
    colour, chroma = name_colour(rgb)
    # Agreement is measured on the families, not the numbers: two readings of
    # a shadowed and a lit red object differ a long way in RGB and not at all
    # in what the object is. Only a change of hue family is evidence of a
    # problem.
    family = colour_family(colour)
    agreeing = sum(
        1 for r in readings if colour_family(name_colour(r)[0]) == family
    )
    return Appearance(
        colour=colour,
        rgb=tuple(int(round(c)) for c in rgb),  # type: ignore[arg-type]
        chroma=round(float(chroma), 2),
        samples=len(readings),
        stability=round(agreeing / len(readings), 4),
    )


def describe_tracks(
    tracks: Iterable[Track], reader: FrameReader | None
) -> dict[int, Appearance]:
    """Colour per entity_id, skipping any that could not be read."""
    if reader is None:
        return {}
    out: dict[int, Appearance] = {}
    try:
        for track in tracks:
            found = describe_track(track, reader)
            if found is not None:
                out[track.entity_id] = found
    finally:
        reader.close()
    return out


class VideoFrameReader:
    """Frames from a video file, read in whatever order they are asked for.

    Seeking per frame is slower than a linear decode but the sampled frames
    are far apart, so a linear pass would decode the entire clip to use seven
    frames of it.
    """

    def __init__(self, path: str | Path) -> None:
        import cv2

        self._cv2 = cv2
        self.path = Path(path)
        self._cap = cv2.VideoCapture(str(self.path))
        if not self._cap.isOpened():
            raise FileNotFoundError(f"cannot open video: {self.path}")

    def frame(self, index: int) -> np.ndarray | None:
        self._cap.set(self._cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, bgr = self._cap.read()
        if not ok or bgr is None:
            return None
        return bgr[:, :, ::-1]  # OpenCV decodes BGR; everything here is RGB

    def close(self) -> None:
        self._cap.release()


def reader_for(meta_path: str) -> FrameReader | None:
    """A reader for a clip's pixels, or None when there are none to read.

    A synthetic clip's path is a `synthetic://` URL with no file behind it,
    and a cached run may name a video that has since been moved. Both are
    ordinary outcomes, not errors: the colour column is simply left empty.
    """
    if not meta_path or "://" in meta_path:
        return None
    path = Path(meta_path)
    if not path.exists():
        return None
    try:
        return VideoFrameReader(path)
    except (ImportError, FileNotFoundError):
        return None
