"""Zones: hand-drawn polygons, rasterised once, intersected with masks.

Occupancy is measured in pixels, not box centroids. A forklift whose box
corner clips a zone is not in the zone; a forklift whose mask is half inside
it is. That distinction is the whole reason this project segments at all, so
it is computed exactly rather than approximated from boxes.

The polygon rasteriser is an even-odd scanline fill written here on purpose:
it keeps the project free of an OpenCV dependency, and makes the one number
every event depends on -- the fraction of an entity's pixels inside a zone --
something we can point at and defend.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .types import Mask


@dataclass(frozen=True)
class Zone:
    name: str
    polygon: tuple[tuple[float, float], ...]

    def __post_init__(self) -> None:
        if len(self.polygon) < 3:
            raise ValueError(f"zone {self.name!r} needs at least 3 points")


@dataclass(frozen=True)
class ZoneSet:
    """Zones for one camera, plus the frame size they were drawn against."""

    camera: str
    width: int
    height: int
    zones: tuple[Zone, ...]

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(z.name for z in self.zones)


def load_zones(path: str | Path) -> ZoneSet:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    zones = tuple(
        Zone(name=z["name"], polygon=tuple((float(x), float(y)) for x, y in z["polygon"]))
        for z in raw["zones"]
    )
    if len(set(z.name for z in zones)) != len(zones):
        raise ValueError("zone names must be unique")
    return ZoneSet(
        camera=raw["camera"],
        width=int(raw["frame"]["width"]),
        height=int(raw["frame"]["height"]),
        zones=zones,
    )


def rasterise(zone: Zone, width: int, height: int) -> np.ndarray:
    """Return a (height, width) bool array: True where the polygon covers a pixel.

    Even-odd scanline fill, sampling each pixel at its centre. A pixel belongs
    to the zone when the polygon encloses the point (x+0.5, y+0.5).
    """
    grid = np.zeros((height, width), dtype=bool)
    pts = zone.polygon
    n = len(pts)

    for y in range(height):
        yc = y + 0.5
        crossings: list[float] = []
        for i in range(n):
            x1, y1 = pts[i]
            x2, y2 = pts[(i + 1) % n]
            if y1 == y2:
                continue  # horizontal edge contributes no crossing
            # half-open in y so a vertex shared by two edges counts once
            lo, hi = (y1, y2) if y1 < y2 else (y2, y1)
            if not (lo <= yc < hi):
                continue
            t = (yc - y1) / (y2 - y1)
            crossings.append(x1 + t * (x2 - x1))
        if not crossings:
            continue
        crossings.sort()
        for a, b in zip(crossings[0::2], crossings[1::2]):
            # half-open in x as well as y: pixel x is filled when a <= x+0.5 < b
            xa = max(0, int(np.ceil(a - 0.5)))
            xb = min(width - 1, int(np.ceil(b - 0.5)) - 1)
            if xb >= xa:
                grid[y, xa : xb + 1] = True
    return grid


class ZoneRaster:
    """Pre-rasterised zones for a frame size. Built once, reused every frame."""

    def __init__(self, zone_set: ZoneSet, width: int, height: int) -> None:
        self.zone_set = zone_set
        self.width = width
        self.height = height
        sx = width / zone_set.width
        sy = height / zone_set.height
        self.masks: dict[str, np.ndarray] = {}
        self.areas: dict[str, int] = {}
        for z in zone_set.zones:
            scaled = z if (sx == 1.0 and sy == 1.0) else Zone(
                z.name, tuple((x * sx, y * sy) for x, y in z.polygon)
            )
            grid = rasterise(scaled, width, height)
            self.masks[z.name] = grid
            self.areas[z.name] = int(grid.sum())

    def overlap(self, mask: Mask) -> dict[str, float]:
        """Fraction of the mask's own pixels that fall inside each zone.

        Returns a value in [0, 1] per zone. Masks are placed into frame
        coordinates using their bounds, and clipped at the frame edge.
        """
        raster = np.frombuffer(mask.data, dtype=np.uint8).reshape(mask.rows, mask.cols)
        total = int(raster.sum())
        if total == 0:
            return {name: 0.0 for name in self.masks}

        # Mask rasters are smaller than the frame; stretch them onto the pixels
        # their bounds cover, using nearest-neighbour index maps.
        x0 = int(np.floor(mask.box.left))
        y0 = int(np.floor(mask.box.top))
        x1 = int(np.ceil(mask.box.right)) + 1
        y1 = int(np.ceil(mask.box.bottom)) + 1
        bw, bh = max(1, x1 - x0), max(1, y1 - y0)

        cols = np.minimum((np.arange(bw) * mask.cols) // bw, mask.cols - 1)
        rows = np.minimum((np.arange(bh) * mask.rows) // bh, mask.rows - 1)
        placed = raster[np.ix_(rows, cols)].astype(bool)

        # Clip to the frame.
        fx0, fy0 = max(0, x0), max(0, y0)
        fx1, fy1 = min(self.width, x0 + bw), min(self.height, y0 + bh)
        if fx1 <= fx0 or fy1 <= fy0:
            return {name: 0.0 for name in self.masks}
        sub = placed[fy0 - y0 : fy1 - y0, fx0 - x0 : fx1 - x0]
        denom = int(sub.sum())
        if denom == 0:
            return {name: 0.0 for name in self.masks}

        out: dict[str, float] = {}
        for name, zgrid in self.masks.items():
            inter = int(np.logical_and(sub, zgrid[fy0:fy1, fx0:fx1]).sum())
            out[name] = inter / denom
        return out
