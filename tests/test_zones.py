"""The polygon rasteriser, checked against areas computed by hand.

Every zone event in the product depends on one number: the fraction of an
entity's pixels inside a polygon. These tests pin that number down.
"""

from __future__ import annotations

from spatialsight.types import Box, Mask
from spatialsight.zones import Zone, ZoneRaster, ZoneSet, rasterise


def test_axis_aligned_square_is_exact():
    z = Zone("s", ((0, 0), (10, 0), (10, 10), (0, 10)))
    assert rasterise(z, 20, 20).sum() == 100


def test_right_triangle_matches_centre_inside_rule():
    # Pixel centres satisfying (x+.5)+(y+.5) < 10 number 45, not the
    # continuous area of 50. Binary masks cannot do better, and the
    # convention must be the same on both axes.
    z = Zone("t", ((0, 0), (10, 0), (0, 10)))
    assert rasterise(z, 20, 20).sum() == 45


def test_concave_polygon_fills_correctly():
    # An L shape: 10x4 plus 4x6 = 64 pixels. Even-odd fill must not bleed
    # across the notch.
    z = Zone("l", ((0, 0), (10, 0), (10, 4), (4, 4), (4, 10), (0, 10)))
    assert rasterise(z, 20, 20).sum() == 64


def test_polygon_clipped_at_frame_edge():
    z = Zone("edge", ((-5, -5), (5, -5), (5, 5), (-5, 5)))
    assert rasterise(z, 10, 10).sum() == 25


def test_zone_needs_three_points():
    try:
        Zone("bad", ((0, 0), (1, 1)))
    except ValueError:
        return
    raise AssertionError("a 2-point zone should be rejected")


def _raster() -> ZoneRaster:
    zs = ZoneSet("cam", 20, 20, (Zone("dock", ((0, 0), (10, 0), (10, 10), (0, 10))),))
    return ZoneRaster(zs, 20, 20)


def test_mask_fully_inside_is_one():
    zr = _raster()
    m = Mask(4, 4, bytes([1] * 16), Box(1, 1, 4, 4))
    assert zr.overlap(m)["dock"] == 1.0


def test_mask_fully_outside_is_zero():
    zr = _raster()
    m = Mask(4, 4, bytes([1] * 16), Box(14, 14, 18, 18))
    assert zr.overlap(m)["dock"] == 0.0


def test_mask_straddling_the_boundary_is_fractional():
    zr = _raster()
    m = Mask(4, 8, bytes([1] * 32), Box(7, 2, 14, 5))
    frac = zr.overlap(m)["dock"]
    assert 0.0 < frac < 1.0


def test_empty_mask_is_zero_not_a_division_error():
    zr = _raster()
    m = Mask(4, 4, bytes(16), Box(1, 1, 4, 4))
    assert zr.overlap(m)["dock"] == 0.0


def test_zones_scale_when_drawn_against_a_different_frame_size():
    # Zones drawn on a 20x20 preview, applied to a 40x40 frame: the zone
    # covers the same proportion of the image, so 4x the pixels.
    zs = ZoneSet("cam", 20, 20, (Zone("dock", ((0, 0), (10, 0), (10, 10), (0, 10))),))
    assert ZoneRaster(zs, 40, 40).areas["dock"] == 400
