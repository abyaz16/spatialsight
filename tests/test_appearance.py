"""Colour read from an entity's own mask pixels.

The scene's actors are painted known colours, so these are not "does it
return something" tests: the answer is known in advance and a wrong one is a
bug, not a judgement call.
"""

from __future__ import annotations

import numpy as np
import pytest

from spatialsight.appearance import (
    MIN_CHROMA,
    Appearance,
    colour_family,
    describe_track,
    describe_tracks,
    mask_pixels,
    name_colour,
    reader_for,
)
from spatialsight.backends.synthetic import BACKGROUND_RGB, SyntheticBackend
from spatialsight.tracker import reconcile
from spatialsight.types import Box, Mask


# ----- naming ---------------------------------------------------------------


@pytest.mark.parametrize(
    "rgb,expected",
    [
        ((196, 32, 32), "red"),
        ((32, 72, 200), "blue"),
        ((36, 150, 60), "green"),
        ((226, 196, 40), "yellow"),
        ((240, 140, 20), "orange"),
        ((40, 200, 210), "cyan"),
        ((140, 40, 200), "purple"),
        ((230, 60, 150), "pink"),
        ((245, 245, 245), "white"),
        ((128, 128, 128), "grey"),
        ((10, 10, 12), "black"),
    ],
)
def test_unambiguous_colours_are_named_correctly(rgb, expected):
    assert name_colour(rgb)[0] == expected


def test_a_dark_grey_is_not_given_a_hue():
    """The bug this guards against produced 'purple' for #393440.

    Saturation is a ratio, so on near-black pixels a two-level difference
    between channels reads as strongly coloured. Judging on absolute chroma
    instead means a dark grey stays grey.
    """
    for hex_rgb in ("#393440", "#2c2a30", "#1a1720", "#35353b"):
        rgb = tuple(int(hex_rgb[i : i + 2], 16) for i in (1, 3, 5))
        assert name_colour(rgb)[0] in {"grey", "black"}, hex_rgb


def test_the_chroma_is_returned_as_the_evidence():
    _, chroma = name_colour((196, 32, 32))
    assert chroma == pytest.approx(164.0)
    assert chroma >= MIN_CHROMA


def test_out_of_range_channels_are_clamped_not_wrapped():
    assert name_colour((300, -20, -20))[0] == "red"


def test_neutrals_are_one_family_and_hues_are_not():
    assert colour_family("black") == colour_family("white") == "neutral"
    assert colour_family("red") != colour_family("blue")


def test_the_hex_is_the_rgb_it_was_judged_from():
    look = Appearance(colour="red", rgb=(196, 32, 32), chroma=164.0, samples=3)
    assert look.hex == "#c42020"


# ----- masking --------------------------------------------------------------


def _solid(colour, size=20):
    frame = np.empty((size, size, 3), dtype=np.uint8)
    frame[:, :] = colour
    return frame


def test_only_the_masked_pixels_are_returned():
    frame = _solid((10, 20, 30))
    frame[2:5, 2:5] = (200, 100, 50)
    mask = Mask(rows=3, cols=3, data=bytes([1] * 9), box=Box(2, 2, 4, 4))
    pixels = mask_pixels(frame, mask)
    assert len(pixels) == 9
    assert set(map(tuple, pixels)) == {(200, 100, 50)}


def test_an_empty_mask_returns_nothing():
    mask = Mask(rows=2, cols=2, data=bytes([0] * 4), box=Box(1, 1, 2, 2))
    assert len(mask_pixels(_solid((1, 2, 3)), mask)) == 0


def test_a_mask_partly_outside_the_frame_is_clipped_not_an_error():
    mask = Mask(rows=4, cols=4, data=bytes([1] * 16), box=Box(-5, -5, 2, 2))
    pixels = mask_pixels(_solid((7, 8, 9)), mask)
    assert len(pixels) > 0


def test_a_mask_entirely_outside_the_frame_returns_nothing():
    mask = Mask(rows=2, cols=2, data=bytes([1] * 4), box=Box(500, 500, 502, 502))
    assert len(mask_pixels(_solid((7, 8, 9)), mask)) == 0


def test_a_mask_coarser_than_its_box_still_covers_it():
    """SAM returns small masks over large boxes; every box pixel must map."""
    frame = _solid((0, 0, 0), size=40)
    frame[10:30, 10:30] = (90, 180, 90)
    mask = Mask(rows=4, cols=4, data=bytes([1] * 16), box=Box(10, 10, 29, 29))
    assert len(mask_pixels(frame, mask)) == 400


# ----- against the painted scene -------------------------------------------


@pytest.fixture(scope="module")
def scene():
    backend = SyntheticBackend("synthetic-01")
    recon = reconcile(backend.detect(("person", "vehicle")))
    looks = describe_tracks(recon.tracks, backend.frame_reader())
    return backend, recon, looks


def test_every_entity_gets_a_colour(scene):
    _, recon, looks = scene
    assert set(looks) == {t.entity_id for t in recon.tracks}
    assert all(look.samples > 0 for look in looks.values())


def test_the_colours_read_back_are_the_colours_painted(scene):
    """The clean entities must match ground truth exactly.

    Entities 3 and 4 are excluded because the scene deliberately swaps their
    ids mid-life, so neither has one true colour -- which is what the next
    test is about.
    """
    backend, _, looks = scene
    truth = {a.truth_id: a.colour for a in backend.truth_actors()}
    for entity_id in (1, 2, 5):
        assert looks[entity_id].colour == truth[entity_id]


def test_no_background_leaks_into_the_reading(scene):
    """A box-based average would land near the background; a mask does not.

    This is the test that justifies the cost of pixel-level segmentation for
    this feature at all.
    """
    _, _, looks = scene
    background = name_colour(BACKGROUND_RGB)[0]
    for entity_id in (1, 2, 5):
        assert looks[entity_id].colour != background


def test_colour_catches_the_id_switch_geometry_cannot_see(scene):
    """The scene's two touching actors swap ids with no geometric jump.

    The tracker has always been unable to see this, and says so. Colour sees
    it: both entities' own frames disagree about what colour they are, and
    they are the only two that do.
    """
    _, recon, looks = scene
    unstable = {eid for eid, look in looks.items() if not look.stable}
    assert unstable == {3, 4}


def test_a_stable_entity_reports_full_agreement(scene):
    _, _, looks = scene
    assert looks[1].stability == 1.0
    assert looks[1].stable


# ----- when there are no pixels --------------------------------------------


class _Blind:
    def frame(self, index):
        return None

    def close(self):
        self.closed = True


def test_an_unreadable_clip_yields_no_colour_rather_than_a_guess(scene):
    _, recon, _ = scene
    assert describe_track(recon.tracks[0], _Blind()) is None


def test_no_reader_means_no_appearances(scene):
    _, recon, _ = scene
    assert describe_tracks(recon.tracks, None) == {}


def test_the_reader_is_closed_even_on_a_blind_clip(scene):
    _, recon, _ = scene
    blind = _Blind()
    describe_tracks(recon.tracks, blind)
    assert getattr(blind, "closed", False)


def test_a_synthetic_path_has_no_frame_reader():
    assert reader_for("synthetic://synthetic-01") is None


def test_a_missing_video_has_no_frame_reader(tmp_path):
    assert reader_for(str(tmp_path / "nope.mp4")) is None


def test_an_empty_path_has_no_frame_reader():
    assert reader_for("") is None
