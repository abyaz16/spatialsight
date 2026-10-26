"""The hysteresis state machine.

Case `test_oscillation_inside_a_zone_is_one_visit` is the one that earns this
module its existence: with a single threshold it returns four visits, and the
events table reports four dock entries where a human counts one.
"""

from __future__ import annotations

import pytest

from spatialsight.events import EventParams, _intervals

FRAMES = list(range(20))


def test_clean_visit():
    overlaps = [0.0] * 5 + [0.9] * 8 + [0.0] * 7
    assert _intervals(FRAMES, overlaps, EventParams()) == [(5, 12, 0.9)]


def test_hovering_below_the_entry_threshold_is_not_a_visit():
    overlaps = ([0.45, 0.35, 0.48, 0.31, 0.44, 0.38] * 3)[:20]
    assert _intervals(FRAMES, overlaps, EventParams()) == []


def test_oscillation_inside_a_zone_is_one_visit():
    overlaps = [0.0] * 2 + [0.9, 0.35, 0.85, 0.33, 0.95, 0.40, 0.88] + [0.0] * 11
    result = _intervals(FRAMES, overlaps, EventParams())
    assert len(result) == 1, "hysteresis must not split one visit into four"
    assert result[0][:2] == (2, 8)


def test_two_visits_separated_by_a_real_absence():
    overlaps = [0.9] * 4 + [0.0] * 10 + [0.9] * 4 + [0.0] * 2
    assert len(_intervals(FRAMES, overlaps, EventParams())) == 2


def test_short_dropout_does_not_end_a_visit():
    overlaps = [0.9] * 5 + [0.0] * 3 + [0.9] * 5 + [0.0] * 7
    result = _intervals(FRAMES, overlaps, EventParams())
    assert len(result) == 1, "a 3-frame dropout is inside max_gap_frames"


def test_visit_still_open_at_the_end_of_the_clip():
    overlaps = [0.0] * 15 + [0.9] * 5
    assert _intervals(FRAMES, overlaps, EventParams()) == [(15, 19, 0.9)]


def test_peak_overlap_is_the_maximum_seen():
    overlaps = [0.6, 0.72, 0.95, 0.55] + [0.0] * 16
    assert _intervals(FRAMES, overlaps, EventParams())[0][2] == 0.95


def test_exit_threshold_above_entry_is_rejected():
    with pytest.raises(ValueError):
        EventParams(enter_at=0.3, exit_at=0.6)


def test_thresholds_outside_zero_to_one_are_rejected():
    with pytest.raises(ValueError):
        EventParams(enter_at=1.5, exit_at=0.5)
