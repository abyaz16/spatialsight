"""ID reconciliation: the repairs it must make, and the ones it must refuse.

The refusals matter more than the repairs. An over-eager stitcher merges two
different people into one entity and silently invents a dwell, which is worse
than reporting two entities where there was one.
"""

from __future__ import annotations

from spatialsight.tracker import TrackerParams, reconcile
from spatialsight.types import Box, Detection


def det(frame, object_id, x, y, w=20, h=20, label="person"):
    return Detection(
        frame=frame,
        object_id=object_id,
        label=label,
        box=Box(x, y, x + w, y + h),
    )


def test_continuous_track_is_left_alone():
    dets = [det(f, "1", f * 2, 50) for f in range(10)]
    r = reconcile(dets)
    assert len(r.tracks) == 1
    assert r.stitches == 0 and r.splits == 0


def test_reacquired_entity_is_stitched_when_it_resumes_its_path():
    # Walks right at 2 px/frame, disappears for 4 frames, reappears exactly
    # where that motion predicts.
    dets = [det(f, "1", f * 2, 50) for f in range(0, 11)]
    dets += [det(f, "7", f * 2, 50) for f in range(15, 21)]
    r = reconcile(dets)
    assert len(r.tracks) == 1
    assert r.stitches == 1
    assert r.tracks[0].source_object_ids == ["1", "7"]


def test_gap_longer_than_the_window_stays_two_entities():
    dets = [det(f, "1", f * 2, 50) for f in range(0, 6)]
    dets += [det(f, "7", f * 2, 50) for f in range(40, 46)]
    r = reconcile(dets)
    assert len(r.tracks) == 2
    assert r.stitches == 0


def test_a_stranger_across_the_room_is_not_stitched():
    # This is the false-positive the predictive gate exists to stop: a
    # different person appears 5 frames later, well off the predicted path.
    dets = [det(f, "1", f * 2, 50) for f in range(0, 11)]
    dets += [det(f, "9", 600, 400) for f in range(15, 21)]
    r = reconcile(dets)
    assert len(r.tracks) == 2, "predicted position must gate the stitch"
    assert r.stitches == 0


def test_teleporting_id_is_split():
    dets = [det(f, "1", f * 2, 50) for f in range(0, 6)]
    dets += [det(f, "1", 800 + f * 2, 600) for f in range(6, 12)]
    r = reconcile(dets)
    assert len(r.tracks) == 2
    assert r.splits == 1


def test_crossing_entities_stay_separate():
    dets = [det(f, "1", f * 10, 50) for f in range(10)]
    dets += [det(f, "2", 90 - f * 10, 50) for f in range(10)]
    r = reconcile(dets)
    assert len(r.tracks) == 2


def test_classes_are_never_stitched_together():
    dets = [det(f, "1", f * 2, 50, label="person") for f in range(0, 6)]
    dets += [det(f, "2", f * 2, 50, label="vehicle") for f in range(8, 14)]
    r = reconcile(dets)
    assert len(r.tracks) == 2


def test_wildly_different_sizes_are_never_stitched():
    dets = [det(f, "1", f * 2, 50, w=20, h=20) for f in range(0, 6)]
    dets += [det(f, "2", f * 2, 50, w=200, h=200) for f in range(8, 14)]
    r = reconcile(dets)
    assert len(r.tracks) == 2


def test_entity_ids_are_assigned_in_order_of_appearance():
    dets = [det(f, "b", 500, 500) for f in range(20, 30)]
    dets += [det(f, "a", 10, 10) for f in range(0, 10)]
    r = reconcile(dets)
    assert [t.entity_id for t in r.tracks] == [1, 2]
    assert r.tracks[0].first_frame == 0


def test_stitch_window_is_configurable():
    dets = [det(f, "1", f * 2, 50) for f in range(0, 11)]
    dets += [det(f, "7", f * 2, 50) for f in range(25, 31)]
    assert len(reconcile(dets).tracks) == 2
    loose = TrackerParams(max_gap_frames=30)
    assert len(reconcile(dets, loose).tracks) == 1


# ----- what MOT17 taught, pinned so it cannot regress ----------------------


def test_partial_box_overlap_alone_does_not_justify_a_stitch():
    """Two different people standing near each other overlap ~10% of their boxes.

    When that was enough to bypass the motion test, it caused 21 wrong merges
    on MOT17 by itself. Overlap is now only conclusive when it is decisive.
    """
    dets = [det(f, "1", 0, 0, w=100, h=260) for f in range(0, 10)]
    # A different person, 10% box overlap, nowhere near the predicted path.
    dets += [det(f, "2", 90, 0, w=100, h=260) for f in range(12, 20)]
    r = reconcile(dets)
    assert len(r.tracks) == 2, "partial overlap must not short-circuit the gate"


def test_decisive_box_overlap_still_reconnects():
    dets = [det(f, "1", 0, 0, w=100, h=260) for f in range(0, 10)]
    dets += [det(f, "2", 5, 0, w=100, h=260) for f in range(12, 20)]
    assert len(reconcile(dets).tracks) == 1


def test_an_ambiguous_reconnection_is_refused_rather_than_guessed():
    """Two candidates fitting one prediction is a coin flip, and a wrong flip
    welds two people into one history that nothing downstream can detect."""
    # Walks right at 5px/frame, so from frame 9 the prediction for frame 12
    # is centred at (80, 150).
    dets = [det(f, "1", f * 5, 100, w=40, h=100) for f in range(0, 10)]
    # Two continuations straddling that prediction, each 10px off it: both
    # well inside the tolerance, and neither better than the other.
    dets += [det(f, "a", 60 + (f - 12) * 5, 90, w=40, h=100) for f in range(12, 18)]
    dets += [det(f, "b", 60 + (f - 12) * 5, 110, w=40, h=100) for f in range(12, 18)]
    r = reconcile(dets)
    assert r.ambiguous >= 1, "the tie must be detected, not broken arbitrarily"
    assert len(r.tracks) == 3, "an ambiguous match must not be guessed"


def test_the_gap_window_is_a_duration_not_a_frame_count():
    """12 frames is 6 seconds at 2fps and 0.4 seconds at 30fps.

    A fixed frame count silently means different things per clip, which is
    how a window tuned on the 2fps simulated clip became far too permissive
    on 30fps footage.
    """
    assert TrackerParams.for_fps(2.0, 1.0).max_gap_frames == 2
    assert TrackerParams.for_fps(30.0, 1.0).max_gap_frames == 30
    assert TrackerParams.for_fps(30.0, 0.5).max_gap_frames == 15
    assert TrackerParams.for_fps(0.1, 1.0).max_gap_frames >= 1
