"""The metrics themselves, checked against cases with known answers.

A measurement you have not tested is just a number with a label on it. These
construct the exact failures the metrics are supposed to name, and assert the
metric names them.
"""

from __future__ import annotations

from spatialsight.backends.synthetic import SyntheticBackend, default_scene
from spatialsight.eval import (
    compression_ratio,
    evaluate_synthetic,
    score_events,
    score_tracking,
    truth_tracks,
)
from spatialsight.events import ZoneEvent
from spatialsight.types import Box, Detection, Track, VideoMeta
from spatialsight.zones import ZoneRaster, load_zones

ZONES = "zones/synthetic-01.json"


def _track(entity_id: int, frames: range, x0: float, label: str = "person") -> Track:
    return Track(
        entity_id=entity_id,
        label=label,
        detections=[
            Detection(
                frame=f,
                object_id=str(entity_id),
                label=label,
                box=Box(x0 + f, 100, x0 + f + 40, 220),
            )
            for f in frames
        ],
    )


def _event(entity_id: int, zone: str, start: float, end: float, label="person") -> ZoneEvent:
    return ZoneEvent(
        entity_id=entity_id,
        label=label,
        zone=zone,
        frame_start=int(start * 2),
        frame_end=int(end * 2),
        entered_at=start,
        exited_at=end,
        dwell_s=end - start,
        peak_overlap=0.9,
        closed=True,
    )


# ----- tracking metrics -----------------------------------------------------


def test_perfect_tracking_scores_one():
    truth = [_track(1, range(10), 0)]
    metrics = score_tracking([_track(1, range(10), 0)], truth)
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.id_switches == 0
    assert metrics.fragmentations == 0


def test_missed_detections_lower_recall_not_precision():
    truth = [_track(1, range(10), 0)]
    metrics = score_tracking([_track(1, range(5), 0)], truth)
    assert metrics.missed == 5
    assert metrics.recall == 0.5
    assert metrics.precision == 1.0


def test_spurious_detections_lower_precision_not_recall():
    truth = [_track(1, range(5), 0)]
    predicted = [_track(1, range(5), 0), _track(2, range(5), 800)]
    metrics = score_tracking(predicted, truth)
    assert metrics.spurious == 5
    assert metrics.recall == 1.0
    assert metrics.precision == 0.5


def test_one_actor_split_across_two_tracks_is_a_fragmentation():
    truth = [_track(1, range(10), 0)]
    predicted = [_track(1, range(5), 0), _track(2, range(5, 10), 0)]
    metrics = score_tracking(predicted, truth)
    assert metrics.fragmentations == 1
    assert metrics.recall == 1.0


def test_classes_are_never_matched_across():
    truth = [_track(1, range(5), 0, label="vehicle")]
    metrics = score_tracking([_track(1, range(5), 0, label="person")], truth)
    assert metrics.matched == 0
    assert metrics.missed == 5


# ----- event metrics --------------------------------------------------------


def test_identical_events_are_all_caught():
    truth = [_event(1, "dock", 0, 10), _event(2, "aisle", 20, 30)]
    metrics = score_events(list(truth), truth)
    assert metrics.caught == 2
    assert metrics.missed == 0
    assert metrics.invented == 0
    assert metrics.mean_abs_dwell_error_s == 0.0


def test_a_missing_event_is_counted_as_missed():
    truth = [_event(1, "dock", 0, 10), _event(2, "aisle", 20, 30)]
    metrics = score_events([truth[0]], truth)
    assert metrics.caught == 1 and metrics.missed == 1 and metrics.invented == 0


def test_an_extra_event_is_counted_as_invented():
    truth = [_event(1, "dock", 0, 10)]
    metrics = score_events([truth[0], _event(9, "aisle", 50, 60)], truth)
    assert metrics.caught == 1 and metrics.invented == 1 and metrics.missed == 0


def test_barely_overlapping_events_do_not_match():
    truth = [_event(1, "dock", 0, 10)]
    # Overlaps 1s out of an 11s span: temporal IoU well under 0.5.
    metrics = score_events([_event(1, "dock", 9, 20)], truth)
    assert metrics.caught == 0
    assert metrics.missed == 1 and metrics.invented == 1


def test_entity_numbering_is_allowed_to_differ():
    truth = [_event(1, "dock", 0, 10)]
    metrics = score_events([_event(7, "dock", 0, 10)], truth)
    assert metrics.caught == 1, "zone and time are what matter, not the id"


def test_wrong_zone_is_not_a_match():
    truth = [_event(1, "dock", 0, 10)]
    metrics = score_events([_event(1, "aisle", 0, 10)], truth)
    assert metrics.caught == 0


def test_dwell_error_is_reported():
    truth = [_event(1, "dock", 0, 10)]
    metrics = score_events([_event(1, "dock", 0, 12)], truth)
    assert metrics.caught == 1
    assert metrics.mean_abs_dwell_error_s == 2.0


# ----- the full report ------------------------------------------------------


def test_synthetic_evaluation_is_perfect_on_detections_and_events():
    backend = SyntheticBackend("synthetic-01")
    scene = default_scene()
    meta = backend.meta()
    raster = ZoneRaster(load_zones(ZONES), meta.width, meta.height)
    report = evaluate_synthetic(
        backend.detect(), scene, raster, meta, "synthetic-01", db_bytes=32768
    )
    assert report.tracking.recall == 1.0
    assert report.tracking.precision == 1.0
    assert report.events.caught == report.events.true_total
    assert report.events.invented == 0
    assert report.compression["ratio"] > 1


def test_the_unfixed_id_switch_is_reported_not_hidden():
    backend = SyntheticBackend("synthetic-01")
    scene = default_scene()
    meta = backend.meta()
    raster = ZoneRaster(load_zones(ZONES), meta.width, meta.height)
    report = evaluate_synthetic(backend.detect(), scene, raster, meta, "synthetic-01")
    assert report.tracking.id_switches > 0, (
        "the scene injects an ID switch between adjacent objects; the report "
        "must not claim a clean sheet"
    )
    assert report.notes, "the limitation must be stated in the report"


def test_truth_tracks_match_the_actor_count():
    assert len(truth_tracks(default_scene())) == len(default_scene().actors)


def test_compression_falls_back_to_a_bitrate_estimate_for_simulated_clips():
    meta = VideoMeta("synthetic://x", 1280, 720, 2.0, 240, 120.0, "abc")
    ratio = compression_ratio(meta, 32768)
    assert ratio["source_bytes"] > 0
    assert ratio["ratio"] > 1


def test_a_correct_unrounded_answer_matches_a_rounded_reference():
    """3.25 and ROUND(3.25, 1) are the same answer; Python and SQLite
    disagree on which way the tie goes, and the grader must not."""
    from spatialsight.query.bench import compare

    verdict = compare([("east_road", 3.3)], ["zone", "m"], [("east_road", 3.25)], ["zone", "m"])
    assert verdict == "exact"
