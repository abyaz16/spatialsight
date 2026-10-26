"""The MOT17 loader and the fragmentation that makes its input realistic.

These run without the dataset present: the loader is exercised against a
small gt.txt written in the real format. The full benchmark needs the real
annotations and is run through `ss mot`.
"""

from __future__ import annotations

import pytest

from spatialsight.datasets.mot import fragment, identity_errors, load_sequence, true_tracks
from spatialsight.tracker import TrackerParams, reconcile

SEQINFO = """[Sequence]
name=TEST-01
imDir=img1
frameRate=30
seqLength=20
imWidth=1920
imHeight=1080
imExt=.jpg
"""

# frame, id, left, top, w, h, conf, class, visibility
ROWS = [
    # identity 1: frames 1-5, then a gap, then 12-16 (an occlusion)
    *[f"{f},1,{100 + f * 10},200,50,120,1,1,1.0" for f in range(1, 6)],
    *[f"{f},1,{100 + f * 10},200,50,120,1,1,1.0" for f in range(12, 17)],
    # identity 2: continuous
    *[f"{f},2,{800 - f * 5},300,50,120,1,1,1.0" for f in range(1, 20)],
    # ignored rows: inactive, wrong class, barely visible
    "3,9,10,10,20,20,0,1,1.0",
    "4,9,10,10,20,20,1,7,1.0",
    "5,9,10,10,20,20,1,1,0.05",
]


@pytest.fixture()
def sequence(tmp_path):
    (tmp_path / "gt").mkdir()
    (tmp_path / "seqinfo.ini").write_text(SEQINFO, encoding="utf-8")
    (tmp_path / "gt" / "gt.txt").write_text("\n".join(ROWS) + "\n", encoding="utf-8")
    return load_sequence(tmp_path, min_visibility=0.25)


def test_metadata_comes_from_seqinfo(sequence):
    assert sequence.name == "TEST-01"
    assert (sequence.meta.width, sequence.meta.height) == (1920, 1080)
    assert sequence.meta.fps == 30
    assert sequence.meta.frame_count == 20


def test_frames_are_converted_to_zero_indexed(sequence):
    # The file is 1-indexed; the rest of the pipeline is not. Off by one here
    # shifts every timestamp in the events table.
    assert min(b.frame for b in sequence.boxes) == 0


def test_inactive_wrong_class_and_invisible_rows_are_dropped(sequence):
    assert sequence.identities == 2, "identity 9 is ignorable on all three counts"


def test_true_tracks_match_the_annotated_identities(sequence):
    tracks = true_tracks(sequence)
    assert len(tracks) == 2
    assert all(t.label == "person" for t in tracks)


def test_fragmentation_splits_at_an_occlusion(sequence):
    # Identity 1 vanishes for 6 frames; a tracker with no patience loses it.
    _, observed = fragment(sequence, patience=0)
    assert observed == 3, "two identities, one of them split by its occlusion"


def test_patience_beyond_the_gap_keeps_the_identity_whole(sequence):
    _, observed = fragment(sequence, patience=10)
    assert observed == 2, "a patient tracker rides out the occlusion"


def test_fragmentation_never_loses_a_detection(sequence):
    for patience in (0, 3, 10):
        detections, _ = fragment(sequence, patience=patience)
        assert len(detections) == len(sequence.boxes)


def test_reconciliation_rejoins_the_occluded_identity(sequence):
    detections, observed = fragment(sequence, patience=0)
    assert observed == 3
    recon = reconcile(
        detections, TrackerParams.for_fps(sequence.meta.fps, 0.5, min_detections=3)
    )
    assert len(recon.tracks) == 2, "the occlusion split should be repaired"
    assert identity_errors(recon.tracks, sequence)["merges"] == 0


def test_identity_errors_counts_a_merge(tmp_path):
    """The merge metric has to be able to see a merge.

    Needs two identities that are *sequential*: the fixture above has them
    overlapping in time, so no reconnection between them is even possible and
    a reckless tracker produces no merge to detect.
    """
    rows = [
        *[f"{f},1,{100 + f * 10},200,50,120,1,1,1.0" for f in range(1, 6)],
        # a different person, later and far away
        *[f"{f},2,{900 + f * 10},700,50,120,1,1,1.0" for f in range(9, 15)],
    ]
    (tmp_path / "gt").mkdir()
    (tmp_path / "seqinfo.ini").write_text(SEQINFO, encoding="utf-8")
    (tmp_path / "gt" / "gt.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
    sequence = load_sequence(tmp_path, min_visibility=0.25)
    assert sequence.identities == 2

    detections, _ = fragment(sequence, patience=0)
    careful = reconcile(
        detections, TrackerParams.for_fps(sequence.meta.fps, 0.5, min_detections=3)
    )
    assert identity_errors(careful.tracks, sequence)["merges"] == 0, (
        "two people 800px apart must never be joined"
    )

    reckless = TrackerParams(
        max_gap_frames=1000,
        stitch_slack=1e6,
        stitch_slack_cap=1e9,
        stitch_max_size_ratio=1e9,
        min_detections=1,
    )
    errors = identity_errors(reconcile(detections, reckless).tracks, sequence)
    assert errors["merges"] >= 1, "a metric that cannot see a merge is useless"


def test_masks_are_optional_and_cost_nothing_when_off(sequence):
    detections, _ = fragment(sequence, patience=0, with_mask=False)
    assert all(d.mask is None for d in detections)
    with_masks, _ = fragment(sequence, patience=0, with_mask=True)
    assert all(d.mask is not None for d in with_masks)
