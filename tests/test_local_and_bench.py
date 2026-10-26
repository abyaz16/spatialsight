"""The pieces added for free local testing: mask conversion, the benchmark's
result comparison, and LLM endpoint configuration.

None of these import torch or ultralytics, so they run on any machine.
"""

from __future__ import annotations

import pytest

from spatialsight import config
from spatialsight.backends.local import _rle, _unrle, polygon_to_mask
from spatialsight.query.bench import compare, run_bench
from spatialsight.query.canned import QUESTIONS
from spatialsight.types import Box
from spatialsight.zones import Zone, ZoneRaster, ZoneSet

# ----- polygon -> mask ------------------------------------------------------


def test_polygon_filling_its_box_gives_a_full_mask():
    box = Box(100, 200, 140, 280)
    square = [(100, 200), (140, 200), (140, 280), (100, 280)]
    mask = polygon_to_mask(square, box, res=16)
    assert mask.rows == mask.cols == 16
    assert mask.pixel_count == 256


def test_polygon_covering_half_its_box_gives_half_a_mask():
    box = Box(0, 0, 40, 40)
    left_half = [(0, 0), (20, 0), (20, 40), (0, 40)]
    mask = polygon_to_mask(left_half, box, res=16)
    assert mask.pixel_count == 128


def test_mask_from_polygon_measures_occupancy_like_any_other_mask():
    # A person whose polygon is the left half of their box, standing with
    # that half inside a zone, is 100% inside it by pixels even though only
    # half the box is.
    zones = ZoneSet("c", 200, 200, (Zone("z", ((0, 0), (20, 0), (20, 200), (0, 200))),))
    raster = ZoneRaster(zones, 200, 200)
    box = Box(0, 50, 40, 90)
    mask = polygon_to_mask([(0, 50), (20, 50), (20, 90), (0, 90)], box, res=16)
    assert raster.overlap(mask)["z"] == pytest.approx(1.0, abs=0.05)


def test_rle_round_trip():
    data = bytes([0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0])
    assert _unrle(_rle(data)) == data


def test_rle_of_a_mask_starting_with_a_set_pixel():
    assert _rle(bytes([1, 0])) == [0, 1, 1]
    assert _unrle([0, 1, 1]) == bytes([1, 0])


# ----- the benchmark's comparison -------------------------------------------


def test_identical_rows_are_exact():
    assert compare([("dock", 4.0)], ["zone", "d"], [("dock", 4.0)], ["z", "dwell"]) == "exact"


def test_row_order_does_not_matter():
    ref = [("a", 1.0), ("b", 2.0)]
    assert compare(ref, ["x", "y"], list(reversed(ref)), ["x", "y"]) == "exact"


def test_float_noise_is_tolerated():
    assert compare([("a", 43.0)], ["z", "v"], [("a", 43.0000001)], ["z", "v"]) == "exact"


def test_extra_columns_with_the_right_rows_are_contains():
    ref = [("dock", 43.0), ("aisle", 30.0)]
    model = [("dock", 43.0, 2), ("aisle", 30.0, 2)]
    assert compare(ref, ["zone", "avg"], model, ["zone", "avg", "visits"]) == "contains"


def test_wrong_values_are_a_miss():
    assert compare([("dock", 43.0)], ["z", "v"], [("dock", 12.0)], ["z", "v"]) == "miss"


# ----- subset: the top of a ranking, or fewer columns -----------------------


def test_top_one_of_a_ranking_is_a_subset_not_a_miss():
    # "Which zone had the longest average dwell?" answered with one row,
    # against a reference that returns the whole ranking.
    ref = [("dock", 43.0, 2), ("aisle", 30.0, 2), ("bench", 14.5, 2)]
    model = [("dock", 43.0)]
    assert compare(ref, ["zone", "avg", "visits"], model, ["zone", "avg"]) == "subset"


def test_bottom_of_a_ranking_is_still_a_miss():
    # The trap a sub-multiset check would fall into: the least-dwelt zone is
    # the opposite answer to the most-dwelt one, and must never score.
    ref = [("dock", 43.0, 2), ("aisle", 30.0, 2), ("bench", 14.5, 2)]
    assert compare(ref, ["zone", "avg", "visits"], [("bench", 14.5)], ["zone", "avg"]) == "miss"


def test_fewer_columns_over_all_rows_is_a_subset():
    ref = [(51.5, 0, "dock"), (34.5, 170, "dock")]
    model = [(51.5, 0), (34.5, 170)]
    assert compare(ref, ["d", "f", "z"], model, ["dwell_s", "frame_start"]) == "subset"


def test_a_subset_must_keep_the_reference_order():
    ref = [("a", 1.0), ("b", 2.0), ("c", 3.0)]
    assert compare(ref, ["x", "y"], [("b", 2.0)], ["x", "y"]) == "miss"


def test_two_model_columns_cannot_claim_one_reference_column():
    ref = [("dock", 43.0, 2)]
    # Both model columns hold the same values; only one reference column does.
    model = [(43.0, 43.0)]
    assert compare(ref, ["z", "avg", "n"], model, ["a", "b"]) == "miss"


def test_more_rows_than_the_reference_is_a_miss():
    ref = [("a", 1.0)]
    assert compare(ref, ["x", "y"], [("a", 1.0), ("b", 2.0)], ["x", "y"]) == "miss"


def test_an_empty_model_result_against_real_rows_is_a_miss():
    assert compare([("a", 1.0)], ["x", "y"], [], ["x", "y"]) == "miss"


# ----- covers: a different projection that still answers the question -------


def test_a_different_projection_carrying_the_required_columns_covers():
    # The real case: the model swapped `entered_at` for `clip`, so its columns
    # are neither a subset nor a superset of the reference's -- but it has the
    # dwell and the frame to seek to, which is what the question asked for.
    ref = [(2, "vehicle", "dock", 51.5, 0, 0.0), (5, "person", "dock", 34.5, 170, 85.0)]
    ref_cols = ["entity_id", "class", "zone", "dwell_s", "frame_start", "entered_at"]
    model = [("c1", 2, "vehicle", "dock", 51.5, 0), ("c1", 5, "person", "dock", 34.5, 170)]
    model_cols = ["clip", "entity_id", "class", "zone", "dwell_s", "frame_start"]
    assert compare(ref, ref_cols, model, model_cols) == "miss", "no requirement, no pass"
    assert (
        compare(ref, ref_cols, model, model_cols, required=("dwell_s", "frame_start"))
        == "covers"
    )


def test_covers_still_refuses_a_missing_required_column():
    ref = [(2, "dock", 51.5, 0)]
    ref_cols = ["entity_id", "zone", "dwell_s", "frame_start"]
    model = [(2, "dock")]  # no dwell, no frame
    assert (
        compare(ref, ref_cols, model, ["entity_id", "zone"], required=("dwell_s", "frame_start"))
        == "miss"
    )


def test_covers_still_refuses_the_bottom_of_a_ranking():
    ref = [("dock", 43.0, 2), ("bench", 14.5, 2)]
    ref_cols = ["zone", "avg_dwell_s", "visits"]
    model = [("x", "bench", 14.5)]
    assert compare(ref, ref_cols, model, ["clip", "zone", "avg"], required=("zone", "avg_dwell_s")) == "miss"


def test_every_question_declares_required_columns_that_its_sql_returns():
    from spatialsight.query.guard import run_readonly

    for q in QUESTIONS:
        assert q.must_have, f"{q.key} declares no required columns"
        columns = run_readonly("data/db/spatialsight.sqlite", q.sql).columns
        missing = [c for c in q.must_have if c not in columns]
        assert not missing, f"{q.key} requires {missing}, which its own SQL does not return"


def test_group_concat_order_is_ignored():
    ref = [(1, "dock,aisle")]
    assert compare(ref, ["e", "zones"], [(1, "aisle,dock")], ["e", "zones"]) == "exact"


def test_two_empty_results_agree():
    assert compare([], ["a"], [], ["a", "b"]) == "exact"


# ----- the benchmark end to end, with a stub model --------------------------


class EchoReferenceClient:
    """Answers every question with its own reference SQL: a perfect model."""

    def __init__(self) -> None:
        self.by_question = {q.question: q.sql for q in QUESTIONS}
        self.model = "echo"
        self.usage = {"calls": 0}

    def complete(self, system: str, user: str, **_kwargs) -> str:
        self.usage["calls"] += 1
        return self.by_question.get(user.split("\n")[0], "SELECT 1")


@pytest.fixture(scope="module")
def db(tmp_path_factory):
    from spatialsight.backends.synthetic import SyntheticBackend
    from spatialsight.runner import run

    path = tmp_path_factory.mktemp("bench") / "b.sqlite"
    run(
        backend=SyntheticBackend("synthetic-01"),
        clip="synthetic-01",
        zones_path="zones/synthetic-01.json",
        db_path=path,
    )
    return path


def test_a_perfect_model_scores_ten_out_of_ten(db):
    report = run_bench(db, EchoReferenceClient())
    assert report.answered == len(QUESTIONS)
    assert report.exact == len(QUESTIONS)


def test_a_model_that_always_refuses_scores_zero(db):
    class Refuser(EchoReferenceClient):
        def complete(self, system, user, **_kwargs):
            self.usage["calls"] += 1
            return "DROP TABLE events"

    report = run_bench(db, Refuser())
    assert report.answered == 0
    assert all(item.verdict == "error" for item in report.items)


# ----- LLM endpoint configuration -------------------------------------------


def test_default_llm_is_muse_spark_on_the_meta_api(monkeypatch):
    for var in ("LLM_BASE_URL", "LLM_MODEL", "LLM_API_KEY", "LLM_API_STYLE", "MODEL_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    cfg = config.load()
    assert cfg.llm_model == "muse-spark-1.3"
    assert cfg.llm_api_style == "responses"
    assert cfg.has_llm is False


def test_a_local_endpoint_needs_no_key_and_speaks_chat(monkeypatch):
    for var in ("LLM_API_KEY", "LLM_API_STYLE", "MODEL_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("LLM_BASE_URL", "http://127.0.0.1:8080/v1")
    monkeypatch.setenv("LLM_MODEL", "llama-3.2-3b")
    cfg = config.load()
    assert cfg.has_llm is True
    assert cfg.llm_api_style == "chat"
    assert cfg.llm_model == "llama-3.2-3b"


def test_a_hosted_free_tier_uses_its_own_key(monkeypatch):
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_STYLE", raising=False)
    monkeypatch.setenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
    monkeypatch.setenv("LLM_API_KEY", "gsk_test")
    cfg = config.load()
    assert cfg.llm_api_key == "gsk_test"
    assert cfg.has_key is False, "a text-model key must not unlock paid SAM calls"
    assert cfg.has_llm is True


def test_meta_key_alone_enables_both(monkeypatch):
    for var in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_API_STYLE", "LLM_MODEL"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("MODEL_API_KEY", "meta_test")
    cfg = config.load()
    assert cfg.has_key and cfg.has_llm
    assert cfg.llm_api_key == "meta_test"


# ----- order only matters for a top-k claim ---------------------------------


def test_same_rows_in_a_different_order_with_a_different_projection_covers():
    # The real case: the model returned all 11 open visits, correct, but
    # sorted by clip instead of by dwell and carrying `clip` where the
    # reference carried `class`.
    ref = [(4, "vehicle", "east_road", 79.4), (29, "person", "west_path", 1.1)]
    ref_cols = ["entity_id", "class", "zone", "dwell_s"]
    model = [(29, "west_path", 1.1), (4, "east_road", 79.4)]  # reverse order
    model_cols = ["entity_id", "zone", "dwell_s"]
    assert compare(ref, ref_cols, model, model_cols, required=("entity_id", "zone")) == "subset"


def test_a_prefix_claim_still_requires_the_right_order():
    ref = [("a", 3.0), ("b", 2.0), ("c", 1.0)]
    # One row out of three asserts it is the top one; 'c' is the bottom.
    assert compare(ref, ["z", "v"], [("c", 1.0)], ["z", "v"], required=("z",)) == "miss"
    assert compare(ref, ["z", "v"], [("a", 3.0)], ["z", "v"], required=("z",)) == "subset"


def test_columns_right_individually_but_rows_paired_wrongly_is_a_miss():
    # Both columns hold the same values as the reference, but the pairing is
    # scrambled: zone 'a' had 1.0, not 2.0. A column-wise check would pass it.
    ref = [("a", 1.0), ("b", 2.0)]
    scrambled = [("a", 2.0), ("b", 1.0)]
    assert compare(ref, ["zone", "v"], scrambled, ["zone", "v"]) == "miss"
