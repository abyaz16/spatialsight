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


def test_wrong_row_count_is_a_miss():
    ref = [("a", 1.0), ("b", 2.0)]
    assert compare(ref, ["x", "y"], ref[:1], ["x", "y"]) == "miss"


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
