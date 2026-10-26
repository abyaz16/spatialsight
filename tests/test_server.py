"""The API layer, exercised without starting a socket.

`Api` is deliberately separate from the HTTP handler so the routes can be
tested as functions. The guard is tested in `test_guard.py`; what matters
here is that the server does not route around it.
"""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from spatialsight import config
from spatialsight.backends.synthetic import SyntheticBackend
from spatialsight.fixtures import build_fixture
from spatialsight.query.guard import GuardError
from spatialsight.runner import run
from spatialsight.server import Api, _rle


@pytest.fixture()
def api(tmp_path):
    cache = tmp_path / "cache"
    db = tmp_path / "db" / "spatialsight.sqlite"
    build_fixture(
        clip="synthetic-01",
        cache_dir=cache,
        manifest_path=db.parent / "clips.json",
    )
    run(
        backend=SyntheticBackend("synthetic-01"),
        clip="synthetic-01",
        zones_path="zones/synthetic-01.json",
        db_path=db,
    )
    cfg = replace(config.load(), db_path=db, cache_dir=cache, api_key=None)
    return Api(cfg)


# ----- run-length helper ----------------------------------------------------


def test_rle_starts_with_a_zero_run():
    assert _rle(bytes([1, 1, 0])) == [0, 2, 1]


def test_rle_round_trips_to_the_same_pixel_count():
    data = bytes([0, 0, 1, 1, 1, 0, 1, 0, 0])
    runs = _rle(data)
    assert sum(runs) == len(data)
    rebuilt = bytearray()
    value = 0
    for run in runs:
        rebuilt.extend([value] * run)
        value ^= 1
    assert bytes(rebuilt) == data


def test_rle_of_an_empty_mask_is_one_run():
    assert _rle(bytes(16)) == [16]


# ----- read routes ----------------------------------------------------------


def test_health_reports_the_database_and_key_state(api):
    health = api.health()
    assert health["ok"] is True
    assert health["database_exists"] is True
    assert health["clips"] == ["synthetic-01"]
    assert health["natural_language"] is False


def test_clips_includes_entity_and_event_counts(api):
    clips = api.clips()["clips"]
    assert len(clips) == 1
    assert clips[0]["entities"] == 5
    assert clips[0]["events"] > 0
    assert clips[0]["simulated"] is True


def test_zones_returns_the_polygons(api):
    zones = api.zones("synthetic-01")
    assert {z["name"] for z in zones["zones"]} == {
        "loading_dock",
        "packing_bench",
        "pick_aisle",
    }


def test_zones_for_an_unknown_clip_raises(api):
    with pytest.raises(FileNotFoundError):
        api.zones("no-such-clip")


def test_events_are_ordered_by_entry_time(api):
    events = api.events("synthetic-01")["events"]
    times = [e["entered_at"] for e in events]
    assert times == sorted(times)


def test_overlay_covers_frames_and_carries_masks(api):
    overlay = api.overlay("synthetic-01")
    assert overlay["frame_count"] == 240
    assert len(overlay["entities"]) == 5
    first = overlay["frames"]["0"]
    assert first and "mask" in first[0]
    mask = first[0]["mask"]
    assert sum(mask["runs"]) == mask["rows"] * mask["cols"]


def test_overlay_entity_ids_agree_with_the_events_table(api):
    overlay_ids = {e["entity_id"] for e in api.overlay("synthetic-01")["entities"]}
    event_ids = {e["entity_id"] for e in api.events("synthetic-01")["events"]}
    assert event_ids <= overlay_ids, "a row must refer to an entity the video can show"


def test_canned_questions_are_served_with_their_sql(api):
    questions = api.canned()["questions"]
    assert len(questions) >= 10
    assert all(q["sql"].strip() for q in questions)


def test_schema_is_served_for_the_prompt(api):
    assert "CREATE TABLE events" in api.schema()["schema"]


# ----- query routes ---------------------------------------------------------


def test_sql_route_returns_rows(api):
    payload = api.run_sql("SELECT zone, COUNT(*) AS n FROM events GROUP BY zone")
    assert payload["columns"] == ["zone", "n"]
    assert payload["row_count"] == 3


def test_sql_route_appends_the_limit(api):
    assert "LIMIT 200" in api.run_sql("SELECT * FROM events")["sql"]


@pytest.mark.parametrize(
    "sql",
    ["DROP TABLE events", "DELETE FROM events", "SELECT 1; DROP TABLE events"],
)
def test_sql_route_refuses_writes(api, sql):
    with pytest.raises(GuardError):
        api.run_sql(sql)


def test_ask_without_a_key_fails_clearly_rather_than_silently(api):
    with pytest.raises(RuntimeError) as err:
        api.ask("which zone is busiest?")
    assert "MODEL_API_KEY" in str(err.value)


def test_manifest_keeps_the_concept_mapping_after_an_ingest(api):
    entry = json.loads(api.manifest_path.read_text(encoding="utf-8"))["synthetic-01"]
    assert entry["concepts"]["vehicle"] == "forklift", (
        "ingest must not overwrite how the cached responses are keyed"
    )
    assert entry["zones_path"].endswith("synthetic-01.json")
    assert entry["zone_names"]


def test_manifest_records_how_to_reproduce_the_detections(tmp_path):
    """Anything a backend keys its cache on must be written beside the clip.

    Without this the overlay endpoint rebuilt the backend with default
    weights and image size, missed the cache keyed on the real ones, and
    tried to re-run the model.
    """
    import json as _json

    from spatialsight.backends.base import SegmentationBackend
    from spatialsight.backends.synthetic import SyntheticBackend as _Synth

    class Keyed(_Synth):
        name = "keyed"

        def manifest_extra(self) -> dict:
            return {"weights": "w.pt", "imgsz": 320}

    db = tmp_path / "db" / "m.sqlite"
    run(
        backend=Keyed("synthetic-01"),
        clip="synthetic-01",
        zones_path="zones/synthetic-01.json",
        db_path=db,
    )
    entry = _json.loads((db.parent / "clips.json").read_text(encoding="utf-8"))["synthetic-01"]
    assert entry["weights"] == "w.pt"
    assert entry["imgsz"] == 320
    # A backend that declares nothing must still write a usable entry.
    assert isinstance(SegmentationBackend.manifest_extra(_Synth("x")), dict)
