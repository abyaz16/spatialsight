"""The question cache: what it must reuse, and what it must never reuse.

The risk a cache introduces is not a miss, it is a confident stale hit. These
tests are mostly about the second kind.
"""

from __future__ import annotations

import pytest

from spatialsight.backends.synthetic import SyntheticBackend
from spatialsight.query.nlsql import ask
from spatialsight.query.qcache import QuestionCache, default_cache
from spatialsight.runner import run

from test_nlsql import StubClient


@pytest.fixture()
def db(tmp_path):
    path = tmp_path / "c.sqlite"
    run(
        backend=SyntheticBackend("synthetic-01"),
        clip="synthetic-01",
        zones_path="zones/synthetic-01.json",
        db_path=path,
    )
    return path


@pytest.fixture()
def cache(tmp_path):
    return QuestionCache(path=tmp_path / "q.json")


VOCAB = {"zones": ["dock"], "classes": ["person"], "clips": ["a"]}


# ----- the key --------------------------------------------------------------


def test_case_and_whitespace_do_not_split_the_key(cache):
    cache.put_sql("Which zone is busiest?", VOCAB, "SELECT 1")
    assert cache.get_sql("  which   zone is BUSIEST? ", VOCAB) == "SELECT 1"


def test_a_different_question_misses(cache):
    cache.put_sql("which zone is busiest?", VOCAB, "SELECT 1")
    assert cache.get_sql("which zone is quietest?", VOCAB) is None


def test_new_vocabulary_invalidates_the_entry(cache):
    """Ingesting a clip with new zone names must not serve the old query.

    'Which zone was busiest?' has a different correct answer once there are
    more zones, and possibly different correct SQL. A key on the question
    alone would serve the stale query forever.
    """
    cache.put_sql("which zone is busiest?", VOCAB, "SELECT 1")
    wider = {**VOCAB, "zones": ["dock", "aisle"]}
    assert cache.get_sql("which zone is busiest?", wider) is None


def test_a_new_clip_invalidates_the_entry(cache):
    cache.put_sql("how many people?", VOCAB, "SELECT 1")
    assert cache.get_sql("how many people?", {**VOCAB, "clips": ["a", "b"]}) is None


def test_the_brief_key_follows_the_rows(cache):
    cols, rows = ["zone"], [("dock",)]
    cache.put_brief("q", cols, rows, "The dock.")
    assert cache.get_brief("q", cols, rows) == "The dock."
    assert cache.get_brief("q", cols, [("aisle",)]) is None


# ----- persistence ----------------------------------------------------------


def test_entries_survive_a_new_instance(tmp_path):
    path = tmp_path / "q.json"
    QuestionCache(path=path).put_sql("q", VOCAB, "SELECT 1")
    assert QuestionCache(path=path).get_sql("q", VOCAB) == "SELECT 1"


def test_a_corrupt_cache_is_a_cold_cache_not_a_crash(tmp_path):
    path = tmp_path / "q.json"
    path.write_text("{not json", encoding="utf-8")
    assert QuestionCache(path=path).get_sql("q", VOCAB) is None


def test_a_stale_version_is_ignored(tmp_path):
    path = tmp_path / "q.json"
    path.write_text('{"version": 0, "entries": {"x": "SELECT 1"}}', encoding="utf-8")
    assert QuestionCache(path=path).stats()["entries"] == 0


def test_disabled_cache_stores_and_returns_nothing(tmp_path):
    c = QuestionCache(path=tmp_path / "q.json", enabled=False)
    c.put_sql("q", VOCAB, "SELECT 1")
    assert c.get_sql("q", VOCAB) is None
    assert not (tmp_path / "q.json").exists()


def test_clear_empties_it(cache):
    cache.put_sql("q", VOCAB, "SELECT 1")
    assert cache.clear() == 1
    assert cache.get_sql("q", VOCAB) is None


def test_the_default_cache_lives_beside_the_database(tmp_path):
    c = default_cache(tmp_path / "db" / "spatialsight.sqlite")
    assert c.path.parent.name == "cache"


# ----- the saving, end to end ----------------------------------------------


def test_a_repeated_question_costs_no_model_calls(db, cache):
    client = StubClient("SELECT zone FROM events", "A brief.")
    first = ask("which zones exist?", db, client, cache=cache)
    assert first.ok
    assert client.usage["calls"] == 2  # SQL, then brief

    again = ask("which zones exist?", db, client, cache=cache)
    assert again.ok
    assert client.usage["calls"] == 2, "a cached question must not call the model"
    assert again.usage["sql_from_cache"] is True
    assert again.brief == first.brief
    assert again.result.rows == first.result.rows


def test_the_cached_sql_is_re_executed_so_rows_stay_live(db, cache):
    """The cache holds the query, never the answer.

    If it held rows, an ingest would make every cached answer silently wrong.
    """
    client = StubClient("SELECT COUNT(*) AS n FROM events")
    first = ask("how many events?", db, client, with_brief=False, cache=cache)
    before = first.result.rows[0][0]

    import sqlite3

    con = sqlite3.connect(db)
    con.execute(
        "INSERT INTO events (clip, entity_id, class, zone, entered_at, exited_at,"
        " dwell_s, frame_start, frame_end, peak_overlap, closed)"
        " VALUES ('synthetic-01', 1, 'person', 'loading_dock', 0, 1, 1, 0, 2, 1.0, 1)"
    )
    con.commit()
    con.close()

    again = ask("how many events?", db, client, with_brief=False, cache=cache)
    assert again.usage["sql_from_cache"] is True
    assert again.result.rows[0][0] == before + 1


def test_sql_that_no_longer_runs_falls_back_to_the_model(db, cache):
    cache.put_sql("anything", _vocab(db), "SELECT gone FROM events")
    client = StubClient("SELECT zone FROM events")
    answer = ask("anything", db, client, with_brief=False, cache=cache)
    assert answer.ok, "a dead cached query must not surface as a failure"
    assert client.usage["calls"] == 1


def test_a_failed_question_is_not_cached(db, cache):
    client = StubClient("SELECT nope FROM events", "SELECT nope FROM events")
    answer = ask("bad", db, client, with_brief=False, cache=cache)
    assert not answer.ok
    assert cache.get_sql("bad", _vocab(db)) is None


def _vocab(db):
    from spatialsight.query.nlsql import schema_context

    return schema_context(db)
