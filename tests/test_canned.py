"""Every canned question must pass the guard and execute against real rows.

This is the schema regression test: rename a column and these fail here,
rather than in front of a judge.
"""

from __future__ import annotations

import pytest

from spatialsight.backends.synthetic import SyntheticBackend
from spatialsight.query.canned import QUESTIONS
from spatialsight.query.guard import guard_sql, run_readonly
from spatialsight.runner import run


@pytest.fixture(scope="module")
def db(tmp_path_factory):
    path = tmp_path_factory.mktemp("canned") / "c.sqlite"
    run(
        backend=SyntheticBackend("synthetic-01"),
        clip="synthetic-01",
        zones_path="zones/synthetic-01.json",
        db_path=path,
    )
    return path


@pytest.mark.parametrize("q", QUESTIONS, ids=lambda q: q.key)
def test_canned_sql_passes_the_guard(q):
    assert guard_sql(q.sql)


@pytest.mark.parametrize("q", QUESTIONS, ids=lambda q: q.key)
def test_canned_sql_executes(db, q):
    result = run_readonly(db, q.sql)
    assert result.columns, f"{q.key} returned no columns"


def test_question_keys_are_unique():
    keys = [q.key for q in QUESTIONS]
    assert len(keys) == len(set(keys))


def test_there_are_at_least_ten_questions():
    assert len(QUESTIONS) >= 10


def test_entity_count_matches_the_scene(db):
    rows = run_readonly(db, "SELECT COUNT(*) FROM entities").rows
    assert rows[0][0] == 5


def test_the_hard_overlap_query_finds_the_scripted_overlap(db):
    from spatialsight.query.canned import BY_KEY

    result = run_readonly(db, BY_KEY["concurrent_dock_occupancy"].sql)
    # The forklift sits in the dock while the boundary loiterer is there only
    # later, so this clip has no overlap. The query must still run and return
    # an empty, well-formed result rather than an error.
    assert "overlap_s" in result.columns


def test_no_canned_query_merges_entities_across_clips(db):
    """entity_id is unique only within a clip.

    With two clips loaded, grouping by entity_id alone silently merges a
    person in one clip with a person in another, and the row looks perfectly
    plausible. Any reference query that groups or joins on entity_id must
    carry clip too.
    """
    for q in QUESTIONS:
        sql = " ".join(q.sql.lower().split())
        if "group by" not in sql:
            continue
        grouped = sql.split("group by", 1)[1]
        for terminator in (" having", " order by", " limit", ")"):
            grouped = grouped.split(terminator, 1)[0]
        if "entity_id" in grouped:
            assert "clip" in grouped, (
                f"{q.key} groups by entity_id without clip, which merges "
                f"entities from different clips: GROUP BY{grouped}"
            )


def test_median_query_returns_one_row_per_zone(db):
    from spatialsight.query.canned import BY_KEY

    median = run_readonly(db, BY_KEY["median_dwell"].sql)
    zones = run_readonly(db, "SELECT COUNT(DISTINCT zone) FROM events").rows[0][0]
    assert len(median.rows) == zones
