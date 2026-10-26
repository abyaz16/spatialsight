"""The SQL guard, including the attacks it is actually meant to stop.

These read like a pen-test because that is what they are. The model is an
untrusted author of SQL, and this is the boundary.
"""

from __future__ import annotations

import sqlite3

import pytest

from spatialsight.query.guard import GuardError, SqlGuard, guard_sql, run_readonly

# ----- what must be allowed ------------------------------------------------


def test_plain_select_passes():
    assert guard_sql("SELECT * FROM events").startswith("SELECT")


def test_cte_passes():
    sql = "WITH d AS (SELECT zone, dwell_s FROM events) SELECT zone FROM d"
    assert guard_sql(sql).startswith("WITH")


def test_markdown_fence_is_stripped():
    assert guard_sql("```sql\nSELECT 1\n```").startswith("SELECT 1")


def test_trailing_semicolon_is_tolerated():
    assert guard_sql("SELECT 1;").startswith("SELECT 1")


def test_limit_is_appended_when_missing():
    assert "LIMIT 200" in guard_sql("SELECT * FROM events")


def test_existing_limit_is_respected():
    assert guard_sql("SELECT * FROM events LIMIT 5").count("LIMIT") == 1


def test_column_named_like_a_keyword_is_not_rejected():
    # 'updated_at' contains 'update'; the check is on whole words.
    assert guard_sql("SELECT updated_at FROM events")


def test_string_literal_containing_a_keyword_is_not_rejected():
    assert guard_sql("SELECT * FROM events WHERE zone = 'delete bay'")


# ----- what must be refused ------------------------------------------------


@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE events",
        "DELETE FROM events",
        "UPDATE events SET dwell_s = 0",
        "INSERT INTO events VALUES (1)",
        "ALTER TABLE events ADD COLUMN x INT",
        "CREATE TABLE t (a INT)",
        "PRAGMA table_info(events)",
        "ATTACH DATABASE '/etc/passwd' AS p",
        "VACUUM",
    ],
)
def test_write_and_admin_statements_are_refused(sql):
    with pytest.raises(GuardError):
        guard_sql(sql)


def test_stacked_statement_is_refused():
    with pytest.raises(GuardError):
        guard_sql("SELECT 1; DROP TABLE events")


def test_statement_hidden_behind_a_line_comment_is_refused():
    # Stripping comments first is what catches this: the ';' is revealed.
    with pytest.raises(GuardError):
        guard_sql("SELECT 1 -- harmless\n; DROP TABLE events")


def test_statement_hidden_behind_a_block_comment_is_refused():
    with pytest.raises(GuardError):
        guard_sql("SELECT 1 /* x */ ; DELETE FROM events")


def test_keyword_disguised_by_a_block_comment_is_refused():
    with pytest.raises(GuardError):
        guard_sql("SELECT * FROM events WHERE 1=1 UNION SELECT 1; DROP/**/TABLE events")


def test_empty_input_is_refused():
    with pytest.raises(GuardError):
        guard_sql("   ")


def test_prose_instead_of_sql_is_refused():
    with pytest.raises(GuardError):
        guard_sql("I cannot answer that question.")


# ----- the engine-level layer ---------------------------------------------


@pytest.fixture()
def db(tmp_path):
    path = tmp_path / "g.sqlite"
    con = sqlite3.connect(path)
    con.executescript(
        "CREATE TABLE events (id INTEGER, zone TEXT, dwell_s REAL);"
        "INSERT INTO events VALUES (1,'dock',4.0),(2,'aisle',9.5),(3,'dock',2.5);"
    )
    con.commit()
    con.close()
    return path


def test_readonly_query_returns_rows(db):
    result = run_readonly(db, "SELECT zone, dwell_s FROM events ORDER BY dwell_s DESC")
    assert result.columns == ["zone", "dwell_s"]
    assert result.rows[0] == ("aisle", 9.5)


def test_aggregate_query_works(db):
    result = run_readonly(
        db, "SELECT zone, AVG(dwell_s) AS avg_dwell FROM events GROUP BY zone"
    )
    assert dict(result.rows)["dock"] == pytest.approx(3.25)


def test_limit_truncation_is_reported(db):
    result = run_readonly(db, "SELECT * FROM events", SqlGuard(max_limit=2))
    assert len(result.rows) == 2
    assert result.truncated is True


def test_write_through_readonly_connection_is_refused(db):
    with pytest.raises(GuardError):
        run_readonly(db, "DELETE FROM events")
    con = sqlite3.connect(db)
    try:
        assert con.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 3
    finally:
        con.close()


def test_the_database_is_untouched_after_a_refused_query(db):
    for attempt in ("DROP TABLE events", "UPDATE events SET zone='x'"):
        with pytest.raises(GuardError):
            run_readonly(db, attempt)
    con = sqlite3.connect(db)
    try:
        zones = {r[0] for r in con.execute("SELECT zone FROM events")}
        assert zones == {"dock", "aisle"}
    finally:
        con.close()


def test_nonexistent_column_is_a_clean_error_not_a_crash(db):
    with pytest.raises(GuardError) as err:
        run_readonly(db, "SELECT no_such_column FROM events")
    assert "rejected" in str(err.value).lower()


def test_a_model_supplied_limit_is_not_reported_as_truncation(db):
    """`LIMIT 1` for a superlative returns exactly what was asked for.

    Reporting it as truncated made the brief warn about missing data that
    nobody wanted.
    """
    result = run_readonly(db, "SELECT zone FROM events ORDER BY dwell_s DESC LIMIT 1")
    assert len(result.rows) == 1
    assert result.truncated is False


def test_our_own_cap_is_still_reported_as_truncation(db):
    result = run_readonly(db, "SELECT * FROM events", SqlGuard(max_limit=2))
    assert len(result.rows) == 2
    assert result.truncated is True
