"""The question -> SQL -> rows -> brief flow, driven by a stub model.

`ask()` otherwise has no coverage until an API key exists, which would make
the first real run of the retry path happen in front of a judge. The stub
replaces only the network call, so the prompt construction, the guard, the
retry and the brief are all the real code.
"""

from __future__ import annotations

import pytest

from spatialsight.backends.synthetic import SyntheticBackend
from spatialsight.query.nlsql import (
    BRIEF_SYSTEM_PROMPT,
    SQL_SYSTEM_PROMPT,
    ask,
    schema_context,
)
from spatialsight.runner import run


@pytest.fixture(scope="module")
def db(tmp_path_factory):
    path = tmp_path_factory.mktemp("nlsql") / "n.sqlite"
    run(
        backend=SyntheticBackend("synthetic-01"),
        clip="synthetic-01",
        zones_path="zones/synthetic-01.json",
        db_path=path,
    )
    return path


class StubClient:
    """Returns scripted replies and records what it was asked."""

    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)
        self.calls: list[tuple[str, str]] = []
        self.usage = {"input_tokens": 0, "output_tokens": 0, "calls": 0}

    def complete(self, system: str, user: str, **_kwargs) -> str:
        self.calls.append((system, user))
        self.usage["calls"] += 1
        return self.replies.pop(0) if self.replies else ""


# ----- the prompt -----------------------------------------------------------


def test_schema_context_reads_the_live_vocabulary(db):
    context = schema_context(db)
    assert "loading_dock" in context["zones"]
    assert set(context["classes"]) == {"person", "vehicle"}
    assert context["clips"] == ["synthetic-01"]


def test_the_prompt_pins_the_schema_and_the_vocabulary(db):
    context = schema_context(db)
    client = StubClient("SELECT zone FROM events")
    ask("anything", db, client, with_brief=False)
    system, _ = client.calls[0]
    assert "events(" in system and "entities(" in system
    assert "loading_dock" in system
    for zone in context["zones"]:
        assert zone in system


def test_the_prompt_warns_the_model_off_wall_clock_time(db):
    client = StubClient("SELECT zone FROM events")
    ask("what happened after 2pm?", db, client, with_brief=False)
    system, _ = client.calls[0]
    assert "no wall-clock time" in system or "wall-clock" in system


# ----- the happy path -------------------------------------------------------


def test_valid_sql_is_executed_and_the_limit_appended(db):
    client = StubClient("SELECT zone, dwell_s FROM events ORDER BY dwell_s DESC")
    answer = ask("longest dwells?", db, client, with_brief=False)
    assert answer.ok
    assert answer.result is not None
    assert answer.result.columns == ["zone", "dwell_s"]
    assert "LIMIT 200" in answer.sql
    assert len(client.calls) == 1


def test_markdown_fenced_sql_is_accepted(db):
    client = StubClient("```sql\nSELECT COUNT(*) AS n FROM events\n```")
    answer = ask("how many events?", db, client, with_brief=False)
    assert answer.ok
    assert answer.result is not None and answer.result.rows[0][0] == 6


def test_the_brief_is_written_over_the_rows(db):
    client = StubClient(
        "SELECT zone, COUNT(*) AS n FROM events GROUP BY zone",
        "The loading dock held the longest visits at 43.0s on average.",
    )
    answer = ask("which zone is busiest?", db, client)
    assert answer.ok
    assert answer.brief and "loading dock" in answer.brief.lower()
    brief_system, brief_user = client.calls[1]
    assert brief_system == BRIEF_SYSTEM_PROMPT
    # The brief must be given the actual rows, not just the question.
    assert "loading_dock" in brief_user
    # Rows go as a table, not indented JSON: one header line, then values.
    assert brief_user.splitlines()[-4] == "zone\tn"
    # And it must be told whether what it is looking at is the whole answer,
    # or it hedges about a complete result being too small.
    assert "complete: true" in brief_user


# ----- the retry ------------------------------------------------------------


def test_a_bad_column_is_retried_once_with_the_error_attached(db):
    client = StubClient(
        "SELECT no_such_column FROM events",
        "SELECT zone FROM events",
    )
    answer = ask("which zone?", db, client, with_brief=False)
    assert answer.ok, "the corrected query should succeed"
    assert len(client.calls) == 2
    _, retry_prompt = client.calls[1]
    assert "no_such_column" in retry_prompt
    assert "rejected" in retry_prompt.lower()


def test_a_refused_statement_is_retried_then_fails_visibly(db):
    client = StubClient("DROP TABLE events", "DELETE FROM events")
    answer = ask("clear yesterday", db, client, with_brief=False)
    assert not answer.ok
    assert answer.error and "SELECT" in answer.error
    # The SQL it tried is still reported, so the failure is inspectable.
    assert "DELETE" in answer.sql
    assert len(answer.attempts) == 2


def test_it_does_not_retry_more_than_once(db):
    client = StubClient("bad one", "bad two", "SELECT zone FROM events")
    answer = ask("x", db, client, with_brief=False)
    assert not answer.ok
    assert len(client.calls) == 2, "a second retry mostly wastes the user's time"


def test_prose_instead_of_sql_fails_cleanly(db):
    client = StubClient(
        "I cannot answer that from this schema.",
        "I still cannot answer that.",
    )
    answer = ask("what is the weather?", db, client, with_brief=False)
    assert not answer.ok
    assert answer.error


def test_no_brief_is_written_when_the_query_failed(db):
    client = StubClient("DROP TABLE events", "DROP TABLE entities")
    answer = ask("delete it all", db, client)
    assert answer.brief is None
    assert client.usage["calls"] == 2, "a failed query must not be narrated"


def test_system_prompt_template_has_every_placeholder_filled(db):
    client = StubClient("SELECT 1")
    ask("x", db, client, with_brief=False)
    system, _ = client.calls[0]
    assert "{schema}" not in system
    assert "{zones}" not in system
    assert "{classes}" not in system
    assert "{clips}" not in system
    assert SQL_SYSTEM_PROMPT.count("{") == 4


# ----- the brief, where it needs no model -----------------------------------


def test_an_empty_result_needs_no_model_call(db):
    """A model handed zero rows invents a reason for them.

    There is exactly one true thing to say about an empty result, so it is
    said locally: correct by construction, and free.
    """
    client = StubClient("SELECT zone FROM events WHERE zone = 'nowhere'")
    answer = ask("who was in nowhere?", db, client)
    assert answer.ok
    assert answer.result is not None and not answer.result.rows
    assert answer.brief == "No rows matched."
    assert client.usage["calls"] == 1, "no brief call for an empty result"


def test_the_brief_is_told_when_a_result_was_cut_short(db):
    from spatialsight.query.guard import SqlGuard

    client = StubClient("SELECT zone FROM events", "A brief.")
    ask("every zone visit", db, client, guard=SqlGuard(max_limit=2))
    _, brief_user = client.calls[1]
    assert "complete: false" in brief_user


def test_only_the_first_rows_go_to_the_brief(db):
    from spatialsight.query.nlsql import BRIEF_MAX_ROWS

    client = StubClient("SELECT zone, dwell_s FROM events", "A brief.")
    answer = ask("all visits", db, client)
    _, brief_user = client.calls[1]
    body = brief_user.split("\n\n")[-1].splitlines()
    assert len(body) - 1 <= BRIEF_MAX_ROWS  # minus the header line
    assert f"rows: {len(answer.result.rows)}" in brief_user


# ----- the rules that exist because the model got them wrong ----------------


def test_the_prompt_rules_out_a_fake_median(db):
    client = StubClient("SELECT zone FROM events")
    ask("median dwell per zone", db, client, with_brief=False)
    system, _ = client.calls[0]
    assert "AVG is not one" in system
    assert "ROW_NUMBER() OVER" in system


def test_the_prompt_requires_a_self_join_to_exclude_itself(db):
    client = StubClient("SELECT zone FROM events")
    ask("who else was there when it left?", db, client, with_brief=False)
    system, _ = client.calls[0]
    assert "a.entity_id <> b.entity_id" in system and "exited_at, not" in system


def test_situational_rules_are_sent_only_when_the_question_needs_them(db):
    plain = StubClient("SELECT zone FROM events")
    ask("how many people were tracked?", db, plain, with_brief=False)
    assert "AVG is not one" not in plain.calls[0][0]
    assert "a.entity_id <> b.entity_id" not in plain.calls[0][0]

    rich = StubClient("SELECT zone FROM events")
    ask("median dwell per zone, and who else was there?", db, rich, with_brief=False)
    assert "AVG is not one" in rich.calls[0][0]
    assert "a.entity_id <> b.entity_id" in rich.calls[0][0]
    assert len(rich.calls[0][0]) > len(plain.calls[0][0])
