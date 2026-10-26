"""Contract tests for the text-model client, over real HTTP against a stub.

`test_nlsql.py` stubs the client object and tests the prompt, the guard and
the retry *of bad SQL*. These test the layer underneath: the HTTP call, the
token accounting, and the transport failures that a stubbed client hides.
"""

from __future__ import annotations

import pytest

from fake_chat_api import FakeChatApi, FakeChatState
from spatialsight._retry import RetryPolicy, call_with_retries, classify
from spatialsight.backends.synthetic import SyntheticBackend
from spatialsight.query.nlsql import SparkClient, ask
from spatialsight.runner import run


@pytest.fixture(scope="module")
def db(tmp_path_factory):
    path = tmp_path_factory.mktemp("llm") / "q.sqlite"
    run(
        backend=SyntheticBackend("synthetic-01"),
        clip="synthetic-01",
        zones_path="zones/synthetic-01.json",
        db_path=path,
    )
    return path


def client_for(base_url, **kwargs):
    kwargs.setdefault("api_style", "chat")
    kwargs.setdefault("retry", RetryPolicy(max_attempts=4, base_delay_s=0.01, max_delay_s=0.05))
    return SparkClient("test-key", base_url, "stub-model", **kwargs)


# ----- the happy path ------------------------------------------------------


def test_a_question_becomes_sql_and_rows(db):
    state = FakeChatState(replies=["SELECT zone, dwell_s FROM events ORDER BY dwell_s DESC"])
    with FakeChatApi(state) as api:
        answer = ask("longest dwells?", db, client_for(api.base_url), with_brief=False)

    assert answer.ok
    assert answer.result is not None
    assert answer.result.columns == ["zone", "dwell_s"]
    assert "LIMIT 200" in answer.sql


def test_the_request_carries_the_system_prompt_and_the_key(db):
    state = FakeChatState(replies=["SELECT 1"])
    with FakeChatApi(state) as api:
        ask("anything", db, client_for(api.base_url), with_brief=False)

    request = state.requests[0]
    assert request["authorization"] == "Bearer test-key"
    roles = [m["role"] for m in request["body"]["messages"]]
    assert roles == ["system", "user"]
    assert "events(" in request["body"]["messages"][0]["content"]
    assert request["body"]["temperature"] == 0, (
        "SQL wants the single most likely answer, not a creative one"
    )


def test_token_usage_is_recorded_for_the_cost_line(db):
    state = FakeChatState(replies=["SELECT 1"])
    with FakeChatApi(state) as api:
        client = client_for(api.base_url)
        ask("anything", db, client, with_brief=False)

    assert client.usage["calls"] == 1
    assert client.usage["input_tokens"] == 42
    assert client.usage["output_tokens"] == 7


def test_a_local_endpoint_needs_no_key():
    """A llama.cpp server ignores the key; requiring one would block the
    offline path for no reason."""
    state = FakeChatState(replies=["SELECT 1"])
    with FakeChatApi(state) as api:
        client = SparkClient(None, api.base_url, "local-model", api_style="chat", require_key=False)
        assert client.complete("sys", "user") == "SELECT 1"


def test_a_missing_key_fails_with_an_actionable_message():
    with pytest.raises(RuntimeError) as err:
        SparkClient("", "https://api.meta.ai/v1", "muse-spark-1.3")
    assert "MODEL_API_KEY" in str(err.value)


# ----- transport failures --------------------------------------------------


def test_a_rate_limit_is_retried_then_answers(db):
    state = FakeChatState(replies=["SELECT zone FROM events"], fail_mode="rate_limit", fail_times=2)
    with FakeChatApi(state) as api:
        client = client_for(api.base_url)
        answer = ask("which zones?", db, client, with_brief=False)

    assert answer.ok, "a rate limit should cost a second, not the answer"
    assert client.usage["retries"] == 2
    assert len(state.requests) == 3


def test_a_server_error_is_retried(db):
    state = FakeChatState(replies=["SELECT zone FROM events"], fail_mode="server_error", fail_times=1)
    with FakeChatApi(state) as api:
        client = client_for(api.base_url)
        assert ask("which zones?", db, client, with_brief=False).ok
    assert client.usage["retries"] == 1


def test_a_bad_request_is_not_retried(db):
    state = FakeChatState(replies=["SELECT 1"], fail_mode="bad_request", fail_times=99)
    with FakeChatApi(state) as api:
        client = client_for(api.base_url)
        with pytest.raises(Exception):
            client.complete("sys", "user")
    assert len(state.requests) == 1, "a 400 will not become a 200 by repetition"


def test_retries_eventually_give_up(db):
    state = FakeChatState(replies=["SELECT 1"], fail_mode="server_error", fail_times=99)
    with FakeChatApi(state) as api:
        client = client_for(api.base_url, retry=RetryPolicy(max_attempts=3, base_delay_s=0.01))
        with pytest.raises(Exception):
            client.complete("sys", "user")
    assert len(state.requests) == 3


def test_an_empty_reply_fails_the_guard_rather_than_the_process(db):
    state = FakeChatState(replies=["", ""])
    with FakeChatApi(state) as api:
        answer = ask("anything", db, client_for(api.base_url), with_brief=False)
    assert not answer.ok
    assert answer.error


def test_the_sql_retry_and_the_transport_retry_are_independent(db):
    """A rejected query is retried once with the error attached; a 429 is
    retried by the transport. Neither should consume the other's budget."""
    state = FakeChatState(
        replies=["DROP TABLE events", "SELECT zone FROM events"],
        fail_mode="rate_limit",
        fail_times=1,
    )
    with FakeChatApi(state) as api:
        client = client_for(api.base_url)
        answer = ask("which zones?", db, client, with_brief=False)

    assert answer.ok, "the corrected SQL should still land"
    assert client.usage["retries"] == 1, "one transport retry"
    assert len(answer.attempts) == 2, "one SQL retry"


# ----- the shared policy ---------------------------------------------------


def test_classify_retries_transport_and_server_faults():
    import httpx
    import openai

    request = httpx.Request("POST", "http://x/v1/chat/completions")

    def status_error(code):
        response = httpx.Response(code, request=request)
        return openai.APIStatusError("boom", response=response, body=None)

    assert classify(openai.APIConnectionError(request=request))[0] is True
    assert classify(status_error(500))[0] is True
    assert classify(status_error(503))[0] is True
    assert classify(status_error(400))[0] is False
    assert classify(status_error(401))[0] is False
    assert classify(status_error(404))[0] is False
    assert classify(ValueError("not an API error"))[0] is False


def test_call_with_retries_stops_at_the_attempt_limit():
    attempts = []

    def always_fails():
        attempts.append(1)
        raise ConnectionResetError("dropped")

    from spatialsight._retry import RetryableError

    def retryable():
        attempts.append(1)
        raise RetryableError("try again")

    with pytest.raises(RetryableError):
        call_with_retries(
            retryable,
            policy=RetryPolicy(max_attempts=3, base_delay_s=0),
            sleep=lambda _: None,
        )
    assert len(attempts) == 3


def test_call_with_retries_returns_on_first_success():
    calls = []

    def succeeds_second_time():
        calls.append(1)
        if len(calls) < 2:
            from spatialsight._retry import RetryableError

            raise RetryableError("once")
        return "ok"

    result = call_with_retries(
        succeeds_second_time,
        policy=RetryPolicy(max_attempts=5, base_delay_s=0),
        sleep=lambda _: None,
    )
    assert result == "ok"
    assert len(calls) == 2
