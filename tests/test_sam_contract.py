"""Contract tests for the SAM 3.1 client, over real HTTP against a stub API.

These are the tests that decide whether the morning the key arrives is dull
or expensive. Everything the client does short of reaching Meta's own servers
runs here: the OpenAI SDK, the request we build, the SSE stream we consume,
the official parser, the mask codec, the cache, the retry policy and the
budget guard.

The failure cases carry most of the weight. A vendor integration tested only
on the happy path is untested, and the costly failures are the quiet ones --
a truncated stream whose bytes are all valid, a 429 during a burst, a refusal
that arrives as HTTP 200.
"""

from __future__ import annotations

import json

import pytest

from fake_meta_api import FakeApiState, FakeMetaApi
from spatialsight.backends.sam import (
    MAX_VIDEO_FRAMES,
    BudgetExceededError,
    IncompleteResponseError,
    RefusedError,
    RetryPolicy,
    SamBackend,
    SamError,
)
from spatialsight.backends.synthetic import SyntheticBackend
from spatialsight.fixtures import encode_scene


@pytest.fixture(scope="module")
def wire_deltas():
    """Real SAM wire format: the simulated scene, encoded as the API returns it.

    Trimmed to the opening frames. These tests are about the protocol, not
    the scene, and encoding base85 masks for 240 frames made them slow enough
    to discourage running them.
    """
    backend = SyntheticBackend("contract")
    meta = backend.meta()
    opening = [d for d in backend.detect() if d.frame < 12]
    return encode_scene(opening, meta, "person")


@pytest.fixture()
def clip(tmp_path):
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"video-bytes" * 64)
    return path


def make_backend(clip, cache_dir, base_url, **kwargs):
    kwargs.setdefault("allow_spend", True)
    kwargs.setdefault("concept_for", {"person": "person"})
    backend = SamBackend(
        clip, api_key="test-key", api_base=base_url, cache_dir=cache_dir, **kwargs
    )
    # Probing a fake MP4 would need ffprobe; the clip's real metadata is not
    # what these tests are about.
    backend._meta = _meta_for(clip)
    return backend


def _meta_for(clip):
    from spatialsight.types import VideoMeta

    return VideoMeta(
        path=str(clip),
        width=1280,
        height=720,
        fps=2.0,
        frame_count=240,
        duration_s=120.0,
        sha256="contract-test-hash",
    )


# ----- the happy path, end to end over HTTP --------------------------------


def test_a_complete_response_is_parsed_into_detections(clip, tmp_path, wire_deltas):
    state = FakeApiState(deltas=wire_deltas)
    with FakeMetaApi(state) as api:
        backend = make_backend(clip, tmp_path / "cache", api.base_url)
        detections = backend.detect(["person"])

    assert detections, "the stub returned records; the client must surface them"
    assert all(d.label == "person" for d in detections)
    assert all(d.mask is not None for d in detections), "masks must decode"
    assert backend.record.requests_billed == 1


def test_the_request_matches_the_documented_shape(clip, tmp_path, wire_deltas):
    state = FakeApiState(deltas=wire_deltas)
    with FakeMetaApi(state) as api:
        make_backend(clip, tmp_path / "cache", api.base_url).detect(["person"])

    assert state.uploads, "the clip must be uploaded to the Files API first"
    body = state.requests[0]["body"]
    assert body["model"] == "sam-3.1"
    assert body["metadata"]["mask_encoding"] == "one_bit"
    assert body["metadata"]["include_confidence"] == "true"

    content = body["input"][0]["content"]
    kinds = {part["type"] for part in content}
    assert kinds == {"input_text", "input_video"}, (
        "a concept prompt and the clip, and nothing else"
    )
    text = next(p for p in content if p["type"] == "input_text")["text"]
    assert text == "person", "a concept prompt is one short noun phrase"
    video = next(p for p in content if p["type"] == "input_video")
    assert video["file_id"] == "file-stub-001", "reference the upload by id"
    assert state.requests[0]["authorization"] == "Bearer test-key"


def test_object_ids_are_namespaced_per_class(clip, tmp_path, wire_deltas):
    """SAM numbers objects per request, so two concepts both return object 1.

    Without namespacing the tracker would fuse a person and a forklift.
    """
    state = FakeApiState(deltas=wire_deltas)
    with FakeMetaApi(state) as api:
        backend = make_backend(
            clip,
            tmp_path / "cache",
            api.base_url,
            concept_for={"person": "person", "vehicle": "forklift"},
        )
        detections = backend.detect(["person", "vehicle"])

    assert state.response_count == 2, "one request per concept, as documented"
    prefixes = {d.object_id.split(":")[0] for d in detections}
    assert prefixes == {"person", "vehicle"}


def test_a_second_run_is_served_from_cache_and_costs_nothing(clip, tmp_path, wire_deltas):
    state = FakeApiState(deltas=wire_deltas)
    cache = tmp_path / "cache"
    with FakeMetaApi(state) as api:
        first = make_backend(clip, cache, api.base_url).detect(["person"])
        backend = make_backend(clip, cache, api.base_url)
        second = backend.detect(["person"])

    assert state.response_count == 1, "the second run must not reach the API"
    assert backend.record.requests_billed == 0
    assert backend.record.est_usd == 0.0
    assert len(first) == len(second)


# ----- the failures that cost money or corrupt data ------------------------


def test_a_truncated_stream_is_refused_and_never_cached(clip, tmp_path, wire_deltas):
    """The dangerous one: valid bytes, silently incomplete.

    If this is accepted, the cache holds a clip whose second half contains
    nobody, the events table agrees, and nothing anywhere reports a problem.
    """
    state = FakeApiState(
        deltas=wire_deltas, fail_mode="truncate", fail_times=99, truncate_after=3
    )
    cache = tmp_path / "cache"
    with FakeMetaApi(state) as api:
        backend = make_backend(
            clip, cache, api.base_url, retry=RetryPolicy(max_attempts=1)
        )
        with pytest.raises(IncompleteResponseError) as err:
            backend.detect(["person"])

    assert "truncated" in str(err.value)
    assert not list(cache.glob("*.json")), "a partial response must not be cached"


def test_a_rate_limit_is_retried_and_then_succeeds(clip, tmp_path, wire_deltas):
    state = FakeApiState(deltas=wire_deltas, fail_mode="rate_limit", fail_times=2)
    with FakeMetaApi(state) as api:
        backend = make_backend(
            clip,
            tmp_path / "cache",
            api.base_url,
            retry=RetryPolicy(max_attempts=4, base_delay_s=0.01, max_delay_s=0.05),
        )
        detections = backend.detect(["person"])

    assert detections
    assert backend.record.retries == 2
    assert state.response_count == 3, "two refusals then the real call"


def test_a_server_error_is_retried(clip, tmp_path, wire_deltas):
    state = FakeApiState(deltas=wire_deltas, fail_mode="server_error", fail_times=1)
    with FakeMetaApi(state) as api:
        backend = make_backend(
            clip,
            tmp_path / "cache",
            api.base_url,
            retry=RetryPolicy(max_attempts=3, base_delay_s=0.01, max_delay_s=0.05),
        )
        assert backend.detect(["person"])
    assert backend.record.retries == 1


def test_a_bad_request_is_not_retried(clip, tmp_path, wire_deltas):
    """A 400 is us being wrong. Retrying spends the same money for the same error."""
    state = FakeApiState(deltas=wire_deltas, fail_mode="bad_request", fail_times=99)
    with FakeMetaApi(state) as api:
        backend = make_backend(
            clip,
            tmp_path / "cache",
            api.base_url,
            retry=RetryPolicy(max_attempts=4, base_delay_s=0.01),
        )
        with pytest.raises(Exception) as err:
            backend.detect(["person"])

    assert "400" in str(err.value) or "invalid" in str(err.value).lower()
    assert state.response_count == 1, "a client error must be fatal, not retried"
    assert backend.record.retries == 0


def test_retries_give_up_rather_than_loop_forever(clip, tmp_path, wire_deltas):
    state = FakeApiState(deltas=wire_deltas, fail_mode="server_error", fail_times=99)
    with FakeMetaApi(state) as api:
        backend = make_backend(
            clip,
            tmp_path / "cache",
            api.base_url,
            retry=RetryPolicy(max_attempts=3, base_delay_s=0.01, max_delay_s=0.02),
        )
        with pytest.raises(Exception):
            backend.detect(["person"])
    assert state.response_count == 3


def test_a_refusal_is_reported_as_a_refusal(clip, tmp_path, wire_deltas):
    state = FakeApiState(deltas=wire_deltas, fail_mode="refusal", fail_times=99)
    with FakeMetaApi(state) as api:
        backend = make_backend(
            clip, tmp_path / "cache", api.base_url, retry=RetryPolicy(max_attempts=1)
        )
        with pytest.raises(RefusedError):
            backend.detect(["person"])


def test_a_concept_with_no_matches_is_a_normal_empty_result(clip, tmp_path):
    """The docs call this a normal result, not an error.

    A loading dock with no forklift in it must yield zero rows, not a crash.
    """
    state = FakeApiState(deltas=[], fail_mode="empty", fail_times=99)
    cache = tmp_path / "cache"
    with FakeMetaApi(state) as api:
        backend = make_backend(clip, cache, api.base_url)
        assert backend.detect(["person"]) == []
    assert list(cache.glob("*.json")), "an empty answer is still a paid answer"


# ----- spending ------------------------------------------------------------


def test_spending_is_refused_unless_explicitly_allowed(clip, tmp_path, wire_deltas):
    state = FakeApiState(deltas=wire_deltas)
    with FakeMetaApi(state) as api:
        backend = make_backend(clip, tmp_path / "cache", api.base_url, allow_spend=False)
        with pytest.raises(SamError) as err:
            backend.detect(["person"])
    assert "--allow-spend" in str(err.value)
    assert state.response_count == 0, "a cache miss must not silently bill"


def test_the_budget_is_a_hard_ceiling(clip, tmp_path, wire_deltas):
    """allow_spend permits spending; it does not make it unlimited."""
    state = FakeApiState(deltas=wire_deltas)
    with FakeMetaApi(state) as api:
        backend = make_backend(
            clip,
            tmp_path / "cache",
            api.base_url,
            budget_usd=0.01,  # one request costs 240 frames = $0.048
            concept_for={"person": "person", "vehicle": "forklift"},
        )
        with pytest.raises(BudgetExceededError) as err:
            backend.detect(["person", "vehicle"])

    assert "budget" in str(err.value)
    assert state.response_count == 0, "nothing may be spent once over budget"


def test_the_budget_stops_the_run_partway(clip, tmp_path, wire_deltas):
    state = FakeApiState(deltas=wire_deltas)
    with FakeMetaApi(state) as api:
        backend = make_backend(
            clip,
            tmp_path / "cache",
            api.base_url,
            budget_usd=0.06,  # enough for one request, not two
            concept_for={"person": "person", "vehicle": "forklift"},
        )
        with pytest.raises(BudgetExceededError):
            backend.detect(["person", "vehicle"])

    assert state.response_count == 1, "the first concept runs, the second is refused"
    assert backend.record.requests_billed == 1


def test_cost_is_estimated_from_the_documented_rate(clip, tmp_path):
    backend = make_backend(clip, tmp_path / "cache", "http://unused/v1")
    # 240 frames at $0.20 per 1,000 = $0.048 per concept.
    assert backend.cost_per_request_usd() == pytest.approx(0.048)
    assert backend.estimate_cost_usd() == pytest.approx(0.048)


def test_the_frame_limit_is_enforced_before_billing(clip, tmp_path):
    backend = make_backend(clip, tmp_path / "cache", "http://unused/v1")
    backend._meta = backend._meta.__class__(
        **{**backend._meta.__dict__, "frame_count": MAX_VIDEO_FRAMES + 1}
    )
    with pytest.raises(ValueError) as err:
        backend.detect(["person"])
    assert str(MAX_VIDEO_FRAMES) in str(err.value)


def test_the_run_record_is_reportable(clip, tmp_path, wire_deltas):
    """The submission quotes a measured cost per video-minute, not an estimate."""
    state = FakeApiState(deltas=wire_deltas)
    with FakeMetaApi(state) as api:
        backend = make_backend(clip, tmp_path / "cache", api.base_url)
        backend.detect(["person"])

    record = backend.record.as_dict()
    assert record["requests_billed"] == 1
    assert record["frames"] == 240
    assert record["est_usd_this_run"] == pytest.approx(0.048)
    assert record["concepts"] == ["person"]
    assert json.dumps(record), "the record must be serialisable for the write-up"


# ----- the retry policy itself ---------------------------------------------


def test_backoff_grows_and_is_capped():
    policy = RetryPolicy(base_delay_s=1.0, max_delay_s=10.0, jitter=0.0)
    assert policy.delay_for(1) == 1.0
    assert policy.delay_for(2) == 2.0
    assert policy.delay_for(3) == 4.0
    assert policy.delay_for(10) == 10.0, "backoff must not grow without bound"


def test_a_server_supplied_retry_after_wins():
    policy = RetryPolicy(base_delay_s=1.0, max_delay_s=30.0, jitter=0.0)
    assert policy.delay_for(4, retry_after=2.0) == 2.0


def test_jitter_spreads_synchronised_retries():
    """Two concepts retrying in lockstep would recreate the burst that caused
    the rate limit in the first place."""
    policy = RetryPolicy(base_delay_s=4.0, jitter=0.25)
    delays = {policy.delay_for(2) for _ in range(20)}
    assert len(delays) > 1
    assert all(6.0 <= d <= 10.0 for d in delays)
