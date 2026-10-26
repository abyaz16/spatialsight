"""Retry policy shared by both vendor clients.

One implementation, because two would mean the one you did not test is the
one that runs during the demo. SAM 3.1 and the text model are different APIs
with the same failure modes, and both are reached through the OpenAI SDK, so
they classify errors identically.

The central decision is *what not to retry*. A 429 or a 5xx is the far end
asking us to come back. A 400 is us being wrong: sending it again spends the
same money to receive the same error, three times more slowly.
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass
from typing import Callable, TypeVar

log = logging.getLogger("spatialsight.retry")

T = TypeVar("T")


@dataclass
class RetryPolicy:
    max_attempts: int = 4
    base_delay_s: float = 1.0
    max_delay_s: float = 30.0
    jitter: float = 0.25

    def delay_for(self, attempt: int, retry_after: float | None = None) -> float:
        """Exponential backoff, capped, jittered, and overridden by the server.

        The jitter matters because a clip issues one request per entity class.
        Synchronised retries would recreate the burst that caused the rate
        limit in the first place.
        """
        if retry_after is not None:
            return min(retry_after, self.max_delay_s)
        delay = min(self.base_delay_s * (2 ** (attempt - 1)), self.max_delay_s)
        return delay * (1.0 + random.uniform(-self.jitter, self.jitter))


class RetryableError(Exception):
    """Raise to tell `call_with_retries` that a failure is worth another go."""


def classify(exc: Exception) -> tuple[bool, float | None]:
    """Is this worth another attempt, and did the server say when?"""
    import openai

    if isinstance(exc, RetryableError):
        return True, None
    if isinstance(exc, (openai.APIConnectionError, openai.APITimeoutError)):
        return True, None
    if isinstance(exc, openai.RateLimitError):
        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", None) or {}
        try:
            raw = headers.get("retry-after") or headers.get("Retry-After")
            return True, float(raw) if raw is not None else None
        except (AttributeError, TypeError, ValueError):
            return True, None
    if isinstance(exc, openai.APIStatusError):
        # 5xx is the server's problem and may pass; 4xx is ours and will not.
        return exc.status_code >= 500, None
    return False, None


def call_with_retries(
    operation: Callable[[], T],
    *,
    policy: RetryPolicy | None = None,
    on_retry: Callable[[Exception, int, float], None] | None = None,
    what: str = "request",
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Run `operation`, retrying the failures that deserve it."""
    policy = policy or RetryPolicy()
    attempt = 0
    while True:
        attempt += 1
        try:
            return operation()
        except Exception as exc:  # noqa: BLE001 - re-raised when not retryable
            retryable, retry_after = classify(exc)
            if not retryable or attempt >= policy.max_attempts:
                raise
            delay = policy.delay_for(attempt, retry_after)
            if on_retry is not None:
                on_retry(exc, attempt, delay)
            else:
                log.warning(
                    "%s failed (%s); attempt %d of %d, retrying in %.1fs",
                    what,
                    type(exc).__name__,
                    attempt,
                    policy.max_attempts,
                    delay,
                )
            sleep(delay)
