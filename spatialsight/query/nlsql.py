"""Natural language to SQL with Muse Spark, and the brief written over the rows.

Two non-obvious decisions:

**The schema is pinned in the system prompt on every call**, never summarised
and never cached across schema versions. It is ~30 lines; the cost of sending
it is trivial next to the cost of the model guessing a column name.

**A failed query is retried exactly once**, with the database's own error text
appended. One retry fixes the overwhelming majority of real failures -- a
misspelled column, a missing GROUP BY -- and a second retry mostly produces a
differently wrong query while the user waits. After that it fails visibly,
showing the SQL it tried. A wrong answer the user can see the SQL for is a
feature demo; a wrong answer they cannot is a liability.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from ..db import SCHEMA_FOR_PROMPT
from .guard import GuardError, QueryResult, SqlGuard, run_readonly
from .qcache import QuestionCache

# SQL is short. The old 700-token ceiling could not be reached by a correct
# answer, and a model that runs away from the task is better cut off than
# paid for.
SQL_MAX_TOKENS = 300

log = logging.getLogger("spatialsight.nlsql")

SQL_SYSTEM_PROMPT = """\
Translate questions about fixed-camera video into one SQLite SELECT.

Schema (use nothing else):
{schema}

zones: {zones}
classes: {classes}
clips: {clips}

Rules:
- Output the SELECT alone. No prose, no markdown fence.
- Rankings need ORDER BY. Include frame_start when a row is a moment someone
  would seek to in the video.
- Identity is (clip, entity_id), never entity_id alone, because entity_id
  restarts in every clip. Counting entities is
  COUNT(DISTINCT clip || ':' || entity_id), and any GROUP BY or JOIN keyed on
  entity_id carries clip too, or two unrelated objects merge into one.
  Grouping by zone or by time alone needs no clip.\
"""

# Rules that only matter for some questions. Sending all of them on every
# call cost ~230 tokens a question for rules irrelevant to most questions;
# the question itself says which apply.
SITUATIONAL_RULES: tuple[tuple[re.Pattern, str], ...] = (
    (
        re.compile(r"\b(minutes?|hourly|per hour)\b", re.I),
        "- Minutes need the division: CAST(entered_at / 60 AS INTEGER). "
        "Casting entered_at by itself gives seconds.",
    ),
    (
        re.compile(
            r"\b(same time|at that time|while|together|overlap\w*|who else|"
            r"other than|alongside|present|there when|with)\b",
            re.I,
        ),
        '- "At the same time" is an interval overlap: '
        "a.entered_at <= b.exited_at AND b.entered_at <= a.exited_at.\n"
        '  "Who else" / "with" means other entities, so a self-join must '
        "exclude the entity itself and stay in one clip:\n"
        "  FROM events a JOIN events b ON a.clip = b.clip "
        "AND a.entity_id <> b.entity_id AND <overlap>",
    ),
    (
        re.compile(r"\b(leave|leaves|left|leaving|exit\w*|depart\w*)\b", re.I),
        '- "Leave" / "left" means events.exited_at, not entities.last_seen.',
    ),
    (
        re.compile(r"\b(median|middle|typical)\b", re.I),
        "- SQLite has no median, and AVG is not one. Median per group, with the "
        "result named median_<measure> (e.g. median_dwell_s):\n"
        "  SELECT g, AVG(v) AS median_v FROM (SELECT g, v,\n"
        "    ROW_NUMBER() OVER (PARTITION BY g ORDER BY v) rn,\n"
        "    COUNT(*) OVER (PARTITION BY g) n FROM t)\n"
        "  WHERE rn IN ((n+1)/2, (n+2)/2) GROUP BY g",
    ),
)


def situational_rules(question: str) -> str:
    return "\n".join(t for pattern, t in SITUATIONAL_RULES if pattern.search(question))

BRIEF_SYSTEM_PROMPT = """\
Write a short operations brief over query results from a fixed camera.

At most four sentences. Rules:
- State only what the rows support. Do not estimate, extrapolate, or infer a
  cause that is not in the data.
- Quote the numbers that matter, with units.
- `complete: true` means these rows are the entire answer. Never call a
  complete result too small or too few: one row that answers the question is
  the answer, not a limitation. Say a result is thin only when
  `complete: false`.
- No preamble, no restating the question, no recommendations unless a number
  in the rows directly implies one.\
"""


@dataclass
class Answer:
    question: str
    sql: str
    result: QueryResult | None
    brief: str | None = None
    attempts: list[str] = field(default_factory=list)
    error: str | None = None
    usage: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.result is not None


def schema_context(db_path: str | Path) -> dict[str, list[str]]:
    """Read the live zone names, classes and clips to pin in the prompt.

    Giving the model the actual vocabulary of the data removes the single
    most common failure: inventing a zone called 'loading dock' when the row
    says 'loading_dock'.
    """
    import sqlite3

    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        def col(sql: str) -> list[str]:
            return [str(r[0]) for r in con.execute(sql) if r[0] is not None]

        return {
            "zones": col("SELECT DISTINCT zone FROM events ORDER BY zone"),
            "classes": col("SELECT DISTINCT class FROM entities ORDER BY class"),
            "clips": col("SELECT DISTINCT clip FROM entities ORDER BY clip"),
        }
    finally:
        con.close()


class SparkClient:
    """Thin wrapper over the Meta Model API for text completion."""

    def __init__(
        self,
        api_key: str | None,
        api_base: str,
        model: str,
        *,
        api_style: str = "responses",
        require_key: bool = True,
        timeout_s: float = 120.0,
        retry=None,
    ) -> None:
        if require_key and not api_key:
            raise RuntimeError(
                "MODEL_API_KEY is not set. Copy .env.example to .env and add the "
                "key from dev.meta.ai, or point LLM_BASE_URL at a local llama.cpp "
                "server (see scripts/llm)."
            )
        if api_style not in ("responses", "chat"):
            raise ValueError(f"api_style must be 'responses' or 'chat', not {api_style!r}")
        from openai import OpenAI

        from .._retry import RetryPolicy

        # A local server ignores the key, but the SDK insists on one.
        self.client = OpenAI(
            base_url=api_base,
            api_key=api_key or "local",
            timeout=timeout_s,
            max_retries=0,  # ours, so they can be counted and logged
        )
        self.model = model
        self.api_style = api_style
        self.retry = retry or RetryPolicy(max_attempts=3, base_delay_s=0.5, max_delay_s=10.0)
        self.usage: dict[str, int] = {
            "input_tokens": 0,
            "output_tokens": 0,
            "calls": 0,
            "retries": 0,
        }

    def complete(self, system: str, user: str, *, max_output_tokens: int = 700) -> str:
        """One completion, retrying only what is worth retrying.

        A rate limit mid-demo should cost a second, not the answer. A bad
        request should fail immediately rather than being sent three more
        times to produce the same error.
        """
        from .._retry import call_with_retries

        return call_with_retries(
            lambda: self._complete_once(system, user, max_output_tokens),
            policy=self.retry,
            on_retry=self._note_retry,
            what="completion",
        )

    def _note_retry(self, exc: Exception, attempt: int, delay: float) -> None:
        self.usage["retries"] = self.usage.get("retries", 0) + 1
        log.warning(
            "%s on attempt %d; retrying in %.1fs", type(exc).__name__, attempt, delay
        )

    def _complete_once(self, system: str, user: str, max_output_tokens: int) -> str:
        if self.api_style == "chat":
            return self._complete_chat(system, user, max_output_tokens)
        response = self.client.responses.create(
            model=self.model,
            instructions=system,
            input=user,
            max_output_tokens=max_output_tokens,
        )
        usage = getattr(response, "usage", None)
        if usage is not None:
            self.usage["input_tokens"] += getattr(usage, "input_tokens", 0) or 0
            self.usage["output_tokens"] += getattr(usage, "output_tokens", 0) or 0
        self.usage["calls"] += 1
        return (response.output_text or "").strip()

    def _complete_chat(self, system: str, user: str, max_output_tokens: int) -> str:
        """Chat completions, for llama.cpp and OpenAI-compatible free tiers."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=max_output_tokens,
            # SQL wants the single most likely answer, not a creative one.
            temperature=0,
        )
        usage = getattr(response, "usage", None)
        if usage is not None:
            self.usage["input_tokens"] += getattr(usage, "prompt_tokens", 0) or 0
            self.usage["output_tokens"] += getattr(usage, "completion_tokens", 0) or 0
        self.usage["calls"] += 1
        content = response.choices[0].message.content if response.choices else ""
        return (content or "").strip()

    @classmethod
    def from_config(cls, cfg) -> "SparkClient":
        return cls(
            cfg.llm_api_key,
            cfg.llm_base_url,
            cfg.llm_model,
            api_style=cfg.llm_api_style,
            require_key=not cfg.llm_explicit,
        )


def ask(
    question: str,
    db_path: str | Path,
    client: SparkClient,
    *,
    guard: SqlGuard | None = None,
    with_brief: bool = True,
    cache: "QuestionCache | None" = None,
) -> Answer:
    """Question in, validated SQL plus rows plus an optional brief out."""
    guard = guard or SqlGuard()
    context = schema_context(db_path)
    system = SQL_SYSTEM_PROMPT.format(
        schema=SCHEMA_FOR_PROMPT,
        zones=", ".join(context["zones"]) or "(none yet)",
        classes=", ".join(context["classes"]) or "(none yet)",
        clips=", ".join(context["clips"]) or "(none yet)",
    )

    extra_rules = situational_rules(question)
    if extra_rules:
        system = f"{system}\n{extra_rules}"

    attempts: list[str] = []
    prompt = question
    last_error = ""

    # A cached question still re-executes its SQL against the live database,
    # so the rows are never stale -- only the translation is reused. A cached
    # query that no longer runs (a dropped clip, a renamed zone) falls through
    # to the model instead of surfacing the error.
    if cache is not None:
        remembered = cache.get_sql(question, context)
        if remembered:
            try:
                result = run_readonly(db_path, remembered, guard)
            except GuardError:
                log.info("cached SQL no longer runs; re-asking the model")
            else:
                return _answer_with_brief(
                    question, result, client, attempts, with_brief, cache, cached_sql=True
                )

    for attempt in range(2):
        raw = client.complete(system, prompt, max_output_tokens=SQL_MAX_TOKENS)
        attempts.append(raw)
        try:
            result = run_readonly(db_path, raw, guard)
        except GuardError as exc:
            last_error = str(exc)
            if attempt == 0:
                # One retry, with the failure handed back verbatim.
                prompt = (
                    f"{question}\n\n"
                    f"Your previous answer was rejected.\n"
                    f"SQL you wrote:\n{raw}\n"
                    f"Error: {last_error}\n"
                    f"Return corrected SQL only."
                )
                continue
            return Answer(
                question=question,
                sql=raw,
                result=None,
                attempts=attempts,
                error=last_error,
                usage=dict(client.usage),
            )

        if cache is not None:
            cache.put_sql(question, context, result.sql)
        return _answer_with_brief(
            question, result, client, attempts, with_brief, cache
        )

    return Answer(
        question=question,
        sql=attempts[-1] if attempts else "",
        result=None,
        attempts=attempts,
        error=last_error,
        usage=dict(client.usage),
    )


def _answer_with_brief(
    question: str,
    result: QueryResult,
    client: SparkClient,
    attempts: list[str],
    with_brief: bool,
    cache: QuestionCache | None,
    *,
    cached_sql: bool = False,
) -> Answer:
    """Attach the brief, from cache where the rows are unchanged."""
    brief = None
    if with_brief:
        if cache is not None:
            brief = cache.get_brief(question, result.columns, result.rows)
        if brief is None:
            brief = write_brief(question, result, client)
            # An empty result's brief is a constant, so storing it would fill
            # the cache with copies of one sentence.
            if cache is not None and result.rows:
                cache.put_brief(question, result.columns, result.rows, brief)
    usage = dict(client.usage)
    usage["sql_from_cache"] = cached_sql
    return Answer(
        question=question,
        sql=result.sql,
        result=result,
        brief=brief,
        attempts=attempts,
        usage=usage,
    )


BRIEF_MAX_ROWS = 24


def _rows_as_tsv(result: QueryResult, limit: int = BRIEF_MAX_ROWS) -> str:
    """Rows as a header line plus tab-separated values.

    Indented JSON spent roughly three tokens per cell on punctuation and a
    repeated key name on every row. A table costs the header once. The model
    reads either equally well; only one of them is paid for per row.
    """
    lines = ["\t".join(result.columns)]
    for row in result.rows[:limit]:
        lines.append("\t".join("" if v is None else str(v) for v in row))
    return "\n".join(lines)


def write_brief(question: str, result: QueryResult, client: SparkClient) -> str:
    """The bottleneck brief: prose over the rows, grounded in the rows.

    Two results need no model at all. An empty result has nothing to write
    prose about, and a model handed zero rows invents a reason for them. Both
    answers below are exactly true by construction, and cost nothing.
    """
    if not result.rows:
        return "No rows matched."

    shown = min(len(result.rows), BRIEF_MAX_ROWS)
    header = (
        f"question: {question}\n"
        f"rows: {len(result.rows)}"
        f"{f' (first {shown} shown)' if shown < len(result.rows) else ''}\n"
        f"complete: {str(not result.truncated).lower()}\n"
    )
    return client.complete(
        BRIEF_SYSTEM_PROMPT,
        header + _rows_as_tsv(result),
        max_output_tokens=200,
    )
