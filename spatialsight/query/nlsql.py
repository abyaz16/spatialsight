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

import json
from dataclasses import dataclass, field
from pathlib import Path

from ..db import SCHEMA_FOR_PROMPT
from .guard import GuardError, QueryResult, SqlGuard, run_readonly

SQL_SYSTEM_PROMPT = """\
You translate questions about video surveillance events into SQLite queries.

The database has exactly these tables and columns:

{schema}

Zones available: {zones}
Entity classes available: {classes}
Clips available: {clips}

Rules:
- Reply with one SQLite SELECT statement and nothing else. No prose, no
  explanation, no markdown fence.
- Use only the tables and columns listed above.
- Times are seconds from the start of the clip. There is no wall-clock time
  and no date column. A question about "after 2pm" cannot be answered from
  this schema: return a query over the clip's own timeline instead, and do
  not invent a date column.
- Because the times are seconds, a question about minutes needs the division:
  CAST(entered_at / 60 AS INTEGER). Casting entered_at straight to an integer
  gives seconds, not minutes.
- Prefer returning frame_start alongside any row a user might want to seek
  to in the video.
- When the question implies a ranking, include ORDER BY.
- entity_id is unique only WITHIN a clip. Entity 1 of one clip is a
  different object from entity 1 of another, so any GROUP BY or JOIN on
  entity_id must include clip as well, or two unrelated objects merge
  into one.
- Never write anything other than a SELECT.\
"""

BRIEF_SYSTEM_PROMPT = """\
You write a short operations brief from query results about a fixed camera.

You are given the user's question, the SQL that answered it, and the rows.

Write at most four sentences. Rules:
- Only state what the rows support. Do not estimate, extrapolate or infer a
  cause that is not in the data.
- Quote the specific numbers that matter, with units.
- If the rows are too few or too uniform to support a conclusion, say exactly
  that in one sentence and stop.
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

        # A local server ignores the key, but the SDK insists on one.
        self.client = OpenAI(base_url=api_base, api_key=api_key or "local", timeout=120)
        self.model = model
        self.api_style = api_style
        self.usage: dict[str, int] = {"input_tokens": 0, "output_tokens": 0, "calls": 0}

    def complete(self, system: str, user: str, *, max_output_tokens: int = 700) -> str:
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

    attempts: list[str] = []
    prompt = question
    last_error = ""

    for attempt in range(2):
        raw = client.complete(system, prompt)
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

        brief = None
        if with_brief:
            brief = write_brief(question, result, client)
        return Answer(
            question=question,
            sql=result.sql,
            result=result,
            brief=brief,
            attempts=attempts,
            usage=dict(client.usage),
        )

    return Answer(
        question=question,
        sql=attempts[-1] if attempts else "",
        result=None,
        attempts=attempts,
        error=last_error,
        usage=dict(client.usage),
    )


def write_brief(question: str, result: QueryResult, client: SparkClient) -> str:
    """The bottleneck brief: prose over the rows, grounded in the rows."""
    rows = result.as_dicts()[:40]
    user = json.dumps(
        {
            "question": question,
            "sql": result.sql,
            "row_count": len(result.rows),
            "truncated": result.truncated,
            "rows": rows,
        },
        indent=2,
        default=str,
    )
    return client.complete(BRIEF_SYSTEM_PROMPT, user, max_output_tokens=400)
