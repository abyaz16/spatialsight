"""Execution accuracy for natural-language questions.

Each reference question is asked in plain English, and the rows the model's
SQL returns are compared with the rows the hand-written SQL returns. Comparing
*results* rather than SQL text is the point: there are many correct ways to
write a query, and only one correct answer.

Two levels are reported, because they mean different things to a user:

* **exact** -- the same rows, ignoring row order and column names. The answer
  is right and shaped the way it was asked for.
* **contains** -- every reference column can be found among the model's
  columns with identical values. The answer is right but carries extra
  columns, which a person reading the table would not mind.

Anything else is a miss, and the miss is kept with its SQL so it can be read.
Numbers are compared after rounding to one decimal, since `AVG` written two
ways can differ in the last float digit without being wrong.
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .canned import QUESTIONS, CannedQuestion
from .guard import GuardError, SqlGuard, run_readonly
from .nlsql import SparkClient, ask


def _norm(value):
    if isinstance(value, float):
        return round(value, 1)
    if isinstance(value, int):
        return round(float(value), 1)
    if isinstance(value, str):
        # GROUP_CONCAT order is unspecified; compare the set it names.
        if "," in value:
            return ",".join(sorted(part.strip() for part in value.split(",")))
        return value.strip()
    return value


def _rows(rows) -> Counter:
    return Counter(tuple(_norm(v) for v in row) for row in rows)


def _columns(rows, width: int) -> list[Counter]:
    return [Counter(_norm(row[i]) for row in rows) for i in range(width)]


def compare(reference_rows, reference_cols, model_rows, model_cols) -> str:
    """'exact', 'contains' or 'miss'."""
    if _rows(reference_rows) == _rows(model_rows) and len(reference_cols) == len(model_cols):
        return "exact"
    if not reference_rows and not model_rows:
        return "exact"
    if len(reference_rows) != len(model_rows) or not model_cols:
        return "miss"
    # Can every reference column be matched to a distinct model column with the
    # same multiset of values? Greedy is enough at these widths.
    ref = _columns(reference_rows, len(reference_cols))
    mod = _columns(model_rows, len(model_cols))
    used: set[int] = set()
    for column in ref:
        match = next(
            (j for j, candidate in enumerate(mod) if j not in used and candidate == column),
            None,
        )
        if match is None:
            return "miss"
        used.add(match)
    return "contains"


@dataclass
class BenchItem:
    key: str
    question: str
    verdict: str
    sql: str
    error: str | None = None
    seconds: float = 0.0
    attempts: int = 0


@dataclass
class BenchReport:
    model: str
    endpoint: str
    items: list[BenchItem] = field(default_factory=list)
    usage: dict = field(default_factory=dict)

    @property
    def exact(self) -> int:
        return sum(1 for i in self.items if i.verdict == "exact")

    @property
    def contains(self) -> int:
        return sum(1 for i in self.items if i.verdict == "contains")

    @property
    def answered(self) -> int:
        return self.exact + self.contains

    def lines(self) -> list[str]:
        out = [f"NL->SQL benchmark: {self.model} via {self.endpoint}", ""]
        for item in self.items:
            mark = {"exact": "PASS ", "contains": "PASS+", "miss": "MISS ", "error": "FAIL "}[
                item.verdict
            ]
            retry = " (retried)" if item.attempts > 1 else ""
            out.append(f"  {mark} {item.key:<28} {item.seconds:5.1f}s{retry}")
            if item.verdict in ("miss", "error"):
                if item.error:
                    out.append(f"         error: {item.error}")
                first = item.sql.strip().splitlines()
                if first:
                    out.append(f"         sql:   {first[0][:100]}")
        n = len(self.items)
        out += [
            "",
            f"  exact     {self.exact}/{n}",
            f"  contains  {self.contains}/{n}   (right rows, extra columns)",
            f"  answered  {self.answered}/{n}",
        ]
        if self.usage:
            out.append(f"  tokens    {self.usage}")
        return out

    def as_dict(self) -> dict:
        return {
            "model": self.model,
            "endpoint": self.endpoint,
            "exact": self.exact,
            "contains": self.contains,
            "answered": self.answered,
            "total": len(self.items),
            "usage": self.usage,
            "items": [
                {
                    "key": i.key,
                    "verdict": i.verdict,
                    "sql": i.sql,
                    "error": i.error,
                    "seconds": round(i.seconds, 2),
                    "attempts": i.attempts,
                }
                for i in self.items
            ],
        }


def run_bench(
    db_path: str | Path,
    client: SparkClient,
    *,
    questions: tuple[CannedQuestion, ...] = QUESTIONS,
    endpoint: str = "",
) -> BenchReport:
    guard = SqlGuard()
    report = BenchReport(model=client.model, endpoint=endpoint)
    for q in questions:
        reference = run_readonly(db_path, q.sql, guard)
        started = time.perf_counter()
        try:
            answer = ask(q.question, db_path, client, guard=guard, with_brief=False)
        except GuardError as exc:  # pragma: no cover - ask() converts these
            report.items.append(BenchItem(q.key, q.question, "error", "", str(exc)))
            continue
        elapsed = time.perf_counter() - started
        if not answer.ok or answer.result is None:
            report.items.append(
                BenchItem(
                    q.key, q.question, "error", answer.sql, answer.error,
                    elapsed, len(answer.attempts),
                )
            )
            continue
        verdict = compare(
            reference.rows, reference.columns, answer.result.rows, answer.result.columns
        )
        report.items.append(
            BenchItem(
                q.key, q.question, verdict, answer.sql, None, elapsed, len(answer.attempts)
            )
        )
    report.usage = dict(client.usage)
    return report
