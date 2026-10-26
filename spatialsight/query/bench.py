"""Execution accuracy for natural-language questions.

Each reference question is asked in plain English, and the rows the model's
SQL returns are compared with the rows the hand-written SQL returns. Comparing
*results* rather than SQL text is the point: there are many correct ways to
write a query, and only one correct answer.

Three levels are reported, because they mean different things to a user:

* **exact** -- the same rows, ignoring row order and column names. The answer
  is right and shaped the way it was asked for.
* **contains** -- every reference column can be found among the model's
  columns with identical values. Right, with extra columns a reader would not
  mind.
* **subset** -- the model returned the *top* of the reference's ranking, with
  a subset of its columns. "Which zone had the longest average dwell?" is
  correctly answered by one row; the reference query returns all three,
  ranked. Marking that wrong would understate the model badly, which is the
  whole reason this distinction exists.

The subset check is deliberately a **prefix** check, not a sub-multiset one.
A sub-multiset check would accept the *bottom* of a ranking as cheerfully as
the top, and the bottom is the opposite answer.

Anything else is a miss, kept with its SQL so it can be read. Numbers are
compared after rounding to one decimal, since `AVG` written two ways can
differ in the last float digit without being wrong.
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


def _norm_rows(rows) -> list[tuple]:
    return [tuple(_norm(v) for v in row) for row in rows]


def _columns(rows: list[tuple], width: int) -> list[Counter]:
    return [Counter(row[i] for row in rows) for i in range(width)]


def _embeds(wanted: list[Counter], available: list[Counter]) -> bool:
    """Can every `wanted` column be matched to a distinct `available` one?"""
    used: set[int] = set()
    for column in wanted:
        match = next(
            (j for j, candidate in enumerate(available) if j not in used and candidate == column),
            None,
        )
        if match is None:
            return False
        used.add(match)
    return True


def _is_top_k(ref: list[tuple], ref_width: int, model: list[tuple], model_width: int) -> bool:
    """Are the model's rows the reference's first k rows, with fewer columns?

    Each model column must equal some reference column over those k rows, in
    order, and no two model columns may claim the same reference column.
    """
    k = len(model)
    head = ref[:k]
    candidates = [
        {j for j in range(ref_width) if [row[j] for row in head] == [row[i] for row in model]}
        for i in range(model_width)
    ]
    if any(not c for c in candidates):
        return False

    def assign(i: int, used: frozenset[int]) -> bool:
        if i == model_width:
            return True
        return any(assign(i + 1, used | {j}) for j in candidates[i] if j not in used)

    return assign(0, frozenset())


def _covers(ref: list[tuple], wanted: list[int], model: list[tuple], model_width: int) -> bool:
    """Does the model's top-k carry every required reference column?"""
    head = ref[: len(model)]
    candidates = [
        {i for i in range(model_width) if [row[i] for row in model] == [row[j] for row in head]}
        for j in wanted
    ]
    if any(not c for c in candidates):
        return False

    def assign(n: int, used: frozenset[int]) -> bool:
        if n == len(candidates):
            return True
        return any(assign(n + 1, used | {i}) for i in candidates[n] if i not in used)

    return assign(0, frozenset())


def compare(reference_rows, reference_cols, model_rows, model_cols, required=()) -> str:
    """'exact', 'contains', 'subset', 'covers' or 'miss'."""
    ref = _norm_rows(reference_rows)
    model = _norm_rows(model_rows)
    ref_width, model_width = len(reference_cols), len(model_cols)

    if not ref and not model:
        return "exact"
    if not model or not model_width:
        return "miss"

    if ref_width == model_width and Counter(ref) == Counter(model):
        return "exact"

    if model_width > ref_width and len(model) == len(ref):
        if _embeds(_columns(ref, ref_width), _columns(model, model_width)):
            return "contains"

    # Required columns gate every remaining verdict. Without this, a model
    # that answers "the five longest dwells, with the frame to seek to" using
    # only entity_id and zone is a strict subset of the reference and would
    # pass, having returned neither the dwell nor the frame.
    wanted = [reference_cols.index(c) for c in required if c in reference_cols]
    has_required = (
        len(wanted) == len(required) and _covers(ref, wanted, model, model_width)
        if required
        else True
    )

    if has_required and len(model) <= len(ref):
        if model_width <= ref_width and _is_top_k(ref, ref_width, model, model_width):
            return "subset"
        # A different projection, carrying everything the question asked for.
        if required:
            return "covers"

    return "miss"


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
    def subset(self) -> int:
        return sum(1 for i in self.items if i.verdict == "subset")

    @property
    def covers(self) -> int:
        return sum(1 for i in self.items if i.verdict == "covers")

    @property
    def answered(self) -> int:
        return self.exact + self.contains + self.subset + self.covers

    def lines(self) -> list[str]:
        out = [f"NL->SQL benchmark: {self.model} via {self.endpoint}", ""]
        for item in self.items:
            mark = {
                "exact": "PASS ",
                "contains": "PASS+",
                "subset": "PASS-",
                "covers": "PASS~",
                "miss": "MISS ",
                "error": "FAIL ",
            }[item.verdict]
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
            f"  subset    {self.subset}/{n}   (top of the ranking, fewer columns)",
            f"  covers    {self.covers}/{n}   (different columns, all required ones present)",
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
            "subset": self.subset,
            "covers": self.covers,
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
            reference.rows,
            reference.columns,
            answer.result.rows,
            answer.result.columns,
            required=q.must_have,
        )
        report.items.append(
            BenchItem(
                q.key, q.question, verdict, answer.sql, None, elapsed, len(answer.attempts)
            )
        )
    report.usage = dict(client.usage)
    return report
