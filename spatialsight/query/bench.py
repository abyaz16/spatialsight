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


def _round1(value: float) -> float:
    """Round half up, as SQLite's ROUND does, not half to even as Python does.

    `round(3.25, 1)` is 3.2 in Python and 3.3 in SQLite. A reference that
    rounds in SQL and a model that does not then disagree on a correct
    answer, and the exact median was being graded a miss for it.
    """
    from decimal import ROUND_HALF_UP, Decimal

    return float(Decimal(str(value)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def _norm(value):
    if isinstance(value, float):
        return _round1(value)
    if isinstance(value, int):
        return float(value)
    if isinstance(value, str):
        # GROUP_CONCAT order is unspecified; compare the set it names.
        if "," in value:
            return ",".join(sorted(part.strip() for part in value.split(",")))
        return value.strip()
    return value


def _norm_rows(rows) -> list[tuple]:
    return [tuple(_norm(v) for v in row) for row in rows]


def _aligns(
    src_rows: list[tuple],
    src_cols: list[int],
    dst_rows: list[tuple],
    dst_width: int,
    *,
    ordered: bool,
) -> bool:
    """Is there an injective map `src_cols` -> destination columns under which
    the two row sets agree?

    `ordered` compares row sequences position by position, which is what a
    top-k claim requires: if the model returned 1 of 3 rows, it is asserting
    that row is the *first*. When the model returned every row it is making
    no such claim, so row order is immaterial and multisets are compared --
    otherwise a correct answer sorted differently reads as wrong.

    The final check is on whole projected rows, not column by column. Matching
    each column's values independently would accept a table whose columns are
    individually right and whose rows pair them up wrongly.
    """
    if len(src_rows) != len(dst_rows):
        return False

    def src_col(j: int) -> list:
        return [row[j] for row in src_rows]

    def dst_col(i: int) -> list:
        return [row[i] for row in dst_rows]

    candidates: list[set[int]] = []
    for j in src_cols:
        values = src_col(j)
        if ordered:
            matches = {i for i in range(dst_width) if dst_col(i) == values}
        else:
            wanted = Counter(values)
            matches = {i for i in range(dst_width) if Counter(dst_col(i)) == wanted}
        if not matches:
            return False
        candidates.append(matches)

    projected_src = [tuple(row[j] for j in src_cols) for row in src_rows]

    def assign(n: int, used: frozenset[int], chosen: list[int]) -> bool:
        if n == len(candidates):
            projected_dst = [tuple(row[i] for i in chosen) for row in dst_rows]
            if ordered:
                return projected_src == projected_dst
            return Counter(projected_src) == Counter(projected_dst)
        for i in sorted(candidates[n]):
            if i not in used and assign(n + 1, used | {i}, chosen + [i]):
                return True
        return False

    return assign(0, frozenset(), [])


def compare(reference_rows, reference_cols, model_rows, model_cols, required=()) -> str:
    """'exact', 'contains', 'subset', 'covers' or 'miss'."""
    ref = _norm_rows(reference_rows)
    model = _norm_rows(model_rows)
    ref_width, model_width = len(reference_cols), len(model_cols)

    if not ref and not model:
        return "exact"
    if not model or not model_width or len(model) > len(ref):
        return "miss"

    if ref_width == model_width and Counter(ref) == Counter(model):
        return "exact"

    # Fewer rows than the reference is a claim about *which* rows: the top
    # ones. Equal rows is no such claim, so order stops mattering.
    ordered = len(model) < len(ref)
    head = ref[: len(model)]

    if model_width > ref_width and not ordered:
        if _aligns(ref, list(range(ref_width)), model, model_width, ordered=False):
            return "contains"

    # Required columns gate every remaining verdict. Without this, a model
    # that answers "the five longest dwells, with the frame to seek to" using
    # only entity_id and zone is a strict subset of the reference and would
    # pass, having returned neither the dwell nor the frame.
    wanted = [reference_cols.index(c) for c in required if c in reference_cols]
    has_required = (
        len(wanted) == len(required)
        and _aligns(head, wanted, model, model_width, ordered=ordered)
        if required
        else True
    )

    if has_required:
        if model_width <= ref_width and _aligns(
            model, list(range(model_width)), head, ref_width, ordered=ordered
        ):
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
