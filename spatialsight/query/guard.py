"""Letting a language model write SQL against your database, safely.

A model that writes SQL will eventually write `DROP TABLE`, not from malice
but because a user asked it to "clear out yesterday" and it obliged. Asking
the model nicely not to is not a control. So there are four independent
layers here, and the SQL has to pass all of them:

1. **Statement shape.** One statement, and it must be a `SELECT` or a `WITH`
   that resolves to one. Comments are stripped first, because `--` is how you
   hide a second statement from a naive regex.
2. **Forbidden keywords**, checked against the stripped text as whole words,
   so a column named `updated_at` is not mistaken for an `UPDATE`.
3. **SQLite's own authorizer**, which vetoes anything but a read at the
   engine level, during preparation, before a row is touched. This is the
   layer that actually holds: it is enforced by the database, not by a
   pattern, so it catches whatever the first two layers did not imagine.
4. **A read-only connection**, so even a successful bypass has nothing to
   write to.

A `LIMIT` is then appended if the model did not supply one, which is a
courtesy to the UI rather than a safety measure.

Layer 3 is the one worth pointing at in a demo. The first two are guesses
about what an attacker might write; the third is a statement about what the
database will permit.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass

DEFAULT_LIMIT = 200

FORBIDDEN = frozenset(
    {
        "insert", "update", "delete", "drop", "alter", "create", "replace",
        "truncate", "attach", "detach", "pragma", "vacuum", "reindex",
        "grant", "revoke", "begin", "commit", "rollback", "savepoint",
        "load_extension", "writefile", "readfile", "edit",
    }
)

_LINE_COMMENT = re.compile(r"--[^\n]*")
_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
_STRING_LITERAL = re.compile(r"'(?:[^']|'')*'")
_LIMIT_CLAUSE = re.compile(r"\blimit\b", re.IGNORECASE)
_LIMIT_VALUE = re.compile(r"\blimit\s+(\d+)\s*$", re.IGNORECASE)
_FENCE = re.compile(r"^\s*```(?:sql)?\s*|\s*```\s*$", re.IGNORECASE)


class GuardError(Exception):
    """The SQL was rejected. The message is safe to show a user."""


@dataclass(frozen=True)
class SqlGuard:
    max_limit: int = DEFAULT_LIMIT

    def clean(self, sql: str) -> str:
        """Strip markdown fencing and trailing semicolons."""
        text = _FENCE.sub("", sql.strip())
        return text.strip().rstrip(";").strip()

    def _strip_for_analysis(self, sql: str) -> str:
        """Remove comments and string literals before looking for keywords.

        String literals go too, so a zone legitimately named 'delete bay'
        does not trip the keyword check.
        """
        text = _BLOCK_COMMENT.sub(" ", sql)
        text = _LINE_COMMENT.sub(" ", text)
        return _STRING_LITERAL.sub("''", text)

    def check(self, sql: str) -> str:
        sql = self.clean(sql)
        if not sql:
            raise GuardError("The model returned no SQL.")

        analysed = self._strip_for_analysis(sql)

        if ";" in analysed:
            raise GuardError("Only one statement is allowed.")

        head = analysed.lstrip().split(None, 1)[0].lower() if analysed.strip() else ""
        if head not in ("select", "with"):
            raise GuardError(
                f"Only SELECT queries are allowed; this one starts with {head.upper()!r}."
            )

        words = set(re.findall(r"[a-z_]+", analysed.lower()))
        banned = sorted(words & FORBIDDEN)
        if banned:
            raise GuardError(f"Query uses forbidden keyword(s): {', '.join(banned)}.")

        return sql

    def with_limit(self, sql: str) -> str:
        if _LIMIT_CLAUSE.search(self._strip_for_analysis(sql)):
            return sql
        return f"{sql}\nLIMIT {self.max_limit}"

    def effective_limit(self, sql: str) -> int:
        """The row cap the executed SQL actually carries.

        Needed because the cap lives in the SQL, so the only way to know a
        result was cut short is to notice it came back exactly at the cap.
        Over-fetching one extra row would mean executing SQL other than the
        SQL shown in the UI, which defeats the point of showing it.
        """
        match = _LIMIT_VALUE.search(self._strip_for_analysis(sql).strip())
        return int(match.group(1)) if match else self.max_limit


def _authorizer(action: int, _arg1, _arg2, _db_name, _trigger):
    """Veto anything that is not a read.

    SQLite calls this during statement preparation for every operation the
    statement would perform. Allowing only reads and the no-op action means
    an unforeseen write syntax is refused by the engine itself.
    """
    allowed = {
        sqlite3.SQLITE_SELECT,
        sqlite3.SQLITE_READ,
        sqlite3.SQLITE_FUNCTION,
    }
    if action in allowed:
        return sqlite3.SQLITE_OK
    return sqlite3.SQLITE_DENY


def guard_sql(sql: str, guard: SqlGuard | None = None) -> str:
    """Run layers 1 and 2, and append a LIMIT. Raises `GuardError`."""
    guard = guard or SqlGuard()
    return guard.with_limit(guard.check(sql))


@dataclass
class QueryResult:
    sql: str
    columns: list[str]
    rows: list[tuple]
    truncated: bool

    def as_dicts(self) -> list[dict]:
        return [dict(zip(self.columns, row)) for row in self.rows]


def run_readonly(
    db_path, sql: str, guard: SqlGuard | None = None, *, timeout_s: float = 5.0
) -> QueryResult:
    """Validate, then execute against a read-only, authorizer-locked connection."""
    guard = guard or SqlGuard()
    safe = guard_sql(sql, guard)

    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=timeout_s)
    try:
        con.set_authorizer(_authorizer)
        # Stop a pathological query from hanging the UI: SQLite calls the
        # progress handler every N opcodes, and a non-zero return aborts.
        budget = {"ops": 0}

        def _progress() -> int:
            budget["ops"] += 1
            return 1 if budget["ops"] > 2000 else 0

        con.set_progress_handler(_progress, 10_000)
        try:
            cur = con.execute(safe)
        except sqlite3.DatabaseError as exc:
            raise GuardError(f"SQLite rejected the query: {exc}") from exc
        columns = [d[0] for d in cur.description or []]
        cap = guard.effective_limit(safe)
        rows = cur.fetchall()
        return QueryResult(
            sql=safe,
            columns=columns,
            rows=[tuple(r) for r in rows],
            truncated=len(rows) >= cap,
        )
    finally:
        con.close()
