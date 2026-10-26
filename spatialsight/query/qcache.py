"""A question cache that stores SQL, not answers.

The distinction is the whole design. Caching the rows would be cheaper still
and wrong: the database changes when a clip is ingested, and a cached answer
would then be confidently stale. Caching the *query* and re-executing it
against the live database costs one SQLite read -- microseconds, no tokens --
and cannot go out of date, because the rows are always read fresh.

The key carries the data's vocabulary, not just the question. "Which zone was
busiest?" has a different correct answer, and potentially different correct
SQL, once a clip with new zone names is ingested; a key of the question alone
would serve the old query forever. So the zone, class and clip lists are
hashed into the key, and ingesting a clip invalidates exactly the entries
that could have changed.

Briefs are keyed on the rows as well, since prose about rows is wrong the
moment the rows differ.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

CACHE_VERSION = 1


def _normalise(question: str) -> str:
    """Fold the differences that never change the right SQL.

    Case and whitespace only. Punctuation stays: "who was there?" and "who
    was there" are the same question, but stripping more than this starts
    merging questions that differ in meaning.
    """
    return re.sub(r"\s+", " ", question.strip().lower())


def _digest(*parts: str) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update(part.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


@dataclass
class QuestionCache:
    """A JSON file of question -> SQL, and (question, rows) -> brief."""

    path: Path
    enabled: bool = True
    hits: int = 0
    misses: int = 0
    _data: dict = field(default_factory=dict)
    _loaded: bool = False

    def _load(self) -> dict:
        if not self._loaded:
            self._loaded = True
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8"))
                if raw.get("version") == CACHE_VERSION:
                    self._data = raw.get("entries", {})
            except (OSError, ValueError):
                # A corrupt or absent cache is a cold cache, never an error.
                self._data = {}
        return self._data

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": CACHE_VERSION, "entries": self._data}
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=1), encoding="utf-8")
        tmp.replace(self.path)

    # ----- SQL -------------------------------------------------------------

    def sql_key(self, question: str, vocabulary: dict[str, list[str]]) -> str:
        vocab = "|".join(
            f"{name}={','.join(vocabulary.get(name, []))}"
            for name in ("zones", "classes", "clips")
        )
        return "sql:" + _digest(_normalise(question), vocab)

    def get_sql(self, question: str, vocabulary: dict[str, list[str]]) -> str | None:
        if not self.enabled:
            return None
        hit = self._load().get(self.sql_key(question, vocabulary))
        if hit is None:
            self.misses += 1
            return None
        self.hits += 1
        return hit

    def put_sql(self, question: str, vocabulary: dict[str, list[str]], sql: str) -> None:
        if not self.enabled:
            return
        self._load()[self.sql_key(question, vocabulary)] = sql
        self._save()

    # ----- briefs ----------------------------------------------------------

    def brief_key(self, question: str, columns: list[str], rows: list[tuple]) -> str:
        body = repr([tuple(r) for r in rows])
        return "brief:" + _digest(_normalise(question), ",".join(columns), body)

    def get_brief(self, question: str, columns: list[str], rows: list[tuple]) -> str | None:
        if not self.enabled:
            return None
        hit = self._load().get(self.brief_key(question, columns, rows))
        if hit is None:
            self.misses += 1
            return None
        self.hits += 1
        return hit

    def put_brief(
        self, question: str, columns: list[str], rows: list[tuple], brief: str
    ) -> None:
        if not self.enabled:
            return
        self._load()[self.brief_key(question, columns, rows)] = brief
        self._save()

    # ----- housekeeping ----------------------------------------------------

    def stats(self) -> dict[str, int]:
        return {
            "entries": len(self._load()),
            "hits": self.hits,
            "misses": self.misses,
        }

    def clear(self) -> int:
        n = len(self._load())
        self._data = {}
        self._save()
        return n


def default_cache(db_path: str | Path, *, enabled: bool = True) -> QuestionCache:
    """Beside the database, under `data/cache/`, which is gitignored.

    It lives next to the database rather than in the repo because it is
    derived data keyed on that database's vocabulary; a cache from someone
    else's clips would key-miss on every entry anyway.
    """
    db_path = Path(db_path)
    return QuestionCache(
        path=db_path.parent.parent / "cache" / "questions.json", enabled=enabled
    )
