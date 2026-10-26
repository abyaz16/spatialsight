"""SQLite store: two tables, obvious joins.

The schema is deliberately flat and boring. A language model writes correct
SQL against two tables with plain column names far more reliably than against
a clever normalised one, and every wrong query it writes costs a judge's
patience. Clip provenance is denormalised onto both tables rather than kept in
a third, for the same reason.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .events import ZoneEvent
from .types import VideoMeta, Track

SCHEMA = """
CREATE TABLE IF NOT EXISTS entities (
  entity_id   INTEGER NOT NULL,
  clip        TEXT    NOT NULL,
  class       TEXT    NOT NULL,   -- 'person' | 'vehicle'
  first_seen  REAL    NOT NULL,   -- seconds into the clip
  last_seen   REAL    NOT NULL,
  frame_first INTEGER NOT NULL,
  frame_last  INTEGER NOT NULL,
  PRIMARY KEY (clip, entity_id)
);

CREATE TABLE IF NOT EXISTS events (
  id           INTEGER PRIMARY KEY AUTOINCREMENT,
  clip         TEXT    NOT NULL,
  entity_id    INTEGER NOT NULL,
  class        TEXT    NOT NULL,
  zone         TEXT    NOT NULL,
  entered_at   REAL    NOT NULL,  -- seconds into the clip
  exited_at    REAL    NOT NULL,
  dwell_s      REAL    NOT NULL,
  frame_start  INTEGER NOT NULL,  -- for video seek
  frame_end    INTEGER NOT NULL,
  peak_overlap REAL    NOT NULL,  -- most of the entity ever inside the zone
  closed       INTEGER NOT NULL,  -- 0 when still in the zone at clip end
  FOREIGN KEY (clip, entity_id) REFERENCES entities (clip, entity_id)
);

CREATE INDEX IF NOT EXISTS idx_events_zone  ON events (zone);
CREATE INDEX IF NOT EXISTS idx_events_time  ON events (entered_at);
CREATE INDEX IF NOT EXISTS idx_events_clip  ON events (clip);
"""

SCHEMA_FOR_PROMPT = """\
CREATE TABLE entities (
  entity_id   INTEGER,  -- unique within a clip
  clip        TEXT,     -- clip name, e.g. 'warehouse-01'
  class        TEXT,    -- 'person' or 'vehicle'
  first_seen  REAL,     -- seconds from the start of the clip
  last_seen   REAL,
  frame_first INTEGER,
  frame_last  INTEGER
);

CREATE TABLE events (
  id           INTEGER,
  clip         TEXT,
  entity_id    INTEGER,  -- joins to entities(entity_id) within the same clip
  class        TEXT,
  zone         TEXT,     -- zone name, e.g. 'loading_dock'
  entered_at   REAL,     -- seconds from the start of the clip
  exited_at    REAL,
  dwell_s      REAL,     -- exited_at - entered_at
  frame_start  INTEGER,
  frame_end    INTEGER,
  peak_overlap REAL,     -- 0..1, the most of the entity ever inside the zone
  closed       INTEGER   -- 0 when the entity was still in the zone at clip end
);

-- Times are seconds into the clip, not wall-clock. One row in `events`
-- is one visit by one entity to one zone.\
"""


def connect(path: str | Path, *, read_only: bool = False) -> sqlite3.Connection:
    path = Path(path)
    if read_only:
        if not path.exists():
            raise FileNotFoundError(path)
        uri = f"file:{path.as_posix()}?mode=ro"
        con = sqlite3.connect(uri, uri=True)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(path)
        con.executescript(SCHEMA)
    con.row_factory = sqlite3.Row
    return con


def write_clip(
    con: sqlite3.Connection,
    clip: str,
    meta: VideoMeta,
    tracks: list[Track],
    events: list[ZoneEvent],
) -> None:
    """Replace everything stored for `clip`. Re-running the pipeline is idempotent."""
    with con:
        con.execute("DELETE FROM events WHERE clip = ?", (clip,))
        con.execute("DELETE FROM entities WHERE clip = ?", (clip,))
        con.executemany(
            "INSERT INTO entities (entity_id, clip, class, first_seen, last_seen,"
            " frame_first, frame_last) VALUES (?,?,?,?,?,?,?)",
            [
                (
                    t.entity_id,
                    clip,
                    t.label,
                    round(meta.to_seconds(t.first_frame), 3),
                    round(meta.to_seconds(t.last_frame), 3),
                    t.first_frame,
                    t.last_frame,
                )
                for t in tracks
            ],
        )
        con.executemany(
            "INSERT INTO events (clip, entity_id, class, zone, entered_at, exited_at,"
            " dwell_s, frame_start, frame_end, peak_overlap, closed)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            [
                (
                    clip,
                    e.entity_id,
                    e.label,
                    e.zone,
                    e.entered_at,
                    e.exited_at,
                    e.dwell_s,
                    e.frame_start,
                    e.frame_end,
                    e.peak_overlap,
                    1 if e.closed else 0,
                )
                for e in events
            ],
        )


def write_manifest(path: Path, clip: str, meta: VideoMeta, extra: dict) -> None:
    """Clip provenance lives beside the DB, not in it, to keep the schema at two tables.

    The clip's entry is **merged**, not replaced. Whoever created the clip
    recorded how to read its cached responses -- which model, which mask
    encoding, which noun phrase per class -- and a later ingest knows none of
    that. Overwriting the entry silently breaks the next cache lookup, which
    then looks like a cache miss rather than a lost key.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    entry = dict(existing.get(clip, {}))
    entry.update(
        {
            "path": meta.path,
            "width": meta.width,
            "height": meta.height,
            "fps": meta.fps,
            "frame_count": meta.frame_count,
            "duration_s": meta.duration_s,
            "sha256": meta.sha256,
            **extra,
        }
    )
    existing[clip] = entry
    path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
