"""A JSON API over the pipeline, on the standard library alone.

The UI does not talk to SQLite. Every query goes through this server, which
goes through the same `SqlGuard` the tests exercise. That is the whole reason
this process exists: porting the guard to TypeScript would mean two
implementations of the one security boundary in the project, and the second
one would not be the tested one.

No framework, because a dependency here would buy nothing: this serves six
routes and a video file.

    GET  /api/health
    GET  /api/clips
    GET  /api/zones?clip=
    GET  /api/events?clip=
    GET  /api/canned
    GET  /api/schema
    POST /api/sql      {"sql": "..."}
    POST /api/ask      {"question": "..."}
    GET  /media/<clip> the clip itself, with range requests
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from . import config
from .query.canned import QUESTIONS
from .query.guard import GuardError, SqlGuard, run_readonly

MAX_BODY_BYTES = 64 * 1024


def _rle(data: bytes) -> list[int]:
    """Run lengths of a 0/1 byte string, starting with a run of zeros.

    Always starts with the zero run so the client needs no parity flag; a
    mask whose first pixel is set simply begins with a run of length 0.
    """
    runs: list[int] = []
    current = 0
    count = 0
    for value in data:
        bit = 1 if value else 0
        if bit == current:
            count += 1
        else:
            runs.append(count)
            current = bit
            count = 1
    runs.append(count)
    return runs


class Api:
    """The request handlers, separated from the HTTP plumbing so they can be tested."""

    def __init__(self, cfg: config.Config) -> None:
        self.cfg = cfg
        self.guard = SqlGuard()
        self._spark = None
        self._spark_lock = threading.Lock()

    # ----- data -----------------------------------------------------------

    @property
    def manifest_path(self) -> Path:
        return self.cfg.db_path.parent / "clips.json"

    def manifest(self) -> dict:
        if not self.manifest_path.exists():
            return {}
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))

    def health(self) -> dict:
        return {
            "ok": True,
            "database": str(self.cfg.db_path),
            "database_exists": self.cfg.db_path.exists(),
            "clips": sorted(self.manifest()),
            "spark_model": self.cfg.llm_model,
            "llm_endpoint": self.cfg.llm_base_url,
            "natural_language": self.cfg.has_llm,
        }

    def clips(self) -> dict:
        manifest = self.manifest()
        counts: dict[str, dict] = {}
        if self.cfg.db_path.exists():
            con = sqlite3.connect(f"file:{self.cfg.db_path}?mode=ro", uri=True)
            try:
                for clip, entities, events in con.execute(
                    "SELECT n.clip, COUNT(DISTINCT n.entity_id),"
                    "       (SELECT COUNT(*) FROM events e WHERE e.clip = n.clip)"
                    " FROM entities n GROUP BY n.clip"
                ):
                    counts[clip] = {"entities": entities, "events": events}
            finally:
                con.close()
        return {
            "clips": [
                {
                    "clip": name,
                    "width": entry.get("width"),
                    "height": entry.get("height"),
                    "fps": entry.get("fps"),
                    "frame_count": entry.get("frame_count"),
                    "duration_s": entry.get("duration_s"),
                    "backend": entry.get("backend"),
                    "simulated": str(entry.get("path", "")).startswith("synthetic://"),
                    **counts.get(name, {"entities": 0, "events": 0}),
                }
                for name, entry in sorted(manifest.items())
            ]
        }

    def zones(self, clip: str) -> dict:
        entry = self.manifest().get(clip, {})
        path = Path(entry.get("zones_path") or f"zones/{clip}.json")
        if not path.is_absolute():
            path = config.REPO_ROOT / path
        if not path.exists():
            raise FileNotFoundError(f"no zone file for clip {clip!r}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["clip"] = clip
        return payload

    def events(self, clip: str | None) -> dict:
        sql = (
            "SELECT e.id, e.clip, e.entity_id, e.class, e.zone, e.entered_at,"
            " e.exited_at, e.dwell_s, e.frame_start, e.frame_end, e.peak_overlap,"
            " e.closed FROM events e"
        )
        params: list = []
        if clip:
            sql += " WHERE e.clip = ?"
            params.append(clip)
        sql += " ORDER BY e.entered_at, e.entity_id"

        con = sqlite3.connect(f"file:{self.cfg.db_path}?mode=ro", uri=True)
        con.row_factory = sqlite3.Row
        try:
            rows = [dict(r) for r in con.execute(sql, params)]
        finally:
            con.close()
        return {"events": rows, "count": len(rows)}

    def overlay(self, clip: str) -> dict:
        """Per-frame masks and boxes, for drawing on top of the video.

        Masks go out run-length encoded rather than as pixels or an SVG path:
        a 24x24 mask is 576 bytes raw and a handful of integers as runs, and
        the client reconstructs the exact binary raster rather than an
        approximation of its outline. Entity ids come from the tracker, so the
        colour of a box on screen is the same identity the events table and
        the SQL rows refer to.
        """
        entry = self.manifest().get(clip)
        if entry is None:
            raise FileNotFoundError(f"unknown clip {clip!r}")

        detections = self._detections_for(clip, entry)
        from .tracker import reconcile

        recon = reconcile(detections)
        frames: dict[int, list[dict]] = {}
        for track in recon.tracks:
            for det in track.detections:
                item: dict = {
                    "entity_id": track.entity_id,
                    "class": det.label,
                    "box": [det.box.left, det.box.top, det.box.right, det.box.bottom],
                    "confidence": det.confidence,
                }
                if det.mask is not None:
                    item["mask"] = {
                        "rows": det.mask.rows,
                        "cols": det.mask.cols,
                        "runs": _rle(det.mask.data),
                    }
                frames.setdefault(det.frame, []).append(item)

        return {
            "clip": clip,
            "width": entry.get("width"),
            "height": entry.get("height"),
            "fps": entry.get("fps"),
            "frame_count": entry.get("frame_count"),
            "entities": [
                {"entity_id": t.entity_id, "class": t.label} for t in recon.tracks
            ],
            "frames": {str(k): v for k, v in sorted(frames.items())},
        }

    def _detections_for(self, clip: str, entry: dict):
        """Reproduce a clip's detections from whatever produced them."""
        if str(entry.get("backend", "")).startswith("yolo"):
            from .backends.local import LocalBackend

            path = Path(entry["path"])
            if not path.is_absolute():
                path = config.REPO_ROOT / path
            # Cached on first ingest, so this is a file read, not a GPU pass.
            return LocalBackend(
                path,
                cache_dir=self.cfg.cache_dir,
                weights=entry.get("weights", "E:/models/yolo11s-seg.pt"),
            ).detect()
        if entry.get("backend") == "fixture" or "concepts" in entry:
            from .backends.sam import CachedSamBackend

            return CachedSamBackend(
                self.manifest_path, clip, cache_dir=self.cfg.cache_dir
            ).detect()
        if str(entry.get("path", "")).startswith("synthetic://"):
            from .backends.synthetic import SyntheticBackend

            return SyntheticBackend(clip).detect()
        raise FileNotFoundError(
            f"clip {clip!r} has no cached segmentation to draw; run `ss ingest` first"
        )

    def canned(self) -> dict:
        return {
            "questions": [
                {"key": q.key, "question": q.question, "sql": q.sql.strip(), "note": q.note}
                for q in QUESTIONS
            ]
        }

    def schema(self) -> dict:
        from .db import SCHEMA_FOR_PROMPT

        return {"schema": SCHEMA_FOR_PROMPT}

    # ----- queries --------------------------------------------------------

    def run_sql(self, sql: str) -> dict:
        result = run_readonly(self.cfg.db_path, sql, self.guard)
        return {
            "sql": result.sql,
            "columns": result.columns,
            "rows": [list(r) for r in result.rows],
            "row_count": len(result.rows),
            "truncated": result.truncated,
        }

    def spark(self):
        from .query.nlsql import SparkClient

        with self._spark_lock:
            if self._spark is None:
                self._spark = SparkClient.from_config(self.cfg)
            return self._spark

    def ask(self, question: str) -> dict:
        from .query.nlsql import ask as ask_spark

        answer = ask_spark(question, self.cfg.db_path, self.spark(), guard=self.guard)
        payload: dict = {
            "question": answer.question,
            "sql": answer.sql,
            "ok": answer.ok,
            "attempts": len(answer.attempts),
            "brief": answer.brief,
            "error": answer.error,
            "usage": answer.usage,
        }
        if answer.result is not None:
            payload.update(
                {
                    "columns": answer.result.columns,
                    "rows": [list(r) for r in answer.result.rows],
                    "row_count": len(answer.result.rows),
                    "truncated": answer.result.truncated,
                }
            )
        return payload


class Handler(BaseHTTPRequestHandler):
    api: Api  # injected by `serve`
    server_version = "SpatialSight/0.1"

    def log_message(self, fmt: str, *args) -> None:
        print(f"  {self.command} {self.path} -> {args[1] if len(args) > 1 else ''}")

    # ----- helpers --------------------------------------------------------

    def _json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _error(self, message: str, status: int = 400) -> None:
        self._json({"error": message}, status)

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        if length > MAX_BODY_BYTES:
            raise ValueError("request body too large")
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    # ----- routes ---------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        clip = (query.get("clip") or [None])[0]
        try:
            if parsed.path == "/api/health":
                return self._json(self.api.health())
            if parsed.path == "/api/clips":
                return self._json(self.api.clips())
            if parsed.path == "/api/zones":
                if not clip:
                    return self._error("clip is required")
                return self._json(self.api.zones(clip))
            if parsed.path == "/api/events":
                return self._json(self.api.events(clip))
            if parsed.path == "/api/overlay":
                if not clip:
                    return self._error("clip is required")
                return self._json(self.api.overlay(clip))
            if parsed.path == "/api/canned":
                return self._json(self.api.canned())
            if parsed.path == "/api/schema":
                return self._json(self.api.schema())
            if parsed.path.startswith("/media/"):
                return self._media(parsed.path[len("/media/"):])
            return self._error("not found", 404)
        except FileNotFoundError as exc:
            self._error(str(exc), 404)
        except sqlite3.OperationalError as exc:
            self._error(
                f"database not ready ({exc}). Run `ss fixture` then `ss ingest`.", 503
            )
        except Exception as exc:  # last resort: never leak a traceback to the client
            self._error(f"{type(exc).__name__}: {exc}", 500)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            body = self._body()
            if parsed.path == "/api/sql":
                sql = (body.get("sql") or "").strip()
                if not sql:
                    return self._error("sql is required")
                return self._json(self.api.run_sql(sql))
            if parsed.path == "/api/ask":
                question = (body.get("question") or "").strip()
                if not question:
                    return self._error("question is required")
                return self._json(self.api.ask(question))
            return self._error("not found", 404)
        except GuardError as exc:
            # A rejected query is an expected outcome, not a server fault.
            self._json({"error": str(exc), "rejected": True}, 422)
        except RuntimeError as exc:
            self._error(str(exc), 503)
        except ValueError as exc:
            self._error(str(exc), 400)
        except Exception as exc:
            self._error(f"{type(exc).__name__}: {exc}", 500)

    def _media(self, name: str) -> None:
        """Serve a clip, honouring range requests so the player can seek."""
        entry = self.api.manifest().get(name)
        if entry is None:
            return self._error(f"unknown clip {name!r}", 404)
        raw = str(entry.get("path", ""))
        if raw.startswith("synthetic://"):
            return self._error(
                f"clip {name!r} is simulated and has no video file; "
                f"the UI draws the scene from its masks instead.",
                404,
            )
        path = Path(raw)
        if not path.is_absolute():
            path = config.REPO_ROOT / path
        if not path.exists():
            return self._error(f"missing file for clip {name!r}: {path}", 404)

        size = path.stat().st_size
        start, end = 0, size - 1
        header = self.headers.get("Range")
        partial = False
        if header and header.startswith("bytes="):
            spec = header[len("bytes="):].split("-", 1)
            if spec[0]:
                start = int(spec[0])
            if len(spec) > 1 and spec[1]:
                end = min(int(spec[1]), size - 1)
            partial = True
        if start >= size:
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.end_headers()
            return

        length = end - start + 1
        self.send_response(206 if partial else 200)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if partial:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        with path.open("rb") as fh:
            fh.seek(start)
            remaining = length
            while remaining > 0:
                chunk = fh.read(min(1 << 16, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)


def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    cfg = config.load()
    Handler.api = Api(cfg)
    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"SpatialSight API on http://{host}:{port}")
    print(f"  database          {cfg.db_path}")
    print(f"  natural language  {cfg.llm_model + ' via ' + cfg.llm_base_url if cfg.has_llm else 'disabled (no key or LLM_BASE_URL)'}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(prog="spatialsight.server")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    serve(**vars(p.parse_args()))
