"""The command line: every stage of the pipeline, runnable on its own.

Subcommands, in the order you would use them:

    ss probe   <video>            read a clip's true fps, size and frame count
    ss cost    <video>            what segmenting it would cost, before paying
    ss resample <video> <out>     trim and re-time a clip to cut frame cost
    ss poster  <video> <out.png>  pull one frame out to draw zones against
    ss fixture                    build the offline fixture cache (no key)
    ss ingest  <clip>             segment, track, compile events, write SQL
    ss events  [clip]             print the events table
    ss sql     "<select ...>"     run SQL through the guard
    ss ask     "<question>"       natural language -> SQL -> rows -> brief
    ss canned  [key]              run the reference questions, no key needed
    ss eval    <clip>             score the pipeline against ground truth
    ss cache                      cache size and contents
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path

from . import config
from .db import connect
from .events import EventParams
from .query.canned import BY_KEY, QUESTIONS
from .query.guard import GuardError, SqlGuard, run_readonly
from .tracker import TrackerParams
from .zones import ZoneRaster, load_zones


def _table(columns: list[str], rows: list[tuple], limit: int = 50) -> str:
    if not columns:
        return "(no columns)"
    shown = rows[:limit]
    widths = [len(c) for c in columns]
    for row in shown:
        for i, value in enumerate(row):
            widths[i] = max(widths[i], len(_fmt(value)))
    out = ["  ".join(c.ljust(widths[i]) for i, c in enumerate(columns))]
    out.append("  ".join("-" * w for w in widths))
    for row in shown:
        out.append("  ".join(_fmt(v).ljust(widths[i]) for i, v in enumerate(row)))
    if len(rows) > limit:
        out.append(f"... {len(rows) - limit} more rows")
    return "\n".join(out)


def _fmt(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.2f}".rstrip("0").rstrip(".")
    return str(value)


def _zones_for(clip: str, explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    candidate = config.REPO_ROOT / "zones" / f"{clip}.json"
    if not candidate.exists():
        raise SystemExit(
            f"no zone file for clip {clip!r}. Expected {candidate}, or pass --zones."
        )
    return candidate


# ----- commands -------------------------------------------------------------


def cmd_probe(args) -> int:
    from .probe import probe_video

    meta = probe_video(args.video, strict=not args.loose)
    print(f"path          {meta.path}")
    print(f"resolution    {meta.width}x{meta.height}")
    print(f"frame rate    {meta.fps:g} fps")
    print(f"frames        {meta.frame_count}")
    print(f"duration      {meta.duration_s:.2f}s")
    print(f"sha256        {meta.sha256[:16]}...")
    return 0


def cmd_cost(args) -> int:
    from .backends.sam import (
        MAX_VIDEO_FRAMES,
        MIN_BILLED_FRAMES,
        PRICE_PER_1K_FRAMES_USD,
    )
    from .probe import probe_video

    meta = probe_video(args.video, strict=not args.loose)
    billed = max(MIN_BILLED_FRAMES, meta.frame_count)
    per_concept = billed / 1000.0 * PRICE_PER_1K_FRAMES_USD
    total = per_concept * args.classes

    print(f"clip          {Path(args.video).name}")
    print(f"frames        {meta.frame_count} (billed as {billed}, 50-frame minimum)")
    print(f"rate          ${PRICE_PER_1K_FRAMES_USD:.2f} per 1,000 frames")
    print(f"per concept   ${per_concept:.4f}")
    print(f"{args.classes} concepts    ${total:.4f}")
    if meta.frame_count > MAX_VIDEO_FRAMES:
        print(
            f"\nWARNING: {meta.frame_count} frames exceeds the "
            f"{MAX_VIDEO_FRAMES}-frame request limit. Resample or trim first."
        )
    if args.budget:
        print(f"\nbudget        ${args.budget:.2f}")
        print(f"clips at this size  {int(args.budget // total) if total else 0}")
    # Resampling is where the real saving is, so show it unprompted.
    print("\nif resampled:")
    for fps in (1, 2, 5):
        frames = max(MIN_BILLED_FRAMES, int(meta.duration_s * fps))
        cost = frames / 1000.0 * PRICE_PER_1K_FRAMES_USD * args.classes
        print(f"  {fps} fps  {frames:>6} frames  ${cost:.4f}")
    return 0


def cmd_resample(args) -> int:
    from .probe import probe_video, resample

    out = resample(args.video, args.out, args.fps, start=args.start, seconds=args.seconds)
    meta = probe_video(out, strict=False)
    print(f"wrote {out} -- {meta.frame_count} frames at {meta.fps:g} fps")
    return 0


def cmd_convert(args) -> int:
    """Re-encode any clip to H.264 MP4 at its own frame rate."""
    from .probe import probe_video, resample

    out = resample(args.video, args.out, fps=None)
    meta = probe_video(out, strict=True)
    print(
        f"wrote {out} -- {meta.width}x{meta.height}, {meta.frame_count} frames "
        f"at {meta.fps:g} fps, {meta.duration_s:.1f}s"
    )
    return 0


def cmd_poster(args) -> int:
    from .probe import extract_frame

    out = extract_frame(args.video, args.out, args.frame)
    print(f"wrote {out} (frame {args.frame}); draw zones against this image")
    return 0


def cmd_fixture(args) -> int:
    from .fixtures import build_fixture

    cfg = config.load()
    info = build_fixture(
        clip=args.clip,
        cache_dir=cfg.cache_dir,
        manifest_path=cfg.db_path.parent / "clips.json",
    )
    print(json.dumps(info, indent=2))
    print("\nNow run:  ss ingest", args.clip)
    return 0


def cmd_ingest(args) -> int:
    cfg = config.load()
    from .runner import run

    zones_path = _zones_for(args.clip, args.zones)
    manifest = cfg.db_path.parent / "clips.json"

    if args.synthetic:
        from .backends.synthetic import SyntheticBackend

        backend = SyntheticBackend(args.clip)
    elif args.video and args.local:
        from .backends.local import LocalBackend

        backend = LocalBackend(
            args.video,
            cache_dir=cfg.cache_dir,
            weights=args.weights,
            device=args.device,
            imgsz=args.imgsz,
        )
    elif args.video:
        from .backends.sam import SamBackend

        if not cfg.has_key:
            raise SystemExit(
                "MODEL_API_KEY is not set. Copy .env.example to .env and add the key "
                "from dev.meta.ai, or use --synthetic / the fixture cache."
            )
        backend = SamBackend(
            args.video,
            api_key=cfg.api_key or "",
            api_base=cfg.api_base,
            model=cfg.sam_model,
            cache_dir=cfg.cache_dir,
            allow_spend=args.allow_spend,
        )
        if args.allow_spend:
            print(f"estimated spend for this clip: ${backend.estimate_cost_usd():.4f}")
    else:
        from .backends.sam import CachedSamBackend

        if not manifest.exists():
            raise SystemExit(
                f"no clip manifest at {manifest}. Run `ss fixture` first, or pass "
                f"--video to segment a real clip."
            )
        backend = CachedSamBackend(manifest, args.clip, cache_dir=cfg.cache_dir)

    report, recon, events = run(
        backend=backend,
        clip=args.clip,
        zones_path=zones_path,
        db_path=cfg.db_path,
        tracker_params=TrackerParams(
            max_gap_frames=args.max_gap, min_detections=args.min_detections
        ),
        event_params=EventParams(enter_at=args.enter_at, exit_at=args.exit_at),
    )
    print("\n".join(report.lines()))
    if recon.repairs and args.verbose:
        print("\nrepairs:")
        for r in recon.repairs:
            print(f"  {r.kind:<7} {r.detail}")
    spend = getattr(backend, "last_run", {})
    if spend.get("seconds"):
        print(
            f"\nsegmented {spend['frames']} frames in {spend['seconds']}s "
            f"({spend['fps']} fps on {spend['device']}), cost $0.00"
        )
    if spend.get("requests_billed"):
        print(f"\nbilled requests  {spend['requests_billed']}")
        print(f"estimated spend  ${spend['est_usd_this_run']:.4f}")
    print(f"\nwrote {len(events)} events to {cfg.db_path}")
    return 0


def cmd_events(args) -> int:
    cfg = config.load()
    sql = (
        "SELECT clip, entity_id, class, zone, entered_at, exited_at, dwell_s,"
        " frame_start, peak_overlap, closed FROM events"
    )
    params: list = []
    if args.clip:
        sql += " WHERE clip = ?"
        params.append(args.clip)
    sql += " ORDER BY entered_at, entity_id"

    con = connect(cfg.db_path, read_only=True)
    try:
        cur = con.execute(sql, params)
        columns = [d[0] for d in cur.description]
        rows = [tuple(r) for r in cur.fetchall()]
    finally:
        con.close()
    print(_table(columns, rows, limit=args.limit))
    print(f"\n{len(rows)} events")
    return 0


def cmd_sql(args) -> int:
    cfg = config.load()
    try:
        result = run_readonly(cfg.db_path, args.query, SqlGuard(max_limit=args.limit))
    except GuardError as exc:
        print(f"rejected: {exc}", file=sys.stderr)
        return 1
    print(result.sql)
    print()
    print(_table(result.columns, result.rows, limit=args.limit))
    print(f"\n{len(result.rows)} rows" + (" (row cap reached)" if result.truncated else ""))
    return 0


def cmd_canned(args) -> int:
    cfg = config.load()
    questions = [BY_KEY[args.key]] if args.key else list(QUESTIONS)
    failures = 0
    for q in questions:
        print(f"\n=== {q.key} ===")
        print(f"Q: {q.question}")
        if q.note:
            print(f"   ({q.note})")
        try:
            result = run_readonly(cfg.db_path, q.sql)
        except GuardError as exc:
            print(f"   FAILED: {exc}")
            failures += 1
            continue
        print()
        print(_table(result.columns, result.rows, limit=args.limit))
    print(f"\n{len(questions) - failures}/{len(questions)} questions answered")
    return 1 if failures else 0


def cmd_ask(args) -> int:
    cfg = config.load()
    from .query.nlsql import SparkClient, ask

    client = SparkClient.from_config(cfg)
    answer = ask(
        args.question,
        cfg.db_path,
        client,
        guard=SqlGuard(max_limit=args.limit),
        with_brief=not args.no_brief,
    )

    print(f"Q: {answer.question}\n")
    print("SQL:")
    print(answer.sql)
    print()
    if not answer.ok:
        print(f"failed after {len(answer.attempts)} attempt(s): {answer.error}")
        return 1
    assert answer.result is not None
    print(_table(answer.result.columns, answer.result.rows, limit=args.limit))
    print(f"\n{len(answer.result.rows)} rows")
    if answer.brief:
        print(f"\nbrief:\n{answer.brief}")
    if answer.usage:
        print(f"\ntokens: {answer.usage}")
    return 0


def cmd_bench(args) -> int:
    cfg = config.load()
    from .query.bench import run_bench
    from .query.nlsql import SparkClient

    client = SparkClient.from_config(cfg)
    report = run_bench(cfg.db_path, client, endpoint=cfg.llm_base_url)
    print("\n".join(report.lines()))
    if args.json:
        Path(args.json).write_text(json.dumps(report.as_dict(), indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


def cmd_eval(args) -> int:
    cfg = config.load()
    from .backends.synthetic import SyntheticBackend, default_scene
    from .eval import evaluate_synthetic

    zones_path = _zones_for(args.clip, args.zones)
    backend = SyntheticBackend(args.clip)
    scene = default_scene()
    meta = backend.meta()
    raster = ZoneRaster(load_zones(zones_path), meta.width, meta.height)

    if args.from_cache:
        from .backends.sam import CachedSamBackend

        manifest = cfg.db_path.parent / "clips.json"
        detections = CachedSamBackend(
            manifest, args.clip, cache_dir=cfg.cache_dir
        ).detect()
        source = "cached SAM wire format"
    else:
        detections = backend.detect()
        source = "synthetic backend"

    db_bytes = cfg.db_path.stat().st_size if cfg.db_path.exists() else 0
    report = evaluate_synthetic(
        detections, scene, raster, meta, args.clip, db_bytes=db_bytes
    )
    print(f"source: {source}\n")
    print("\n".join(report.lines()))
    if args.json:
        Path(args.json).write_text(
            json.dumps(report.as_dict(), indent=2), encoding="utf-8"
        )
        print(f"\nwrote {args.json}")
    return 0


def cmd_cache(args) -> int:
    cfg = config.load()
    from .cache import ResponseCache

    cache = ResponseCache(cfg.cache_dir)
    stats = cache.stats()
    print(f"directory  {cfg.cache_dir}")
    print(f"entries    {stats['entries']}")
    print(f"size       {stats['bytes']:,} bytes")
    for path in sorted(cfg.cache_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        meta = payload.get("meta", {})
        print(
            f"  {path.stem[:12]}  {meta.get('clip', '?'):<16} "
            f"{meta.get('concept', '?'):<12} {len(payload['deltas']):>5} lines  "
            f"${meta.get('est_usd', 0):.4f}"
        )
    return 0


def cmd_schema(args) -> int:
    from .db import SCHEMA_FOR_PROMPT

    print(SCHEMA_FOR_PROMPT)
    return 0


# ----- wiring ---------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ss",
        description="SpatialSight: video understanding compiled into a queryable schema.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = p.add_subparsers(dest="command", required=True)

    probe = sub.add_parser("probe", help="read a clip's true fps, size and frames")
    probe.add_argument("video")
    probe.add_argument("--loose", action="store_true", help="skip codec checks")
    probe.set_defaults(func=cmd_probe)

    cost = sub.add_parser("cost", help="what segmenting a clip would cost")
    cost.add_argument("video")
    cost.add_argument("--classes", type=int, default=2, help="number of concepts")
    cost.add_argument("--budget", type=float, default=None)
    cost.add_argument("--loose", action="store_true")
    cost.set_defaults(func=cmd_cost)

    res = sub.add_parser("resample", help="trim and re-time a clip to cut frame cost")
    res.add_argument("video")
    res.add_argument("out")
    res.add_argument("--fps", type=float, default=2.0)
    res.add_argument("--start", type=float, default=0.0)
    res.add_argument("--seconds", type=float, default=None)
    res.set_defaults(func=cmd_resample)

    conv = sub.add_parser("convert", help="re-encode any clip to H.264 MP4")
    conv.add_argument("video")
    conv.add_argument("out")
    conv.set_defaults(func=cmd_convert)

    poster = sub.add_parser("poster", help="extract one frame to draw zones against")
    poster.add_argument("video")
    poster.add_argument("out")
    poster.add_argument("--frame", type=int, default=0)
    poster.set_defaults(func=cmd_poster)

    fix = sub.add_parser("fixture", help="build the offline fixture cache")
    fix.add_argument("--clip", default="synthetic-01")
    fix.set_defaults(func=cmd_fixture)

    ing = sub.add_parser("ingest", help="segment, track, compile events, write SQL")
    ing.add_argument("clip")
    ing.add_argument("--video", default=None, help="segment this file with SAM 3.1")
    ing.add_argument("--synthetic", action="store_true", help="simulate, no parser")
    ing.add_argument(
        "--local",
        action="store_true",
        help="segment --video with the free local stand-in (YOLO11-seg) instead of SAM 3.1",
    )
    ing.add_argument("--weights", default="E:/models/yolo11s-seg.pt")
    ing.add_argument("--device", default="0", help="GPU index, or cpu")
    ing.add_argument("--imgsz", type=int, default=960)
    ing.add_argument("--zones", default=None)
    ing.add_argument("--allow-spend", action="store_true", help="permit paid API calls")
    ing.add_argument("--max-gap", type=int, default=12)
    ing.add_argument(
        "--min-detections",
        type=int,
        default=3,
        help="frames an entity must be seen in before it counts (noise filter)",
    )
    ing.add_argument("--enter-at", type=float, default=0.50)
    ing.add_argument("--exit-at", type=float, default=0.30)
    ing.add_argument("-v", "--verbose", action="store_true")
    ing.set_defaults(func=cmd_ingest)

    ev = sub.add_parser("events", help="print the events table")
    ev.add_argument("clip", nargs="?", default=None)
    ev.add_argument("--limit", type=int, default=50)
    ev.set_defaults(func=cmd_events)

    sq = sub.add_parser("sql", help="run SQL through the guard")
    sq.add_argument("query")
    sq.add_argument("--limit", type=int, default=200)
    sq.set_defaults(func=cmd_sql)

    can = sub.add_parser("canned", help="run the reference questions (no key needed)")
    can.add_argument("key", nargs="?", default=None)
    can.add_argument("--limit", type=int, default=20)
    can.set_defaults(func=cmd_canned)

    a = sub.add_parser("ask", help="natural language question (needs a key)")
    a.add_argument("question")
    a.add_argument("--limit", type=int, default=200)
    a.add_argument("--no-brief", action="store_true")
    a.set_defaults(func=cmd_ask)

    bench = sub.add_parser(
        "bench", help="ask the ten reference questions in English and score the SQL"
    )
    bench.add_argument("--json", default=None)
    bench.set_defaults(func=cmd_bench)

    ev2 = sub.add_parser("eval", help="score the pipeline against ground truth")
    ev2.add_argument("clip", nargs="?", default="synthetic-01")
    ev2.add_argument("--zones", default=None)
    ev2.add_argument(
        "--from-cache",
        action="store_true",
        help="score the real parse path rather than the simulator",
    )
    ev2.add_argument("--json", default=None, help="also write the report here")
    ev2.set_defaults(func=cmd_eval)

    c = sub.add_parser("cache", help="cache size and contents")
    c.set_defaults(func=cmd_cache)

    s = sub.add_parser("schema", help="print the schema pinned into the SQL prompt")
    s.set_defaults(func=cmd_schema)

    return p


def main(argv: list[str] | None = None) -> int:
    # Keep numpy single-threaded: OpenBLAS aborts on this machine otherwise.
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    # Model output contains characters a Windows console code page cannot
    # print (a narrow no-break space killed the first live brief). Replace
    # rather than crash: a mangled glyph is better than a lost answer.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (GuardError, FileNotFoundError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except sqlite3.OperationalError as exc:
        print(f"database error: {exc}", file=sys.stderr)
        print(
            "If the database is missing, run `ss fixture` then `ss ingest synthetic-01`.",
            file=sys.stderr,
        )
        return 1
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
