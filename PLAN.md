# SpatialSight — locked spec

**One line:** pixel-level video understanding compiled into a relational schema you can query in plain English.

**Theme:** 2 — Push the Frontier. **Track:** 1 (indie) unless a real team/company exists.

## Scope contract (do not expand)

- Input: **recorded video files.** No live RTSP.
- One camera, one fixed angle, 2 clips for the eval + 1 for the demo.
- Entity classes: **2** (e.g. person, forklift/vehicle). Not "all objects".
- Zones: 3-5 hand-drawn polygons per camera, defined once in a JSON file.
- Events: enter, exit, dwell. Nothing else.
- No codec work. No model training. No auth. No multi-tenant. No deploy.

## Cut from the original pitch

| Cut | Why |
|---|---|
| Muse Code as a pipeline component | It is an agentic **coding agent** — a dev tool. Listing it as a runtime encoder signals the stack page was skimmed. Use it to build this. |
| LLM-dictated compression logic | An LLM is a worse bitrate controller than 3 lines of heuristic. Adds cost, subtracts credibility. |
| 80% storage reduction / ROI encoding | Real but a 2000s technique. Dilutes "technical ambition" instead of adding it. |
| Live RTSP ingest | Whole class of failure modes, zero judging points. No judge will verify liveness. |
| Per-frame SAM 3.1 | Will exhaust the $150 credit grant almost immediately. |

Compression survives as **one metric line**, not a feature:
> the derived event table is ~N,000x smaller than the source footage and answers the questions people open the video for.

## Architecture

```
video file
  -> ffmpeg: extract keyframes at 1-2 fps          [free, local]
  -> SAM 3.1 (Meta Model API): masks per keyframe  [$ - the only paid step]
  -> tracker: IOU + centroid match across frames   [our code, deterministic]
  -> event extractor: masks x zone polygons        [our code, deterministic]
  -> SQLite: events table
  -> Muse Spark: (a) NL -> SQL   (b) bottleneck brief over result rows
  -> Next.js: video + zone overlay + table + query box; click row seeks video
```

**The two deterministic layers are where the craft is.** Tracking and event extraction must be our own code, not a model call — that is what makes the numbers defensible and the cost survivable.

### Schema

```sql
CREATE TABLE entities (
  entity_id   INTEGER PRIMARY KEY,
  class       TEXT NOT NULL,          -- 'person' | 'vehicle'
  first_seen  REAL NOT NULL,          -- seconds into clip
  last_seen   REAL NOT NULL
);

CREATE TABLE events (
  id          INTEGER PRIMARY KEY,
  entity_id   INTEGER REFERENCES entities(entity_id),
  zone        TEXT NOT NULL,
  entered_at  REAL NOT NULL,
  exited_at   REAL,
  dwell_s     REAL,                   -- exited_at - entered_at
  frame_start INTEGER NOT NULL,       -- for video seek
  frame_end   INTEGER
);
```

Flat, 2 tables, obvious joins. Spark writes correct SQL against this reliably; it will not against a clever schema.

### NL -> SQL rules (non-negotiable)

- Schema pinned in the system prompt, every time.
- **Reject anything that is not a single SELECT.** Regex + sqlite read-only connection. Two layers.
- Force `LIMIT 200`.
- Show the generated SQL in the UI, always. Transparency reads as engineering maturity and turns a wrong answer into a feature demo.
- On SQL error: one retry with the error text appended. Then fail visibly.

## Measurement — the thing almost nobody will bring

Use a public tracking dataset with existing annotations (MOT17 / MOT20 style) so entity ground truth is **free**. Hand-label only the zone events on 2 clips.

Report, in the submission:
- entity tracking: precision / recall / ID switches vs. provided annotations
- zone events: caught / missed / invented, counted by hand
- cost per video-minute, measured not estimated
- one honest failure mode we found and did not fix

A measured negative beats an unmeasured claim with these judges.

## Cost control

1. **Before processing any long clip, run 10 frames and read the actual billed cost.** Extrapolate. Decide fps from that number.
2. Cache every API response to disk keyed by frame hash. Never pay twice for the same frame.
3. Keep one 30-second clip as the dev loop. Full clips only when the pipeline is frozen.
4. Budget: ~60% of credits on the eval clips, ~20% demo clip, ~20% reserve. Reserve is not optional.

## Stretch (only if core is green by Day 6)

Muse Image synthesizes 2-3 rare events the footage never contains; show the pipeline catches them. Buys a novelty claim for ~1 day of work. **Skip without guilt if behind.**

## 9-day plan (Oct 26 - Nov 4, 21:00 ET)

| Day | Deliverable |
|---|---|
| 1 | Credits live. Cost probe. Frames -> SAM -> masks on one 30s clip, cached. |
| 2 | Tracker. IDs stable across keyframes on the dev clip. |
| 3 | Zone JSON + event extractor -> SQLite rows that are correct by inspection. |
| 4 | **Freeze pipeline.** Run both eval clips. First numbers. |
| 5 | Spark NL->SQL + guardrails. 10 canned questions must all work. |
| 6 | Next.js: video, overlay, table, query box, row-click seek. |
| 7 | Bottleneck brief. Hand-label ground truth. Final numbers. |
| 8 | **Demo video (2 min).** Script it. Record it. This day is not for code. |
| 9 | Submission text, README, buffer. Submit by 18:00 ET, not 20:59. |

Day 8 is sacred. A working project with no demo video loses to a worse project with one.

## Demo video script (2 min)

1. 0:00 The problem, over real footage: "something slowed down yesterday. Finding out means watching 9 hours."
2. 0:15 Masks appearing on entities. Silent. Let it land.
3. 0:30 The events table filling. "That video is now rows."
4. 0:45 Type: *"which zone had the longest average dwell after 2pm?"* Show the SQL. Show the answer.
5. 1:10 Click a row -> video seeks to that exact moment. **This is the beat that proves the data is real.**
6. 1:30 The numbers. Precision, recall, cost per minute.
7. 1:50 The one-liner.

## Pre-window prep (now -> Oct 25)

**Check the Official Rules first** (published before Oct 26) for whether pre-existing code is allowed. If it is not, prep is limited to the no-code items below.

- Apply. **Deadline Oct 25 23:59 ET, 20,000-spot cap, rolling review.** Apply this week.
- No-code-risk prep: download datasets, pick clips, hand-label ground truth, write the demo script, read the SAM 3.1 + Spark API docs.
- If pre-existing code is permitted: ffmpeg extraction, tracker, zone math, schema, and the Next.js shell are all API-free and can be ready on Day 0.

---

# Day 1 revisions (Oct 26) — after reading the actual API docs

The locked spec above was written before the SAM 3.1 docs were read. Four
things in it were wrong, and the corrections made the project stronger rather
than smaller. Recorded here because the reasoning matters more than the spec.

## 1. SAM 3.1 takes video natively, and cost is no longer a constraint

The spec assumed per-frame image calls and budgeted around them. In fact SAM
3.1 accepts a whole clip in one request, tracks objects through it with
identity preserved, and bills **$0.20 per 1,000 frames** — not $2.50 per
1,000 images. A 2-minute clip at 2 fps costs about **$0.05**. The $150 grant
is roughly **750,000 frames**.

Consequences:
- No keyframe extraction step. The clip goes to the API whole.
- The "$150 will evaporate" risk is gone. The reserve stays anyway.
- `ss cost` still exists and still runs before any paid call, because being
  able to state the measured cost per video-minute is worth more than the
  saving.

## 2. The tracker changed role, and became more defensible, not less

The spec had us writing a tracker from scratch because SAM gave only
per-frame masks. It gives tracked ids. So the layer became **ID
reconciliation**: repairing the two failures SAM's own tracking leaves behind
(a re-appearance getting a fresh id; an id teleporting). That is a smaller
claim and a true one, and it is still our deterministic code that the numbers
rest on.

The stitch gate was rewritten during the build. A plain drift radius accepted
a *different* person 360 px away as a continuation, because at a ten-frame gap
the radius covers most of the frame. It now extrapolates the entity's own
motion and tests the residual against the prediction. Caught by a test, not by
inspection.

## 3. There is an official client library, and it changed the test strategy

`meta-sam-parser` (pip) and `@meta-sam/parser` (npm) parse the response
grammar and decode the base85 mask payloads. Writing either by hand would
have been a day of work and a liability.

More importantly, the parser ships a mask **encoder**. So the simulated scene
is rendered into *genuine SAM wire format*, written to the response cache, and
read back through the official parser. The entire production path — grammar,
mask codec, cache replay, tracker, zones, events, SQL — is exercised with no
key. The only untested piece until the key arrives is the HTTP call.

This is the single highest-leverage thing found on Day 1.

## 4. Masks come with bounds, and must be placed before they mean anything

A mask is smaller than the frame and is positioned by the record's box, with
`x=0;y=0` always. Occupancy therefore requires placing the mask into frame
coordinates and clipping at the edge — which is where a naive implementation
silently reports garbage. Covered by tests.

## Architecture, as built

```
clip.mp4
  -> ffprobe: true fps, resolution, exact frame count
  -> SAM 3.1: one concept prompt per class, whole clip  [the only paid step]
     -> response cached to disk, keyed by hash+model+prompt+encoding
  -> ID reconciliation: stitch re-appearances, split teleports    [ours]
  -> zone occupancy: mask x rasterised polygon, per frame         [ours]
  -> event compilation: hysteresis -> enter / exit / dwell        [ours]
  -> SQLite: entities, events
  -> Muse Spark: (a) NL -> SQL, four-layer guard  (b) brief over rows
  -> Next.js: video + overlay + table + query box; click row seeks video
```

## What is done as of end of Day 1

Days 1–3 of the plan, plus parts of 5, 6 and 7:

- [x] Probe, cost estimator, resample, poster extraction
- [x] SAM 3.1 client with disk cache and opt-in spending
- [x] Fixture generator: simulated scene -> real SAM wire format
- [x] ID reconciliation with predictive stitching
- [x] Polygon rasteriser and pixel-level occupancy
- [x] Hysteresis event compiler
- [x] SQLite schema, idempotent writes
- [x] Four-layer SQL guard
- [x] Muse Spark NL->SQL and brief (code complete, needs the key)
- [x] Ten canned questions, doubling as the NL->SQL benchmark
- [x] Eval: tracking and event metrics, two error sources separated
- [x] JSON API and Next.js UI with row-click seek
- [x] 125 tests, none needing a key

## What the key unblocks (Oct 28)

1. Cost probe on 10 frames, then confirm the per-minute figure.
2. One real clip through `ss ingest --video --allow-spend`.
3. Verify the live response parses identically to the fixtures.
4. Turn on `ss ask`; run the ten canned questions as NL and score the SQL.
5. Hand-label zone events on the real clip; produce the real numbers.

## Still open

- **The Official Rules.** Not published on either public page as of today.
  They are expected in the acceptance email or Discord. The one clause that
  matters is whether pre-existing code is allowed. Mitigation: every commit
  in this repo is dated inside the window, so if prior work is barred,
  nothing here is affected.
