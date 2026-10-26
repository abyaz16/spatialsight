# SpatialSight

**Pixel-level video understanding compiled into a relational schema you can query in plain English.**

A fixed camera records nine hours. Something slowed down yesterday afternoon.
Finding out what means watching the footage. SpatialSight segments the clip
with SAM 3.1, resolves each object into a stable entity, works out when that
entity entered and left hand-drawn zones, and writes the result to two SQL
tables. You then ask a question in English, see the SQL it generated, and
click a row to jump the video to the exact frame that row came from.

Built for the Meta Global AI Developer Hackathon (Oct 26 – Nov 4, 2026),
theme **Push the Frontier**, indie track.

---

## Try it in two minutes, with no API key

Everything below runs offline. `ss fixture` generates a cached SAM response
for a simulated clip in genuine SAM 3.1 wire format, so the real parser, the
real mask codec and the real pipeline all run without spending a cent. (The
cache itself is gitignored, so a paid response can never be committed by
accident.)

```bash
python -m venv .venv
PIP_TARGET= PYTHONPATH= ./.venv/Scripts/python.exe -m pip install -r requirements.txt

./scripts/ss fixture              # write the cached SAM responses
./scripts/ss ingest synthetic-01  # segment -> track -> events -> SQLite
./scripts/ss canned               # ten reference questions, answered
./scripts/ss eval --from-cache    # score the pipeline against ground truth
./scripts/dev                     # API on :8000, UI on http://localhost:3000
```

> **Two environment quirks this repo works around on the author's machine.**
> A global `pip --target` and `PYTHONPATH` would shadow the virtualenv, so the
> scripts clear `PYTHONPATH`; and OpenBLAS aborts numpy with more than one
> thread, so they set `OPENBLAS_NUM_THREADS=1`. Both are set for you in
> `scripts/*`.

## Natural language, with or without the Meta key

`ss ask` and `ss bench` run against any OpenAI-compatible endpoint, so the
text side can be exercised before the Meta Model API key arrives:

```bash
cp .env.example .env
# Either: the Meta Model API (Muse Spark) --
#   MODEL_API_KEY=...
# or any OpenAI-compatible endpoint, e.g. a Meta open-weights model --
#   LLM_BASE_URL=https://openrouter.ai/api/v1
#   LLM_MODEL=meta-llama/llama-3.3-70b-instruct
#   LLM_API_KEY=...
#   LLM_API_STYLE=chat

./scripts/ss ask "which zone had the longest average dwell?"
./scripts/ss bench                # score the ten reference questions
```

A text-model key never unlocks paid SAM 3.1 calls; those need `MODEL_API_KEY`.

## Real footage before the key

`ss ingest --local` swaps SAM 3.1 for a free local stand-in (YOLO11-seg with
ByteTrack) that emits the same `Detection`s, so the tracker, zones, events and
SQL cannot tell which ran:

```bash
./scripts/ss convert raw.avi data/clips/plaza-01.mp4   # H.264 MP4, as SAM wants
./scripts/ss poster data/clips/plaza-01.mp4 frame.png  # draw zones against this
./scripts/ss ingest plaza-01 --video data/clips/plaza-01.mp4 --local
```

It is a test stand-in, not a Meta model, and no submission number comes from
it. See [docs/FINDINGS.md](docs/FINDINGS.md) for what it found.

---

## How it works

```
clip.mp4
  │
  ├─ ffprobe ─────────── true fps, resolution, exact frame count
  │
  ├─ SAM 3.1 ─────────── one concept prompt per class, over the whole video
  │   (the only paid step: $0.20 per 1,000 frames)
  │   └─ cached to disk, keyed by clip hash + model + prompt + encoding
  │
  ├─ ID reconciliation ─ stitch re-appearances, split teleports      ← our code
  ├─ zone occupancy ──── mask × rasterised polygon, per frame        ← our code
  ├─ event compilation ─ hysteresis → enter / exit / dwell           ← our code
  │
  ├─ SQLite ──────────── entities, events
  │
  ├─ Muse Spark ──────── question → SQL (four-layer guard) → brief
  └─ Next.js ─────────── video + overlay + table + query box, row-click seeks
```

### The three deterministic layers are the point

SAM 3.1 tracks objects through video and keeps an identity per object, so the
cheap thing to do would be to take its output as the answer. The reason not to
is that two failure modes survive, and both corrupt any count built on top:

| Failure | What it does to the data | What we do |
|---|---|---|
| An entity leaves the frame and returns | Gets a new object id, so one forklift is counted as three | **Stitch**, if the new object appears where the old one's motion predicted it would |
| One object id teleports across the frame | Invents a dwell that never happened | **Split** into two entities at the discontinuity |
| An entity stands on a zone boundary | Flickers in and out, inventing dozens of entries | **Hysteresis**: enter above 50% of its pixels, leave below 30% |

The stitch test is a *prediction*, not a radius. Extrapolating an entity's own
motion across the gap and asking whether the candidate landed near where it
was heading is what separates "the same forklift, reacquired" from "a
different person who appeared five seconds later across the room". A plain
drift radius cannot tell those apart, because at a ten-frame gap the radius
covers most of the frame. ([`tracker.py`](spatialsight/tracker.py))

Occupancy is measured in **pixels, not box centroids**. A forklift whose
bounding box clips a zone corner is not in the zone; one whose mask is half
inside it is. That distinction is the only reason to segment rather than
detect, so it is computed exactly, against a polygon rasterised once per zone.
([`zones.py`](spatialsight/zones.py))

### Letting a model write SQL, safely

Four independent layers, all of which the SQL must pass
([`query/guard.py`](spatialsight/query/guard.py)):

1. **Statement shape** — one statement, `SELECT` or a `WITH` that resolves to
   one. Comments are stripped first, because `--` is how a second statement
   hides from a naive regex.
2. **Forbidden keywords** — matched as whole words against the stripped text,
   so a column called `updated_at` is not mistaken for an `UPDATE`.
3. **SQLite's own authorizer** — vetoes anything but a read, during statement
   preparation, before a row is touched.
4. **A read-only connection** — so a successful bypass has nothing to write to.

Layer 3 is the one worth pointing at. The first two are guesses about what an
attacker might write; the third is a statement about what the database will
permit. The guard lives in Python and the UI has no database access at all,
because porting it to TypeScript would mean two implementations of the one
security boundary in the project — and the second would not be the tested one.

**The generated SQL is always on screen**, including when it fails. A wrong
answer whose SQL is visible is something a user can judge and correct; an
answer with nothing behind it has to be taken on faith.

---

## Measurement

Three levels, because they answer different questions.

### Against MOT17 — our code on real human motion

```bash
./scripts/ss mot            # 7 sequences, 519 hand-annotated identities
```

The boxes are human annotations, so detection error is exactly zero and
anything wrong is ours. Realistic fragmentation is derived from the
annotations' own occlusion gaps, as an upstream tracker would produce.

| | true | upstream | ours | merges | recovered |
|---|---|---|---|---|---|
| 7 sequences | 519 | 937 | 753 | 7.1% | 44% |

Reconciliation removes 44% of the upstream over-count (1.8× → 1.5×) while
merging 7.1% of identities. The first attempt merged 18.9%; the gap between
those two numbers is three bugs that only ground truth could expose, written
up in [docs/FINDINGS.md](docs/FINDINGS.md).

Merges and fragments are deliberately *not* weighted equally. A fragment
reports one person as two entities — the count is too high but every history
is real. A merge reports two people as one, with a dwell assembled from two
different people: the count is too low *and* the history is fabricated, and
nothing downstream can detect it.

### Against the simulated clip — the layers end to end

Run `./scripts/ss eval --from-cache`. The report separates two errors that have
different fixes:

- **Tracking** — precision, recall, ID switches and fragmentations, from
  matching predicted tracks to true actors frame by frame.
- **Zone events** — caught / missed / invented, plus dwell error, matched on
  zone, class and temporal overlap.

On the simulated clip the truth is exact, so these measure *our code alone*.
On real footage the truth is hand-labelled and the same functions run, so the
two are directly comparable and the gap between them is what the model
contributes.

### Current numbers (simulated clip, 240 frames, 5 actors)

| | |
|---|---|
| Detection precision / recall | 1.000 / 1.000 |
| Entities | 5 predicted vs 5 true |
| Events caught / missed / invented | 6 / 0 / 0 |
| Dwell error | 0.00 s mean |
| ID switches | **2** |
| Fragmentations | **2** |
| Event table vs footage | ~1,800× smaller |

### The failure mode we did not fix

The simulated scene swaps the object ids of two people at the moment they pass
each other. The tracker does not catch it, and the eval reports it rather than
hiding it. This is not an oversight: an ID switch between two objects that are
touching produces **no geometric discontinuity**, so no amount of motion
modelling can see it. Fixing it needs an appearance embedding per entity,
which is a different system. The honest statement is that zone-level analytics
are correct here and per-entity histories are not, across a crossing.

---

## Cost control

SAM 3.1 video is $0.20 per 1,000 frames, so cost is a design input, not an
afterthought.

```bash
./scripts/ss cost clip.mp4 --budget 150   # before paying anything
```

- **Every response is cached** on disk, keyed by clip hash + model + prompt +
  mask encoding. Re-running the pipeline after editing the tracker is free.
- **`ss resample`** re-times a clip before segmenting. A loading dock at 2 fps
  costs a fifteenth of the same clip at 30 fps and loses nothing of interest.
- **Spending is opt-in.** A cache miss raises rather than silently billing;
  paid calls need `--allow-spend`.
- Documented limits (15,000 frames per request, 50-frame billing minimum, 16
  tracked objects per frame) are enforced locally, so a bad request fails
  before it is billed rather than after.

---

## Commands

```
ss probe    <video>           true fps, resolution, exact frame count
ss cost     <video>           what segmenting would cost, before paying
ss resample <video> <out>     trim and re-time to cut frame cost
ss poster   <video> <out.png> one frame, to draw zones against
ss fixture                    build the offline cached clip (no key)
ss ingest   <clip>            segment, track, compile events, write SQL
ss events   [clip]            print the events table
ss sql      "<select ...>"    run SQL through the guard
ss ask      "<question>"      English → SQL → rows → brief  (needs a key)
ss canned   [key]             the ten reference questions   (no key)
ss eval     [clip]            score against the simulated clip
ss mot                        score reconciliation against MOT17 ground truth
ss cache                      what has been paid for, and what it cost
ss schema                     the schema pinned into the SQL prompt
```

## Layout

```
spatialsight/
  types.py          Detection / Mask / Track — the seam between stages
  probe.py          ffprobe: fps, size, exact frame count, codec checks
  backends/
    base.py         the one interface the pipeline knows about
    sam.py          SAM 3.1 over the Meta Model API, plus a cached replay
    synthetic.py    a simulated camera with exact ground truth
  cache.py          disk cache for responses, keyed by request
  fixtures.py       renders the simulated scene into real SAM wire format
  tracker.py        ID reconciliation: stitch and split
  zones.py          polygon rasteriser and pixel-level occupancy
  events.py         hysteresis → enter / exit / dwell
  db.py             two tables, obvious joins
  runner.py         orchestration, deliberately thin
  eval.py           metrics, with the two error sources separated
  query/
    guard.py        the four-layer SQL boundary
    nlsql.py        Muse Spark: question → SQL, and the brief
    canned.py       ten reference questions with hand-written SQL
  server.py         JSON API on the standard library alone
  cli.py            every stage, runnable on its own
web/                Next.js: video, overlay, table, query box
tests/              125 tests, no API key required
zones/              hand-drawn polygons per camera
```

## Testing

```bash
./scripts/test
```

125 tests, none of which need a key or a network. The pipeline is covered
end to end through the **real** parse path: the simulated scene is encoded
into genuine SAM wire format using the official parser's own mask encoder,
written to the response cache, and read back through the official parser. The
only thing a live key adds is the HTTP call itself.

## Models used

| Model | Used for |
|---|---|
| `sam-3.1` | Segmentation and tracking, via the Meta Model API |
| `muse-spark-1.3` | Question → SQL, and the brief over the result rows |
| Muse Code | Used to build this. Not a runtime component. |
