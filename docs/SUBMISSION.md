# Submission draft

Fill the bracketed numbers from `./scripts/ss eval <clip> --json` on **Day 7**.
Submit by **18:00 ET on Nov 4**, not 20:59.

---

## One line

Pixel-level video understanding compiled into a relational schema you can
query in plain English.

## Track and theme

Indie developer track. Theme: **Push the Frontier**.

## The problem

A fixed camera produces footage nobody watches. The questions people actually
have about it are relational — *which zone held things up, for how long, and
who was there* — but the data is pixels. Answering takes a person scrubbing a
timeline, so in practice the question goes unanswered and the footage is kept
only for disputes.

## What it does

Segments a clip with SAM 3.1, resolves each object into a stable entity,
measures when that entity's pixels enter and leave hand-drawn zones, and
writes two SQL tables. You ask a question in English, see the generated SQL,
and click a result row to jump the video to the exact frame it came from.

## What is ours, and why that is the interesting part

SAM 3.1 does segmentation and tracking well, so the work is not in calling it.
It is in the three deterministic layers between the model and the table, each
of which fixes a failure that would otherwise corrupt every count built on it:

1. **ID reconciliation.** An entity that leaves the frame and returns gets a
   fresh object id, so one forklift reads as three. We stitch the two ids
   together — but only when the second appears where the first one's own
   motion predicted. A plain distance radius accepts a different person across
   the room, because at a ten-frame gap the radius covers most of the frame.
2. **Pixel-level zone occupancy.** Occupancy is the fraction of an entity's
   *own mask* inside a polygon, not whether a box corner clipped it. This is
   the only reason to segment rather than detect, so it is computed exactly
   against a polygon rasterised once per zone.
3. **Hysteresis.** An entity standing on a zone boundary flickers, and each
   flicker invents an entry and an exit. Entry requires more than 50% of its
   pixels inside, exit fewer than 30%. One visit, not forty.

## Measurement

| Metric | Simulated clip | Real clip `[name]` |
|---|---|---|
| Detection precision | 1.000 | `[ ]` |
| Detection recall | 1.000 | `[ ]` |
| Entities (predicted vs true) | 5 vs 5 | `[ ]` |
| Events caught | 6 of 6 | `[ ]` |
| Events missed | 0 | `[ ]` |
| Events invented | 0 | `[ ]` |
| Mean dwell error | 0.00 s | `[ ]` |
| ID switches | 2 | `[ ]` |
| Cost per video-minute | $0.00 (cached) | `$[ ]` measured |
| Event table vs footage | ~1,800× smaller | `[ ]×` |

On the simulated clip the ground truth is exact by construction, so those
numbers measure **our code alone**, with the model's error removed. On the
real clip the truth is hand-labelled and the same code runs, so the difference
between the two columns is what the model contributes. Separating them was
deliberate: they have different fixes.

### The failure mode we did not fix

When two people cross, SAM can swap their ids, and we do not catch it. This is
not an oversight. An ID switch between two objects that are touching produces
**no geometric discontinuity**, so no amount of motion modelling can detect
it; it needs an appearance embedding per entity, which is a different system.
The eval reports it as `id_switches` rather than hiding it. Zone-level
analytics on this clip are correct; per-entity histories across a crossing are
not.

## Letting a model write SQL against your database

Four independent layers, all of which the SQL must pass: statement shape
(comments stripped first, so a hidden second statement is revealed), a
whole-word keyword deny list, **SQLite's own authorizer** vetoing anything but
a read during statement preparation, and a read-only connection underneath.

The third layer is the one that holds. The first two are guesses about what an
attacker might write; the authorizer is a statement about what the database
will permit. The guard lives in one place, in Python, and the UI has no
database access at all — porting it to TypeScript would have meant two
implementations of the project's single security boundary, and the second
would not have been the tested one.

The generated SQL is always visible, including on failure.

## Reproducibility

`./scripts/test` runs 125 tests with no API key and no network. The pipeline is
covered end to end through the real parse path: the simulated scene is encoded
into genuine SAM 3.1 wire format using the official parser's own mask encoder,
written to the response cache, and read back through the official parser. A
judge can clone the repo, run three commands, and get the simulated clip's
numbers without credentials.

## Models used

- `sam-3.1` — segmentation and tracking, Meta Model API
- `muse-spark-1.3` — question → SQL, and the brief over result rows
- Muse Code — used to build this; not a runtime component

## Scope, stated plainly

Recorded files, one fixed camera, two entity classes, 3–5 zones per camera,
three event types. No live ingest, no multi-camera, no re-identification
across cameras, no model training. Each of those is a different project, and
claiming them would be claiming untested code.

## Links

- Repo: `[url]`
- Demo video: `[url]`
