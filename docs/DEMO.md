# Demo video — 2 minutes, shot list

Record on **Nov 2 (Day 8)**. That day is not for code. A working project with
no demo video loses to a worse project with one.

## Before recording

```bash
./scripts/ss ingest <real-clip> --video data/clips/<real-clip>.mp4   # cached by now
./scripts/ss eval <real-clip> --json docs/numbers.json
./scripts/dev
```

Check: the browser window is 1600 px wide or more, the events table shows the
real clip, and `ss ask` answers all ten canned questions. Close every other
tab. Hide bookmarks.

## Shot list

| Time | On screen | Said |
|---|---|---|
| 0:00 | Raw footage playing, no overlay | "A fixed camera recorded nine hours yesterday. Something slowed down. Finding out means watching it." |
| 0:15 | Masks fade in on the entities. **Silence.** Let it land for a full four seconds. | — |
| 0:30 | Zones appear; the events table fills row by row | "That video is now rows. One row is one entity, in one zone, for one period." |
| 0:45 | Type: *which zone had the longest average dwell?* | "You ask in English." |
| 0:55 | The generated SQL appears, then the answer | "It shows you the SQL. If it got the question wrong, you can see that it did." |
| 1:10 | **Click a row. The video jumps to that frame, the entity highlighted.** Hold for three seconds. | "And the row goes back to the footage it came from." |
| 1:25 | Terminal: `ss sql "DROP TABLE events"` → refused | "A model writing SQL will eventually write DROP TABLE. Four layers stop it; the one that matters is SQLite's own authorizer." |
| 1:35 | `ss eval` output on screen | "Precision, recall, events caught and missed, measured against ground truth. And two ID switches we don't fix — when two people cross, there's no geometry to see it by. The eval says so." |
| 1:52 | The one-liner, over the first frame | "Pixel-level video understanding, compiled into a schema you can query in plain English." |

## The beat that matters

**1:10.** Everything before it is a claim; the row-to-frame jump is the proof
the table is derived from the footage rather than asserted alongside it. If
the recording is going badly, protect that shot and cut anything else.

## What not to do

- No slides. No architecture diagram. The repo has one.
- Do not narrate the stack. Judges can read `README.md`.
- Do not claim real-time, live ingest, or multi-camera. None are built.
- Do not hide the ID switch. Stating a measured limitation reads as
  confidence; omitting it reads as not having measured.
