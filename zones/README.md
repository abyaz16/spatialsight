# Drawing zones

One JSON file per camera, named after the clip: `zones/<clip>.json`.

```json
{
  "camera": "warehouse-01",
  "frame": { "width": 1920, "height": 1080 },
  "zones": [
    { "name": "loading_dock", "polygon": [[120, 80], [900, 80], [900, 620], [120, 620]] }
  ]
}
```

## How to produce the coordinates

```bash
./scripts/ss poster data/clips/warehouse-01.mp4 docs/warehouse-01.png
```

Open the PNG in any image editor, read the pixel coordinates off the corners
you want, and type them in clockwise. Three to five zones per camera is the
sweet spot: fewer and the questions get vague, more and the events table fills
with noise from zones nobody asks about.

## Rules

- **`frame` must be the size you read the coordinates against**, not
  necessarily the clip's size. Zones are scaled automatically if the clip is a
  different resolution, so a polygon drawn on a 1920×1080 still works on a
  1280×720 re-encode of the same shot.
- At least 3 points. Concave shapes are fine — the rasteriser is an even-odd
  scanline fill and handles the notch correctly.
- Zone names must be unique, and should be `snake_case`: they go into SQL
  rows and into the model's prompt, and `loading_dock` is easier for both to
  get right than `Loading Dock #2`.
- Zones may overlap. An entity in two overlapping zones produces an event in
  each, which is usually what you want for a nested area like `dock` inside
  `warehouse_floor`.

## Choosing the thresholds

Occupancy is the fraction of an entity's own mask inside the polygon, so the
right thresholds depend on how the camera sees the zone, not on the zone's
size. The defaults (`--enter-at 0.50`, `--exit-at 0.30`) suit a camera looking
down at a floor area.

For a zone viewed at a shallow angle, where an entity's mask is mostly its own
height and only its feet are ever inside, lower both:

```bash
./scripts/ss ingest warehouse-01 --enter-at 0.25 --exit-at 0.12
```

Keep `exit-at` meaningfully below `enter-at`. Setting them equal removes the
hysteresis and brings back the boundary flicker it exists to prevent.
