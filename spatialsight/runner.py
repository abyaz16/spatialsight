"""Orchestration: clip in, rows out.

Deliberately thin. Every decision worth defending lives in `tracker`,
`zones` or `events`; this file only wires them together in the right order
and reports what each stage did.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .backends.base import SegmentationBackend
from .db import connect, write_clip, write_manifest
from .events import EventParams, ZoneEvent, compile_events
from .tracker import Reconciliation, TrackerParams, reconcile
from .types import Detection, VideoMeta
from .zones import ZoneRaster, ZoneSet, load_zones


@dataclass
class RunReport:
    clip: str
    backend: str
    meta: VideoMeta
    detections: int
    raw_object_ids: int
    entities: int
    stitches: int
    splits: int
    events: int
    zones: tuple[str, ...]
    zone_pixels: dict[str, int] = field(default_factory=dict)

    def lines(self) -> list[str]:
        return [
            f"clip          {self.clip}  ({self.backend} backend)",
            f"video         {self.meta.width}x{self.meta.height} @ {self.meta.fps:g} fps, "
            f"{self.meta.frame_count} frames, {self.meta.duration_s:.1f}s",
            f"detections    {self.detections} across {self.raw_object_ids} raw object ids",
            f"entities      {self.entities}  "
            f"(+{self.stitches} stitched, +{self.splits} split)",
            f"zones         {', '.join(self.zones)}",
            f"events        {self.events}",
        ]


def run(
    backend: SegmentationBackend,
    clip: str,
    zones_path: str | Path,
    db_path: str | Path,
    labels: tuple[str, ...] = ("person", "vehicle"),
    tracker_params: TrackerParams | None = None,
    event_params: EventParams | None = None,
) -> tuple[RunReport, Reconciliation, list[ZoneEvent]]:
    meta = backend.meta()
    zone_set: ZoneSet = load_zones(zones_path)
    raster = ZoneRaster(zone_set, meta.width, meta.height)

    detections: list[Detection] = backend.detect(labels)
    recon = reconcile(detections, tracker_params)
    events = compile_events(recon.tracks, raster, meta, event_params)

    db_path = Path(db_path)
    con = connect(db_path)
    try:
        write_clip(con, clip, meta, recon.tracks, events)
    finally:
        con.close()
    write_manifest(
        db_path.parent / "clips.json",
        clip,
        meta,
        {
            "backend": backend.name,
            # Two distinct keys on purpose: `zones_path` is where the polygons
            # live, `zone_names` is what they are called. Collapsing them into
            # one `zones` key makes a later ingest overwrite the path with a
            # list of names, which then reads as a missing zone file.
            "zones_path": str(Path(zones_path)),
            "zone_names": list(zone_set.names),
        },
    )

    report = RunReport(
        clip=clip,
        backend=backend.name,
        meta=meta,
        detections=len(detections),
        raw_object_ids=len({d.object_id for d in detections}),
        entities=len(recon.tracks),
        stitches=recon.stitches,
        splits=recon.splits,
        events=len(events),
        zones=zone_set.names,
        zone_pixels=dict(raster.areas),
    )
    return report, recon, events
