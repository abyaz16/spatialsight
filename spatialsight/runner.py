"""Orchestration: clip in, rows out.

Deliberately thin. Every decision worth defending lives in `tracker`,
`zones` or `events`; this file only wires them together in the right order
and reports what each stage did.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .appearance import Appearance, describe_tracks, reader_for
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
    dropped: int
    events: int
    zones: tuple[str, ...]
    zone_pixels: dict[str, int] = field(default_factory=dict)
    coloured: int = 0
    colour_conflicts: int = 0

    def lines(self) -> list[str]:
        out = [
            f"clip          {self.clip}  ({self.backend} backend)",
            f"video         {self.meta.width}x{self.meta.height} @ {self.meta.fps:g} fps, "
            f"{self.meta.frame_count} frames, {self.meta.duration_s:.1f}s",
            f"detections    {self.detections} across {self.raw_object_ids} raw object ids",
            f"entities      {self.entities}  "
            f"(+{self.stitches} stitched, +{self.splits} split, "
            f"-{self.dropped} dropped as noise)",
            f"zones         {', '.join(self.zones)}",
            f"events        {self.events}",
        ]
        if self.coloured:
            out.append(
                f"colour        read for {self.coloured} of {self.entities} entities"
            )
            if self.colour_conflicts:
                out.append(
                    f"              {self.colour_conflicts} changed colour mid-life "
                    f"-- identity suspect"
                )
        return out


def run(
    backend: SegmentationBackend,
    clip: str,
    zones_path: str | Path,
    db_path: str | Path,
    labels: tuple[str, ...] = ("person", "vehicle"),
    tracker_params: TrackerParams | None = None,
    event_params: EventParams | None = None,
    with_colour: bool = True,
) -> tuple[RunReport, Reconciliation, list[ZoneEvent]]:
    meta = backend.meta()
    zone_set: ZoneSet = load_zones(zones_path)
    raster = ZoneRaster(zone_set, meta.width, meta.height)

    detections: list[Detection] = backend.detect(labels)
    recon = reconcile(detections, tracker_params)
    events = compile_events(recon.tracks, raster, meta, event_params)

    # Colour is read after reconciliation, not before: it describes an
    # entity, and an entity does not exist until the fragments have been
    # stitched. Reading it per detection would also pay for the same pixels
    # repeatedly.
    appearances: dict[int, Appearance] = {}
    if with_colour:
        reader = getattr(backend, "frame_reader", lambda: reader_for(meta.path))()
        appearances = describe_tracks(recon.tracks, reader)

    db_path = Path(db_path)
    con = connect(db_path)
    try:
        write_clip(con, clip, meta, recon.tracks, events, appearances)
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
            # Whatever the backend keys its cache on, so a later reader can
            # rebuild it identically instead of re-running the model.
            **getattr(backend, "manifest_extra", dict)(),
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
        dropped=recon.dropped,
        events=len(events),
        zones=zone_set.names,
        zone_pixels=dict(raster.areas),
        coloured=len(appearances),
        colour_conflicts=sum(1 for a in appearances.values() if not a.stable),
    )
    return report, recon, events
