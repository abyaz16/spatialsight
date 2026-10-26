"""A free, local segmentation backend: YOLO11-seg with ByteTrack, on the GPU.

This is a **stand-in** for SAM 3.1, used to run real footage end to end
before the Meta Model API key arrives. It is not a Meta model and the
submission's numbers do not come from it. Its job is to answer one question
early: does the pipeline downstream of segmentation hold up on real pixels,
with real occlusions, real jitter and a tracker that is not ours?

It emits exactly what the SAM backend emits -- `Detection`s with a box, a
mask placed by that box, and a per-class object id -- so the tracker, the
zones, the events and the SQL cannot tell which one ran. Swapping back to
SAM 3.1 is a command-line flag, not a code change.

Two choices worth knowing about:

**Masks are re-rasterised from YOLO's polygons**, not taken from its mask
tensor. The tensor lives in the model's letterboxed input space; the polygon
is already in frame coordinates. Rasterising the polygon over its own box at
a fixed resolution produces a mask in the same convention SAM uses (a small
raster stretched over its bounds), with the same scanline fill the zones use.

**Results are cached**, keyed by the clip's hash and every parameter that
changes the answer, exactly as SAM responses are. A GPU pass over a minute of
video is quick but not free, and the tracker gets re-run far more often than
the detector.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Sequence

from ..probe import probe_video
from ..types import Box, Detection, Mask, VideoMeta
from ..zones import Zone, rasterise

# COCO class ids, folded into the two classes the schema knows about.
COCO_TO_LABEL = {
    0: "person",
    1: "vehicle",  # bicycle
    2: "vehicle",  # car
    3: "vehicle",  # motorcycle
    5: "vehicle",  # bus
    7: "vehicle",  # truck
}

MASK_RES = 32


def _rle(data: bytes) -> list[int]:
    runs: list[int] = []
    current, count = 0, 0
    for value in data:
        bit = 1 if value else 0
        if bit == current:
            count += 1
        else:
            runs.append(count)
            current, count = bit, 1
    runs.append(count)
    return runs


def _unrle(runs: list[int]) -> bytes:
    out = bytearray()
    value = 0
    for run in runs:
        out.extend([value] * run)
        value ^= 1
    return bytes(out)


def polygon_to_mask(polygon, box: Box, res: int = MASK_RES) -> Mask:
    """Rasterise a frame-space polygon over its own box, at `res` x `res`."""
    width = max(1e-6, box.width)
    height = max(1e-6, box.height)
    local = tuple(
        ((x - box.left) / width * res, (y - box.top) / height * res) for x, y in polygon
    )
    grid = rasterise(Zone("mask", local), res, res)
    return Mask(rows=res, cols=res, data=grid.astype("uint8").tobytes(), box=box)


class LocalBackend:
    """YOLO11-seg + ByteTrack, run locally. A stand-in, not the submission model."""

    name = "yolo11-seg+bytetrack (local stand-in)"

    def __init__(
        self,
        video_path: str | Path,
        *,
        weights: str | Path = "E:/models/yolo11s-seg.pt",
        cache_dir: str | Path = "data/cache",
        conf: float = 0.30,
        imgsz: int = 960,
        device: str | int = 0,
    ) -> None:
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(self.video_path)
        self.weights = Path(weights)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.conf = conf
        self.imgsz = imgsz
        self.device = device
        self._meta: VideoMeta | None = None
        self.last_run: dict[str, object] = {}

    def meta(self) -> VideoMeta:
        if self._meta is None:
            self._meta = probe_video(self.video_path, strict=False)
        return self._meta

    def manifest_extra(self) -> dict:
        """The parameters this backend's cache key is built from."""
        return {
            "weights": str(self.weights),
            "imgsz": self.imgsz,
            "conf": self.conf,
            "mask_res": MASK_RES,
            "tracker": "bytetrack",
        }

    def _cache_path(self) -> Path:
        meta = self.meta()
        key = "|".join(
            [
                meta.sha256,
                self.weights.name,
                f"conf={self.conf}",
                f"imgsz={self.imgsz}",
                f"res={MASK_RES}",
                "bytetrack",
            ]
        )
        digest = hashlib.sha256(key.encode()).hexdigest()[:40]
        return self.cache_dir / f"local-{digest}.json"

    def detect(self, labels: Sequence[str] = ("person", "vehicle")) -> list[Detection]:
        wanted = set(labels)
        path = self._cache_path()
        if path.exists():
            records = json.loads(path.read_text(encoding="utf-8"))["detections"]
            self.last_run = {"cached": True, "frames": self.meta().frame_count}
        else:
            records = self._run_model()
            path.write_text(
                json.dumps(
                    {
                        "clip": self.video_path.name,
                        "model": self.weights.name,
                        "tracker": "bytetrack",
                        "detections": records,
                    }
                ),
                encoding="utf-8",
            )
        return [self._to_detection(r) for r in records if r["label"] in wanted]

    def _run_model(self) -> list[dict]:
        import time

        import torch
        from ultralytics import YOLO

        device = self.device if torch.cuda.is_available() else "cpu"
        if str(device) == "cpu":
            # oneDNN pre-allocates generously per primitive; on a machine near
            # its commit limit that fails with "could not create a primitive".
            # The plain kernels are slower but allocate far less.
            torch.backends.mkldnn.enabled = False
            torch.set_num_threads(max(1, int(os.environ.get("SS_CPU_THREADS", "4"))))
        model = YOLO(str(self.weights))
        started = time.perf_counter()
        records: list[dict] = []
        frame = -1
        for result in model.track(
            source=str(self.video_path),
            stream=True,
            persist=True,
            tracker="bytetrack.yaml",
            conf=self.conf,
            imgsz=self.imgsz,
            classes=sorted(COCO_TO_LABEL),
            device=device,
            retina_masks=False,
            verbose=False,
        ):
            frame += 1
            boxes = result.boxes
            if boxes is None or boxes.id is None or result.masks is None:
                continue
            ids = boxes.id.int().tolist()
            classes = boxes.cls.int().tolist()
            confs = boxes.conf.tolist()
            xyxy = boxes.xyxy.tolist()
            polygons = result.masks.xy
            for i, track_id in enumerate(ids):
                label = COCO_TO_LABEL.get(classes[i])
                if label is None or i >= len(polygons) or len(polygons[i]) < 3:
                    continue
                left, top, right, bottom = xyxy[i]
                box = Box(left, top, right, bottom)
                mask = polygon_to_mask([tuple(p) for p in polygons[i]], box)
                records.append(
                    {
                        "frame": frame,
                        # Namespaced by class, as with SAM: the tracker ids are
                        # per-model, and two classes must never share one.
                        "object_id": f"{label}:{track_id}",
                        "label": label,
                        "box": [left, top, right, bottom],
                        "confidence": round(float(confs[i]), 4),
                        "mask": {"rows": mask.rows, "cols": mask.cols, "runs": _rle(mask.data)},
                    }
                )
        elapsed = time.perf_counter() - started
        self.last_run = {
            "cached": False,
            "frames": frame + 1,
            "seconds": round(elapsed, 2),
            "fps": round((frame + 1) / elapsed, 1) if elapsed else 0.0,
            "device": str(device),
        }
        return records

    @staticmethod
    def _to_detection(record: dict) -> Detection:
        left, top, right, bottom = record["box"]
        box = Box(left, top, right, bottom)
        m = record["mask"]
        mask = Mask(rows=m["rows"], cols=m["cols"], data=_unrle(m["runs"]), box=box)
        return Detection(
            frame=record["frame"],
            object_id=record["object_id"],
            label=record["label"],
            box=box,
            mask=mask,
            confidence=record.get("confidence"),
        )
