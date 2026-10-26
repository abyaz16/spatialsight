"""Reading a clip's true dimensions, frame rate and frame count.

Every timestamp in the events table is derived from `fps`, so this is not a
detail: guessing 30 fps on a 25 fps clip puts every answer 20% out, and the
error is invisible because the numbers still look plausible. The frame count
comes from an exact packet count rather than `duration * fps`, which rounds.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from .types import VideoMeta

FFPROBE_HINT = (
    "ffprobe was not found on PATH. Install ffmpeg and reopen the shell:\n"
    "  winget install Gyan.FFmpeg      (Windows)\n"
    "  brew install ffmpeg             (macOS)\n"
    "  sudo apt install ffmpeg         (Debian/Ubuntu)"
)

# SAM 3.1 accepts these containers and codecs; anything else is rejected
# server-side, so it is worth refusing locally with a clearer message.
SUPPORTED_CONTAINERS = {".mp4", ".mov"}
SUPPORTED_CODECS = {"h264", "hevc", "mpeg4"}


def ffprobe_path() -> str:
    exe = shutil.which("ffprobe")
    if exe is None:
        raise RuntimeError(FFPROBE_HINT)
    return exe


def ffmpeg_path() -> str:
    exe = shutil.which("ffmpeg")
    if exe is None:
        raise RuntimeError(FFPROBE_HINT.replace("ffprobe", "ffmpeg"))
    return exe


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def _rate(value: str) -> float:
    """Parse ffprobe's rational frame rates, e.g. '30000/1001'."""
    if "/" in value:
        num, den = value.split("/", 1)
        den_f = float(den)
        return float(num) / den_f if den_f else 0.0
    return float(value)


def probe_video(path: str | Path, *, strict: bool = True) -> VideoMeta:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    out = subprocess.run(
        [
            ffprobe_path(),
            "-v", "error",
            "-select_streams", "v:0",
            "-count_packets",
            "-show_entries",
            "stream=width,height,avg_frame_rate,r_frame_rate,nb_read_packets,codec_name",
            "-show_entries", "format=duration",
            "-of", "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(out.stdout)
    if not data.get("streams"):
        raise ValueError(f"{path.name} has no video stream")
    stream = data["streams"][0]

    codec = stream.get("codec_name", "")
    if strict:
        if path.suffix.lower() not in SUPPORTED_CONTAINERS:
            raise ValueError(
                f"{path.name}: SAM 3.1 accepts MP4 or MOV, not {path.suffix}. "
                f"Convert it with `ss convert`."
            )
        if codec not in SUPPORTED_CODECS:
            raise ValueError(
                f"{path.name}: codec {codec!r} is not accepted by SAM 3.1 "
                f"(need h264, hevc or mpeg4). Convert it with `ss convert`."
            )

    fps = _rate(stream.get("avg_frame_rate") or "0") or _rate(
        stream.get("r_frame_rate") or "0"
    )
    if fps <= 0:
        raise ValueError(f"{path.name}: could not determine frame rate")

    frame_count = int(stream.get("nb_read_packets") or 0)
    duration = float(data.get("format", {}).get("duration") or 0.0)
    if frame_count <= 0:
        frame_count = int(round(duration * fps))

    return VideoMeta(
        path=str(path),
        width=int(stream["width"]),
        height=int(stream["height"]),
        fps=fps,
        frame_count=frame_count,
        duration_s=duration or (frame_count / fps),
        sha256=sha256_file(path),
    )


def resample(
    src: str | Path, dst: str | Path, fps: float, *, start: float = 0.0, seconds: float | None = None
) -> Path:
    """Write a trimmed, re-timed copy of a clip.

    SAM bills per frame, so sampling a 30 fps clip down to 2 fps is a 15x
    cost reduction for a loading dock, where nothing of interest happens
    inside half a second. The output keeps H.264 in MP4 with faststart, which
    is what the API wants and avoids a server-side remux.
    """
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [ffmpeg_path(), "-y"]
    if start:
        cmd += ["-ss", str(start)]
    cmd += ["-i", str(src)]
    if seconds is not None:
        cmd += ["-t", str(seconds)]
    cmd += [
        "-vf", f"fps={fps}",
        "-an",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(dst),
    ]
    subprocess.run(cmd, capture_output=True, text=True, check=True)
    return dst


def extract_frame(src: str | Path, dst: str | Path, frame: int = 0) -> Path:
    """Pull one frame out as a PNG, for drawing zones against."""
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            ffmpeg_path(), "-y",
            "-i", str(src),
            "-vf", f"select=eq(n\\,{frame})",
            "-vframes", "1",
            str(dst),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return dst
