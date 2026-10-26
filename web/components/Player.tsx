"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  entityColour,
  formatSeconds,
  type Overlay,
  type OverlayItem,
  type ZoneSet,
} from "@/lib/api";

type Props = {
  clip: string;
  overlay: Overlay;
  zones: ZoneSet;
  simulated: boolean;
  frame: number;
  onFrameChange: (frame: number) => void;
  highlightEntity: number | null;
};

/**
 * The video and its overlay.
 *
 * Two things here are load-bearing for the demo.
 *
 * Masks are drawn from run-length data onto a small offscreen canvas at the
 * mask's own resolution, then stretched into the box it occupies. That is
 * exactly how the pipeline places a mask in frame coordinates, so what is on
 * screen is the same geometry the zone occupancy was computed from, not a
 * prettier approximation of it.
 *
 * Seeking is driven by frame index, never by seconds. A row in the events
 * table records the frame an event started, and `frame / fps` is the only
 * conversion anywhere. Rounding a timestamp back to a frame is how a seek
 * lands one frame off and the overlay stops matching the picture.
 */
export default function Player({
  clip,
  overlay,
  zones,
  simulated,
  frame,
  onFrameChange,
  highlightEntity,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const maskCanvas = useRef<HTMLCanvasElement | null>(null);
  const [playing, setPlaying] = useState(false);

  const { width, height, fps, frame_count: frameCount } = overlay;
  const items: OverlayItem[] = useMemo(
    () => overlay.frames[String(frame)] ?? [],
    [overlay, frame],
  );

  const scaleX = zones.frame.width === 0 ? 1 : width / zones.frame.width;
  const scaleY = zones.frame.height === 0 ? 1 : height / zones.frame.height;

  // ----- drawing ----------------------------------------------------------

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.clearRect(0, 0, width, height);

    if (simulated) {
      // No footage for a simulated clip, so the scene is the backdrop.
      ctx.fillStyle = "#11161d";
      ctx.fillRect(0, 0, width, height);
      ctx.strokeStyle = "rgba(255,255,255,0.04)";
      ctx.lineWidth = 1;
      for (let x = 0; x < width; x += 64) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
        ctx.stroke();
      }
      for (let y = 0; y < height; y += 64) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(width, y);
        ctx.stroke();
      }
    }

    // Zones first, so entities sit on top of them.
    for (const zone of zones.zones) {
      ctx.beginPath();
      zone.polygon.forEach(([x, y], index) => {
        const px = x * scaleX;
        const py = y * scaleY;
        if (index === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      });
      ctx.closePath();
      ctx.fillStyle = "rgba(79,156,249,0.08)";
      ctx.fill();
      ctx.strokeStyle = "rgba(120,180,255,0.55)";
      ctx.setLineDash([6, 4]);
      ctx.lineWidth = 2;
      ctx.stroke();
      ctx.setLineDash([]);

      const [fx, fy] = zone.polygon[0];
      ctx.font = "600 15px ui-monospace, monospace";
      ctx.fillStyle = "rgba(170,205,255,0.9)";
      ctx.fillText(zone.name, fx * scaleX + 8, fy * scaleY + 22);
    }

    // Entities.
    for (const item of items) {
      const [left, top, right, bottom] = item.box;
      const colour = entityColour(item.entity_id);
      const dimmed = highlightEntity !== null && highlightEntity !== item.entity_id;
      ctx.globalAlpha = dimmed ? 0.25 : 1;

      if (item.mask) {
        drawMask(ctx, item.mask, left, top, right, bottom, colour, maskCanvas);
      }

      ctx.strokeStyle = colour;
      ctx.lineWidth = highlightEntity === item.entity_id ? 4 : 2;
      ctx.strokeRect(left, top, right - left, bottom - top);

      const label = `${item.class} #${item.entity_id}`;
      ctx.font = "600 14px ui-monospace, monospace";
      const textWidth = ctx.measureText(label).width;
      ctx.fillStyle = colour;
      ctx.fillRect(left, Math.max(0, top - 22), textWidth + 12, 20);
      ctx.fillStyle = "#0b0e13";
      ctx.fillText(label, left + 6, Math.max(13, top - 7));
      ctx.globalAlpha = 1;
    }

    // Frame stamp, so a screenshot is self-describing.
    ctx.font = "600 14px ui-monospace, monospace";
    ctx.fillStyle = "rgba(0,0,0,0.55)";
    ctx.fillRect(width - 186, height - 36, 174, 26);
    ctx.fillStyle = "#cfe3ff";
    ctx.fillText(
      `frame ${frame} · ${formatSeconds(frame / fps)}`,
      width - 176,
      height - 18,
    );
  }, [
    width,
    height,
    fps,
    frame,
    items,
    zones,
    scaleX,
    scaleY,
    simulated,
    highlightEntity,
  ]);

  useEffect(() => {
    draw();
  }, [draw]);

  // ----- playback ---------------------------------------------------------

  useEffect(() => {
    if (!playing) return;
    // Advance on the clip's own frame rate; the overlay is per-frame data, so
    // playing faster than the samples exist would just drop them.
    const interval = window.setInterval(
      () =>
        onFrameChange(frame + 1 >= frameCount ? 0 : frame + 1),
      Math.max(40, 1000 / fps),
    );
    return () => window.clearInterval(interval);
  }, [playing, frame, frameCount, fps, onFrameChange]);

  // Keep a real <video> in step when the clip has footage.
  useEffect(() => {
    const video = videoRef.current;
    if (!video || simulated) return;
    const target = frame / fps;
    if (Math.abs(video.currentTime - target) > 0.5 / fps) {
      video.currentTime = target;
    }
  }, [frame, fps, simulated]);

  return (
    <div className="player">
      <div className="stage" style={{ aspectRatio: `${width} / ${height}` }}>
        {!simulated && (
          <video
            ref={videoRef}
            className="stage-video"
            src={`/media/${encodeURIComponent(clip)}`}
            muted
            playsInline
            preload="auto"
          />
        )}
        <canvas
          ref={canvasRef}
          className="stage-canvas"
          width={width}
          height={height}
        />
      </div>

      <div className="controls">
        <button
          type="button"
          className="button"
          onClick={() => setPlaying((value) => !value)}
        >
          {playing ? "Pause" : "Play"}
        </button>
        <input
          className="scrub"
          type="range"
          min={0}
          max={Math.max(0, frameCount - 1)}
          value={frame}
          onChange={(event) => {
            setPlaying(false);
            onFrameChange(Number(event.target.value));
          }}
        />
        <span className="frame-readout">
          {frame} / {frameCount - 1}
        </span>
      </div>

      <p className="caption">
        {items.length} object{items.length === 1 ? "" : "s"} segmented in this
        frame
        {simulated && " · simulated clip: the scene is drawn from its masks"}
      </p>
    </div>
  );
}

/** Reconstruct a run-length mask and stretch it over the box it covers. */
function drawMask(
  ctx: CanvasRenderingContext2D,
  mask: { rows: number; cols: number; runs: number[] },
  left: number,
  top: number,
  right: number,
  bottom: number,
  colour: string,
  scratch: React.RefObject<HTMLCanvasElement | null>,
) {
  if (!scratch.current) scratch.current = document.createElement("canvas");
  const offscreen = scratch.current;
  offscreen.width = mask.cols;
  offscreen.height = mask.rows;
  const octx = offscreen.getContext("2d");
  if (!octx) return;

  const image = octx.createImageData(mask.cols, mask.rows);
  const rgb = hexToRgb(colour);
  let index = 0;
  let value = 0; // runs always begin with a run of zeros
  for (const run of mask.runs) {
    for (let i = 0; i < run; i += 1) {
      if (value === 1) {
        const offset = index * 4;
        image.data[offset] = rgb[0];
        image.data[offset + 1] = rgb[1];
        image.data[offset + 2] = rgb[2];
        image.data[offset + 3] = 150;
      }
      index += 1;
    }
    value = value === 0 ? 1 : 0;
  }
  octx.putImageData(image, 0, 0);

  // Nearest-neighbour, matching how the pipeline places the mask.
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(offscreen, left, top, right - left, bottom - top);
  ctx.imageSmoothingEnabled = true;
}

function hexToRgb(hex: string): [number, number, number] {
  const value = Number.parseInt(hex.replace("#", ""), 16);
  return [(value >> 16) & 255, (value >> 8) & 255, value & 255];
}
