"use client";

import { entityColour, formatSeconds, type ZoneEvent } from "@/lib/api";

type Props = {
  events: ZoneEvent[];
  frame: number;
  onSeek: (frame: number) => void;
  onHighlight: (entityId: number | null) => void;
  highlightEntity: number | null;
};

/**
 * The events table: the whole clip as rows.
 *
 * A row is highlighted while the playhead is inside its frame range, which
 * is what makes the table legible as a timeline rather than a list. Clicking
 * one seeks to the frame the event began, and that round trip -- row to
 * video, video to row -- is the demo's proof that the table is derived from
 * the footage rather than asserted alongside it.
 */
export default function EventsTable({
  events,
  frame,
  onSeek,
  onHighlight,
  highlightEntity,
}: Props) {
  return (
    <section className="panel">
      <h2>
        Events <span className="count">{events.length}</span>
      </h2>
      <div className="table-scroll tall">
        <table>
          <thead>
            <tr>
              <th>entity</th>
              <th>class</th>
              <th>zone</th>
              <th>entered</th>
              <th>exited</th>
              <th>dwell</th>
              <th>peak</th>
              <th>frames</th>
            </tr>
          </thead>
          <tbody>
            {events.map((event) => {
              const live = frame >= event.frame_start && frame <= event.frame_end;
              const dimmed =
                highlightEntity !== null && highlightEntity !== event.entity_id;
              return (
                <tr
                  key={event.id}
                  className={[
                    "seekable",
                    live ? "live" : "",
                    dimmed ? "dimmed" : "",
                  ]
                    .filter(Boolean)
                    .join(" ")}
                  onClick={() => {
                    onSeek(event.frame_start);
                    onHighlight(event.entity_id);
                  }}
                >
                  <td>
                    <span
                      className="swatch"
                      style={{ background: entityColour(event.entity_id) }}
                    />
                    #{event.entity_id}
                  </td>
                  <td>{event.class}</td>
                  <td>{event.zone}</td>
                  <td>{formatSeconds(event.entered_at)}</td>
                  <td>
                    {formatSeconds(event.exited_at)}
                    {event.closed === 0 && <span className="badge open">open</span>}
                  </td>
                  <td>{event.dwell_s.toFixed(1)}s</td>
                  <td>{(event.peak_overlap * 100).toFixed(0)}%</td>
                  <td className="muted">
                    {event.frame_start}–{event.frame_end}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="caption">
        <strong>peak</strong> is the most of the entity&rsquo;s own pixels ever
        inside the zone. Entry needs more than 50%, exit fewer than 30%, so an
        entity standing on a boundary produces one visit rather than a flicker
        of them.
      </p>
    </section>
  );
}
