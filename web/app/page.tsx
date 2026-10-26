"use client";

import { useCallback, useEffect, useState } from "react";
import EventsTable from "@/components/EventsTable";
import Player from "@/components/Player";
import QueryPanel from "@/components/QueryPanel";
import {
  api,
  type CannedQuestion,
  type Clip,
  type Health,
  type Overlay,
  type ZoneEvent,
  type ZoneSet,
} from "@/lib/api";

export default function Page() {
  const [health, setHealth] = useState<Health | null>(null);
  const [clips, setClips] = useState<Clip[]>([]);
  const [clip, setClip] = useState<string | null>(null);
  const [zones, setZones] = useState<ZoneSet | null>(null);
  const [overlay, setOverlay] = useState<Overlay | null>(null);
  const [events, setEvents] = useState<ZoneEvent[]>([]);
  const [canned, setCanned] = useState<CannedQuestion[]>([]);
  const [frame, setFrame] = useState(0);
  const [highlight, setHighlight] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    (async () => {
      try {
        const [h, c, q] = await Promise.all([
          api.health(),
          api.clips(),
          api.canned(),
        ]);
        setHealth(h);
        setClips(c.clips);
        setCanned(q.questions);
        if (c.clips.length > 0) setClip(c.clips[0].clip);
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      }
    })();
  }, []);

  useEffect(() => {
    if (!clip) return;
    setFrame(0);
    setHighlight(null);
    (async () => {
      try {
        const [z, o, ev] = await Promise.all([
          api.zones(clip),
          api.overlay(clip),
          api.events(clip),
        ]);
        setZones(z);
        setOverlay(o);
        setEvents(ev.events);
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      }
    })();
  }, [clip]);

  const onSeek = useCallback((target: number) => setFrame(target), []);
  const current = clips.find((c) => c.clip === clip) ?? null;

  if (error) {
    return (
      <main className="shell">
        <h1>SpatialSight</h1>
        <div className="panel">
          <p className="error">{error}</p>
          <p className="notice">
            Start the API and build a clip first:
            <br />
            <code>./scripts/serve</code>
            <br />
            <code>./scripts/ss fixture &amp;&amp; ./scripts/ss ingest synthetic-01</code>
          </p>
        </div>
      </main>
    );
  }

  return (
    <main className="shell">
      <header className="masthead">
        <div>
          <h1>SpatialSight</h1>
          <p className="tagline">
            Pixel-level video understanding compiled into a relational schema
            you can query in plain English.
          </p>
        </div>
        <div className="meta">
          {clips.length > 1 && (
            <select
              className="select"
              value={clip ?? ""}
              onChange={(event) => setClip(event.target.value)}
            >
              {clips.map((c) => (
                <option key={c.clip} value={c.clip}>
                  {c.clip}
                </option>
              ))}
            </select>
          )}
          {current && (
            <dl className="stats">
              <div>
                <dt>entities</dt>
                <dd>{current.entities}</dd>
              </div>
              <div>
                <dt>events</dt>
                <dd>{current.events}</dd>
              </div>
              <div>
                <dt>duration</dt>
                <dd>{current.duration_s.toFixed(0)}s</dd>
              </div>
              <div>
                <dt>NL&nbsp;&rarr;&nbsp;SQL</dt>
                <dd>{health?.natural_language ? health.spark_model : "off"}</dd>
              </div>
            </dl>
          )}
        </div>
      </header>

      <div className="columns">
        <div className="left">
          {overlay && zones && current ? (
            <Player
              clip={current.clip}
              overlay={overlay}
              zones={zones}
              simulated={current.simulated}
              frame={frame}
              onFrameChange={onSeek}
              highlightEntity={highlight}
            />
          ) : (
            <div className="panel">
              <p className="notice">Loading clip…</p>
            </div>
          )}
          <EventsTable
            events={events}
            frame={frame}
            onSeek={onSeek}
            onHighlight={setHighlight}
            highlightEntity={highlight}
          />
        </div>

        <div className="right">
          <QueryPanel
            naturalLanguage={Boolean(health?.natural_language)}
            canned={canned}
            onSeek={onSeek}
            onHighlight={setHighlight}
          />
        </div>
      </div>

      <footer className="footer">
        <span>
          {current?.simulated
            ? "Simulated clip with exact ground truth — the deterministic layers are measured against it."
            : "Real footage segmented with SAM 3.1."}
        </span>
        {highlight !== null && (
          <button type="button" className="button" onClick={() => setHighlight(null)}>
            Clear entity #{highlight}
          </button>
        )}
      </footer>
    </main>
  );
}
