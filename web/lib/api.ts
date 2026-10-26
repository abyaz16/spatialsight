// Types mirroring the Python API. Kept hand-written and small: the server is
// six routes, and a code generator would be more machinery than the contract.

export type Clip = {
  clip: string;
  width: number;
  height: number;
  fps: number;
  frame_count: number;
  duration_s: number;
  backend: string | null;
  simulated: boolean;
  entities: number;
  events: number;
};

export type ZoneSet = {
  clip: string;
  camera: string;
  frame: { width: number; height: number };
  zones: { name: string; polygon: [number, number][] }[];
};

export type ZoneEvent = {
  id: number;
  clip: string;
  entity_id: number;
  class: string;
  zone: string;
  entered_at: number;
  exited_at: number;
  dwell_s: number;
  frame_start: number;
  frame_end: number;
  peak_overlap: number;
  closed: number;
};

export type MaskRle = { rows: number; cols: number; runs: number[] };

export type OverlayItem = {
  entity_id: number;
  class: string;
  box: [number, number, number, number];
  confidence: number | null;
  mask?: MaskRle;
};

export type Overlay = {
  clip: string;
  width: number;
  height: number;
  fps: number;
  frame_count: number;
  entities: { entity_id: number; class: string }[];
  frames: Record<string, OverlayItem[]>;
};

export type QueryResult = {
  sql: string;
  columns: string[];
  rows: unknown[][];
  row_count: number;
  truncated: boolean;
};

export type AskResult = QueryResult & {
  question: string;
  ok: boolean;
  attempts: number;
  brief: string | null;
  error: string | null;
};

export type Health = {
  ok: boolean;
  database: string;
  database_exists: boolean;
  clips: string[];
  spark_model: string;
  natural_language: boolean;
};

export type CannedQuestion = {
  key: string;
  question: string;
  sql: string;
  note: string;
};

async function get<T>(path: string): Promise<T> {
  const response = await fetch(path, { cache: "no-store" });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error ?? `HTTP ${response.status}`);
  return payload as T;
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error ?? `HTTP ${response.status}`);
  return payload as T;
}

export const api = {
  health: () => get<Health>("/api/health"),
  clips: () => get<{ clips: Clip[] }>("/api/clips"),
  zones: (clip: string) => get<ZoneSet>(`/api/zones?clip=${encodeURIComponent(clip)}`),
  events: (clip: string) =>
    get<{ events: ZoneEvent[]; count: number }>(
      `/api/events?clip=${encodeURIComponent(clip)}`,
    ),
  overlay: (clip: string) =>
    get<Overlay>(`/api/overlay?clip=${encodeURIComponent(clip)}`),
  canned: () => get<{ questions: CannedQuestion[] }>("/api/canned"),
  sql: (sql: string) => post<QueryResult>("/api/sql", { sql }),
  ask: (question: string) => post<AskResult>("/api/ask", { question }),
};

// One stable colour per entity, so the box on the video, the row in the table
// and the row in a SQL result all agree about who they are talking about.
const PALETTE = [
  "#4f9cf9",
  "#f2994a",
  "#27ae60",
  "#bb6bd9",
  "#eb5757",
  "#56ccf2",
  "#f2c94c",
  "#2d9cdb",
];

export function entityColour(entityId: number): string {
  return PALETTE[(entityId - 1 + PALETTE.length) % PALETTE.length];
}

export function formatSeconds(value: number): string {
  const minutes = Math.floor(value / 60);
  const seconds = value - minutes * 60;
  return `${minutes}:${seconds.toFixed(1).padStart(4, "0")}`;
}
