"use client";

import { useState } from "react";
import { api, type AskResult, type CannedQuestion, type QueryResult } from "@/lib/api";

type Props = {
  naturalLanguage: boolean;
  canned: CannedQuestion[];
  onSeek: (frame: number) => void;
  onHighlight: (entityId: number | null) => void;
};

type State = {
  loading: boolean;
  sql: string | null;
  result: QueryResult | null;
  brief: string | null;
  error: string | null;
  rejected: boolean;
  attempts: number;
};

const EMPTY: State = {
  loading: false,
  sql: null,
  result: null,
  brief: null,
  error: null,
  rejected: false,
  attempts: 0,
};

/**
 * Ask a question, see the SQL, click a row.
 *
 * The generated SQL is always on screen, including when the query failed.
 * That is a deliberate product decision rather than a debug affordance: a
 * wrong answer whose SQL is visible is something a user can judge and
 * correct, and an answer with no SQL behind it is something they have to
 * take on faith. It also means a rejected query shows *why* it was rejected.
 */
export default function QueryPanel({
  naturalLanguage,
  canned,
  onSeek,
  onHighlight,
}: Props) {
  const [question, setQuestion] = useState("");
  const [state, setState] = useState<State>(EMPTY);

  async function runAsk(text: string) {
    if (!text.trim()) return;
    setState({ ...EMPTY, loading: true });
    try {
      const answer: AskResult = await api.ask(text);
      setState({
        loading: false,
        sql: answer.sql,
        result: answer.ok ? answer : null,
        brief: answer.brief,
        error: answer.ok ? null : answer.error,
        rejected: false,
        attempts: answer.attempts,
      });
    } catch (error) {
      setState({
        ...EMPTY,
        error: error instanceof Error ? error.message : String(error),
      });
    }
  }

  async function runSql(sql: string) {
    setState({ ...EMPTY, loading: true });
    try {
      const result = await api.sql(sql);
      setState({ ...EMPTY, sql: result.sql, result });
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setState({ ...EMPTY, sql, error: message, rejected: true });
    }
  }

  const frameColumn = state.result
    ? state.result.columns.findIndex((c) => c === "frame_start" || c === "person_frame")
    : -1;
  const entityColumn = state.result
    ? state.result.columns.findIndex((c) => c === "entity_id")
    : -1;

  return (
    <section className="panel">
      <h2>Ask</h2>

      {naturalLanguage ? (
        <form
          className="ask-form"
          onSubmit={(event) => {
            event.preventDefault();
            void runAsk(question);
          }}
        >
          <input
            className="ask-input"
            placeholder="which zone had the longest average dwell?"
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
          />
          <button className="button primary" type="submit" disabled={state.loading}>
            {state.loading ? "Asking..." : "Ask"}
          </button>
        </form>
      ) : (
        <p className="notice">
          Natural language is off because <code>MODEL_API_KEY</code> is not set.
          The reference questions below run against the same database and the
          same guard, so everything else is live.
        </p>
      )}

      <div className="chips">
        {canned.map((q) => (
          <button
            key={q.key}
            type="button"
            className="chip"
            title={q.note || q.question}
            onClick={() => {
              if (naturalLanguage) {
                setQuestion(q.question);
                void runAsk(q.question);
              } else {
                void runSql(q.sql);
              }
            }}
          >
            {q.question}
          </button>
        ))}
      </div>

      {state.sql && (
        <div className="sql-block">
          <div className="sql-head">
            <span>SQL{state.attempts > 1 ? ` (after ${state.attempts} attempts)` : ""}</span>
            {state.rejected && <span className="badge refused">refused by the guard</span>}
          </div>
          <pre>{state.sql}</pre>
        </div>
      )}

      {state.error && (
        <p className={state.rejected ? "error refused-text" : "error"}>{state.error}</p>
      )}

      {state.brief && (
        <div className="brief">
          <h3>Brief</h3>
          <p>{state.brief}</p>
        </div>
      )}

      {state.result && (
        <div className="result">
          <div className="result-head">
            {state.result.row_count} row{state.result.row_count === 1 ? "" : "s"}
            {state.result.truncated && " (row cap reached)"}
            {frameColumn >= 0 && (
              <span className="hint">click a row to seek the video</span>
            )}
          </div>
          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  {state.result.columns.map((column) => (
                    <th key={column}>{column}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {state.result.rows.map((row, index) => {
                  const frame = frameColumn >= 0 ? Number(row[frameColumn]) : null;
                  const entity =
                    entityColumn >= 0 ? Number(row[entityColumn]) : null;
                  const seekable = frame !== null && Number.isFinite(frame);
                  return (
                    <tr
                      key={index}
                      className={seekable ? "seekable" : undefined}
                      onClick={() => {
                        if (seekable) onSeek(frame);
                        onHighlight(entity !== null && Number.isFinite(entity) ? entity : null);
                      }}
                    >
                      {row.map((cell, cellIndex) => (
                        <td key={cellIndex}>{format(cell)}</td>
                      ))}
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </section>
  );
}

function format(value: unknown): string {
  if (value === null || value === undefined) return "";
  if (typeof value === "number") {
    return Number.isInteger(value) ? String(value) : value.toFixed(2);
  }
  return String(value);
}
