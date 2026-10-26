"""A stand-in Meta Model API, speaking the documented protocol over real HTTP.

The SAM 3.1 client is the one part of this system that cannot be exercised by
fixtures: fixtures prove the *parser* works, and prove nothing about the
request we send, the stream we read, or what happens when the far end
misbehaves. Those are exactly the things that fail on the morning of a demo.

So this serves the real endpoints -- `POST /v1/files` and a streaming
`POST /v1/responses` -- over a real socket, and the production client talks
to it through the real OpenAI SDK. The only thing not exercised is Meta's
own server.

It can also be told to misbehave, which is the point. A vendor integration
that has only ever been tested against the happy path is untested:

* `rate_limit` -- 429s before succeeding, which is what a burst of requests
  gets on any metered API.
* `server_error` -- 500s, which retry logic must distinguish from a 400.
* `truncate` -- the stream dies mid-response. This is the dangerous one: the
  bytes received are *valid*, just incomplete, so nothing downstream can tell
  that half the video is missing.
* `refusal` -- the model declines, which is a successful HTTP request.
* `empty` -- a concept with no matches, which the docs call a normal result
  and which must not be mistaken for a failure.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


@dataclass
class FakeApiState:
    """What the server should do, and a record of what it was asked."""

    deltas: list[str] = field(default_factory=list)
    fail_mode: str | None = None
    fail_times: int = 0
    truncate_after: int = 1
    requests: list[dict] = field(default_factory=list)
    uploads: list[dict] = field(default_factory=list)
    _failures_served: int = 0

    def should_fail(self) -> bool:
        if not self.fail_mode or self._failures_served >= self.fail_times:
            return False
        self._failures_served += 1
        return True

    @property
    def response_count(self) -> int:
        return len(self.requests)


def _sse(event_type: str, payload: dict) -> bytes:
    return f"event: {event_type}\ndata: {json.dumps(payload)}\n\n".encode()


class _Handler(BaseHTTPRequestHandler):
    state: FakeApiState

    def log_message(self, *args) -> None:  # keep the test output readable
        pass

    def _body(self) -> bytes:
        length = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(length) if length else b""

    def _json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802
        if self.path.endswith("/files"):
            return self._files()
        if self.path.endswith("/responses"):
            return self._responses()
        self._json({"error": {"message": "not found"}}, 404)

    def _files(self) -> None:
        raw = self._body()
        self.state.uploads.append(
            {
                "bytes": len(raw),
                "authorization": self.headers.get("Authorization"),
                "purpose": b'name="purpose"' in raw or b"user_data" in raw,
            }
        )
        self._json({"id": "file-stub-001", "object": "file", "purpose": "user_data"})

    def _responses(self) -> None:
        raw = self._body()
        try:
            request = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            request = {}
        self.state.requests.append(
            {
                "body": request,
                "authorization": self.headers.get("Authorization"),
            }
        )

        if self.state.should_fail():
            mode = self.state.fail_mode
            if mode == "rate_limit":
                self.send_response(429)
                self.send_header("Content-Type", "application/json")
                self.send_header("Retry-After", "0")
                body = json.dumps({"error": {"message": "rate limit exceeded"}}).encode()
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if mode == "server_error":
                return self._json({"error": {"message": "upstream failure"}}, 500)
            if mode == "bad_request":
                return self._json(
                    {"error": {"message": "invalid prompt or metadata"}}, 400
                )
            if mode == "model_not_found":
                return self._json(
                    {"error": {"message": "model not found", "code": "model_not_found"}},
                    404,
                )

        self._stream(request)

    def _stream(self, request: dict) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        response = {
            "id": "resp-stub-001",
            "object": "response",
            "model": request.get("model", "sam-3.1"),
            "status": "in_progress",
            "output": [],
        }
        lane = {"item_id": "msg-001", "output_index": 0, "content_index": 0}

        self.wfile.write(_sse("response.created", {"type": "response.created", "response": response}))
        self.wfile.write(
            _sse("response.in_progress", {"type": "response.in_progress", "response": response})
        )
        self.wfile.write(
            _sse(
                "response.output_item.added",
                {
                    "type": "response.output_item.added",
                    "output_index": 0,
                    "item": {"id": "msg-001", "type": "message", "role": "assistant", "content": [], "status": "in_progress"},
                },
            )
        )
        self.wfile.write(
            _sse(
                "response.content_part.added",
                {
                    "type": "response.content_part.added",
                    **lane,
                    "part": {"type": "output_text", "text": "", "annotations": []},
                },
            )
        )

        mode = self.state.fail_mode
        if mode == "refusal":
            text = "I cannot segment this."
            self.wfile.write(
                _sse(
                    "response.refusal.delta",
                    {"type": "response.refusal.delta", **lane, "delta": text},
                )
            )
            self.wfile.write(
                _sse(
                    "response.refusal.done",
                    {"type": "response.refusal.done", **lane, "refusal": text},
                )
            )
            # A refusal still closes its content part and output item; the SDK
            # rejects a response completed with items left open.
            self.wfile.write(
                _sse(
                    "response.content_part.done",
                    {"type": "response.content_part.done", **lane, "part": {"type": "refusal", "refusal": text}},
                )
            )
            item = {
                "id": "msg-001",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "refusal", "refusal": text}],
            }
            self.wfile.write(
                _sse(
                    "response.output_item.done",
                    {"type": "response.output_item.done", "output_index": 0, "item": item},
                )
            )
            self.wfile.write(
                _sse(
                    "response.completed",
                    {"type": "response.completed", "response": {**response, "status": "completed", "output": [item]}},
                )
            )
            self.wfile.flush()
            return

        deltas = [] if mode == "empty" else list(self.state.deltas)
        sent = 0
        for delta in deltas:
            if mode == "truncate" and sent >= self.state.truncate_after:
                # Stop writing and hang up. The bytes already sent are valid,
                # which is precisely what makes this failure dangerous.
                self.wfile.flush()
                try:
                    self.connection.close()
                except OSError:
                    pass
                return
            self.wfile.write(
                _sse("response.output_text.delta", {"type": "response.output_text.delta", **lane, "delta": delta})
            )
            sent += 1
        self.wfile.flush()

        text = "".join(deltas)
        self.wfile.write(
            _sse("response.output_text.done", {"type": "response.output_text.done", **lane, "text": text})
        )
        self.wfile.write(
            _sse(
                "response.content_part.done",
                {"type": "response.content_part.done", **lane, "part": {"type": "output_text", "text": text, "annotations": []}},
            )
        )
        self.wfile.write(
            _sse(
                "response.output_item.done",
                {
                    "type": "response.output_item.done",
                    "output_index": 0,
                    "item": {
                        "id": "msg-001",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": text, "annotations": []}],
                    },
                },
            )
        )
        completed = {
            **response,
            "status": "completed",
            "output": [
                {
                    "id": "msg-001",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": text, "annotations": []}],
                }
            ],
            "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        }
        self.wfile.write(_sse("response.completed", {"type": "response.completed", "response": completed}))
        self.wfile.flush()


class FakeMetaApi:
    """Runs the stub on a free port for the lifetime of a `with` block."""

    def __init__(self, state: FakeApiState) -> None:
        self.state = state
        handler = type("BoundHandler", (_Handler,), {"state": state})
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        host, port = self.server.server_address[:2]
        return f"http://{host}:{port}/v1"

    def __enter__(self) -> "FakeMetaApi":
        self.thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
