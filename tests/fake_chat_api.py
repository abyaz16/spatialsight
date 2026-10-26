"""A stub chat-completions endpoint, for testing the text-model client.

The same argument as `fake_meta_api`: the question-to-SQL path reaches a
vendor over HTTP, and the interesting failures are the ones that happen to
that call rather than to the SQL. A rate limit during a demo should cost a
second, not the answer.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


@dataclass
class FakeChatState:
    replies: list[str] = field(default_factory=list)
    fail_mode: str | None = None
    fail_times: int = 0
    requests: list[dict] = field(default_factory=list)
    _failures_served: int = 0
    _reply_index: int = 0

    def should_fail(self) -> bool:
        if not self.fail_mode or self._failures_served >= self.fail_times:
            return False
        self._failures_served += 1
        return True

    def next_reply(self) -> str:
        if not self.replies:
            return ""
        reply = self.replies[min(self._reply_index, len(self.replies) - 1)]
        self._reply_index += 1
        return reply


class _Handler(BaseHTTPRequestHandler):
    state: FakeChatState

    def log_message(self, *args) -> None:
        pass

    def _json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        if status == 429:
            self.send_header("Retry-After", "0")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            request = json.loads(raw)
        except json.JSONDecodeError:
            request = {}
        self.state.requests.append(
            {"body": request, "authorization": self.headers.get("Authorization")}
        )

        if self.state.should_fail():
            mode = self.state.fail_mode
            status = {"rate_limit": 429, "server_error": 500, "bad_request": 400}.get(mode, 500)
            return self._json({"error": {"message": f"stub {mode}"}}, status)

        content = self.state.next_reply()
        self._json(
            {
                "id": "chatcmpl-stub",
                "object": "chat.completion",
                "model": request.get("model", "stub-model"),
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 42, "completion_tokens": 7, "total_tokens": 49},
            }
        )


class FakeChatApi:
    def __init__(self, state: FakeChatState) -> None:
        self.state = state
        handler = type("BoundChatHandler", (_Handler,), {"state": state})
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        host, port = self.server.server_address[:2]
        return f"http://{host}:{port}/v1"

    def __enter__(self) -> "FakeChatApi":
        self.thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
