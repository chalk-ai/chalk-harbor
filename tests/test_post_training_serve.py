import itertools
import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from chalk_harbor.post_training import trainer


class _Replicas:
    """Two vLLM replicas behind a round-robin balancer; the second sees new adapters late."""

    def __init__(self, lagging_requests: int) -> None:
        super().__init__()
        self.lagging_requests = lagging_requests
        self.turn = itertools.count()
        self.loads: list[dict[str, Any]] = []
        self.chat_requests = 0


def _handler(state: _Replicas) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path == "/v1/load_lora_adapter":
                state.loads.append(body)
                self._send(200, "ok")
                return
            state.chat_requests += 1
            replica = next(state.turn) % 2
            if replica == 1 and state.chat_requests <= state.lagging_requests:
                self._send(
                    404, json.dumps({"error": f"model {body['model']} does not exist"})
                )
            else:
                self._send(
                    200, json.dumps({"choices": [{"message": {"content": "h"}}]})
                )

        def _send(self, status: int, text: str) -> None:
            data = text.encode()
            self.send_response(status)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, format: str, *args: Any) -> None:
            pass

    return Handler


@pytest.fixture
def replicas(request: pytest.FixtureRequest) -> Iterator[tuple[str, _Replicas]]:
    state = _Replicas(lagging_requests=request.param)
    server = ThreadingHTTPServer(("127.0.0.1", 0), _handler(state))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}", state
    server.shutdown()


@pytest.mark.parametrize("replicas", [0], indirect=True)
def test_loads_then_confirms_every_replica_serves(
    replicas: tuple[str, _Replicas], monkeypatch: pytest.MonkeyPatch
) -> None:
    url, state = replicas
    monkeypatch.setattr(trainer.time, "sleep", lambda _: None)
    trainer.load_lora_adapter(url, "adapter-x-1", "/chalk/adapters/adapter-x-1", 30.0)
    assert state.loads == [
        {
            "lora_name": "adapter-x-1",
            "lora_path": "/chalk/adapters/adapter-x-1",
            "load_inplace": True,
        }
    ]
    assert state.chat_requests == trainer._SERVE_PROBE_STREAK


@pytest.mark.parametrize("replicas", [6], indirect=True)
def test_waits_for_a_lagging_replica(
    replicas: tuple[str, _Replicas], monkeypatch: pytest.MonkeyPatch
) -> None:
    url, state = replicas
    monkeypatch.setattr(trainer.time, "sleep", lambda _: None)
    trainer.load_lora_adapter(url, "adapter-x-1", "/chalk/adapters/adapter-x-1", 30.0)
    # The lagging replica's 404s reset the streak, so success needs a full streak after them.
    assert state.chat_requests >= 6 + trainer._SERVE_PROBE_STREAK


@pytest.mark.parametrize("replicas", [10_000], indirect=True)
def test_gives_up_at_the_deadline(
    replicas: tuple[str, _Replicas], monkeypatch: pytest.MonkeyPatch
) -> None:
    url, _ = replicas
    clock = itertools.count(0, 10)
    monkeypatch.setattr(trainer.time, "monotonic", lambda: float(next(clock)))
    monkeypatch.setattr(trainer.time, "sleep", lambda _: None)
    with pytest.raises(RuntimeError, match="does not serve adapter-x-1"):
        trainer.load_lora_adapter(
            url, "adapter-x-1", "/chalk/adapters/adapter-x-1", 120.0
        )
