"""Synthetic models and a dependency-free ASGI transport, not fake app results."""
import asyncio
import json
from pathlib import Path
from urllib.parse import urlencode, urlsplit

import numpy as np

from mcp.environments import CustomEnvironment
from mcp.utils import PyMDPInterface

ROOT = Path(__file__).resolve().parents[1]
ONE_CELL = {"A": [[[1]], [[1], [0]]], "B": [[[[1]]]], "C": [[0], [0, 0]], "D": [[1]]}


def ok(result):
    assert "error" not in result, result
    return result


def one_cell(interface=None, **kwargs):
    interface = PyMDPInterface(**kwargs) if interface is None else interface
    ok(interface.create_agent("a", ONE_CELL))
    ok(interface.create_grid_world_env("e", [1, 1], []))
    return interface


def chain():
    interface = PyMDPInterface()
    B = np.zeros((3, 3, 1))
    B[1, 0, 0] = B[2, 1, 0] = B[2, 2, 0] = 1
    model = {"A": [np.eye(3).tolist()], "B": [B.tolist()], "C": [[0, 0, 0]], "D": [[1, 0, 0]]}
    ok(interface.create_agent("chain", model))
    env = CustomEnvironment("chain-env", ["s0", "s1", "s2"], ["s0", "s1", "s2"],
                            {"s0": {"next": "s1"}, "s1": {"next": "s2"}, "s2": {"next": "s2"}}, {})
    interface.environments["chain-env"] = env
    result = ok(interface.run_simulation("chain", "chain-env", 2))
    return interface, result


def rectangular(**settings):
    from mcp.testing.fixtures import create_gridworld_generative_model
    interface = PyMDPInterface(**settings)
    model = create_gridworld_generative_model([2, 3], [1, 2])
    model["C"] = [[0, 1, 2, 0, 0, 3], [0, 0]]
    ok(interface.create_agent("rectangle", model))
    ok(interface.create_grid_world_env("rectangle-env", [2, 3], [[1, 2]]))
    return interface, ok(interface.run_simulation("rectangle", "rectangle-env", 3))


def http_scope(method, path, query=None, body=b"", headers=None):
    values = [(b"content-length", str(len(body)).encode())]
    if body:
        values.append((b"content-type", b"application/json"))
    values.extend((key.lower().encode(), value.encode()) for key, value in (headers or {}).items())
    return {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1", "method": method, "scheme": "http", "root_path": "",
            "path": path, "raw_path": path.encode(), "query_string": urlencode(query or {}).encode(),
            "headers": values, "client": ("127.0.0.1", 12345), "server": ("testserver", 8080)}


async def asgi_request(app, method, path, value=None, *, query=None, raw=None, headers=None):
    data = raw if raw is not None else (json.dumps(value, allow_nan=False).encode() if value is not None else b"")
    sent, received = [], False

    async def receive():
        nonlocal received
        if not received:
            received = True
            return {"type": "http.request", "body": data, "more_body": False}
        await asyncio.Future()

    async def send(message):
        sent.append(message)

    try:
        await app(http_scope(method, path, query, data, headers), receive, send)
    except Exception:
        # Starlette sends the registered 500 response, then re-raises to its HTTP
        # server. Observe that actual response without concealing unsent failures.
        if not any(message["type"] == "http.response.start" and message["status"] == 500 for message in sent):
            raise
    start = next(message for message in sent if message["type"] == "http.response.start")
    body = b"".join(message.get("body", b"") for message in sent if message["type"] == "http.response.body")
    return start["status"], {key.decode().title(): value.decode() for key, value in start["headers"]}, body


def document(response):
    status, headers, data = response
    assert headers["Content-Type"].startswith("application/json"), response
    return status, json.loads(data)


class ASGISession:
    """Exercise MCPClient's real request/codec against the real production ASGI app."""
    def __init__(self, app):
        self.app, self.closed, self.requests = app, False, []

    def request(self, **options):
        self.requests.append(options)
        return ASGIResponse(self, options)

    async def close(self):
        self.closed = True


class ASGIResponse:
    def __init__(self, session, options):
        self.session, self.options = session, options
        self.content = self

    async def __aenter__(self):
        parts = urlsplit(self.options["url"])
        self.status, self.headers, self.data = await asgi_request(
            self.session.app, self.options["method"], parts.path, self.options.get("json"),
            query=self.options.get("params"), headers=self.options.get("headers"))
        return self

    async def __aexit__(self, *error):
        return False

    async def iter_chunked(self, size):
        for offset in range(0, len(self.data), size):
            yield self.data[offset:offset + size]


class SSEConnection:
    """Drive a real bodyless ASGI GET and collect its actual SSE response frames."""
    def __init__(self, app):
        self.app = app
        self.incoming, self.outgoing = asyncio.Queue(), asyncio.Queue()
        self.buffer = ""

    async def open(self):
        await self.incoming.put({"type": "http.request", "body": b"", "more_body": False})
        self.task = asyncio.create_task(self.app(http_scope("GET", "/sse"), self.incoming.get, self.outgoing.put))
        event, endpoint = await self.event()
        assert event == "endpoint", (event, endpoint)
        parts = urlsplit(endpoint)
        self.identity = parts.query.split("=", 1)[1]
        return self

    async def event(self):
        while "\n\n" not in self.buffer:
            message = await asyncio.wait_for(self.outgoing.get(), timeout=10)
            if message["type"] == "http.response.start":
                assert message["status"] == 200, message
            elif message["type"] == "http.response.body":
                self.buffer += message.get("body", b"").decode()
        block, self.buffer = self.buffer.split("\n\n", 1)
        rows = dict(line.split(": ", 1) for line in block.splitlines() if ": " in line)
        return rows.get("event"), rows.get("data")

    async def post(self, frame):
        return await asgi_request(self.app, "POST", "/messages", frame, query={"session_id": self.identity})

    async def response(self):
        event, data = await self.event()
        assert event == "message", (event, data)
        return json.loads(data)

    async def close(self):
        await self.incoming.put({"type": "http.disconnect"})
        await asyncio.wait_for(self.task, timeout=10)


INITIALIZE = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
    "protocolVersion": "2025-03-26", "capabilities": {},
    "clientInfo": {"name": "synthetic-contract", "version": "1"}}}
INITIALIZED = {"jsonrpc": "2.0", "method": "notifications/initialized"}
