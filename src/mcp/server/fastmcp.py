"""Bundled MCP JSON-RPC implementation (2025-03-26, stdio or HTTP+SSE)."""
import asyncio
import inspect
import json
import logging
import os
import sys
import uuid
from contextlib import asynccontextmanager, redirect_stdout
from dataclasses import dataclass, field

from ..api import InvalidArguments, Tool, UnknownTool
from ..utils import NumpyEncoder

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8050
DEFAULT_TRANSPORT = "sse"
PROTOCOL_VERSION = "2025-03-26"
logger = logging.getLogger(__name__)


@dataclass
class Context:
    request_context: object


class RequestContext:
    def __init__(self, lifespan_context):
        self.lifespan_context = lifespan_context


@dataclass
class ProtocolSession:
    phase: str = "new"
    request_count: int = 0


@dataclass
class Channel:
    session: ProtocolSession = field(default_factory=ProtocolSession)
    responses: asyncio.Queue = field(default_factory=lambda: asyncio.Queue(maxsize=128))
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


def encode(value):
    return json.dumps(value, cls=NumpyEncoder, allow_nan=False, separators=(",", ":"))


def parse(value):
    def invalid_constant(value):
        raise ValueError(f"non-JSON constant {value}")
    return json.loads(value, parse_constant=invalid_constant)


def rpc_error(identity, code, message):
    return {"jsonrpc": "2.0", "id": identity, "error": {"code": code, "message": message}}


class FastMCP:
    def __init__(self, name, description, lifespan, host=None, port=None, transport=None,
                 max_message_bytes=1024 * 1024, max_channels=128, max_requests=10000,
                 max_pending_responses=128):
        self.name, self.description, self.lifespan = name, description, lifespan
        self.host = host or os.environ.get("HOST", DEFAULT_HOST)
        self.port = int(port if port is not None else os.environ.get("PORT", DEFAULT_PORT))
        if not 1 <= self.port <= 65535:
            raise ValueError("port is outside 1..65535")
        self.transport = transport or os.environ.get("TRANSPORT", DEFAULT_TRANSPORT)
        self.max_message_bytes, self.max_channels, self.max_requests = max_message_bytes, max_channels, max_requests
        self.max_pending_responses = max_pending_responses
        if any(type(n) is not int or n <= 0 for n in (max_message_bytes, max_channels, max_requests, max_pending_responses)):
            raise ValueError("protocol budgets must be positive integers")
        self.tools = {}
        self._execution_lock = asyncio.Lock()

    def tool(self, name=None, description=None):
        def register(function):
            tool = Tool(name or function.__name__, function, description)
            if tool.name in self.tools:
                raise ValueError("duplicate tool")
            self.tools[tool.name] = tool
            return function
        return register

    def _registry(self, context):
        return getattr(context, "registry", None)

    def descriptors(self, context):
        registry = self._registry(context)
        return registry.descriptors() if registry is not None else [t.descriptor() for t in self.tools.values()]

    async def handle_request(self, value, lifespan_context, session):
        if isinstance(value, list):
            if not value or len(value) > 64:
                return rpc_error(None, -32600, "invalid or oversized batch")
            responses = []
            for item in value:
                response = await self._safe_request(item, lifespan_context, session)
                if response is not None:
                    responses.append(response)
            return responses or None
        return await self._safe_request(value, lifespan_context, session)

    async def _safe_request(self, value, context, session):
        try:
            return await self._request(value, context, session)
        except Exception:
            logger.exception("protocol dispatch failed")
            if isinstance(value, dict) and "id" not in value and isinstance(value.get("method"), str):
                return None
            identity = value.get("id") if isinstance(value, dict) else None
            return rpc_error(identity if type(identity) in (int, str) else None, -32603, "internal error")

    async def _request(self, value, context, session):
        if not isinstance(value, dict) or value.get("jsonrpc") != "2.0" or not isinstance(value.get("method"), str):
            return rpc_error(None, -32600, "invalid JSON-RPC request")
        notification = "id" not in value
        identity = value.get("id")
        if not notification and (type(identity) not in (int, str)):
            return rpc_error(None, -32600, "request id must be a string or integer")
        if set(value) - {"jsonrpc", "id", "method", "params"}:
            return None if notification else rpc_error(identity, -32600, "unknown request fields")
        params = value.get("params", {})
        if not isinstance(params, dict):
            return None if notification else rpc_error(identity, -32602, "params must be an object")
        params = dict(params)
        if "_meta" in params:
            metadata = params.pop("_meta")
            if not isinstance(metadata, dict):
                return None if notification else rpc_error(identity, -32602, "_meta must be an object")
            if "progressToken" in metadata and type(metadata["progressToken"]) not in (str, int):
                return None if notification else rpc_error(identity, -32602, "invalid progress token")
        method = value["method"]
        if notification:
            if method == "notifications/initialized" and not params and session.phase == "initializing":
                session.phase = "ready"
            # No notification receives a response, nor can it execute a tool.
            return None
        if session.request_count >= self.max_requests:
            return rpc_error(identity, -32600, "session request budget exhausted")
        session.request_count += 1
        if method == "initialize":
            info = params.get("clientInfo")
            if (session.phase != "new" or not isinstance(params.get("protocolVersion"), str)
                    or not isinstance(params.get("capabilities"), dict)
                    or not isinstance(info, dict)
                    or any(not isinstance(info.get(k), str) or not info[k] for k in ("name", "version"))):
                return rpc_error(identity, -32602, "invalid initialization or already initialized")
            session.phase = "initializing"
            return {"jsonrpc": "2.0", "id": identity, "result": {
                "protocolVersion": PROTOCOL_VERSION, "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": self.name, "version": "1.0.3"},
            }}
        if method == "ping":
            if params:
                return rpc_error(identity, -32602, "ping takes no parameters")
            return {"jsonrpc": "2.0", "id": identity, "result": {}}
        if method not in ("tools/list", "tools/call"):
            return rpc_error(identity, -32601, "method not found")
        if session.phase != "ready":
            return rpc_error(identity, -32600, "initialize and initialized notification are required")
        if method == "tools/list":
            if params:
                return rpc_error(identity, -32602, "pagination is not supported")
            return {"jsonrpc": "2.0", "id": identity, "result": {"tools": self.descriptors(context)}}
        if set(params) - {"name", "arguments"} or not isinstance(params.get("name"), str):
            return rpc_error(identity, -32602, "tools/call requires name and native arguments")
        try:
            async with self._execution_lock:
                registry = self._registry(context)
                if registry is not None:
                    result = registry.call(params["name"], params.get("arguments", {}))
                else:
                    if params["name"] not in self.tools:
                        raise UnknownTool(params["name"])
                    result = self.tools[params["name"]].call(
                        params.get("arguments", {}), Context(RequestContext(context)))
                    if inspect.isawaitable(result):
                        result = await result
                text = encode(result)
        except (InvalidArguments, UnknownTool) as error:
            return rpc_error(identity, -32602, str(error))
        except Exception as error:
            logger.exception("tool execution failed")
            result = {"error": str(error), "error_code": "operation_failed"}
            text = encode(result)
        return {"jsonrpc": "2.0", "id": identity, "result": {
            "content": [{"type": "text", "text": text}],
            "isError": isinstance(result, dict) and "error" in result,
        }}

    async def run_stdio(self):
        output, input_stream = sys.stdout, getattr(sys.stdin, "buffer", sys.stdin)
        session = ProtocolSession()
        # Dependency/tool diagnostics cannot contaminate the protocol stream.
        with redirect_stdout(sys.stderr):
            async with self.lifespan(self) as context:
                while True:
                    line = input_stream.readline(self.max_message_bytes + 2)
                    if not line:
                        break
                    size = len(line) if isinstance(line, bytes) else len(line.encode("utf-8", errors="surrogatepass"))
                    if size > self.max_message_bytes:
                        newline = b"\n" if isinstance(line, bytes) else "\n"
                        while line and not line.endswith(newline):
                            line = input_stream.readline(self.max_message_bytes + 2)
                        response = rpc_error(None, -32600, "message exceeds byte budget")
                    else:
                        try:
                            value = parse(line)
                        except (ValueError, UnicodeError):
                            response = rpc_error(None, -32700, "parse error")
                        else:
                            response = await self.handle_request(value, context, session)
                    if response is not None:
                        output.write(encode(response) + "\n")
                        output.flush()

    def create_sse_app(self):
        from fastapi import FastAPI, HTTPException, Request
        from fastapi.responses import Response, StreamingResponse

        @asynccontextmanager
        async def lifespan(app):
            async with self.lifespan(self) as context:
                app.state.context = context
                app.state.channels = {}
                try:
                    yield
                finally:
                    app.state.channels.clear()

        app = FastAPI(title=self.name, lifespan=lifespan)

        @app.get("/sse")
        async def connect(request: Request):
            if len(app.state.channels) >= self.max_channels:
                raise HTTPException(503, "channel budget exhausted")
            identity = uuid.uuid4().hex
            channel = Channel(responses=asyncio.Queue(maxsize=self.max_pending_responses))
            app.state.channels[identity] = channel

            async def events():
                yield f"event: endpoint\ndata: /messages?session_id={identity}\n\n"
                while not await request.is_disconnected():
                    try:
                        response = await asyncio.wait_for(channel.responses.get(), timeout=15)
                    except asyncio.TimeoutError:
                        yield ": heartbeat\n\n"
                    else:
                        yield f"event: message\ndata: {encode(response)}\n\n"

            class ChannelResponse(StreamingResponse):
                async def __call__(self, scope, receive, send):
                    try:
                        await super().__call__(scope, receive, send)
                    finally:
                        # Disconnect can cancel send while the iterator is suspended
                        # at yield; its own finally would not run until it is closed.
                        app.state.channels.pop(identity, None)
                        await self.body_iterator.aclose()

            return ChannelResponse(events(), media_type="text/event-stream",
                                   headers={"Cache-Control": "no-cache"})

        @app.post("/messages")
        async def message(request: Request, session_id: str):
            channel = app.state.channels.get(session_id)
            if channel is None:
                raise HTTPException(404, "unknown or expired channel")
            chunks, size = [], 0
            async for chunk in request.stream():
                size += len(chunk)
                if size > self.max_message_bytes:
                    raise HTTPException(413, "message exceeds byte budget")
                chunks.append(chunk)
            async with channel.lock:
                if app.state.channels.get(session_id) is not channel:
                    raise HTTPException(404, "expired channel")
                if channel.responses.full():
                    raise HTTPException(429, "response queue is full")
                try:
                    value = parse(b"".join(chunks))
                except (ValueError, UnicodeError):
                    response = rpc_error(None, -32700, "parse error")
                else:
                    response = await self.handle_request(value, app.state.context, channel.session)
                if response is not None:
                    channel.responses.put_nowait(response)
            # JSON-RPC response is delivered once, only on this SSE channel.
            return Response(status_code=202)

        return app

    async def run_sse(self):
        import uvicorn
        await uvicorn.Server(uvicorn.Config(self.create_sse_app(), host=self.host,
                                           port=self.port, workers=1)).serve()

    async def run(self):
        if self.transport == "stdio":
            await self.run_stdio()
        elif self.transport == "sse":
            await self.run_sse()
        else:
            raise ValueError("transport must be stdio or sse")
