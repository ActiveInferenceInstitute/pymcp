"""Actual bundled MCP lifecycle, stdio frames and isolated SSE correlation."""
import asyncio
import copy
import json
import os
import subprocess
import sys
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from mcp.api import ToolRegistry
from mcp.server.fastmcp import FastMCP, ProtocolSession
from mcp.utils import PyMDPInterface
from tests.contract_support import (
    INITIALIZE,
    INITIALIZED,
    ROOT,
    SSEConnection,
    document,
)


def make_server(**settings):
    interface = PyMDPInterface()
    context = SimpleNamespace(interface=interface, registry=ToolRegistry(interface))

    @asynccontextmanager
    async def lifespan(server):
        yield context

    return FastMCP("contract", "Actual production protocol", lifespan, **settings), context


def request(identity, method, params=None):
    result = {"jsonrpc": "2.0", "id": identity, "method": method}
    if params is not None:
        result["params"] = params
    return result


async def ready(server, context):
    session = ProtocolSession()
    initialized = await server.handle_request(INITIALIZE, context, session)
    assert initialized["result"]["protocolVersion"] == "2025-03-26"
    assert await server.handle_request(INITIALIZED, context, session) is None
    return session


def test_native_lifecycle_schema_notifications_domain_and_rpc_errors():
    async def scenario():
        server, context = make_server()
        session = ProtocolSession()
        before = await server.handle_request(request(0, "tools/list"), context, session)
        assert before["error"]["code"] == -32600
        await server.handle_request(INITIALIZE, context, session)
        pending = await server.handle_request(request(2, "tools/list"), context, session)
        assert pending["error"]["code"] == -32600
        await server.handle_request(INITIALIZED, context, session)
        listing = await server.handle_request(request(3, "tools/list"), context, session)
        tools = {item["name"]: item for item in listing["result"]["tools"]}
        assert {"create_agent", "commit_action", "get_session_history", "analyze_free_energy"} <= tools.keys()
        assert not {"_normalize_A_matrix", "set_debug_mode", "get_tool"} & tools.keys()
        assert tools["create_agent"]["inputSchema"]["properties"]["generative_model"]["type"] == "object"
        arguments = {"name": "native", "generative_model": {"A": [[[1]]], "B": [[[[1]]]]}}
        created = await server.handle_request(request(4, "tools/call", {"name": "create_agent", "arguments": arguments}), context, session)
        assert created["result"]["isError"] is False
        assert json.loads(created["result"]["content"][0]["text"])["id"] == "native"
        from pymdp.agent import Agent
        assert isinstance(context.interface.agents["native"], Agent)
        bad = await server.handle_request(request(5, "tools/call", {"name": "infer_states", "arguments": {"agent_id": "native", "observation": "0"}}), context, session)
        assert bad["error"]["code"] == -32602
        unknown = await server.handle_request(request(6, "tools/call", {"name": "not_a_tool"}), context, session)
        assert unknown["error"]["code"] == -32602
        missing = await server.handle_request(request(7, "tools/call", {"name": "get_agent", "arguments": {"agent_id": "missing"}}), context, session)
        assert missing["result"]["isError"] is True
        assert json.loads(missing["result"]["content"][0]["text"])["error_code"] == "not_found"
        assert (await server.handle_request(request(8, "not_a_method"), context, session))["error"]["code"] == -32601
        assert (await server.handle_request({"jsonrpc": "1.0", "id": 9, "method": "ping"}, context, session))["error"]["code"] == -32600
        notification = {"jsonrpc": "2.0", "method": "tools/call", "params": {"name": "create_agent", "arguments": {**arguments, "name": "notification-must-not-create"}}}
        assert await server.handle_request(notification, context, session) is None
        assert context.interface.get_agent("notification-must-not-create")["error_code"] == "not_found"
        with patch.object(server, "descriptors", side_effect=RuntimeError("synthetic router failure")):
            assert (await server.handle_request(request(10, "tools/list"), context, session))["error"]["code"] == -32603
    asyncio.run(scenario())


def test_standard_metadata_id_reuse_and_bounded_session_requests():
    async def scenario():
        server, context = make_server(max_requests=4)
        session = await ready(server, context)
        frame = request(2, "tools/list", {"_meta": {"progressToken": "caller-progress"}})
        before = copy.deepcopy(frame)
        assert "result" in await server.handle_request(frame, context, session)
        assert frame == before
        assert "result" in await server.handle_request(request(2, "ping"), context, session)
        batch = [INITIALIZED, request(3, "ping")]
        assert await server.handle_request(batch, context, session) == [{"jsonrpc": "2.0", "id": 3, "result": {}}]
        assert (await server.handle_request(request(4, "ping"), context, session))["error"]["code"] == -32600
    asyncio.run(scenario())


def test_stdio_is_real_jsonrpc_only_not_an_eof_smoke(tmp_path):
    frames = [INITIALIZE, INITIALIZED, request(2, "tools/list"),
              request(3, "tools/call", {"name": "create_agent", "arguments": {
                  "name": "wire-agent", "generative_model": {"A": [[[1]]], "B": [[[[1]]]]}}}),
              request(4, "tools/call", {"name": "get_agent", "arguments": {"agent_id": "wire-agent"}}),
              request(5, "tools/call", {"name": "get_agent", "arguments": {"agent_id": "absent"}}),
              request(6, "not_a_method")]
    oversized = json.dumps(request("must-not-execute", "ping", {"_meta": {"payload": "x" * (1024 * 1024)}})).encode()
    payload = (b"\n".join(json.dumps(frame).encode() for frame in frames) + b"\n{bad-json}\n"
               + oversized + b"\n\xff\n" + json.dumps(request(7, "ping")).encode() + b"\n")
    environment = dict(os.environ, PYTHONPATH=str(ROOT / "src"), MCP_OUTPUT_DIR=str(tmp_path / "server"))
    environment.pop("MCP_LIMITS", None)
    process = subprocess.Popen([sys.executable, "-B", str(ROOT / "src" / "main.py"), "--transport", "stdio"],
                               cwd=tmp_path, env=environment, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        output, diagnostics = process.communicate(payload, timeout=90)
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=10)
    assert process.returncode == 0, diagnostics.decode(errors="replace")
    responses = [json.loads(line) for line in output.splitlines()]
    assert [item["id"] for item in responses] == [1, 2, 3, 4, 5, 6, None, None, None, 7]
    assert responses[0]["result"]["protocolVersion"] == "2025-03-26"
    assert any(tool["name"] == "create_agent" for tool in responses[1]["result"]["tools"])
    assert json.loads(responses[2]["result"]["content"][0]["text"])["id"] == "wire-agent"
    assert json.loads(responses[3]["result"]["content"][0]["text"])["A"] == [[[1.0]]]
    assert responses[4]["result"]["isError"] is True
    assert responses[5]["error"]["code"] == -32601
    assert responses[6]["error"]["code"] == -32700
    assert responses[7]["error"]["code"] in (-32600, -32700)
    assert responses[8]["error"]["code"] == -32700
    assert responses[9]["result"] == {}


def test_sse_bodyless_channels_are_correlated_once_and_cleaned_on_disconnect():
    async def scenario():
        server, _ = make_server()
        app = server.create_sse_app()
        async with app.router.lifespan_context(app):
            first, second = await SSEConnection(app).open(), await SSEConnection(app).open()
            try:
                assert first.identity != second.identity
                status, _, body = await first.post(INITIALIZE)
                assert status == 202 and body == b""
                assert (await first.response())["id"] == 1
                await first.post(INITIALIZED)
                await second.post(request(7, "tools/list"))
                assert (await second.response())["error"]["code"] == -32600
                await second.post(INITIALIZE)
                assert (await second.response())["id"] == 1
                await second.post(INITIALIZED)
                calls = [request(8, "tools/call", {"name": "infer_states_from_observation", "arguments": {
                    "A_matrix": [[0.9, 0.1], [0.1, 0.9]], "prior": [0.5, 0.5], "observation": [outcome]}})
                    for outcome in (0, 1)]
                posted = await asyncio.gather(first.post(calls[0]), second.post(calls[1]))
                assert all(status == 202 and body == b"" for status, _, body in posted)
                left, right = await asyncio.gather(first.response(), second.response())
                assert left["id"] == right["id"] == 8
                assert json.loads(left["result"]["content"][0]["text"])["posterior"][0] == pytest.approx(0.9)
                assert json.loads(right["result"]["content"][0]["text"])["posterior"][0] == pytest.approx(0.1)
                with pytest.raises(asyncio.TimeoutError):
                    await asyncio.wait_for(first.event(), 0.05)
                with pytest.raises(asyncio.TimeoutError):
                    await asyncio.wait_for(second.event(), 0.05)
            finally:
                await first.close()
                await second.close()
            assert document(await first.post(request(9, "ping")))[0] == 404
            assert app.state.channels == {}
    asyncio.run(scenario())


def test_sse_budgets_reject_before_dispatch_and_release_on_disconnect():
    from tests.contract_support import asgi_request
    async def scenario():
        server, _ = make_server(max_channels=1, max_message_bytes=256, max_pending_responses=2)
        app = server.create_sse_app()
        async with app.router.lifespan_context(app):
            connection = SSEConnection(app)
            connection.outgoing = asyncio.Queue(maxsize=1)  # A real blocked ASGI send.
            await connection.open()
            try:
                assert document(await asgi_request(app, "GET", "/sse"))[0] == 503
                assert (await connection.post(INITIALIZE))[0] == 202
                assert "result" in await connection.response()
                await connection.post(INITIALIZED)
                oversized = await asgi_request(app, "POST", "/messages", raw=b"x" * 257,
                                               query={"session_id": connection.identity})
                assert document(oversized)[0] == 413
                for identity in range(2, 14):
                    status, _, _ = await connection.post(request(identity, "ping"))
                    if status == 429:
                        break
                    assert status == 202
                    await asyncio.sleep(0)
                else:
                    pytest.fail("blocked SSE consumer did not meet its explicit two-response budget")
            finally:
                await connection.close()
            replacement = await SSEConnection(app).open()
            await replacement.close()
    asyncio.run(scenario())
