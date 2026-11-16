"""The real REST app, HTTP codec/toolkit and explicit packaged direct client."""
import asyncio
import builtins
import copy
import io
import json
from unittest.mock import patch
import xml.etree.ElementTree as ET

import pytest
from PIL import Image

from mcp.client import ClientError, DirectMCPClient, MCPClient, MCPToolKit, RemoteError
from mcp.client.config import DEFAULT_CONFIG, get_server_url, load_config, merge_configs, save_config
from mcp.server.app import create_app
from tests.contract_support import ASGISession, ONE_CELL, asgi_request, document, ok


def assert_png(value):
    assert value.startswith(b"\x89PNG\r\n\x1a\n")
    with Image.open(io.BytesIO(value)) as image:
        assert image.format == "PNG" and image.width > 1 and image.height > 1
        image.verify()


def assert_svg(value):
    assert ET.fromstring(value).tag == "{http://www.w3.org/2000/svg}svg"


def test_real_rest_native_envelope_objects_ids_history_and_content_types(tmp_path):
    async def scenario():
        app = create_app(output_root=tmp_path / "server", instance_id="synthetic-owner")
        transport = ASGISession(app)
        async with MCPClient(server_url="http://testserver", session=transport, config={}) as client:
            toolkit = MCPToolKit(client)
            ping = await client.ping()
            assert ping["status"] == "ok" and ping["instance_id"] == "synthetic-owner"
            native = {"params": {"name": "a", "generative_model": ONE_CELL}}
            status, envelope = document(await asgi_request(app, "POST", "/tools/create_agent", native))
            assert status == 200 and set(envelope) == {"result"} and envelope["result"]["id"] == "a"
            assert "result" not in envelope["result"]
            from pymdp.agent import Agent
            assert isinstance(app.state.interface.agents["a"], Agent)
            state = await client.call_tool("infer_states", {"agent_id": "a", "observation": [0, 0]})
            assert state["posterior_states"] == [[1.0]]
            environment = await client.create_environment("e", [1, 1], [])
            assert environment["id"] == "e"
            assert environment["reward_positions"] == []
            run = await toolkit.run_simulation("a", "e", num_steps=2)
            assert run["num_steps"] == run["requested_steps"] == 2
            assert run["simulation_id"] == run["history"]["simulation_id"]
            assert [step["next_state"] for step in run["timesteps"]] == [[0], [0]]
            session = await client.create_session("a", "e", "session")
            first = await client.run_session(session["id"], num_steps=2)
            second = await client.run_session(session["id"], num_steps=1)
            history = await client.visualize_session(session["id"], "json")
            assert first["simulation_id"] != second["simulation_id"]
            assert history["num_steps"] == 3
            assert [step["simulation_id"] for step in history["timesteps"]] == [
                first["simulation_id"], first["simulation_id"], second["simulation_id"]]
            assert_png(await client.visualize_agent("a", "png"))
            assert_svg(await client.visualize_simulation(run["simulation_id"], "svg"))
            assert_png(await client.visualize_session("session", "png"))
            assert_png(await client.analyze_free_energy("session", "png"))
            assert_svg(await client.plot_belief_dynamics("session", "svg"))
            measured = await client.analyze_free_energy("session")
            assert all(record["status"] == "measured" and record["free_energy"] == 0
                       for record in measured["records"])
            assert [record["simulation_id"] for record in measured["records"]] == history["simulation_ids"][:1] * 2 + history["simulation_ids"][1:]
            statistics = await client.plot_belief_dynamics("session")
            assert statistics["num_steps"] == 3 and statistics["factors"][0]["state_changes"] == 0
            local = tmp_path / "trusted-local" / "session.svg"
            assert await toolkit.visualize_session("session", str(local), "svg") == str(local)
            assert_svg(local.read_bytes())
            assert all(item.get("json") is None for item in transport.requests if item["method"] == "GET")
        assert transport.closed is False
    asyncio.run(scenario())


@pytest.mark.parametrize("path,payload,status", [
    ("/tools/absent", {"params": {}}, 404),
    ("/tools/get_agent", {"params": {"agent_id": "absent"}}, 404),
    ("/tools/create_agent", {"params": {"name": "bad", "generative_model": "{}"}}, 422),
    ("/tools/get_agent", {"agent_id": "absent"}, 422),
    ("/tools/run_simulation", {"params": {"agent_id": "a", "env_id": "e", "num_timesteps": 2}}, 422),
    ("/environments", {"name": "e", "type": "custom"}, 422),
    ("/environments", {"name": "e", "grid_size": [1, 1], "reward_positions": [], "config": {}}, 422),
])
def test_bad_native_shapes_and_unsupported_controls_are_structured_not_500(tmp_path, path, payload, status):
    async def scenario():
        app = create_app(output_root=tmp_path)
        actual, data = document(await asgi_request(app, "POST", path, payload))
        assert actual == status and set(data) == {"error"}
        assert data["error"]["code"] and data["error"]["message"]
    asyncio.run(scenario())


def test_session_idempotence_conflicts_missing_and_removed_capabilities(tmp_path):
    async def scenario():
        app = create_app(output_root=tmp_path)
        core = app.state.interface
        ok(core.create_agent("a", ONE_CELL))
        ok(core.create_grid_world_env("e", [1, 1], []))
        ok(core.create_grid_world_env("other", [1, 1], []))
        payload = {"agent_id": "a", "env_id": "e", "session_id": "s"}
        assert document(await asgi_request(app, "POST", "/sessions", payload))[0] == 200
        assert document(await asgi_request(app, "POST", "/sessions", payload))[0] == 200
        status, body = document(await asgi_request(app, "POST", "/sessions", {**payload, "env_id": "other"}))
        assert status == 409 and body["error"]["code"] == "conflict"
        for method, route in (("POST", "/sessions/missing/run"), ("DELETE", "/agents/a"),
                              ("PUT", "/environments/e"), ("POST", "/sessions/s/reset")):
            status, body = document(await asgi_request(app, method, route, {}))
            assert status in (404, 405) and set(body) == {"error"}
        bad, body = document(await asgi_request(app, "GET", "/agents/a/visualize", query={"format": "webp"}))
        assert bad == 422 and set(body) == {"error"}
        with patch.object(app.state.registry, "call", side_effect=RuntimeError("synthetic backend exception")):
            status, body = document(await asgi_request(app, "GET", "/agents/a"))
            assert status == 500 and body == {"error": {"code": "internal_error", "message": "internal server error"}}
    asyncio.run(scenario())


def test_real_client_errors_propagate_without_a_local_success_fallback(tmp_path):
    async def scenario():
        app = create_app(output_root=tmp_path)
        client = MCPClient(server_url="http://testserver", session=ASGISession(app), config={})
        with pytest.raises(RemoteError) as error:
            await client.get_agent("missing")
        assert error.value.status == 404 and error.value.payload["error"]["code"] == "not_found"
        with pytest.raises(RemoteError) as error:
            await client.create_agent("bad", {"A": [[[0]]], "B": [[[[1]]]]})
        assert error.value.status == 422
        with pytest.raises(ValueError):
            await client._make_request("GET", "/agents", data={"unsupported": True})
    asyncio.run(scenario())


def test_config_merge_disk_roundtrip_and_client_ownership(tmp_path):
    before = copy.deepcopy(DEFAULT_CONFIG)
    override = {"server": {"host": "::1", "port": 9999}, "auth": {"enabled": False, "token": "synthetic-token"}}
    merged = merge_configs(DEFAULT_CONFIG, override)
    merged["auth"]["token"] = "changed"
    assert override["auth"]["token"] == "synthetic-token" and DEFAULT_CONFIG == before
    path = tmp_path / "config.json"
    save_config(override, path)
    assert load_config(path)["auth"]["token"] == "synthetic-token"
    assert get_server_url(load_config(path)) == "http://[::1]:9999"
    first, second = MCPClient(config=override), MCPClient(config={})
    first.config["server"]["port"] = 1
    first.config["auth"]["enabled"] = True
    assert second.config == before and DEFAULT_CONFIG == before
    assert override["server"]["port"] == 9999 and override["auth"]["enabled"] is False
    path.write_text("{malformed")
    with pytest.raises(json.JSONDecodeError):
        load_config(path)
    for unsupported in ({"auth": {"username": "unused"}}, {"options": {"retry_attempts": 3}}):
        with pytest.raises(ValueError):
            MCPClient(config=unsupported)


def test_auth_enablement_and_request_headers_never_bleed_between_requests(tmp_path):
    async def scenario():
        app = create_app(output_root=tmp_path)
        disabled_transport, enabled_transport = ASGISession(app), ASGISession(app)
        disabled = MCPClient("http://testserver", session=disabled_transport,
                             config={"auth": {"enabled": False, "token": "synthetic-stored"}})
        enabled = MCPClient("http://testserver", session=enabled_transport,
                            config={"auth": {"enabled": True, "token": "synthetic-enabled"}})
        headers = {"Authorization": "caller-header", "X-Case": "first"}
        await disabled._make_request("GET", "/ping", headers=headers)
        await enabled._make_request("GET", "/ping", headers=headers)
        await disabled.ping()
        assert headers == {"Authorization": "caller-header", "X-Case": "first"}
        assert "Authorization" not in disabled_transport.requests[0]["headers"]
        assert enabled_transport.requests[0]["headers"]["Authorization"] == "Bearer synthetic-enabled"
        assert "Authorization" not in disabled_transport.requests[1]["headers"]
    asyncio.run(scenario())


def test_explicit_direct_client_works_without_source_main_or_http_fallback(tmp_path):
    original_import = builtins.__import__
    def without_main(name, *args, **kwargs):
        if name == "main":
            raise ModuleNotFoundError("installed packages do not include src/main.py")
        return original_import(name, *args, **kwargs)
    async def scenario():
        direct, independent = DirectMCPClient(), DirectMCPClient()
        with patch("builtins.__import__", side_effect=without_main), patch.object(MCPClient, "_make_request", side_effect=AssertionError("no HTTP fallback")):
            agent = await direct.create_agent("a", ONE_CELL)
            env = await MCPToolKit(direct).create_gridworld_environment("e", [1, 1], [])
            run = await direct.run_simulation(agent["id"], env["id"], num_steps=2)
            assert run["num_steps"] == 2
            assert_png(await direct.visualize_simulation(run["simulation_id"]))
            local = tmp_path / "trusted" / "agent.svg"
            await MCPToolKit(direct).visualize_agent_model("a", str(local), "svg")
            assert_svg(local.read_bytes())
            with pytest.raises(ClientError):
                await independent.call_tool("get_agent", {"agent_id": "a"})
            with pytest.raises(ClientError):
                await direct.call_tool("not-a-real-tool", {})
    asyncio.run(scenario())


def test_codec_rejects_non_json_constants_and_oversized_bodies_without_guessing_format():
    class Response:
        status = 200
        def __init__(self, data, content_type, repeats=1):
            self.data, self.headers, self.content, self.repeats = data, {"Content-Type": content_type}, self, repeats
        async def iter_chunked(self, size):
            for _ in range(self.repeats):
                yield self.data
    async def scenario():
        for value in (b"NaN", b"Infinity", b"{invalid}"):
            with pytest.raises(ClientError, match="invalid JSON"):
                await MCPClient.decode_response(Response(value, "application/json"))
        with pytest.raises(ClientError, match="byte budget"):
            await MCPClient.decode_response(Response(b"x" * 1024 * 1024, "image/png", 33))
        with pytest.raises(ClientError, match="Content-Type"):
            await MCPClient.decode_response(Response(b'{"result": {}}', "text/plain"))
    asyncio.run(scenario())


def test_toolkit_cannot_label_png_bytes_as_svg_in_a_trusted_local_destination(tmp_path):
    async def scenario():
        direct = DirectMCPClient()
        await direct.create_agent("a", ONE_CELL)
        wrong = tmp_path / "wrong.svg"
        with pytest.raises(ValueError, match="suffix"):
            await MCPToolKit(direct).visualize_agent_model("a", str(wrong))
        assert not wrong.exists()
    asyncio.run(scenario())
