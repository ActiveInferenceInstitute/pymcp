"""Explicit HTTP client: one tool unwrap, real resource bytes, no local fallback."""
import copy
import json
from urllib.parse import quote, urlsplit

import aiohttp

from ..utils import NumpyEncoder
from .config import DEFAULT_CONFIG, get_server_url, load_config, merge_configs, validate_config


class ClientError(RuntimeError):
    pass


class RemoteError(ClientError):
    def __init__(self, status, payload):
        self.status, self.payload = status, payload
        super().__init__(f"HTTP {status}: {payload}")


def _reject_constant(value):
    raise ValueError("non-JSON numeric constant: " + value)


class MCPClient:
    def __init__(self, server_url=None, auth_token=None, session=None, config=None):
        self.config = load_config() if config is None else validate_config(merge_configs(DEFAULT_CONFIG, config))
        self.server_url = (server_url if server_url is not None else get_server_url(self.config)).rstrip("/")
        parsed = urlsplit(self.server_url)
        if (parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username
                or parsed.password or parsed.query or parsed.fragment):
            raise ValueError("server_url must be an explicit HTTP(S) endpoint without embedded credentials")
        parsed.port  # Validate a supplied port without contacting the server.
        self.auth_token = auth_token if auth_token is not None else self.config["auth"]["token"]
        if self.auth_token is not None and not isinstance(self.auth_token, str):
            raise ValueError("auth_token must be a string")
        self.session = session
        self._owns_session = session is None

    async def _ensure_session(self):
        if self.session is None:
            self.session = aiohttp.ClientSession()
            self._owns_session = True
        elif self.session.closed:
            raise ClientError("the supplied HTTP session is closed")

    async def __aenter__(self):
        await self._ensure_session()
        return self

    async def close(self):
        if self._owns_session and self.session is not None:
            await self.session.close()
            self.session = None

    async def __aexit__(self, *exception):
        await self.close()

    def _headers(self, supplied=None):
        headers = {key: value for key, value in dict(supplied or {}).items() if key.lower() != "authorization"}
        if self.config["auth"]["enabled"] and self.auth_token:
            headers["Authorization"] = "Bearer " + self.auth_token
        return headers

    @staticmethod
    async def decode_response(response):
        chunks, total = [], 0
        async for chunk in response.content.iter_chunked(65536):
            total += len(chunk)
            if total > 32 * 1024 * 1024:
                raise ClientError("response exceeds the byte budget")
            chunks.append(chunk)
        data = b"".join(chunks)
        kind = response.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
        if response.status >= 400:
            try:
                payload = json.loads(data, parse_constant=_reject_constant)
            except (ValueError, UnicodeError):
                payload = data.decode("utf-8", errors="replace")
            raise RemoteError(response.status, payload)
        if kind == "application/json" or kind.endswith("+json"):
            try:
                value = json.loads(data, parse_constant=_reject_constant)
            except (ValueError, UnicodeError) as error:
                raise ClientError("invalid JSON response") from error
            if isinstance(value, dict) and "error" in value:
                raise RemoteError(response.status, value)
            return value
        if kind in ("image/png", "image/svg+xml"):
            return data
        raise ClientError("unsupported response Content-Type: " + kind)

    async def _make_request(self, method, endpoint, data=None, params=None, headers=None):
        if method == "GET" and data is not None:
            raise ValueError("GET bodies are unsupported")
        await self._ensure_session()
        options = {
            "method": method, "url": self.server_url + endpoint,
            "params": copy.deepcopy(params), "headers": self._headers(headers),
            "timeout": aiohttp.ClientTimeout(total=self.config["options"]["timeout"]),
        }
        if data is not None:
            # Convert array-like caller values at the HTTP boundary only.
            options["json"] = json.loads(json.dumps(data, cls=NumpyEncoder, allow_nan=False))
        async with self.session.request(**options) as response:
            return await self.decode_response(response)

    async def ping(self):
        return await self._make_request("GET", "/ping")

    async def get_tools(self):
        return (await self._make_request("GET", "/tools"))["tools"]

    async def call_tool(self, tool_id, params):
        response = await self._make_request("POST", "/tools/" + quote(tool_id, safe=""), {"params": params})
        if not isinstance(response, dict) or set(response) != {"result"} or not isinstance(response["result"], dict):
            raise ClientError("invalid canonical tool response envelope")
        result = response["result"]
        if "error" in result:
            raise RemoteError(200, result)
        return result

    async def create_agent(self, name, generative_model, normalize=False):
        return await self.call_tool("create_agent", {
            "name": name, "generative_model": generative_model, "normalize": normalize})

    async def get_agents(self):
        return (await self._make_request("GET", "/agents"))["agents"]

    async def get_agent(self, agent_id):
        return await self._make_request("GET", "/agents/" + quote(agent_id, safe=""))

    async def create_environment(self, name, grid_size, reward_positions, *,
                                  start_position=None, reset_mode="origin", terminate_on_reward=True):
        return await self._make_request("POST", "/environments", {
            "name": name, "type": "grid_world", "grid_size": grid_size,
            "reward_positions": reward_positions, "start_position": start_position,
            "reset_mode": reset_mode, "terminate_on_reward": terminate_on_reward})

    async def get_environments(self):
        return (await self._make_request("GET", "/environments"))["environments"]

    async def get_environment(self, env_id):
        return await self._make_request("GET", "/environments/" + quote(env_id, safe=""))

    async def create_session(self, agent_id, env_id, session_id):
        return await self._make_request("POST", "/sessions", {
            "agent_id": agent_id, "env_id": env_id, "session_id": session_id})

    async def get_sessions(self):
        return (await self._make_request("GET", "/sessions"))["sessions"]

    async def run_session(self, session_id, num_steps=10, stop_on_done=False):
        return await self._make_request("POST", "/sessions/" + quote(session_id, safe="") + "/run",
                                        {"num_steps": num_steps, "stop_on_done": stop_on_done})

    async def run_simulation(self, agent_id, env_id, num_steps=10, save_history=True, stop_on_done=False):
        return await self.call_tool("run_simulation", {
            "agent_id": agent_id, "env_id": env_id, "num_steps": num_steps,
            "save_history": save_history, "stop_on_done": stop_on_done})

    async def visualize_agent(self, agent_id, format="png"):
        return await self._make_request("GET", "/agents/" + quote(agent_id, safe="") + "/visualize",
                                        params={"format": format})

    async def visualize_simulation(self, simulation_id, format="png"):
        return await self._make_request("GET", "/simulations/" + quote(simulation_id, safe="") + "/visualize",
                                        params={"format": format})

    async def visualize_session(self, session_id, format="png"):
        return await self._make_request("GET", "/sessions/" + quote(session_id, safe="") + "/visualize",
                                        params={"format": format})

    async def analyze_free_energy(self, session_id, format="json"):
        return await self._make_request("GET", "/analyze/free_energy/" + quote(session_id, safe=""),
                                        params={"format": format})

    async def plot_belief_dynamics(self, session_id, format="json"):
        return await self._make_request("GET", "/analyze/beliefs/" + quote(session_id, safe=""),
                                        params={"format": format})
