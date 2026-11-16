"""Explicit in-process client. No HTTP probing, source-main import or fallback."""
from pathlib import Path
import tempfile

from ..api import InvalidArguments, ToolRegistry, UnknownTool
from ..utils import PyMDPInterface
from .core import ClientError


class DirectMCPClient:
    def __init__(self, interface=None, *, limits=None, output_root=None):
        if interface is not None and (limits is not None or output_root is not None):
            raise ValueError("an injected interface retains its own limits/output custody")
        self.interface = interface if interface is not None else PyMDPInterface(limits, output_root)
        if not isinstance(self.interface, PyMDPInterface):
            raise ValueError("interface must be a real PyMDPInterface")
        self.registry = ToolRegistry(self.interface)

    async def call_tool(self, tool_id, params):
        try:
            result = self.registry.call(tool_id, params)
        except (InvalidArguments, UnknownTool) as error:
            raise ClientError(str(error)) from error
        if isinstance(result, dict) and "error" in result:
            raise ClientError(f"{result.get('error_code', 'operation_failed')}: {result['error']}")
        return result

    async def get_tools(self):
        return self.registry.descriptors()

    async def create_agent(self, name, generative_model, normalize=False):
        return await self.call_tool("create_agent", {
            "name": name, "generative_model": generative_model, "normalize": normalize})

    async def create_session(self, agent_id, env_id, session_id):
        return await self.call_tool("create_session", {
            "agent_id": agent_id, "env_id": env_id, "session_id": session_id})

    async def run_session(self, session_id, num_steps=10, stop_on_done=False):
        return await self.call_tool("run_session", {
            "session_id": session_id, "num_steps": num_steps, "stop_on_done": stop_on_done})

    async def run_simulation(self, agent_id, env_id, num_steps=10, save_history=True, stop_on_done=False):
        return await self.call_tool("run_simulation", {
            "agent_id": agent_id, "env_id": env_id, "num_steps": num_steps,
            "save_history": save_history, "stop_on_done": stop_on_done})

    async def _image(self, tool_name, identity_key, identity, format):
        if format not in ("png", "svg"):
            raise ValueError("image format must be png or svg")
        root = self.interface.output_policy.root
        if root is not None:
            root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="pymcp-direct-", dir=root) as directory:
            target = str(Path(directory) / ("figure." + format))
            result = await self.call_tool(tool_name, {identity_key: identity, "output_file": target})
            return self.interface.output_policy.read(result["figure_path"])

    async def visualize_agent(self, agent_id, format="png"):
        if format == "json":
            return await self.call_tool("get_agent", {"agent_id": agent_id})
        return await self._image("visualize_agent_model", "agent_id", agent_id, format)

    async def visualize_simulation(self, simulation_id, format="png"):
        if format == "json":
            return await self.call_tool("get_simulation", {"simulation_id": simulation_id})
        return await self._image("visualize_simulation", "simulation_id", simulation_id, format)

    async def visualize_session(self, session_id, format="png"):
        from ..visualization.simulation import visualize_simulation
        history = await self.call_tool("get_session_history", {"session_id": session_id})
        if format == "json":
            return history
        if format not in ("png", "svg"):
            raise ValueError("image format must be png or svg")
        root = self.interface.output_policy.root
        if root is not None:
            root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="pymcp-direct-", dir=root) as directory:
            target = str(Path(directory) / ("figure." + format))
            with self.registry.lock:
                result = visualize_simulation(history, target, policy=self.interface.output_policy)
            return self.interface.output_policy.read(result["figure_path"])

    async def analyze_free_energy(self, session_id, format="json"):
        if format == "json":
            from ..visualization.free_energy import measured_records
            history = await self.call_tool("get_session_history", {"session_id": session_id})
            return {"session_id": session_id, "records": measured_records(history)}
        return await self._image("analyze_free_energy", "session_id", session_id, format)

    async def plot_belief_dynamics(self, session_id, format="json"):
        if format == "json":
            from ..belief_dynamics import calculate_belief_statistics
            history = await self.call_tool("get_session_history", {"session_id": session_id})
            return calculate_belief_statistics([step["beliefs"] for step in history["timesteps"]])
        return await self._image("plot_belief_dynamics", "session_id", session_id, format)
