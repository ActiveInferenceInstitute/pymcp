"""Native tool conveniences; HTTP images are downloaded, never local path claims."""
from pathlib import Path

from ..outputs import OutputPolicy


class MCPToolKit:
    def __init__(self, client):
        self.client = client

    async def create_agent_from_model(self, name, A, B, C, D=None, **settings):
        model = {"A": A, "B": B, "C": C, **settings}
        if D is not None:
            model["D"] = D
        return await self.client.call_tool("create_agent", {"name": name, "generative_model": model})

    async def create_gridworld_agent(self, name, grid_size, reward_positions, *,
                                     policy_len=1, inference_horizon=5, action_precision=1.0):
        return await self.client.call_tool("create_gridworld_agent", {
            "name": name, "grid_size": grid_size, "reward_positions": reward_positions,
            "policy_len": policy_len, "inference_horizon": inference_horizon,
            "action_precision": action_precision})

    async def create_gridworld_environment(self, name, grid_size, reward_positions, **settings):
        return await self.client.call_tool("create_grid_world_env", {
            "name": name, "grid_size": grid_size, "reward_positions": reward_positions, **settings})

    async def run_simulation(self, agent_id, env_id, num_steps=10, save_history=True, stop_on_done=False):
        return await self.client.call_tool("run_simulation", {
            "agent_id": agent_id, "env_id": env_id, "num_steps": num_steps,
            "save_history": save_history, "stop_on_done": stop_on_done})

    @staticmethod
    def _download(value, output_file, format):
        if output_file is None:
            return value
        if not isinstance(value, bytes):
            raise ValueError("local image output requires actual response bytes")
        if format not in ("png", "svg") or Path(output_file).suffix.lower() != "." + format:
            raise ValueError("local output suffix must match the requested PNG/SVG format")
        authority = OutputPolicy()  # Explicit client-local trusted destination.
        target = authority.path(Path(output_file))
        with authority.writer(target) as stream:
            stream.write(value)
        return str(target)

    async def visualize_agent_model(self, agent_id, output_file=None, format="png"):
        return self._download(await self.client.visualize_agent(agent_id, format), output_file, format)

    async def visualize_simulation(self, simulation_id, output_file=None, format="png"):
        return self._download(await self.client.visualize_simulation(simulation_id, format), output_file, format)

    async def visualize_session(self, session_id, output_file=None, format="png"):
        return self._download(await self.client.visualize_session(session_id, format), output_file, format)

    async def analyze_free_energy(self, session_id, output_file=None, format=None):
        format = ("png" if output_file is not None else "json") if format is None else format
        return self._download(await self.client.analyze_free_energy(session_id, format), output_file, format)

    async def plot_belief_dynamics(self, session_id, output_file=None, format=None):
        format = ("png" if output_file is not None else "json") if format is None else format
        return self._download(await self.client.plot_belief_dynamics(session_id, format), output_file, format)

    async def validate_generative_model(self, generative_model, check_normalization=True):
        return await self.client.call_tool("validate_generative_model", {
            "generative_model": generative_model, "check_normalization": check_normalization})
