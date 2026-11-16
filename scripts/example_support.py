"""Shared example CLI and real required-operation reporting, never fallbacks."""
import argparse
import asyncio
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mcp.client import MCPClient, MCPToolKit
from mcp.outputs import OutputPolicy


def write_json(directory, name, value):
    return OutputPolicy().write_json(Path(directory) / name, value)


async def grid_case(client, directory, label, shape, rewards, steps):
    toolkit = MCPToolKit(client)
    suffix = uuid.uuid4().hex[:12]
    agent = await toolkit.create_gridworld_agent(label + "Agent" + suffix, shape, rewards)
    environment = await toolkit.create_gridworld_environment(label + "Env" + suffix, shape, rewards)
    result = await toolkit.run_simulation(agent["id"], environment["id"], num_steps=steps)
    write_json(directory, "simulation.json", result)
    write_json(directory, "model.json", await client.call_tool("get_agent", {"agent_id": agent["id"]}))
    await toolkit.visualize_agent_model(agent["id"], str(Path(directory) / "model.png"))
    await toolkit.visualize_simulation(result["simulation_id"], str(Path(directory) / "simulation.png"))
    return {"agent_id": agent["id"], "env_id": environment["id"],
            "simulation_id": result["simulation_id"], "num_steps": result["num_steps"],
            "total_reward": result["total_reward"],
            "vfe_statuses": [entry["status"] for entry in result["free_energy_trace"]]}


async def evaluate_cases(client, cases, directory):
    """Attempt every independent case; failure stays failure in the exit summary."""
    cases = list(cases)
    if not cases or len({name for name, _ in cases}) != len(cases):
        raise ValueError("nonempty, uniquely named cases required")
    results = {}
    for name, operation in cases:
        destination = Path(directory) / name
        destination.mkdir(parents=True, exist_ok=False)
        try:
            result = await operation(client, destination)
        except Exception as error:
            results[name] = {"status": "failed", "error": str(error)}
        else:
            results[name] = {"status": "completed", "result": result}
    write_json(directory, "summary.json", {"cases": results})
    return 0 if all(result["status"] == "completed" for result in results.values()) else 1


async def run_remote(cases, *, host="127.0.0.1", port=8080, use_ssl=False, output_dir=None):
    base = Path(output_dir) if output_dir is not None else ROOT / "scripts" / "output"
    directory = base / ("run_" + uuid.uuid4().hex)
    directory.mkdir(parents=True, exist_ok=False)
    print(f"Outputs: {directory}")
    scheme = "https" if use_ssl else "http"
    host = f"[{host}]" if ":" in host and not host.startswith("[") else host
    try:
        async with MCPClient(server_url=f"{scheme}://{host}:{port}") as client:
            ping = await client.ping()
            if ping["status"] != "ok":
                raise RuntimeError("server did not report readiness")
            write_json(directory, "available_tools.json", await client.get_tools())
            return await evaluate_cases(client, cases, directory)
    except Exception as error:
        write_json(directory, "summary.json", {
            "error": str(error), "cases": {name: {"status": "not_run"} for name, _ in cases}})
        print(str(error), file=sys.stderr)
        return 1


def example_cli(description, cases, argv=None):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--use-ssl", action="store_true")
    parser.add_argument("--output-dir")
    options = parser.parse_args(argv)
    return asyncio.run(run_remote(cases, **vars(options)))
