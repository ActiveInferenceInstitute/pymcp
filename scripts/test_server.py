#!/usr/bin/env python3
"""Exercise real REST/session/byte contracts; required failure exits nonzero."""
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mcp.client import MCPToolKit
from scripts.example_support import example_cli, write_json


async def server_case(client, directory):
    toolkit = MCPToolKit(client)
    suffix = uuid.uuid4().hex[:12]
    model = {"A": [[[1]], [[1], [0]]], "B": [[[[1]]]], "C": [[0], [0, 0]], "D": [[1]]}
    agent = await client.create_agent("ProbeAgent" + suffix, model)
    env = await client.create_environment("ProbeEnv" + suffix, [1, 1], [])
    session = await client.create_session(agent["id"], env["id"], "ProbeSession" + suffix)
    first = await client.run_session(session["id"], num_steps=2)
    second = await client.run_session(session["id"], num_steps=1)
    history = await client.visualize_session(session["id"], format="json")
    if first["simulation_id"] == second["simulation_id"] or history["num_steps"] != 3:
        raise RuntimeError("session accumulation/identity contract failed")
    await toolkit.visualize_agent_model(agent["id"], str(Path(directory) / "model.png"))
    await toolkit.visualize_session(session["id"], str(Path(directory) / "session.svg"), format="svg")
    energy = await toolkit.analyze_free_energy(session["id"])
    beliefs = await toolkit.plot_belief_dynamics(session["id"])
    write_json(directory, "history.json", history)
    write_json(directory, "analysis.json", {"energy": energy, "beliefs": beliefs})
    return {"session_id": session["id"], "simulation_ids": [first["simulation_id"], second["simulation_id"]],
            "num_steps": history["num_steps"]}


def main(argv=None):
    return example_cli(__doc__, (("server_contract", server_case),), argv)


if __name__ == "__main__":
    raise SystemExit(main())
