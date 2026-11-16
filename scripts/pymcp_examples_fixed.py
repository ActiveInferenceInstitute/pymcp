#!/usr/bin/env python3
"""Four real demonstrations: agent, grid, exact Bayes/VFE and explicit commits."""
from pathlib import Path
import sys
import uuid

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.example_support import example_cli, grid_case, write_json
from mcp.client import MCPToolKit
from mcp.outputs import OutputPolicy


async def agent_demo(client, directory):
    B = np.zeros((3, 3, 3))
    for state in range(3):
        for action, offset in enumerate((0, 1, -1)):
            B[(state + offset) % 3, state, action] = 1
    A = [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]]
    model = {"A": [[[0.8, 0.1], [0.1, 0.2], [0.1, 0.7]], A, A],
             "B": [np.eye(2)[:, :, None].tolist(), B.tolist()],
             "C": [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
             "D": [[0.5, 0.5], [1 / 3, 1 / 3, 1 / 3]], "A_dependencies": [[0], [1], [1]]}
    agent = await client.create_agent("AgentDemo" + uuid.uuid4().hex[:12], model)
    result = await client.call_tool("infer_states", {"agent_id": agent["id"], "observation": [2, 2, 0]})
    write_json(directory, "agent.json", await client.get_agent(agent["id"]))
    write_json(directory, "inference.json", result)
    return {"agent_id": agent["id"], "posterior_states": result["posterior_states"]}


async def gridworld_tutorial(client, directory):
    return await grid_case(client, directory, "Tutorial", [3, 3], [[2, 2]], 10)


async def free_energy_calculation(client, directory):
    A = [[0.9, 0.1, 0.3], [0.1, 0.9, 0.7]]
    prior, observation = [1 / 3, 1 / 3, 1 / 3], [1]
    bayes = await client.call_tool("infer_states_from_observation", {
        "A_matrix": A, "prior": prior, "observation": observation})
    components = await client.call_tool("calculate_free_energy", {
        "A_matrix": A, "prior": prior, "observation": observation, "q_states": bayes["posterior"]})
    result = {"A_matrix": A, "prior": prior, "observation": observation,
              "posterior": bayes["posterior"], "evidence": bayes["evidence"], "vfe": components}
    write_json(directory, "bayes_vfe.json", result)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    try:
        x = np.arange(len(prior))
        ax.bar(x - 0.15, prior, 0.3, label="Prior")
        ax.bar(x + 0.15, bayes["posterior"], 0.3, label="Actual posterior")
        ax.set(xlabel="State", ylabel="Probability")
        ax.legend()
        OutputPolicy().save_figure(fig, Path(directory) / "belief_update.png")
    finally:
        plt.close(fig)
    return result


async def agent_loop(client, directory):
    toolkit = MCPToolKit(client)
    suffix = uuid.uuid4().hex[:12]
    agent = await toolkit.create_gridworld_agent("LoopAgent" + suffix, [3, 3], [[2, 2]],
                                                action_precision=2.0, inference_horizon=3)
    env = await toolkit.create_gridworld_environment("LoopEnv" + suffix, [3, 3], [[2, 2]])
    current = await client.call_tool("reset_environment", {"env_id": env["id"]})
    await client.call_tool("reset_agent", {"agent_id": agent["id"]})
    transitions = []
    for step in range(10):
        state = await client.call_tool("infer_states", {"agent_id": agent["id"], "observation": current["observation"]})
        policy = await client.call_tool("infer_policies", {"agent_id": agent["id"]})
        selected = await client.call_tool("sample_action", {"agent_id": agent["id"]})
        following = await client.call_tool("step_environment", {
            "env_id": env["id"], "agent_id": agent["id"], "action": selected["action"],
            "action_id": selected["action_id"]})
        transitions.append({"step": step, "input": current, "inference": state, "policy": policy,
                            "selection": selected, "transition": following})
        current = following
    # This explicit operation sequence is not falsely assigned a registered run ID.
    result = {"mode": "manual", "agent_id": agent["id"], "env_id": env["id"],
              "stop_on_done": False, "transitions": transitions}
    write_json(directory, "manual_transitions.json", result)
    return {"agent_id": agent["id"], "env_id": env["id"], "completed_transitions": len(transitions)}


CASES = (("agent_demo", agent_demo), ("gridworld_tutorial", gridworld_tutorial),
         ("free_energy_calculation", free_energy_calculation), ("agent_loop", agent_loop))


def main(argv=None):
    return example_cli(__doc__, CASES, argv)


if __name__ == "__main__":
    raise SystemExit(main())
