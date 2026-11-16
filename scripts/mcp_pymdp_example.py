#!/usr/bin/env python3
"""Native model and compatible grid demonstrations over canonical REST."""
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.example_support import example_cli, grid_case, write_json


async def random_model_case(client, directory):
    model = await client.call_tool("define_generative_model", {
        "A_dims": [[2, 3]], "B_dims": [[3, 3, 2]], "seed": 0})
    agent = await client.create_agent("RandomAgent" + uuid.uuid4().hex[:12], model)
    states = await client.call_tool("infer_states", {"agent_id": agent["id"], "observation": [0]})
    policies = await client.call_tool("infer_policies", {"agent_id": agent["id"]})
    selected = await client.call_tool("sample_action", {"agent_id": agent["id"]})
    write_json(directory, "model.json", await client.get_agent(agent["id"]))
    write_json(directory, "inference.json", {"states": states, "policies": policies, "selection": selected})
    return {"agent_id": agent["id"], "action_id": selected["action_id"]}


async def compatible_grid_case(client, directory):
    return await grid_case(client, directory, "BasicGrid", [3, 3], [[2, 2]], 10)


CASES = (("random_model", random_model_case), ("gridworld", compatible_grid_case))


def main(argv=None):
    return example_cli(__doc__, CASES, argv)


if __name__ == "__main__":
    raise SystemExit(main())
