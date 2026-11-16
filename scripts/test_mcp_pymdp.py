#!/usr/bin/env python3
"""Explicit local-core demonstration; no HTTP fallback or fabricated history."""
import argparse
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mcp.outputs import OutputPolicy
from mcp.utils import PyMDPInterface


def require(result):
    if "error" in result:
        raise RuntimeError(result["error"])
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parent / "output"))
    options = parser.parse_args(argv)
    output = Path(options.output_dir) / ("direct_" + uuid.uuid4().hex)
    output.mkdir(parents=True, exist_ok=False)
    interface = PyMDPInterface()
    try:
        agent = require(interface.create_gridworld_agent("LocalAgent", [3, 3], [[2, 2]]))
        env = require(interface.create_grid_world_env("LocalEnv", [3, 3], [[2, 2]]))
        result = require(interface.run_simulation(agent["id"], env["id"], num_steps=10))
        OutputPolicy().write_json(output / "simulation.json", result)
        require(interface.visualize_simulation(result["simulation_id"], str(output / "simulation.png")))
        return 0
    except Exception as error:
        OutputPolicy().write_json(output / "failure.json", {"error": str(error)})
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
