# MCP & PyMDP Integration

## What is pymcp?

**pymcp** (`pymdp-mcp`) is an implementation of the
[Model Context Protocol (MCP)](https://modelcontextprotocol.io) — plus a plain
FastAPI HTTP API — that exposes the
[PyMDP](https://github.com/infer-actively/pymdp) active inference framework.
It lets MCP-compatible clients (Claude Desktop, Windsurf, custom agents) and
plain HTTP callers create active inference agents, define generative models, run
inference, sample actions, and simulate environments.

The project is a thin wrapper: it routes every call to the **unmodified PyMDP
library** through a single `PyMDPInterface` class, so results match using PyMDP
directly. There is no mocked or simplified re-implementation of the algorithms.

## Architecture

The integration has four layers:

1. **MCP server** — `src/main.py` declares MCP tools on the bundled FastMCP
   implementation (`src/mcp/server/fastmcp.py`), served over SSE or stdio.
2. **FastAPI HTTP server** — `src/mcp/server/app.py` exposes the same interface
   as a REST API (`/ping`, `/tools`, `/agents`, `/environments`, `/sessions`).
3. **PyMDP interface** — `PyMDPInterface` in `src/mcp/utils.py` is the bridge
   between the servers and PyMDP. It holds in-memory registries of agents,
   environments, and simulation/session history, and converts JSON ⇄ PyMDP's
   NumPy object arrays.
4. **PyMDP (dependency)** — the live library from the pip package
   `inferactively-pymdp` (`pymdp.agent.Agent`, `pymdp.envs.Env`, `pymdp.utils`,
   `pymdp.maths`).

## Data flow

1. A client calls a tool — over MCP (SSE/stdio) or an HTTP endpoint.
2. The server looks up the corresponding `PyMDPInterface` method.
3. `PyMDPInterface` calls the PyMDP API for the requested operation.
4. Numeric results (NumPy arrays/object arrays) are serialized to plain JSON via
   `NumpyEncoder` and `_convert_from_obj_array`.
5. The JSON response is returned to the client; visualizations/logs are written
   to files as requested.

## Core capabilities

- **Agent creation and model definition** — `create_agent`,
  `define_generative_model`, `create_gridworld_agent`.
- **Environment creation and stepping** — `create_grid_world_env`,
  `step_environment`, `reset_environment`.
- **Inference and action selection** — `infer_states`, `infer_policies`,
  `sample_action`.
- **Simulation** — `run_simulation` (full perception–action loop).
- **Introspection** — `get_agent`, `get_environment`, `get_all_functions`.
- **Analysis & visualization** — free-energy calculation, belief-dynamics and
  free-energy plots, session-based history.

See [mcp_tools.md](mcp_tools.md) for the full reference.

## Use cases

- **Research & education** — explore active inference models interactively.
- **Applied agents** — give LLM-driven systems belief-based planning.
- **Cognitive science / neuroscience** — model perception, decision-making, and
  simulated experiments on the free-energy principle.

## References

- [PyMDP GitHub repository](https://github.com/infer-actively/pymdp)
- [PyMDP documentation](https://pymdp.readthedocs.io/)
- [Model Context Protocol specification](https://modelcontextprotocol.io)
- Heins, C., Millidge, B., Demekas, D., Klein, B., Friston, K., Couzin, I. D., &
  Tschantz, A. (2022). pymdp: A Python library for active inference in discrete
  state spaces. *Journal of Open Source Software*, 7(73), 4098.
  doi:[10.21105/joss.04098](https://doi.org/10.21105/joss.04098)
