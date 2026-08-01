# pymcp (pymdp-mcp)

**A Model Context Protocol (MCP) server and HTTP API for the PyMDP active inference library.**

`pymcp` exposes the discrete-state-space active inference framework
[PyMDP](https://github.com/infer-actively/pymdp) through two clean interfaces:

1. **MCP tools** — declared in `src/main.py` over the FastMCP framework
   (`src/mcp/server/fastmcp.py`), which speaks the Model Context Protocol over
   SSE or stdio so any MCP-compatible client (Claude Desktop, Windsurf, etc.) can
   create agents, run inference, and simulate environments.
2. **A FastAPI HTTP server** — `src/mcp/server/app.py`, which serves the same
   capabilities as a plain REST API (`/ping`, `/tools`, `/agents`,
   `/environments`, `/sessions`, …) over uvicorn.

Both are thin wrappers around the `PyMDPInterface` class in `src/mcp/utils.py`,
which calls the real, unmodified PyMDP library — there are no mock or simplified
implementations, and results are identical to using PyMDP directly.

---

## Overview

| Package | `pymdp-mcp` (source layout under `src/mcp/`) |
| --- | --- |
| Python | `>=3.8` |
| Active inference engine | [PyMDP](https://github.com/infer-actively/pymdp) (pip package `inferactively-pymdp`) |
| Protocol | Model Context Protocol (SSE / stdio) + REST (FastAPI) |
| License | MIT |

The `PyMDPInterface` keeps in-memory registries of **agents**, **environments**,
and **simulation history**, and converts between JSON payloads and PyMDP's NumPy
object arrays so every tool takes plain JSON in and returns plain JSON out.

## Installation

### 1. Install this package

Using [uv](https://docs.astral.sh/uv/):

```bash
uv pip install -e .
```

This installs the `pymcp` server/client code and its runtime dependencies
(FastAPI, uvicorn, numpy, matplotlib, aiohttp, …).

### 2. Install PyMDP

PyMDP is **not** bundled as a dependency, so you must install it separately:

```bash
uv pip install inferactively-pymdp
```

> The package talks to the PyMDP library through its public API
> (`pymdp.agent.Agent`, `pymdp.envs.Env`, `pymdp.utils`, `pymdp.maths`). There is
> no vendored `pymdp-clone` directory in this repository.

## Quickstart (using the interface directly)

The quickest way to get a feel for the API is through `PyMDPInterface` directly:

```python
from mcp.utils import PyMDPInterface

pymdp = PyMDPInterface()

# A minimal generative model with 1 hidden state and 2 observations
model = pymdp.define_generative_model(
    A_dims=[[2, 3]],        # one observation modality: [obs_dim, state_dim]
    B_dims=[[3, 3, 2]],     # one state factor: [state_to, state_from, action_dim]
)

# Create an agent from that model
pymdp.create_agent("agent_1", model)

# Create a 3x3 grid world with a reward in the corner
env = pymdp.create_grid_world_env("world_1", [3, 3], [[2, 2]])
print(env)

# Step through a short simulation
result = pymdp.run_simulation("agent_1", "world_1", num_steps=5)
print(result)
```

## The `PyMDPInterface` methods

All MCP tools and HTTP endpoints are thin wrappers over these methods.

| Method | Description |
| --- | --- |
| `create_agent(name, generative_model)` | Create an active inference agent from a generative model dict (`A`, `B`, optional `C`, `D`). |
| `define_generative_model(A_dims, B_dims)` | Build random `A`/`B` matrices (plus default `C`, `D`) from dimension lists. |
| `create_gridworld_agent(name, grid_size, reward_positions, ...)` | Create a gridworld agent with position + reward observation models. |
| `create_grid_world_env(name, grid_size, reward_positions)` | Create a grid world environment with rewards. |
| `infer_states(agent_id, observation, method="FPI")` | Infer hidden-state posterior beliefs for an observation (FPI, BP, …). |
| `infer_policies(agent_id, planning_horizon=None)` | Optimize beliefs over policies via expected free energy. |
| `sample_action(agent_id, planning_horizon=None)` | Sample an action from the policy posterior. |
| `step_environment(env_id, action)` | Advance an environment given an action. |
| `run_simulation(agent_id, env_id, num_steps=10, ...)` | Run a full agent–environment simulation loop. |
| `reset_environment(env_id)` | Reset an environment to its initial state. |
| `get_agent(name)` / `get_environment(name)` | Retrieve stored agents/environments as JSON. |
| `get_all_functions()` | List all public methods available on the interface. |

Additional helpers exist in `utils.py` (e.g. `validate_generative_model`,
`calculate_free_energy`, `infer_states_from_observation`, `get_all_agents`,
`get_all_simulations`) and visualization methods (`visualize_belief_dynamics`,
`plot_belief_dynamics`, `analyze_free_energy`) are attached by
`src/mcp/visualization/extensions.py`. See the docs for details.

## Server run modes

There are two independent servers in this repository.

### 1. FastAPI HTTP server (`src/mcp/server/app.py`)

Serves the interface as a REST API. Run it with uvicorn:

```bash
uvicorn mcp.server.app:app
```

The module exposes `app = create_app()` at import time, so any uvicorn invocation
works. Endpoints include:

- `GET /ping` — health check
- `GET /tools`, `POST /tools/{tool_id}` — list and invoke tools
- `GET /agents`, `GET /agents/{agent_id}` — list/retrieve agents
- `GET /environments`, `POST /environments`, `GET /environments/{env_id}` — manage environments
- `GET /simulations`, `POST /sessions`, `POST /sessions/{id}/run` — run and inspect simulations
- `/agents/{id}/visualize`, `/sessions/{id}/visualize`, `/analyze/free_energy/{id}`, `/analyze/beliefs/{id}` — analysis and visualization

### 2. MCP server (`src/main.py` + `src/mcp/server/fastmcp.py`)

The MCP tools are declared in `src/main.py` using the bundled FastMCP
implementation. It supports two transports:

- **SSE** (default) — exposes `/schema`, `/tools`, `/invoke`, and `/sse`.
- **stdio** — reads JSON requests on stdin and writes responses to stdout, so an
  MCP client can launch the server itself.

### Configuration environment variables

| Variable | Default | Description |
| --- | --- | --- |
| `HOST` | `0.0.0.0` | Host to bind (SSE server). |
| `PORT` | `8050` | Port to listen on (SSE server). |
| `TRANSPORT` | `sse` | Transport: `sse` or `stdio`. |

## Testing

The test suite lives under `tests/` and is configured in `pyproject.toml`
(`testpaths = ["tests"]`, `python_files = "test_*.py"`). Run it with pytest:

```bash
pytest
```

The tests exercise the `PyMDPInterface` directly and through the MCP layer, and
verify that the interface reproduces the results of direct PyMDP calls. Outputs
and plots are written under `tests/output/`.

## Documentation

See the [`docs/`](docs/) directory for detailed guides:

- [MCP & PyMDP integration](docs/mcp_pymdp_integration.md)
- [MCP tools reference](docs/mcp_tools.md)
- [Implementation details](docs/implementation_details.md)
- [Advanced features](docs/advanced_features.md)
- [PyMDP integration](docs/pymdp_integration.md)

## Citation

If you use this project in research, please cite the underlying PyMDP library:

```bibtex
@article{Heins2022,
  doi = {10.21105/joss.04098},
  url = {https://doi.org/10.21105/joss.04098},
  year = {2022},
  publisher = {The Open Journal},
  volume = {7},
  number = {73},
  pages = {4098},
  author = {Conor Heins and Beren Millidge and Daphne Demekas and Brennan Klein and Karl Friston and Iain D. Couzin and Alexander Tschantz},
  title = {pymdp: A Python library for active inference in discrete state spaces},
  journal = {Journal of Open Source Software}
}
```

## License

This project is licensed under the [MIT License](LICENSE).
Copyright (c) 2025 Cole Medin.
