# Advanced Features

This guide covers the non-core capabilities that **actually exist** in the
codebase. Earlier written material claimed features such as JAX acceleration
(`enable_jax`, `jax_infer_states`), temporal deep active inference
(`set_temporal_horizon`, `deep_planning`), information-theoretic metrics
(`calculate_kl_divergence`, `calculate_entropy`), learning tools (`enable_learning`,
`update_model`), and complex multi-agent scenarios (`create_simulation`,
`analyze_simulation`, `register_environment`, `connect_agent_to_environment`).
**None of these are implemented.** This repository does not wrap PyMDP's
experimental JAX backend or its learning/example modules. What follows is the
accurate set of advanced functionality.

## Gridworld agents and environments

Beyond raw `Agent` objects, the interface ships purpose-built gridworld support:

- `create_gridworld_agent` builds an agent whose observation model combines a
  position observation (identity over grid cells) and a reward observation, with
  prior preferences biased toward reward cells.
- `create_grid_world_env` builds a navigable gridworld environment with reward
  locations.

These compose with `run_simulation` to produce a navigable foraging-style loop.

## Environments module

The `src/mcp/environments/` package provides reusable environment helpers
(`GridWorldEnvironment`, `CustomEnvironment`, and `create_*` factory functions)
used by the interface and available programmatically via the `mcp` package.

## Visualization and analysis

The `src/mcp/visualization/` package, wired into the interface by
`visualization/extensions.py`, adds visualization methods to `PyMDPInterface`:

- **Belief dynamics** — `visualize_belief_dynamics(session_id, output_file)` and
  `plot_belief_dynamics(session_id, output_file)` render heatmaps of how an
  agent's beliefs evolve over a simulation, with a panel per state factor.
- **Free-energy analysis** — `analyze_free_energy(session_id, output_file)`
  produces plots and metrics of the agent's free-energy-minimization process.
- **Generative-model / simulation visualization** — helpers in
  `visualization/generative_model.py`, `visualization/simulation.py`, and
  `visualization/free_energy.py`.

These are exercised in the test suite, which writes sample plots under
`tests/output/`.

## Free-energy calculation

`calculate_free_energy(A_matrix, prior, observation, q_states)` and
`infer_states_from_observation(A_matrix, prior, observation)` are standalone
analysis helpers built on `pymdp.maths.get_joint_likelihood`. They let you probe
free-energy components (energy, entropy, divergence, accuracy) without running a
full agent loop.

## Sessions

The FastAPI server (`src/mcp/server/app.py`) maintains persistent **sessions**
that bind an agent to an environment. You create a session, run it for `N`
steps, and query/visualize the accumulated history:

```bash
# create a session
POST /sessions            {"agent_id": "...", "env_id": "..."}
# run it
POST /sessions/{id}/run   {"steps": 10}
# inspect / visualize
GET  /sessions
GET  /sessions/{id}/visualize?format=png|svg|json
GET  /analyze/free_energy/{id}
GET  /analyze/beliefs/{id}
```

Sessions keep simulation history in `PyMDPInterface.sessions` so state persists
across calls.

## Clients

`src/mcp/client.py` (with `src/mcp/client/`) provides two ways to talk to the
servers:

- **`MCPClient`** — network client that talks to a running FastAPI/HTTP server
  over aiohttp (`ping`, `get_tools`, `call_tool`).
- **`DirectMCPClient`** — calls the MCP tools in `src/main.py` in-process with no
  server, useful for testing and development.

There is also an `MCPToolKit` (`src/mcp/client/tools.py`) that wraps common
workflows as higher-level convenience calls.

## Configuration

Server behavior is controlled by environment variables:

| Variable | Default | Description |
| --- | --- | --- |
| `HOST` | `0.0.0.0` | Host to bind (MCP SSE server). |
| `PORT` | `8050` | Port to listen on (MCP SSE server). |
| `TRANSPORT` | `sse` | MCP transport: `sse` or `stdio`. |

The FastAPI app (`mcp.server.app:app`) is served with uvicorn and listens on the
bind host/port you pass to uvicorn.

## What is intentionally out of scope

This project is a focused wrapper around PyMDP's **discrete-state-space
inference**, **policy optimization**, and **action sampling**. It does not
re-implement or expose PyMDP's experimental JAX acceleration, its standalone
example/learning scripts, or custom information-theoretic utilities beyond the
free-energy helpers listed above. If you need those, use PyMDP directly.
