# Implementation Details

This document describes how the project actually works, at the level of the
source code in `src/`.

## Layered architecture

1. **MCP server layer** — `src/main.py` declares MCP tools (`@mcp.tool()`) on
   the bundled FastMCP implementation in `src/mcp/server/fastmcp.py`, which runs
   over SSE (default) or stdio.
2. **FastAPI HTTP layer** — `src/mcp/server/app.py` builds a FastAPI app
   (`create_app()`, `app = create_app()` at module scope) exposing the same
   capability set as REST endpoints, served with uvicorn.
3. **PyMDP interface layer** — `PyMDPInterface` in `src/mcp/utils.py` is the
   single bridge between the servers and PyMDP.
4. **PyMDP (dependency)** — the unmodified `inferactively-pymdp` pip package.

## `PyMDPInterface`

The central class. Its constructor keeps in-memory state:

```python
def __init__(self):
    self.agents = {}             # name -> pymdp Agent
    self.environments = {}       # name -> environment object/dict
    self.simulation_history = {}
    self.computation_logs = {}
    self.debug_mode = True
    self.sessions = {}           # session id -> {agent, environment, history}
```

Agent creation normalizes the supplied `A`/`B` (and optional `C`/`D`) JSON into
the NumPy object arrays PyMDP expects, then constructs a PyMDP `Agent` and stores
it by name:

```python
def create_agent(self, name, generative_model):
    A = generative_model.get("A", [])
    B = generative_model.get("B", [])
    C = generative_model.get("C", None)
    D = generative_model.get("D", None)
    # ... normalize A columns to sum to 1, B along each action ...
    agent = Agent(A=A_obj, B=B_obj, C=C_obj, D=D_obj)
    self.agents[name] = agent
    return {"name": name, "num_observation_modalities": ..., "num_state_factors": ...}
```

Environment creation builds a step/reset environment (used for gridworld) and
stores it in `self.environments`:

```python
def create_grid_world_env(self, name, grid_size, reward_positions):
    # returns environment metadata and registers the environment
    self.environments[name] = env
    return {"name": name, ...}
```

## MCP server (`src/main.py`)

Each tool is an async function that receives an MCP `Context`, pulls the shared
`PyMDPInterface` from the lifespan context, calls the corresponding
`PyMDPInterface` method, and serializes the result with `NumpyEncoder`:

```python
@mcp.tool()
async def create_agent(ctx: Context, name: str, generative_model: str) -> str:
    pymdp_interface = ctx.request_context.lifespan_context.pymdp_interface
    model_dict = json.loads(generative_model)
    result = pymdp_interface.create_agent(name, model_dict)
    return json.dumps({"result": {"agent": result}}, cls=NumpyEncoder)
```

`src/mcp/server/fastmcp.py` provides a minimal `FastMCP` class that registers
tools via `@mcp.tool()`, extracts their parameters from the function signature,
and serves them over SSE (`/schema`, `/tools`, `/invoke`, `/sse`) or stdio
(line-delimited JSON on stdin/stdout). It reads `HOST`/`PORT`/`TRANSPORT` from the
environment (defaults `0.0.0.0`, `8050`, `sse`).

## FastAPI HTTP server (`src/mcp/server/app.py`)

Built with `create_app()` and exposed as `app`. Highlights:

- `GET /ping` — health check.
- `GET /tools`, `POST /tools/{tool_id}` — list and invoke tools.
- `GET /agents`, `GET /agents/{agent_id}` — list/retrieve agents.
- `GET /environments`, `POST /environments`, `GET /environments/{env_id}` —
  manage environments.
- `GET /simulations`, `POST /sessions`, `POST /sessions/{id}/run` — run and
  query simulation sessions.
- `/agents/{id}/visualize`, `/sessions/{id}/visualize`,
  `/analyze/free_energy/{id}`, `/analyze/beliefs/{id}` — analysis and
  visualization.

It maintains a module-level `pymdp_interface` plus a `sessions` registry so
state persists across requests.

## Data conversion

- `NumpyEncoder` (in `src/mcp/utils.py`) serializes `np.integer`,
  `np.floating`, `np.ndarray`, and `np.bool_` to native JSON types.
- `_convert_to_obj_array` / `_convert_from_obj_array` / `_ensure_native_types`
  convert between plain JSON lists and PyMDP's nested NumPy object arrays.

## Visualization

`src/mcp/visualization/extensions.py`'s `_add_visualization_methods()` attaches
methods to `PyMDPInterface` at runtime: `plot_belief_dynamics` (heatmaps of
belief evolution, one panel per state factor) and `analyze_free_energy` (plots +
metrics of the free-energy process). The `mcp` package's `visualization` module
also exports generative-model, free-energy, and simulation plotting helpers.

## Clients

`src/mcp/client.py` and `src/mcp/client/` provide:

- **`MCPClient`** — async aiohttp client against the HTTP server (`ping`,
  `get_tools`, `call_tool`); can log requests/responses and write outputs to a
  directory.
- **`DirectMCPClient`** — calls the `src/main.py` tools in-process with a mock
  context, no server needed (useful for tests/development).
- **`MCPToolKit`** (`client/tools.py`) — higher-level convenience wrappers.

There is also `get_best_client(output_dir=None)`, which probes for a running
server at `http://localhost:8050/schema` and falls back to `DirectMCPClient`.

## Session management

Sessions bind an agent to an environment and accumulate a per-timestep history
(`observations`, `actions`, `rewards`, `states`, `policies`, `free_energies`).
They are stored in `PyMDPInterface.sessions` and queried/visualized through the
HTTP layer.
