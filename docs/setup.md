# Setup and Usage

This page covers installing pymcp, running its two servers, and driving the core
`PyMDPInterface` directly. For the architecture behind these pieces see
[`ARCHITECTURE.md`](ARCHITECTURE.md); for the full method surface see
[`mcp_tools.md`](mcp_tools.md).

## Requirements

- Python **>= 3.12** (declared in `pyproject.toml`, matching `uv.lock`).
- Package manager: [`uv`](https://docs.astral.sh/uv/) (the repository ships a
  committed `uv.lock`).
- No network is needed for a local install once dependencies are resolved; the
  MCP/REST servers themselves listen on localhost/loopback by default.

## Install

```bash
uv pip install -e .                      # installs fastapi, uvicorn, numpy, matplotlib, aiohttp, ...
uv pip install inferactively-pymdp       # the engine, already declared in pyproject.toml
```

`inferactively-pymdp` is a declared dependency (`>=1.0.5`); step two just makes
the engine explicit. There is **no vendored PyMDP clone** — the interface talks
to the installed package through its public API.

For reproducible development, prefer the committed lock — the gate's `setup`
arm does exactly this:

```bash
./run.sh setup        # = uv sync --frozen --python 3.12
```

## Quickstart (interface directly)

```python
from mcp.utils import PyMDPInterface

pymdp = PyMDPInterface()

# 1 observation modality [obs, state], 1 control factor
model = pymdp.define_generative_model(A_dims=[[2, 3]], B_dims=[[3, 3, 2]])
pymdp.create_agent("agent_1", model)

# 3x3 grid world with a reward in the bottom-right
pymdp.create_grid_world_env("world_1", [3, 3], [[2, 2]])

result = pymdp.run_simulation("agent_1", "world_1", num_steps=5)
print(result["timesteps"])
```

## Run a server

### MCP server (`src/main.py`)

```bash
TRANSPORT=sse uv run src/main.py      # SSE (default)
TRANSPORT=stdio uv run src/main.py    # stdio
```

### FastAPI HTTP server (`src/mcp/server/app.py`)

```bash
uvicorn mcp.server.app:app
```

`scripts/start_fastapi_server.py` is a convenience launcher for the same app.
The legacy `scripts/start_mcp_server.py` predates both and is kept only because
the example scripts under `scripts/` call it; see
[`../scripts/README.md`](../scripts/README.md).

## Configuration

| Variable | Default | Description |
| --- | --- | --- |
| `HOST` | `0.0.0.0` | Host to bind (SSE server). |
| `PORT` | `8050` | Port to listen on (SSE server). |
| `TRANSPORT` | `sse` | Transport: `sse` or `stdio`. |
| `CORS_ORIGINS` | `http://localhost:8080,http://127.0.0.1:8080` | Allowed origins for the FastAPI server. |
| `MCP_OUTPUT_DIR` | `./output` | Directory where visualization files are written (path-traversal safe). |

## Next steps

- [Testing](testing.md) — the 54-test suite and CI gate.
- [Advanced features](advanced_features.md) — environments, sessions, clients.
- [Implementation details](implementation_details.md) — serialization and layering.
