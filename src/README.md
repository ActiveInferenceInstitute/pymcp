# `src/` — implementation

The Python source tree for pymcp. Both servers are thin wrappers over one core
class, `PyMDPInterface` in [`mcp/utils.py`](mcp/utils.py), so MCP and REST behave
identically.

## Layout

```
src/
  main.py                  # MCP server entry point (FastMCP over SSE / stdio)
  mcp/
    __init__.py            # package version and metadata
    utils.py               # PyMDPInterface — the core integration layer
    belief_dynamics.py     # belief-trajectory helpers
    client.py, client/     # MCPClient (network) and DirectMCPClient (in-process)
    environments/          # gridworld + custom-environment helpers
    server/
      app.py               # FastAPI HTTP app (uvicorn mcp.server.app:app)
      fastmcp.py           # bundled FastMCP implementation used by src/main.py
    testing/               # test fixtures, helpers, and setup utilities
    visualization.py, visualization/  # model / free-energy / belief-dynamics plots
```

## Entry points

| Module | Role |
| --- | --- |
| `main.py` | MCP server (SSE default, `TRANSPORT=stdio` optional). |
| `mcp/server/app.py` | FastAPI REST server. |
| `mcp/utils.py` | `PyMDPInterface` — every tool and endpoint delegates here. |

See [`../docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md) for the module diagram
and request lifecycle, and [`../docs/implementation_details.md`](../docs/implementation_details.md)
for serialization and session handling.

## Running

Nothing in this tree is executed on its own except the entry points above; see
[`../README.md`](../README.md) for install and run commands and
[`../docs/setup.md`](../docs/setup.md) for the configuration variables.

## Conventions

- Keep the MCP layer and the FastAPI layer thin: add behaviour to
  `PyMDPInterface`, then expose it, so both surfaces stay consistent.
- Convert NumPy/JAX values to plain JSON at the boundary (see
  [`mcp/utils.py`](mcp/utils.py)).
- Do not vendor PyMDP; call the installed `inferactively-pymdp` package.
