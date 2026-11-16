# `src/` — implementation

The Python source tree for pymcp. MCP, REST and explicit direct clients use
one strict registry and `PyMDPInterface`; their instances do not share state.

## Layout

```
src/
  main.py                  # MCP server entry point (FastMCP over SSE / stdio)
  mcp/
    __init__.py            # package version and metadata
    api.py                 # explicit tool registry and strict native schemas
    models.py              # model/settings admission and trusted limits
    utils.py               # native inference, committed transitions and histories
    outputs.py             # primary/companion output custody
    belief_dynamics.py     # actual belief-history analysis
    client/                # HTTP client, explicit direct client and toolkit
    environments/          # real grid/custom objects
    server/
      app.py               # FastAPI create_app factory; no eager global app
      fastmcp.py           # bundled MCP JSON-RPC, stdio and SSE
    testing/               # shared test inputs and helpers
    visualization/         # actual model/history renderers
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

- Keep transports thin: add admitted core behavior and an explicit registry
  entry/schema together; public attributes are not automatically tools.
- Convert NumPy/JAX values to plain JSON at the boundary (see
  [`mcp/utils.py`](mcp/utils.py)).
- Do not vendor PyMDP; call the installed `inferactively-pymdp` package.
