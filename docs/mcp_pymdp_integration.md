# MCP & PyMDP Integration

## What is pymcp?

**pymcp** (`pymdp-mcp`) is an implementation of the
[Model Context Protocol (MCP)](https://modelcontextprotocol.io) — plus a plain
FastAPI HTTP API — that exposes the
[PyMDP](https://github.com/infer-actively/pymdp) active inference framework.
It lets MCP-compatible clients (Claude Desktop, Windsurf, custom agents) and
plain HTTP callers create active inference agents, define generative models, run
inference, sample actions, and simulate environments.

The project uses the real installed PyMDP Agent for its admitted filtering and
planning subset. Model admission, environment ownership, transition commits,
transport schemas and measured reporting are explicit wrapper contracts; this
is not a promise to expose every upstream capability unchanged.

## Architecture

The integration has four layers:

1. **MCP server** — `src/main.py` starts the bundled JSON-RPC
   implementation (`src/mcp/server/fastmcp.py`) over SSE or stdio.
2. **FastAPI HTTP server** — `src/mcp/server/app.py:create_app` constructs
   an independent REST instance.
3. **Registry and core** — `mcp.api.ToolRegistry` exposes explicit strict
   schemas and dispatches to `PyMDPInterface` in `src/mcp/utils.py`.
   The core owns native agents, environment objects, priors, pending actions,
   bounded histories and logs.
4. **PyMDP dependency** — the installed `inferactively-pymdp` Agent receives
   batched arrays internally; public model interchange remains unbatched.

## Data flow

1. A client invokes a discovered tool with a native JSON argument object.
2. The registry admits its explicit schema and dispatches under its lock.
3. The core validates model, support, lifecycle and resource constraints.
4. Real PyMDP inference/planning or the requested environment/measurement
   operation produces a native result.
5. MCP encodes JSON-RPC tool content; REST returns one result envelope.
   Images use actual PNG/SVG bytes, not server paths masquerading as downloads.

## Core capabilities

The [tool reference](mcp_tools.md) owns the complete registry, model interchange,
observation → inference → selection → successful transition/commit lifecycle,
reset policies, measured VFE/EFE conventions and bounded retention contract.
The [setup guide](setup.md) distinguishes bundled MCP, REST and explicit
in-process clients. Instances share a contract, not cross-process state.

See [mcp_tools.md](mcp_tools.md) for the full reference.

## Use cases

- **Research & education** — explore active inference models interactively.
- **Applied agents** — give LLM-driven systems belief-based planning.
- **Cognitive science / neuroscience** — model perception, decision-making, and
  simulated experiments on the free-energy principle.

## References

- [PyMDP GitHub repository](https://github.com/infer-actively/pymdp)
- [PyMDP documentation](https://pymdp-rtd.readthedocs.io/en/latest/)
- [Model Context Protocol specification](https://modelcontextprotocol.io)
- Heins, C., Millidge, B., Demekas, D., Klein, B., Friston, K., Couzin, I. D., &
  Tschantz, A. (2022). pymdp: A Python library for active inference in discrete
  state spaces. *Journal of Open Source Software*, 7(73), 4098.
  doi:[10.21105/joss.04098](https://doi.org/10.21105/joss.04098)
