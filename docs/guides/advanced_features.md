# Advanced Features

The [explicit tool registry](../reference/mcp_tools.md) defines the supported wrapper
surface. Native Agent execution may use PyMDP's JAX implementation, but no
`enable_jax`, custom inference, learning-update, comparison or distributed
execution API is exposed.

## Grid and custom environments

`create_gridworld_agent` and `create_grid_world_env` compose when their grid
dimensions/reward modalities agree. Grid actions include stay and preserve
rectangular row/column conventions. `GridWorldEnvironment` and
`CustomEnvironment` are real step/reset objects, not serialized dictionaries
with invented methods.

Custom observation modalities remain fixed. Empty rewards stay empty.
Grid reset defaults to origin and custom reset to first state;
`reset_mode="configured"` and `stop_on_done=True` are explicit choices.
Terminal flags do not silently shorten the default fixed-step execution.

## Visualization and analysis

The actual interface methods invoke renderers under `src/mcp/visualization/`
and belief-history analysis, not retired extension injection.
Plots use complete measured policy/energy/history records, distinguish
pre-transition state from next-state fields and include final endpoints.
The model overview labels its selection; full model JSON is available
separately. Tests inspect actual artists, files and animation frames in
isolated temporary outputs.

Numerical Bayes/VFE helpers use stable log-space calculations and explicit
support checks. Signed entropy, KL/accuracy and full negative/positive EFE
conventions are in the [measurement contract](../reference/mcp_tools.md).
No arbitrary decomposition or fallback energy curve is supplied.

## Sessions and clients

Sessions are core-owned, bounded and process-local. Repeated runs accumulate
real history and run IDs; restart loses that state. Bindings are idempotent
only when the existing agent/environment pair agrees.

Import `MCPClient`, `DirectMCPClient` and `MCPToolKit` from `mcp.client`.
The HTTP client unwraps one native result envelope. The direct client
dispatches explicitly to the same registry without a server, source-main
import, mock context or remote-failure fallback.
Images are actual PNG/SVG bytes; toolkit downloads belong to the caller.

## Operating boundary

Use `mcp.server.app:create_app --factory --workers 1` for direct uvicorn
launch, or the supported launcher in [setup](setup.md). No reload,
multi-worker state sharing or server authentication is implemented.
MCP defaults to loopback; CORS and optional client token headers do not
authorize external exposure.

