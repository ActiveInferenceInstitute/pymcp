# Implementation Details

The [architecture](ARCHITECTURE.md) maps the modules; the
[tool reference](mcp_tools.md) defines public arguments, lifecycle and
measurement semantics. These are supported contracts, not descriptions of
retired duplicate implementations.

## Dispatch and state

`ToolRegistry` is the single explicit public surface. Its strict schemas reject
unknown parameters and dispatch to an owned `PyMDPInterface`.
`create_app()` creates independent REST state; there is no eager global `app`
or parallel transport-local session registry. `src/main.py` starts the bundled
MCP protocol over stdio or SSE.

Native Agent configuration is immutable. Admission validates shapes, support,
finite values and bounded dimensions before allocating. Normalization is
opt-in and exported model JSON is complete and unbatched. The core retains
predictive priors and pending action tokens separately from the native model.

## Runs and measurements

Runs store actual pre-transition observations, beliefs, physical states,
actions and next-state fields. `num_steps` is executed work;
`requested_steps` retains the request. Sessions retain cumulative history and
actual run IDs. Resetting an environment alone does not reset agent beliefs.

VFE uses the actual observation/prior and signed entropy term. Full policy
posteriors and native negative EFE are retained, with explicit sign conversion
for positive EFE. Unsupported/infinite measurements remain explicit;
arbitrary risk/ambiguity terms and fallback traces are absent.
Run logs cannot include later runs; evicted detail is unavailable.

## Protocol and output custody

MCP implements the `2025-03-26` initialize/initialized/tools lifecycle,
notifications and JSON-RPC error distinctions. Stdio stdout is protocol-only.
SSE responses belong to one bounded channel; response lifetime owns cleanup
on normal or backpressured disconnect.

REST returns one native result envelope. PNG/SVG endpoints return actual bytes
and matching content types. `OutputPolicy` confines both primary files and
companions, including symlink parents/files. A toolkit download writes to the
caller's trusted local destination, not an arbitrary server path.

## Clients and processes

Import `MCPClient`, `DirectMCPClient` and `MCPToolKit` from `mcp.client`.
Direct mode is explicit registry dispatch; it does not import `src/main.py`,
probe HTTP, construct mock contexts or fall back after a remote failure.
Client/remote errors propagate. Config contains `server{host,port,use_ssl}`,
`auth{enabled,token}` and `options{timeout}`; enabled auth controls outbound
headers only, not server authentication or automatic retries.

The REST launcher is loopback/single-worker/no-reload. The example runner
tracks its own PID and instance nonce, verifies readiness, enforces deadlines,
preserves failed/not-run dispositions and reaps only its owned process.
See [setup](setup.md) and [example commands](../scripts/README.md).

## Limits and qualification

Trusted launch configuration bounds model elements, policy combinations,
steps, entities, retained runs/logs, protocol traffic and animation frames.
These are admission limits, not measured memory/performance promises.
Docker provisions uv 0.13.8, a frozen no-dev/non-editable environment and
starts the installed interpreter directly. Source tests do not certify an
image build or installed-artifact runtime.

