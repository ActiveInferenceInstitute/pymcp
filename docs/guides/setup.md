# Setup and Usage

## Requirements and provisioning

Python >=3.12 and the resolved project dependencies are required.
`inferactively-pymdp>=1.0.5` is declared and locked at 1.0.5; do not install
another backend independently of the lock. This project owns the bundled
`mcp` namespace, not the official SDK package.

When dependency provisioning is authorized, the existing setup arm uses
`./run.sh setup` and the frozen lock. Provisioning is separate from starting an
already prepared environment. Docker uses uv 0.13.8, frozen/no-dev/non-editable
installation and direct interpreter startup; source proof is not image proof.

For the commands below, set `PROJECT_PY` to the absolute Python executable of
that prepared environment. Run from the repository root, explicitly selecting
this checkout's `src` when using source rather than an installed artifact.

## Quickstart

The executable matching-grid example is in the
[root quickstart](../../README.md#quickstart-interface-directly).
Do not pair an arbitrary model with an incompatible environment.
The [tool reference](../reference/mcp_tools.md) defines model shapes, commit/reset semantics,
measured results and unsupported operations.

## Bundled MCP

```bash
PYTHONPATH="$PWD/src" "$PROJECT_PY" -B src/main.py --transport stdio
PYTHONPATH="$PWD/src" "$PROJECT_PY" -B src/main.py --transport sse --host 127.0.0.1 --port 8050
```

This is the bundled `2025-03-26` tools protocol, not an official-SDK adoption.
Stdio requires the actual initialize/initialized/tools lifecycle; starting a
process and sending EOF is not protocol acceptance. SSE opens `/sse` and sends
JSON-RPC messages to the returned session-specific `/messages` endpoint.

## REST

```bash
PYTHONPATH="$PWD/src" "$PROJECT_PY" -B scripts/start_fastapi_server.py --host 127.0.0.1 --port 8080 --output-dir /path/to/owned-output
```

The equivalent application factory is `mcp.server.app:create_app`; if invoking
uvicorn directly, use `--factory --workers 1 --host 127.0.0.1`.
There is no eager `mcp.server.app:app`. Reload and multiple workers are not
supported because registries are process-local. The retired
`scripts/start_mcp_server.py` is not an entry point.

## Configuration and exposure

| Setting | Contract |
| --- | --- |
| `HOST`, `PORT`, `TRANSPORT` | MCP defaults: `127.0.0.1`, `8050`, `sse`; explicit CLI options select a launch |
| `MCP_OUTPUT_DIR` | Server-owned output root for confined primary/companion files |
| `MCP_LIMITS` | Trusted launch-time resource limits, not per-tool caller overrides |
| `CORS_ORIGINS` | Browser-origin policy, not authentication |

Non-loopback exposure is an explicit owner decision. No server authentication
is implemented. Client config `server{host,port,use_ssl}`,
`auth{enabled,token}`, `options{timeout}` controls destinations, timeout and
optional outbound token headers; it does not create a server auth/retry layer.

`DirectMCPClient` is an explicit in-process alternative from `mcp.client`.
It never probes a remote server or falls back after a remote failure.
MCP, REST and direct instances share a contract, not state.

## Verification

Use [the test inventory](../../tests/README.md) for focused source contracts and
[the run guide](RUN_GUIDE.md) for the canonical gate. Counts and timings belong
to [generated gate facts](../_generated/gate-facts.md), not current prose.
Historical facts are not refreshed by a focused test run.

