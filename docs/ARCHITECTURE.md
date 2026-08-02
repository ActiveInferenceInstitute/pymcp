# Architecture

This document explains how `pymcp` is put together: where the code lives, how a
request flows from a client to PyMDP and back, and the design decisions that
keep the MCP and REST surfaces consistent.

## Module layout

```mermaid
flowchart TD
    subgraph src["src/"]
        main["main.py<br/>entry point + @mcp.tool declarations"]
        subgraph mcp["mcp/"]
            subgraph server["server/"]
                fastmcp["fastmcp.py<br/>FastMCP framework (SSE/stdio)"]
                app["app.py<br/>FastAPI REST server"]
            end
            utils["utils.py<br/>PyMDPInterface (the core)"]
            subgraph envs["environments/"]
                grid["gridworld.py"]
                custom["custom.py"]
            end
            subgraph viz["visualization/"]
                sim["simulation.py"]
                fe["free_energy.py"]
                gm["generative_model.py"]
                ext["extensions.py"]
            end
            belief["belief_dynamics.py"]
            client["client.py"] 
        end
    end

    main --> fastmcp
    fastmcp --> utils
    app --> utils
    utils --> envs
    utils --> viz
    utils --> belief
    parser["PyMDP (inferactively-pymdp)"] <--> utils
```

## Request lifecycle

A single tool call travels through four layers. The goal is that **every layer
speaks JSON**, and only `PyMDPInterface` touches NumPy/JAX objects.

```mermaid
sequenceDiagram
    participant C as Client (MCP / HTTP)
    participant S as Server (FastMCP / FastAPI)
    participant I as PyMDPInterface
    participant P as PyMDP Agent/env

    C->>S: JSON tool invocation
    S->>I: method(name, *json args)
    Note over I: validate + normalize JSON → NumPy
    I->>P: PyMDP call (infer_states, infer_policies, ...)
    P-->>I: NumPy/JAX result
    Note over I: convert → plain JSON (NumpyEncoder)
    I-->>S: JSON result dict
    S-->>C: JSON response
```

## Shared core: `PyMDPInterface`

Both servers delegate to the same methods, so behavior is identical whichever
surface you use.

```mermaid
flowchart LR
    subgraph State["PyMDPInterface in-memory state"]
        A["agents<br/>name → PyMDP Agent"]
        Env["environments<br/>name → env dict"]
        H["simulation_history<br/>id → timesteps"]
        C["computation_logs<br/>per-agent trace"]
    end

    Create["create_agent"] --> A
    Grid["create_grid_world_env"] --> Env
    Sim["run_simulation"] --> H
    Inf["infer_states / infer_policies / sample_action"] --> A
    Inf --> C
```

## Key design decisions

1. **One interface, two transports.** Both the hand-rolled FastMCP layer
   (`src/main.py`) and the FastAPI server (`src/mcp/server/app.py`) are thin
   wrappers over `PyMDPInterface` (`src/mcp/utils.py`). No business logic leaks
   into either server; everything routes through `utils.py`.
2. **JSON boundary.** All public methods take/return JSON-serializable values.
   NumPy and JAX arrays are converted via `NumpyEncoder` and the
   `_convert_from_obj_array` / `_ensure_native_types` helpers.
3. **Real PyMDP, no mocks.** There is no `pymdp-clone` in the tree and no
   simplified implementation. The interface calls the installed
   `inferactively-pymdp` package's public API; the test suite verifies results
   match direct PyMDP calls.
4. **Immutable engine model.** Current PyMDP agents are immutable (equinox
   dataclasses). Configuration like `policy_len`, `inference_algo`, and learning
   flags is therefore passed at **construction time** in `create_agent`, not
   assigned afterwards.
5. **Batched observation convention.** Modern PyMDP uses a leading batch
   dimension. The interface adds/removes these automatically, so callers never
   see them (`_to_batch`, `_unwrap`, `_onehot_observations`).
