# MCP Tools Reference

MCP, REST and `DirectMCPClient` dispatch through the explicit
[`TOOL_NAMES` registry](../../src/mcp/api.py). Discovery returns strict native
JSON argument schemas. Unknown tools, extra fields and invalid types are
rejected; arbitrary public attributes are not remotely callable. Separate
server/client instances share this contract, not mutable state.

## Agent creation and model definition

`create_agent(name, generative_model, normalize=False)` accepts native,
unbatched families of arrays:

- `A`: likelihood matrices `[outcome, state]`.
- `B`: transition matrices `[next_state, current_state, action]`.
- `C`: finite signed outcome utilities, not probability distributions.
- `D`: normalized state priors.
- Optional `pA`/`pB`: finite nonnegative concentration arrays with the
  corresponding shapes and supported columns.

`A` and `B` are required. Shape, finite-value, support and allocation limits
are checked before construction. Normalization is explicit, never silently
applied to invalid input. Admission and exported copies do not mutate the
caller's arrays. `get_agent(agent_id)` returns the complete unbatched model,
settings and dimension metadata, including singleton dimensions.

Settings are admitted at construction by [`models.py`](../../src/mcp/models.py).
Filtering supports `fpi` and single-factor `exact`; unsupported smoothing,
algorithms and per-call method/horizon changes fail rather than being ignored.
Passing learning configuration or concentrations does not expose a learning
update API.

`define_generative_model(A_dims, B_dims, seed=None)` constructs bounded
normalized models. `create_gridworld_agent(name, grid_size, reward_positions,
...)` constructs the position/reward model matching that grid. Use it with
`create_grid_world_env` of the same dimensions and rewards; an arbitrary
two-state model cannot observe a nine-state grid.

## Observation, action and reset lifecycle

1. Obtain an actual environment observation. `infer_states(agent_id,
   observation)` requires exactly one integer outcome per declared modality.
2. `infer_policies(agent_id)` requires inferred beliefs; it never invents
   an observation. It retains the complete policy posterior and the native
   negative expected-free-energy vector, with its sign-reversed EFE.
3. `sample_action(agent_id)` returns an action and an `action_id`.
   Repeated selection returns the pending action without advancing the prior.
4. `step_environment(env_id, action, agent_id, action_id)` commits that action
   only after a successful matching transition. A caller executing a real
   external transition instead uses `commit_action(agent_id, action_id)`.
   Commit advances the predictive prior exactly once; do not do both.
5. `reset_agent` restores `D` and invalidates pending tokens.
   `reset_environment` resets the physical environment; it is a distinct
   operation, not a replacement for resetting the agent.

Grid actions are up/right/down/left/stay. Grid reset defaults to the origin;
custom reset defaults to its first state. Explicit `reset_mode="configured"`
uses the configured start. `terminate_on_reward` controls the terminal flag;
`run_simulation`/`run_session` retain fixed-step execution unless
`stop_on_done=True` is explicitly requested. Empty `reward_positions=[]`
remains empty. These defaults are not new scientific policy decisions.

## Runs and sessions

`run_simulation(agent_id, env_id, num_steps=10, save_history=True,
stop_on_done=False)` returns an actual `simulation_id`, executed `num_steps`,
requested `requested_steps`, rewards and measured history. Records distinguish
pre-transition observations, beliefs and physical state from next-state fields;
trajectory plots include the final endpoint.

`create_session(session_id, agent_id, env_id)` binds existing resources.
Repeating the same binding is idempotent; conflicting reuse fails.
`run_session` continues that session and retains cumulative history and actual
run IDs. State is process-local, bounded and lost on restart; no distributed
or durable session guarantee is provided.

## Measurements and plots

`calculate_free_energy(A_matrix, prior, observation, q_states)` uses
`F = KL(q || prior) - E_q[log p(observation | state)]`.
The entropy contribution `sum(q * log(q))` is signed, not positive entropy.
`infer_states_from_observation` computes the corresponding stable Bayes
posterior and evidence. Zero support is rejected or explicitly reported as
infinite/unavailable; nonfinite measurements are not fabricated as finite JSON.

`visualize_agent_model`, `visualize_simulation`,
`visualize_belief_dynamics`, `plot_belief_dynamics` and `analyze_free_energy`
consume actual model/history data. No arbitrary risk/ambiguity decomposition
or synthetic free-energy fallback is supplied. The model overview labels its
first-two-entry selection and B control 0; model JSON remains complete.

Server output confinement applies to primary files and every reported
companion, including symlink parents and existing symlink files. Trusted local
core output and client-side downloads are separate custody boundaries.
HTTP PNG/SVG responses contain image bytes with matching content types; request
`format=svg` explicitly when downloading to an SVG filename.

Core allocation, steps, retained runs/logs, protocol bytes/channels/queues and
animation frames are bounded by trusted launch configuration. Expired detail
is reported unavailable, never borrowed from a later run. These caps are not
measured RSS, latency or throughput guarantees.

## Registry inventory

| Group | Tools |
| --- | --- |
| Models | `create_agent`, `define_generative_model`, `create_gridworld_agent`, `validate_generative_model`, `get_agent`, `get_all_agents` |
| Filtering/control | `infer_states`, `infer_policies`, `sample_action`, `commit_action`, `reset_agent` |
| Environments | `create_grid_world_env`, `step_environment`, `reset_environment`, `get_environment`, `get_all_environments` |
| Runs | `run_simulation`, `get_simulation`, `get_all_simulations`, `get_computation_logs` |
| Sessions | `create_session`, `get_session`, `get_session_history`, `run_session`, `get_all_sessions` |
| Plots | `visualize_agent_model`, `visualize_simulation`, `visualize_belief_dynamics`, `plot_belief_dynamics`, `analyze_free_energy` |
| Numerical helpers | `calculate_free_energy`, `infer_states_from_observation` |
| Discovery | `get_all_functions` |

Delete/update CRUD, comparison, distributed execution, custom inference and
learning-update tools are not offered. Consult actual discovery for argument
schemas rather than treating the full upstream
[PyMDP API](https://pymdp-rtd.readthedocs.io/en/latest/) as this wrapper's API.

## Transport envelopes

REST `POST /tools/create_agent` accepts an object, not a JSON-string model:

```json
{"params":{"name":"one","generative_model":{"A":[[[1]]],"B":[[[[1]]]],"D":[[1]]},"normalize":false}}
```

HTTP success has one `result` envelope; `MCPClient` unwraps it once.
MCP uses JSON-RPC initialization followed by `notifications/initialized`,
`tools/list` and `tools/call`, with protocol version `2025-03-26`.
Stdio stdout contains protocol frames only. SSE opens a bodyless `/sse`
channel and posts to its `/messages?session_id=...` endpoint; responses are
delivered once on that channel. Disconnect releases its budget even when
the consumer blocks an ASGI send. See [setup](../guides/setup.md) for launch boundaries.

