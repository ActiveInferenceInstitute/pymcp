# MCP Tools Reference

This document describes the tools that this project actually exposes. Every tool
is a thin wrapper over a method on `PyMDPInterface` (`src/mcp/utils.py`), which
calls the real PyMDP library. The two servers — the MCP tools in `src/main.py`
(delivered over SSE or stdio) and the REST API in `src/mcp/server/app.py` — both
route to these same methods, so their behavior is identical.

> **Note on scope.** Earlier drafts of these docs listed tools such as
> `enable_jax`, `run_tmaze_example`, `calculate_kl_divergence`,
> `calculate_entropy`, `set_temporal_horizon`, `deep_planning`,
> `register_environment`, `connect_agent_to_environment`, `enable_learning`,
> `update_model`, `create_simulation`, `analyze_simulation`, `jax_infer_states`
> and `run_learning_example`. **None of those exist in the code.** The only tools
> available are the `PyMDPInterface` methods below.

## The core methods

### Agent creation and model definition

#### `create_agent(name, generative_model)`

Create an active inference agent from a generative-model dict containing `A`
(observation likelihood) and `B` (state transitions), with optional `C` (prior
preferences) and `D` (prior beliefs). `A` and `B` are required.

```python
pymdp.create_agent("agent_1", {
    "A": [[...]],   # observation model
    "B": [[...]],   # transition model
    "C": [[...]],   # optional priors over outcomes
    "D": [[...]],   # optional priors over states
})
```

Returns agent metadata (number of observation modalities, state factors, and
controls).

#### `define_generative_model(A_dims, B_dims)`

Build **random** `A` and `B` matrices from dimension lists, plus default flat `C`
and uniform `D` priors.

- `A_dims`: list of `[obs_dim, state_dim]` per observation modality.
- `B_dims`: list of `[state_to, state_from, action_dim]` per state factor.

```python
model = pymdp.define_generative_model(
    A_dims=[[3, 4]],
    B_dims=[[4, 4, 4]],
)
```

#### `create_gridworld_agent(name, grid_size, reward_positions, ...)`

Create an agent tailored to a gridworld: a position observation (identity map of
the grid) plus a reward observation, with strong preference for reward cells.

```python
pymdp.create_gridworld_agent(
    "agent_1",
    grid_size=[3, 3],
    reward_positions=[[2, 2]],
    action_precision=1.0,
    inference_horizon=5,
)
```

### Environments

#### `create_grid_world_env(name, grid_size, reward_positions)`

Create a gridworld environment. Returns environment metadata.

```python
env = pymdp.create_grid_world_env("world_1", [3, 3], [[2, 2]])
```

#### `step_environment(env_id, action)`

Advance an environment by one step given an action index (or list of indices).
Returns the new observation, reward, and status.

```python
result = pymdp.step_environment("world_1", [1])
```

#### `reset_environment(env_id)`

Reset an environment to its initial state and return the initial observation.

```python
pymdp.reset_environment("world_1")
```

### Inference and action selection

#### `infer_states(agent_id, observation, method="FPI")`

Compute a posterior over hidden states given an observation. `method` selects the
message-passing algorithm PyMDP uses (`FPI`, `BP`, etc.).

```python
result = pymdp.infer_states("agent_1", [0, 2])
# -> {"posterior_states": [[...], ...]}
```

#### `infer_policies(agent_id, planning_horizon=None)`

Optimize the belief over policies by expected free energy, returning the policy
posterior and expected free energy.

```python
result = pymdp.infer_policies("agent_1")
```

#### `sample_action(agent_id, planning_horizon=None)`

Sample an action from the agent's policy posterior.

```python
result = pymdp.sample_action("agent_1")
```

### Simulation

#### `run_simulation(agent_id, env_id, num_steps=10, ...)`

Run the full perception–action loop: infer states, infer policies, sample an
action, step the environment, and record the history (`observations`, `actions`,
`rewards`, `states`, `policies`, `free_energies`).

```python
result = pymdp.run_simulation("agent_1", "world_1", num_steps=10)
```

### Introspection

#### `get_agent(name)` / `get_environment(name)`

Retrieve a stored agent or environment as serializable JSON.

#### `get_all_functions()`

Return the list of all public methods available on the interface.

```python
pymdp.get_all_functions()
# -> {"functions": ["calculate_free_energy", "create_agent", ...]}
```

## Additional helpers

The interface also exposes a few non-core helpers used for analysis:

| Method | Description |
| --- | --- |
| `validate_generative_model(model, check_normalization=True)` | Validate structure and normalization of `A`/`B` matrices. |
| `calculate_free_energy(A_matrix, prior, observation, q_states)` | Compute variational free energy (accuracy/complexity decomposition). |
| `infer_states_from_observation(A_matrix, prior, observation)` | Single-step posterior via Bayes' rule (`pymdp.maths.get_joint_likelihood`). |
| `get_all_agents` / `get_all_environments` / `get_all_simulations` | List everything stored in the interface registries. |

Visualization methods (`visualize_belief_dynamics`, `plot_belief_dynamics`,
`analyze_free_energy`, `visualize_simulation`) are attached to the interface by
`src/mcp/visualization/extensions.py`.

## How a tool call flows

1. A client calls a tool (over MCP SSE/stdio, or an HTTP endpoint).
2. The server looks up the corresponding `PyMDPInterface` method.
3. `PyMDPInterface` calls the unmodified PyMDP API.
4. Results (NumPy object arrays) are converted to plain JSON via `NumpyEncoder`
   and `_convert_from_obj_array`, and returned to the client.

For the full underlying PyMDP API, see the [PyMDP documentation](https://pymdp.readthedocs.io/).
