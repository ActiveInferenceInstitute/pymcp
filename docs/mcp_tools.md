# MCP Tools for PyMDP

This document details the MCP tools provided by the PyMDP-MCP server and shows how they map directly to the PyMDP API. Each tool is implemented by calling the corresponding PyMDP functions without any simplification or mocking.

## Agent Creation and Configuration

### `create_agent`

Creates a new active inference agent with the specified parameters.

**MCP Tool:**
```python
@mcp.tool()
def create_agent(name: str, generative_model: dict, inference_params: dict = None):
    """Create a new active inference agent."""
    # Extract parameters from generative_model
    A = convert_from_json(generative_model.get("A"))
    B = convert_from_json(generative_model.get("B"))
    C = convert_from_json(generative_model.get("C"))
    D = convert_from_json(generative_model.get("D", None))
    
    # Create PyMDP agent directly using the original Agent class
    agent = pymdp.agent.Agent(A=A, B=B, C=C, D=D, **inference_params or {})
    
    # Register agent in registry
    agent_registry.register_agent(name, agent)
    
    return {"status": "success", "agent_name": name}
```

**Corresponding PyMDP API:**
```python
from pymdp.agent import Agent

# Create an agent directly
agent = Agent(A=A, B=B, C=C, D=D, **inference_params)
```

### `define_generative_model`

Defines or updates the generative model for an existing agent.

**MCP Tool:**
```python
@mcp.tool()
def define_generative_model(agent_name: str, A: list = None, B: list = None, 
                           C: list = None, D: list = None):
    """Define or update the generative model for an agent."""
    agent = agent_registry.get_agent(agent_name)
    
    # Update the agent's generative model components directly
    if A is not None:
        agent.A = convert_from_json(A)
    if B is not None:
        agent.B = convert_from_json(B)
    if C is not None:
        agent.C = convert_from_json(C)
    if D is not None:
        agent.D = convert_from_json(D)
    
    return {"status": "success", "agent_name": agent_name}
```

**Corresponding PyMDP API:**
```python
# Update agent's generative model directly
agent.A = A
agent.B = B
agent.C = C
agent.D = D
```

## Inference and Planning

### `infer_states`

Performs inference on hidden states given an observation.

**MCP Tool:**
```python
@mcp.tool()
def infer_states(agent_name: str, observation: list, method: str = "FPI", 
                num_iter: int = 10, dF: float = 0.001, grad_descent: bool = False):
    """Infer hidden states given an observation."""
    agent = agent_registry.get_agent(agent_name)
    
    # Direct call to the agent's infer_states method
    qs = agent.infer_states(
        observation=observation,
        method=method,
        num_iter=num_iter,
        dF=dF,
        grad_descent=grad_descent
    )
    
    # Convert numpy arrays to lists for JSON serialization
    posterior_beliefs = [q.tolist() for q in qs]
    
    return {"posterior_beliefs": posterior_beliefs}
```

**Corresponding PyMDP API:**
```python
# Infer states directly
qs = agent.infer_states(
    observation=observation,
    method=method,
    num_iter=num_iter,
    dF=dF,
    grad_descent=grad_descent
)
```

### `infer_policies`

Computes the posterior over policies and their expected free energies.

**MCP Tool:**
```python
@mcp.tool()
def infer_policies(agent_name: str, use_utility: bool = True, 
                  use_states_info_gain: bool = True, use_param_info_gain: bool = False):
    """Infer policies based on expected free energy."""
    agent = agent_registry.get_agent(agent_name)
    
    # Direct call to the agent's infer_policies method
    q_pi, efe = agent.infer_policies(
        use_utility=use_utility,
        use_states_info_gain=use_states_info_gain,
        use_param_info_gain=use_param_info_gain
    )
    
    return {
        "policy_beliefs": q_pi.tolist(),
        "expected_free_energy": efe.tolist()
    }
```

**Corresponding PyMDP API:**
```python
# Infer policies directly
q_pi, efe = agent.infer_policies(
    use_utility=use_utility,
    use_states_info_gain=use_states_info_gain,
    use_param_info_gain=use_param_info_gain
)
```

### `sample_action`

Samples an action from the agent's policy distribution.

**MCP Tool:**
```python
@mcp.tool()
def sample_action(agent_name: str, modalities: list = None):
    """Sample an action from the agent's policy distribution."""
    agent = agent_registry.get_agent(agent_name)
    
    # Direct call to the agent's sample_action method
    action = agent.sample_action(modalities=modalities)
    
    if isinstance(action, list):
        return {"action": action}
    else:
        return {"action": [action]}
```

**Corresponding PyMDP API:**
```python
# Sample action directly
action = agent.sample_action(modalities=modalities)
```

## Environment Interaction

### `step_environment`

Updates the environment given an agent's action.

**MCP Tool:**
```python
@mcp.tool()
def step_environment(env_name: str, action: list):
    """Update the environment given an action."""
    env = environment_registry.get_environment(env_name)
    
    # Direct call to the environment's step method
    observation, reward, done, info = env.step(action)
    
    return {
        "observation": observation.tolist() if isinstance(observation, np.ndarray) else observation,
        "reward": float(reward) if isinstance(reward, (np.ndarray, np.float_)) else reward,
        "done": done,
        "info": info
    }
```

**Corresponding PyMDP API:**
```python
# Step environment directly
observation, reward, done, info = env.step(action)
```

### `reset_environment`

Resets the environment to its initial state.

**MCP Tool:**
```python
@mcp.tool()
def reset_environment(env_name: str):
    """Reset the environment to its initial state."""
    env = environment_registry.get_environment(env_name)
    
    # Direct call to the environment's reset method
    observation = env.reset()
    
    return {
        "observation": observation.tolist() if isinstance(observation, np.ndarray) else observation
    }
```

**Corresponding PyMDP API:**
```python
# Reset environment directly
observation = env.reset()
```

## Advanced Features

### `calculate_free_energy`

Calculates the variational free energy for a given belief and observation.

**MCP Tool:**
```python
@mcp.tool()
def calculate_free_energy(agent_name: str, beliefs: list, observation: list):
    """Calculate variational free energy for given beliefs and observation."""
    agent = agent_registry.get_agent(agent_name)
    
    # Convert JSON to NumPy arrays
    beliefs_np = convert_from_json(beliefs)
    
    # Use PyMDP's mathematical functions directly
    F = pymdp.maths.calculate_free_energy(
        beliefs_np, 
        agent.A, 
        agent.B, 
        agent.C, 
        agent.D, 
        observation
    )
    
    return {"free_energy": float(F)}
```

**Corresponding PyMDP API:**
```python
from pymdp.maths import calculate_free_energy

# Calculate free energy directly
F = calculate_free_energy(beliefs, A, B, C, D, observation)
```

### `run_tmaze_example`

Runs the T-maze example from the PyMDP repository.

**MCP Tool:**
```python
@mcp.tool()
def run_tmaze_example(reward_location: str = "left", agent_type: str = "sophisticated", 
                     num_trials: int = 10):
    """Run the T-maze example from PyMDP examples."""
    import pymdp.examples as examples
    
    # Use the example directly from PyMDP
    results = examples.tmaze.run_tmaze_demo(
        reward_location=reward_location,
        agent_type=agent_type,
        num_trials=num_trials
    )
    
    return {
        "trial_results": results["trial_results"],
        "agent_beliefs": [q.tolist() for q in results["agent_beliefs"]]
    }
```

**Corresponding PyMDP Example:**
```python
from pymdp.examples import tmaze

# Run example directly
results = tmaze.run_tmaze_demo(
    reward_location=reward_location,
    agent_type=agent_type,
    num_trials=num_trials
)
```

## Integration with PyMDP JAX Support

### `enable_jax`

Enables JAX acceleration for an agent.

**MCP Tool:**
```python
@mcp.tool()
def enable_jax(agent_name: str):
    """Enable JAX acceleration for an agent."""
    agent = agent_registry.get_agent(agent_name)
    
    # Import JAX module from PyMDP
    import pymdp.jax as jax_pymdp
    
    # Convert agent to JAX-enabled version
    jax_agent = jax_pymdp.convert_to_jax_agent(agent)
    
    # Update registry with JAX agent
    agent_registry.register_agent(agent_name, jax_agent)
    
    return {"status": "success", "jax_enabled": True}
```

**Corresponding PyMDP API:**
```python
from pymdp.jax import convert_to_jax_agent

# Enable JAX directly
jax_agent = convert_to_jax_agent(agent)
```

## Summary

As demonstrated above, all MCP tools directly map to PyMDP API calls without any simplification or mocking. The MCP server provides a thin wrapper around the PyMDP library, ensuring that the behavior and results are identical to using PyMDP directly.

For a complete reference of the PyMDP API, see the [PyMDP documentation](https://pymdp.readthedocs.io/). 