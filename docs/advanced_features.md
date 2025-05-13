# Advanced PyMDP Features via MCP

This guide covers advanced PyMDP features available through the MCP server, demonstrating how the integration provides full access to PyMDP's capabilities.

## Custom Inference Methods

PyMDP provides multiple inference algorithms for different use cases. The MCP server exposes all of these options through the appropriate parameters:

```python
# Using variational message passing for inference
mcp.call("pymdp", "infer_states", {
    "agent_name": "my_agent",
    "observation": [0, 2],
    "method": "VMP",
    "num_iter": 10,
    "dF": 0.001
})

# Using marginal message passing for inference
mcp.call("pymdp", "infer_states", {
    "agent_name": "my_agent",
    "observation": [0, 2],
    "method": "MMP",
    "num_iter": 10,
    "dF": 0.001
})
```

## Information-Theoretic Metrics

PyMDP calculates various information-theoretic measures that are useful for analyzing agent behavior. These are accessible through MCP tools:

```python
# Calculate the Kullback-Leibler divergence between beliefs
kl_div = mcp.call("pymdp", "calculate_kl_divergence", {
    "agent_name": "my_agent",
    "beliefs_q": prior_beliefs,
    "beliefs_p": posterior_beliefs
})

# Calculate entropy of beliefs
entropy = mcp.call("pymdp", "calculate_entropy", {
    "agent_name": "my_agent",
    "beliefs": agent_beliefs
})
```

## Temporal Deep Active Inference

The MCP server supports PyMDP's implementation of temporal deep active inference, where agents can plan over multiple time steps:

```python
# Set up a temporal horizon for planning
mcp.call("pymdp", "set_temporal_horizon", {
    "agent_name": "my_agent",
    "horizon": 5
})

# Perform deep planning
planning_results = mcp.call("pymdp", "deep_planning", {
    "agent_name": "my_agent",
    "current_observation": [0, 1]
})
```

## Custom Environment Integration

The MCP server can connect PyMDP agents to custom environments:

```python
# Register a custom environment
mcp.call("pymdp", "register_environment", {
    "env_name": "my_custom_env",
    "observation_space": [3, 4],
    "action_space": [2, 2]
})

# Connect an agent to the environment
mcp.call("pymdp", "connect_agent_to_environment", {
    "agent_name": "my_agent",
    "env_name": "my_custom_env"
})
```

## Learning of Generative Models

PyMDP supports learning of generative models through Bayesian inference. The MCP server exposes these capabilities:

```python
# Enable learning of the A matrix
mcp.call("pymdp", "enable_learning", {
    "agent_name": "my_agent",
    "parameter": "A",
    "learning_rate": 0.1
})

# Update generative model based on experience
mcp.call("pymdp", "update_model", {
    "agent_name": "my_agent",
    "observation": [1, 0],
    "action": [0, 1]
})
```

## Complex Simulation Scenarios

The MCP server supports setting up and running complex PyMDP simulations:

```python
# Set up a multi-agent simulation
simulation_id = mcp.call("pymdp", "create_simulation", {
    "agent_names": ["agent1", "agent2"],
    "environment": "social_env",
    "duration": 100
})

# Run simulation
results = mcp.call("pymdp", "run_simulation", {
    "simulation_id": simulation_id
})

# Analyze results
analysis = mcp.call("pymdp", "analyze_simulation", {
    "simulation_id": simulation_id,
    "metrics": ["free_energy", "accuracy", "complexity"]
})
```

## JAX Acceleration

PyMDP includes experimental JAX support for accelerated computation. The MCP server provides access to these features:

```python
# Enable JAX acceleration for an agent
mcp.call("pymdp", "enable_jax", {
    "agent_name": "my_agent"
})

# Run accelerated inference
results = mcp.call("pymdp", "jax_infer_states", {
    "agent_name": "my_agent",
    "observation": [0, 2]
})
```

## Integration with PyMDP Examples

The MCP server includes all standard PyMDP examples, accessible through dedicated tools:

```python
# Run the T-maze example
tmaze_results = mcp.call("pymdp", "run_tmaze_example", {
    "reward_location": "left",
    "agent_type": "sophisticated"
})

# Run the agent learning example
learning_results = mcp.call("pymdp", "run_learning_example", {
    "learning_rate": 0.1,
    "num_trials": 50
})
```

These examples match the functionality available in the PyMDP repository examples directory, providing identical results.

# Advanced PyMDP-MCP Features

This document covers the advanced features available in the PyMDP-MCP integration, based on the actual implementation in the source code.

## Belief Dynamics Visualization

The implementation includes specialized visualization tools for tracking belief dynamics over time:

```python
# Visualize belief dynamics for a session
result = await client.call_tool("visualize_belief_dynamics", {
    "session_id": "session_1234",
    "output_file": "beliefs.png"
})
```

This generates a detailed heatmap visualization showing how agent beliefs evolved during a simulation, with separate plots for each state factor.

## Free Energy Analysis

Advanced analysis of free energy components is available through dedicated visualization tools:

```python
# Analyze free energy components from a simulation
result = await client.call_tool("analyze_free_energy", {
    "session_id": "session_1234",
    "output_file": "free_energy.png"
})
```

This produces both visualizations and detailed metrics about the agent's free energy minimization process.

## State Inference Methods

Multiple state inference algorithms are supported, with the ability to configure detailed parameters:

```python
# Infer states using various methods
result = await client.call_tool("infer_states", {
    "agent_name": "agent_1",
    "observation": [0, 2],
    "method": "FPI",  # Options: FPI, VMP, MMP, BP
    "num_iter": 10,
    "dF": 0.001
})
```

## Session Management

The implementation includes a comprehensive session management system for maintaining state across multiple interactions:

```python
# Run a simulation with session tracking
result = await client.call_tool("run_simulation", {
    "agent_id": "agent_1",
    "env_id": "env_1",
    "num_timesteps": 10,
    "save_history": true
})

# Session ID is returned for future reference
session_id = result["session_id"]

# Use session ID to query results later
beliefs = await client.call_tool("get_belief_history", {
    "session_id": session_id
})
```

## Custom Environment Creation

The implementation supports creating customized environments beyond the standard grid world:

```python
# Create a custom environment
result = await client.call_tool("create_custom_env", {
    "name": "custom_1",
    "observation_space": [3, 5],
    "action_space": [4],
    "initial_state": [0, 2],
    "transition_dynamics": "custom_dynamics_json"
})
```

## Direct vs. Network Client

The implementation provides both direct and network-based client implementations:

```python
# Direct client for local operation
direct_client = DirectMCPClient(output_dir="./output")
result = await direct_client.call_tool("create_agent", 
    name="direct_agent",
    generative_model={"A": [...], "B": [...]}
)

# Network client for remote operation
network_client = MCPClient(host="remote-server", port=8080)
result = await network_client.call_tool("create_agent", 
    name="remote_agent",
    generative_model={"A": [...], "B": [...]}
)
```

## Computation Logging

Detailed logging of computational processes is available:

```python
# Enable detailed computation logging
result = await client.call_tool("set_debug_mode", {
    "enabled": true
})

# Run inference with detailed logging
result = await client.call_tool("infer_states", {
    "agent_name": "agent_1",
    "observation": [0, 1],
    "save_computation": true
})

# Retrieve computation details
logs = await client.call_tool("get_computation_logs", {
    "agent_name": "agent_1"
})
```

## Output Directory Management

The implementation includes automatic creation and management of timestamped output directories:

```python
# Client automatically creates a timestamped output directory
client = MCPClient(output_dir="auto")

# Directory path follows a standardized format
# Example: scripts/output/pymcp_examples_YYYYMMDD_HHMMSS/
```

All visualizations, logs, and data are saved to these directories with appropriate timestamps.

## Free Energy Calculation

Direct calculation of free energy components is available for advanced analysis:

```python
# Calculate free energy with decomposition into components
result = await client.call_tool("calculate_free_energy", {
    "A_matrix": A_matrix_list,
    "prior": prior_belief_list,
    "observation": [1, 0, 2],
    "q_states": posterior_list
})

# Returns accuracy and complexity terms separately
accuracy = result["accuracy"]
complexity = result["complexity"]
free_energy = result["free_energy"]
```

## Asynchronous Operation

The implementation is fully asynchronous, supporting concurrent operations:

```python
# Run multiple operations concurrently
import asyncio

async def run_concurrent_operations():
    tasks = [
        client.call_tool("infer_states", {"agent_name": "agent_1", "observation": [0, 1]}),
        client.call_tool("infer_states", {"agent_name": "agent_2", "observation": [2, 0]}),
        client.call_tool("step_environment", {"env_name": "env_1", "action": [1]})
    ]
    results = await asyncio.gather(*tasks)
    return results
``` 