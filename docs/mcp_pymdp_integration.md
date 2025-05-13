# MCP Server for PyMDP: Integration Documentation

## What is PyMDP-MCP?

PyMDP-MCP is a fully-featured implementation of the [Model Context Protocol (MCP)](https://modelcontextprotocol.io) server that provides access to the [PyMDP](https://github.com/infer-actively/pymdp) active inference framework. This integration allows AI assistants and other MCP clients to create, simulate, and analyze active inference agents through a standardized interface.

## Core Functionality

This server provides a bridge between the MCP specification and PyMDP's implementation of active inference for Markov Decision Processes. It exposes PyMDP's core functionality through MCP tools that can be called by any compatible client.

### PyMDP Features Available Through MCP

This MCP server provides access to the following PyMDP capabilities:

1. **Agent Creation and Configuration**
   - Create active inference agents with customizable parameters
   - Define and modify generative models (A, B, C, D matrices)
   - Configure inference parameters (e.g., inference methods, planning horizon)

2. **Inference and Learning**
   - Perform variational inference on hidden states
   - Compute expected free energy
   - Optimize policies
   - Sample actions based on posterior beliefs
   - Update beliefs through message passing

3. **Environment Interaction**
   - Simulate agent-environment interactions
   - Sample observations from generative models
   - Track agent performance and belief dynamics

4. **Visualization and Analysis**
   - Generate plots of belief dynamics
   - Visualize agent behavior in environments
   - Analyze information-theoretic metrics

## Implementation Details

### MCP Tool Integration

All PyMDP functionality is exposed as MCP tools, following the MCP specification. These tools can be called by any MCP-compatible client, such as Claude or other AI assistants.

### Authentication and Access Control

The server implements appropriate authentication and access control mechanisms as per the MCP specification.

### Error Handling

The server provides detailed error messages and appropriate error codes for all possible failure cases, following the MCP specification.

## Connection to Original PyMDP Framework

This MCP server is built directly on top of the official [PyMDP library](https://github.com/infer-actively/pymdp), without any mocking or simplification of methods. It provides complete access to PyMDP's functionality for active inference in discrete state spaces, as described in their [JOSS paper](https://joss.theoj.org/papers/10.21105/joss.04098).

The server maintains compatibility with PyMDP's API and implementation details, ensuring that all models and simulations are fully compatible with the standalone library.

## Use Cases

### Research and Education

- Explore active inference models without writing code
- Test hypotheses about agent behavior under different conditions
- Teach active inference concepts through interactive demonstrations

### Applied AI Systems

- Prototype active inference agents for specific tasks
- Integrate belief-based planning into existing systems
- Develop hybrid systems combining active inference with other approaches

### Cognitive Science and Neuroscience

- Model perception, learning, and decision-making
- Test theories of brain function based on free energy principles
- Simulate experiments with artificial agents

## References

- [PyMDP GitHub Repository](https://github.com/infer-actively/pymdp)
- [PyMDP Documentation](https://pymdp.readthedocs.io/)
- [Model Context Protocol Specification](https://modelcontextprotocol.io)
- Heins, C., Millidge, B., Demekas, D., Klein, B., Friston, K., Couzin, I. D., & Tschantz, A. (2022). pymdp: A Python library for active inference in discrete state spaces. Journal of Open Source Software, 7(73), 4098.

# MCP-PyMDP Integration

## Overview

MCP-PyMDP integrates the Message-based Cognitive Protocol (MCP) server with the PyMDP active inference library, providing a standardized interface for building, deploying, and interacting with active inference agents.

## Architecture

The integration consists of these primary components:

1. **MCP Server**: Implemented in `src/main.py` using the FastMCP framework, exposing PyMDP functionality as MCP tools
2. **PyMDP Interface**: The `PyMDPInterface` class in `src/mcp/utils.py` provides a bridge between MCP tools and PyMDP functionality
3. **MCP Client**: Implemented in `src/mcp/client.py`, providing both network-based and direct client implementations for interacting with the MCP server
4. **Environments**: Implementations of standard environments for agent simulations in `src/mcp/environments/`
5. **Visualization**: Tools for visualizing active inference models and simulations in `src/mcp/visualization.py`

## Implementation Details

### PyMDP Interface

The `PyMDPInterface` class manages:

- Creating and storing agents with their generative models (A, B, C, D matrices)
- Managing environments and their state
- Performing inference and action selection
- Running simulations and storing their history
- Providing visualization tools for belief dynamics and free energy

### MCP Tools

The MCP server exposes the following core tools:

- `create_agent`: Create an active inference agent with specified generative model parameters
- `define_generative_model`: Define random A and B matrices based on specified dimensions
- `infer_states`: Perform state inference given an observation
- `infer_policies`: Optimize policies based on expected free energy
- `sample_action`: Sample an action from the agent's policy distribution
- `create_grid_world_env`: Create a simple grid world environment
- `step_environment`: Update an environment given an agent's action
- `run_simulation`: Run a simulation for a specified number of timesteps
- `visualize_simulation`: Generate visualization of a simulation

### Client Implementation

Two client implementations are provided:

1. **MCPClient**: Network-based client for communicating with a remote MCP server
2. **DirectMCPClient**: Direct client that calls MCP tools without going through a server, useful for testing and development

## Data Flow

1. Client sends a request to the MCP server with a tool name and parameters
2. Server processes the request and calls the corresponding method in the PyMDPInterface
3. PyMDPInterface interacts with PyMDP to perform the requested operation
4. Results are serialized to JSON and returned to the client
5. Output files (visualizations, logs) are saved to a timestamped directory

## Advanced Features

The integration provides several advanced features beyond basic active inference:

- Belief dynamics visualization for tracking agent beliefs over time
- Free energy analysis tools for diagnosing agent performance
- Support for custom environments beyond the standard grid world
- Asynchronous operation for running multiple simulations in parallel
- Session management for maintaining state across multiple tool calls

## Performance Considerations

The implementation includes several optimizations:

- JSON serialization/deserialization of NumPy arrays for efficient network transmission
- Caching of intermediate results to avoid redundant computations
- Support for both direct and network-based client operation to minimize overhead when appropriate
- Session management to maintain state between calls and reduce initialization costs

## Running Examples

The project includes several example scripts in the `scripts` directory that demonstrate how to use the MCP-PyMDP integration. All scripts must be run using `python3` (not `python`).

### Starting the MCP Server

```bash
# Navigate to the scripts directory
cd scripts

# Start the server in the background
python3 start_mcp_server.py --port 8090 --output-dir /tmp/mcp_test_output > /tmp/mcp_server.log 2>&1 &
```

### Running Individual Examples

```bash
# Grid World Examples
python3 mcp_gridworld_examples.py --port 8090 --output-dir /tmp/gridworld_examples

# PyMDP Basic Example
python3 mcp_pymdp_example.py --port 8090 --output-dir /tmp/pymdp_example

# Fixed PyMCP Examples
python3 pymcp_examples_fixed.py --port 8090 --output-dir /tmp/pymcp_examples_fixed
```

### Running All Examples

The `run_all_examples.py` script will start a server, run all the example scripts, and then shut down the server:

```bash
python3 run_all_examples.py --port 8090 --output-dir /tmp/all_examples
```

Example outputs are saved to the specified output directory or to timestamped directories under `scripts/output/` by default. 