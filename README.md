# PyMCP a.k.a. PyMDP-MCP: Active Inference and Markov Decision Processes in Python (PyMDP) for Agents with Model Context Protocol (MCP)

<p align="center">
  <img src="public/PyMDP-MCP.png" alt="PyMDP-MCP Integration" width="600">
</p>

A full-featured implementation of the [Model Context Protocol (MCP)](https://modelcontextprotocol.io) server for Active Inference and Markov Decision Processes, enabling AI agents to perform simulations using the complete, unmodified [PyMDP](https://github.com/infer-actively/pymdp) framework.

Developed initially by [Daniel Friedman](https://github.com/docxology) at the Active Inference Institute, building off of the PyMDP package.

## Overview

This project implements a comprehensive MCP server that provides AI agents with direct access to the PyMDP library's active inference capabilities. The server includes the full PyMDP codebase (in the `pymdp-clone` directory) and exposes its functionality through MCP tools, allowing clients to create and simulate active inference agents, define generative models, run inference, and optimize policies.

The implementation aims to follow the best practices laid out by Anthropic for building MCP servers, allowing seamless integration with any MCP-compatible client.

## PyMDP Integration

This MCP server uses the complete, unmodified PyMDP library directly from its [official GitHub repository](https://github.com/infer-actively/pymdp). Key aspects of this integration:

- No mock methods or simplified implementations
- Direct calls to PyMDP's API
- Identical behavior to using PyMDP directly
- Support for all PyMDP features, including advanced capabilities

For detailed information about the integration, see the [PyMDP Integration documentation](docs/pymdp_integration.md).

## Features

The server provides comprehensive access to PyMDP functionality:

1. **`create_agent`**: Create an active inference agent with specified parameters
2. **`define_generative_model`**: Define A, B, and C matrices for the generative model
3. **`infer_states`**: Infer hidden states given an observation
4. **`infer_policies`**: Optimize policies based on expected free energy
5. **`sample_action`**: Sample an action from the agent's policy distribution
6. **`step_environment`**: Update the environment given an agent's action
7. **`visualize_simulation`**: Generate visualizations of agent simulations

For a complete list of available tools and their direct mappings to PyMDP API calls, see the [MCP Tools documentation](docs/mcp_tools.md).

## Prerequisites

- Python 3.12+
- PyMDP and required scientific computing packages (installed automatically)
- Docker if running the MCP server as a container (recommended)

## Installation

### Using uv

1. Install uv if you don't have it:
   ```bash
   pip install uv
   ```

2. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/pymdp-mcp.git
   cd pymdp-mcp
   ```

3. Install dependencies:
   ```bash
   uv pip install -e .
   ```

4. Create a `.env` file based on `.env.example`:
   ```bash
   cp .env.example .env
   ```

### Using Docker (Recommended)

1. Build the Docker image:
   ```bash
   docker build -t pymdp/mcp --build-arg PORT=8050 .
   ```

2. Create a `.env` file based on `.env.example`

## Configuration

The following environment variables can be configured in your `.env` file:

| Variable | Description | Example |
|----------|-------------|----------|
| `TRANSPORT` | Transport protocol (sse or stdio) | `sse` |
| `HOST` | Host to bind to when using SSE transport | `0.0.0.0` |
| `PORT` | Port to listen on when using SSE transport | `8050` |

## Running the Server

### Using uv

#### SSE Transport

```bash
# Set TRANSPORT=sse in .env then:
uv run src/main.py
```

Or use the starter script in the scripts directory:

```bash
cd scripts
python start_mcp_server.py
```

The MCP server will run as an API endpoint that you can connect to with the configuration shown below.

#### Stdio Transport

With stdio, the MCP client itself can spin up the MCP server, so nothing to run at this point.

### Using Docker

#### SSE Transport

```bash
docker run --env-file .env -p:8050:8050 pymdp/mcp
```

The MCP server will run as an API endpoint within the container that you can connect to with the configuration shown below.

#### Stdio Transport

With stdio, the MCP client itself can spin up the MCP server container, so nothing to run at this point.

## Integration with MCP Clients

### SSE Configuration

Once you have the server running with SSE transport, you can connect to it using this configuration:

```json
{
  "mcpServers": {
    "pymdp": {
      "transport": "sse",
      "url": "http://localhost:8050/sse"
    }
  }
}
```

> **Note for Windsurf users**: Use `serverUrl` instead of `url` in your configuration:
> ```json
> {
>   "mcpServers": {
>     "pymdp": {
>       "transport": "sse",
>       "serverUrl": "http://localhost:8050/sse"
>     }
>   }
> }
> ```

Make sure to update the port if you are using a value other than the default 8050.

### Python with Stdio Configuration

Add this server to your MCP configuration for Claude Desktop, Windsurf, or any other MCP client:

```json
{
  "mcpServers": {
    "pymdp": {
      "command": "your/path/to/pymdp-mcp/.venv/Scripts/python.exe",
      "args": ["your/path/to/pymdp-mcp/src/main.py"],
      "env": {
        "TRANSPORT": "stdio"
      }
    }
  }
}
```

### Docker with Stdio Configuration

```json
{
  "mcpServers": {
    "pymdp": {
      "command": "docker",
      "args": ["run", "--rm", "-i", 
               "-e", "TRANSPORT=stdio", 
               "pymdp/mcp"],
      "env": {
        "TRANSPORT": "stdio"
      }
    }
  }
}
```

## Using the MCP Tools

### Creating an Agent

```
create_agent(name="foraging_agent", generative_model={"A": A_json, "B": B_json, "C": C_json, "D": D_json})
```

### Inferring States

```
infer_states(agent_name="foraging_agent", observation=[1, 2])
```

### Inferring Policies

```
infer_policies(agent_name="foraging_agent")
```

### Sampling an Action

```
sample_action(agent_name="foraging_agent")
```

## Supported Models and Environments

The server supports all models and environments available in PyMDP:
1. Discrete state-space active inference models
2. Customizable generative models with arbitrary factor dimensions
3. Standard POMDP and MDP environments
4. All example environments from the PyMDP repository

## Documentation

For comprehensive documentation, see the `/docs` directory:

- [PyMDP-MCP Integration Overview](docs/mcp_pymdp_integration.md)
- [MCP Tools and PyMDP API Mapping](docs/mcp_tools.md)
- [Implementation Details](docs/implementation_details.md)
- [Advanced Features](docs/advanced_features.md)
- [PyMDP Integration](docs/pymdp_integration.md)

## Running Tests

```bash
python -m unittest discover tests
```

The test suite includes comprehensive tests that verify the MCP server produces identical results to direct PyMDP API calls.

## Examples

The repository includes example scripts demonstrating how to use the MCP server:

### Gridworld Examples

To run the gridworld examples, you have several options:

1. Run with a separate server:
   ```bash
   # In one terminal:
   cd scripts
   python start_mcp_server.py
   
   # In another terminal:
   cd scripts
   python mcp_gridworld_examples.py
   ```

2. Run in direct mode (without a server):
   ```bash
   cd scripts
   python mcp_gridworld_examples.py --direct
   ```

3. Use the all-in-one script:
   ```bash
   cd scripts
   ./run_with_server.sh
   ```

These examples demonstrate both 3x3 and 4x4 gridworld environments with active inference agents.

## Building Your Own Server

This template provides a foundation for building more complex MCP servers for active inference. To build your own:

1. Add more environment types and simulation capabilities
2. Implement additional active inference methods using the `@mcp.tool()` decorator
3. Extend the functionality with custom PyMDP extensions
4. Add visualization tools for displaying agent behavior and belief dynamics

## Citation

If you use this MCP server in your research, please cite both this repository and the original PyMDP paper:

```
@article{Heins2022,
  doi = {10.21105/joss.04098},
  url = {https://doi.org/10.21105/joss.04098},
  year = {2022},
  publisher = {The Open Journal},
  volume = {7},
  number = {73},
  pages = {4098},
  author = {Conor Heins and Beren Millidge and Daphne Demekas and Brennan Klein and Karl Friston and Iain D. Couzin and Alexander Tschantz},
  title = {pymdp: A Python library for active inference in discrete state spaces},
  journal = {Journal of Open Source Software}
}
```

# MCP-PyMDP

## Model Context Protocol for Active Inference with PyMDP

MCP-PyMDP provides a Model Context Protocol (MCP) server implementation that enables Large Language Models (LLMs) to access PyMDP's active inference functionality. This integration allows AI assistants to create, simulate, and analyze active inference agents through a standardized protocol.

## What is Model Context Protocol?

The [Model Context Protocol (MCP)](https://modelcontextprotocol.io) is an open protocol that standardizes how applications provide context to LLMs and how LLMs can interact with external tools. It enables seamless integration between LLM applications and external data sources/tools, similar to how USB-C provides a standardized way to connect various devices.

## What is PyMDP?

[PyMDP](https://github.com/infer-actively/pymdp) is a Python library for active inference in discrete state spaces. It provides tools for building and simulating agents based on the Free Energy Principle and active inference framework.

## Project Structure

- `src/mcp/server/`: MCP server implementation
- `src/mcp/client/`: MCP client implementation
- `src/mcp/utils.py`: PyMDP integration layer
- `scripts/`: Example scripts and server startup
- `tests/`: Test suites for MCP-PyMDP functionality

## Features

Through the MCP protocol, this project enables:

1. **Agent Creation and Configuration**
   - Create active inference agents with customizable parameters
   - Define and modify generative models (A, B, C, D matrices)
   - Configure inference parameters

2. **Inference and Learning**
   - Perform variational inference on hidden states
   - Compute expected free energy
   - Optimize policies
   - Sample actions based on posterior beliefs

3. **Environment Interaction**
   - Simulate agent-environment interactions
   - Track agent performance and belief dynamics

4. **Visualization and Analysis**
   - Generate plots of belief dynamics
   - Visualize agent behavior in environments

## Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/mcp-pymdp.git
cd mcp-pymdp

# Install dependencies
pip install -e .
```

### Running the Server

```bash
# Start the MCP server
python scripts/start_mcp_server.py --host localhost --port 8080
```

### Using with Claude or Other LLMs

The server exposes MCP-compatible endpoints that can be used by any LLM client that supports the Model Context Protocol, such as Claude Desktop.

Example prompt for Claude:

```
Using the MCP server running at http://localhost:8080, create a simple active inference agent in a grid world environment and have it navigate to a reward.
```

## MCP Tools

The server exposes several MCP tools that LLMs can use:

- `create_agent`: Create a custom agent with specified observation and state dimensions
- `create_gridworld_agent`: Create an agent specifically for grid world environments
- `create_environment`: Create a simulation environment
- `infer_states`: Perform state inference for an agent
- `infer_policies`: Perform policy inference
- `sample_action`: Sample an action from the agent's policy posterior
- `run_simulation`: Run a full simulation with an agent in an environment

## Examples

Check out the `scripts/pymcp_examples.py` script for examples of how to use the MCP client to interact with the server.

```bash
python scripts/pymcp_examples.py --host localhost --port 8080
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The PyMDP development team for their excellent active inference library
- The Model Context Protocol team for developing the open protocol
