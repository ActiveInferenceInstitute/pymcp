# PyMDP-MCP Implementation Details

This document provides technical details on how the PyMDP framework is integrated with the Model Context Protocol (MCP) server.

## Architecture Overview

The PyMDP-MCP implementation follows a layered architecture:

1. **MCP Server Layer**: Implemented with FastMCP in `src/main.py`, handles API requests and routing
2. **PyMDP Interface Layer**: The `PyMDPInterface` class in `src/mcp/utils.py` acts as a bridge between MCP tools and PyMDP
3. **Client Implementation Layer**: Provides both network-based and direct client access in `src/mcp/client.py`
4. **PyMDP Core**: Uses the original PyMDP library from the pymdp-clone directory for all active inference algorithms

## Core Components

### PyMDP Interface

The central component of the integration is the `PyMDPInterface` class which:

- Maintains registries of agents, environments, and simulation histories
- Provides methods that map directly to MCP tools
- Handles data conversion between JSON and PyMDP's NumPy-based data structures
- Manages serialization of complex objects and arrays
- Provides rich visualization capabilities

```python
class PyMDPInterface:
    def __init__(self):
        """Initialize the PyMDP interface."""
        self.agents = {}
        self.environments = {}
        self.simulation_history = {}
        self.computation_logs = {}
        self.debug_mode = True
        self.sessions = {}
```

### MCP Server Implementation

The MCP server is implemented using the FastMCP framework, with tools defined as async functions:

```python
@mcp.tool()
async def create_agent(ctx: Context, name: str, generative_model: str) -> str:
    """Create an active inference agent with specified parameters."""
    try:
        pymdp_interface = ctx.request_context.lifespan_context.pymdp_interface
        model_dict = json.loads(generative_model)
        result = pymdp_interface.create_agent(name, model_dict)
        return json.dumps({"result": {"agent": result}}, indent=2, cls=NumpyEncoder)
    except Exception as e:
        return json.dumps({"error": f"Error creating agent: {str(e)}"})
```

### Agent Management

Agents are created and stored in the `PyMDPInterface.agents` dictionary:

```python
def create_agent(self, name: str, generative_model: Dict) -> Dict:
    """Create an agent from a generative model."""
    # Extract generative model components
    A = generative_model.get("A", [])
    B = generative_model.get("B", [])
    C = generative_model.get("C", None)
    D = generative_model.get("D", None)
    
    # Convert to PyMDP format
    A_obj = self._convert_to_obj_array(A)
    B_obj = self._convert_to_obj_array(B)
    C_obj = self._convert_to_obj_array(C) if C is not None else None
    D_obj = self._convert_to_obj_array(D) if D is not None else None
    
    # Create the PyMDP agent
    agent = Agent(A=A_obj, B=B_obj, C=C_obj, D=D_obj)
    
    # Store the agent
    self.agents[name] = agent
    
    return {"name": name, "details": "Agent created successfully"}
```

### Environment Simulation

The implementation includes built-in support for grid world environments:

```python
def create_grid_world_env(self, name: str, grid_size: List[int], reward_positions: List[List[int]]) -> Dict:
    """Create a grid world environment."""
    # Create the environment object with state and methods
    env = {
        "type": "grid_world",
        "grid_size": grid_size,
        "reward_positions": reward_positions,
        "agent_pos": [0, 0],  # Default starting position
        "step_count": 0
    }
    
    # Store environment and return reference
    self.environments[name] = env
    return {"name": name, "details": env}
```

### Data Conversion Utilities

The implementation includes specialized utilities for converting between JSON and PyMDP's NumPy arrays:

```python
class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles NumPy types."""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)
```

### Client Implementation

Two client implementations are provided:

1. **MCPClient**: For network-based communication

```python
class MCPClient:
    """Client for communicating with the MCP server"""
    
    async def call_tool(self, tool_id, params=None):
        """Call an MCP tool on the server."""
        # Prepare request data
        request_data = {"params": params or {}}
        
        # Make request to server
        tool_url = f"{self.server_url}/tools/{tool_id}"
        async with self.session.post(tool_url, json=request_data) as response:
            response_data = await response.json()
            return response_data
```

2. **DirectMCPClient**: For direct function calls without network overhead

```python
class DirectMCPClient:
    """Client that directly calls MCP tools without going through the server."""
    
    async def call_tool(self, tool_name, **params):
        """Call a tool directly."""
        # Get PyMDP interface
        interface = get_pymdp_interface()
        
        # Call the method directly on the interface
        if hasattr(interface, tool_name):
            method = getattr(interface, tool_name)
            result = method(**params)
            return {"result": result}
        else:
            return {"error": f"Tool {tool_name} not found"}
```

## Visualization Capabilities

The implementation includes extensive visualization support through a monkey-patched extension system:

```python
def _add_visualization_methods():
    """Add visualization methods to PyMDPInterface class."""
    
    def plot_belief_dynamics(self, session_id: str, output_file: str = "belief_dynamics.png") -> Dict:
        """Generate a visualization of belief dynamics over time for a session."""
        # Implementation...
    
    def analyze_free_energy(self, session_id: str, output_file: str = "free_energy.png") -> Dict:
        """Analyze and visualize free energy components from a session."""
        # Implementation...
    
    # Add methods to the class
    PyMDPInterface.plot_belief_dynamics = plot_belief_dynamics
    PyMDPInterface.analyze_free_energy = analyze_free_energy
```

## Session Management

The implementation supports persistent sessions to maintain state across multiple tool calls:

```python
# Session example from implementation
session = {
    "agent_id": agent_id,
    "env_id": env_id,
    "history": {
        "timesteps": [],
        "metadata": {
            "start_time": time.time(),
            "agent_details": self.get_agent(agent_id),
            "env_details": self.get_environment(env_id)
        }
    }
}
self.sessions[session_id] = session
```

## Performance Considerations

The implementation includes several optimization strategies:

1. **Caching**: Storing computation results to avoid redundant operations
2. **Custom Serialization**: Efficient conversion between NumPy arrays and JSON
3. **Direct Client Option**: Bypassing network overhead for local usage
4. **Session Management**: Maintaining state to avoid reinitialization costs
5. **Asynchronous Processing**: Supporting concurrent operations with asyncio 