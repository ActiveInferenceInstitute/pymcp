"""
MCP-PyMDP - Message-based Cognitive Protocol for Active Inference with PyMDP.

This package provides a server and client implementation for the Message-based Cognitive Protocol (MCP)
using the PyMDP active inference framework.

The package includes the following components:
- MCP server implementation
- MCP client for interacting with the server
- Environment implementations for agent simulations
- Visualization utilities for Active Inference models
- Testing utilities for the MCP toolkit
"""

# Version information
__version__ = "0.1.0"

# Import client components
from .client import (
    MCPClient,
    MCPToolKit,
    load_config,
    save_config,
    get_server_url
)

# Import visualization module
from .visualization import (
    visualize_generative_model,
    visualize_free_energy_components,
    visualize_variational_free_energy,
    visualize_simulation,
    create_belief_heatmap,
    visualize_policy_evolution
)

# Import environments
from .environments import (
    GridWorldEnvironment,
    create_grid_world_env,
    CustomEnvironment,
    create_custom_env
)

# Define top-level exports
__all__ = [
    # Version
    '__version__',
    
    # Client
    'MCPClient',
    'MCPToolKit',
    'load_config',
    'save_config',
    'get_server_url',
    
    # Visualization
    'visualize_generative_model',
    'visualize_free_energy_components',
    'visualize_variational_free_energy',
    'visualize_simulation',
    'create_belief_heatmap',
    'visualize_policy_evolution',
    
    # Environments
    'GridWorldEnvironment',
    'create_grid_world_env',
    'CustomEnvironment',
    'create_custom_env'
] 