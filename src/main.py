from mcp.server.fastmcp import FastMCP, Context
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from dataclasses import dataclass
import asyncio
import json
import os
import numpy as np
from typing import List, Dict, Union, Optional, Any
import argparse
import sys

from mcp.utils import PyMDPInterface, NumpyEncoder

# Create a dataclass for our application context
@dataclass
class PyMDPContext:
    """Context for the PyMDP MCP server."""
    pymdp_interface: object

@asynccontextmanager
async def pymdp_lifespan(server: FastMCP) -> AsyncIterator[PyMDPContext]:
    """
    Manages the PyMDP interface lifecycle.
    
    Args:
        server: The FastMCP server instance
        
    Yields:
        PyMDPContext: The context containing the PyMDP interface
    """
    # Create and return the PyMDP interface 
    pymdp_interface = PyMDPInterface()
    
    try:
        yield PyMDPContext(pymdp_interface=pymdp_interface)
    finally:
        # No explicit cleanup needed
        pass

# Initialize FastMCP server with the PyMDP interface as context
mcp = FastMCP(
    "pymdp-mcp",
    description="MCP server for Active Inference and Markov Decision Processes using PyMDP",
    lifespan=pymdp_lifespan,
    host=os.getenv("HOST", "0.0.0.0"),
    port=os.getenv("PORT", "8050")
)        

@mcp.tool()
async def create_agent(ctx: Context, name: str, 
                     generative_model: str) -> str:
    """Create an active inference agent with specified parameters.
    
    Args:
        ctx: The MCP server provided context
        name: Name to identify this agent
        generative_model: JSON string of generative model parameters (A, B, C matrices)
    """
    try:
        pymdp_interface = ctx.request_context.lifespan_context.pymdp_interface
        model_dict = json.loads(generative_model)
        result = pymdp_interface.create_agent(name, model_dict)
        return json.dumps({"result": {"agent": result}}, indent=2, cls=NumpyEncoder)
    except Exception as e:
        return json.dumps({"error": f"Error creating agent: {str(e)}"})

@mcp.tool()
async def define_generative_model(ctx: Context, A_dims: str, B_dims: str) -> str:
    """Define random A and B matrices based on dimensions.
    
    Args:
        ctx: The MCP server provided context
        A_dims: JSON string of A matrix dimensions [[obs_dim1, state_dim1], [obs_dim2, state_dim2], ...]
        B_dims: JSON string of B matrix dimensions [[state_dim1, state_dim1, control_dim1], ...]
    """
    try:
        pymdp_interface = ctx.request_context.lifespan_context.pymdp_interface
        A_dims_list = json.loads(A_dims)
        B_dims_list = json.loads(B_dims)
        result = pymdp_interface.define_generative_model(A_dims_list, B_dims_list)
        return json.dumps({"result": result}, indent=2, cls=NumpyEncoder)
    except Exception as e:
        return json.dumps({"error": f"Error defining generative model: {str(e)}"})

@mcp.tool()
async def infer_states(ctx: Context, agent_name: str, observation: str, method: str = 'FPI') -> str:
    """Infer hidden states given an observation.
    
    Args:
        ctx: The MCP server provided context
        agent_name: Name of the agent
        observation: JSON string of observation indices
        method: Inference method to use ('FPI', 'VMP', 'MMP', 'BP')
    """
    try:
        pymdp_interface = ctx.request_context.lifespan_context.pymdp_interface
        obs_list = json.loads(observation)
        result = pymdp_interface.infer_states(agent_name, obs_list, method)
        return json.dumps({"result": result}, indent=2, cls=NumpyEncoder)
    except Exception as e:
        return json.dumps({"error": f"Error inferring states: {str(e)}"})

@mcp.tool()
async def infer_policies(ctx: Context, agent_name: str, planning_horizon: int = None) -> str:
    """Optimize policies based on expected free energy.
    
    Args:
        ctx: The MCP server provided context
        agent_name: Name of the agent
        planning_horizon: Optional planning horizon for temporal planning (number of steps to plan ahead)
    """
    try:
        pymdp_interface = ctx.request_context.lifespan_context.pymdp_interface
        result = pymdp_interface.infer_policies(agent_name, planning_horizon=planning_horizon)
        return json.dumps({"result": result}, indent=2, cls=NumpyEncoder)
    except Exception as e:
        return json.dumps({"error": f"Error inferring policies: {str(e)}"})

@mcp.tool()
async def sample_action(ctx: Context, agent_name: str, planning_horizon: int = None) -> str:
    """Sample an action from the agent's policy distribution.
    
    Args:
        ctx: The MCP server provided context
        agent_name: Name of the agent
        planning_horizon: Optional planning horizon for temporal planning (number of steps to plan ahead)
    """
    try:
        pymdp_interface = ctx.request_context.lifespan_context.pymdp_interface
        result = pymdp_interface.sample_action(agent_name, planning_horizon=planning_horizon)
        return json.dumps({"result": result}, indent=2, cls=NumpyEncoder)
    except Exception as e:
        return json.dumps({"error": f"Error sampling action: {str(e)}"})

@mcp.tool()
async def create_grid_world_env(ctx: Context, name: str, grid_size: str, 
                             reward_locations: str) -> str:
    """Create a simple grid world environment.
    
    Args:
        ctx: The MCP server provided context
        name: Name to identify this environment
        grid_size: JSON string of grid dimensions [height, width]
        reward_locations: JSON string of reward positions [[x1, y1], [x2, y2], ...]
    """
    try:
        pymdp_interface = ctx.request_context.lifespan_context.pymdp_interface
        grid_size_list = json.loads(grid_size)
        reward_locations_list = json.loads(reward_locations)
        result = pymdp_interface.create_grid_world_env(name, grid_size_list, reward_locations_list)
        return json.dumps({"result": result}, indent=2, cls=NumpyEncoder)
    except Exception as e:
        return json.dumps({"error": f"Error creating environment: {str(e)}"})

@mcp.tool()
async def step_environment(ctx: Context, env_name: str, action: str) -> str:
    """Update the environment given an agent's action.
    
    Args:
        ctx: The MCP server provided context
        env_name: Name of the environment
        action: JSON string of action indices
    """
    try:
        pymdp_interface = ctx.request_context.lifespan_context.pymdp_interface
        action_list = json.loads(action)
        result = pymdp_interface.step_environment(env_name, action_list)
        return json.dumps({"result": result}, indent=2, cls=NumpyEncoder)
    except Exception as e:
        return json.dumps({"error": f"Error stepping environment: {str(e)}"})

@mcp.tool()
async def run_simulation(ctx: Context, agent_name: str, env_name: str, 
                      num_timesteps: int) -> str:
    """Run a simulation for a specified number of timesteps.
    
    Args:
        ctx: The MCP server provided context
        agent_name: Name of the agent
        env_name: Name of the environment
        num_timesteps: Number of timesteps to simulate
    """
    try:
        pymdp_interface = ctx.request_context.lifespan_context.pymdp_interface
        result = pymdp_interface.run_simulation(agent_name, env_name, num_timesteps)
        return json.dumps({"result": result}, indent=2, cls=NumpyEncoder)
    except Exception as e:
        return json.dumps({"error": f"Error running simulation: {str(e)}"})

@mcp.tool()
async def visualize_simulation(ctx: Context, simulation_id: str, 
                           output_file: str = "simulation.png") -> str:
    """Generate a visualization of a simulation.
    
    Args:
        ctx: The MCP server provided context
        simulation_id: ID of the simulation to visualize
        output_file: Filename for saving the visualization
    """
    try:
        pymdp_interface = ctx.request_context.lifespan_context.pymdp_interface
        result = pymdp_interface.visualize_simulation(simulation_id, output_file)
        return json.dumps({"result": result}, indent=2, cls=NumpyEncoder)
    except Exception as e:
        return json.dumps({"error": f"Error visualizing simulation: {str(e)}"})

@mcp.tool()
async def get_agent(ctx: Context, name: str) -> str:
    """Retrieve an agent by name.
    
    Args:
        ctx: The MCP server provided context
        name: Name of the agent
    """
    try:
        pymdp_interface = ctx.request_context.lifespan_context.pymdp_interface
        result = pymdp_interface.get_agent(name)
        return json.dumps({"result": result}, indent=2, cls=NumpyEncoder)
    except Exception as e:
        return json.dumps({"error": f"Error retrieving agent: {str(e)}"})

@mcp.tool()
async def get_environment(ctx: Context, name: str) -> str:
    """Retrieve an environment by name.
    
    Args:
        ctx: The MCP server provided context
        name: Name of the environment
    """
    try:
        pymdp_interface = ctx.request_context.lifespan_context.pymdp_interface
        result = pymdp_interface.get_environment(name)
        return json.dumps({"result": result}, indent=2, cls=NumpyEncoder)
    except Exception as e:
        return json.dumps({"error": f"Error retrieving environment: {str(e)}"})

@mcp.tool()
async def get_all_agents(ctx: Context) -> str:
    """Get all stored agents.
    
    Args:
        ctx: The MCP server provided context
    """
    try:
        pymdp_interface = ctx.request_context.lifespan_context.pymdp_interface
        result = pymdp_interface.get_all_agents()
        return json.dumps({"result": result}, indent=2, cls=NumpyEncoder)
    except Exception as e:
        return json.dumps({"error": f"Error retrieving agents: {str(e)}"})
        
@mcp.tool()
async def get_all_environments(ctx: Context) -> str:
    """Get all stored environments.
    
    Args:
        ctx: The MCP server provided context
    """
    try:
        pymdp_interface = ctx.request_context.lifespan_context.pymdp_interface
        result = pymdp_interface.get_all_environments()
        return json.dumps({"result": result}, indent=2, cls=NumpyEncoder)
    except Exception as e:
        return json.dumps({"error": f"Error retrieving environments: {str(e)}"})

@mcp.tool()
async def get_all_simulations(ctx: Context) -> str:
    """Get all stored simulation results.
    
    Args:
        ctx: The MCP server provided context
    """
    try:
        pymdp_interface = ctx.request_context.lifespan_context.pymdp_interface
        result = pymdp_interface.get_all_simulations()
        return json.dumps({"result": result}, indent=2, cls=NumpyEncoder)
    except Exception as e:
        return json.dumps({"error": f"Error retrieving simulations: {str(e)}"})

@mcp.tool()
async def get_all_functions(ctx: Context) -> str:
    """Get all available PyMDP functions.
    
    Args:
        ctx: The MCP server provided context
    """
    try:
        pymdp_interface = ctx.request_context.lifespan_context.pymdp_interface
        result = pymdp_interface.get_all_functions()
        return json.dumps({"result": result}, indent=2, cls=NumpyEncoder)
    except Exception as e:
        return json.dumps({"error": f"Error retrieving functions: {str(e)}"})

def run_tests(args: Dict[str, Any]) -> None:
    """
    Run tests using the run_and_organize.sh script
    
    Parameters
    ----------
    args : Dict[str, Any]
        Command line arguments
    """
    print("Running tests with run_and_organize.sh...")
    
    # Build command based on arguments
    cmd = "cd tests && ./run_and_organize.sh"
    
    # Add category if specified
    if args.get('category'):
        cmd += f" --category {args['category']}"
    
    # Add file if specified
    if args.get('file'):
        cmd += f" --file {args['file']}"
    
    # Add summary flag if specified
    if args.get('summary'):
        cmd += " --summary"
    
    # Add verbose flag if specified
    if args.get('verbose'):
        cmd += " --verbose"
    
    # Run the command
    print(f"Executing: {cmd}")
    status = os.system(cmd)
    
    if status == 0:
        print("Tests completed successfully!")
    else:
        print(f"Tests failed with status code: {status}")
    
    sys.exit(status)

def main() -> None:
    """
    Main entry point for MCP-PyMDP

    This script provides examples of how to use the MCP (Model Communication Protocol)
    implementation with PyMDP for active inference agents.
    """
    parser = argparse.ArgumentParser(description="MCP-PyMDP: Active Inference with Model Communication Protocol")
    
    # Add subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Add test command
    test_parser = subparsers.add_parser("test", help="Run tests")
    test_parser.add_argument("--category", "-c", help="Test category (interface, mcp, advanced, visualization, belief_dynamics, core, all)")
    test_parser.add_argument("--file", "-f", help="Specific test file to run")
    test_parser.add_argument("--summary", "-s", action="store_true", help="Generate a detailed test summary")
    test_parser.add_argument("--verbose", "-v", action="store_true", help="Increase verbosity")
    
    # ... other subparsers ...
    
    # Parse arguments
    args = vars(parser.parse_args())
    
    # Handle commands
    if args["command"] == "test":
        run_tests(args)
    # ... handle other commands ...
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
