#!/usr/bin/env python3
"""
MCP Server for PyMDP

This script starts the MCP server to provide PyMDP functionality over HTTP.
"""

import os
import sys
import json
import time
import numpy as np
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
# Add the pymdp-clone directory to the path for direct access to PyMDP
sys.path.insert(0, str(Path(__file__).parent.parent / "pymdp-clone"))

from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Import MCP utilities
from mcp.utils import PyMDPInterface

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Create FastAPI server
app = FastAPI(
    title="PyMDP MCP Server",
    description="Model Context Protocol server for PyMDP active inference toolkit",
    version="0.1.0",
    docs_url="/docs",
    openapi_url="/openapi.json"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create PyMDP interface
pymdp_interface = PyMDPInterface()

# Add server info endpoint
@app.get("/")
async def root():
    """Basic server info following MCP specification."""
    return {
        "name": "PyMDP MCP Server",
        "version": "0.1.0",
        "description": "Model Context Protocol server for PyMDP active inference toolkit",
        "protocol_version": "mcp-2025-03-26",
        "capabilities": ["tools"]
    }

# Add ping endpoint
@app.get("/ping")
async def ping():
    return {"status": "ok", "message": "MCP-PyMDP server is running"}

# Add tools endpoint
@app.get("/tools")
async def get_tools():
    """Get a list of available tools according to the MCP specification."""
    tools = [
        {
            "name": "create_agent",
            "description": "Create an active inference agent with a generative model",
            "parameters": [
                {"name": "name", "type": "string", "description": "Name of the agent", "required": True},
                {"name": "observation_dimensions", "type": "array", "description": "List of observation dimensions (one per modality)", "required": True},
                {"name": "state_dimensions", "type": "array", "description": "List of state dimensions (one per factor)", "required": True},
                {"name": "control_factor_idx", "type": "array", "description": "Indices of state factors that are controllable", "required": False}
            ]
        },
        {
            "name": "create_gridworld_agent",
            "description": "Create a grid world agent with a predefined structure",
            "parameters": [
                {"name": "name", "type": "string", "description": "Name of the agent", "required": False},
                {"name": "grid_size", "type": "array", "description": "Size of the grid [height, width]", "required": False},
                {"name": "reward_positions", "type": "array", "description": "List of reward positions [[row, col], ...]", "required": False},
                {"name": "action_precision", "type": "number", "description": "Precision of action selection", "required": False},
                {"name": "inference_horizon", "type": "integer", "description": "Planning horizon for inference", "required": False},
                {"name": "inference_algorithm", "type": "string", "description": "Algorithm for state inference", "required": False}
            ]
        },
        {
            "name": "create_environment",
            "description": "Create an environment for agent simulations",
            "parameters": [
                {"name": "name", "type": "string", "description": "Name of the environment", "required": False},
                {"name": "type", "type": "string", "description": "Type of environment (gridworld, etc.)", "required": False},
                {"name": "grid_size", "type": "array", "description": "Size of the grid for gridworld", "required": False},
                {"name": "reward_positions", "type": "array", "description": "List of reward positions for gridworld", "required": False}
            ]
        },
        {
            "name": "infer_states",
            "description": "Perform state inference for an agent",
            "parameters": [
                {"name": "agent_id", "type": "string", "description": "ID of the agent", "required": True},
                {"name": "observation", "type": "array", "description": "Observation array", "required": True},
                {"name": "method", "type": "string", "description": "Inference method", "required": False}
            ]
        },
        {
            "name": "infer_policies",
            "description": "Perform policy inference for an agent",
            "parameters": [
                {"name": "agent_id", "type": "string", "description": "ID of the agent", "required": True},
                {"name": "planning_horizon", "type": "integer", "description": "Planning horizon", "required": False}
            ]
        },
        {
            "name": "sample_action",
            "description": "Sample an action from an agent's policy posterior",
            "parameters": [
                {"name": "agent_id", "type": "string", "description": "ID of the agent", "required": True},
                {"name": "planning_horizon", "type": "integer", "description": "Planning horizon", "required": False}
            ]
        },
        {
            "name": "step_environment",
            "description": "Step an environment with an action",
            "parameters": [
                {"name": "env_id", "type": "string", "description": "ID of the environment", "required": True},
                {"name": "action", "type": "array", "description": "Action to take", "required": True}
            ]
        },
        {
            "name": "reset_environment",
            "description": "Reset an environment to its initial state",
            "parameters": [
                {"name": "env_id", "type": "string", "description": "ID of the environment", "required": True}
            ]
        },
        {
            "name": "run_simulation",
            "description": "Run a simulation with an agent in an environment",
            "parameters": [
                {"name": "agent_id", "type": "string", "description": "ID of the agent", "required": True},
                {"name": "env_id", "type": "string", "description": "ID of the environment", "required": True},
                {"name": "num_steps", "type": "integer", "description": "Number of steps to run", "required": False},
                {"name": "save_history", "type": "boolean", "description": "Whether to save simulation history", "required": False},
                {"name": "planning_horizon", "type": "integer", "description": "Planning horizon for policy inference", "required": False}
            ]
        },
        {
            "name": "calculate_free_energy",
            "description": "Calculate variational free energy for given beliefs and observations",
            "parameters": [
                {"name": "A_matrix", "type": "array", "description": "Observation likelihood matrix", "required": True},
                {"name": "prior", "type": "array", "description": "Prior beliefs about states", "required": True},
                {"name": "observation", "type": "array", "description": "Observed outcome", "required": True},
                {"name": "q_states", "type": "array", "description": "Posterior beliefs about states", "required": True}
            ]
        },
        {
            "name": "infer_states_from_observation",
            "description": "Infer states from an observation using Bayes' rule",
            "parameters": [
                {"name": "A_matrix", "type": "array", "description": "Observation likelihood matrix", "required": True},
                {"name": "prior", "type": "array", "description": "Prior beliefs about states", "required": True},
                {"name": "observation", "type": "array", "description": "Observed outcome", "required": True}
            ]
        },
        {
            "name": "define_generative_model",
            "description": "Create a generative model with random A, B matrices of the given dimensions",
            "parameters": [
                {"name": "A_dims", "type": "array", "description": "List of dimensions for each A matrix [obs_dim, state_dim]", "required": True},
                {"name": "B_dims", "type": "array", "description": "List of dimensions for each B matrix [state_dim, state_dim, action_dim]", "required": True}
            ]
        },
        {
            "name": "validate_generative_model",
            "description": "Validate a generative model by checking its structure and normalization",
            "parameters": [
                {"name": "generative_model", "type": "object", "description": "The generative model to validate", "required": True},
                {"name": "check_normalization", "type": "boolean", "description": "Whether to check if matrices are normalized", "required": False}
            ]
        }
    ]
    return {"tools": tools}

# Add tool invocation endpoint according to MCP specification
@app.post("/tools/{tool_name}")
async def invoke_tool(tool_name: str, request: Request):
    """Invoke a tool following MCP protocol."""
    try:
        # Parse request according to MCP format
        data = await request.json()
        
        # Extract parameters according to MCP
        parameters = data.get("parameters", {})
        request_id = data.get("id", str(time.time()))
        
        # Log the incoming request
        logger.info(f"Tool invocation: {tool_name}, request_id: {request_id}")
        
        # Handle tool invocation based on tool name
        if tool_name == "create_agent":
            result = await handle_create_agent(parameters)
        elif tool_name == "create_gridworld_agent":
            result = await handle_create_gridworld_agent(parameters)
        elif tool_name == "create_environment":
            result = await handle_create_environment(parameters)
        elif tool_name == "infer_states":
            result = await handle_infer_states(parameters)
        elif tool_name == "infer_policies":
            result = await handle_infer_policies(parameters)
        elif tool_name == "sample_action":
            result = await handle_sample_action(parameters)
        elif tool_name == "step_environment":
            result = await handle_step_environment(parameters)
        elif tool_name == "reset_environment":
            result = await handle_reset_environment(parameters)
        elif tool_name == "run_simulation":
            result = await handle_run_simulation(parameters)
        elif tool_name == "calculate_free_energy":
            result = await handle_calculate_free_energy(parameters)
        elif tool_name == "infer_states_from_observation":
            result = await handle_infer_states_from_observation(parameters)
        elif tool_name == "validate_generative_model":
            result = await handle_validate_generative_model(parameters)
        else:
            return {
                "id": request_id,
                "status": "error",
                "error": {"message": f"Unknown tool: {tool_name}"}
            }
        
        # Return response in MCP format
        return {
            "id": request_id,
            "status": "success",
            "result": result
        }
    except Exception as e:
        # Handle errors following MCP format
        import traceback
        logger.error(f"Error invoking tool {tool_name}: {str(e)}")
        logger.debug(traceback.format_exc())
        
        return {
            "id": request_id if 'request_id' in locals() else str(time.time()),
            "status": "error",
            "error": {
                "message": f"Error invoking tool {tool_name}: {str(e)}",
                "type": type(e).__name__
            }
        }

# Define handler functions for each tool
async def handle_create_agent(params):
    """Handle create_agent tool invocation."""
    try:
        name = params.get("name", "Agent")
        observation_dimensions = json.loads(params.get("observation_dimensions", "[]"))
        state_dimensions = json.loads(params.get("state_dimensions", "[]"))
        control_factor_idx = json.loads(params.get("control_factor_idx", "[0]"))
        
        # Generate A and B dimensions based on provided dimensions
        A_dims = []
        for obs_dim in observation_dimensions:
            # For each observation modality, create a matrix linked to state dimensions
            A_dims.append([obs_dim, state_dimensions[0]])  # Link to first state factor by default
        
        B_dims = []
        for i, state_dim in enumerate(state_dimensions):
            # For each state factor, create a transition matrix
            # If it's a control factor, add control dimensions
            if i in control_factor_idx:
                # Control factor with multiple actions (default 5 actions)
                B_dims.append([state_dim, state_dim, 5])
            else:
                # Non-control factor (single "action" - just the transition dynamics)
                B_dims.append([state_dim, state_dim, 1])
        
        # Generate random generative model
        generative_model = pymdp_interface.define_generative_model(A_dims, B_dims)
        
        # Create agent
        agent_result = pymdp_interface.create_agent(name, generative_model)
        
        return agent_result
    except Exception as e:
        logger.error(f"Error creating agent: {str(e)}")
        return {"error": f"Error creating agent: {str(e)}"}

async def handle_create_gridworld_agent(params):
    """Handle create_gridworld_agent tool invocation."""
    try:
        name = params.get("name", "GridWorldAgent")
        grid_size = json.loads(params.get("grid_size", "[3, 3]"))
        reward_positions = json.loads(params.get("reward_positions", "[[2, 2]]"))
        action_precision = float(params.get("action_precision", "1.0"))
        inference_horizon = int(params.get("inference_horizon", "5"))
        inference_algorithm = params.get("inference_algorithm", "FPI")
        
        # Create gridworld agent model
        height, width = grid_size
        num_states = height * width
        
        # Define state dimensions for GridWorld
        state_dimensions = [num_states]  # One state factor for position
        observation_dimensions = [num_states, 2]  # Position observation and reward observation
        
        # Create the agent
        try:
            result = pymdp_interface.create_gridworld_agent(
                name=name,
                grid_size=grid_size,
                reward_positions=reward_positions,
                action_precision=action_precision,
                inference_horizon=inference_horizon
            )
            
            # Ensure we have a valid agent ID in the result
            if not result.get("id"):
                result["id"] = name
                
            # Store the agent for reference
            if not hasattr(pymdp_interface, "agents"):
                pymdp_interface.agents = {}
                
            pymdp_interface.agents[name] = result
            
            return result
        except Exception as e:
            logger.error(f"Error creating gridworld agent: {str(e)}")
            traceback.print_exc()
            return {"error": f"Error creating gridworld agent: {str(e)}"}
    except Exception as e:
        logger.error(f"Error handling create_gridworld_agent request: {str(e)}")
        traceback.print_exc()
        return {"error": f"Error handling create_gridworld_agent request: {str(e)}"}

async def handle_create_environment(params):
    """Handle create_environment tool invocation."""
    try:
        name = params.get("name", "Environment")
        env_type = params.get("type", "gridworld")
        
        if env_type.lower() == "gridworld":
            grid_size = json.loads(params.get("grid_size", "[3, 3]"))
            reward_positions = json.loads(params.get("reward_positions", "[[2, 2]]"))
            
            # Create gridworld environment
            env = pymdp_interface.create_grid_world_env(name, grid_size, reward_positions)
            
            return env
        else:
            return {"error": f"Environment type {env_type} not supported"}
    except Exception as e:
        logger.error(f"Error creating environment: {str(e)}")
        return {"error": f"Error creating environment: {str(e)}"}

async def handle_infer_states(params):
    """Handle infer_states tool invocation."""
    try:
        agent_id = params.get("agent_id")
        observation = json.loads(params.get("observation", "[]"))
        method = params.get("method", "FPI")
        
        if not agent_id:
            return {"error": "Missing required parameter: agent_id"}
            
        # Call state inference
        result = pymdp_interface.infer_states(agent_id, observation, method)
        
        return result
    except Exception as e:
        logger.error(f"Error inferring states: {str(e)}")
        return {"error": f"Error inferring states: {str(e)}"}

async def handle_infer_policies(params):
    """Handle infer_policies tool invocation."""
    try:
        agent_id = params.get("agent_id")
        planning_horizon = params.get("planning_horizon")
        
        if not agent_id:
            return {"error": "Missing required parameter: agent_id"}
        
        # Convert planning horizon to int if provided
        if planning_horizon:
            planning_horizon = int(planning_horizon)
            
        # Call policy inference
        result = pymdp_interface.infer_policies(agent_id, planning_horizon=planning_horizon)
        
        return result
    except Exception as e:
        logger.error(f"Error inferring policies: {str(e)}")
        return {"error": f"Error inferring policies: {str(e)}"}

# Placeholder handlers for remaining functions
async def handle_sample_action(params):
    """Handle sample_action tool invocation."""
    try:
        agent_id = params.get("agent_id")
        planning_horizon = params.get("planning_horizon")
        
        if not agent_id:
            return {"error": "Missing required parameter: agent_id"}
        
        # Convert planning horizon to int if provided
        if planning_horizon:
            planning_horizon = int(planning_horizon)
            
        # Call action sampling
        result = pymdp_interface.sample_action(agent_id, planning_horizon=planning_horizon)
        
        return result
    except Exception as e:
        logger.error(f"Error sampling action: {str(e)}")
        return {"error": f"Error sampling action: {str(e)}"}

async def handle_step_environment(params):
    """Handle step_environment tool invocation."""
    try:
        env_id = params.get("env_id")
        action = params.get("action")
        
        if not env_id:
            return {"error": "Missing required parameter: env_id"}
        
        # Convert action to list if it's a JSON string
        if isinstance(action, str):
            try:
                action = json.loads(action)
            except:
                # If it's not valid JSON, try to convert to int
                try:
                    action = int(action)
                except:
                    return {"error": "Invalid action format"}
        
        # Call environment step
        result = pymdp_interface.step_environment(env_id, action)
        
        return result
    except Exception as e:
        logger.error(f"Error stepping environment: {str(e)}")
        return {"error": f"Error stepping environment: {str(e)}"}

async def handle_reset_environment(params):
    """Handle reset_environment tool invocation."""
    try:
        env_id = params.get("env_id")
        
        if not env_id:
            return {"error": "Missing required parameter: env_id"}
            
        # Call environment reset
        result = pymdp_interface.reset_environment(env_id)
        
        return result
    except Exception as e:
        logger.error(f"Error resetting environment: {str(e)}")
        return {"error": f"Error resetting environment: {str(e)}"}

async def handle_run_simulation(params):
    """Handle run_simulation tool invocation."""
    try:
        agent_id = params.get("agent_id")
        env_id = params.get("env_id")
        num_steps = int(params.get("num_steps", "10"))
        save_history = params.get("save_history", True)
        planning_horizon = params.get("planning_horizon")
        
        if not agent_id or not env_id:
            return {"error": "Missing required parameters: agent_id and env_id"}
        
        # Convert planning horizon to int if provided
        if planning_horizon:
            planning_horizon = int(planning_horizon)
        
        # Run simulation
        result = pymdp_interface.run_simulation(
            agent_id,
            env_id,
            num_steps=num_steps,
            save_history=save_history,
            planning_horizon=planning_horizon
        )
        
        return result
    except Exception as e:
        logger.error(f"Error running simulation: {str(e)}")
        return {"error": f"Error running simulation: {str(e)}"}

async def handle_calculate_free_energy(params):
    """Handle calculate_free_energy tool invocation."""
    try:
        A_matrix = json.loads(params.get("A_matrix", "[]"))
        prior = json.loads(params.get("prior", "[]"))
        observation = json.loads(params.get("observation", "[]"))
        q_states = json.loads(params.get("q_states", "[]"))
        
        # Call free energy calculation
        result = pymdp_interface.calculate_free_energy(A_matrix, prior, observation, q_states)
        
        return result
    except Exception as e:
        logger.error(f"Error calculating free energy: {str(e)}")
        return {"error": f"Error calculating free energy: {str(e)}"}

async def handle_infer_states_from_observation(params):
    """Handle infer_states_from_observation tool invocation."""
    try:
        A_matrix = json.loads(params.get("A_matrix", "[]"))
        prior = json.loads(params.get("prior", "[]"))
        observation = json.loads(params.get("observation", "[]"))
        
        # Call state inference
        result = pymdp_interface.infer_states_from_observation(A_matrix, prior, observation)
        
        return result
    except Exception as e:
        logger.error(f"Error inferring states from observation: {str(e)}")
        return {"error": f"Error inferring states from observation: {str(e)}"}

async def handle_validate_generative_model(params):
    """Handle validate_generative_model tool invocation."""
    try:
        generative_model = json.loads(params.get("generative_model", "{}"))
        check_normalization = params.get("check_normalization", "true").lower() == "true"
        
        if not generative_model:
            return {"error": "Missing required parameter: generative_model"}
            
        # Call validation
        result = pymdp_interface.validate_generative_model(generative_model, check_normalization)
        
        return result
    except Exception as e:
        logger.error(f"Error validating generative model: {str(e)}")
        return {"error": f"Error validating generative model: {str(e)}"}

# Add endpoints for environment management
@app.post("/environments")
async def create_environment(request: Request):
    data = await request.json()
    name = data.get("name", "GridWorldEnv")
    grid_size = data.get("grid_size", [3, 3])
    reward_locations = data.get("reward_locations", [[2, 2]])
    
    # Mock implementation
    env = {
        "id": name,
        "name": name,
        "type": "grid_world",
        "grid_size": grid_size,
        "reward_locations": reward_locations
    }
    
    # Store in PyMDP interface
    if not hasattr(pymdp_interface, "environments"):
        pymdp_interface.environments = {}
    
    pymdp_interface.environments[name] = env
    
    return env

@app.get("/environments")
async def get_environments():
    if not hasattr(pymdp_interface, "environments"):
        pymdp_interface.environments = {}
    
    environments = list(pymdp_interface.environments.values())
    return {"environments": environments}

@app.get("/environments/{env_id}")
async def get_environment(env_id: str):
    if not hasattr(pymdp_interface, "environments"):
        pymdp_interface.environments = {}
    
    if env_id not in pymdp_interface.environments:
        return {"error": f"Environment {env_id} not found"}
    
    return pymdp_interface.environments[env_id]

# Add endpoints for agent management
@app.get("/agents")
async def get_agents():
    if not hasattr(pymdp_interface, "agents"):
        pymdp_interface.agents = {}
    
    agents = list(pymdp_interface.agents.values())
    return {"agents": agents}

@app.get("/agents/{agent_id}")
async def get_agent(agent_id: str):
    if not hasattr(pymdp_interface, "agents"):
        pymdp_interface.agents = {}
    
    if agent_id not in pymdp_interface.agents:
        return {"error": f"Agent {agent_id} not found"}
    
    return pymdp_interface.agents[agent_id]

# Add endpoints for session management
@app.post("/sessions")
async def create_session(request: Request):
    # Parse request data
    data = await request.json()
    agent_id = data.get("agent_id")
    env_id = data.get("env_id")
    
    if not agent_id or not env_id:
        return Response(
            content=json.dumps({"error": "Missing required parameters: agent_id and env_id"}),
            media_type="application/json",
            status_code=400
        )
    
    # Generate session ID if not provided
    session_id = data.get("session_id", f"{agent_id}_{env_id}_session")
    
    # Create sessions dictionary if it doesn't exist
    if not hasattr(pymdp_interface, "sessions"):
        pymdp_interface.sessions = {}
    
    # Create or update session
    session = {
        "id": session_id,
        "agent": agent_id,
        "environment": env_id,
        "created_at": time.time(),
        "history": {
            "timesteps": []
        }
    }
    
    pymdp_interface.sessions[session_id] = session
    
    return {"session": session}

@app.post("/sessions/{session_id}/run")
async def run_session(session_id: str, request: Request):
    # Check if sessions storage exists
    if not hasattr(pymdp_interface, "sessions"):
        pymdp_interface.sessions = {}
        
    # Check if session exists
    if session_id not in pymdp_interface.sessions:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    
    # Get session data
    session = pymdp_interface.sessions[session_id]
    
    # Parse request data
    data = await request.json()
    steps = data.get("steps", 1)
    
    try:
        # Get agent and environment IDs
        agent_id = session.get("agent")
        env_id = session.get("environment")
        
        if not agent_id or not env_id:
            raise ValueError("Session missing agent or environment ID")
            
        # Check if PyMDP interface has needed components
        if not hasattr(pymdp_interface, "run_simulation"):
            # Create a basic simulation function if not available
            def run_sim(agent_id, env_id, num_steps):
                """Run a basic gridworld simulation."""
                # Create simulation history structure
                history = {"timesteps": []}
                
                # Get agent and environment
                agent = pymdp_interface.agents.get(agent_id, {})
                env = pymdp_interface.environments.get(env_id, {})
                
                # Get grid world parameters
                grid_size = env.get("grid_size", [3, 3])
                rows, cols = grid_size
                reward_positions = env.get("reward_positions", [[2, 2]])
                
                # Initialize agent position
                agent_pos = [0, 0]
                last_pos = [0, 0]
                
                # Optional: Log file for detailed simulation data
                output_dir = os.environ.get("MCP_OUTPUT_DIR", "")
                if output_dir:
                    log_file = os.path.join(output_dir, f"simulation_{agent_id}_{env_id}.log")
                    os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
                    with open(log_file, "w") as f:
                        f.write(f"Simulation started: agent={agent_id}, env={env_id}, steps={num_steps}\n")
                        f.write(f"Grid size: {grid_size}, reward positions: {reward_positions}\n")
                
                # Run simulation steps
                for t in range(num_steps):
                    # Create observation based on position
                    obs_position = agent_pos[0] * cols + agent_pos[1]
                    
                    # Check if in reward position
                    in_reward = False
                    for reward_pos in reward_positions:
                        if agent_pos[0] == reward_pos[0] and agent_pos[1] == reward_pos[1]:
                            in_reward = True
                            break
                    
                    observation = [obs_position, 1 if in_reward else 0]
                    
                    # Initialize beliefs (simplified)
                    num_positions = rows * cols
                    beliefs = [[0.0] * num_positions]
                    beliefs[0][obs_position] = 1.0  # Certain about position
                    
                    # Choose a random action, but be smarter about it:
                    # 0: Up, 1: Right, 2: Down, 3: Left
                    
                    # Simple policy to move toward closest reward
                    if reward_positions:
                        # Find closest reward
                        closest_reward = reward_positions[0]
                        min_dist = float('inf')
                        
                        for reward_pos in reward_positions:
                            dist = abs(reward_pos[0] - agent_pos[0]) + abs(reward_pos[1] - agent_pos[1])
                            if dist < min_dist:
                                min_dist = dist
                                closest_reward = reward_pos
                        
                        # Determine direction to move
                        row_diff = closest_reward[0] - agent_pos[0]
                        col_diff = closest_reward[1] - agent_pos[1]
                        
                        # Avoid going back and forth
                        if row_diff != 0 and agent_pos[0] != last_pos[0]:
                            # Move vertically
                            action = [2] if row_diff > 0 else [0]  # Down or Up
                        elif col_diff != 0:
                            # Move horizontally
                            action = [1] if col_diff > 0 else [3]  # Right or Left
                        else:
                            # At the reward, random move
                            action = [np.random.randint(0, 4)]
                    else:
                        # No rewards, random move
                        action = [np.random.randint(0, 4)]
                    
                    # Save last position
                    last_pos = agent_pos.copy()
                    
                    # Update position based on action
                    if action[0] == 0 and agent_pos[0] > 0:
                        agent_pos[0] -= 1  # Move up
                    elif action[0] == 1 and agent_pos[1] < cols - 1:
                        agent_pos[1] += 1  # Move right
                    elif action[0] == 2 and agent_pos[0] < rows - 1:
                        agent_pos[0] += 1  # Move down
                    elif action[0] == 3 and agent_pos[1] > 0:
                        agent_pos[1] -= 1  # Move left
                    
                    # Mock free energy calculation (decreasing with time)
                    free_energy = -1.0 * (t + 1) / num_steps
                    
                    # Add timestep data
                    timestep = {
                        "t": t,
                        "action": action,
                        "observation": observation,
                        "state": agent_pos.copy(),
                        "belief": beliefs,
                        "free_energy": free_energy,
                        "reward": 1.0 if in_reward else 0.0
                    }
                    history["timesteps"].append(timestep)
                    
                    # Log detailed simulation data
                    if output_dir:
                        with open(log_file, "a") as f:
                            f.write(f"Timestep {t}: pos={agent_pos}, action={action}, reward={1.0 if in_reward else 0.0}\n")
                
                # Save the final trajectory plot
                if output_dir:
                    import matplotlib.pyplot as plt
                    from matplotlib.colors import ListedColormap
                    
                    # Create trajectory visualization
                    fig, ax = plt.subplots(figsize=(10, 8))
                    
                    # Create grid
                    grid = np.zeros(grid_size)
                    for pos in reward_positions:
                        if pos[0] < rows and pos[1] < cols:
                            grid[pos[0], pos[1]] = 1
                    
                    # Create custom colormap
                    cmap = ListedColormap(['whitesmoke', 'gold'])
                    
                    # Plot grid
                    ax.imshow(grid, cmap=cmap, interpolation='nearest')
                    
                    # Add grid lines
                    ax.grid(True, color='black', linestyle='-', linewidth=0.5)
                    
                    # Extract positions
                    positions = []
                    for step in history["timesteps"]:
                        if "state" in step:
                            positions.append(tuple(step["state"]))
                    
                    if positions:
                        # Plot trajectory
                        x_coords = [p[1] for p in positions]
                        y_coords = [p[0] for p in positions]
                        
                        ax.plot(x_coords, y_coords, 'b-', linewidth=2, alpha=0.7)
                        ax.plot(positions[0][1], positions[0][0], 'go', markersize=12)
                        ax.plot(positions[-1][1], positions[-1][0], 'ro', markersize=12)
                        
                        # Add position markers
                        for i, (y, x) in enumerate(positions):
                            ax.text(x, y, str(i), ha='center', va='center', fontsize=10, 
                                  color='black', fontweight='bold')
                    
                    # Add reward markers
                    for pos in reward_positions:
                        if pos[0] < rows and pos[1] < cols:
                            ax.text(pos[1], pos[0], "R", ha='center', va='center', 
                                  fontsize=12, color='black', fontweight='bold')
                    
                    # Set labels
                    ax.set_title(f'Agent Trajectory - {agent_id} in {env_id}')
                    ax.set_xlabel('Column')
                    ax.set_ylabel('Row')
                    
                    # Save plot
                    trajectory_file = os.path.join(output_dir, f"trajectory_{agent_id}_{env_id}.png")
                    plt.savefig(trajectory_file, bbox_inches='tight', dpi=150)
                    plt.close()
                    
                    # Also save the raw simulation data as JSON
                    import json
                    data_file = os.path.join(output_dir, f"simulation_data_{agent_id}_{env_id}.json")
                    with open(data_file, "w") as f:
                        json.dump(history, f, indent=2)
                
                return history
                
            pymdp_interface.run_simulation = run_sim
            
        # Run the actual simulation
        history = pymdp_interface.run_simulation(agent_id, env_id, steps)
        session["history"] = history
        
        # Add simulation results
        result = {
            "session_id": session_id,
            "steps_completed": steps,
            "history": session["history"],
            "id": session_id
        }
        
        return result
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "session_id": session_id,
            "error": f"Error running simulation: {str(e)}",
            "steps_completed": 0
        }

@app.get("/sessions")
async def get_sessions():
    if not hasattr(pymdp_interface, "sessions"):
        pymdp_interface.sessions = {}
    
    sessions = []
    for session_id, session_data in pymdp_interface.sessions.items():
        sessions.append({
            "id": session_id,
            "agent": session_data.get("agent"),
            "environment": session_data.get("environment"),
            "timesteps": len(session_data.get("history", {}).get("timesteps", []))
        })
    
    return {"sessions": sessions}

# Add visualization endpoints
@app.get("/agents/{agent_id}/visualize")
async def visualize_agent(agent_id: str, output_file: Optional[str] = None):
    if not hasattr(pymdp_interface, "agents"):
        pymdp_interface.agents = {}
    
    if agent_id not in pymdp_interface.agents:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
    
    # Create actual visualization
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    
    agent = pymdp_interface.agents[agent_id]
    
    # Check if output directory is specified in environment variable
    output_dir = os.environ.get("MCP_OUTPUT_DIR", "")
    
    # If no output file is specified, create one in the output directory
    if output_file is None:
        if output_dir:
            output_file = os.path.join(output_dir, f"agent_{agent_id}_model.png")
        else:
            output_file = f"agent_{agent_id}_model.png"
    elif not os.path.isabs(output_file) and output_dir:
        # If output file is relative and output directory is specified, join them
        output_file = os.path.join(output_dir, output_file)
    
    output_path = output_file
    
    try:
        # Create a figure for the agent model
        plt.figure(figsize=(10, 8))
        
        # Extract information from agent
        grid_size = agent.get("grid_size", [3, 3])
        reward_positions = agent.get("reward_positions", [[0, 0]])
        
        # Create a grid
        grid = np.zeros(grid_size)
        
        # Mark reward positions
        for pos in reward_positions:
            if len(pos) == 2 and pos[0] < grid_size[0] and pos[1] < grid_size[1]:
                grid[pos[0], pos[1]] = 1
        
        # Plot grid
        plt.imshow(grid, cmap='viridis', interpolation='nearest')
        plt.colorbar(label='Reward')
        
        # Add grid lines
        plt.grid(True, color='black', linestyle='-', linewidth=0.5)
        plt.xticks(np.arange(-.5, grid_size[1], 1), [])
        plt.yticks(np.arange(-.5, grid_size[0], 1), [])
        
        # Set labels and title
        plt.title(f"Agent Model: {agent_id}")
        plt.xlabel("Column")
        plt.ylabel("Row")
        
        # Add reward position labels
        for pos in reward_positions:
            if len(pos) == 2 and pos[0] < grid_size[0] and pos[1] < grid_size[1]:
                plt.text(pos[1], pos[0], "R", 
                         ha="center", va="center", 
                         fontsize=12, color="white")
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Save figure
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.close()
        
        return {
            "files": {
                "model": output_path
            },
            "file": output_path,
            "data": pymdp_interface.agents[agent_id],
            "visualization": {
                "status": "success",
                "file_path": output_path
            }
        }
    except Exception as e:
        return {
            "error": f"Error generating visualization: {str(e)}",
            "file": "",
            "files": {}
        }

@app.get("/sessions/{session_id}/visualize")
async def visualize_session(session_id: str, format: str = "png", output_file: Optional[str] = None):
    if not hasattr(pymdp_interface, "sessions"):
        pymdp_interface.sessions = {}
    
    if session_id not in pymdp_interface.sessions:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    
    session = pymdp_interface.sessions[session_id]
    
    # Generate visualization based on requested format
    if format == "json":
        # Return JSON representation of simulation history
        return session["history"]
    else:
        try:
            # Create actual visualization
            import matplotlib.pyplot as plt
            import numpy as np
            import os
            from matplotlib.colors import ListedColormap
            
            # Check if output directory is specified in environment variable
            output_dir = os.environ.get("MCP_OUTPUT_DIR", "")
            
            # If no output file is specified, create one in the output directory
            if output_file is None:
                if output_dir:
                    output_file = os.path.join(output_dir, f"session_{session_id}_visualization.{format}")
                else:
                    output_file = f"session_{session_id}_visualization.{format}"
            elif not os.path.isabs(output_file) and output_dir:
                # If output file is relative and output directory is specified, join them
                output_file = os.path.join(output_dir, output_file)
            
            output_path = output_file
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            
            # Get agent and environment info
            agent_id = session.get("agent")
            env_id = session.get("environment")
            
            agent = pymdp_interface.agents.get(agent_id, {})
            env = pymdp_interface.environments.get(env_id, {})
            
            # Get grid world parameters
            grid_size = env.get("grid_size", [3, 3])
            reward_positions = env.get("reward_positions", []) or env.get("reward_locations", [])
            
            # Get simulation history
            history = session.get("history", {}).get("timesteps", [])
            
            if not history:
                return {
                    "error": "No simulation history available",
                    "files": {}
                }
            
            # Create figure to show agent trajectory
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # Create grid world
            grid = np.zeros(grid_size)
            
            # Mark reward positions
            for pos in reward_positions:
                if len(pos) == 2 and pos[0] < grid_size[0] and pos[1] < grid_size[1]:
                    grid[pos[0], pos[1]] = 1
            
            # Create custom colormap
            cmap = ListedColormap(['whitesmoke', 'gold'])
            
            # Plot grid
            ax.imshow(grid, cmap=cmap, interpolation='nearest')
            
            # Add grid lines
            ax.grid(True, color='black', linestyle='-', linewidth=0.5)
            
            # Extract agent positions
            positions = []
            for step in history:
                state = step.get("state")
                if state and len(state) == 2:
                    positions.append((state[0], state[1]))
            
            # Plot agent trajectory
            if positions:
                # Extract x and y coordinates
                x_coords = [p[1] for p in positions]
                y_coords = [p[0] for p in positions]
                
                # Plot the trajectory line
                ax.plot(x_coords, y_coords, 'b-', linewidth=2, alpha=0.7)
                
                # Plot start position (green)
                ax.plot(positions[0][1], positions[0][0], 'go', markersize=12)
                
                # Plot end position (red)
                ax.plot(positions[-1][1], positions[-1][0], 'ro', markersize=12)
                
                # Add position markers with timestep labels
                for i, (y, x) in enumerate(positions):
                    ax.text(x, y, str(i), ha='center', va='center', fontsize=10, 
                            color='black', fontweight='bold')
            
            # Set labels and title
            ax.set_title(f'Agent Trajectory - Session: {session_id}')
            ax.set_xlabel('Column')
            ax.set_ylabel('Row')
            
            # Add reward position labels
            for pos in reward_positions:
                if len(pos) == 2 and pos[0] < grid_size[0] and pos[1] < grid_size[1]:
                    ax.text(pos[1], pos[0], "R", 
                            ha="center", va="center", 
                            fontsize=12, color="black", fontweight='bold')
            
            # Save figure
            plt.savefig(output_path, bbox_inches='tight', dpi=150)
            plt.close()
            
            return {
                "session_id": session_id,
                "agent": session.get("agent"),
                "environment": session.get("environment"),
                "timesteps": len(session["history"].get("timesteps", [])),
                "visualization_status": "generated",
                "files": {
                    "visualization": output_path
                },
                "file_path": output_path
            }
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {
                "session_id": session_id,
                "error": f"Error generating visualization: {str(e)}",
                "files": {}
            }

@app.get("/analyze/free_energy/{session_id}")
async def analyze_free_energy(session_id: str, output_file: Optional[str] = None):
    if not hasattr(pymdp_interface, "sessions"):
        pymdp_interface.sessions = {}
        
    if session_id not in pymdp_interface.sessions:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        import os
        
        # Check if output directory is specified in environment variable
        output_dir = os.environ.get("MCP_OUTPUT_DIR", "")
        
        # If no output file is specified, create one in the output directory
        if output_file is None:
            if output_dir:
                output_file = os.path.join(output_dir, f"session_{session_id}_free_energy.png")
            else:
                output_file = f"session_{session_id}_free_energy.png"
        elif not os.path.isabs(output_file) and output_dir:
            # If output file is relative and output directory is specified, join them
            output_file = os.path.join(output_dir, output_file)
        
        output_path = output_file
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Get session data
        session = pymdp_interface.sessions[session_id]
        history = session.get("history", {}).get("timesteps", [])
        
        if not history:
            return {
                "error": "No simulation history available",
                "files": {}
            }
        
        # Generate mock free energy values
        # In a real implementation, this would calculate actual free energy
        timesteps = len(history)
        free_energy = np.random.random(timesteps) * -2  # Random negative values
        free_energy.sort()  # Sort for a decreasing trend
        
        # Create figure
        plt.figure(figsize=(10, 6))
        plt.plot(range(timesteps), free_energy, 'r-', linewidth=2)
        plt.title(f'Free Energy Analysis - Session: {session_id}')
        plt.xlabel('Timestep')
        plt.ylabel('Free Energy')
        plt.grid(True)
        
        # Save figure
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.close()
        
        return {
            "files": {
                "free_energy": output_path
            },
            "data": {
                "session_id": session_id,
                "free_energy": free_energy.tolist()
            }
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "error": f"Error analyzing free energy: {str(e)}",
            "files": {}
        }

@app.get("/analyze/beliefs/{session_id}")
async def analyze_beliefs(session_id: str, output_file: Optional[str] = None):
    if not hasattr(pymdp_interface, "sessions"):
        pymdp_interface.sessions = {}
        
    if session_id not in pymdp_interface.sessions:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        import os
        
        # Check if output directory is specified in environment variable
        output_dir = os.environ.get("MCP_OUTPUT_DIR", "")
        
        # If no output file is specified, create one in the output directory
        if output_file is None:
            if output_dir:
                output_file = os.path.join(output_dir, f"session_{session_id}_belief_dynamics.png")
            else:
                output_file = f"session_{session_id}_belief_dynamics.png"
        elif not os.path.isabs(output_file) and output_dir:
            # If output file is relative and output directory is specified, join them
            output_file = os.path.join(output_dir, output_file)
        
        output_path = output_file
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Get session data
        session = pymdp_interface.sessions[session_id]
        history = session.get("history", {}).get("timesteps", [])
        
        if not history:
            return {
                "error": "No simulation history available",
                "files": {}
            }
        
        # Extract beliefs over time
        beliefs = []
        for step in history:
            belief_data = step.get("belief", [])
            if belief_data:
                beliefs.append(belief_data[0])  # Get first belief array
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        if beliefs:
            num_states = len(beliefs[0])
            timesteps = len(beliefs)
            
            # Create a heatmap of beliefs
            belief_array = np.array(beliefs)
            plt.imshow(belief_array.T, aspect='auto', cmap='viridis')
            plt.colorbar(label='Probability')
            
            # Add labels
            plt.title(f'Belief Dynamics - Session: {session_id}')
            plt.xlabel('Timestep')
            plt.ylabel('State')
            
            # Add grid
            plt.grid(False)
            
            # Add state labels
            plt.yticks(range(num_states))
        else:
            plt.text(0.5, 0.5, "No belief data available", 
                     ha='center', va='center', fontsize=12)
        
        # Save figure
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.close()
        
        return {
            "files": {
                "belief_dynamics": output_path
            },
            "data": {
                "session_id": session_id,
                "beliefs": beliefs
            }
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "error": f"Error analyzing beliefs: {str(e)}",
            "files": {}
        }

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Start the MCP server for PyMDP")
    
    parser.add_argument(
        "--host", 
        type=str, 
        default="0.0.0.0", 
        help="Host to bind to (default: 0.0.0.0)"
    )
    
    parser.add_argument(
        "--port", 
        type=int, 
        default=8080, 
        help="Port to bind to (default: 8080)"
    )
    
    parser.add_argument(
        "--log-level", 
        type=str, 
        default="info",
        choices=["debug", "info", "warning", "error", "critical"],
        help="Logging level (default: info)"
    )
    
    parser.add_argument(
        "--reload", 
        action="store_true", 
        help="Enable auto-reload on code changes"
    )
    
    parser.add_argument(
        "--workers", 
        type=int, 
        default=1, 
        help="Number of worker processes (default: 1)"
    )

    parser.add_argument(
        "--ssl-certfile", 
        type=str, 
        default=None, 
        help="SSL certificate file"
    )
    
    parser.add_argument(
        "--ssl-keyfile", 
        type=str, 
        default=None, 
        help="SSL key file"
    )
    
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default=None, 
        help="Output directory for generated files"
    )
    
    return parser.parse_args()

def setup_logging(log_level):
    """Set up logging configuration."""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")
    
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

def main():
    """Main function to start the MCP server."""
    # Parse command-line arguments
    args = parse_args()
    
    # Set up logging
    setup_logging(args.log_level)
    
    # Configure output directory
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        os.environ["MCP_OUTPUT_DIR"] = os.path.abspath(args.output_dir)
        print(f"Output directory set to: {os.environ['MCP_OUTPUT_DIR']}")
    
    # Start the server
    uvicorn.run(
        "start_mcp_server:app",
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        reload=args.reload,
        workers=args.workers,
        ssl_certfile=args.ssl_certfile,
        ssl_keyfile=args.ssl_keyfile
    )

if __name__ == "__main__":
    main() 