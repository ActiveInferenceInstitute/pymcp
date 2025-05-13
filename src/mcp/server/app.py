"""
MCP Server FastAPI Application.

This module provides a FastAPI application for the MCP server.
"""

import os
import json
import sys
import importlib.util
from pathlib import Path
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging
from typing import Dict, Any
import traceback
import time

# Import PyMDP interface using absolute import instead of relative
from mcp.utils import PyMDPInterface

# Import visualization extension module to add methods to PyMDPInterface
from mcp.visualization.extensions import _add_visualization_methods

# Create a global instance of the PyMDP interface
pymdp_interface = PyMDPInterface()

logger = logging.getLogger("mcp.server")

def create_app():
    """Create the FastAPI application for the MCP server.
    
    Returns
    -------
    FastAPI
        FastAPI application
    """
    # Create FastAPI app
    app = FastAPI(
        title="MCP-PyMDP Server",
        description="Message-based Cognitive Protocol for Active Inference with PyMDP",
        version="0.1.0"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Add ping endpoint
    @app.get("/ping")
    async def ping():
        return {"status": "ok", "message": "MCP-PyMDP server is running"}
    
    # Add tools endpoint
    @app.get("/tools")
    async def get_tools():
        tools = [
            {"id": "create_agent", "name": "Create Agent", "description": "Create a custom agent from a generative model"},
            {"id": "create_gridworld_agent", "name": "Create Gridworld Agent", "description": "Create an agent for a grid world environment"},
            {"id": "create_environment", "name": "Create Environment", "description": "Create a simulation environment"},
            {"id": "infer_states", "name": "Infer States", "description": "Perform state inference for an agent"},
            {"id": "run_simulation", "name": "Run Simulation", "description": "Run a simulation with an agent in an environment"}
        ]
        return {"tools": tools}

    # Add tool invocation endpoints
    @app.post("/tools/{tool_id}")
    async def invoke_tool(tool_id: str, request: Request):
        # Parse request data
        data = await request.json()
        params = data.get("params", {})
        
        # Handle different tools
        if tool_id == "create_gridworld_agent":
            name = params.get("name", "GridWorldAgent")
            grid_size = params.get("grid_size", [3, 3])
            reward_positions = params.get("reward_positions", [[2, 2]])
            
            # Create grid world agent using PyMDP interface
            height, width = grid_size
            result = pymdp_interface.create_grid_world_env(name, [height, width], reward_positions)
            
            # Create an agent with appropriate parameters for this environment
            agent_name = f"{name}_Agent"
            
            # The grid size determines the state factor dimensions
            
            # A simple grid world agent has two observation modalities:
            # 1. Position observation (height*width possibilities)
            # 2. Reward observation (2 possibilities: reward or no reward)
            A_dims = [[height*width, height*width], [2, height*width]]
            
            # One state factor with 4 possible actions (up, down, left, right)
            B_dims = [[height*width, height*width, 4]]
            
            # Generate the generative model
            model_result = pymdp_interface.define_generative_model(A_dims, B_dims)
            
            if "error" in model_result:
                return {"result": {"error": model_result["error"]}}
                
            # Create the agent
            agent_result = pymdp_interface.create_agent(agent_name, model_result)
            
            if "error" in agent_result:
                return {"result": {"error": agent_result["error"]}}
                
            # Return the agent information with ID explicitly set to match name
            agent_result["id"] = agent_name
            agent_result["env_created"] = result
            
            return {"result": {"agent": agent_result}}
            
        elif tool_id == "create_agent":
            name = params.get("name")
            model = params.get("model", {})
            
            # Create agent using PyMDP interface
            result = pymdp_interface.create_agent(name, model)
            
            # Make sure the ID is set to the agent's name
            if "error" not in result:
                result["id"] = name
            
            return {"result": {"agent": result}}
        
        elif tool_id == "create_environment":
            name = params.get("name")
            type = params.get("type", "grid_world")
            grid_size = params.get("grid_size", [3, 3])
            reward_locations = params.get("reward_locations", [[2, 2]])
            
            # Create the environment
            result = pymdp_interface.create_grid_world_env(name, grid_size, reward_locations)
            
            # Make sure the ID is set
            if "error" not in result:
                result["id"] = name
                result["name"] = name
                result["type"] = type
                
            return {"result": result}
            
        elif tool_id == "infer_states":
            agent_name = params.get("agent_name")
            observation = params.get("observation", [0])
            method = params.get("method", "FPI")
            
            result = pymdp_interface.infer_states(agent_name, observation, method)
            
            return {"result": result}
            
        elif tool_id == "infer_policies":
            agent_name = params.get("agent_name")
            
            result = pymdp_interface.infer_policies(agent_name)
            
            return {"result": result}
            
        elif tool_id == "sample_action":
            agent_name = params.get("agent_name")
            
            result = pymdp_interface.sample_action(agent_name)
            
            return {"result": result}
            
        elif tool_id == "run_simulation":
            agent_id = params.get("agent_id")
            env_id = params.get("env_id")
            steps = params.get("steps", 10)
            
            # Check if the agent and environment exist
            agent_result = pymdp_interface.get_agent(agent_id)
            if "error" in agent_result:
                return {"result": {"error": f"Agent '{agent_id}' not found"}}
                
            env_result = pymdp_interface.get_environment(env_id)
            if "error" in env_result:
                return {"result": {"error": f"Environment '{env_id}' not found"}}
            
            # Create a session if one doesn't exist
            session_id = f"{agent_id}_{env_id}_session"
            
            if not hasattr(pymdp_interface, "sessions"):
                pymdp_interface.sessions = {}
                
            if session_id not in pymdp_interface.sessions:
                session = {
                    "id": session_id,
                    "agent": agent_id,
                    "environment": env_id,
                    "history": {
                        "timesteps": []
                    }
                }
                pymdp_interface.sessions[session_id] = session
            
            # Now run the simulation
            result = pymdp_interface.run_simulation(agent_id, env_id, steps)
            
            # Ensure the simulation has an ID
            if "error" not in result:
                result["id"] = session_id
            
            return {"result": result}
            
        elif tool_id == "step_environment":
            env_name = params.get("env_name")
            action = params.get("action", [0])
            
            result = pymdp_interface.step_environment(env_name, action)
            
            return {"result": result}
            
        elif tool_id == "visualize_simulation":
            simulation_id = params.get("simulation_id")
            output_file = params.get("output_file", "simulation.png")
            
            result = pymdp_interface.visualize_simulation(simulation_id, output_file)
            
            return {"result": result}
            
        else:
            # Check if the tool is available in the PyMDP interface
            available_functions = pymdp_interface.get_all_functions()
            if tool_id in available_functions.get("functions", []):
                # Call the method dynamically
                method = getattr(pymdp_interface, tool_id, None)
                if method:
                    try:
                        result = method(**params)
                        return {"result": result}
                    except Exception as e:
                        return {"result": {"error": f"Error calling {tool_id}: {str(e)}"}}
            
            # Unknown tool
            raise HTTPException(status_code=404, detail=f"Tool {tool_id} not found")
    
    # Add endpoints for environment management
    @app.post("/environments")
    async def create_environment(request: Request):
        data = await request.json()
        name = data.get("name")
        
        # Default to grid_world environment type
        type = data.get("type", "grid_world")
        
        # Get specific parameters from the request body
        grid_size = data.get("grid_size", [3, 3])
        reward_locations = data.get("reward_locations", [[2, 2]])
        
        # For backward compatibility, also check in params
        params = data.get("params", {})
        if not grid_size and "grid_size" in params:
            grid_size = params["grid_size"]
        if not reward_locations and "reward_locations" in params:
            reward_locations = params["reward_locations"]
            
        # Create the environment
        if type == "grid_world":
            result = pymdp_interface.create_grid_world_env(name, grid_size, reward_locations)
        else:
            return {"error": f"Environment type {type} not supported"}
        
        # Make sure the ID is set
        if "error" not in result:
            result["id"] = name
            result["name"] = name
            result["type"] = type
        
        return result
    
    @app.get("/environments")
    async def get_environments():
        result = pymdp_interface.get_all_environments()
        return {"environments": result.get("environments", [])}
    
    @app.get("/environments/{env_id}")
    async def get_environment(env_id: str):
        result = pymdp_interface.get_environment(env_id)
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        
        return result
    
    # Add endpoints for agent management
    @app.get("/agents")
    async def get_agents():
        result = pymdp_interface.get_all_agents()
        return {"agents": result.get("agents", [])}
    
    @app.get("/agents/{agent_id}")
    async def get_agent(agent_id: str):
        result = pymdp_interface.get_agent(agent_id)
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        
        return result
    
    # Add endpoints for simulation management
    @app.get("/simulations")
    async def get_simulations():
        result = pymdp_interface.get_all_simulations()
        return {"simulations": result.get("simulations", [])}
    
    @app.post("/sessions")
    async def create_session(request: Request):
        """Create a new simulation session.
        
        A session connects an agent with an environment for simulation.
        
        Returns:
            Session information
        """
        try:
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
            
            # Check agent exists
            agent_result = pymdp_interface.get_agent(agent_id)
            if "error" in agent_result:
                return Response(
                    content=json.dumps({"error": f"Agent '{agent_id}' not found"}),
                    media_type="application/json", 
                    status_code=404
                )
            
            # Check environment exists
            env_result = pymdp_interface.get_environment(env_id)
            
            # Handle case when environment is not a dict (GridWorldEnv object)
            if isinstance(env_result, dict) and "error" in env_result:
                return Response(
                    content=json.dumps({"error": f"Environment '{env_id}' not found"}),
                    media_type="application/json",
                    status_code=404
                )
            
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
        except Exception as e:
            logger.error(f"Error creating session: {str(e)}")
            logger.error(traceback.format_exc())
            return {"error": f"Error creating session: {str(e)}"}
    
    @app.post("/sessions/{session_id}/run")
    async def run_session(session_id: str, request: Request):
        """Run a simulation session for a specified number of steps.
        
        Args:
            session_id: ID of the session to run
            request: Request with simulation parameters
            
        Returns:
            Simulation results
        """
        try:
            # Check if sessions storage exists
            if not hasattr(pymdp_interface, "sessions"):
                pymdp_interface.sessions = {}
                
            # Check if session exists
            if session_id not in pymdp_interface.sessions:
                raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
            
            # Get session data
            session = pymdp_interface.sessions[session_id]
            agent_id = session.get("agent")
            env_id = session.get("environment")
            
            if not agent_id or not env_id:
                raise HTTPException(status_code=400, detail="Session is missing agent or environment")
            
            # Parse request data
            data = await request.json()
            steps = data.get("steps", 1)
            
            # Run simulation
            result = pymdp_interface.run_simulation(agent_id, env_id, steps)
            
            # Check for errors
            if "error" in result:
                raise HTTPException(status_code=500, detail=result["error"])
            
            # Add session info to result
            result["session_id"] = session_id
            
            return result
        except Exception as e:
            logger.error(f"Error running session: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=str(e))

    # Add visualization endpoints
    @app.get("/agents/{agent_id}/visualize")
    async def visualize_agent(agent_id: str):
        try:
            if agent_id not in pymdp_interface.agents:
                raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
            
            # Generate visualization using the PyMDPInterface method
            output_file = f"agent_{agent_id}_model.png"
            result = pymdp_interface.visualize_agent_model(agent_id, output_file)
            
            # Check if visualization was successful
            if "error" in result:
                raise HTTPException(status_code=500, detail=result["error"])
                
            # If visualization file was generated, create a response with the file path
            if "file_path" in result and os.path.exists(result["file_path"]):
                # Get the agent data for additional information
                agent_data = pymdp_interface.get_agent(agent_id)
                
                return {
                    "files": {
                        "model": result["file_path"]
                    },
                    "data": agent_data,
                    "visualization": result
                }
            else:
                raise HTTPException(status_code=500, detail="Failed to generate agent visualization")
        except Exception as e:
            logger.error(f"Error visualizing agent: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/sessions/{session_id}/visualize")
    async def visualize_session(session_id: str, format: str = "png"):
        """Visualize a simulation session.
        
        Args:
            session_id: ID of the session to visualize
            format: Output format (png, svg, or json)
            
        Returns:
            Visualization data or error message
        """
        try:
            # Check if sessions storage exists
            if not hasattr(pymdp_interface, "sessions"):
                pymdp_interface.sessions = {}
            
            # Check if session exists
            if session_id not in pymdp_interface.sessions:
                return Response(
                    content=json.dumps({"error": f"Session {session_id} not found"}),
                    media_type="application/json",
                    status_code=404
                )
            
            # Get session data
            session = pymdp_interface.sessions[session_id]
            agent_id = session.get("agent")
            env_id = session.get("environment")
            
            # Check if we have history data to visualize
            if "history" not in session or not session["history"].get("timesteps"):
                return Response(
                    content=json.dumps({"error": "No simulation data available for visualization"}),
                    media_type="application/json",
                    status_code=400
                )
            
            # Generate visualization based on requested format
            if format == "json":
                # Return JSON representation of simulation history
                return session["history"]
            elif format in ["png", "svg"]:
                # Call the PyMDP interface to generate visualization
                output_file = f"session_{session_id}_visualization.{format}"
                result = pymdp_interface.visualize_simulation(session_id, output_file)
                
                # Check if visualization was successful
                if "error" in result:
                    return Response(
                        content=json.dumps(result),
                        media_type="application/json",
                        status_code=500
                    )
                
                # If visualization file was generated, return it
                if "file_path" in result and os.path.exists(result["file_path"]):
                    with open(result["file_path"], "rb") as f:
                        image_data = f.read()
                
                    media_type = "image/png" if format == "png" else "image/svg+xml"
                    return Response(content=image_data, media_type=media_type)
                else:
                    # Return placeholder visualization with session data
                    visualization_data = {
                        "session_id": session_id,
                        "agent": agent_id,
                        "environment": env_id,
                        "timesteps": len(session["history"].get("timesteps", [])),
                        "visualization_status": "generated"
                    }
                
                    return visualization_data
            else:
                return Response(
                    content=json.dumps({"error": f"Unsupported format: {format}. Use 'png', 'svg', or 'json'"}),
                    media_type="application/json",
                    status_code=400
                )
        except Exception as e:
            logger.error(f"Error visualizing session: {str(e)}")
            logger.error(traceback.format_exc())
            return Response(
                content=json.dumps({"error": f"Error visualizing session: {str(e)}"}),
                media_type="application/json",
                status_code=500
            )
    
    @app.get("/analyze/free_energy/{session_id}")
    async def analyze_free_energy(session_id: str):
        try:
            # Check if exists in PyMDPInterface
            if session_id not in pymdp_interface.sessions:
                raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
            
            # Generate free energy analysis
            output_file = f"session_{session_id}_free_energy.png"
            result = pymdp_interface.analyze_free_energy(session_id, output_file)
            
            return {
                "files": {
                    "free_energy": output_file
                },
                "data": result
            }
        except Exception as e:
            logger.error(f"Error analyzing free energy: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/analyze/beliefs/{session_id}")
    async def analyze_beliefs(session_id: str):
        try:
            # Check if exists in PyMDPInterface
            if session_id not in pymdp_interface.sessions:
                raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
            
            # Generate belief dynamics visualization
            output_file = f"session_{session_id}_belief_dynamics.png"
            result = pymdp_interface.plot_belief_dynamics(session_id, output_file)
            
            return {
                "files": {
                    "belief_dynamics": output_file
                },
                "data": result
            }
        except Exception as e:
            logger.error(f"Error analyzing beliefs: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=str(e))
            
    # Add endpoint to create gridworld environment
    @app.post("/tools/create_environment")
    async def create_gridworld_environment(request: Request):
        try:
            data = await request.json()
            params = data.get("params", {})
            
            name = params.get("name", "GridWorldEnv")
            grid_size = params.get("grid_size", [3, 3])
            reward_locations = params.get("reward_locations", [[2, 2]])
            
            # Create environment
            environment = {
                "type": "grid_world",
                "name": name,
                "grid_size": grid_size,
                "reward_locations": reward_locations
            }
            
            # Store in the interface
            pymdp_interface.environments[name] = environment
            
            return {"id": name, **environment}
        except Exception as e:
            logger.error(f"Error creating environment: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=str(e))
    
    # Add route to list sessions
    @app.get("/sessions")
    async def get_sessions():
        """Get all available sessions.
        
        Returns:
            List of session IDs
        """
        try:
            # Initialize sessions if not already done
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
        except Exception as e:
            logger.error(f"Error getting sessions: {str(e)}")
            logger.error(traceback.format_exc())
            return Response(
                content=json.dumps({"error": f"Error getting sessions: {str(e)}"}),
                media_type="application/json",
                status_code=500
            )
    
    return app

# Instantiate the app at the module level for Uvicorn
app = create_app()

if __name__ == "__main__":
    # This block is now only for direct execution of this file,
    # Uvicorn will use the 'app' instance defined above.
    uvicorn.run(app, host="0.0.0.0", port=8080) 