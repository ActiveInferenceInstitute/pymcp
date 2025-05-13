"""
MCP Client Tools.

This module provides high-level wrappers for MCP tools.
"""

import os
import json
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
from pathlib import Path
import asyncio

from .core import MCPClient

class MCPToolKit:
    """High-level toolkit for working with MCP tools.
    
    This class provides convenience methods for working with MCP tools
    without having to directly use the low-level MCP client.
    
    Parameters
    ----------
    client : MCPClient
        MCP client to use
    
    Attributes
    ----------
    client : MCPClient
        MCP client for communicating with the server
    """
    
    def __init__(self, client: MCPClient):
        """Initialize the MCP toolkit."""
        self.client = client
    
    async def create_agent_from_model(
        self,
        name: str,
        A: Union[List[List[float]], List[np.ndarray]],
        B: Union[List[List[float]], List[np.ndarray]],
        C: Union[List[List[float]], List[np.ndarray]],
        D: Optional[Union[List[List[float]], List[np.ndarray]]] = None
    ) -> Dict[str, Any]:
        """Create an agent from generative model matrices.
        
        Parameters
        ----------
        name : str
            Agent name
        A : List[List[float]] or List[np.ndarray]
            A matrices (observation model)
        B : List[List[float]] or List[np.ndarray]
            B matrices (transition model)
        C : List[List[float]] or List[np.ndarray]
            C matrices (preferences)
        D : List[List[float]] or List[np.ndarray], optional
            D matrices (prior beliefs), by default None
        
        Returns
        -------
        Dict[str, Any]
            Agent information
        """
        # Convert numpy arrays to lists if needed
        A_list = [a.tolist() if isinstance(a, np.ndarray) else a for a in A]
        B_list = [b.tolist() if isinstance(b, np.ndarray) else b for b in B]
        C_list = [c.tolist() if isinstance(c, np.ndarray) else c for c in C]
        
        # Create model dictionary
        model = {
            "A": A_list,
            "B": B_list,
            "C": C_list
        }
        
        # Add D matrices if provided
        if D is not None:
            D_list = [d.tolist() if isinstance(d, np.ndarray) else d for d in D]
            model["D"] = D_list
        
        # Create agent
        return await self.client.create_agent(name, model)
    
    async def create_gridworld_agent(
        self,
        name: str,
        grid_size: List[int],
        reward_positions: List[List[int]]
    ) -> Dict[str, Any]:
        """Create an agent for a grid world environment.
        
        Parameters
        ----------
        name : str
            Agent name
        grid_size : List[int]
            Grid size [height, width]
        reward_positions : List[List[int]]
            List of reward positions [[row, col], ...]
        
        Returns
        -------
        Dict[str, Any]
            Agent information
        """
        # Call grid world agent tool
        params = {
            "name": name,
            "grid_size": grid_size,
            "reward_positions": reward_positions
        }
        
        result = await self.client.call_tool("create_gridworld_agent", params)
        return result.get("agent", {})
    
    async def create_gridworld_environment(
        self,
        name: str,
        grid_size: List[int],
        reward_positions: List[List[int]]
    ) -> Dict[str, Any]:
        """Create a grid world environment.
        
        Parameters
        ----------
        name : str
            Environment name
        grid_size : List[int]
            Grid size [height, width]
        reward_positions : List[List[int]]
            List of reward positions [[row, col], ...]
        
        Returns
        -------
        Dict[str, Any]
            Environment information
        """
        # Create environment parameters
        params = {
            "name": name,
            "grid_size": grid_size,
            "reward_locations": reward_positions
        }
        
        # Create environment
        result = await self.client.call_tool("create_environment", params)
        env = result.get("result", {})
        
        # Ensure environment has an ID
        if "id" not in env and "error" not in env:
            env["id"] = name
            
        return env
    
    async def create_custom_environment(
        self,
        name: str,
        states: List[str],
        observations: List[str],
        transitions: Dict[str, Dict[str, str]],
        rewards: Dict[str, float]
    ) -> Dict[str, Any]:
        """Create a custom environment.
        
        Parameters
        ----------
        name : str
            Environment name
        states : List[str]
            List of state names
        observations : List[str]
            List of observation names
        transitions : Dict[str, Dict[str, str]]
            Transitions dictionary (state -> action -> next state)
        rewards : Dict[str, float]
            Rewards dictionary (state -> reward)
        
        Returns
        -------
        Dict[str, Any]
            Environment information
        """
        # Create params dictionary
        params = {
            "states": states,
            "observations": observations,
            "transitions": transitions,
            "rewards": rewards
        }
        
        # Create environment
        return await self.client.create_environment(name, "custom", params)
    
    async def run_simulation(
        self, 
        agent_id: str, 
        env_id: str, 
        num_steps: int = 10, 
        save_history: bool = True, 
        planning_horizon: Optional[int] = None, 
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """Run a simulation with an agent in an environment.
        
        Parameters
        ----------
        agent_id : str
            Agent ID
        env_id : str
            Environment ID
        num_steps : int, optional
            Number of steps to run, by default 10
        save_history : bool, optional
            Whether to save history, by default True
        planning_horizon : int, optional
            Planning horizon for policy inference, by default None
        output_dir : str, optional
            Directory to save outputs, by default None
        
        Returns
        -------
        Dict[str, Any]
            Simulation results
        """
        # Prepare parameters
        params = {
            "agent_id": agent_id,
            "environment_id": env_id,
            "num_steps": num_steps,
            "save_history": save_history
        }
        
        # Add optional parameters
        if planning_horizon is not None:
            params["planning_horizon"] = planning_horizon
        
        if output_dir:
            params["output_dir"] = output_dir
        
        # Call the tool
        result = await self.client.call_tool("run_simulation", params)
        
        # Extract the result from the response if needed
        if isinstance(result, dict) and "result" in result:
            return result["result"]
        
        return result
    
    async def visualize_agent_model(
        self,
        agent_id: str,
        output_file: Optional[str] = None,
        format: str = "png"
    ) -> str:
        """Visualize an agent's generative model.
        
        Parameters
        ----------
        agent_id : str
            Agent ID
        output_file : str, optional
            Path to save the visualization, by default None
        format : str, optional
            Visualization format, by default "png"
        
        Returns
        -------
        str
            Path to the saved visualization
        """
        # Get visualization
        params = {}
        if output_file is not None:
            params["output_file"] = output_file
        
        result = await self.client.visualize_agent(agent_id, format, params)
        
        # Return file path
        return result.get("file", "")
    
    async def visualize_simulation(
        self,
        session_id: str,
        output_file: Optional[str] = None,
        format: str = "png",
        include_beliefs: bool = True,
        include_policies: bool = True
    ) -> Dict[str, str]:
        """Visualize a simulation.
        
        Parameters
        ----------
        session_id : str
            Session ID
        output_file : str, optional
            Path to save the visualization, by default None
        format : str, optional
            Visualization format, by default "png"
        include_beliefs : bool, optional
            Whether to include belief visualization, by default True
        include_policies : bool, optional
            Whether to include policy visualization, by default True
        
        Returns
        -------
        Dict[str, str]
            Dictionary of visualization file paths
        """
        # Get visualization
        params = {
            "include_beliefs": include_beliefs,
            "include_policies": include_policies
        }
        if output_file is not None:
            params["output_file"] = output_file
        
        result = await self.client.visualize_session(session_id, format, params)
        
        # Return file paths
        files = result.get("files", {})
        if isinstance(files, list):
            return {"files": files}
        else:
            return files
    
    async def analyze_free_energy(
        self,
        session_id: str,
        output_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """Analyze free energy from a simulation.
        
        Parameters
        ----------
        session_id : str
            Session ID
        output_file : str, optional
            Path to save the analysis, by default None
        
        Returns
        -------
        Dict[str, Any]
            Analysis results
        """
        # Call free energy analysis tool
        params = {
            "session_id": session_id
        }
        if output_file is not None:
            params["output_file"] = output_file
        
        result = await self.client.call_tool("analyze_free_energy", params)
        return result
    
    async def plot_belief_dynamics(
        self,
        session_id: str,
        output_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """Plot belief dynamics from a simulation.
        
        Parameters
        ----------
        session_id : str
            Session ID
        output_file : str, optional
            Path to save the plot, by default None
        
        Returns
        -------
        Dict[str, Any]
            Plot results
        """
        # Call belief dynamics tool
        params = {
            "session_id": session_id
        }
        if output_file is not None:
            params["output_file"] = output_file
        
        result = await self.client.call_tool("plot_belief_dynamics", params)
        return result
    
    async def validate_generative_model(
        self,
        model: Dict[str, List],
        check_normalization: bool = True
    ) -> Dict[str, Any]:
        """Validate a generative model.
        
        Parameters
        ----------
        model : Dict[str, List]
            Generative model (A, B, C, D matrices)
        check_normalization : bool, optional
            Whether to check normalization, by default True
        
        Returns
        -------
        Dict[str, Any]
            Validation results
        """
        # Call validation tool
        params = {
            "model": model,
            "check_normalization": check_normalization
        }
        
        result = await self.client.call_tool("validate_generative_model", params)
        return result
    
    async def compare_generative_models(
        self,
        model1: Dict[str, List],
        model2: Dict[str, List]
    ) -> Dict[str, Any]:
        """Compare two generative models.
        
        Parameters
        ----------
        model1 : Dict[str, List]
            First generative model
        model2 : Dict[str, List]
            Second generative model
        
        Returns
        -------
        Dict[str, Any]
            Comparison results
        """
        # Call comparison tool
        params = {
            "model1": model1,
            "model2": model2
        }
        
        result = await self.client.call_tool("compare_generative_models", params)
        return result
    
    async def run_agent_in_environment(
        self, 
        agent_id: str, 
        env_id: str, 
        num_steps: int = 10,
        save_history: bool = True, 
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """Run an agent in an environment (alias for run_simulation).
        
        Parameters
        ----------
        agent_id : str
            Agent ID
        env_id : str
            Environment ID
        num_steps : int, optional
            Number of steps to run, by default 10
        save_history : bool, optional
            Whether to save history, by default True
        output_dir : str, optional
            Directory to save outputs, by default None
        
        Returns
        -------
        Dict[str, Any]
            Simulation results
        """
        return await self.run_simulation(
            agent_id=agent_id,
            env_id=env_id,
            num_steps=num_steps,
            save_history=save_history,
            output_dir=output_dir
        ) 