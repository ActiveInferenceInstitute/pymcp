from typing import List, Dict, Tuple, Any, Union, Optional
import numpy as np
import matplotlib.pyplot as plt
import time
import json
import os
import logging

from pymdp.agent import Agent
from pymdp.envs import Env
import pymdp.utils as utils

# Set up logging
logger = logging.getLogger("mcp.utils")

# Add custom JSON encoder for NumPy types
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

class PyMDPInterface:
    """Interface class for PyMDP simulations and active inference."""
    
    def __init__(self):
        """Initialize the PyMDP interface."""
        self.agents = {}
        self.environments = {}
        self.simulation_history = {}
        self.computation_logs = {}  # Store detailed computation logs for each agent/simulation
        self.debug_mode = True  # Enable detailed logging by default
        self.sessions = {}  # For storing simulation sessions
    
    def create_agent(self, name: str, generative_model: Dict) -> Dict:
        """Create an agent from a generative model."""
        try:
            if self.debug_mode:
                print(f"Creating agent {name} with generative model")
                
            # Import required PyMDP modules
            from pymdp.agent import Agent
            import numpy as np
            
            # Extract generative model components
            A = generative_model.get("A", [])
            B = generative_model.get("B", [])
            C = generative_model.get("C", None)
            D = generative_model.get("D", None)
            
            # Convert lists to proper PyMDP numpy object arrays
            if A:
                # First convert list of lists to numpy arrays
                if isinstance(A[0], list):
                    A_arrays = [np.array(a) for a in A]
                else:
                    A_arrays = A  # Assume already numpy arrays
                
                # Normalize A matrices to ensure columns sum to 1
                A_arrays = [self._normalize_A_matrix(a) for a in A_arrays]
                
                # Then convert to object array as required by PyMDP
                A_obj = np.empty(len(A_arrays), dtype=object)
                for i, a in enumerate(A_arrays):
                    A_obj[i] = a
            
            if B:
                # First convert list of lists to numpy arrays
                if isinstance(B[0], list):
                    B_arrays = [np.array(b) for b in B]
                else:
                    B_arrays = B  # Assume already numpy arrays
                
                # Normalize B matrices along the "to" dimension
                for i, B_mat in enumerate(B_arrays):
                    for c in range(B_mat.shape[-1]):  # For each control state
                        for s in range(B_mat.shape[1]):  # For each state being transitioned from
                            # Normalize over states to transition to (axis 0)
                            if np.sum(B_mat[:, s, c]) > 0:
                                B_mat[:, s, c] = B_mat[:, s, c] / np.sum(B_mat[:, s, c])
                    B_arrays[i] = B_mat
                    
                # Then convert to object array as required by PyMDP
                B_obj = np.empty(len(B_arrays), dtype=object)
                for i, b in enumerate(B_arrays):
                    B_obj[i] = b
                
            if C is not None:
                # First convert list of lists to numpy arrays
                if isinstance(C[0], list):
                    C_arrays = [np.array(c) for c in C]
                else:
                    C_arrays = C  # Assume already numpy arrays
                
                # Then convert to object array as required by PyMDP
                C_obj = np.empty(len(C_arrays), dtype=object)
                for i, c in enumerate(C_arrays):
                    C_obj[i] = c
                
            if D is not None:
                # First convert list of lists to numpy arrays
                if isinstance(D[0], list):
                    D_arrays = [np.array(d) for d in D]
                else:
                    D_arrays = D  # Assume already numpy arrays
                
                # Then convert to object array as required by PyMDP
                D_obj = np.empty(len(D_arrays), dtype=object)
                for i, d in enumerate(D_arrays):
                    D_obj[i] = d
            
            # Create the PyMDP agent
            agent = Agent(A=A_obj, B=B_obj, C=C_obj if C is not None else None, D=D_obj if D is not None else None)
            
            # Add additional parameters if provided
            if "inference_horizon" in generative_model:
                agent.inference_horizon = int(generative_model["inference_horizon"])
            
            if "action_precision" in generative_model:
                agent.action_precision = float(generative_model["action_precision"])
                
            if "inference_algo" in generative_model:
                agent.inference_algo = generative_model["inference_algo"]
            
            # Store the agent
            self.agents[name] = agent
            
            # Return information about the agent
            result = {
                "name": name,
                "id": name,  # Set ID to be the same as name for consistency
                "num_observation_modalities": len(agent.A),
                "num_state_factors": len(agent.B),
                "num_controls": [b.shape[-1] for b in agent.B]
            }
            
            return result
        except Exception as e:
            import traceback
            logger.error(f"Error creating agent: {str(e)}")
            logger.error(traceback.format_exc())
            return {"error": str(e)}
    
    def define_generative_model(self, A_dims: List[List[int]], B_dims: List[List[int]]) -> Dict:
        """
        Create a generative model with random A, B matrices of the given dimensions.
        
        Args:
            A_dims: List of dimensions for each A matrix [obs_dim, state_dim]
            B_dims: List of dimensions for each B matrix [state_dim, state_dim, action_dim]
            
        Returns:
            Dict with A, B, C, D matrices
        """
        try:
            # Create random A matrices (observation model)
            A = []
            for dims in A_dims:
                if len(dims) == 2:
                    # Basic observation model: [obs_dim, state_dim]
                    obs_dim, state_dim = dims
                    a = np.random.dirichlet(np.ones(state_dim), size=obs_dim)
                    # Ensure normalization (columns sum to 1)
                    a = self._normalize_A_matrix(a)
                    A.append(a)
            
            # Create random B matrices (transition model)
            B = []
            for dims in B_dims:
                if len(dims) == 3:
                    # Basic transition model: [state_dim, state_dim, action_dim]
                    state_dim_to, state_dim_from, action_dim = dims
                    b = np.zeros((state_dim_to, state_dim_from, action_dim))
                    for a in range(action_dim):
                        b[:, :, a] = np.random.dirichlet(np.ones(state_dim_from), size=state_dim_to)
                    B.append(b)
            
            # Create default prior preferences (C) - flat preferences
            C = []
            for dims in A_dims:
                obs_dim = dims[0]
                c = np.zeros(obs_dim)
                C.append(c)
            
            # Create default prior beliefs (D) - uniform distribution
            D = []
            for dims in B_dims:
                state_dim = dims[0]
                d = np.ones(state_dim) / state_dim
                D.append(d)
            
            # Return the generative model
            return {
                "A": A,
                "B": B,
                "C": C,
                "D": D
            }
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"error": f"Error defining generative model: {str(e)}"}
            
    def create_gridworld_agent(self, name, grid_size, reward_positions, action_precision=1.0, inference_horizon=5):
        """
        Create a gridworld agent with the specified parameters.
        
        Args:
            name: Name of the agent
            grid_size: Grid dimensions [height, width]
            reward_positions: List of reward positions [[row, col], ...]
            action_precision: Precision parameter for action selection
            inference_horizon: Horizon for policy inference
            
        Returns:
            Dict with agent information
        """
        try:
            import traceback
            # Extract grid dimensions
            height, width = grid_size
            num_states = height * width
            
            # Define state dimensions for GridWorld
            state_dimensions = [num_states]
            control_factor_idx = [0]  # Single control factor
            
            # Create A and B dimensions for GridWorld
            A_dims = [[num_states, num_states], [2, num_states]]  # Location and reward observations
            B_dims = [[num_states, num_states, 5]]  # 5 actions: up, right, down, left, stay
            
            # Generate generative model with basic structure
            generative_model = self.define_generative_model(A_dims, B_dims)
            
            # Manually set A matrices to ensure proper normalization
            # A[0]: Position observation (identity matrix)
            A1 = np.eye(num_states)
            # Ensure columns sum to 1 by normalizing
            A1 = self._normalize_A_matrix(A1)
            
            # A[1]: Reward observation (2 states: no reward, reward)
            A2 = np.zeros((2, num_states))
            # Set reward positions (state 1 = reward)
            for row, col in reward_positions:
                pos = row * width + col
                A2[1, pos] = 1.0  # Reward observation
            # Set non-reward positions (state 0 = no reward)
            for s in range(num_states):
                if A2[1, s] == 0:
                    A2[0, s] = 1.0
            # Ensure columns sum to 1
            A2 = self._normalize_A_matrix(A2)
            
            # Replace the generated A matrices with our normalized ones
            generative_model["A"] = [A1, A2]
            
            # Add reward preferences (C vector)
            C = [[0.0] * num_states, [0.0, 4.0]]  # Neutral location preference, strong reward preference
            for row, col in reward_positions:
                pos = row * width + col
                C[0][pos] = 1.0  # Set reward location to high preference
            generative_model["C"] = C
            
            # Add additional parameters
            generative_model["action_precision"] = action_precision
            generative_model["inference_horizon"] = inference_horizon
            
            # Create agent
            agent_result = self.create_agent(name, generative_model)
            
            # Store additional metadata
            agent_result["grid_size"] = grid_size
            agent_result["reward_positions"] = reward_positions
            
            return agent_result
        except Exception as e:
            print(f"Error creating gridworld agent: {str(e)}")
            traceback.print_exc()
            return {"error": f"Error creating gridworld agent: {str(e)}"}
    
    def _normalize_A_matrix(self, A_matrix: np.ndarray) -> np.ndarray:
        """
        Normalize an A matrix so that columns sum to 1.
        
        Args:
            A_matrix: The A matrix to normalize [obs_dim, state_dim]
            
        Returns:
            Normalized A matrix
        """
        # For A matrices, we need to normalize columns (probabilities over observations given a state)
        # Each column in A corresponds to a specific state and sums to 1 over observations
        
        # Check if matrix is already normalized
        if np.allclose(np.sum(A_matrix, axis=0), 1.0):
            return A_matrix
            
        # Get column sums
        col_sums = np.sum(A_matrix, axis=0, keepdims=True)
        
        # Replace zeros with ones to avoid division by zero
        col_sums[col_sums == 0] = 1.0
        
        # Normalize columns
        A_normalized = A_matrix / col_sums
        
        return A_normalized
    
    def validate_generative_model(self, generative_model: Dict, check_normalization: bool = True) -> Dict:
        """
        Validate a generative model by checking its structure and normalization.
        
        Args:
            generative_model: The generative model to validate
            check_normalization: Whether to check if matrices are normalized
            
        Returns:
            Dict with validation results
        """
        try:
            import numpy as np
            result = {"valid": True, "issues": []}
            
            # Check required components
            if "A" not in generative_model:
                result["valid"] = False
                result["issues"].append("Missing A matrices (observation model)")
            
            if "B" not in generative_model:
                result["valid"] = False
                result["issues"].append("Missing B matrices (transition model)")
            
            # Early return if missing components
            if not result["valid"]:
                return result
            
            # Extract components
            A = generative_model.get("A", [])
            B = generative_model.get("B", [])
            C = generative_model.get("C", None)
            D = generative_model.get("D", None)
            
            # Convert to numpy arrays for validation
            A_arrays = [np.array(a) for a in A]
            B_arrays = [np.array(b) for b in B]
            
            # Check A matrices
            for i, a in enumerate(A_arrays):
                # Check shape - should be 2D
                if len(a.shape) != 2:
                    result["valid"] = False
                    result["issues"].append(f"A[{i}] has invalid shape: expected 2D, got {len(a.shape)}D")
                
                # Check normalization - columns should sum to 1
                if check_normalization:
                    is_normalized = True
                    for col in range(a.shape[1]):
                        col_sum = np.sum(a[:, col])
                        if not np.isclose(col_sum, 1.0, rtol=1e-5):
                            is_normalized = False
                            result["issues"].append(f"A[{i}] column {col} sums to {col_sum}, not 1.0")
                    
                    if not is_normalized:
                        result["valid"] = False
            
            # Check B matrices
            for i, b in enumerate(B_arrays):
                # Check shape - should be 3D for transition model
                if len(b.shape) != 3:
                    result["valid"] = False
                    result["issues"].append(f"B[{i}] has invalid shape: expected 3D, got {len(b.shape)}D")
                
                # Check normalization - for each action, from_state columns should sum to 1
                if check_normalization and len(b.shape) == 3:
                    for a in range(b.shape[2]):  # For each action
                        for s in range(b.shape[1]):  # For each from_state
                            col_sum = np.sum(b[:, s, a])
                            if not np.isclose(col_sum, 1.0, rtol=1e-5):
                                result["valid"] = False
                                result["issues"].append(f"B[{i}][:, {s}, {a}] sums to {col_sum}, not 1.0")
            
            # Return validation results
            return result
        except Exception as e:
            import traceback
            logger.error(f"Error validating generative model: {str(e)}")
            logger.error(traceback.format_exc())
            return {"valid": False, "error": str(e)}
    
    def infer_states(self, agent_id: str, observation: List[int], method: str = "FPI") -> Dict:
        """
        Perform state inference for the agent.
        
        Args:
            agent_id: ID of the agent
            observation: List of discrete observations
            method: Inference method (FPI, BP, etc.)
            
        Returns:
            Dict with posterior state beliefs
        """
        try:
            if self.debug_mode:
                print(f"State inference for agent {agent_id} with observation {observation}")
                
            # Check if agent exists
            if agent_id not in self.agents:
                return {"error": f"Agent {agent_id} not found"}
            
            agent = self.agents[agent_id]
            
            # Convert observation to numpy array if it's a list
            if isinstance(observation, list):
                observation = np.array(observation)
            
            # Check if the agent has an infer_states method
            if hasattr(agent, 'infer_states'):
                # Call PyMDP agent's method
                posterior = agent.infer_states(observation)
                
                # Convert posterior to list format for JSON serialization
                posterior_list = []
                for q in posterior:
                    if hasattr(q, 'tolist'):
                        posterior_list.append(q.tolist())
                    else:
                        posterior_list.append(q)
                    
                return {"posterior_states": posterior_list}
            else:
                # Use custom method if available
                if callable(getattr(agent, "infer_states", None)):
                    posterior = agent["infer_states"](observation)
                else:
                    return {"error": f"Agent {agent_id} does not have state inference capability"}
            
            # Convert numpy arrays to lists if needed
            if hasattr(posterior, 'tolist'):
                posterior = posterior.tolist()
            
            return {"posterior_states": posterior}
            
        except Exception as e:
            import traceback
            logger.error(f"Error in state inference: {str(e)}")
            logger.error(traceback.format_exc())
            return {"error": f"Error in state inference: {str(e)}"}
    
    def infer_policies(self, agent_id: str, save_computation: bool = True, planning_horizon: int = None) -> Dict:
        """
        Perform policy inference for the agent.
        
        Args:
            agent_id: ID of the agent
            save_computation: Whether to save computation details
            planning_horizon: Inference planning horizon (None uses agent default)
            
        Returns:
            Dict with policy posterior and expected free energy
        """
        try:
            if self.debug_mode:
                print(f"Policy inference for agent {agent_id}")
                
            # Check if agent exists
            if agent_id not in self.agents:
                return {"error": f"Agent {agent_id} not found"}
            
            agent = self.agents[agent_id]
            
            # Check if the agent has an infer_policies method
            policy_posterior = None
            expected_free_energy = None
            
            if hasattr(agent, 'infer_policies'):
                # Call PyMDP agent's method
                horizon = planning_horizon if planning_horizon is not None else agent.inference_horizon if hasattr(agent, 'inference_horizon') else 5
                policy_posterior, expected_free_energy = agent.infer_policies()
            else:
                # Use custom method if available
                if callable(getattr(agent, "infer_policies", None)):
                    policy_posterior, expected_free_energy = agent["infer_policies"]()
                else:
                    return {"error": f"Agent {agent_id} does not have policy inference capability"}
            
            # Convert numpy arrays to lists
            if hasattr(policy_posterior, 'tolist'):
                policy_posterior = policy_posterior.tolist()
            if hasattr(expected_free_energy, 'tolist'):
                expected_free_energy = expected_free_energy.tolist()
            
            # Log computation details
            if save_computation and self.debug_mode:
                if agent_id not in self.computation_logs:
                    self.computation_logs[agent_id] = []
                    
                self.computation_logs[agent_id].append({
                    "operation": "policy_inference",
                    "timestamp": time.time(),
                    "planning_horizon": horizon if 'horizon' in locals() else None,
                    "policy_posterior": str(policy_posterior)[:100] + "..." if len(str(policy_posterior)) > 100 else str(policy_posterior),
                    "expected_free_energy": str(expected_free_energy)[:100] + "..." if len(str(expected_free_energy)) > 100 else str(expected_free_energy)
                })
            
            return {
                "policy_posterior": policy_posterior,
                "expected_free_energy": expected_free_energy
            }
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"error": f"Error in policy inference: {str(e)}"}
    
    def sample_action(self, agent_id: str, planning_horizon: int = None) -> Dict:
        """
        Sample an action from the agent's policy posterior.
        
        Args:
            agent_id: ID of the agent
            planning_horizon: Planning horizon for policy inference (None uses agent default)
            
        Returns:
            Dict with the sampled action
        """
        try:
            if self.debug_mode:
                print(f"Sampling action for agent {agent_id}")
                
            # Check if agent exists
            if agent_id not in self.agents:
                return {"error": f"Agent {agent_id} not found"}
            
            agent = self.agents[agent_id]
            
            # Run policy inference if needed
            if planning_horizon is not None:
                # Use custom planning horizon
                policy_result = self.infer_policies(agent_id, planning_horizon=planning_horizon)
                if "error" in policy_result:
                    return {"error": f"Error in policy inference: {policy_result['error']}"}
            
            # Sample action
            action = None
            if hasattr(agent, 'sample_action'):
                # Call PyMDP agent's method
                action = agent.sample_action()
            else:
                # Use custom method if available
                if callable(getattr(agent, "sample_action", None)):
                    action = agent["sample_action"]()
                else:
                    return {"error": f"Agent {agent_id} does not have action sampling capability"}
            
            # Convert numpy arrays to lists
            if hasattr(action, 'tolist'):
                action = action.tolist()
            elif isinstance(action, np.ndarray):
                action = action.tolist()
            
            return {"action": action}
        except Exception as e:
            import traceback
            logger.error(f"Error in action sampling: {str(e)}")
            logger.error(traceback.format_exc())
            return {"error": f"Error in action sampling: {str(e)}"}
    
    def create_grid_world_env(self, name: str, grid_size: List[int], reward_positions: List[List[int]]) -> Dict:
        """
        Create a grid world environment with the given parameters.
        
        Args:
            name: Name of the environment
            grid_size: Grid dimensions [height, width]
            reward_positions: List of reward positions [[row, col], ...]
            
        Returns:
            Dict with environment information
        """
        try:
            import numpy as np
            
            # Extract dimensions
            height, width = grid_size
            num_states = height * width
            
            # Create a basic environment with the given dimensions
            env = {
                "id": name,
                "name": name,
                "type": "gridworld",
                "grid_size": grid_size,
                "reward_positions": reward_positions,
                "current_state": 0,  # Start at top-left corner
                "num_states": num_states,
                "is_mock_env": True  # Flag to indicate this is not a PyMDP Env
            }
            
            # Define step and reset functions
            def step(action):
                """Take a step in the environment based on the action.
                
                Args:
                    action: Action index (0=up, 1=right, 2=down, 3=left, 4=stay)
                    
                Returns:
                    Tuple of (observation, reward)
                """
                # Get current position
                current_pos = env["current_state"]
                current_row = current_pos // width
                current_col = current_pos % width
                
                # Calculate new position based on action
                new_row, new_col = current_row, current_col
                
                if isinstance(action, list) and len(action) > 0:
                    action = action[0]  # Extract action from list if needed
                
                # Convert to int if needed
                if isinstance(action, float):
                    action = int(action)
                
                # Apply action
                if action == 0:  # Up
                    new_row = max(0, current_row - 1)
                elif action == 1:  # Right
                    new_col = min(width - 1, current_col + 1)
                elif action == 2:  # Down
                    new_row = min(height - 1, current_row + 1)
                elif action == 3:  # Left
                    new_col = max(0, current_col - 1)
                elif action == 4:  # Stay
                    pass  # No change
                
                # Calculate new state index
                new_pos = new_row * width + new_col
                env["current_state"] = new_pos
                
                # Check if reached reward
                reward = 0
                for row, col in reward_positions:
                    if new_row == row and new_col == col:
                        reward = 1
                        break
                
                # Return observation (position and reward)
                return [new_pos, reward], reward
            
            def reset():
                """Reset the environment to the initial state.
                
                Returns:
                    Initial observation
                """
                # Reset to top-left corner
                env["current_state"] = 0
                
                # Return initial observation (position and reward)
                return [0, 0]
            
            # Attach functions to the environment
            env["step"] = step
            env["reset"] = reset
            
            # Store the environment
            self.environments[name] = env
            
            return env
        except Exception as e:
            import traceback
            print(f"Error creating grid world environment: {str(e)}")
            traceback.print_exc()
            return {"error": f"Error creating grid world environment: {str(e)}"}
    
    def step_environment(self, env_id: str, action: Union[List[int], int]) -> Dict:
        """
        Take a step in the environment with the specified action.
        
        Args:
            env_id: ID of the environment
            action: Action to take (can be a single integer or list)
            
        Returns:
            Dict with observation, reward, done, and info
        """
        try:
            if self.debug_mode:
                print(f"Taking step in environment {env_id} with action {action}")
                
            # Check if environment exists
            if env_id not in self.environments:
                return {"error": f"Environment {env_id} not found"}
            
            env = self.environments[env_id]
            
            # Convert action format if needed
            if isinstance(action, list) and len(action) == 1:
                action = action[0]
            
            # Take step in environment
            if hasattr(env, 'step') and callable(env.step):
                observation, reward = env.step(action)
            else:
                # Use step method from the environment dictionary
                if callable(getattr(env, "step", None)):
                    observation, reward = env["step"](action)
                else:
                    return {"error": f"Environment {env_id} does not have step capability"}
            
            # Convert numpy arrays to lists
            if hasattr(observation, 'tolist'):
                observation = observation.tolist()
            if hasattr(reward, 'tolist'):
                reward = reward.tolist()
            
            # Log step details
            if self.debug_mode:
                if env_id not in self.computation_logs:
                    self.computation_logs[env_id] = []
                    
                self.computation_logs[env_id].append({
                    "operation": "environment_step",
                    "timestamp": time.time(),
                    "action": action,
                    "observation": observation,
                    "reward": reward
                })
            
            return {
                "observation": observation,
                "reward": reward
            }
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"error": f"Error taking environment step: {str(e)}"}
    
    def run_simulation(self, agent_id: str, env_id: str, num_steps: int = 10, 
                     save_history: bool = True, planning_horizon: int = None) -> Dict:
        """
        Run a simulation with an agent in an environment.
        
        Args:
            agent_id: ID of the agent
            env_id: ID of the environment
            num_steps: Number of steps to run
            save_history: Whether to save simulation history
            planning_horizon: Planning horizon for policy inference (None uses agent default)
            
        Returns:
            Dict with simulation results
        """
        try:
            if self.debug_mode:
                print(f"Running simulation with agent {agent_id} in environment {env_id} for {num_steps} steps")
                
            # Check if agent and environment exist
            if agent_id not in self.agents:
                return {"error": f"Agent {agent_id} not found"}
            
            if env_id not in self.environments:
                return {"error": f"Environment {env_id} not found"}
            
            agent = self.agents[agent_id]
            env = self.environments[env_id]
            
            # Reset environment
            reset_result = self.reset_environment(env_id)
            if "error" in reset_result:
                return {"error": f"Error resetting environment: {reset_result['error']}"}
            
            initial_observation = reset_result.get("observation", [0])
            
            # Initialize history
            history = {
                "agent_id": agent_id,
                "env_id": env_id,
                "start_time": time.time(),
                "observations": [initial_observation],
                "actions": [],
                "rewards": [0.0],
                "states": [],
                "policies": [],
                "free_energies": []
            }
            
            # Run simulation steps
            current_observation = initial_observation
            
            for step in range(num_steps):
                # Infer states
                state_result = self.infer_states(agent_id, current_observation)
                if "error" in state_result:
                    return {"error": f"Error in state inference at step {step}: {state_result['error']}"}
                
                posterior_states = state_result.get("posterior_states", [])
                
                # Infer policies
                policy_result = self.infer_policies(agent_id, planning_horizon=planning_horizon)
                if "error" in policy_result:
                    return {"error": f"Error in policy inference at step {step}: {policy_result['error']}"}
                
                policy_posterior = policy_result.get("policy_posterior", [])
                expected_free_energy = policy_result.get("expected_free_energy", [])
                
                # Sample action
                action_result = self.sample_action(agent_id)
                if "error" in action_result:
                    return {"error": f"Error in action sampling at step {step}: {action_result['error']}"}
                
                action = action_result.get("action", [0])
                
                # Take step in environment
                step_result = self.step_environment(env_id, action)
                if "error" in step_result:
                    return {"error": f"Error in environment step at step {step}: {step_result['error']}"}
                
                observation = step_result.get("observation", [0])
                reward = step_result.get("reward", 0.0)
                
                # Update current observation
                current_observation = observation
                
                # Save history
                if save_history:
                    history["observations"].append(observation)
                    history["actions"].append(action)
                    history["rewards"].append(reward)
                    history["states"].append(posterior_states)
                    history["policies"].append(policy_posterior)
                    history["free_energies"].append(expected_free_energy)
                
                # Check if done
                if reward > 0:
                    history["done_step"] = step
                    break
            
            # Add end time
            history["end_time"] = time.time()
            history["total_steps"] = len(history["actions"])
            history["total_reward"] = sum(history["rewards"])
            
            # Log computation details
            if self.debug_mode:
                if "simulations" not in self.computation_logs:
                    self.computation_logs["simulations"] = {}
                
                simulation_id = f"{agent_id}_{env_id}_{int(time.time())}"
                self.computation_logs["simulations"][simulation_id] = {
                    "agent_id": agent_id,
                    "env_id": env_id,
                    "num_steps": num_steps,
                    "planning_horizon": planning_horizon,
                    "total_reward": history["total_reward"],
                    "timestamp": time.time()
                }
            
            return {
                "simulation_id": simulation_id if "simulation_id" in locals() else f"{agent_id}_{env_id}_{int(time.time())}",
                "agent_id": agent_id,
                "env_id": env_id,
                "total_steps": history["total_steps"],
                "total_reward": history["total_reward"],
                "history": history if save_history else None
            }
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"error": f"Error running simulation: {str(e)}"}
    
    def reset_environment(self, env_id: str) -> Dict:
        """
        Reset an environment to its initial state.
        
        Args:
            env_id: ID of the environment
            
        Returns:
            Dict with initial observation and state
        """
        try:
            if env_id not in self.environments:
                return {"error": f"Environment {env_id} not found"}
                
            env = self.environments[env_id]
            
            # Check if environment has a reset method
            if hasattr(env, "reset") and callable(env.reset):
                # Real PyMDP environment
                obs = env.reset()
                state = env.state
                
                # Convert to appropriate format
                observation = [int(obs)] if isinstance(obs, (int, float)) else [int(o) for o in obs]
                agent_pos = state
                
                return {"observation": observation, "state": agent_pos}
            else:
                # Mock environment
                # For GridWorld, set agent to top-left
                if env.get("type") == "grid_world" or env.get("type") == "gridworld":
                    env["current_state"] = 0
                    return {"observation": [0], "state": [0, 0]}
                else:
                    return {"observation": [0], "state": [0]}
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"error": f"Error resetting environment: {str(e)}"}
    
    def visualize_belief_dynamics(self, session_id: str, output_file: str = None) -> Dict:
        """
        Visualize belief dynamics over a simulation.
        
        Args:
            session_id: ID of the simulation session
            output_file: Path to save the visualization
            
        Returns:
            Dict with visualization information
        """
        try:
            # Check if session exists
            if not hasattr(self, "sessions") or session_id not in self.sessions:
                return {"error": f"Session '{session_id}' not found"}
                
            session = self.sessions[session_id]
            
            # Check if session has history
            if "history" not in session:
                return {"error": f"Session '{session_id}' has no history"}
                
            history = session["history"]
            
            # Check if timesteps exist
            if "timesteps" not in history or not history["timesteps"]:
                return {"error": f"Session '{session_id}' has no timesteps"}
                
            # Extract beliefs from each timestep
            beliefs_over_time = []
            for step in history["timesteps"]:
                if "beliefs" in step:
                    beliefs_over_time.append(step["beliefs"])
            
            # If no beliefs were found, return error
            if not beliefs_over_time:
                return {"error": "No belief data found in simulation history"}
                
            # Get the number of state factors
            num_factors = len(beliefs_over_time[0])
            
            # Import visualization libraries
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec
            from matplotlib.colors import LinearSegmentedColormap
            
            # Create a figure with subplots for each state factor
            num_timesteps = len(beliefs_over_time)
            fig = plt.figure(figsize=(12, 4 * num_factors))
            gs = gridspec.GridSpec(num_factors, 1, height_ratios=[1] * num_factors)
            
            # Plot beliefs for each state factor
            for factor_idx in range(num_factors):
                ax = plt.subplot(gs[factor_idx])
                
                # Extract beliefs for this factor over time
                factor_beliefs = np.array([step[factor_idx] for step in beliefs_over_time])
                
                # Check if we have data for this factor
                if factor_beliefs.size == 0:
                    continue
                    
                # Number of states for this factor
                num_states = factor_beliefs.shape[1]
                
                # Create heatmap
                cmap = plt.cm.viridis
                im = ax.imshow(factor_beliefs.T, aspect='auto', cmap=cmap, vmin=0, vmax=1)
                
                # Add colorbar
                plt.colorbar(im, ax=ax, label='Belief Probability')
                
                # Add labels
                ax.set_title(f'Belief Dynamics for State Factor {factor_idx}')
                ax.set_xlabel('Time Step')
                ax.set_ylabel('State')
                
                # Set y-ticks to show state indices
                ax.set_yticks(range(num_states))
                ax.set_yticklabels([f'State {i}' for i in range(num_states)])
                
                # Set x-ticks to show timesteps
                ax.set_xticks(range(num_timesteps))
                ax.set_xticklabels(range(num_timesteps))
                
                # Add grid
                ax.grid(False)
            
            # Adjust layout to prevent overlap
            plt.tight_layout()
            
            # Save figure if output file provided
            if output_file:
                plt.savefig(output_file, dpi=100, bbox_inches='tight')
                plt.close()
                return {"file": output_file}
            else:
                # Return plot as base64 encoded image
                import io
                import base64
                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
                plt.close()
                buf.seek(0)
                img_str = base64.b64encode(buf.read()).decode('utf-8')
                return {"image": f"data:image/png;base64,{img_str}"}
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"error": f"Error visualizing belief dynamics: {str(e)}"}
    
    def get_agent(self, name: str) -> Dict:
        """
        Retrieve an agent by name.
        
        Args:
            name: Name of the agent
            
        Returns:
            Dict with agent parameters
        """
        if name not in self.agents:
            return {"error": f"Agent '{name}' not found"}
            
        agent = self.agents[name]
        
        try:
            # Convert matrices to serializable format
            A_list = self._convert_from_obj_array(agent.A)
            B_list = self._convert_from_obj_array(agent.B)
            
            # C is optional
            C_list = None
            if hasattr(agent, 'C'):
                C_list = self._convert_from_obj_array(agent.C)
                
            result = {
                "name": name,
                "A": A_list,
                "B": B_list,
                "num_observation_modalities": len(agent.A),
                "num_state_factors": len(agent.B),
                "num_controls": [b.shape[-1] for b in agent.B]
            }
            
            if C_list is not None:
                result["C"] = C_list
                
            return result
            
        except Exception as e:
            return {"error": f"Error retrieving agent: {str(e)}"}
    
    def get_environment(self, name: str) -> Dict:
        """
        Get information about a specific environment.
        
        Args:
            name: Name of the environment
            
        Returns:
            Dict with environment information
        """
        if name not in self.environments:
            return {"error": f"Environment '{name}' not found"}
            
        env = self.environments[name]
        
        # Handle different environment types
        if hasattr(env, "shape") and hasattr(env, "position"):
            # This is a GridWorldEnv instance
            response = {
                "name": name,
                "type": "grid_world",
                "grid_size": list(env.shape),
                "agent_pos": list(env.position) if hasattr(env, "position") else [0, 0],
                "reward_locations": [[row, col] for row, col in env.reward_coords] if hasattr(env, "reward_coords") else []
            }
            return response
        else:
            # For dictionary-based environments, just return the dict
            if isinstance(env, dict):
                return env
            else:
                # For any other type, return basic info
                return {
                    "name": name,
                    "type": "custom",
                    "custom_env": str(type(env))
                }
    
    def get_all_functions(self) -> Dict:
        """Get a list of all available functions in the PyMDP interface."""
        # Get all methods that don't start with _ (private methods)
        methods = [method for method in dir(self) 
                  if callable(getattr(self, method)) and not method.startswith('_')]
        
        return {"functions": methods}
    
    def _convert_to_obj_array(self, data_json: List) -> Any:
        """Convert JSON serializable list to PyMDP object array."""
        # For simple types, just return numpy array
        if not isinstance(data_json, list):
            return np.array(data_json)
        
        # For A, B matrices which are nested lists
        if all(isinstance(item, list) for item in data_json):
            # For multidimensional arrays
            num_items = len(data_json)
            arr = np.empty(num_items, dtype=object)
            
            for i in range(num_items):
                # Convert each element in the list
                arr[i] = self._convert_to_obj_array(data_json[i])
            
            return arr
        else:
            # Simple 1D array
            return np.array(data_json)
    
    def _convert_from_obj_array(self, obj_array: Any) -> List:
        """Convert PyMDP object array to JSON serializable list."""
        # If a scalar value, directly return
        if isinstance(obj_array, (int, float, bool)) or obj_array is None:
            return obj_array
            
        # Handle NumPy scalar types (int64, float64, etc.)
        if isinstance(obj_array, (np.integer, np.floating, np.bool_)):
            return obj_array.item()  # Convert to native Python type
            
        # If NumPy array, convert to list
        if isinstance(obj_array, np.ndarray):
            if obj_array.dtype == 'O':  # Object array
                return [self._convert_from_obj_array(obj) for obj in obj_array]
            else:  # Regular array
                # Make sure all numpy values are converted to native Python types
                result = obj_array.tolist()
                if isinstance(result, list):
                    return self._ensure_native_types(result)
                return result
                
        # If already a list, process recursively
        if isinstance(obj_array, list):
            return self._ensure_native_types(obj_array)
            
        # Default - attempt to convert
        try:
            # For any other type that might be JSON serializable
            return obj_array
        except (TypeError, ValueError):
            # If conversion fails, return string representation
            return str(obj_array)
            
    def _ensure_native_types(self, lst: List) -> List:
        """Convert all numpy types in a nested list to native Python types."""
        if not isinstance(lst, list) and not isinstance(lst, np.ndarray):
            # Handle scalar numpy types
            if isinstance(lst, (np.integer, np.floating, np.bool_)):
                return lst.item()  # Convert to native Python type
            return lst
            
        # If it's a numpy array, convert to list first
        if isinstance(lst, np.ndarray):
            lst = lst.tolist()
            
        result = []
        for item in lst:
            if isinstance(item, (np.integer, np.floating, np.bool_)):
                # Convert numpy scalar to native Python type
                result.append(item.item())
            elif isinstance(item, list) or isinstance(item, np.ndarray):
                # Recursively handle nested lists or arrays
                result.append(self._ensure_native_types(item))
            else:
                # Keep other types as is
                result.append(item)
        return result

    def calculate_free_energy(self, A_matrix: List, prior: List, observation: List, q_states: List) -> Dict:
        """
        Calculate variational free energy for given beliefs and observations.
        
        Args:
            A_matrix: Observation likelihood matrix (A)
            prior: Prior beliefs about states
            observation: Observed outcome
            q_states: Posterior beliefs about states
            
        Returns:
            Dict with calculated free energy and components
        """
        try:
            import numpy as np
            from pymdp.maths import get_joint_likelihood
            
            # Convert inputs to numpy arrays
            A = np.array(A_matrix)
            prior = np.array(prior)
            obs = np.array(observation)
            q = np.array(q_states)
            
            # Ensure vectors are properly shaped
            if len(prior.shape) == 1:
                prior = prior.reshape(-1, 1)
            if len(q.shape) == 1:
                q = q.reshape(-1, 1)
            
            # Get observation likelihood for the actual observation
            likelihood = get_joint_likelihood(A, obs)
            
            # Calculate the energy term: -log(likelihood * prior)
            if likelihood.shape == prior.shape:
                energy = -np.log(np.sum(likelihood * prior) + 1e-16)
            else:
                energy = -np.log(np.sum(likelihood @ prior) + 1e-16)
            
            # Calculate the entropy term: sum(q * log(q))
            entropy = np.sum(q * np.log(q + 1e-16))
            
            # Calculate the divergence term: sum(q * log(q / prior))
            div = np.sum(q * np.log((q + 1e-16) / (prior + 1e-16)))
            
            # Total free energy = energy + entropy
            free_energy = energy + entropy
            
            # Alternative calculation: divergence - accuracy
            accuracy = np.sum(q * np.log(likelihood + 1e-16))
            free_energy_alt = div - accuracy
            
            return {
                "free_energy": float(free_energy),
                "energy": float(energy),
                "entropy": float(entropy),
                "divergence": float(div),
                "accuracy": float(accuracy),
                "free_energy_alt": float(free_energy_alt)
            }
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"error": f"Error calculating free energy: {str(e)}"}

    def infer_states_from_observation(self, A_matrix: List, prior: List, observation: List) -> Dict:
        """
        Infer states from an observation using Bayes' rule.
        
        Args:
            A_matrix: Observation likelihood matrix (A)
            prior: Prior beliefs about states
            observation: Observed outcome
            
        Returns:
            Dict with posterior state beliefs
        """
        try:
            import numpy as np
            from pymdp.maths import get_joint_likelihood
            
            # Convert inputs to numpy arrays
            A = np.array(A_matrix)
            prior = np.array(prior)
            obs = np.array(observation)
            
            # Ensure prior is properly shaped
            if len(prior.shape) == 1:
                prior = prior.reshape(-1, 1)
            
            # Get observation likelihood for the actual observation
            likelihood = get_joint_likelihood(A, obs)
            
            # Calculate posterior using Bayes' rule: P(s|o) ∝ P(o|s) * P(s)
            if likelihood.shape == prior.shape:
                joint = likelihood * prior
            else:
                joint = likelihood @ prior
                
            # Normalize to get proper posterior
            posterior = joint / np.sum(joint)
            
            # Convert to list format
            posterior_list = posterior.flatten().tolist()
            
            return {
                "posterior": posterior_list,
                "likelihood": likelihood.tolist(),
                "prior": prior.flatten().tolist()
            }
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"error": f"Error inferring states from observation: {str(e)}"}

def get_pymdp_interface():
    """
    Helper function to create a new PyMDPInterface instance.
    Used by the MCP server to create a context.
    """
    return PyMDPInterface() 