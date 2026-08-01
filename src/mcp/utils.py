import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pymdp.utils as utils
from pymdp.agent import Agent
from pymdp.envs import Env

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
        # Per-agent inference state for the sequential infer -> plan -> act flow
        self.priors = {}       # agent_id -> list of (1, num_states) prior arrays (D)
        self.last_qs = {}      # agent_id -> last posterior over hidden states
        self.last_q_pi = {}    # agent_id -> last policy posterior
        self.log_counter = 0   # monotonic counter for deterministic log keys

    @staticmethod
    def _to_batch(mat):
        """Add a leading (batch) dimension of 1 to a numpy matrix for the
        batched PyMDP inference convention."""
        import numpy as np
        return np.asarray(mat, dtype=float)[None]

    @staticmethod
    def _unwrap(value):
        """Strip leading size-1 batch/time axes from a PyMDP result and
        return a plain 1-D python list for JSON serialization."""
        import numpy as np
        arr = np.asarray(value)
        while arr.ndim > 1 and arr.shape[0] == 1:
            arr = arr[0]
        return arr.tolist()

    def _onehot_observations(self, agent, observation):
        """Convert integer observation indices (one per modality) into the
        batched one-hot format the current PyMDP expects."""
        import numpy as np
        return [np.eye(agent.num_obs[m])[int(observation[m])][None]
                for m in range(agent.num_modalities)]

    @property
    def _next_log_key(self):
        self.log_counter += 1
        return f"log_{int(time.time()*1000)}_{self.log_counter}"

    @staticmethod
    def _step_gridworld(env: Dict, action) -> Tuple:
        """Inline grid-world transition for a dict environment.

        Returns the (observation, reward) pair, where observation is
        ``[new_state_index, reward]`` (position modality + reward modality).
        """
        height, width = env["grid_size"]
        cp = int(env["current_state"])
        row, col = divmod(cp, width)
        if isinstance(action, (list, tuple)) and len(action) > 0:
            action = action[0]
        action = int(action)
        if action == 0:      # up
            row = max(0, row - 1)
        elif action == 1:    # right
            col = min(width - 1, col + 1)
        elif action == 2:    # down
            row = min(height - 1, row + 1)
        elif action == 3:    # left
            col = max(0, col - 1)
        # action == 4 (stay) leaves row/col unchanged
        new_pos = row * width + col
        env["current_state"] = new_pos
        reward = 0
        for rp in env.get("reward_positions", []):
            if rp[0] == row and rp[1] == col:
                reward = 1
                break
        return [new_pos, reward], reward

    def create_agent(self, name: str, generative_model: Dict) -> Dict:
        """Create an agent from a generative model."""
        try:
            if self.debug_mode:
                print(f"Creating agent {name} with generative model")

            # Import required PyMDP modules
            import numpy as np
            from pymdp.agent import Agent

            # Extract generative model components
            A = generative_model.get("A", [])
            B = generative_model.get("B", [])
            C = generative_model.get("C", None)
            D = generative_model.get("D", None)

            # A and B observation/transition matrices are required by PyMDP.
            if not A or not B:
                return {"error": "Generative model must contain non-empty 'A' and 'B' matrices"}

            # Normalize observation model columns (probabilities over
            # observations given a state) and coerce to float arrays.
            for idx, a in enumerate(A):
                arr = np.asarray(a, dtype=float)
                if arr.ndim != 2:
                    return {"error": (
                        f"A[{idx}] is {arr.ndim}D; the interface supports per-factor 2D "
                        "observation models. Multi-factor coupled observation models need "
                        "explicit A_dependencies (Check modality dimension)."
                    )}
            A_arrays = [self._normalize_A_matrix(np.asarray(a, dtype=float)) for a in A]

            # Transition matrices: normalize each distribution over destination
            # states (axis 0) for every (source state, control) pair.
            B_arrays = []
            for b in B:
                bmat = np.asarray(b, dtype=float)
                for s in range(bmat.shape[-2]):
                    for c in range(bmat.shape[-1]):
                        colsum = bmat[:, s, c].sum()
                        if colsum > 0:
                            bmat[:, s, c] = bmat[:, s, c] / colsum
                B_arrays.append(bmat)

            # Current PyMDP uses a batched convention: each matrix carries a
            # leading batch dimension of 1 (batch_size=1).
            A_b = [self._to_batch(a) for a in A_arrays]
            B_b = [self._to_batch(b) for b in B_arrays]
            C_b = [self._to_batch(c) for c in C] if C is not None else None
            D_b = [self._to_batch(d) for d in D] if D is not None else None

            num_controls = [b.shape[-1] for b in B_arrays]

            # Default prior beliefs (D): uniform over each state factor.
            if D_b is None:
                D_b = [np.ones((1, b.shape[0])) / b.shape[0] for b in B_arrays]

            # Build construction kwargs that PyMDP accepts at __init__ time
            # (the equinox Agent is immutable after construction).
            agent_kwargs = dict(
                C=C_b,
                D=D_b,
                policy_len=int(generative_model.get("policy_len", 1)),
                num_controls=num_controls,
                control_fac_idx=list(range(len(B_arrays))),
            )
            if "inference_horizon" in generative_model:
                agent_kwargs["inference_horizon"] = int(generative_model["inference_horizon"])
            if "inference_algo" in generative_model:
                agent_kwargs["inference_algo"] = generative_model["inference_algo"]
            # Learning configuration is passed at construction (the equinox
            # Agent is immutable, so learning params cannot be assigned later).
            for lkey in ("learn_A", "learn_B", "learn_C", "learn_D", "learning_mode"):
                if lkey in generative_model:
                    agent_kwargs[lkey] = generative_model[lkey]
            for pkey in ("pA", "pB"):
                if generative_model.get(pkey) is not None:
                    agent_kwargs[pkey] = [np.asarray(p, dtype=float)[None] for p in generative_model[pkey]]
            # 'action_precision' is accepted for interface compatibility but has
            # no direct equivalent in the current PyMDP; it is intentionally ignored.

            # Create the PyMDP agent
            agent = Agent(A=A_b, B=B_b, **agent_kwargs)

            # Store the agent plus its prior and inference state
            self.agents[name] = agent
            self.priors[name] = D_b
            self.last_qs[name] = None
            self.last_q_pi[name] = None

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
                else:
                    logger.warning(f"Skipping A dimension entry of unexpected length: {dims} (expected 2)")

            # Create random B matrices (transition model)
            B = []
            for dims in B_dims:
                if len(dims) == 3:
                    # Basic transition model: [state_dim, state_dim, action_dim]
                    state_dim_to, state_dim_from, action_dim = dims
                    b = np.zeros((state_dim_to, state_dim_from, action_dim))
                    for a in range(action_dim):
                        b[:, :, a] = np.random.dirichlet(np.ones(state_dim_from), size=state_dim_to)
                    # Normalize each distribution over destination states (axis 0)
                    for s in range(state_dim_from):
                        for a in range(action_dim):
                            colsum = b[:, s, a].sum()
                            if colsum > 0:
                                b[:, s, a] = b[:, s, a] / colsum
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

            # Validate reward positions stay within the grid (negative indices
            # would otherwise silently wrap via numpy negative indexing).
            for row, col in reward_positions:
                if not (0 <= row < height and 0 <= col < width):
                    return {"error": f"Reward position ({row}, {col}) is out of bounds for a {height}x{width} grid"}

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

        # Normalize columns; non-positive-sum columns become uniform so that
        # a degenerate (all-zero or negative) state still yields a valid
        # probability distribution over observations instead of an invalid
        # all-zero (or negative) column.
        A_normalized = A_matrix / np.where(col_sums > 0, col_sums, 1.0)
        num_obs = A_normalized.shape[0]
        for col in range(A_normalized.shape[1]):
            if col_sums[0, col] <= 0:
                A_normalized[:, col] = 1.0 / num_obs

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

    def infer_states(self, agent_id: str, observation: List[int], method: str = "FPI",
                     save_intermediate_results: bool = False) -> Dict:
        """
        Perform state inference for the agent using the current PyMDP API.

        Args:
            agent_id: ID of the agent
            observation: List of discrete observation indices (one per modality)
            method: Inference method (FPI, BP, etc.). Retained for interface
                compatibility; the effective algorithm is set at agent construction.
            save_intermediate_results: Whether to attach computation details.

        Returns:
            Dict with posterior state beliefs, computation details and a log key
        """
        import numpy as np
        try:
            if self.debug_mode:
                print(f"State inference for agent {agent_id} with observation {observation}")

            if agent_id not in self.agents:
                return {"error": f"Agent {agent_id} not found"}

            agent = self.agents[agent_id]
            prior = self.priors.get(agent_id)

            # Current PyMDP expects one-hot observations (batched) and an
            # explicit empirical prior over hidden states.
            o_vec = self._onehot_observations(agent, observation)

            t0 = time.time()
            qs = agent.infer_states(o_vec, prior)
            computation_time = time.time() - t0

            self.last_qs[agent_id] = qs

            # One posterior vector per hidden-state factor.
            posterior_list = [self._unwrap(q) for q in qs]

            # Always attach computation details + a log key so callers (and
            # visualization) can trace a result back to its inputs.
            log_key = self._next_log_key
            comp_details = {
                "method": method,
                "observation": list(observation),
                "timestamp": time.time(),
                "computation_time": computation_time,
                "final_beliefs": posterior_list,
            }
            if agent_id not in self.computation_logs:
                self.computation_logs[agent_id] = []
            self.computation_logs[agent_id].append(comp_details)

            result = {
                "posterior_states": posterior_list,
                "method": method,
                "log_key": log_key,
                "computation_details": comp_details,
            }

            return result

        except Exception as e:
            import traceback
            logger.error(f"Error in state inference: {str(e)}")
            logger.error(traceback.format_exc())
            return {"error": f"Error in state inference: {str(e)}"}

    def infer_policies(self, agent_id: str, save_computation: bool = True, planning_horizon: int = None) -> Dict:
        """
        Perform policy inference for the agent using the current PyMDP API.

        The number of policies is fixed at agent construction (policy_len);
        `planning_horizon` is accepted for interface compatibility.

        Args:
            agent_id: ID of the agent
            save_computation: Whether to save computation details
            planning_horizon: Ignored (policies are fixed at construction).

        Returns:
            Dict with policy posterior and expected free energy
        """
        if self.debug_mode:
            print(f"Policy inference for agent {agent_id}")

        if agent_id not in self.agents:
            return {"error": f"Agent {agent_id} not found"}

        agent = self.agents[agent_id]

        # Policy inference consumes the posterior over states, so infer hidden
        # states from the prior when state inference was not run first.
        qs = self.last_qs.get(agent_id)
        if qs is None:
            prior = self.priors.get(agent_id)
            default_obs = self._onehot_observations(agent, [0] * agent.num_modalities)
            qs = agent.infer_states(default_obs, prior)
            self.last_qs[agent_id] = qs

        q_pi, expected_free_energy = agent.infer_policies(qs)
        self.last_q_pi[agent_id] = q_pi

        policy_posterior = self._unwrap(q_pi)
        efe_list = self._unwrap(expected_free_energy)

        # Log computation details
        if save_computation and self.debug_mode:
            if agent_id not in self.computation_logs:
                self.computation_logs[agent_id] = []
            self.computation_logs[agent_id].append({
                "operation": "policy_inference",
                "timestamp": time.time(),
                "planning_horizon": planning_horizon,
                "num_policies": len(policy_posterior),
                "policy_posterior": str(policy_posterior)[:100],
                "expected_free_energy": str(efe_list)[:100],
            })

        return {
            "policy_posterior": policy_posterior,
            "expected_free_energy": efe_list,
        }

    def sample_action(self, agent_id: str, planning_horizon: int = None) -> Dict:
        """
        Sample an action from the agent's policy posterior.

        Args:
            agent_id: ID of the agent
            planning_horizon: Ignored (policies are fixed at construction).

        Returns:
            Dict with the sampled action (an integer per control factor)
        """
        try:
            if self.debug_mode:
                print(f"Sampling action for agent {agent_id}")

            if agent_id not in self.agents:
                return {"error": f"Agent {agent_id} not found"}

            agent = self.agents[agent_id]

            # Sample from the policy posterior, inferring it first if needed.
            q_pi = self.last_q_pi.get(agent_id)
            if q_pi is None:
                qs = self.last_qs.get(agent_id)
                if qs is None:
                    prior = self.priors.get(agent_id)
                    default_obs = self._onehot_observations(agent, [0] * agent.num_modalities)
                    qs = agent.infer_states(default_obs, prior)
                    self.last_qs[agent_id] = qs
                q_pi, _ = agent.infer_policies(qs)
                self.last_q_pi[agent_id] = q_pi

            action = agent.sample_action(q_pi)
            # action shape is (batch, num_control_factors); strip the batch.
            act = self._unwrap(action)
            if isinstance(act, list):
                action_out = [int(a) for a in act]
            else:
                action_out = [int(act)]

            return {"action": action_out}
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
                "type": "grid_world",
                "grid_size": list(grid_size),
                "reward_positions": list(reward_positions),
                "reward_locations": [list(rp) for rp in reward_positions],
                "agent_pos": [0, 0],  # Start at top-left corner
                "current_state": 0,  # Start at top-left corner
                "num_states": num_states,
                "is_mock_env": True,  # Flag to indicate this is not a PyMDP Env
            }
            # NOTE: step/reset behaviour is implemented inline in
            # step_environment()/reset_environment(); the env dict deliberately
            # holds only JSON-serializable data (no attached functions).

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

            # Take a step.
            if isinstance(env, dict) and "grid_size" in env:
                observation, reward = self._step_gridworld(env, action)
            elif hasattr(env, 'step') and callable(env.step):
                # Real PyMDP-style environment object
                observation, reward = env.step(action)
            else:
                return {"error": f"Environment {env_id} does not have step capability"}

            # Convert numpy arrays to lists
            if hasattr(observation, 'tolist'):
                observation = observation.tolist()
            if hasattr(reward, 'tolist'):
                reward = reward.tolist()

            # Track the agent's row/column position for grid-style dict envs.
            position = None
            if isinstance(env, dict) and "grid_size" in env and "current_state" in env:
                height, width = env["grid_size"]
                cs = env["current_state"]
                if isinstance(cs, (int, float)) and width:
                    row, col = int(cs) // int(width), int(cs) % int(width)
                    position = [row, col]
                    env["agent_pos"] = position

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
                "reward": reward,
                "position": position
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

            current_observation = reset_result.get("observation", [0] * agent.num_modalities)

            timesteps = []
            free_energy_trace = []
            t_start = time.time()

            for step in range(num_steps):
                # Infer hidden states (also records computation details)
                state_result = self.infer_states(agent_id, current_observation, save_intermediate_results=True)
                if "error" in state_result:
                    return {"error": f"Error in state inference at step {step}: {state_result['error']}"}
                posterior_states = state_result.get("posterior_states", [])

                # Infer policies and expected free energy
                policy_result = self.infer_policies(agent_id, planning_horizon=planning_horizon)
                if "error" in policy_result:
                    return {"error": f"Error in policy inference at step {step}: {policy_result['error']}"}
                policy_posterior = policy_result.get("policy_posterior", [])
                efe = policy_result.get("expected_free_energy", [])

                # Sample an action
                action_result = self.sample_action(agent_id)
                if "error" in action_result:
                    return {"error": f"Error in action sampling at step {step}: {action_result['error']}"}
                action = action_result.get("action", 0)

                # Step the environment
                step_result = self.step_environment(env_id, action)
                if "error" in step_result:
                    return {"error": f"Error in environment step at step {step}: {step_result['error']}"}
                observation = step_result.get("observation", [0] * agent.num_modalities)
                reward = step_result.get("reward", 0.0)

                free_energy_trace.append(float(efe[0]) if efe else None)
                timesteps.append({
                    "observation": list(observation) if isinstance(observation, (list, tuple)) else observation,
                    "state": posterior_states,
                    "action": action,
                    "reward": reward,
                    "policy_posterior": policy_posterior,
                    "expected_free_energy": efe,
                })

                current_observation = observation

            simulation_id = f"{agent_id}_{env_id}_{int(time.time())}"
            history = {
                "agent_id": agent_id,
                "env_id": env_id,
                "num_steps": num_steps,
                "num_timesteps": len(timesteps),
                "timesteps": timesteps,
                "free_energy_trace": free_energy_trace,
                "start_time": t_start,
                "end_time": time.time(),
            }
            # Store history under both the timestamped id and the stable
            # "agent_env" key for downstream visualization lookups.
            self.simulation_history[simulation_id] = history
            self.simulation_history[f"{agent_id}_{env_id}"] = history

            # Log computation details
            if self.debug_mode:
                if "simulations" not in self.computation_logs:
                    self.computation_logs["simulations"] = {}
                self.computation_logs["simulations"][simulation_id] = {
                    "agent_id": agent_id,
                    "env_id": env_id,
                    "num_steps": num_steps,
                    "planning_horizon": planning_horizon,
                    "total_reward": float(sum(ts["reward"] for ts in timesteps)),
                    "timestamp": time.time(),
                }

            return {
                "simulation_id": simulation_id,
                "id": simulation_id,
                "agent_id": agent_id,
                "env_id": env_id,
                "timesteps": timesteps,
                "free_energy_trace": free_energy_trace,
                "total_steps": len(timesteps),
                "total_reward": float(sum(ts["reward"] for ts in timesteps)),
                "history": history if save_history else None,
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
                    env["agent_pos"] = [0, 0]
                    return {"observation": [0, 0], "state": [0, 0]}
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
            import matplotlib.gridspec as gridspec
            import matplotlib.pyplot as plt
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
                import base64
                import io
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

    def get_all_agents(self) -> Dict:
        """Return a summary of every stored agent."""
        agents = []
        for name in self.agents:
            summary = self.get_agent(name)
            if "error" in summary:
                summary = {"name": name, "error": summary["error"]}
            agents.append(summary)
        return {"agents": agents}

    def get_all_environments(self) -> Dict:
        """Return a summary of every stored environment."""
        envs = []
        for name in self.environments:
            info = self.get_environment(name)
            if "error" not in info:
                info = {"name": name, "type": info.get("type"), "grid_size": info.get("grid_size")}
            envs.append(info)
        return {"environments": envs}

    def get_all_simulations(self) -> Dict:
        """Return a summary of every stored simulation."""
        sims = []
        for sim_id, history in self.simulation_history.items():
            if isinstance(history, dict) and "timesteps" in history:
                sims.append({
                    "simulation_id": sim_id,
                    "num_timesteps": len(history.get("timesteps", [])),
                    "agent_id": history.get("agent_id"),
                    "env_id": history.get("env_id"),
                })
        return {"simulations": sims}

    def visualize_agent_model(self, agent_id: str, output_file: str = "agent_model.png") -> Dict:
        """Generate a visualization of an agent's generative model structure."""
        if agent_id not in self.agents:
            return {"error": f"Agent '{agent_id}' not found"}
        agent = self.agents[agent_id]
        try:
            os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            # A matrix heatmap
            A_mats = [np.asarray(a).sum(axis=0) if np.asarray(a).ndim > 2 else np.asarray(a) for a in agent.A]
            if A_mats:
                ax = axes[0]
                im = ax.imshow(A_mats[0], aspect="auto")
                ax.set_title("Observation model (A) — modality 0")
                fig.colorbar(im, ax=ax)
            B_mats = [np.asarray(b) for b in agent.B]
            if B_mats:
                ax = axes[1]
                b0 = B_mats[0]
                if b0.ndim > 2:
                    b0 = b0.reshape(b0.shape[0], -1)
                im = ax.imshow(b0, aspect="auto")
                ax.set_title("Transition model (B) — factor 0")
                fig.colorbar(im, ax=ax)
            fig.tight_layout()
            fig.savefig(output_file, dpi=120)
            plt.close(fig)
            return {"figure_path": output_file, "agent_id": agent_id}
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"error": f"Error visualizing agent model: {str(e)}"}

    def visualize_simulation(self, simulation_id: str, output_file: str = "simulation.png") -> Dict:
        """Render the main simulation figure plus companion visualization files.

        Uses the simulation history stored by ``run_simulation`` (looked up by
        the timestamped id or the stable ``agent_id_env_id`` key).
        """
        import matplotlib.gridspec as gridspec
        base, ext = os.path.splitext(output_file)
        try:
            # Locate the stored history (by timestamped id or stable agent_env key).
            history = self.simulation_history.get(simulation_id)
            if history is None:
                for key, h in self.simulation_history.items():
                    if (isinstance(h, dict) and h.get("agent_id")
                            and f"{h['agent_id']}_{h['env_id']}" == simulation_id):
                        history = h
                        break
            if history is None:
                return {"error": f"Simulation '{simulation_id}' not found"}

            timesteps = history.get("timesteps", [])
            os.makedirs(os.path.dirname(os.path.abspath(output_file)) or ".", exist_ok=True)

            additional = []
            # Belief heatmap companion
            beliefs = [ts.get("state", []) for ts in timesteps if ts.get("state")]
            if beliefs and beliefs[0]:
                n_factor = len(beliefs[0])
                fig, axes = plt.subplots(n_factor, 1, figsize=(10, 3 * n_factor), squeeze=False)
                for f in range(n_factor):
                    rows = []
                    for b in beliefs:
                        if f < len(b) and len(b[f]):
                            rows.append(b[f])
                    if rows:
                        arr = np.asarray(rows)
                        axes[f, 0].imshow(arr.T, aspect="auto", cmap="viridis")
                        axes[f, 0].set_title(f"Belief heatmap — factor {f}")
                fig.tight_layout()
                bh = base + "_belief_heatmap.png"
                fig.savefig(bh, dpi=110)
                plt.close(fig)
                additional.append(bh)

            # Policy posterior companion
            policies = [ts.get("policy_posterior", []) for ts in timesteps]
            if policies and any(p for p in policies):
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot([sum(0 for _ in p) for p in policies], alpha=0)
                for k in range(max((len(p) for p in policies if p), default=0)):
                    ax.plot([p[k] if k < len(p) else 0 for p in policies], label=f"policy {k}")
                ax.set_title("Policy posterior over time")
                ax.legend()
                fig.tight_layout()
                pp = base + "_policy_posterior.png"
                fig.savefig(pp, dpi=110)
                plt.close(fig)
                additional.append(pp)

            # Free-energy / EFE companions
            fe_trace = history.get("free_energy_trace") or [ts.get("expected_free_energy", []) for ts in timesteps]
            efe_trace = [ts.get("expected_free_energy", []) for ts in timesteps if ts.get("expected_free_energy")]
            if any(efe_trace):
                fig, ax = plt.subplots(figsize=(8, 4))
                for k in range(max(len(e) for e in efe_trace)):
                    ax.plot([e[k] if k < len(e) else 0 for e in efe_trace], label=f"efe {k}")
                ax.set_title("Expected free energy over time")
                ax.legend()
                fig.tight_layout()
                efe_png = base + "_expected_free_energy.png"
                fig.savefig(efe_png, dpi=110)
                plt.close(fig)
                additional.append(efe_png)
            if isinstance(fe_trace, list) and any(v is not None for v in fe_trace if isinstance(v, (int, float))):
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot([float(v) for v in fe_trace if isinstance(v, (int, float))])
                ax.set_title("Free energy trace")
                fig.tight_layout()
                fe_png = base + "_free_energy.png"
                fig.savefig(fe_png, dpi=110)
                plt.close(fig)
                additional.append(fe_png)

            # Main figure: free-energy trace
            fig, ax = plt.subplots(figsize=(10, 5))
            if isinstance(fe_trace, list) and any(isinstance(v, (int, float)) for v in fe_trace):
                ax.plot([float(v) for v in fe_trace if isinstance(v, (int, float))])
            ax.set_title(f"Simulation: {simulation_id}")
            ax.set_xlabel("Time step")
            ax.set_ylabel("Free energy")
            fig.tight_layout()
            fig.savefig(output_file, dpi=110)
            plt.close(fig)

            # Computation-log companion (JSON): top-level entries are dicts with
            # a timestamp / free-energy / iterations key (as expected by tests),
            # followed by the agent's recorded computation details.
            agent_id = history.get("agent_id")
            agent_logs = self.computation_logs.get(agent_id, []) if agent_id else []
            logs = {"generated_at": time.time(), "num_timesteps": len(timesteps)}
            for idx, entry in enumerate(agent_logs):
                logs[f"computation_log_{idx}"] = entry if isinstance(entry, dict) else {"detail": str(entry)}
            log_file = base + "_computation_logs.json"
            with open(log_file, "w") as f:
                json.dump(logs, f, indent=2)
            additional.append(log_file)

            return {"figure_path": output_file, "additional_visualizations": additional}
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"error": f"Error visualizing simulation: {str(e)}"}

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

        # Handle other array-like objects (e.g. JAX arrays) by coercing to
        # numpy and converting to native Python lists.
        try:
            arr = np.asarray(obj_array)
            if arr.dtype != object:
                result = arr.tolist()
                if isinstance(result, list):
                    return self._ensure_native_types(result)
                return result
        except Exception:
            pass

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
                # Coerce other array-likes (e.g. JAX arrays) to native types.
                try:
                    sub = np.asarray(item)
                    if sub.ndim == 0 and sub.dtype != object:
                        result.append(sub.item())
                    elif sub.dtype != object:
                        result.append(self._ensure_native_types(sub.tolist()))
                    else:
                        result.append(item)
                except Exception:
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
