"""
Test Fixtures for PyMDP with MCP.

This module provides common test fixtures for generative models and environments.
"""

import numpy as np

def create_simple_generative_model(num_obs=2, num_states=2, num_actions=2):
    """Create a simple generative model with specified dimensions.
    
    Parameters
    ----------
    num_obs : int, optional
        Number of observations, by default 2
    num_states : int, optional
        Number of states, by default 2
    num_actions : int, optional
        Number of actions, by default 2
    
    Returns
    -------
    dict
        Generative model with A, B, C, D matrices
    """
    # Create A matrix (observation model)
    A = np.zeros((num_obs, num_states))
    
    # Create identity-like mapping from states to observations 
    # (state i most likely generates observation i)
    for i in range(min(num_obs, num_states)):
        A[i, i] = 0.9
    
    # Fill in remaining probabilities to ensure proper normalization
    for col in range(num_states):
        col_sum = A[:, col].sum()
        if col_sum < 1.0:
            remaining = 1.0 - col_sum
            for row in range(num_obs):
                if A[row, col] == 0:  # Only add to cells that are still zero
                    A[row, col] = remaining / (num_obs - np.count_nonzero(A[:, col]))
    
    # Create B matrix (transition model)
    B = np.zeros((num_states, num_states, num_actions))
    
    # For each action, create a different transition dynamic
    for a in range(num_actions):
        # Create action-dependent transition matrix
        if a == 0:  # First action: tends to maintain state
            for s in range(num_states):
                B[s, s, a] = 0.9
                # Distribute remaining probability to other states
                for s_next in range(num_states):
                    if s_next != s:
                        B[s_next, s, a] = 0.1 / (num_states - 1) if num_states > 1 else 0
        else:  # Other actions: tend to transition to next state (cycling at the end)
            for s in range(num_states):
                next_s = (s + 1) % num_states
                B[next_s, s, a] = 0.8
                # Distribute remaining probability
                remaining = 0.2
                for s_next in range(num_states):
                    if s_next != next_s:
                        B[s_next, s, a] = remaining / (num_states - 1) if num_states > 1 else 0
    
    # Create C matrix (preferences over observations)
    C = np.zeros(num_obs)
    C[0] = 1.0  # Prefer first observation type
    
    # Create D matrix (prior over initial states)
    D = np.ones(num_states) / num_states  # Uniform prior
    
    # Convert to nested lists for JSON compatibility
    return {
        'A': [A.tolist()],
        'B': [B.tolist()],
        'C': [C.tolist()],
        'D': [D.tolist()]
    }

def create_multimodal_generative_model(num_modalities=2, state_dims=[2], control_dims=[2]):
    """Create a generative model with multiple observation modalities.
    
    Parameters
    ----------
    num_modalities : int, optional
        Number of observation modalities, by default 2
    state_dims : list, optional
        Dimensions of each state factor, by default [2]
    control_dims : list, optional
        Dimensions of each control factor, by default [2]
    
    Returns
    -------
    dict
        Generative model with A, B, C, D matrices
    """
    # Initialize lists for matrices
    A = []
    C = []
    
    # Create A and C matrices for each modality
    for m in range(num_modalities):
        # Determine number of observations for this modality (vary slightly between modalities)
        num_obs = state_dims[0] + (m % 2)  # Add 0 or 1 to make modalities slightly different
        
        # Create A matrix for this modality
        A_m = np.zeros((num_obs,) + tuple(state_dims))
        
        # Fill A matrix with reasonable values
        # For simplicity, we'll make observations roughly correspond to states
        for i in range(min(num_obs, state_dims[0])):
            # Create a slice to set a specific entry in the N-dimensional array
            idx = (i,) + (i,) + (0,) * (len(state_dims) - 1)
            A_m[idx] = 0.9
        
        # Normalize A matrix over observations
        # For each state combination
        for state_indices in np.ndindex(tuple(state_dims)):
            # Create slices to access this state combination for all observations
            obs_slice = (slice(None),) + state_indices
            
            # Get sum for this slice
            col_sum = A_m[obs_slice].sum()
            
            # Normalize if needed
            if col_sum < 1.0:
                remaining = 1.0 - col_sum
                non_zero_count = np.count_nonzero(A_m[obs_slice])
                zero_indices = np.where(A_m[obs_slice] == 0)[0]
                
                if len(zero_indices) > 0:
                    for zero_idx in zero_indices:
                        # Create a tuple with the observation index and state indices
                        full_idx = (zero_idx,) + state_indices
                        A_m[full_idx] = remaining / len(zero_indices)
        
        # Add to list of A matrices
        A.append(A_m.tolist())
        
        # Create C matrix for this modality (preferences)
        C_m = np.zeros(num_obs)
        C_m[0] = 1.0  # Prefer first observation
        C.append(C_m.tolist())
    
    # Create B matrix for each state factor
    B = []
    for i, state_dim in enumerate(state_dims):
        # Determine control dimension for this factor
        control_dim = control_dims[min(i, len(control_dims) - 1)]
        
        # Create B matrix
        B_i = np.zeros((state_dim, state_dim, control_dim))
        
        # Fill B matrix with reasonable transition dynamics
        for a in range(control_dim):
            if a == 0:  # First action: tends to maintain state
                for s in range(state_dim):
                    B_i[s, s, a] = 0.9
                    # Distribute remaining probability
                    for s_next in range(state_dim):
                        if s_next != s:
                            B_i[s_next, s, a] = 0.1 / (state_dim - 1) if state_dim > 1 else 0
            else:  # Other actions: tend to transition to next state
                for s in range(state_dim):
                    next_s = (s + 1) % state_dim
                    B_i[next_s, s, a] = 0.8
                    # Distribute remaining probability
                    for s_next in range(state_dim):
                        if s_next != next_s:
                            B_i[s_next, s, a] = 0.2 / (state_dim - 1) if state_dim > 1 else 0
        
        # Add to list of B matrices
        B.append(B_i.tolist())
    
    # Create D matrix for each state factor
    D = []
    for state_dim in state_dims:
        D_i = np.ones(state_dim) / state_dim  # Uniform prior
        D.append(D_i.tolist())
    
    return {'A': A, 'B': B, 'C': C, 'D': D}

def create_gridworld_generative_model(grid_size=[3, 3], reward_position=None):
    """Create a generative model for a grid world environment.
    
    Parameters
    ----------
    grid_size : list, optional
        Grid dimensions [height, width], by default [3, 3]
    reward_position : list, optional
        Position of reward [row, col], by default None (bottom-right)
    
    Returns
    -------
    dict
        Generative model with A, B, C, D matrices
    """
    # Set default reward position to bottom-right if not specified
    if reward_position is None:
        reward_position = [grid_size[0] - 1, grid_size[1] - 1]
    
    # Calculate total number of states
    num_states = grid_size[0] * grid_size[1]
    num_actions = 4  # Up, right, down, left
    
    # Calculate reward state index
    reward_state = reward_position[0] * grid_size[1] + reward_position[1]
    
    # Create A matrix for position observations (one-to-one mapping)
    A1 = np.eye(num_states)
    
    # Create A matrix for reward observations (binary: reward or no reward)
    A2 = np.zeros((2, num_states))
    A2[1, reward_state] = 1.0  # Reward observation at reward state
    A2[0, :] = 1.0  # No reward observation everywhere
    A2[0, reward_state] = 0.0  # Except at reward state
    
    # Create B matrix for transitions
    B = np.zeros((num_states, num_states, num_actions))
    
    # Fill in transitions for each position and action
    for pos in range(num_states):
        row, col = divmod(pos, grid_size[1])
        
        # UP action (0)
        next_row = max(0, row - 1)
        next_pos = next_row * grid_size[1] + col
        B[next_pos, pos, 0] = 1.0
        
        # RIGHT action (1)
        next_col = min(grid_size[1] - 1, col + 1)
        next_pos = row * grid_size[1] + next_col
        B[next_pos, pos, 1] = 1.0
        
        # DOWN action (2)
        next_row = min(grid_size[0] - 1, row + 1)
        next_pos = next_row * grid_size[1] + col
        B[next_pos, pos, 2] = 1.0
        
        # LEFT action (3)
        next_col = max(0, col - 1)
        next_pos = row * grid_size[1] + next_col
        B[next_pos, pos, 3] = 1.0
    
    # Create C matrix for preferences
    C1 = np.zeros(num_states)  # Neutral preference over positions
    C2 = np.array([0.0, 4.0])  # Strong preference for reward
    
    # Create D matrix for initial state (top-left)
    D = np.zeros(num_states)
    D[0] = 1.0
    
    # Return the generative model
    return {
        'A': [A1.tolist(), A2.tolist()],
        'B': [B.tolist()],
        'C': [C1.tolist(), C2.tolist()],
        'D': [D.tolist()]
    }

def create_grid_world_environment(grid_size=[3, 3], reward_positions=None):
    """Create a grid world environment configuration.
    
    Parameters
    ----------
    grid_size : list, optional
        Grid dimensions [height, width], by default [3, 3]
    reward_positions : list, optional
        List of reward positions [[row, col], ...], by default None
    
    Returns
    -------
    dict
        Grid world environment configuration
    """
    # Set default reward position to bottom-right if not specified
    if reward_positions is None:
        reward_positions = [[grid_size[0] - 1, grid_size[1] - 1]]
    
    # Create environment configuration
    return {
        'type': 'grid_world',
        'grid_size': grid_size,
        'reward_locations': reward_positions,
        'agent_pos': [0, 0],  # Start at top-left
        'current_state': [0, 0]
    }

def create_custom_environment(name="custom_env", states=None, observations=None):
    """Create a custom environment configuration.
    
    Parameters
    ----------
    name : str, optional
        Environment name, by default "custom_env"
    states : list, optional
        List of state names, by default None
    observations : list, optional
        List of observation names, by default None
    
    Returns
    -------
    dict
        Custom environment configuration
    """
    # Set default states and observations if not specified
    if states is None:
        states = ["state1", "state2", "state3"]
    
    if observations is None:
        observations = ["obs1", "obs2", "obs3", "reward", "no_reward"]
    
    # Create transitions (simple cycle by default)
    transitions = {}
    for i, state in enumerate(states):
        next_state = states[(i + 1) % len(states)]
        prev_state = states[(i - 1) % len(states)]
        
        transitions[state] = {
            "forward": next_state,
            "backward": prev_state,
            "stay": state
        }
    
    # Create rewards (reward in the last state)
    rewards = {state: 0.0 for state in states}
    rewards[states[-1]] = 1.0
    
    # Create environment configuration
    return {
        'type': 'custom',
        'name': name,
        'states': states,
        'observations': observations,
        'transitions': transitions,
        'rewards': rewards,
        'current_state': states[0]
    } 