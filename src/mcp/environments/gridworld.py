"""
Grid World Environment for PyMDP with MCP.

This module provides a grid world environment implementation for Active Inference agents.
"""

import numpy as np

class GridWorldEnvironment:
    """Grid world environment for Active Inference agents.
    
    Parameters
    ----------
    grid_size : list
        Size of the grid [height, width]
    reward_locations : list
        List of reward locations [[row, col], ...]
    start_position : list, optional
        Starting position [row, col], by default [0, 0]
    
    Attributes
    ----------
    grid_size : list
        Size of the grid [height, width]
    reward_locations : list
        List of reward locations [[row, col], ...]
    agent_pos : list
        Current agent position [row, col]
    num_states : int
        Total number of states in the grid
    """
    
    def __init__(self, grid_size, reward_locations, start_position=None):
        """Initialize the grid world environment."""
        self.grid_size = grid_size
        self.reward_locations = reward_locations
        
        # Set default starting position to top-left
        if start_position is None:
            start_position = [0, 0]
        
        self.agent_pos = start_position.copy()
        self.num_states = grid_size[0] * grid_size[1]
        
        # Define actions: 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT
        self.actions = {
            0: [-1, 0],   # UP
            1: [0, 1],    # RIGHT
            2: [1, 0],    # DOWN
            3: [0, -1]    # LEFT
        }
    
    def reset(self, start_position=None):
        """Reset the environment to initial state.
        
        Parameters
        ----------
        start_position : list, optional
            Starting position [row, col], by default None (use current start position)
        
        Returns
        -------
        dict
            Observation after reset
        """
        if start_position is not None:
            self.agent_pos = start_position.copy()
        else:
            self.agent_pos = [0, 0]  # Default to top-left
        
        return self.get_observation()
    
    def step(self, action):
        """Take a step in the environment.
        
        Parameters
        ----------
        action : int or list
            Action to take (0=UP, 1=RIGHT, 2=DOWN, 3=LEFT)
        
        Returns
        -------
        dict
            Observation after taking the action
        """
        # Extract action index if provided as a list
        if isinstance(action, list) and len(action) > 0:
            action = action[0]
        
        # Check if action is valid
        if action not in self.actions:
            # Invalid action, no movement
            return self.get_observation()
        
        # Get direction vector for the action
        delta_row, delta_col = self.actions[action]
        
        # Calculate new position
        new_row = self.agent_pos[0] + delta_row
        new_col = self.agent_pos[1] + delta_col
        
        # Check if new position is valid
        if 0 <= new_row < self.grid_size[0] and 0 <= new_col < self.grid_size[1]:
            # Update position
            self.agent_pos = [new_row, new_col]
        
        # Return new observation
        return self.get_observation()
    
    def get_observation(self):
        """Get the current observation from the environment.
        
        Returns
        -------
        dict
            Observation dictionary
        """
        # Calculate state index
        state_idx = self.agent_pos[0] * self.grid_size[1] + self.agent_pos[1]
        
        # Calculate modality 1: position (one-hot encoded)
        position_obs = state_idx
        
        # Calculate modality 2: reward (binary)
        reward_obs = 0  # Default to no reward
        reward_value = 0  # Default reward value
        
        # Check if agent is at a reward location
        for reward_location in self.reward_locations:
            if self.agent_pos == reward_location:
                reward_obs = 1
                reward_value = 1
                break
        
        return {
            "observation": [position_obs, reward_obs],
            "state": [state_idx],
            "position": self.agent_pos.copy(),
            "reward": reward_value
        }
    
    def get_state_index(self, position=None):
        """Get the state index for a position.
        
        Parameters
        ----------
        position : list, optional
            Position [row, col], by default None (use current position)
        
        Returns
        -------
        int
            State index
        """
        if position is None:
            position = self.agent_pos
        
        return position[0] * self.grid_size[1] + position[1]
    
    def get_position_from_index(self, index):
        """Get the position from a state index.
        
        Parameters
        ----------
        index : int
            State index
        
        Returns
        -------
        list
            Position [row, col]
        """
        row = index // self.grid_size[1]
        col = index % self.grid_size[1]
        
        return [row, col]
    
    def get_valid_actions(self, position=None):
        """Get valid actions from a position.
        
        Parameters
        ----------
        position : list, optional
            Position [row, col], by default None (use current position)
        
        Returns
        -------
        list
            List of valid action indices
        """
        if position is None:
            position = self.agent_pos
        
        row, col = position
        valid_actions = []
        
        # Check each action
        if row > 0:
            valid_actions.append(0)  # UP
        if col < self.grid_size[1] - 1:
            valid_actions.append(1)  # RIGHT
        if row < self.grid_size[0] - 1:
            valid_actions.append(2)  # DOWN
        if col > 0:
            valid_actions.append(3)  # LEFT
        
        return valid_actions
    
    def get_random_action(self):
        """Get a random valid action.
        
        Returns
        -------
        int
            Random valid action index
        """
        valid_actions = self.get_valid_actions()
        return np.random.choice(valid_actions)
    
    def has_reached_reward(self):
        """Check if agent has reached a reward location.
        
        Returns
        -------
        bool
            True if agent is at a reward location, False otherwise
        """
        return self.agent_pos in self.reward_locations
    
    def is_terminal(self):
        """Check if current state is terminal.
        
        In this implementation, reaching a reward location is a terminal state.
        
        Returns
        -------
        bool
            True if current state is terminal, False otherwise
        """
        return self.has_reached_reward()
    
    def render(self, mode='ascii'):
        """Render the environment.
        
        Parameters
        ----------
        mode : str, optional
            Rendering mode, by default 'ascii'
        
        Returns
        -------
        str
            ASCII grid representation
        """
        if mode != 'ascii':
            raise NotImplementedError(f"Rendering mode {mode} not implemented")
        
        # Create empty grid
        grid = []
        for _ in range(self.grid_size[0]):
            row = ['.'] * self.grid_size[1]
            grid.append(row)
        
        # Add reward locations
        for reward_location in self.reward_locations:
            row, col = reward_location
            grid[row][col] = 'R'
        
        # Add agent
        row, col = self.agent_pos
        grid[row][col] = 'A'
        
        # Convert to string
        result = ''
        for row in grid:
            result += ' '.join(row) + '\n'
        
        return result
    
    def to_dict(self):
        """Convert environment to dictionary representation.
        
        Returns
        -------
        dict
            Dictionary representation of the environment
        """
        return {
            'type': 'grid_world',
            'grid_size': self.grid_size,
            'reward_locations': self.reward_locations,
            'agent_pos': self.agent_pos,
            'num_states': self.num_states,
            'current_state': self.agent_pos
        }

def create_grid_world_env(name, grid_size, reward_locations, start_position=None):
    """Create a grid world environment.
    
    This is a convenience function that returns both the environment object
    and a dictionary representation for the MCP interface.
    
    Parameters
    ----------
    name : str
        Name of the environment
    grid_size : list
        Size of the grid [height, width]
    reward_locations : list
        List of reward locations [[row, col], ...]
    start_position : list, optional
        Starting position [row, col], by default None
    
    Returns
    -------
    tuple
        (environment_dict, environment_object)
    """
    # Create environment
    env = GridWorldEnvironment(grid_size, reward_locations, start_position)
    
    # Create dictionary representation
    env_dict = env.to_dict()
    env_dict['name'] = name
    
    return env_dict, env 