"""
Custom Environment for PyMDP with MCP.

This module provides a custom environment implementation for Active Inference agents.
"""

import numpy as np
from collections import defaultdict

class CustomEnvironment:
    """Custom environment for Active Inference agents.
    
    This environment allows for defining arbitrary states, observations,
    transitions, and rewards for simulating complex environments.
    
    Parameters
    ----------
    name : str
        Name of the environment
    states : list
        List of state names
    observations : list
        List of observation names
    transitions : dict
        Dictionary mapping states to actions to next states
    rewards : dict
        Dictionary mapping states to reward values
    current_state : str, optional
        Initial state, by default None (use first state in list)
    
    Attributes
    ----------
    name : str
        Name of the environment
    states : list
        List of state names
    observations : list
        List of observation names
    transitions : dict
        Dictionary mapping states to actions to next states
    rewards : dict
        Dictionary mapping states to reward values
    current_state : str
        Current state of the environment
    """
    
    def __init__(self, name, states, observations, transitions, rewards, current_state=None):
        """Initialize the custom environment."""
        self.name = name
        self.states = states
        self.observations = observations
        self.transitions = transitions
        self.rewards = rewards
        
        # Set default initial state if not provided
        if current_state is None and len(states) > 0:
            current_state = states[0]
        
        self.current_state = current_state
        
        # Create mappings for indexing
        self.state_to_index = {state: i for i, state in enumerate(states)}
        self.index_to_state = {i: state for i, state in enumerate(states)}
        self.obs_to_index = {obs: i for i, obs in enumerate(observations)}
        self.index_to_obs = {i: obs for i, obs in enumerate(observations)}
        
        # Create action mapping for convenience
        self.actions = list(set(action for state_actions in transitions.values() 
                                for action in state_actions.keys()))
        self.action_to_index = {action: i for i, action in enumerate(self.actions)}
        self.index_to_action = {i: action for i, action in enumerate(self.actions)}
        
        # Create state-observation mapping
        self.state_observations = self._create_state_observation_mapping()
    
    def _create_state_observation_mapping(self):
        """Create mapping from states to observations.
        
        By default, we create a simple mapping where each state emits a specific observation
        and potentially a reward observation if the state has a reward.
        
        Returns
        -------
        dict
            Dictionary mapping states to list of observation indices
        """
        state_observations = {}
        
        for state in self.states:
            # Default observation is the one with the same name or index as the state
            default_obs = None
            
            # Try to find matching observation by name
            for obs in self.observations:
                if obs == state or obs.lower() == state.lower():
                    default_obs = obs
                    break
            
            # If no matching observation found, use state index
            if default_obs is None and len(self.observations) > self.state_to_index[state]:
                default_obs = self.observations[self.state_to_index[state]]
            
            # If still no observation found, use the first observation
            if default_obs is None and len(self.observations) > 0:
                default_obs = self.observations[0]
            
            # Create observation list
            if default_obs is not None:
                obs_indices = [self.obs_to_index[default_obs]]
                
                # Add reward observation if state has reward
                if state in self.rewards and self.rewards[state] > 0:
                    # Try to find "reward" observation
                    reward_obs = None
                    for obs in self.observations:
                        if obs.lower() == "reward":
                            reward_obs = obs
                            break
                    
                    if reward_obs is not None:
                        obs_indices.append(self.obs_to_index[reward_obs])
                
                state_observations[state] = obs_indices
            else:
                # No observations available
                state_observations[state] = []
        
        return state_observations
    
    def reset(self, state=None):
        """Reset the environment to initial state.
        
        Parameters
        ----------
        state : str, optional
            Initial state, by default None (use first state in list)
        
        Returns
        -------
        dict
            Observation after reset
        """
        if state is not None and state in self.states:
            self.current_state = state
        elif len(self.states) > 0:
            self.current_state = self.states[0]
        
        return self.get_observation()
    
    def step(self, action):
        """Take a step in the environment.
        
        Parameters
        ----------
        action : str, int, or list
            Action to take (can be action name, index, or list containing either)
        
        Returns
        -------
        dict
            Observation after taking the action
        """
        # Extract action if provided as a list
        if isinstance(action, list) and len(action) > 0:
            action = action[0]
        
        # Convert action index to name if needed
        if isinstance(action, int) and action in self.index_to_action:
            action = self.index_to_action[action]
        
        # Check if action is valid for current state
        if self.current_state in self.transitions and action in self.transitions[self.current_state]:
            # Update state
            self.current_state = self.transitions[self.current_state][action]
        
        # Return new observation
        return self.get_observation()
    
    def get_observation(self):
        """Get the current observation from the environment.
        
        Returns
        -------
        dict
            Observation dictionary
        """
        # Get state index
        state_idx = self.state_to_index[self.current_state]
        
        # Get observations for current state
        if self.current_state in self.state_observations:
            obs_indices = self.state_observations[self.current_state]
        else:
            obs_indices = []
        
        # Get reward
        reward_value = self.rewards.get(self.current_state, 0)
        
        return {
            "observation": obs_indices,
            "state": [state_idx],
            "current_state": self.current_state,
            "reward": reward_value
        }
    
    def get_valid_actions(self, state=None):
        """Get valid actions from a state.
        
        Parameters
        ----------
        state : str, optional
            State name, by default None (use current state)
        
        Returns
        -------
        list
            List of valid action names
        """
        if state is None:
            state = self.current_state
        
        if state in self.transitions:
            return list(self.transitions[state].keys())
        else:
            return []
    
    def get_valid_action_indices(self, state=None):
        """Get valid action indices from a state.
        
        Parameters
        ----------
        state : str, optional
            State name, by default None (use current state)
        
        Returns
        -------
        list
            List of valid action indices
        """
        valid_actions = self.get_valid_actions(state)
        return [self.action_to_index[action] for action in valid_actions 
                if action in self.action_to_index]
    
    def get_random_action(self, state=None):
        """Get a random valid action from a state.
        
        Parameters
        ----------
        state : str, optional
            State name, by default None (use current state)
        
        Returns
        -------
        str
            Random valid action name
        """
        valid_actions = self.get_valid_actions(state)
        if valid_actions:
            return np.random.choice(valid_actions)
        else:
            return None
    
    def has_reward(self, state=None):
        """Check if a state has a reward.
        
        Parameters
        ----------
        state : str, optional
            State name, by default None (use current state)
        
        Returns
        -------
        bool
            True if state has a reward, False otherwise
        """
        if state is None:
            state = self.current_state
        
        return state in self.rewards and self.rewards[state] > 0
    
    def is_terminal(self, state=None):
        """Check if a state is terminal.
        
        In this implementation, states with no outgoing transitions are terminal.
        
        Parameters
        ----------
        state : str, optional
            State name, by default None (use current state)
        
        Returns
        -------
        bool
            True if state is terminal, False otherwise
        """
        if state is None:
            state = self.current_state
        
        return (state not in self.transitions or 
                not self.transitions[state] or 
                self.has_reward(state))
    
    def render(self, mode='text'):
        """Render the environment.
        
        Parameters
        ----------
        mode : str, optional
            Rendering mode, by default 'text'
        
        Returns
        -------
        str
            Text representation of the environment
        """
        if mode != 'text':
            raise NotImplementedError(f"Rendering mode {mode} not implemented")
        
        result = f"Environment: {self.name}\n"
        result += f"Current state: {self.current_state}\n"
        
        # Show valid actions
        valid_actions = self.get_valid_actions()
        if valid_actions:
            result += f"Valid actions: {', '.join(valid_actions)}\n"
        else:
            result += "No valid actions (terminal state)\n"
        
        # Show reward if any
        if self.has_reward():
            result += f"Reward: {self.rewards[self.current_state]}\n"
        else:
            result += "No reward in current state\n"
        
        return result
    
    def to_dict(self):
        """Convert environment to dictionary representation.
        
        Returns
        -------
        dict
            Dictionary representation of the environment
        """
        return {
            'type': 'custom',
            'name': self.name,
            'states': self.states,
            'observations': self.observations,
            'transitions': self.transitions,
            'rewards': self.rewards,
            'current_state': self.current_state
        }

def create_custom_env(name, states, observations, transitions=None, rewards=None, current_state=None):
    """Create a custom environment.
    
    This is a convenience function that returns both the environment object
    and a dictionary representation for the MCP interface.
    
    Parameters
    ----------
    name : str
        Name of the environment
    states : list
        List of state names
    observations : list
        List of observation names
    transitions : dict, optional
        Dictionary mapping states to actions to next states, by default None
    rewards : dict, optional
        Dictionary mapping states to reward values, by default None
    current_state : str, optional
        Initial state, by default None
    
    Returns
    -------
    tuple
        (environment_dict, environment_object)
    """
    # Set default transitions if not provided (simple cycle)
    if transitions is None:
        transitions = {}
        for i, state in enumerate(states):
            next_state = states[(i + 1) % len(states)]
            prev_state = states[(i - 1) % len(states)]
            
            transitions[state] = {
                "forward": next_state,
                "backward": prev_state,
                "stay": state
            }
    
    # Set default rewards if not provided (reward in last state)
    if rewards is None:
        rewards = {state: 0.0 for state in states}
        if states:
            rewards[states[-1]] = 1.0
    
    # Create environment
    env = CustomEnvironment(name, states, observations, transitions, rewards, current_state)
    
    # Create dictionary representation
    env_dict = env.to_dict()
    
    return env_dict, env 