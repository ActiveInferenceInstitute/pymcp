#!/usr/bin/env python3

import sys
import os
import numpy as np

# Add the pymdp-clone directory to the path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'pymdp-clone'))

from pymdp.agent import Agent
import pymdp.utils as utils

def test_agent_creation():
    """Test creating a PyMDP agent with proper object array handling."""
    print("Testing PyMDP agent creation...")
    
    # Create A matrices (observation model)
    A_mat = np.array([[0.9, 0.1], [0.1, 0.9]])
    A = np.empty(1, dtype=object)
    A[0] = A_mat
    
    # Create B matrices (transition model) with proper normalization
    B_mat = np.zeros((2, 2, 2))  # (to_state, from_state, control)
    
    # Fill B matrices with values
    B_mat[:, :, 0] = np.array([[0.9, 0.1], [0.1, 0.9]])  # Control 0
    B_mat[:, :, 1] = np.array([[0.5, 0.5], [0.5, 0.5]])  # Control 1
    
    # Normalize B matrices along the "to" dimension
    for c in range(B_mat.shape[2]):  # For each control state
        for s in range(B_mat.shape[1]):  # For each state being transitioned from
            # Normalize over states to transition to (axis 0)
            if np.sum(B_mat[:, s, c]) > 0:
                B_mat[:, s, c] = B_mat[:, s, c] / np.sum(B_mat[:, s, c])
    
    B = np.empty(1, dtype=object)
    B[0] = B_mat
    
    # Create C matrices (prior preferences)
    C = np.empty(1, dtype=object)
    C[0] = np.array([1.0, 0.0])  # Prefer first observation
    
    # Create the agent
    agent = Agent(A=A, B=B, C=C)
    print("Agent created successfully!")
    
    # Test state inference
    observation = [0]  # Observe state 0
    qs = agent.infer_states(observation)
    print(f"Posterior states: {[q.tolist() for q in qs]}")
    
    # Test policy inference
    q_pi, G = agent.infer_policies()
    print(f"Policy posterior: {q_pi.tolist()}")
    print(f"Expected free energy: {G.tolist()}")
    
    # Test action selection
    action = agent.sample_action()
    print(f"Sampled action: {action}")
    
    return agent

if __name__ == "__main__":
    agent = test_agent_creation() 