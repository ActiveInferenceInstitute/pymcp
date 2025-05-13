"""
Belief Dynamics Module for MCP-PyMDP

This module implements functions for analyzing, tracking, and visualizing 
belief dynamics in active inference agents. The functions are used by 
the MCP server tools and can be used directly by client applications.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Any


def extract_belief_history(history: Dict) -> Tuple[List, int, List[int]]:
    """
    Extract belief history data from a simulation history.
    
    Parameters
    ----------
    history : Dict
        The simulation history from PyMDP
        
    Returns
    -------
    Tuple[List, int, List[int]]
        A tuple containing (beliefs, num_timesteps, num_states_per_factor)
    """
    # Extract data from history
    timesteps = history.get('timesteps', [])
    if not timesteps:
        return [], 0, []
    
    # Extract belief history
    beliefs = [ts.get('beliefs', []) for ts in timesteps]
    if not beliefs or not beliefs[0]:
        return [], 0, []
        
    num_factors = len(beliefs[0])
    num_timesteps = len(beliefs)
    
    # Determine number of states for each factor
    num_states_per_factor = []
    for f in range(num_factors):
        if len(beliefs[0]) > f:
            num_states_per_factor.append(len(beliefs[0][f]))
        else:
            num_states_per_factor.append(0)
    
    return beliefs, num_timesteps, num_states_per_factor


def create_belief_dynamics_visualization(
    history: Dict, 
    output_file: str,
    title: str = "Belief Dynamics Over Time",
    show_state_labels: bool = True,
    custom_state_labels: Optional[List[List[str]]] = None
) -> str:
    """
    Create a visualization of belief dynamics throughout a simulation
    
    Parameters
    ----------
    history : Dict
        The simulation history from PyMDP
    output_file : str
        The path to save the visualization
    title : str, optional
        The title for the visualization, by default "Belief Dynamics Over Time"
    show_state_labels : bool, optional
        Whether to show state labels on the y-axis, by default True
    custom_state_labels : Optional[List[List[str]]], optional
        Custom labels for states, organized by factor, by default None
        
    Returns
    -------
    str
        The path to the saved visualization, or None if creation failed
    """
    beliefs, num_timesteps, num_states_per_factor = extract_belief_history(history)
    
    if not beliefs or num_timesteps == 0:
        return None
        
    num_factors = len(beliefs[0])
    
    # Create a figure
    fig = plt.figure(figsize=(15, 5 * num_factors))
    fig.suptitle(title, fontsize=24)
    
    # Create subplots for each state factor
    for f in range(num_factors):
        ax = fig.add_subplot(num_factors, 1, f+1)
        
        # Extract belief arrays for this factor
        factor_beliefs = np.array([b[f] for b in beliefs if len(b) > f])
        
        # Only proceed if we have valid data
        if factor_beliefs.size == 0 or len(factor_beliefs.shape) < 2:
            continue
            
        num_states = factor_beliefs.shape[1]
        
        # Create heatmap of beliefs over time
        im = ax.imshow(factor_beliefs.T, aspect='auto', cmap='viridis', 
                      extent=[0, num_timesteps-1, -0.5, num_states-0.5])
        
        plt.colorbar(im, ax=ax, label="Belief Probability")
        
        # Add labels
        ax.set_title(f"Factor {f} Belief Dynamics", fontsize=18)
        ax.set_xlabel("Timestep", fontsize=14)
        ax.set_ylabel("State", fontsize=14)
        
        # Add state labels if not too many and enabled
        if show_state_labels and num_states <= 20:
            ax.set_yticks(range(num_states))
            
            # Use custom labels if provided, otherwise use default
            if custom_state_labels and f < len(custom_state_labels) and custom_state_labels[f]:
                labels = custom_state_labels[f][:num_states]  # Ensure we don't have more labels than states
                ax.set_yticklabels(labels)
            else:
                ax.set_yticklabels([f"State {s}" for s in range(num_states)])
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    
    # Save the figure
    plt.savefig(output_file, dpi=100)
    plt.close(fig)
    
    return output_file


def calculate_belief_statistics(beliefs: List) -> Dict:
    """
    Calculate statistics about belief dynamics
    
    Parameters
    ----------
    beliefs : List
        List of belief states across timesteps
        
    Returns
    -------
    Dict
        Dictionary containing statistics about belief dynamics
    """
    if not beliefs or not beliefs[0]:
        return {"error": "No belief data available"}
    
    num_factors = len(beliefs[0])
    num_timesteps = len(beliefs)
    
    stats = {
        "num_factors": num_factors,
        "num_timesteps": num_timesteps,
        "factors": []
    }
    
    for f in range(num_factors):
        # Extract belief arrays for this factor
        factor_beliefs = np.array([b[f] for b in beliefs if len(b) > f])
        
        # Skip if no valid data
        if factor_beliefs.size == 0 or len(factor_beliefs.shape) < 2:
            stats["factors"].append({"error": f"No valid data for factor {f}"})
            continue
            
        num_states = factor_beliefs.shape[1]
        
        # Calculate entropy over time
        entropy = -np.sum(factor_beliefs * np.log(factor_beliefs + 1e-10), axis=1)
        
        # Calculate state with maximum belief at each timestep
        max_belief_states = np.argmax(factor_beliefs, axis=1)
        
        # Calculate belief volatility (how often the most likely state changes)
        state_changes = np.sum(np.diff(max_belief_states) != 0)
        
        factor_stats = {
            "num_states": num_states,
            "mean_entropy": float(np.mean(entropy)),
            "max_entropy": float(np.max(entropy)),
            "min_entropy": float(np.min(entropy)),
            "entropy_over_time": entropy.tolist(),
            "max_belief_states": max_belief_states.tolist(),
            "state_changes": int(state_changes),
            "state_change_rate": float(state_changes) / (num_timesteps - 1) if num_timesteps > 1 else 0
        }
        
        stats["factors"].append(factor_stats)
    
    return stats 