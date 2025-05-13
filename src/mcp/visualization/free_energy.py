"""
Free Energy Visualization Module.

This module provides functions for visualizing the free energy components in Active Inference models.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt

def visualize_free_energy_components(history, output_file):
    """Create visualization of free energy components throughout simulation
    
    Parameters
    ----------
    history : dict
        Simulation history dictionary containing timesteps with free energy data
    output_file : str
        Path to save the visualization
        
    Returns
    -------
    str or None
        Path to the saved visualization file, or None if visualization failed
    """
    # Extract relevant data
    timesteps = history.get('timesteps', [])
    if not timesteps:
        return None
        
    # Check if we have expected free energy components
    has_efe = any('expected_free_energy_components' in ts for ts in timesteps)
    
    if not has_efe:
        return None
        
    # Extract components
    time_points = []
    ambiguity_terms = []
    risk_terms = []
    efe_values = []
    
    for t, ts in enumerate(timesteps):
        if 'expected_free_energy_components' in ts:
            components = ts['expected_free_energy_components']
            time_points.append(t)
            
            # Extract component values - adapt these based on your actual data structure
            if isinstance(components, dict):
                ambiguity_terms.append(components.get('ambiguity', 0))
                risk_terms.append(components.get('risk', 0))
                efe_values.append(components.get('total', 0))
            elif isinstance(components, list) and len(components) >= 3:
                ambiguity_terms.append(components[0])
                risk_terms.append(components[1])
                efe_values.append(components[2])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot components
    if ambiguity_terms:
        ax.plot(time_points, ambiguity_terms, 'b-', label='Ambiguity', linewidth=2)
    if risk_terms:
        ax.plot(time_points, risk_terms, 'r-', label='Risk', linewidth=2)
    if efe_values:
        ax.plot(time_points, efe_values, 'k--', label='Total EFE', linewidth=2)
        
    ax.set_title("Expected Free Energy Components", fontsize=20)
    ax.set_xlabel("Timestep", fontsize=16)
    ax.set_ylabel("Value", fontsize=16)
    ax.legend(fontsize=14)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save the figure
    plt.savefig(output_file, dpi=150)
    plt.close(fig)
    
    # Also save the raw data as JSON for analysis
    json_file = os.path.splitext(output_file)[0] + "_data.json"
    efe_data = {
        "timesteps": time_points,
        "ambiguity": ambiguity_terms,
        "risk": risk_terms,
        "total_efe": efe_values
    }
    with open(json_file, "w") as f:
        json.dump(efe_data, f, indent=2)
    
    return output_file

def visualize_variational_free_energy(history, output_file):
    """Create visualization of variational free energy over time
    
    Parameters
    ----------
    history : dict
        Simulation history dictionary containing free_energy_trace field
    output_file : str
        Path to save the visualization
        
    Returns
    -------
    str or None
        Path to the saved visualization file, or None if visualization failed
    """
    # Extract free energy trace from history
    fe_trace = history.get('free_energy_trace', [])
    
    if not fe_trace:
        return None
        
    # Extract timesteps and free energy values
    timesteps = [entry.get('timestep', i) for i, entry in enumerate(fe_trace)]
    fe_values = [entry.get('free_energy', 0) for entry in fe_trace]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot free energy over time
    ax.plot(timesteps, fe_values, 'b-o', linewidth=2)
    
    ax.set_title("Variational Free Energy over Time", fontsize=18)
    ax.set_xlabel("Timestep", fontsize=14)
    ax.set_ylabel("Free Energy", fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Add trend line
    if len(timesteps) > 1:
        z = np.polyfit(timesteps, fe_values, 1)
        p = np.poly1d(z)
        ax.plot(timesteps, p(timesteps), "r--", alpha=0.8, 
                label=f"Trend: {z[0]:.4f}x + {z[1]:.4f}")
        ax.legend(fontsize=12)
    
    plt.tight_layout()
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save the figure
    plt.savefig(output_file, dpi=150)
    plt.close(fig)
    
    # Also save the raw data as JSON for analysis
    json_file = os.path.splitext(output_file)[0] + "_data.json"
    fe_data = {
        "timesteps": timesteps,
        "free_energy": fe_values
    }
    with open(json_file, "w") as f:
        json.dump(fe_data, f, indent=2)
    
    return output_file 