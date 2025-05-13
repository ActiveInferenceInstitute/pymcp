"""
Simulation Visualization Module.

This module provides functions for visualizing Active Inference agent simulations.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.animation as animation

def visualize_simulation(simulation_history, output_file, include_beliefs=True, include_policies=True):
    """Create comprehensive visualization of a simulation
    
    Parameters
    ----------
    simulation_history : dict
        Simulation history dictionary with timesteps
    output_file : str
        Path to save the visualization
    include_beliefs : bool, optional
        Whether to include belief visualization, by default True
    include_policies : bool, optional
        Whether to include policy/action visualization, by default True
        
    Returns
    -------
    dict
        Dictionary containing paths to all generated visualizations
    """
    # Extract data from history
    timesteps = simulation_history.get('timesteps', [])
    agent_name = simulation_history.get('agent_name', 'Unknown Agent')
    env_name = simulation_history.get('env_name', 'Unknown Environment')
    
    if not timesteps:
        return {"error": "No timesteps found in simulation history"}
    
    # Create figure with flexible layout based on what's included
    if include_beliefs and include_policies:
        fig = plt.figure(figsize=(18, 12))
        gs = GridSpec(3, 3, figure=fig)
        axs = {
            'trajectory': fig.add_subplot(gs[0, :]),
            'beliefs': fig.add_subplot(gs[1, :2]),
            'policies': fig.add_subplot(gs[1, 2]),
            'reward': fig.add_subplot(gs[2, :])
        }
    elif include_beliefs:
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 2, figure=fig)
        axs = {
            'trajectory': fig.add_subplot(gs[0, :]),
            'beliefs': fig.add_subplot(gs[1, :]),
            'reward': fig.add_subplot(gs[2, :])
        }
    elif include_policies:
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 2, figure=fig)
        axs = {
            'trajectory': fig.add_subplot(gs[0, :]),
            'policies': fig.add_subplot(gs[1, 0]),
            'reward': fig.add_subplot(gs[2, :])
        }
    else:
        fig = plt.figure(figsize=(14, 8))
        gs = GridSpec(2, 1, figure=fig)
        axs = {
            'trajectory': fig.add_subplot(gs[0, :]),
            'reward': fig.add_subplot(gs[1, :])
        }
    
    # Set overall title
    fig.suptitle(f"Simulation: {agent_name} in {env_name}", fontsize=20)
    
    # Extract trajectory data
    trajectory_x = []
    trajectory_y = []
    
    # Extract reward data
    rewards = []
    cumulative_reward = 0
    cumulative_rewards = []
    
    # Extract belief and policy data if needed
    if include_beliefs:
        belief_data = []
    
    if include_policies:
        policy_data = []
        action_data = []
    
    # Process timesteps
    for i, ts in enumerate(timesteps):
        # Extract state if available (might be in different formats)
        state = ts.get('state', [])
        position = ts.get('position', [])
        
        # Try to extract position from state or position field
        if position and len(position) >= 2:
            trajectory_x.append(position[0])
            trajectory_y.append(position[1])
        elif state and len(state) >= 2:
            trajectory_x.append(state[0])
            trajectory_y.append(state[1])
        elif isinstance(state, list) and len(state) == 1 and isinstance(state[0], int):
            # Convert 1D state to 2D position (assuming grid environment)
            # This assumes a grid layout with row-major indexing
            grid_size = int(np.sqrt(len(timesteps[0].get('beliefs', [[]])[0]))) if timesteps[0].get('beliefs') else 0
            if grid_size > 0:
                row = state[0] // grid_size
                col = state[0] % grid_size
                trajectory_x.append(col)
                trajectory_y.append(row)
        
        # Extract reward
        reward = ts.get('reward', 0)
        rewards.append(reward)
        cumulative_reward += reward
        cumulative_rewards.append(cumulative_reward)
        
        # Extract beliefs if requested
        if include_beliefs and 'beliefs' in ts:
            belief_data.append(ts['beliefs'])
        
        # Extract policy and action data if requested
        if include_policies:
            if 'policy_posterior' in ts:
                policy_data.append(ts['policy_posterior'])
            if 'action' in ts:
                action_data.append(ts['action'])
    
    # Plot trajectory
    axs['trajectory'].plot(trajectory_x, trajectory_y, 'bo-', markersize=8, linewidth=2)
    axs['trajectory'].set_title('Agent Trajectory', fontsize=16)
    axs['trajectory'].set_xlabel('X Position', fontsize=12)
    axs['trajectory'].set_ylabel('Y Position', fontsize=12)
    axs['trajectory'].grid(True, alpha=0.3)
    
    # Add trajectory point labels
    for i, (x, y) in enumerate(zip(trajectory_x, trajectory_y)):
        axs['trajectory'].annotate(str(i), (x, y), xytext=(5, 5), 
                                  textcoords='offset points', fontsize=10)
                                  
    # Add start and end markers
    if trajectory_x and trajectory_y:
        axs['trajectory'].plot(trajectory_x[0], trajectory_y[0], 'go', markersize=12, label='Start')
        axs['trajectory'].plot(trajectory_x[-1], trajectory_y[-1], 'ro', markersize=12, label='End')
        axs['trajectory'].legend(fontsize=12)
    
    # Plot rewards
    time_points = list(range(len(rewards)))
    axs['reward'].bar(time_points, rewards, alpha=0.6, label='Reward')
    axs['reward'].plot(time_points, cumulative_rewards, 'r-', linewidth=2, label='Cumulative')
    axs['reward'].set_title('Rewards', fontsize=16)
    axs['reward'].set_xlabel('Timestep', fontsize=12)
    axs['reward'].set_ylabel('Reward Value', fontsize=12)
    axs['reward'].legend(fontsize=12)
    axs['reward'].grid(True, alpha=0.3)
    
    # Plot beliefs if requested and available
    if include_beliefs and belief_data:
        plot_beliefs(axs['beliefs'], belief_data)
    
    # Plot policies if requested and available
    if include_policies and (policy_data or action_data):
        if policy_data:
            plot_policies(axs['policies'], policy_data, action_data)
        elif action_data:
            plot_actions(axs['policies'], action_data)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save the main figure
    plt.savefig(output_file, dpi=150)
    plt.close(fig)
    
    # Generate additional visualizations
    additional_visualizations = []
    
    # Create belief dynamics visualization if beliefs are available
    if include_beliefs and belief_data and len(belief_data) > 0:
        belief_file = os.path.splitext(output_file)[0] + "_belief_heatmap.png"
        create_belief_heatmap(belief_data, belief_file)
        additional_visualizations.append(belief_file)
    
    # Create policy evolution visualization if policies are available
    if include_policies and policy_data and len(policy_data) > 0:
        policy_file = os.path.splitext(output_file)[0] + "_policy_posterior.png"
        visualize_policy_evolution(policy_data, policy_file)
        additional_visualizations.append(policy_file)
    
    # Create free energy visualization if available
    if 'free_energy_trace' in simulation_history and simulation_history['free_energy_trace']:
        fe_file = os.path.splitext(output_file)[0] + "_free_energy.png"
        visualize_free_energy(simulation_history['free_energy_trace'], fe_file)
        additional_visualizations.append(fe_file)
    
    # Return information about all visualizations
    return {
        "figure_path": output_file,
        "additional_visualizations": additional_visualizations,
        "files": [output_file] + additional_visualizations
    }

def plot_beliefs(ax, belief_data):
    """Plot belief evolution on the given axis
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Matplotlib axis to plot on
    belief_data : list
        List of belief arrays for each timestep
    """
    if not belief_data or not belief_data[0]:
        ax.text(0.5, 0.5, "No belief data available", 
                ha='center', va='center', fontsize=12)
        return
    
    # Determine the state factor to visualize (use the first one)
    factor_idx = 0
    
    # Extract beliefs for the chosen factor
    factor_beliefs = [ts[factor_idx] if len(ts) > factor_idx else [] for ts in belief_data]
    
    # Determine number of states in this factor
    if not factor_beliefs or not factor_beliefs[0]:
        ax.text(0.5, 0.5, "Empty belief data", 
                ha='center', va='center', fontsize=12)
        return
    
    num_states = len(factor_beliefs[0])
    timesteps = list(range(len(factor_beliefs)))
    
    # Create a beliefs over time plot
    for state_idx in range(num_states):
        state_beliefs = [beliefs[state_idx] if len(beliefs) > state_idx else 0 
                         for beliefs in factor_beliefs]
        ax.plot(timesteps, state_beliefs, '-o', 
                linewidth=2, label=f'State {state_idx}')
    
    ax.set_title(f'Belief Evolution (Factor {factor_idx})', fontsize=16)
    ax.set_xlabel('Timestep', fontsize=12)
    ax.set_ylabel('Belief Probability', fontsize=12)
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)
    
    # Only show legend if there are not too many states
    if num_states <= 10:
        ax.legend(fontsize=10, loc='upper right')

def plot_policies(ax, policy_data, action_data=None):
    """Plot policy posterior evolution on the given axis
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Matplotlib axis to plot on
    policy_data : list
        List of policy posterior arrays for each timestep
    action_data : list, optional
        List of action arrays for each timestep
    """
    if not policy_data:
        ax.text(0.5, 0.5, "No policy data available", 
                ha='center', va='center', fontsize=12)
        return
    
    # Determine the number of policies
    num_policies = len(policy_data[0]) if policy_data[0] else 0
    timesteps = list(range(len(policy_data)))
    
    # Create a policies over time plot
    for policy_idx in range(min(num_policies, 5)):  # Limit to 5 policies for clarity
        policy_probs = [policies[policy_idx] if len(policies) > policy_idx else 0 
                       for policies in policy_data]
        ax.plot(timesteps, policy_probs, '-o', 
                linewidth=2, label=f'Policy {policy_idx}')
    
    # Add selected actions if available
    if action_data:
        # Create a twin y-axis for actions
        ax2 = ax.twinx()
        
        # Extract the first control factor's actions
        actions = [a[0] if len(a) > 0 else None for a in action_data]
        action_timesteps = [t for t, a in enumerate(actions) if a is not None]
        action_values = [a for a in actions if a is not None]
        
        if action_values:
            ax2.step(action_timesteps, action_values, 'r-', linewidth=2, where='post')
            ax2.set_ylabel('Selected Action', fontsize=12, color='r')
            ax2.tick_params(axis='y', labelcolor='r')
    
    ax.set_title('Policy Posterior Evolution', fontsize=16)
    ax.set_xlabel('Timestep', fontsize=12)
    ax.set_ylabel('Policy Probability', fontsize=12)
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)
    
    # Only show legend if there are not too many policies
    if num_policies <= 5:
        ax.legend(fontsize=10, loc='upper right')

def plot_actions(ax, action_data):
    """Plot actions on the given axis when policy data is not available
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Matplotlib axis to plot on
    action_data : list
        List of action arrays for each timestep
    """
    if not action_data:
        ax.text(0.5, 0.5, "No action data available", 
                ha='center', va='center', fontsize=12)
        return
    
    # Extract the first control factor's actions
    actions = [a[0] if len(a) > 0 else None for a in action_data]
    valid_timesteps = [t for t, a in enumerate(actions) if a is not None]
    valid_actions = [a for a in actions if a is not None]
    
    if not valid_actions:
        ax.text(0.5, 0.5, "No valid actions found", 
                ha='center', va='center', fontsize=12)
        return
    
    ax.step(valid_timesteps, valid_actions, 'r-o', linewidth=2, where='post')
    ax.set_title('Selected Actions', fontsize=16)
    ax.set_xlabel('Timestep', fontsize=12)
    ax.set_ylabel('Action', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Set y-ticks to integers if actions are discrete
    if all(isinstance(a, int) for a in valid_actions):
        min_action = min(valid_actions)
        max_action = max(valid_actions)
        ax.set_yticks(list(range(min_action, max_action + 1)))

def create_belief_heatmap(belief_data, output_file):
    """Create a heatmap visualization of beliefs over time
    
    Parameters
    ----------
    belief_data : list
        List of belief arrays for each timestep
    output_file : str
        Path to save the visualization
        
    Returns
    -------
    str
        Path to the saved visualization file
    """
    if not belief_data or not belief_data[0]:
        return None
    
    # Determine the number of state factors
    num_factors = len(belief_data[0])
    num_timesteps = len(belief_data)
    
    # Create a figure with subplots for each factor
    fig_height = 4 * num_factors
    fig, axes = plt.subplots(num_factors, 1, figsize=(12, fig_height))
    
    # Convert to array of axes if there's only one factor
    if num_factors == 1:
        axes = [axes]
    
    # For each state factor
    for factor_idx in range(num_factors):
        ax = axes[factor_idx]
        
        # Extract beliefs for this factor
        factor_beliefs = [ts[factor_idx] if len(ts) > factor_idx else [] for ts in belief_data]
        
        # Check if we have valid data
        if not factor_beliefs or not factor_beliefs[0]:
            ax.text(0.5, 0.5, f"No belief data for factor {factor_idx}", 
                    ha='center', va='center', fontsize=12)
            continue
        
        # Create a 2D array for the heatmap
        num_states = len(factor_beliefs[0])
        belief_matrix = np.zeros((num_states, num_timesteps))
        
        # Fill the matrix
        for t, beliefs in enumerate(factor_beliefs):
            if len(beliefs) == num_states:
                belief_matrix[:, t] = beliefs
        
        # Plot the heatmap
        im = ax.imshow(belief_matrix, aspect='auto', cmap='viridis', 
                       origin='lower', vmin=0, vmax=1)
        
        # Add colorbar
        plt.colorbar(im, ax=ax)
        
        # Set labels
        ax.set_title(f'Belief Heatmap - Factor {factor_idx}', fontsize=16)
        ax.set_xlabel('Timestep', fontsize=12)
        ax.set_ylabel('State', fontsize=12)
        
        # Set x-ticks
        ax.set_xticks(list(range(0, num_timesteps, max(1, num_timesteps // 10))))
        
        # Set y-ticks
        ax.set_yticks(list(range(num_states)))
        ax.set_yticklabels([f'State {s}' for s in range(num_states)])
    
    plt.tight_layout()
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save the figure
    plt.savefig(output_file, dpi=150)
    plt.close(fig)
    
    return output_file

def visualize_policy_evolution(policy_data, output_file):
    """Create visualization of policy posterior evolution over time
    
    Parameters
    ----------
    policy_data : list
        List of policy posterior arrays for each timestep
    output_file : str
        Path to save the visualization
        
    Returns
    -------
    str
        Path to the saved visualization file
    """
    if not policy_data:
        return None
    
    # Determine the number of policies
    num_policies = len(policy_data[0]) if policy_data[0] else 0
    num_timesteps = len(policy_data)
    
    if num_policies == 0:
        return None
    
    # Create a 2D array for the heatmap
    policy_matrix = np.zeros((num_policies, num_timesteps))
    
    # Fill the matrix
    for t, policies in enumerate(policy_data):
        if len(policies) == num_policies:
            policy_matrix[:, t] = policies
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot the heatmap
    im = ax.imshow(policy_matrix, aspect='auto', cmap='viridis', 
                   origin='lower', vmin=0, vmax=1)
    
    # Add colorbar
    plt.colorbar(im, ax=ax)
    
    # Set labels
    ax.set_title('Policy Posterior Evolution', fontsize=18)
    ax.set_xlabel('Timestep', fontsize=14)
    ax.set_ylabel('Policy', fontsize=14)
    
    # Set x-ticks
    ax.set_xticks(list(range(0, num_timesteps, max(1, num_timesteps // 10))))
    
    # Set y-ticks
    ax.set_yticks(list(range(num_policies)))
    ax.set_yticklabels([f'Policy {p}' for p in range(num_policies)])
    
    plt.tight_layout()
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save the figure
    plt.savefig(output_file, dpi=150)
    plt.close(fig)
    
    return output_file

def visualize_free_energy(fe_trace, output_file):
    """Create visualization of free energy over time
    
    Parameters
    ----------
    fe_trace : list
        List of free energy values for each timestep
    output_file : str
        Path to save the visualization
        
    Returns
    -------
    str
        Path to the saved visualization file
    """
    if not fe_trace:
        return None
    
    # Extract timesteps and free energy values
    timesteps = [entry.get('timestep', i) for i, entry in enumerate(fe_trace)]
    fe_values = [entry.get('free_energy', 0) for entry in fe_trace]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot free energy over time
    ax.plot(timesteps, fe_values, 'b-o', linewidth=2)
    
    # Set labels
    ax.set_title('Free Energy over Time', fontsize=18)
    ax.set_xlabel('Timestep', fontsize=14)
    ax.set_ylabel('Free Energy', fontsize=14)
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
    
    return output_file

def create_simulation_animation(simulation_history, output_file, interval=200):
    """Create an animation of the agent's behavior in the environment
    
    Parameters
    ----------
    simulation_history : dict
        Simulation history dictionary with timesteps
    output_file : str
        Path to save the animation
    interval : int, optional
        Interval between frames in milliseconds, by default 200
        
    Returns
    -------
    str
        Path to the saved animation file
    """
    # Extract data from history
    timesteps = simulation_history.get('timesteps', [])
    agent_name = simulation_history.get('agent_name', 'Unknown Agent')
    env_name = simulation_history.get('env_name', 'Unknown Environment')
    
    if not timesteps:
        return None
    
    # Extract trajectory data
    trajectory_x = []
    trajectory_y = []
    
    # Process timesteps to extract positions
    for ts in timesteps:
        # Extract state if available (might be in different formats)
        state = ts.get('state', [])
        position = ts.get('position', [])
        
        # Try to extract position from state or position field
        if position and len(position) >= 2:
            trajectory_x.append(position[0])
            trajectory_y.append(position[1])
        elif state and len(state) >= 2:
            trajectory_x.append(state[0])
            trajectory_y.append(state[1])
        elif isinstance(state, list) and len(state) == 1 and isinstance(state[0], int):
            # Convert 1D state to 2D position (assuming grid environment)
            grid_size = int(np.sqrt(len(timesteps[0].get('beliefs', [[]])[0]))) if timesteps[0].get('beliefs') else 0
            if grid_size > 0:
                row = state[0] // grid_size
                col = state[0] % grid_size
                trajectory_x.append(col)
                trajectory_y.append(row)
    
    if not trajectory_x or not trajectory_y:
        return None
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Set title and labels
    ax.set_title(f"Agent Trajectory: {agent_name} in {env_name}", fontsize=16)
    ax.set_xlabel('X Position', fontsize=12)
    ax.set_ylabel('Y Position', fontsize=12)
    
    # Calculate bounds for the plot
    padding = 1
    min_x, max_x = min(trajectory_x) - padding, max(trajectory_x) + padding
    min_y, max_y = min(trajectory_y) - padding, max(trajectory_y) + padding
    
    # Set bounds
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    
    # Turn on grid
    ax.grid(True, alpha=0.3)
    
    # Initialize with empty plot
    line, = ax.plot([], [], 'bo-', markersize=8, linewidth=2)
    point, = ax.plot([], [], 'ro', markersize=12)
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12)
    
    # Initialization function for animation
    def init():
        line.set_data([], [])
        point.set_data([], [])
        time_text.set_text('')
        return line, point, time_text
    
    # Animation function
    def animate(i):
        line.set_data(trajectory_x[:i+1], trajectory_y[:i+1])
        point.set_data(trajectory_x[i], trajectory_y[i])
        time_text.set_text(f'Timestep: {i}')
        return line, point, time_text
    
    # Create animation
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=len(trajectory_x), interval=interval, 
                                   blit=True)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save the animation
    anim.save(output_file, writer='pillow', fps=int(1000/interval))
    plt.close(fig)
    
    return output_file 