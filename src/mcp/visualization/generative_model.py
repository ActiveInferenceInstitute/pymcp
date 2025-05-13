"""
Generative Model Visualization Module.

This module provides functions for visualizing Active Inference generative models (A, B, C, D matrices).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import os

def visualize_generative_model(agent_name, matrices, output_file):
    """Create comprehensive visualization of the generative model (A, B, C, D matrices)
    
    Parameters
    ----------
    agent_name : str
        Name of the agent whose generative model is being visualized
    matrices : dict
        Dictionary containing the A, B, C, D matrices
        Format: {'A': [...], 'B': [...], 'C': [...], 'D': [...]}
    output_file : str
        Path to save the visualization
        
    Returns
    -------
    str
        Path to the saved visualization file
    """
    A, B, C = matrices.get('A', []), matrices.get('B', []), matrices.get('C', [])
    D = matrices.get('D', [])
    
    # Convert lists to numpy arrays if needed
    A = [np.array(a) if not isinstance(a, np.ndarray) else a for a in A]
    B = [np.array(b) if not isinstance(b, np.ndarray) else b for b in B]
    C = [np.array(c) if not isinstance(c, np.ndarray) else c for c in C]
    if D:
        D = [np.array(d) if not isinstance(d, np.ndarray) else d for d in D]
    
    fig = plt.figure(figsize=(20, 16))
    
    # Create a 3x4 grid layout
    gs = gridspec.GridSpec(3, 4, figure=fig, height_ratios=[1, 1, 0.8])
    
    # Title
    fig.suptitle(f"Active Inference Generative Model: {agent_name}", fontsize=24)
    
    # Helper functions for matrix visualization
    def plot_heatmap(ax, matrix, title, xlabel=None, ylabel=None, cmap='viridis'):
        if matrix.ndim == 1:
            # For 1D arrays (like C and D), reshape to column vector for display
            matrix = matrix.reshape(-1, 1)
            im = ax.imshow(matrix, cmap=cmap, aspect='auto')
            ax.set_xticks([])
        else:
            im = ax.imshow(matrix, cmap=cmap, aspect='auto')
        
        # Add colorbar
        plt.colorbar(im, ax=ax)
        
        # Add title and labels
        ax.set_title(title, fontsize=16)
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=12)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=12)
        
        # Add values in cells if matrix is not too large
        if matrix.size <= 100:  # Adjust threshold as needed
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1] if matrix.ndim > 1 else 1):
                    if matrix.ndim > 1:
                        value = matrix[i, j]
                    else:
                        value = matrix[i]
                    ax.text(j, i, f"{value:.2f}", ha="center", va="center", 
                            color="white" if value > 0.5 else "black", fontsize=10)
                            
    # Create a custom diverging colormap for preferences and prior beliefs
    pref_cmap = LinearSegmentedColormap.from_list('preference', 
                                               ['blue', 'white', 'red'], 
                                               N=256)
    
    # 1. Visualize A matrices (Observation Model)
    ax_A_title = fig.add_subplot(gs[0, 0:2])
    ax_A_title.set_title("A Matrices (Observation Model)", fontsize=20)
    ax_A_title.axis('off')
    
    # Create subplots for each A matrix (up to 2 for simplicity)
    a_plots = min(len(A), 2)
    for i in range(a_plots):
        ax = fig.add_subplot(gs[0, i])
        A_i = A[i]
        if A_i.ndim == 2:
            plot_heatmap(ax, A_i, f"A[{i}]", 
                         ylabel="Observations", xlabel="Hidden States")
        else:
            # For higher-dimensional A matrices, we'll flatten all but the first dimension
            shape_str = ' × '.join([str(d) for d in A_i.shape])
            obs_dim = A_i.shape[0]
            state_dims = np.prod(A_i.shape[1:])
            A_i_flat = A_i.reshape(obs_dim, state_dims)
            plot_heatmap(ax, A_i_flat, f"A[{i}] ({shape_str})", 
                         ylabel="Observations", xlabel="Hidden States (flattened)")
    
    # 2. Visualize B matrices (Transition Model)
    ax_B_title = fig.add_subplot(gs[1, 0:2])
    ax_B_title.set_title("B Matrices (Transition Model)", fontsize=20)
    ax_B_title.axis('off')
    
    # Create subplots for each B matrix (up to 2 for simplicity)
    b_plots = min(len(B), 2)
    for i in range(b_plots):
        ax = fig.add_subplot(gs[1, i])
        B_i = B[i]
        
        # Show just the first control state for simplicity
        plot_heatmap(ax, B_i[:, :, 0], f"B[{i}], control=0", 
                     ylabel="Next State", xlabel="Current State")
    
    # 3. Visualize C matrices (Preferences)
    ax_C_title = fig.add_subplot(gs[2, 0:2])
    ax_C_title.set_title("C Vectors (Preferences)", fontsize=20)
    ax_C_title.axis('off')
    
    # Create subplots for each C vector (up to 2 for simplicity)
    c_plots = min(len(C), 2)
    for i in range(c_plots):
        ax = fig.add_subplot(gs[2, i])
        plot_heatmap(ax, C[i], f"C[{i}]", 
                     ylabel="Preferences", cmap=pref_cmap)
    
    # 4. Visualize D matrices (Prior Beliefs) if present
    if D:
        ax_D_title = fig.add_subplot(gs[2, 2:4])
        ax_D_title.set_title("D Vectors (Prior Beliefs)", fontsize=20)
        ax_D_title.axis('off')
        
        # Create subplots for each D vector (up to 2 for simplicity)
        d_plots = min(len(D), 2)
        for i in range(d_plots):
            ax = fig.add_subplot(gs[2, 2+i])
            plot_heatmap(ax, D[i], f"D[{i}]", 
                         ylabel="Prior Beliefs", cmap='Blues')
    
    # Add legend/information
    info_ax = fig.add_subplot(gs[0, 2:4])
    info_ax.axis('off')
    info_text = (
        "Active Inference Generative Model Components:\n\n"
        "A Matrices: Likelihood mapping (observation given state)\n"
        "B Matrices: Transition dynamics (next state given current state and action)\n"
        "C Vectors: Preference distribution over observations\n"
        "D Vectors: Prior beliefs over initial states\n\n"
        f"Model Dimensions:\n"
        f"- Observation Modalities: {len(A)}\n"
        f"- State Factors: {len(B)}\n"
        f"- Control States: {[b.shape[2] for b in B]}\n"
    )
    info_ax.text(0, 0.5, info_text, fontsize=14, va='center')
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save the figure
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle
    plt.savefig(output_file, dpi=150)
    plt.close(fig)
    
    return output_file 