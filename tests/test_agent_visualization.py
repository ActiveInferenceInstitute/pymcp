import unittest
import numpy as np
import sys
import os
import json
import matplotlib
import time
matplotlib.use('Agg')  # Use non-interactive Agg backend for testing

# Setup output directory
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
GEN_MODEL_DIR = os.path.join(OUTPUT_DIR, "generative_models")
FREE_ENERGY_DIR = os.path.join(OUTPUT_DIR, "free_energy")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(GEN_MODEL_DIR, exist_ok=True)
os.makedirs(FREE_ENERGY_DIR, exist_ok=True)

# Add the src directory directly to path
src_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Add the pymdp-clone directory to path to ensure we're using the real PyMDP library
pymdp_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'pymdp-clone')
if pymdp_dir not in sys.path:
    sys.path.insert(0, pymdp_dir)

# Import directly from the utils module
from mcp.utils import PyMDPInterface

# Import real PyMDP for comparison
import pymdp
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec

class TestAgentVisualization(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        self.pymdp_interface = PyMDPInterface()
        
        # Ensure output directory exists
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
    def tearDown(self):
        """Clean up after tests."""
        # We don't clean up the output directory here to preserve the visualizations
        pass
        
    def visualize_generative_model(self, agent_name, matrices, output_file):
        """Create comprehensive visualization of the generative model (A, B, C, D matrices)"""
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
        
        # Save the figure
        # Always save to generative_models directory
        base_name = os.path.basename(output_file)
        output_file = os.path.join(GEN_MODEL_DIR, base_name)
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle
        plt.savefig(output_file, dpi=150)
        plt.close(fig)
        
        return output_file
        
    def visualize_belief_dynamics(self, history, output_file):
        """Create a visualization of belief dynamics throughout a simulation"""
        # Ensure we're saving to belief_dynamics directory
        output_dir = os.path.join(os.path.dirname(OUTPUT_DIR), "output", "belief_dynamics")
        os.makedirs(output_dir, exist_ok=True)
        
        # Get just the filename if a path was provided
        filename = os.path.basename(output_file)
        # Create the new output path
        output_path = os.path.join(output_dir, filename)
        
        # Use the modular belief_dynamics implementation
        from mcp.belief_dynamics import create_belief_dynamics_visualization
        return create_belief_dynamics_visualization(history, output_path)
        
    def visualize_free_energy_components(self, history, output_file):
        """Create visualization of free energy components throughout simulation"""
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
        
        # Always save to free_energy directory
        output_name = os.path.basename(output_file)
        output_file = os.path.join(FREE_ENERGY_DIR, output_name)
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
        
    def test_simple_agent_visualization(self):
        """Test creating and visualizing a simple agent with 2D state space"""
        # Create a simple agent
        A = [
            [[0.9, 0.1], [0.1, 0.9]]  # Simple A matrix for one modality
        ]
        B = [
            [[[0.9, 0.1], [0.1, 0.9]], [[0.5, 0.5], [0.5, 0.5]]]  # Transitions for two actions
        ]
        C = [
            [1.0, 0.0]  # Preference for first observation type
        ]
        D = [
            [0.5, 0.5]  # Flat prior over states
        ]
        
        generative_model = {
            "A": A,
            "B": B,
            "C": C,
            "D": D
        }
        
        # Create agent
        result = self.pymdp_interface.create_agent("simple_agent", generative_model)
        
        # Visualize generative model even if agent creation fails
        output_file = os.path.join(GEN_MODEL_DIR, "simple_agent_generative_model.png")
        self.visualize_generative_model("simple_agent", generative_model, output_file)
        
        # Verify the visualization was created
        self.assertTrue(os.path.exists(output_file))
        self.assertGreater(os.path.getsize(output_file), 0)
        
    def test_multi_modality_agent_visualization(self):
        """Test creating and visualizing an agent with multiple observation modalities"""
        # Create a multi-modality agent (two observation modalities, one state factor)
        A1 = np.array([
            [0.9, 0.1],  # P(Obs1=0 | States)
            [0.1, 0.9]   # P(Obs1=1 | States)
        ])
        
        A2 = np.array([
            [0.8, 0.2],  # P(Obs2=0 | States)
            [0.2, 0.8]   # P(Obs2=1 | States)
        ])
        
        # B matrix - single state factor with transitions for 2 actions
        B = np.array([
            # Next state given current state and action
            [[0.9, 0.1], [0.1, 0.9]],  # Action 0
            [[0.5, 0.5], [0.5, 0.5]]   # Action 1
        ])
        B = np.transpose(B, (1, 2, 0))  # Reshape to PyMDP expected format
        
        # C matrices - preferences over observations
        C1 = np.array([0.8, 0.2])  # Preference for first modality
        C2 = np.array([0.3, 0.7])  # Preference for second modality
        
        # D matrix - prior over initial states
        D = np.array([0.7, 0.3])  # Prior for single state factor
        
        # Convert to lists for the interface
        A = [A1.tolist(), A2.tolist()]
        B = [B.tolist()]
        C = [C1.tolist(), C2.tolist()]
        D = [D.tolist()]
        
        generative_model = {
            "A": A,
            "B": B,
            "C": C,
            "D": D
        }
        
        # Create agent
        result = self.pymdp_interface.create_agent("multi_modality_agent", generative_model)
        
        # Visualize generative model even if agent creation fails
        output_file = os.path.join(GEN_MODEL_DIR, "multi_modality_agent_generative_model.png")
        self.visualize_generative_model("multi_modality_agent", generative_model, output_file)
        
        # Verify the visualization was created
        self.assertTrue(os.path.exists(output_file))
        self.assertGreater(os.path.getsize(output_file), 0)
        
    def test_grid_world_simulation_visualization(self):
        """Test running and visualizing a grid world simulation"""
        # Create a simpler grid world agent and environment (3x3 instead of 5x5)
        grid_size = [3, 3]
        num_states = grid_size[0] * grid_size[1]
        num_actions = 4  # Up, down, left, right
        
        # A matrix - observations conditioned on hidden states
        # One-to-one mapping between position and observation
        A1 = np.eye(num_states)
        # Second modality for rewards
        A2 = np.zeros((2, num_states))
        # Set reward location at bottom right
        reward_pos = num_states - 1  # Bottom right (2,2)
        A2[1, reward_pos] = 1.0  # Reward observation
        A2[0, :reward_pos] = 1.0  # No reward for other positions
        
        # B matrix - transitions conditioned on actions
        B = np.zeros((num_states, num_states, num_actions))
        
        # Fill in transitions for each action
        for pos in range(num_states):
            row, col = pos // grid_size[1], pos % grid_size[1]
            
            # For each position, define the next state for each action
            # UP (action 0)
            next_row = max(0, row - 1)
            next_pos = next_row * grid_size[1] + col
            B[next_pos, pos, 0] = 1.0
            
            # RIGHT (action 1)
            next_col = min(grid_size[1] - 1, col + 1)
            next_pos = row * grid_size[1] + next_col
            B[next_pos, pos, 1] = 1.0
            
            # DOWN (action 2)
            next_row = min(grid_size[0] - 1, row + 1)
            next_pos = next_row * grid_size[1] + col
            B[next_pos, pos, 2] = 1.0
            
            # LEFT (action 3)
            next_col = max(0, col - 1)
            next_pos = row * grid_size[1] + next_col
            B[next_pos, pos, 3] = 1.0
        
        # C matrix - preferences over observations (reward preference)
        C1 = np.zeros(num_states)  # Neutral preference for positions
        C2 = np.array([0.0, 4.0])  # Strong preference for reward
        
        # D matrix - start at top left
        D1 = np.zeros(num_states)
        D1[0] = 1.0  # Top-left position
        
        # Convert to lists for the interface
        A = [A1.tolist(), A2.tolist()]
        B = [B.tolist()]
        C = [C1.tolist(), C2.tolist()]
        D = [D1.tolist()]
        
        generative_model = {
            "A": A,
            "B": B,
            "C": C,
            "D": D
        }
        
        # Create agent and environment
        agent_result = self.pymdp_interface.create_agent("grid_agent", generative_model)
        reward_locations = [[2, 2]]  # Bottom-right in 3x3 grid
        env_result = self.pymdp_interface.create_grid_world_env("grid_env", grid_size, reward_locations)
        
        # First, visualize the generative model regardless of agent creation success
        gm_file = os.path.join(GEN_MODEL_DIR, "grid_agent_generative_model.png")
        self.visualize_generative_model("grid_agent", generative_model, gm_file)
        self.assertTrue(os.path.exists(gm_file))
        
        # If agent creation was successful, run a simulation
        if "error" not in agent_result and "error" not in env_result:
            # Run simulation for 20 steps (doubled from 10 for more detailed visualization)
            result = self.pymdp_interface.run_simulation("grid_agent", "grid_env", 20)
            
            # If simulation was successful, visualize everything
            if "error" not in result:
                # Belief dynamics visualization
                history_key = f"grid_agent_grid_env"
                if history_key in self.pymdp_interface.simulation_history:
                    history = self.pymdp_interface.simulation_history[history_key]
                    
                    # Belief dynamics - save to belief_dynamics folder
                    belief_output_dir = os.path.join(os.path.dirname(OUTPUT_DIR), "output", "belief_dynamics")
                    os.makedirs(belief_output_dir, exist_ok=True)
                    belief_file = os.path.join(belief_output_dir, "grid_agent_belief_dynamics.png")
                    self.visualize_belief_dynamics(history, belief_file)
                    
                    # Standard simulation visualization - save to visualization folder
                    vis_output_dir = os.path.join(os.path.dirname(OUTPUT_DIR), "output", "visualization")
                    os.makedirs(vis_output_dir, exist_ok=True)
                    sim_file = os.path.join(vis_output_dir, "grid_agent_simulation.png")
                    viz_result = self.pymdp_interface.visualize_simulation(history_key, sim_file)
                    
                    # Verify files were created
                    for file in [belief_file, sim_file]:
                        if os.path.exists(file):
                            self.assertGreater(os.path.getsize(file), 0)
                            print(f"Generated visualization: {os.path.basename(file)} in {os.path.dirname(file)}")
        
        # Check which files were saved
        expected_output_files = [
            (os.path.join(GEN_MODEL_DIR, "grid_agent_generative_model.png")),
            (os.path.join(os.path.dirname(OUTPUT_DIR), "output", "belief_dynamics", "grid_agent_belief_dynamics.png")),
            (os.path.join(os.path.dirname(OUTPUT_DIR), "output", "visualization", "grid_agent_simulation.png"))
        ]
        
        for filepath in expected_output_files:
            if os.path.exists(filepath):
                print(f"Created visualization: {os.path.basename(filepath)} in {os.path.dirname(filepath)}")
            else:
                print(f"Missing expected visualization: {filepath}")

    def test_free_energy_visualization(self):
        """Test visualizing free energy components from a simulation."""
        # Create a simple agent for testing EFE calculations
        A = [
            [[0.9, 0.1], [0.1, 0.9]]  # One observation modality
        ]
        B = [
            [[[0.9, 0.1], [0.1, 0.9]], [[0.5, 0.5], [0.5, 0.5]]]  # Transitions for control states
        ]
        C = [
            [1.0, 0.0]  # Preference for first observation
        ]
        D = [
            [0.5, 0.5]  # Uniform prior
        ]
        
        generative_model = {
            "A": A,
            "B": B,
            "C": C,
            "D": D
        }
        
        # Create agent
        result = self.pymdp_interface.create_agent("fe_agent", generative_model)
        
        # Create a simple simulation history with mock free energy components
        history = {
            "simulation_id": "fe_agent_simulation",
            "agent_name": "fe_agent",
            "env_name": "mock_env",
            "timesteps": []
        }
        
        # Add timesteps with mock free energy components
        for t in range(10):  # Doubled from 5 to 10 timesteps
            # Create mock expected free energy components
            efe_components = {
                "ambiguity": 0.2 - 0.03 * t,  # Decreasing ambiguity over time
                "risk": 0.5 - 0.1 * t,       # Decreasing risk over time
                "total": 0.7 - 0.13 * t      # Total EFE
            }
            
            timestep = {
                "timestep": t,
                "observation": [0],
                "state": [0],
                "action": [0],
                "beliefs": [[0.9, 0.1]],
                "expected_free_energy_components": efe_components,
                "expected_free_energy": [0.3, 0.4]  # Mock EFE values for two policies
            }
            
            history["timesteps"].append(timestep)
        
        # Add free energy trace
        history["free_energy_trace"] = [
            {"timestep": t, "free_energy": 1.0 - 0.15 * t} for t in range(10)  # Doubled from 5 to 10
        ]
        
        # Store the simulation history
        self.pymdp_interface.simulation_history["fe_agent_simulation"] = history
        
        # Generate free energy visualization
        output_file = os.path.join(FREE_ENERGY_DIR, "free_energy_test.png")
        self.visualize_free_energy_components(history, output_file)
        
        # Check that the file was created
        self.assertTrue(os.path.exists(output_file), f"Free energy visualization file {output_file} wasn't created")
        
        # Also create a standard simulation visualization which should include free energy
        fig_path = os.path.join(OUTPUT_DIR, "fe_simulation_viz.png")
        result = self.pymdp_interface.visualize_simulation("fe_agent_simulation", fig_path)
        
        # Verify additional files are created in free_energy directory
        fe_files = os.listdir(FREE_ENERGY_DIR)
        self.assertGreater(len(fe_files), 1, f"Expected more than just README in free_energy directory, got: {fe_files}")
        
        # Check if free energy JSON data file was created
        fe_data_file = os.path.splitext(output_file)[0] + "_data.json"
        self.assertTrue(os.path.exists(fe_data_file), f"Free energy data file {fe_data_file} wasn't created")

if __name__ == '__main__':
    unittest.main() 