"""
Additional visualization methods for PyMDPInterface class.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any

# Import PyMDPInterface to extend it
from src.mcp.utils import PyMDPInterface

# Add visualization methods to PyMDPInterface
def _add_visualization_methods():
    """Add visualization methods to PyMDPInterface class."""
    
    def plot_belief_dynamics(self, session_id: str, output_file: str = "belief_dynamics.png") -> Dict:
        """
        Generate a visualization of belief dynamics over time for a session.
        
        Args:
            session_id: The ID of the session to visualize
            output_file: Filename for the output visualization
            
        Returns:
            Dict with visualization info and file path
        """
        if not hasattr(self, "sessions") or session_id not in self.sessions:
            return {"error": f"Session '{session_id}' not found"}
            
        try:
            session = self.sessions[session_id]
            history = session.get("history", {})
            
            if not history or "timesteps" not in history or not history["timesteps"]:
                return {"error": "No simulation data available for visualization"}
            
            # Extract belief data from history
            timesteps = history["timesteps"]
            
            # Determine how many state factors we have
            first_step_with_beliefs = None
            for step in timesteps:
                if "beliefs" in step:
                    first_step_with_beliefs = step
                    break
            
            if first_step_with_beliefs is None:
                return {"error": "No belief data found in session history"}
            
            num_factors = len(first_step_with_beliefs["beliefs"])
            
            # Create directory for output if needed
            output_dir = os.path.dirname(output_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            
            # Create figure with subplots for each state factor
            fig, axes = plt.subplots(num_factors, 1, figsize=(10, 3 * num_factors))
            
            # Make axes accessible for single factor case
            if num_factors == 1:
                axes = [axes]
            
            # Plot belief dynamics for each factor
            for factor_idx in range(num_factors):
                ax = axes[factor_idx]
                
                # Collect beliefs for this factor across all timesteps
                factor_beliefs = []
                for step in timesteps:
                    if "beliefs" in step and len(step["beliefs"]) > factor_idx:
                        factor_beliefs.append(step["beliefs"][factor_idx])
                
                if not factor_beliefs:
                    ax.text(0.5, 0.5, "No belief data for this factor", 
                           ha='center', va='center', transform=ax.transAxes)
                    continue
                
                # Convert to numpy array for plotting
                # Each row is a timestep, each column is a state
                belief_array = np.array(factor_beliefs)
                
                # Plot heatmap of beliefs over time
                im = ax.imshow(belief_array.T, aspect='auto', cmap='viridis', 
                              interpolation='nearest', origin='lower')
                ax.set_title(f"Belief Dynamics - State Factor {factor_idx}")
                ax.set_xlabel("Timestep")
                ax.set_ylabel("State")
                ax.set_yticks(range(belief_array.shape[1]))
                plt.colorbar(im, ax=ax)
            
            # Add overall title
            plt.suptitle(f"Belief Dynamics for Session: {session_id}", fontsize=16)
            plt.tight_layout(rect=[0, 0.03, 1, 0.97])
            
            # Save the figure
            plt.savefig(output_file)
            plt.close(fig)
            
            return {
                "file_path": output_file,
                "session_id": session_id,
                "num_factors": num_factors,
                "num_timesteps": len(timesteps)
            }
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"error": f"Error visualizing belief dynamics: {str(e)}"}
    
    def analyze_free_energy(self, session_id: str, output_file: str = "free_energy.png") -> Dict:
        """
        Analyze and visualize free energy components from a session.
        
        Args:
            session_id: The ID of the session to analyze
            output_file: Filename for the output visualization
            
        Returns:
            Dict with analysis results and file path
        """
        if not hasattr(self, "sessions") or session_id not in self.sessions:
            return {"error": f"Session '{session_id}' not found"}
            
        try:
            session = self.sessions[session_id]
            history = session.get("history", {})
            
            if not history or "timesteps" not in history or not history["timesteps"]:
                return {"error": "No simulation data available for analysis"}
            
            # Create directory for output if needed
            output_dir = os.path.dirname(output_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            
            # Extract free energy data from history if available
            timesteps = history["timesteps"]
            
            # Check if we have free energy data
            free_energy_values = []
            for step in timesteps:
                if "free_energy" in step:
                    free_energy_values.append(step["free_energy"])
                elif "computation_details" in step and "final_free_energy" in step["computation_details"]:
                    free_energy_values.append(step["computation_details"]["final_free_energy"])
            
            if not free_energy_values:
                return {"error": "No free energy data found in session history"}
            
            # Create figure for free energy analysis
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot free energy over time
            ax.plot(range(len(free_energy_values)), free_energy_values, marker='o', linestyle='-')
            ax.set_title(f"Free Energy Analysis - Session {session_id}")
            ax.set_xlabel("Timestep")
            ax.set_ylabel("Variational Free Energy")
            ax.grid(True)
            
            # Add horizontal line at zero for reference
            ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
            
            # Save the figure
            plt.tight_layout()
            plt.savefig(output_file)
            plt.close(fig)
            
            # Calculate summary statistics
            mean_fe = np.mean(free_energy_values)
            min_fe = np.min(free_energy_values)
            max_fe = np.max(free_energy_values)
            
            return {
                "file_path": output_file,
                "session_id": session_id,
                "num_timesteps": len(timesteps),
                "free_energy_summary": {
                    "mean": float(mean_fe),
                    "min": float(min_fe),
                    "max": float(max_fe)
                },
                "free_energy_values": free_energy_values
            }
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"error": f"Error analyzing free energy: {str(e)}"}
    
    # Add methods to the class
    PyMDPInterface.plot_belief_dynamics = plot_belief_dynamics
    PyMDPInterface.analyze_free_energy = analyze_free_energy

# Execute the function to add methods
_add_visualization_methods() 