"""
Test module for belief dynamics functionality.
"""

import os
import sys
import unittest
import numpy as np

# Add the src directory to the path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import matplotlib
# Use non-interactive backend
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mcp.belief_dynamics import (
    extract_belief_history,
    create_belief_dynamics_visualization,
    calculate_belief_statistics
)

class TestBeliefDynamics(unittest.TestCase):
    """Test class for belief dynamics module."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a sample history with beliefs
        self.sample_history = {
            'timesteps': [
                {'beliefs': [[0.8, 0.1, 0.1], [0.7, 0.3]]},
                {'beliefs': [[0.6, 0.3, 0.1], [0.5, 0.5]]},
                {'beliefs': [[0.4, 0.5, 0.1], [0.3, 0.7]]},
                {'beliefs': [[0.2, 0.7, 0.1], [0.2, 0.8]]},
                {'beliefs': [[0.1, 0.8, 0.1], [0.1, 0.9]]}
            ]
        }
        
        # Create output directory
        self.output_dir = os.path.join('tests', 'output', 'belief_dynamics')
        os.makedirs(self.output_dir, exist_ok=True)
    
    def test_extract_belief_history(self):
        """Test extracting belief history from simulation data."""
        beliefs, num_timesteps, num_states = extract_belief_history(self.sample_history)
        
        # Verify results
        self.assertEqual(num_timesteps, 5)
        self.assertEqual(len(beliefs), 5)
        self.assertEqual(num_states, [3, 2])
        
        # Verify empty history handling
        empty_beliefs, empty_timesteps, empty_states = extract_belief_history({})
        self.assertEqual(empty_timesteps, 0)
        self.assertEqual(len(empty_beliefs), 0)
        
    def test_create_belief_dynamics_visualization(self):
        """Test creating belief dynamics visualization."""
        # Define the base output directory
        base_output_dir = os.path.join('tests', 'output')
        belief_output_dir = os.path.join(base_output_dir, 'belief_dynamics')
        os.makedirs(belief_output_dir, exist_ok=True)

        output_file = os.path.join(belief_output_dir, 'test_belief_vis.png')
        
        # Create visualization
        vis_path = create_belief_dynamics_visualization(
            self.sample_history, 
            output_file,
            title="Test Belief Dynamics"
        )
        
        # Verify file was created
        self.assertTrue(os.path.exists(output_file))
        self.assertEqual(vis_path, output_file)
        
        # Test with custom labels
        custom_labels = [
            ["State A", "State B", "State C"],
            ["Position 1", "Position 2"]
        ]
        
        custom_output = os.path.join(belief_output_dir, 'test_belief_custom_labels.png')
        custom_vis_path = create_belief_dynamics_visualization(
            self.sample_history,
            custom_output,
            custom_state_labels=custom_labels
        )
        
        # Verify custom labeled file was created
        self.assertTrue(os.path.exists(custom_output))
        
        # Test with empty history
        empty_path = create_belief_dynamics_visualization({}, os.path.join(belief_output_dir, 'empty.png'))
        self.assertIsNone(empty_path)
        
    def test_calculate_belief_statistics(self):
        """Test calculating statistics from belief data."""
        # Extract beliefs
        beliefs, _, _ = extract_belief_history(self.sample_history)
        
        # Calculate statistics
        stats = calculate_belief_statistics(beliefs)
        
        # Verify structure and content
        self.assertEqual(stats['num_factors'], 2)
        self.assertEqual(stats['num_timesteps'], 5)
        self.assertEqual(len(stats['factors']), 2)
        
        # Verify specific factor statistics
        factor0 = stats['factors'][0]
        self.assertEqual(factor0['num_states'], 3)
        self.assertTrue('mean_entropy' in factor0)
        self.assertTrue('state_changes' in factor0)
        self.assertTrue('max_belief_states' in factor0)
        
        # Verify empty input handling
        empty_stats = calculate_belief_statistics([])
        self.assertTrue('error' in empty_stats)
        
if __name__ == '__main__':
    unittest.main() 