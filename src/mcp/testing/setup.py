"""
Test Setup Utilities for PyMDP with MCP.

This module provides utility functions for setting up test environments and directories.
"""

import os
import sys
import matplotlib

def setup_test_environment(use_non_interactive_backend=True):
    """Set up the test environment by configuring paths and matplotlib.
    
    Parameters
    ----------
    use_non_interactive_backend : bool, optional
        Whether to use a non-interactive matplotlib backend, by default True
    
    Returns
    -------
    dict
        Dictionary containing configuration parameters and paths
    """
    # Use a non-interactive matplotlib backend for testing
    if use_non_interactive_backend:
        matplotlib.use('Agg')
    
    # Get current directory (assuming this is being called from a test file)
    current_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
    
    # Get parent directory (project root)
    project_root = os.path.dirname(os.path.dirname(current_dir))
    
    # Set up source and PyMDP directories
    src_dir = os.path.join(project_root, 'src')
    pymdp_dir = os.path.join(project_root, 'pymdp-clone')
    
    # Add to Python path if not already present
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    
    if pymdp_dir not in sys.path:
        sys.path.insert(0, pymdp_dir)
    
    # Return configuration
    return {
        'project_root': project_root,
        'src_dir': src_dir,
        'pymdp_dir': pymdp_dir,
        'test_dir': current_dir
    }

def setup_output_directories(base_dir=None, create_subdirs=None):
    """Set up output directories for test results.
    
    Parameters
    ----------
    base_dir : str, optional
        Base directory for outputs, by default None (uses 'output' in test directory)
    create_subdirs : list, optional
        List of subdirectories to create, by default None
    
    Returns
    -------
    dict
        Dictionary of created directory paths
    """
    # Get current directory if not provided
    if base_dir is None:
        current_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
        base_dir = os.path.join(current_dir, 'output')
    
    # Create the main output directory
    os.makedirs(base_dir, exist_ok=True)
    
    # Set default subdirectories if not provided
    if create_subdirs is None:
        create_subdirs = [
            'generative_models',
            'free_energy',
            'belief_dynamics',
            'simulations',
            'logs'
        ]
    
    # Create subdirectories
    dir_paths = {'base': base_dir}
    for subdir in create_subdirs:
        path = os.path.join(base_dir, subdir)
        os.makedirs(path, exist_ok=True)
        dir_paths[subdir] = path
    
    return dir_paths

def clean_output_directory(directory, file_extensions=None):
    """Clean the specified output directory by removing files.
    
    Parameters
    ----------
    directory : str
        Directory to clean
    file_extensions : list, optional
        List of file extensions to remove, by default None (all files)
    
    Returns
    -------
    int
        Number of files removed
    """
    if not os.path.exists(directory):
        return 0
    
    files_removed = 0
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        
        # Skip directories (only remove files)
        if os.path.isdir(item_path):
            continue
        
        # Check if we should remove this file based on extension
        if file_extensions is None or any(item.endswith(ext) for ext in file_extensions):
            os.unlink(item_path)
            files_removed += 1
    
    return files_removed 