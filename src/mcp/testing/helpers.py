"""
Test Helpers for PyMDP with MCP.

This module provides utility functions for testing PyMDP with MCP.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def compare_matrices(matrix1, matrix2, rtol=1e-5, atol=1e-8):
    """Compare two matrices for approximate equality.
    
    Parameters
    ----------
    matrix1 : list or numpy.ndarray
        First matrix to compare
    matrix2 : list or numpy.ndarray
        Second matrix to compare
    rtol : float, optional
        Relative tolerance, by default 1e-5
    atol : float, optional
        Absolute tolerance, by default 1e-8
    
    Returns
    -------
    bool
        True if matrices are approximately equal, False otherwise
    """
    # Convert lists to numpy arrays if needed
    if isinstance(matrix1, list):
        matrix1 = np.array(matrix1)
    if isinstance(matrix2, list):
        matrix2 = np.array(matrix2)
    
    # Check shapes
    if matrix1.shape != matrix2.shape:
        return False
    
    # Check values
    return np.allclose(matrix1, matrix2, rtol=rtol, atol=atol)

def compare_generative_models(model1, model2, rtol=1e-5, atol=1e-8):
    """Compare two generative models for approximate equality.
    
    Parameters
    ----------
    model1 : dict
        First generative model to compare
    model2 : dict
        Second generative model to compare
    rtol : float, optional
        Relative tolerance, by default 1e-5
    atol : float, optional
        Absolute tolerance, by default 1e-8
    
    Returns
    -------
    dict
        Dictionary with comparison results for each matrix type
    """
    results = {}
    
    # Check A matrices
    if 'A' in model1 and 'A' in model2:
        A1 = model1['A']
        A2 = model2['A']
        
        # Check number of modalities
        if len(A1) != len(A2):
            results['A'] = False
        else:
            # Compare each modality
            A_results = []
            for i in range(len(A1)):
                if i < len(A1) and i < len(A2):
                    A_results.append(compare_matrices(A1[i], A2[i], rtol, atol))
                else:
                    A_results.append(False)
            
            results['A'] = all(A_results)
    else:
        results['A'] = 'missing'
    
    # Check B matrices
    if 'B' in model1 and 'B' in model2:
        B1 = model1['B']
        B2 = model2['B']
        
        # Check number of factors
        if len(B1) != len(B2):
            results['B'] = False
        else:
            # Compare each factor
            B_results = []
            for i in range(len(B1)):
                if i < len(B1) and i < len(B2):
                    B_results.append(compare_matrices(B1[i], B2[i], rtol, atol))
                else:
                    B_results.append(False)
            
            results['B'] = all(B_results)
    else:
        results['B'] = 'missing'
    
    # Check C matrices
    if 'C' in model1 and 'C' in model2:
        C1 = model1['C']
        C2 = model2['C']
        
        # Check number of modalities
        if len(C1) != len(C2):
            results['C'] = False
        else:
            # Compare each modality
            C_results = []
            for i in range(len(C1)):
                if i < len(C1) and i < len(C2):
                    C_results.append(compare_matrices(C1[i], C2[i], rtol, atol))
                else:
                    C_results.append(False)
            
            results['C'] = all(C_results)
    else:
        results['C'] = 'missing'
    
    # Check D matrices
    if 'D' in model1 and 'D' in model2:
        D1 = model1['D']
        D2 = model2['D']
        
        # Check number of factors
        if len(D1) != len(D2):
            results['D'] = False
        else:
            # Compare each factor
            D_results = []
            for i in range(len(D1)):
                if i < len(D1) and i < len(D2):
                    D_results.append(compare_matrices(D1[i], D2[i], rtol, atol))
                else:
                    D_results.append(False)
            
            results['D'] = all(D_results)
    else:
        results['D'] = 'missing'
    
    # Overall result
    results['overall'] = all(result is True for result in results.values())
    
    return results

def save_test_results(results, output_dir, test_name, include_timestamp=True):
    """Save test results to a file.
    
    Parameters
    ----------
    results : dict
        Test results to save
    output_dir : str
        Directory to save results to
    test_name : str
        Name of the test
    include_timestamp : bool, optional
        Whether to include a timestamp in the filename, by default True
    
    Returns
    -------
    str
        Path to the saved results file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Add timestamp to test name if requested
    if include_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{test_name}_{timestamp}.json"
    else:
        filename = f"{test_name}.json"
    
    # Add metadata to results
    results_with_metadata = results.copy()
    results_with_metadata['metadata'] = {
        'test_name': test_name,
        'timestamp': datetime.now().isoformat(),
    }
    
    # Save results to file
    output_path = os.path.join(output_dir, filename)
    with open(output_path, 'w') as f:
        json.dump(results_with_metadata, f, indent=2)
    
    return output_path

def generate_test_report(results_dir, output_file, report_title="Test Results"):
    """Generate a markdown report from test results.
    
    Parameters
    ----------
    results_dir : str
        Directory containing test result files
    output_file : str
        Path to save the report to
    report_title : str, optional
        Title for the report, by default "Test Results"
    
    Returns
    -------
    str
        Path to the generated report
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Get all JSON result files
    result_files = [f for f in os.listdir(results_dir) if f.endswith('.json')]
    
    # Sort by modification time (newest first)
    result_files.sort(key=lambda f: os.path.getmtime(os.path.join(results_dir, f)), reverse=True)
    
    # Generate report
    with open(output_file, 'w') as f:
        # Write header
        f.write(f"# {report_title}\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Write summary
        f.write("## Summary\n\n")
        f.write(f"Total test results: {len(result_files)}\n\n")
        
        # Write table of contents
        f.write("## Contents\n\n")
        for i, filename in enumerate(result_files):
            test_name = os.path.splitext(filename)[0]
            f.write(f"{i+1}. [{test_name}](#{test_name.lower().replace('_', '-')})\n")
        f.write("\n")
        
        # Write details for each test
        f.write("## Test Details\n\n")
        for filename in result_files:
            result_path = os.path.join(results_dir, filename)
            test_name = os.path.splitext(filename)[0]
            
            # Read test results
            with open(result_path, 'r') as rf:
                try:
                    result_data = json.load(rf)
                except json.JSONDecodeError:
                    f.write(f"### {test_name}\n\n")
                    f.write("*Error: Could not parse JSON file*\n\n")
                    continue
            
            # Write test section
            f.write(f"### {test_name}\n\n")
            
            # Write metadata if available
            if 'metadata' in result_data:
                metadata = result_data['metadata']
                if 'timestamp' in metadata:
                    f.write(f"Timestamp: {metadata['timestamp']}\n\n")
            
            # Write overall result if available
            if 'overall' in result_data:
                overall = result_data['overall']
                if isinstance(overall, bool):
                    status = "✅ Passed" if overall else "❌ Failed"
                    f.write(f"**Status: {status}**\n\n")
            
            # Write detailed results
            f.write("```json\n")
            # Remove metadata for cleaner output
            if 'metadata' in result_data:
                result_data_clean = result_data.copy()
                del result_data_clean['metadata']
                f.write(json.dumps(result_data_clean, indent=2))
            else:
                f.write(json.dumps(result_data, indent=2))
            f.write("\n```\n\n")
    
    return output_file

def plot_test_results(results, output_file, title="Test Results"):
    """Plot test results visually.
    
    Parameters
    ----------
    results : dict
        Test results to plot
    output_file : str
        Path to save the plot to
    title : str, optional
        Title for the plot, by default "Test Results"
    
    Returns
    -------
    str
        Path to the saved plot
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Extract metrics to plot
    metrics = []
    values = []
    
    for key, value in results.items():
        # Skip metadata and overall
        if key in ['metadata', 'overall']:
            continue
        
        # Handle different value types
        if isinstance(value, bool):
            metrics.append(key)
            values.append(1.0 if value else 0.0)
        elif isinstance(value, (int, float)):
            metrics.append(key)
            values.append(value)
        elif isinstance(value, dict) and 'value' in value:
            metrics.append(key)
            values.append(value['value'])
    
    # If no metrics to plot, return None
    if not metrics:
        return None
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Define colors based on values
    colors = ['green' if v >= 0.8 else 'orange' if v >= 0.5 else 'red' for v in values]
    
    # Create bars
    bars = ax.bar(metrics, values, color=colors)
    
    # Add value labels
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.2f}' if isinstance(value, float) else str(value),
                ha='center', va='bottom', fontsize=10)
    
    # Customize plot
    ax.set_ylim(0, 1.1)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Metrics', fontsize=12)
    ax.set_ylabel('Values', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    
    # Save plot
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close(fig)
    
    return output_file

def memory_profile(func):
    """Decorator to profile memory usage of a function.
    
    Parameters
    ----------
    func : function
        Function to profile
    
    Returns
    -------
    function
        Wrapped function with memory profiling
    """
    def wrapper(*args, **kwargs):
        try:
            import psutil
            import os
            import gc
            
            # Force garbage collection
            gc.collect()
            
            # Get initial memory usage
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Call function
            start_time = datetime.now()
            result = func(*args, **kwargs)
            end_time = datetime.now()
            
            # Force garbage collection again
            gc.collect()
            
            # Get final memory usage
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Calculate metrics
            duration = (end_time - start_time).total_seconds()
            memory_change = final_memory - initial_memory
            
            # Print results
            print(f"\nMemory Profile for {func.__name__}:")
            print(f"  Duration: {duration:.4f} seconds")
            print(f"  Initial memory: {initial_memory:.2f} MB")
            print(f"  Final memory: {final_memory:.2f} MB")
            print(f"  Memory change: {memory_change:.2f} MB")
            
            return result
        
        except ImportError:
            # If psutil is not available, just run the function
            return func(*args, **kwargs)
    
    return wrapper 