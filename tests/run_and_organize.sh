#!/bin/bash
# run_and_organize.sh - Run tests and organize output files

# Display help text
display_help() {
    echo "PyMDP-MCP Test Runner and Organizer"
    echo ""
    echo "Usage: ./run_and_organize.sh [options]"
    echo ""
    echo "Options:"
    echo "  -h, --help             Display this help message"
    echo "  -c, --category [name]  Run tests in category (interface, mcp, advanced, visualization, core, all)"
    echo "                         Default: all (runs ALL tests)"
    echo "  -f, --file [path]      Run a specific test file"
    echo "  -k, --keep             Keep existing output files"
    echo "  -v, --verbose          Increase verbosity"
    echo "  -s, --summary          Generate a detailed test summary"
    echo ""
    echo "Examples:"
    echo "  ./run_and_organize.sh              # Runs ALL tests"
    echo "  ./run_and_organize.sh --category visualization"
    echo "  ./run_and_organize.sh --file test_agent_visualization.py --summary"
    echo ""
}

# Parameters to pass to run_tests.py
# Default to running all tests
PARAMS="--organize --category all"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            display_help
            exit 0
            ;;
        -c|--category)
            # Override the default category
            PARAMS=$(echo "$PARAMS" | sed 's/--category all//')
            PARAMS="$PARAMS --category $2"
            shift 2
            ;;
        -f|--file)
            # If a file is specified, remove the category parameter
            PARAMS=$(echo "$PARAMS" | sed 's/--category all//')
            PARAMS="$PARAMS --file $2"
            shift 2
            ;;
        -k|--keep)
            PARAMS="$PARAMS --keep-old"
            shift
            ;;
        -v|--verbose)
            PARAMS="$PARAMS --verbose"
            shift
            ;;
        -s|--summary)
            PARAMS="$PARAMS --summary"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            display_help
            exit 1
            ;;
    esac
done

# Check if run_tests.py exists in parent directory
if [ ! -f "../run_tests.py" ]; then
    echo "Error: run_tests.py not found in parent directory"
    exit 1
fi

# Run the tests with the specified parameters
echo "Running tests with parameters: $PARAMS"
echo "Note: By default this runs ALL tests in ALL categories unless overridden with --category or --file"
cd ..
python3 run_tests.py $PARAMS

# Return to the tests directory
cd tests

# Check if output directory exists
if [ ! -d "output" ]; then
    echo "Error: output directory not found"
    exit 1
fi

echo ""
echo "Test run complete."
echo "Output files are organized in the following directories:"
ls -la output/

# Check test outputs and warn about empty directories
echo ""
echo "Files by category:"
for dir in output/*/; do
    count=$(find "$dir" -type f | wc -l)
    dir_name=$(basename "$dir")
    
    if [ $count -eq 0 ]; then
        echo "  $dir_name: WARNING - No files found. Tests may have failed or not been run."
    else
        echo "  $dir_name: $count files"
    fi
done

# Add specific message about belief_dynamics
if [ -d "output/belief_dynamics" ]; then
    belief_count=$(find "output/belief_dynamics" -type f | wc -l)
    if [ $belief_count -eq 0 ]; then
        echo ""
        echo "NOTE: The belief_dynamics directory is empty. You need to run visualization tests"
        echo "      or tests that include belief tracking to populate this directory."
        echo "      Try: ./run_and_organize.sh --category visualization"
    fi
fi

exit 0 