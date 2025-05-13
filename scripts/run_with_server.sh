#!/bin/bash
# Run the MCP examples with a local server

# Set default values
PORT=8090
OUTPUT_DIR="$(pwd)/outputs/run_$(date +%Y%m%d_%H%M%S)"
EXAMPLE_SCRIPT="mcp_gridworld_examples.py"

# Parse command line arguments
while [ "$#" -gt 0 ]; do
  case "$1" in
    --port)
      PORT="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --example)
      EXAMPLE_SCRIPT="$2"
      shift 2
      ;;
    *)
      echo "Unknown parameter: $1"
      echo "Usage: $0 [--port PORT] [--output-dir DIR] [--example SCRIPT]"
      exit 1
      ;;
  esac
done

# Create output directory
mkdir -p "$OUTPUT_DIR"
echo "Output will be saved to: $OUTPUT_DIR"

# Define the server output directory
SERVER_OUTPUT_DIR="$OUTPUT_DIR/server"
mkdir -p "$SERVER_OUTPUT_DIR"

# Define a function to clean up the server process
cleanup() {
    echo "Stopping MCP server..."
    if [ -n "$SERVER_PID" ]; then
        kill $SERVER_PID
        wait $SERVER_PID 2>/dev/null
    fi
    exit
}

# Set up a trap to clean up on exit
trap cleanup EXIT INT TERM

# Start the server in the background
echo "Starting MCP server on port $PORT..."
python3 start_mcp_server.py --port $PORT --output-dir "$SERVER_OUTPUT_DIR" &
SERVER_PID=$!

# Wait for the server to start
echo "Waiting for server to start..."
MAX_RETRIES=10
for i in $(seq 1 $MAX_RETRIES); do
    if curl -s "http://localhost:$PORT/ping" >/dev/null 2>&1; then
        echo "Server started successfully!"
        break
    fi
    
    # Check if server process is still running
    if ! ps -p $SERVER_PID > /dev/null; then
        echo "Error: Server process failed to start"
        exit 1
    fi
    
    echo "Retry $i/$MAX_RETRIES: Server not ready yet..."
    sleep 1
done

# Verify server is running
if ! curl -s "http://localhost:$PORT/ping" >/dev/null 2>&1; then
    echo "Error: Failed to connect to server after $MAX_RETRIES retries"
    exit 1
fi

# Create example output directory
EXAMPLE_OUTPUT_DIR="$OUTPUT_DIR/$(basename $EXAMPLE_SCRIPT .py)"
mkdir -p "$EXAMPLE_OUTPUT_DIR"

# Run the example
echo "Running $EXAMPLE_SCRIPT..."
python3 "$EXAMPLE_SCRIPT" --port $PORT --output-dir "$EXAMPLE_OUTPUT_DIR"
EXAMPLE_EXIT_CODE=$?

if [ $EXAMPLE_EXIT_CODE -eq 0 ]; then
    echo "Example completed successfully!"
else
    echo "Example failed with exit code $EXAMPLE_EXIT_CODE"
fi

# Exit (cleanup will be called automatically)
exit $EXAMPLE_EXIT_CODE 