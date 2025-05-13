"""Comprehensive MCP Functionality Tests for PyMDP-MCP.

This file tests all aspects of the MCP server implementation for PyMDP,
including server initialization, tool registration, and end-to-end functionality.
"""

import unittest
import asyncio
import json
import sys
import os
import numpy as np
import matplotlib
from contextlib import asynccontextmanager
matplotlib.use('Agg')  # Use non-interactive Agg backend for testing

# Setup output directory
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
GEN_MODEL_DIR = os.path.join(OUTPUT_DIR, "generative_models")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(GEN_MODEL_DIR, exist_ok=True)

# Add the src directory directly to path
src_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Add the pymdp-clone directory to path to ensure we're using the real PyMDP library
pymdp_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'pymdp-clone')
if pymdp_dir not in sys.path:
    sys.path.insert(0, pymdp_dir)

# Import MCP components
from mcp.server.fastmcp import FastMCP, Context
from mcp.utils import PyMDPInterface, get_pymdp_interface

# Import main which contains the MCP tool definitions
import main

class TestMCPFunctionality(unittest.TestCase):
    """Test the full MCP functionality for PyMDP."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a test context
        self.mock_lifespan_context = type('MockLifespanContext', (), {'pymdp_interface': PyMDPInterface()})
        self.mock_request_context = type('MockRequestContext', (), {'lifespan_context': self.mock_lifespan_context})
        self.ctx = Context(request_context=self.mock_request_context)
        
        # Ensure output directory exists
        os.makedirs(os.path.join(OUTPUT_DIR, "logs"), exist_ok=True)
        
    def run_async_test(self, coroutine):
        """Helper to run async test functions."""
        return asyncio.run(coroutine)
    
    def test_mcp_server_initialization(self):
        """Test MCP server initialization."""
        # Create a test lifespan function
        @asynccontextmanager
        async def test_lifespan(server):
            try:
                yield {"test_key": "test_value"}
            finally:
                pass
        
        # Initialize MCP server
        server = FastMCP(
            "test-server",
            "Test server for PyMDP-MCP",
            lifespan=test_lifespan,
            host="localhost",
            port="9000",
            transport="stdio"
        )
        
        # Verify server properties
        self.assertEqual(server.name, "test-server")
        self.assertEqual(server.description, "Test server for PyMDP-MCP")
        self.assertEqual(server.host, "localhost")
        self.assertEqual(server.port, "9000")
        self.assertEqual(server.transport, "stdio")
        
        # Verify server has empty tools at initialization
        self.assertEqual(len(server.tools), 0)
    
    def test_tool_registration(self):
        """Test registering a function as an MCP tool."""
        # Create a test server
        @asynccontextmanager
        async def test_lifespan(server):
            try:
                yield {"test_key": "test_value"}
            finally:
                pass
        
        server = FastMCP(
            "test-server",
            "Test server for PyMDP-MCP",
            lifespan=test_lifespan
        )
        
        # Define and register a test tool
        @server.tool()
        async def test_tool(ctx, param1: str, param2: int = 42):
            """Test tool documentation."""
            return f"{param1}: {param2}"
        
        # Verify tool was registered
        self.assertIn("test_tool", server.tools)
        self.assertIn("test_tool", server.tool_descriptions)
        
        # Verify tool description
        tool_desc = server.tool_descriptions["test_tool"]
        self.assertEqual(tool_desc["name"], "test_tool")
        self.assertEqual(tool_desc["description"], "Test tool documentation.")
        
        # Verify parameter descriptions
        self.assertIn("param1", tool_desc["parameters"])
        self.assertIn("param2", tool_desc["parameters"])
        self.assertEqual(tool_desc["parameters"]["param1"]["type"], "string")
        self.assertEqual(tool_desc["parameters"]["param2"]["type"], "number")
        self.assertTrue(tool_desc["parameters"]["param1"]["required"])
        self.assertFalse(tool_desc["parameters"]["param2"]["required"])
    
    def test_create_agent_tool(self):
        """Test the create_agent MCP tool."""
        # Define a simple generative model
        A = [
            [[0.9, 0.1], [0.1, 0.9]]  # One observation modality
        ]
        B = [
            [[[0.9, 0.1], [0.1, 0.9]], [[0.5, 0.5], [0.5, 0.5]]]  # Transitions for control states
        ]
        generative_model = {
            "A": A,
            "B": B
        }
        
        # Call the tool
        result = self.run_async_test(main.create_agent(
            self.ctx,
            "test_agent",
            json.dumps(generative_model)
        ))
        
        # Parse result
        result_obj = json.loads(result)
        
        # Verify agent was created correctly
        self.assertEqual(result_obj["name"], "test_agent")
        self.assertEqual(result_obj["num_observation_modalities"], 1)
        self.assertEqual(result_obj["num_state_factors"], 1)
        
        # Save result to output directory
        with open(os.path.join(OUTPUT_DIR, "logs", "mcp_create_agent_result.json"), "w") as f:
            f.write(result)
        # Save generative model to generative_models directory
        with open(os.path.join(GEN_MODEL_DIR, "test_agent_generative_model.json"), "w") as f:
            json.dump(generative_model, f, indent=2)
    
    def test_schema_generation(self):
        """Test OpenAPI schema generation for the MCP server."""
        # Create a test server with tools
        @asynccontextmanager
        async def test_lifespan(server):
            try:
                yield {"test_key": "test_value"}
            finally:
                pass
        
        server = FastMCP(
            "test-server",
            "Test server for PyMDP-MCP",
            lifespan=test_lifespan
        )
        
        # Register a few test tools
        @server.tool()
        async def tool1(ctx, param1: str):
            """Tool 1 documentation."""
            return param1
        
        @server.tool()
        async def tool2(ctx, param1: int, param2: bool = False):
            """Tool 2 documentation."""
            return param1 if param2 else 0
        
        # Generate schema
        schema = asyncio.run(server.get_schema())
        
        # Verify schema structure
        self.assertEqual(schema["info"]["title"], "test-server")
        self.assertEqual(schema["info"]["description"], "Test server for PyMDP-MCP")
        
        # Save schema to output directory
        with open(os.path.join(OUTPUT_DIR, "logs", "mcp_schema.json"), "w") as f:
            json.dump(schema, f, indent=2)
    
    def test_get_pymdp_interface(self):
        """Test the get_pymdp_interface function."""
        # Get the interface
        interface = get_pymdp_interface()
        
        # Verify it's an instance of PyMDPInterface
        self.assertIsInstance(interface, PyMDPInterface)

if __name__ == "__main__":
    unittest.main()
