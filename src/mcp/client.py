"""
MCP Client Module

This module provides client implementations for interacting with MCP servers.
It includes both a network-based client for communicating with remote MCP servers
and a direct client for using MCP tools without a server.
"""

import os
import json
import time
import datetime
import asyncio
import sys
import aiohttp
from mcp.server.fastmcp import Context

# Add the pymdp-clone directory to the path
pymdp_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'pymdp-clone'))
if pymdp_dir not in sys.path:
    sys.path.insert(0, pymdp_dir)

# Import utils for direct client functionality
from utils import get_pymdp_interface

def create_output_dir(name_prefix="output"):
    """Create a timestamped output directory for storing results.
    
    Args:
        name_prefix: Prefix for the directory name (default: "output")
        
    Returns:
        str: Path to the created output directory
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                             "scripts", f"{name_prefix}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Created output directory: {output_dir}")
    return output_dir

class MCPClient:
    """Client for communicating with the MCP server"""
    
    def __init__(self, host="localhost", port=8080, use_ssl=False, output_dir=None):
        """Initialize the MCP client.
        
        Args:
            host: Hostname of the MCP server
            port: Port number of the MCP server
            use_ssl: Whether to use HTTPS instead of HTTP
            output_dir: Directory to save outputs
        """
        protocol = "https" if use_ssl else "http"
        self.server_url = f"{protocol}://{host}:{port}"
        self.output_dir = output_dir
        self.session = None
        self.direct_client = None
        
        # Create output directory if specified
        if self.output_dir and not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
    
    async def __aenter__(self):
        """Async context manager entry point."""
        if self.session is None:
            self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit point."""
        if self.session is not None:
            await self.session.close()
            self.session = None
    
    async def ping(self):
        """Ping the MCP server to check connectivity.
        
        Returns:
            dict: Response from the server or error information
        """
        try:
            async with aiohttp.ClientSession() as session:
                try:
                    ping_url = f"{self.server_url}/ping"
                    async with session.get(ping_url, timeout=5) as response:
                        if response.status == 200:
                            return await response.json()
                        else:
                            return {"status": "error", "message": f"Server returned status {response.status}"}
                except (aiohttp.ClientConnectorError, asyncio.TimeoutError) as e:
                    return {"status": "error", "message": f"Connection error: {str(e)}"}
        except Exception as e:
            return {"status": "error", "message": f"General error: {str(e)}"}
    
    async def get_tools(self):
        """Get the list of available tools from the MCP server.
        
        Returns:
            list: List of available tools
        """
        try:
            async with aiohttp.ClientSession() as session:
                tools_url = f"{self.server_url}/tools"
                async with session.get(tools_url, timeout=5) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("tools", [])
                    else:
                        print(f"Failed to get tools: Server returned status {response.status}")
                        return []
        except Exception as e:
            print(f"Error getting tools: {str(e)}")
            return []
    
    async def call_tool(self, tool_id, params=None):
        """Call an MCP tool on the server.
        
        Args:
            tool_id: ID of the tool to call
            params: Parameters to pass to the tool
            
        Returns:
            dict: The parsed result from the tool
        """
        if params is None:
            params = {}
            
        # Log request
        if self.output_dir:
            req_file = os.path.join(self.output_dir, f"{tool_id}_request_{int(time.time())}.json")
            with open(req_file, 'w') as f:
                json.dump({"tool_id": tool_id, "params": params}, f, indent=2)
        
        # Prepare request data
        request_data = {"params": params}
        
        # Make request to server
        try:
            # Use existing session if available, otherwise create a new one
            session_to_use = self.session or aiohttp.ClientSession()
            tool_url = f"{self.server_url}/tools/{tool_id}"
            
            try:
                async with session_to_use.post(tool_url, json=request_data, timeout=30) as response:
                    response_text = await response.text()
                    
                    # Try to parse JSON response
                    try:
                        response_data = json.loads(response_text)
                    except json.JSONDecodeError:
                        response_data = {"error": f"Invalid JSON response: {response_text[:100]}..."}
                    
                    # Save response
                    if self.output_dir:
                        resp_file = os.path.join(self.output_dir, f"{tool_id}_response_{int(time.time())}.json")
                        with open(resp_file, 'w') as f:
                            json.dump(response_data, f, indent=2)
                    
                    # Close session if we created it here
                    if session_to_use is not self.session:
                        await session_to_use.close()
                    
                    return response_data
            except (aiohttp.ClientConnectorError, asyncio.TimeoutError) as e:
                error_data = {"status": "error", "error": f"Connection error: {str(e)}"}
                print(f"Connection error calling tool {tool_id}: {str(e)}")
                
                # Close session if we created it here
                if session_to_use is not self.session:
                    await session_to_use.close()
                
                return error_data
        except Exception as e:
            error_data = {"status": "error", "error": f"General error: {str(e)}"}
            print(f"Error calling tool {tool_id}: {str(e)}")
            
            # Log error
            if self.output_dir:
                error_file = os.path.join(self.output_dir, f"{tool_id}_error_{int(time.time())}.json")
                with open(error_file, 'w') as f:
                    json.dump(error_data, f, indent=2)
            
            return error_data

class DirectMCPClient:
    """Client that directly calls MCP tools without going through the server.
    
    This is useful for testing and development when the server isn't running.
    """
    
    def __init__(self, output_dir=None):
        """Initialize the direct MCP client.
        
        Args:
            output_dir: Directory to save outputs
        """
        self.output_dir = output_dir
        self.results = {}
        
        # Create a PyMDP interface instance
        self.pymdp_interface = get_pymdp_interface()
        
        # Create a context to pass to the tools
        mock_lifespan_context = type('MockLifespanContext', (), {'pymdp_interface': self.pymdp_interface})
        mock_request_context = type('MockRequestContext', (), {'lifespan_context': mock_lifespan_context})
        self.ctx = Context(request_context=mock_request_context)
    
    async def call_tool(self, tool_name, **params):
        """Call an MCP tool directly.
        
        Args:
            tool_name: Name of the tool to call
            **params: Parameters to pass to the tool
            
        Returns:
            The parsed result from the tool
        """
        # Import main which has all tool implementations
        import main
        
        # Log request
        if self.output_dir:
            req_file = os.path.join(self.output_dir, f"{tool_name}_direct_request_{int(time.time())}.json")
            with open(req_file, 'w') as f:
                json.dump({"name": tool_name, "parameters": params}, f, indent=2)
        
        try:
            # Get the tool function from main
            tool_func = getattr(main, tool_name)
            
            # Call the tool
            result_json = await tool_func(self.ctx, **params)
            result = json.loads(result_json)
            
            # Save response
            if self.output_dir:
                resp_file = os.path.join(self.output_dir, f"{tool_name}_direct_response_{int(time.time())}.json")
                with open(resp_file, 'w') as f:
                    json.dump(result, f, indent=2)
            
            # Store result
            self.results[tool_name] = result
            return result
        except Exception as e:
            error_data = {"status": "error", "error": f"Direct client error: {str(e)}"}
            if self.output_dir:
                error_file = os.path.join(self.output_dir, f"{tool_name}_direct_error_{int(time.time())}.json")
                with open(error_file, 'w') as f:
                    json.dump(error_data, f, indent=2)
            return error_data

async def get_best_client(output_dir=None):
    """Determine and return the best available MCP client.
    
    This function attempts to connect to a running MCP server.
    If successful, it returns an MCPClient; otherwise, it returns a DirectMCPClient.
    
    Args:
        output_dir: Directory to save outputs
        
    Returns:
        Union[MCPClient, DirectMCPClient]: The most appropriate client
    """
    try:
        # Try to connect to the server
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(f"http://localhost:8050/schema", timeout=2) as response:
                    if response.status == 200:
                        print("Connected to MCP server. Using server-based client.")
                        return MCPClient(output_dir=output_dir)
                    else:
                        print("MCP server returned unexpected status. Using direct client.")
                        return DirectMCPClient(output_dir=output_dir)
            except (aiohttp.ClientConnectorError, asyncio.TimeoutError):
                print("MCP server not available. Using direct client.")
                return DirectMCPClient(output_dir=output_dir)
    except Exception as e:
        print(f"Error checking MCP server: {str(e)}. Using direct client.")
        return DirectMCPClient(output_dir=output_dir) 