"""
FastMCP implementation for the Model Context Protocol.

This module provides a server implementation for the MCP protocol,
designed to be efficient and easy to use for exposing PyMDP functionality.
"""

import asyncio
import json
import os
import inspect
from typing import Dict, Any, Callable, Optional, AsyncIterator, List, Union
from dataclasses import dataclass
from contextlib import asynccontextmanager
import time
import sys

# Constants
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = "8050"
DEFAULT_TRANSPORT = "sse"

@dataclass
class Context:
    """Context object for MCP tool calls."""
    request_context: Any

class RequestContext:
    """Context for the current request."""
    def __init__(self, lifespan_context: Any):
        self.lifespan_context = lifespan_context
        self.start_time = time.time()
        self.tool_calls: List[Dict[str, Any]] = []

class FastMCP:
    """
    FastMCP server implementation.
    
    This class provides a server that exposes MCP tools to clients.
    """
    
    def __init__(self, name: str, description: str, lifespan, 
                 host: str = None, port: str = None, transport: str = None):
        """
        Initialize an MCP server.
        
        Args:
            name: Name of the MCP server
            description: Description of the MCP server
            lifespan: Async context manager for server lifespan
            host: Host to bind to (for SSE transport)
            port: Port to listen on (for SSE transport) 
            transport: Transport protocol ('sse' or 'stdio')
        """
        self.name = name
        self.description = description
        self.lifespan = lifespan
        
        # Get configuration from environment if not provided
        self.host = host or os.environ.get("HOST", DEFAULT_HOST)
        self.port = port or os.environ.get("PORT", DEFAULT_PORT)
        self.transport = transport or os.environ.get("TRANSPORT", DEFAULT_TRANSPORT)
        
        # Initialize storage for tools
        self.tools: Dict[str, Callable] = {}
        self.tool_descriptions: Dict[str, Dict[str, Any]] = {}
    
    def tool(self, name: str = None, description: str = None):
        """
        Decorator to register a function as an MCP tool.
        
        Args:
            name: Optional override for the tool name
            description: Optional description for the tool
            
        Returns:
            Decorator function
        """
        def decorator(func):
            # Get tool name
            tool_name = name or func.__name__
            
            # Get tool description from docstring if not provided
            tool_desc = description
            if not tool_desc and func.__doc__:
                tool_desc = func.__doc__.strip()
            
            # Parse function signature for parameters
            sig = inspect.signature(func)
            parameters = {}
            
            for param_name, param in sig.parameters.items():
                # Skip context parameter
                if param_name == "ctx":
                    continue
                
                # Extract parameter type and default value
                param_type = "string"  # Default type
                if param.annotation != inspect.Parameter.empty:
                    if param.annotation == int:
                        param_type = "number"
                    elif param.annotation == bool:
                        param_type = "boolean"
                    elif param.annotation == dict or param.annotation == Dict:
                        param_type = "object"
                    elif param.annotation == list or param.annotation == List:
                        param_type = "array"
                
                # Determine if parameter is required
                required = param.default == inspect.Parameter.empty
                
                # Create parameter description
                parameters[param_name] = {
                    "type": param_type,
                    "required": required
                }
                
                # Add default value if available
                if not required:
                    parameters[param_name]["default"] = param.default
            
            # Store tool information
            self.tools[tool_name] = func
            self.tool_descriptions[tool_name] = {
                "name": tool_name,
                "description": tool_desc,
                "parameters": parameters
            }
            
            return func
        
        return decorator
    
    async def handle_request(self, request_data: Dict[str, Any], lifespan_context: Any) -> Dict[str, Any]:
        """
        Handle an incoming MCP request.
        
        Args:
            request_data: The request data
            lifespan_context: The lifespan context
            
        Returns:
            Response data
        """
        # Extract request information
        request_id = request_data.get("id")
        tool_name = request_data.get("name")
        parameters = request_data.get("parameters", {})
        
        # Check if tool exists
        if tool_name not in self.tools:
            return {
                "id": request_id,
                "status": "error",
                "error": f"Unknown tool: {tool_name}"
            }
        
        # Create request context
        request_context = RequestContext(lifespan_context)
        context = Context(request_context=request_context)
        
        try:
            # Call the tool function
            tool_func = self.tools[tool_name]
            result = await tool_func(context, **parameters)
            
            # Return successful response
            return {
                "id": request_id,
                "status": "success",
                "result": result
            }
        except Exception as e:
            # Return error response
            import traceback
            traceback.print_exc()
            return {
                "id": request_id,
                "status": "error",
                "error": str(e)
            }
    
    async def get_schema(self) -> Dict[str, Any]:
        """
        Get the OpenAPI schema for the MCP server.
        
        Returns:
            OpenAPI schema
        """
        tools = []
        for tool_name, tool_desc in self.tool_descriptions.items():
            tools.append({
                "name": tool_name,
                "description": tool_desc.get("description", ""),
                "parameters": {
                    "type": "object",
                    "properties": tool_desc.get("parameters", {}),
                    "required": [
                        param_name for param_name, param_desc in 
                        tool_desc.get("parameters", {}).items() 
                        if param_desc.get("required", False)
                    ]
                }
            })
        
        return {
            "openapi": "3.1.0",
            "info": {
                "title": self.name,
                "description": self.description,
                "version": "1.0.0"
            },
            "servers": [
                {
                    "url": f"http://{self.host}:{self.port}"
                }
            ],
            "paths": {
                "/tools": {
                    "get": {
                        "summary": "Get available tools",
                        "responses": {
                            "200": {
                                "description": "List of available tools",
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "array",
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "name": {"type": "string"},
                                                    "description": {"type": "string"},
                                                    "parameters": {"type": "object"}
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "components": {
                "schemas": {}
            }
        }
    
    async def run_sse(self):
        """Run the MCP server using SSE transport."""
        try:
            from fastapi import FastAPI, Request
            from fastapi.responses import StreamingResponse
            from fastapi.middleware.cors import CORSMiddleware
            import uvicorn
        except ImportError:
            print("Error: FastAPI and uvicorn are required for SSE transport")
            print("Install with: pip install fastapi uvicorn")
            sys.exit(1)
        
        app = FastAPI(title=self.name, description=self.description)
        
        # Add CORS middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Create lifespan context
        @app.on_event("startup")
        async def startup():
            app.state.lifespan_cm = self.lifespan(self)
            app.state.lifespan_context = await app.state.lifespan_cm.__aenter__()
        
        @app.on_event("shutdown")
        async def shutdown():
            await app.state.lifespan_cm.__aexit__(None, None, None)
        
        # Define routes
        @app.get("/schema")
        async def get_schema():
            return await self.get_schema()
        
        @app.get("/tools")
        async def get_tools():
            return list(self.tool_descriptions.values())
        
        @app.post("/invoke")
        async def invoke_tool(request_data: dict):
            return await self.handle_request(request_data, app.state.lifespan_context)
        
        @app.get("/sse")
        async def sse_endpoint(request: Request):
            async def event_generator():
                # Send initial connection message
                yield f"data: {json.dumps({'type': 'connection', 'status': 'connected'})}\n\n"
                
                try:
                    # Keep connection alive
                    while True:
                        if await request.is_disconnected():
                            break
                        
                        # Process incoming requests
                        request_data = await request.json()
                        response = await self.handle_request(request_data, app.state.lifespan_context)
                        
                        # Send response
                        yield f"data: {json.dumps(response)}\n\n"
                        
                        # Brief pause to prevent tight loop
                        await asyncio.sleep(0.01)
                except Exception as e:
                    yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
            
            return StreamingResponse(
                event_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Content-Type": "text/event-stream"
                }
            )
        
        # Run the server
        uvicorn.run(app, host=self.host, port=int(self.port))
    
    async def run_stdio(self):
        """Run the MCP server using stdio transport."""
        # Create lifespan context
        async with self.lifespan(self) as lifespan_context:
            # Process input until EOF
            while True:
                try:
                    line = sys.stdin.readline()
                    if not line:
                        break
                    
                    # Parse request
                    request_data = json.loads(line)
                    
                    # Handle request
                    response = await self.handle_request(request_data, lifespan_context)
                    
                    # Send response
                    sys.stdout.write(json.dumps(response) + "\n")
                    sys.stdout.flush()
                except EOFError:
                    break
                except json.JSONDecodeError:
                    # Invalid JSON, send error response
                    sys.stdout.write(json.dumps({
                        "status": "error",
                        "error": "Invalid JSON request"
                    }) + "\n")
                    sys.stdout.flush()
                except Exception as e:
                    # General error, send error response
                    sys.stdout.write(json.dumps({
                        "status": "error",
                        "error": str(e)
                    }) + "\n")
                    sys.stdout.flush()
    
    async def run(self):
        """Run the MCP server with the configured transport."""
        if self.transport == "sse":
            await self.run_sse()
        elif self.transport == "stdio":
            await self.run_stdio()
        else:
            raise ValueError(f"Unsupported transport: {self.transport}")

# Shortcut function for tool decorator
tool = FastMCP.tool 