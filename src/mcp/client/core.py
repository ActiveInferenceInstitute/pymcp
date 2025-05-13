"""
MCP Client Core.

This module provides the main MCP client implementation for interacting with the MCP server.
"""

import os
import json
import asyncio
import logging
import aiohttp
from typing import Dict, Any, List, Optional, Union, Callable

from .config import load_config, get_server_url

# Set up logging
logger = logging.getLogger("mcp.client")

class MCPClient:
    """MCP client for interacting with the MCP server.
    
    Parameters
    ----------
    server_url : str, optional
        URL of the MCP server, by default None (use configuration)
    auth_token : str, optional
        Authentication token, by default None
    session : aiohttp.ClientSession, optional
        HTTP session to use, by default None (create a new session)
    config : Dict[str, Any], optional
        Client configuration, by default None (load from file)
    
    Attributes
    ----------
    server_url : str
        URL of the MCP server
    session : aiohttp.ClientSession
        HTTP session for making requests
    auth_token : str
        Authentication token for the MCP server
    config : Dict[str, Any]
        Client configuration
    """
    
    def __init__(
        self,
        server_url: Optional[str] = None,
        auth_token: Optional[str] = None,
        session: Optional[aiohttp.ClientSession] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize the MCP client."""
        # Load configuration if not provided
        if config is None:
            self.config = load_config()
        else:
            self.config = config
        
        # Set server URL
        if server_url is None:
            self.server_url = get_server_url(self.config)
        else:
            self.server_url = server_url
        
        # Set authentication token
        if auth_token is None:
            auth_config = self.config.get("auth", {})
            self.auth_token = auth_config.get("token")
        else:
            self.auth_token = auth_token
        
        # Set HTTP session
        self.session = session
        self._owns_session = session is None
    
    async def __aenter__(self):
        """Enter async context manager."""
        if self.session is None:
            self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit async context manager."""
        if self._owns_session and self.session is not None:
            await self.session.close()
            self.session = None
    
    async def _ensure_session(self):
        """Ensure that a session is available."""
        if self.session is None:
            self.session = aiohttp.ClientSession()
            self._owns_session = True
    
    async def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Make a request to the MCP server.
        
        Parameters
        ----------
        method : str
            HTTP method (GET, POST, etc.)
        endpoint : str
            API endpoint
        data : Dict[str, Any], optional
            Request data, by default None
        params : Dict[str, Any], optional
            Query parameters, by default None
        headers : Dict[str, Any], optional
            Request headers, by default None
        
        Returns
        -------
        Dict[str, Any]
            Response data
        
        Raises
        ------
        Exception
            If the request fails or the response is invalid
        """
        await self._ensure_session()
        
        # Create URL
        url = f"{self.server_url}{endpoint}"
        
        # Set default headers
        if headers is None:
            headers = {}
        
        # Add authentication if available
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        
        # Set content type
        if data is not None and "Content-Type" not in headers:
            headers["Content-Type"] = "application/json"
        
        # Make request
        options = self.config.get("options", {})
        timeout = aiohttp.ClientTimeout(total=options.get("timeout", 30))
        
        try:
            async with self.session.request(
                method=method,
                url=url,
                json=data,
                params=params,
                headers=headers,
                timeout=timeout
            ) as response:
                # Read response
                response_text = await response.text()
                
                # Check for success
                if response.status >= 400:
                    error_message = f"HTTP error {response.status}: {response_text}"
                    logger.error(error_message)
                    raise Exception(error_message)
                
                # Parse JSON response
                try:
                    return json.loads(response_text)
                except json.JSONDecodeError:
                    error_message = f"Invalid JSON response: {response_text}"
                    logger.error(error_message)
                    raise Exception(error_message)
        
        except asyncio.TimeoutError:
            error_message = f"Request timed out: {url}"
            logger.error(error_message)
            raise Exception(error_message)
        
        except aiohttp.ClientError as e:
            error_message = f"HTTP client error: {str(e)}"
            logger.error(error_message)
            raise Exception(error_message)
    
    async def ping(self) -> Dict[str, Any]:
        """Ping the MCP server to check if it's available.
        
        Returns
        -------
        Dict[str, Any]
            Server status
        """
        return await self._make_request("GET", "/ping")
    
    async def get_tools(self) -> List[Dict[str, Any]]:
        """Get the list of available tools from the MCP server.
        
        Returns
        -------
        List[Dict[str, Any]]
            List of available tools
        """
        response = await self._make_request("GET", "/tools")
        return response.get("tools", [])
    
    async def get_tool(self, tool_id: str) -> Dict[str, Any]:
        """Get information about a specific tool.
        
        Parameters
        ----------
        tool_id : str
            Tool ID
        
        Returns
        -------
        Dict[str, Any]
            Tool information
        """
        response = await self._make_request("GET", f"/tools/{tool_id}")
        return response.get("tool", {})
    
    async def call_tool(
        self,
        tool_id: str,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Call a tool on the MCP server.
        
        Parameters
        ----------
        tool_id : str
            Tool ID
        params : Dict[str, Any]
            Tool parameters
        
        Returns
        -------
        Dict[str, Any]
            Tool result
        """
        data = {"params": params}
        response = await self._make_request("POST", f"/tools/{tool_id}", data=data)
        return response.get("result", {})
    
    async def create_agent(
        self,
        name: str,
        model: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create a new agent on the MCP server.
        
        Parameters
        ----------
        name : str
            Agent name
        model : Dict[str, Any]
            Agent model (generative model)
        
        Returns
        -------
        Dict[str, Any]
            Agent information
        """
        data = {
            "name": name,
            "model": model
        }
        response = await self._make_request("POST", "/agents", data=data)
        return response.get("agent", {})
    
    async def get_agents(self) -> List[Dict[str, Any]]:
        """Get the list of agents from the MCP server.
        
        Returns
        -------
        List[Dict[str, Any]]
            List of agents
        """
        response = await self._make_request("GET", "/agents")
        return response.get("agents", [])
    
    async def get_agent(self, agent_id: str) -> Dict[str, Any]:
        """Get information about a specific agent.
        
        Parameters
        ----------
        agent_id : str
            Agent ID
        
        Returns
        -------
        Dict[str, Any]
            Agent information
        """
        response = await self._make_request("GET", f"/agents/{agent_id}")
        return response.get("agent", {})
    
    async def update_agent(
        self,
        agent_id: str,
        name: Optional[str] = None,
        model: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Update an agent on the MCP server.
        
        Parameters
        ----------
        agent_id : str
            Agent ID
        name : str, optional
            Agent name, by default None
        model : Dict[str, Any], optional
            Agent model (generative model), by default None
        
        Returns
        -------
        Dict[str, Any]
            Updated agent information
        """
        data = {}
        if name is not None:
            data["name"] = name
        if model is not None:
            data["model"] = model
        
        response = await self._make_request("PUT", f"/agents/{agent_id}", data=data)
        return response.get("agent", {})
    
    async def delete_agent(self, agent_id: str) -> bool:
        """Delete an agent from the MCP server.
        
        Parameters
        ----------
        agent_id : str
            Agent ID
        
        Returns
        -------
        bool
            True if successful, False otherwise
        """
        response = await self._make_request("DELETE", f"/agents/{agent_id}")
        return response.get("success", False)
    
    async def create_environment(
        self,
        name: str,
        type: str,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create a new environment on the MCP server.
        
        Parameters
        ----------
        name : str
            Environment name
        type : str
            Environment type
        params : Dict[str, Any]
            Environment parameters
        
        Returns
        -------
        Dict[str, Any]
            Environment information
        """
        data = {
            "name": name,
            "type": type,
            "params": params
        }
        response = await self._make_request("POST", "/environments", data=data)
        return response.get("environment", {})
    
    async def get_environments(self) -> List[Dict[str, Any]]:
        """Get the list of environments from the MCP server.
        
        Returns
        -------
        List[Dict[str, Any]]
            List of environments
        """
        response = await self._make_request("GET", "/environments")
        return response.get("environments", [])
    
    async def get_environment(self, env_id: str) -> Dict[str, Any]:
        """Get information about a specific environment.
        
        Parameters
        ----------
        env_id : str
            Environment ID
        
        Returns
        -------
        Dict[str, Any]
            Environment information
        """
        response = await self._make_request("GET", f"/environments/{env_id}")
        return response.get("environment", {})
    
    async def update_environment(
        self,
        env_id: str,
        name: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Update an environment on the MCP server.
        
        Parameters
        ----------
        env_id : str
            Environment ID
        name : str, optional
            Environment name, by default None
        params : Dict[str, Any], optional
            Environment parameters, by default None
        
        Returns
        -------
        Dict[str, Any]
            Updated environment information
        """
        data = {}
        if name is not None:
            data["name"] = name
        if params is not None:
            data["params"] = params
        
        response = await self._make_request("PUT", f"/environments/{env_id}", data=data)
        return response.get("environment", {})
    
    async def delete_environment(self, env_id: str) -> bool:
        """Delete an environment from the MCP server.
        
        Parameters
        ----------
        env_id : str
            Environment ID
        
        Returns
        -------
        bool
            True if successful, False otherwise
        """
        response = await self._make_request("DELETE", f"/environments/{env_id}")
        return response.get("success", False)
    
    async def create_session(
        self,
        agent_id: str,
        env_id: str,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create a new session on the MCP server.
        
        Parameters
        ----------
        agent_id : str
            Agent ID
        env_id : str
            Environment ID
        params : Dict[str, Any], optional
            Session parameters, by default None
        
        Returns
        -------
        Dict[str, Any]
            Session information
        """
        data = {
            "agent_id": agent_id,
            "env_id": env_id
        }
        if params is not None:
            for key, value in params.items():
                data[key] = value
        
        response = await self._make_request("POST", "/sessions", data=data)
        return response
    
    async def get_sessions(self) -> List[Dict[str, Any]]:
        """Get the list of sessions from the MCP server.
        
        Returns
        -------
        List[Dict[str, Any]]
            List of sessions
        """
        response = await self._make_request("GET", "/sessions")
        return response.get("sessions", [])
    
    async def get_session(self, session_id: str) -> Dict[str, Any]:
        """Get information about a specific session.
        
        Parameters
        ----------
        session_id : str
            Session ID
        
        Returns
        -------
        Dict[str, Any]
            Session information
        """
        response = await self._make_request("GET", f"/sessions/{session_id}")
        return response.get("session", {})
    
    async def run_session(
        self,
        session_id: str,
        steps: int = 1,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Run a session on the MCP server.
        
        Parameters
        ----------
        session_id : str
            Session ID
        steps : int, optional
            Number of steps to run, by default 1
        params : Dict[str, Any], optional
            Run parameters, by default None
        
        Returns
        -------
        Dict[str, Any]
            Session information after running
        """
        data = {"steps": steps}
        if params is not None:
            data["params"] = params
        
        response = await self._make_request("POST", f"/sessions/{session_id}/run", data=data)
        return response.get("session", {})
    
    async def reset_session(
        self,
        session_id: str,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Reset a session on the MCP server.
        
        Parameters
        ----------
        session_id : str
            Session ID
        params : Dict[str, Any], optional
            Reset parameters, by default None
        
        Returns
        -------
        Dict[str, Any]
            Session information after reset
        """
        data = {}
        if params is not None:
            data["params"] = params
        
        response = await self._make_request("POST", f"/sessions/{session_id}/reset", data=data)
        return response.get("session", {})
    
    async def delete_session(self, session_id: str) -> bool:
        """Delete a session from the MCP server.
        
        Parameters
        ----------
        session_id : str
            Session ID
        
        Returns
        -------
        bool
            True if successful, False otherwise
        """
        response = await self._make_request("DELETE", f"/sessions/{session_id}")
        return response.get("success", False)
    
    async def run_simulation(
        self,
        agent_id: str,
        env_id: str,
        steps: int,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Run a simulation on the MCP server.
        
        Parameters
        ----------
        agent_id : str
            Agent ID
        env_id : str
            Environment ID
        steps : int
            Number of steps to run
        params : Dict[str, Any], optional
            Simulation parameters, by default None
        
        Returns
        -------
        Dict[str, Any]
            Simulation results
        """
        # Create a session if it doesn't exist
        try:
            session_data = await self.create_session(agent_id, env_id, params)
        except Exception as e:
            return {"error": f"Error creating session: {str(e)}"}
            
        session_id = session_data.get("id")
            
        if not session_id:
            # Try to generate a predictable session ID
            session_id = f"{agent_id}_{env_id}_session"
            
        # Run the simulation using the tool directly
        run_params = {
            "agent_id": agent_id,
            "env_id": env_id,
            "steps": steps
        }
        if params is not None:
            for key, value in params.items():
                if key not in run_params:
                    run_params[key] = value
                    
        result = await self.call_tool("run_simulation", run_params)
        
        # If we got a result with a result field, use that
        if "result" in result:
            result = result["result"]
            
        # Add the session ID if it's not present
        if "id" not in result:
            result["id"] = session_id
            
        return result
    
    async def visualize_agent(
        self,
        agent_id: str,
        format: str = "png",
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Visualize an agent on the MCP server.
        
        Parameters
        ----------
        agent_id : str
            Agent ID
        format : str, optional
            Visualization format, by default "png"
        params : Dict[str, Any], optional
            Visualization parameters, by default None
        
        Returns
        -------
        Dict[str, Any]
            Visualization information
        """
        query_params = {"format": format}
        if params is not None:
            data = {"params": params}
        else:
            data = None
        
        response = await self._make_request(
            "GET",
            f"/agents/{agent_id}/visualize",
            data=data,
            params=query_params
        )
        return response.get("visualization", {})
    
    async def visualize_session(
        self,
        session_id: str,
        format: str = "png",
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Visualize a session on the MCP server.
        
        Parameters
        ----------
        session_id : str
            Session ID
        format : str, optional
            Visualization format, by default "png"
        params : Dict[str, Any], optional
            Visualization parameters, by default None
        
        Returns
        -------
        Dict[str, Any]
            Visualization information
        """
        query_params = {"format": format}
        if params is not None:
            data = {"params": params}
        else:
            data = None
        
        response = await self._make_request(
            "GET",
            f"/sessions/{session_id}/visualize",
            data=data,
            params=query_params
        )
        return response.get("visualization", {}) 