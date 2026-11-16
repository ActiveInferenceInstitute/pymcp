"""Source entry point for the bundled MCP transport; no resolver at startup."""
import argparse
import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
import json
import logging
import os

from mcp.api import ToolRegistry
from mcp.server.fastmcp import FastMCP
from mcp.utils import PyMDPInterface


@dataclass
class PyMDPContext:
    pymdp_interface: PyMDPInterface
    registry: ToolRegistry


@asynccontextmanager
async def pymdp_lifespan(server):
    interface = PyMDPInterface(
        limits=json.loads(os.environ["MCP_LIMITS"]) if "MCP_LIMITS" in os.environ else None,
        output_root=os.environ.get("MCP_OUTPUT_DIR", "output"))
    yield PyMDPContext(interface, ToolRegistry(interface))


def create_server(**settings):
    return FastMCP("pymdp-mcp", "Bounded PyMDP filtering tools", pymdp_lifespan, **settings)


mcp = create_server()


def main():
    parser = argparse.ArgumentParser(description="PyMDP Model Context Protocol server")
    parser.add_argument("--transport", choices=("stdio", "sse"), default=os.environ.get("TRANSPORT", "sse"))
    parser.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8050")))
    options = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    asyncio.run(create_server(**vars(options)).run())


if __name__ == "__main__":
    main()
