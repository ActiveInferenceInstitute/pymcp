#!/usr/bin/env python3
"""Canonical single-process REST launcher; loopback unless explicitly configured."""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main(argv=None):
    import uvicorn

    from mcp.server.app import create_app
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8080")))
    parser.add_argument("--output-dir", default=os.environ.get("MCP_OUTPUT_DIR", "output"))
    options = parser.parse_args(argv)
    if not 1 <= options.port <= 65535:
        parser.error("port is outside 1..65535")
    app = create_app(output_root=options.output_dir)
    uvicorn.run(app, host=options.host, port=options.port, workers=1, reload=False)


if __name__ == "__main__":
    main()
