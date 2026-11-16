"""Single-process REST adapter over the explicit production tool contract."""
import json
import logging
import os
import uuid

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException

from ..api import InvalidArguments, ToolRegistry, UnknownTool
from ..outputs import OutputPolicy
from ..utils import PyMDPInterface


def create_app(interface=None, output_root=None, instance_id=None):
    root = output_root if output_root is not None else os.environ.get("MCP_OUTPUT_DIR", "output")
    if interface is None:
        limits = json.loads(os.environ["MCP_LIMITS"]) if "MCP_LIMITS" in os.environ else None
        interface = PyMDPInterface(limits=limits, output_root=root)
    else:
        interface.output_policy = OutputPolicy(root)
    registry = ToolRegistry(interface)
    app = FastAPI(title="PyMDP REST tools", version="1.0.3")
    app.state.interface, app.state.registry = interface, registry
    app.state.instance_id = instance_id if instance_id is not None else os.environ.get("MCP_INSTANCE_ID")
    origins = [v.strip() for v in os.environ.get(
        "CORS_ORIGINS", "http://localhost:8080,http://127.0.0.1:8080").split(",") if v.strip()]
    # Browser CORS configuration is not authentication.
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=False,
                       allow_methods=["GET", "POST"], allow_headers=["Content-Type", "Authorization"])

    @app.exception_handler(HTTPException)
    async def http_error(request, error):
        detail = error.detail if isinstance(error.detail, dict) else {"code": "http_error", "message": str(error.detail)}
        return JSONResponse({"error": detail}, status_code=error.status_code, headers=error.headers)

    @app.exception_handler(Exception)
    async def unexpected_error(request, error):
        logging.getLogger(__name__).error("Unhandled HTTP request failure",
                                         exc_info=(type(error), error, error.__traceback__))
        return JSONResponse({"error": {"code": "internal_error", "message": "internal server error"}},
                            status_code=500)

    async def body(request):
        chunks, size = [], 0
        async for chunk in request.stream():
            size += len(chunk)
            if size > 1024 * 1024:
                raise HTTPException(413, "request exceeds byte budget")
            chunks.append(chunk)
        try:
            def invalid_constant(value):
                raise ValueError(value)
            value = json.loads(b"".join(chunks), parse_constant=invalid_constant)
        except (ValueError, UnicodeError):
            raise HTTPException(400, "invalid JSON") from None
        if not isinstance(value, dict):
            raise HTTPException(422, "request body must be an object")
        return value

    def domain(result):
        if isinstance(result, dict) and "error" in result:
            code = result.get("error_code", "operation_failed")
            status = {"not_found": 404, "invalid_argument": 422, "conflict": 409, "gone": 410}.get(code, 500)
            raise HTTPException(status, {"code": code, "message": result["error"]})
        return result

    def invoke(name, arguments):
        try:
            result = registry.call(name, arguments)
        except UnknownTool:
            raise HTTPException(404, {"code": "unknown_tool", "message": name}) from None
        except InvalidArguments as error:
            raise HTTPException(422, {"code": "invalid_argument", "message": str(error)}) from None
        return domain(result)

    def output_format(request, format):
        if set(request.query_params) - {"format"}:
            raise HTTPException(422, "unsupported query option")
        if request.headers.get("content-length", "0") != "0":
            raise HTTPException(422, "GET bodies are not supported")
        if format not in ("json", "png", "svg"):
            raise HTTPException(422, "format must be json, png or svg")
        return format

    def image(result, format):
        result = domain(result)
        try:
            data = interface.output_policy.read(result["figure_path"])
        except FileNotFoundError:
            raise HTTPException(404, "generated image is unavailable") from None
        return Response(data, media_type="image/png" if format == "png" else "image/svg+xml")

    @app.get("/ping")
    async def ping():
        return {"status": "ok", "instance_id": app.state.instance_id, "pid": os.getpid()}

    @app.get("/tools")
    async def tools():
        return {"tools": registry.descriptors()}

    @app.post("/tools/{tool_id}")
    async def call_tool(tool_id: str, request: Request):
        value = await body(request)
        if set(value) != {"params"}:
            raise HTTPException(422, "tool request requires exactly one params object")
        return {"result": invoke(tool_id, value["params"])}

    @app.post("/environments")
    async def create_environment(request: Request):
        value = await body(request)
        if value.pop("type", "grid_world") != "grid_world":
            raise HTTPException(422, "HTTP environment creation supports grid_world only")
        return invoke("create_grid_world_env", value)

    @app.get("/environments")
    async def environments():
        return invoke("get_all_environments", {})

    @app.get("/environments/{env_id}")
    async def environment(env_id: str):
        return invoke("get_environment", {"env_id": env_id})

    @app.get("/agents")
    async def agents():
        return invoke("get_all_agents", {})

    @app.get("/agents/{agent_id}")
    async def agent(agent_id: str):
        return invoke("get_agent", {"agent_id": agent_id})

    @app.get("/simulations")
    async def simulations():
        return invoke("get_all_simulations", {})

    @app.post("/sessions")
    async def create_session(request: Request):
        value = await body(request)
        return invoke("create_session", value)

    @app.post("/sessions/{session_id}/run")
    async def run_session(session_id: str, request: Request):
        value = await body(request)
        if "session_id" in value:
            raise HTTPException(422, "session identity belongs in the route")
        return invoke("run_session", {"session_id": session_id, **value})

    @app.get("/sessions")
    async def sessions():
        return invoke("get_all_sessions", {})

    @app.get("/agents/{agent_id}/visualize")
    async def visualize_agent(agent_id: str, request: Request, format: str = "png"):
        output_format(request, format)
        if format == "json":
            return invoke("get_agent", {"agent_id": agent_id})
        return image(invoke("visualize_agent_model", {
            "agent_id": agent_id, "output_file": f"agent_{uuid.uuid4().hex}.{format}"}), format)

    @app.get("/simulations/{simulation_id}/visualize")
    async def visualize_run(simulation_id: str, request: Request, format: str = "png"):
        output_format(request, format)
        if format == "json":
            return invoke("get_simulation", {"simulation_id": simulation_id})
        return image(invoke("visualize_simulation", {
            "simulation_id": simulation_id, "output_file": f"simulation_{uuid.uuid4().hex}.{format}"}), format)

    @app.get("/sessions/{session_id}/visualize")
    async def visualize_session(session_id: str, request: Request, format: str = "png"):
        from ..visualization.simulation import visualize_simulation
        output_format(request, format)
        history = invoke("get_session_history", {"session_id": session_id})
        if format == "json":
            return history
        with registry.lock:
            try:
                result = visualize_simulation(history, f"session_{uuid.uuid4().hex}.{format}",
                                               policy=interface.output_policy)
            except ValueError as error:
                raise HTTPException(422, str(error)) from None
        return image(result, format)

    @app.get("/analyze/free_energy/{session_id}")
    async def analyze_energy(session_id: str, request: Request, format: str = "json"):
        from ..visualization.free_energy import measured_records
        output_format(request, format)
        if format == "json":
            history = invoke("get_session_history", {"session_id": session_id})
            try:
                return {"session_id": session_id, "records": measured_records(history)}
            except ValueError as error:
                raise HTTPException(422, str(error)) from None
        return image(invoke("analyze_free_energy", {
            "session_id": session_id, "output_file": f"vfe_{uuid.uuid4().hex}.{format}"}), format)

    @app.get("/analyze/beliefs/{session_id}")
    async def analyze_beliefs(session_id: str, request: Request, format: str = "json"):
        from ..belief_dynamics import calculate_belief_statistics
        output_format(request, format)
        if format == "json":
            history = invoke("get_session_history", {"session_id": session_id})
            try:
                return calculate_belief_statistics([t["beliefs"] for t in history["timesteps"]])
            except ValueError as error:
                raise HTTPException(422, str(error)) from None
        return image(invoke("plot_belief_dynamics", {
            "session_id": session_id, "output_file": f"beliefs_{uuid.uuid4().hex}.{format}"}), format)

    return app




if __name__ == "__main__":
    import uvicorn
    uvicorn.run(create_app(), host=os.environ.get("HOST", "127.0.0.1"),
                port=int(os.environ.get("PORT", "8080")), workers=1)
