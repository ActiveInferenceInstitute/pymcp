"""Explicit discovery, strict native argument schemas and one domain dispatcher."""
import inspect
import threading
from typing import Any, Union, get_type_hints

from pydantic import ConfigDict, ValidationError, create_model

TOOL_NAMES = (
    "create_agent", "define_generative_model", "create_gridworld_agent",
    "validate_generative_model", "infer_states", "infer_policies", "sample_action",
    "commit_action", "reset_agent", "create_grid_world_env", "step_environment",
    "reset_environment", "run_simulation", "get_agent", "get_environment",
    "get_all_agents", "get_all_environments", "get_all_simulations", "get_all_functions",
    "get_simulation", "get_computation_logs", "create_session", "get_session",
    "get_session_history", "run_session", "get_all_sessions",
    "visualize_agent_model", "visualize_simulation", "visualize_belief_dynamics",
    "plot_belief_dynamics", "analyze_free_energy", "calculate_free_energy",
    "infer_states_from_observation",
)


class InvalidArguments(ValueError):
    pass


class UnknownTool(KeyError):
    pass


class Tool:
    def __init__(self, name, function, description=None):
        self.name, self.function = name, function
        self.description = description or inspect.getdoc(function) or name.replace("_", " ")
        hints = get_type_hints(function)
        fields = {}
        self.contextual = "ctx" in inspect.signature(function).parameters
        for key, parameter in inspect.signature(function).parameters.items():
            if key in ("ctx", "self"):
                continue
            if parameter.kind not in (parameter.POSITIONAL_OR_KEYWORD, parameter.KEYWORD_ONLY):
                raise ValueError("tools require explicit named parameters")
            annotation = hints.get(key, Any)
            default = ... if parameter.default is parameter.empty else parameter.default
            if default is None:
                annotation = Union[annotation, None]
            fields[key] = (annotation, default)
        self.arguments = create_model(name + "Arguments", __config__=ConfigDict(extra="forbid", strict=True), **fields)

    def descriptor(self):
        return {"name": self.name, "description": self.description,
                "inputSchema": self.arguments.model_json_schema()}

    def call(self, arguments, context=None):
        if not isinstance(arguments, dict):
            raise InvalidArguments("tool arguments must be an object")
        try:
            values = self.arguments.model_validate(arguments).model_dump(mode="python")
        except ValidationError as error:
            raise InvalidArguments(str(error)) from error
        if self.contextual:
            values["ctx"] = context
        return self.function(**values)


class ToolRegistry:
    def __init__(self, interface):
        self.interface = interface
        self.tools = {name: Tool(name, getattr(interface, name)) for name in TOOL_NAMES}
        self.lock = threading.RLock()

    def descriptors(self):
        return [tool.descriptor() for tool in self.tools.values()]

    def call(self, name, arguments):
        if name not in self.tools:
            raise UnknownTool(name)
        with self.lock:
            return self.tools[name].call(arguments)
