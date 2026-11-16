"""Independently owned HTTP client configuration; token-only client headers."""
import copy
import json
import math
from numbers import Real
import os
from pathlib import Path

from ..environments.common import integer

DEFAULT_CONFIG = {
    "server": {"host": "localhost", "port": 8080, "use_ssl": False},
    "auth": {"enabled": False, "token": None},
    "options": {"timeout": 30},
}


def get_config_path():
    configured = os.environ.get("MCP_CONFIG_PATH")
    if configured:
        return Path(configured)
    home = Path.home() / ".mcp" / "config.json"
    local = Path.cwd() / ".mcp.json"
    return home if home.exists() or not local.exists() else local


def merge_configs(base_config, override_config):
    if not isinstance(base_config, dict) or not isinstance(override_config, dict):
        raise ValueError("configuration must be an object")
    result = copy.deepcopy(base_config)
    for key, value in override_config.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def validate_config(config):
    if set(config) - set(DEFAULT_CONFIG):
        raise ValueError("unsupported client configuration section")
    for section, defaults in DEFAULT_CONFIG.items():
        if not isinstance(config.get(section), dict) or set(config[section]) - set(defaults):
            raise ValueError(f"unsupported {section} configuration option")
    server, auth, options = config["server"], config["auth"], config["options"]
    if not isinstance(server["host"], str) or not server["host"] or "/" in server["host"]:
        raise ValueError("host must be a hostname")
    integer(server["port"], "port", 1, 65535)
    if type(server["use_ssl"]) is not bool or type(auth["enabled"]) is not bool:
        raise ValueError("use_ssl and auth.enabled must be booleans")
    if auth["token"] is not None and not isinstance(auth["token"], str):
        raise ValueError("auth.token must be a string or null")
    timeout = options["timeout"]
    if isinstance(timeout, bool) or not isinstance(timeout, Real) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be positive and finite")
    return config


def load_config(config_path=None):
    path = Path(config_path) if config_path is not None else get_config_path()
    user = json.loads(path.read_text()) if path.exists() else {}
    return validate_config(merge_configs(DEFAULT_CONFIG, user))


def save_config(config, config_path=None):
    validated = validate_config(merge_configs(DEFAULT_CONFIG, config))
    path = Path(config_path) if config_path is not None else get_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(validated, allow_nan=False, indent=2))
    return True


def get_server_url(config=None):
    config = load_config() if config is None else validate_config(merge_configs(DEFAULT_CONFIG, config))
    server = config["server"]
    host = server["host"]
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    return f"{'https' if server['use_ssl'] else 'http'}://{host}:{server['port']}"
