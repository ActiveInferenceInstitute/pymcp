"""Synthetic local children exercise the real owner; no provider or fake success."""
import asyncio
from contextlib import contextmanager
import json
import os
import signal
import socket
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import pytest

from mcp.client import DirectMCPClient
from scripts import run_all_examples as owner
from scripts.example_support import evaluate_cases, grid_case
from scripts.pymcp_examples_fixed import free_energy_calculation
from tests.contract_support import ROOT


SERVER_SOURCE = '''import json, os, socket, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
Path(__file__).with_suffix(".pid").write_text(str(os.getpid()))
if "--large" in sys.argv:
    print("x" * 2_000_000, flush=True)
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        nonce = "wrong-owner" if "--wrong" in sys.argv else os.environ["MCP_INSTANCE_ID"]
        pid = os.getpid() + 1 if "--wrong-pid" in sys.argv else os.getpid()
        data = json.dumps({"status": "ok", "pid": pid, "instance_id": nonce}).encode()
        self.send_response(200 if self.path == "/ping" else 404)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)
server = HTTPServer(("127.0.0.1", 0), Handler, bind_and_activate=False)
server.socket.close()
server.socket = socket.socket(fileno=int(os.environ["SYNTHETIC_LISTENER_FD"]))
server.server_address = server.socket.getsockname()
server.server_name, server.server_port = "localhost", server.server_address[1]
server.serve_forever(poll_interval=0.01)
'''


@contextmanager
def local_server_fixture(tmp_path, monkeypatch):
    """Pass a reserved listener to the fixture child; no free-port race."""
    script = tmp_path / "synthetic_server.py"
    script.write_text(SERVER_SOURCE)
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    port = listener.getsockname()[1]
    original = subprocess.Popen
    spawned = []
    def spawn(command, *arguments, **options):
        if len(command) > 2 and str(script) in command:
            options["pass_fds"] = (listener.fileno(),)
            options["env"] = dict(options.get("env", os.environ), SYNTHETIC_LISTENER_FD=str(listener.fileno()))
        process = original(command, *arguments, **options)
        spawned.append(process)
        return process
    with monkeypatch.context() as scoped:
        scoped.setattr(owner.subprocess, "Popen", spawn)
        try:
            yield script, port, spawned
        finally:
            listener.close()
            for process in spawned:
                if process.returncode is None:
                    owner.reap(process)


def test_owned_readiness_uses_pid_nonce_and_large_output_cannot_fill_a_pipe(tmp_path, monkeypatch):
    with local_server_fixture(tmp_path, monkeypatch) as (script, port, spawned):
        service = owner.OwnedServer([sys.executable, "-B", str(script), "--large"], tmp_path / "server.log", port, startup_timeout=5)
        try:
            assert service.start() == service.process.pid
            assert service.process.returncode is None
            assert (tmp_path / "server.log").stat().st_size >= 2_000_000
        finally:
            service.close()
        assert service.process.returncode is not None and service.stream is None
        assert all(process.returncode is not None for process in spawned)


@pytest.mark.parametrize("wrong_marker", ["--wrong", "--wrong-pid"])
def test_wrong_instance_marker_is_never_accepted_and_owned_child_is_reaped(tmp_path, monkeypatch, wrong_marker):
    with local_server_fixture(tmp_path, monkeypatch) as (script, port, spawned):
        service = owner.OwnedServer([sys.executable, "-B", str(script), wrong_marker], tmp_path / "wrong.log", port, startup_timeout=0.5)
        with pytest.raises(owner.StartupError, match="timed out"):
            service.start()
        assert service.process.returncode is not None and service.stream is None
        assert all(process.returncode is not None for process in spawned)


def test_early_server_exit_preserves_exact_exit_code_and_future_cases_are_not_run(tmp_path):
    first, second = tmp_path / "first.py", tmp_path / "second.py"
    first.write_text("raise SystemExit(0)\n")
    second.write_text("raise SystemExit(0)\n")
    output = tmp_path / "early-exit"
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))  # Reserve, but do not listen or contact an unrelated service.
        status = owner.run_batch([first, second], output, reservation.getsockname()[1], startup_timeout=2,
                                 server_command=[sys.executable, "-B", "-c", "raise SystemExit(7)"])
    result = json.loads((output / "summary.json").read_text())
    assert status == result["exit_code"] == 7
    assert result["results"][str(first)]["status"] == result["results"][str(second)]["status"] == "not_run"


def test_critical_example_failure_preserves_seven_and_reaps_server(tmp_path, monkeypatch):
    probe, later = tmp_path / "test_server.py", tmp_path / "later.py"
    marker = tmp_path / "must-not-run"
    probe.write_text("raise SystemExit(7)\n")
    later.write_text(f"from pathlib import Path\nPath({str(marker)!r}).write_text('ran')\n")
    with local_server_fixture(tmp_path, monkeypatch) as (script, port, spawned):
        output = tmp_path / "critical"
        status = owner.run_batch([probe, later], output, port, startup_timeout=5,
                                 server_command=[sys.executable, "-B", str(script)])
        result = json.loads((output / "summary.json").read_text())
        assert status == 7 and result["exit_code"] == 7
        assert result["results"][str(probe)] == {"status": "failed", "exit_code": 7}
        assert result["results"][str(later)]["status"] == "not_run" and not marker.exists()
        assert all(process.returncode is not None for process in spawned)


def test_noncritical_failure_does_not_hide_later_case_outcomes(tmp_path, monkeypatch):
    failed, later = tmp_path / "failed.py", tmp_path / "later.py"
    failed.write_text("raise SystemExit(7)\n")
    later.write_text("raise SystemExit(0)\n")
    with local_server_fixture(tmp_path, monkeypatch) as (script, port, spawned):
        output = tmp_path / "independent"
        status = owner.run_batch([failed, later], output, port, startup_timeout=5,
                                 server_command=[sys.executable, "-B", str(script)])
        results = json.loads((output / "summary.json").read_text())["results"]
        assert status == 7
        assert results[str(failed)] == {"status": "failed", "exit_code": 7}
        assert results[str(later)] == {"status": "completed", "exit_code": 0}
        assert all(process.returncode is not None for process in spawned)


def test_timed_out_example_is_attempted_failed_not_not_run_and_reaped(tmp_path, monkeypatch):
    script = tmp_path / "timeout.py"
    script.write_text("import time\ntime.sleep(60)\n")
    with local_server_fixture(tmp_path, monkeypatch) as (server, port, spawned):
        output = tmp_path / "timeout-run"
        status = owner.run_batch([script], output, port, startup_timeout=5, example_timeout=0.2,
                                 server_command=[sys.executable, "-B", str(server)])
        results = json.loads((output / "summary.json").read_text())["results"]
        assert status != 0 and results[str(script)]["status"] == "failed"
        assert all(process.returncode is not None for process in spawned)


def test_interrupt_and_unexpected_owner_failures_reap_actual_server(tmp_path, monkeypatch):
    example = tmp_path / "example.py"
    example.write_text("raise SystemExit(0)\n")
    for index, error in enumerate((owner.StopRequested(signal.SIGTERM), RuntimeError("synthetic wait failure"))):
        with local_server_fixture(tmp_path, monkeypatch) as (server, port, spawned):
            output = tmp_path / f"failure-{index}"
            with patch.object(owner, "run_child", side_effect=error):
                status = owner.run_batch([example], output, port, startup_timeout=5,
                                         server_command=[sys.executable, "-B", str(server)])
            assert status == (128 + signal.SIGTERM if index == 0 else 1)
            assert json.loads((output / "summary.json").read_text())["results"][str(example)]["status"] == "failed"
            assert all(process.returncode is not None for process in spawned)


def test_run_child_timeout_reaps_before_returning_to_its_caller(tmp_path, monkeypatch):
    original, processes = subprocess.Popen, []
    def spawn(*arguments, **options):
        process = original(*arguments, **options)
        processes.append(process)
        return process
    monkeypatch.setattr(owner.subprocess, "Popen", spawn)
    try:
        with pytest.raises(subprocess.TimeoutExpired):
            owner.run_child([sys.executable, "-B", "-c", "import time; time.sleep(60)"], tmp_path / "child.log", 0.2)
        assert len(processes) == 1 and processes[0].returncode is not None
    finally:
        for process in processes:
            if process.returncode is None:
                owner.reap(process)


def test_shell_wrapper_exec_preserves_the_owned_python_exit_code(tmp_path):
    interpreter = tmp_path / "synthetic-python"
    interpreter.write_text("#!/bin/sh\nexit 7\n")
    interpreter.chmod(0o700)
    result = subprocess.run(["/bin/sh", str(ROOT / "scripts" / "run_with_server.sh")],
                            cwd=tmp_path, env=dict(os.environ, PYTHON=str(interpreter)), timeout=10, check=False)
    assert result.returncode == 7


def test_case_summary_keeps_failure_and_attempts_all_independent_cases(tmp_path):
    async def broken(client, output):
        raise RuntimeError("synthetic case failure")
    async def real_math(client, output):
        return await free_energy_calculation(client, output)
    status = asyncio.run(evaluate_cases(DirectMCPClient(), [("broken", broken), ("math", real_math)], tmp_path))
    results = json.loads((tmp_path / "summary.json").read_text())["cases"]
    assert status == 1 and results["broken"]["status"] == "failed"
    assert results["math"]["status"] == "completed"
    np.testing.assert_allclose(results["math"]["result"]["posterior"], [1 / 17, 9 / 17, 7 / 17])
    assert results["math"]["result"]["evidence"] == pytest.approx(17 / 30)
    assert (tmp_path / "math" / "belief_update.png").read_bytes().startswith(b"\x89PNG")


def test_real_grid_case_preserves_actual_ids_counts_rewards_and_measured_status(tmp_path):
    client = DirectMCPClient()
    result = asyncio.run(grid_case(client, tmp_path, "Synthetic", [1, 1], [], 2))
    assert result["num_steps"] == 2 and result["total_reward"] == 0
    assert result["vfe_statuses"] == ["measured", "measured"]
    history = client.interface.get_simulation(result["simulation_id"])
    assert history["agent_id"] == result["agent_id"] and history["env_id"] == result["env_id"]
    assert (tmp_path / "simulation.png").read_bytes().startswith(b"\x89PNG")
