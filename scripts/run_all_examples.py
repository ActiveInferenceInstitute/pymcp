#!/usr/bin/env python3
"""Own one exact REST child and preserve every example outcome."""
import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
EXAMPLES = ("test_server.py", "mcp_pymdp_example.py", "mcp_gridworld_examples.py", "pymcp_examples_fixed.py")


class StartupError(RuntimeError):
    def __init__(self, message, exit_code=1):
        super().__init__(message)
        self.exit_code = (128 - exit_code if exit_code < 0 else exit_code) or 1


class StopRequested(BaseException):
    def __init__(self, signum):
        self.signum = signum


@contextmanager
def handled_signals():
    def stop(signum, frame):
        raise StopRequested(signum)
    previous = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def reap(process):
    if process.poll() is None:
        try:
            process.terminate()
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


class OwnedServer:
    def __init__(self, command, log_file, port, *, host="127.0.0.1", startup_timeout=30, env=None):
        if not math.isfinite(startup_timeout) or startup_timeout <= 0 or startup_timeout > 300:
            raise ValueError("startup_timeout must be in (0,300]")
        self.command, self.log_file, self.port, self.host = command, Path(log_file), port, host
        self.startup_timeout = startup_timeout
        self.env = dict(os.environ if env is None else env)
        self.instance_id = uuid.uuid4().hex
        self.env["MCP_INSTANCE_ID"] = self.instance_id
        self.process = self.stream = None

    def start(self):
        try:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            self.stream = self.log_file.open("wb")
            self.process = subprocess.Popen(self.command, cwd=ROOT, env=self.env,
                                            stdout=self.stream, stderr=subprocess.STDOUT)
            deadline = time.monotonic() + self.startup_timeout
            host = f"[{self.host}]" if ":" in self.host else self.host
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise StartupError(f"owned server exited {self.process.returncode}; see {self.log_file}",
                                       self.process.returncode)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    with urlopen(f"http://{host}:{self.port}/ping", timeout=min(1, remaining)) as response:
                        data = response.read(65537)
                        payload = json.loads(data) if len(data) <= 65536 else None
                        ready = (response.status == 200 and isinstance(payload, dict)
                                 and payload.get("status") == "ok"
                                 and payload.get("instance_id") == self.instance_id
                                 and payload.get("pid") == self.process.pid)
                    if ready and self.process.poll() is None:
                        return self.process.pid
                except (URLError, OSError, ValueError):
                    pass  # Bounded readiness polling, never a mutating tool retry.
                time.sleep(min(0.05, max(0, deadline - time.monotonic())))
            raise StartupError(f"owned server readiness timed out; see {self.log_file}")
        except BaseException:
            try:
                self.close()
            except Exception as error:
                print(f"Server cleanup failed: {error}", file=sys.stderr)
            raise

    def close(self):
        try:
            if self.process is not None:
                reap(self.process)
        finally:
            if self.stream is not None:
                self.stream.close()
                self.stream = None


def run_child(command, log_file, timeout):
    with Path(log_file).open("wb") as stream:
        process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        try:
            return process.wait(timeout=timeout)
        except BaseException:
            reap(process)
            raise


def run_batch(examples, output_dir, port=8090, *, startup_timeout=30, example_timeout=300, server_command=None):
    if (isinstance(examples, (str, bytes)) or not examples or len(set(map(str, examples))) != len(examples)
            or not math.isfinite(example_timeout) or not 0 < example_timeout <= 3600):
        raise ValueError("nonempty unique examples and bounded example_timeout required")
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    command = server_command if server_command is not None else [
        sys.executable, str(SCRIPTS / "start_fastapi_server.py"), "--host", "127.0.0.1",
        "--port", str(port), "--output-dir", str(output / "server")]
    server = OwnedServer(command, output / "server.log", port, startup_timeout=startup_timeout)
    results = {str(script): {"status": "not_run"} for script in examples}
    status = 0
    current_script = None
    with handled_signals():
        try:
            server.start()
            for index, script in enumerate(examples):
                current_script = str(script)
                results[current_script] = {"status": "running"}
                path = Path(script).resolve()
                destination = output / f"example_{index}"
                destination.mkdir()
                code = run_child([sys.executable, str(path), "--host", "127.0.0.1", "--port", str(port),
                                  "--output-dir", str(destination)], output / f"example_{index}.log", example_timeout)
                code = 128 - code if code < 0 else code
                results[str(script)] = {"status": "completed" if code == 0 else "failed", "exit_code": code}
                if code and status == 0:
                    status = code
                if code and path.name == "test_server.py":
                    break
        except StopRequested as error:
            status = status or 128 + error.signum
            if current_script is not None and results[current_script]["status"] == "running":
                results[current_script] = {"status": "failed", "signal": error.signum}
            results["interruption"] = {"status": "failed", "signal": error.signum}
        except Exception as error:
            status = status or (error.exit_code if isinstance(error, StartupError) else 1)
            if current_script is not None and results[current_script]["status"] == "running":
                results[current_script] = {"status": "failed", "error": str(error)}
            results["owner_error"] = {"status": "failed", "error": str(error)}
        finally:
            try:
                server.close()
            except Exception as error:
                status = status or 1
                results["cleanup"] = {"status": "failed", "error": str(error)}
    (output / "summary.json").write_text(json.dumps({"exit_code": status, "results": results}, indent=2))
    print(json.dumps({"exit_code": status, "output_dir": str(output), "results": results}))
    return status


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--output-dir", default=str(SCRIPTS / "outputs"))
    parser.add_argument("--example", action="append")
    parser.add_argument("--startup-timeout", type=float, default=30)
    parser.add_argument("--example-timeout", type=float, default=300)
    options = parser.parse_args(argv)
    examples = [str((SCRIPTS / path).resolve()) for path in (options.example or EXAMPLES)]
    output = Path(options.output_dir) / ("run_" + uuid.uuid4().hex)
    return run_batch(examples, output, options.port, startup_timeout=options.startup_timeout,
                     example_timeout=options.example_timeout)


if __name__ == "__main__":
    raise SystemExit(main())
