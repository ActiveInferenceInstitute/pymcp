#!/usr/bin/env python3
"""Pinned GNN conformance check.

This script is the single source of truth for the `gnn conformance (pinned)`
CI job: `.github/workflows/ci.yml` runs `./run.sh gnn`, which runs this file.
Nothing about the check lives in YAML.

What it does, in order:

1. Read `.github/gnn-pin.json` and verify its schema (`gnn-importer-pin/v1`),
   repository, tag and revision.
2. Install the pinned parser into an isolated `.gnn-venv/` (never the project
   environment, so `uv run --frozen` cannot prune it) and reuse it only when the
   recorded version *and* git commit match the pin revision, so a same-version
   pin change cannot silently reuse older code.
3. Parse and syntax-validate every `*.gnn.md` sketch in the repository and
   type-check them. When the repository has no sketch, parse the pinned
   upstream `input/gnn_files/basics/static_perception.md` exemplar instead.

The type check accepts one value: the boolean `True`. `GNNTypeChecker` returns
`True` for a clean model and `2` for its warning sentinel, so equality tests are
unsound here — `not 2` is `False` and `False == 0` — and the verdict is
therefore an identity check.

Exit status: 0 when the pin, parser and sketches agree; 1 otherwise, with the
reason on stderr and a JSON verdict on stdout.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PIN_PATH = Path(".github/gnn-pin.json")
PARSER_ENV = ".gnn-venv"
PIN_SCHEMA = "gnn-importer-pin/v1"
PARSER_DEPS = ("defusedxml", "psutil", "pyyaml")
EXEMPLAR = "input/gnn_files/basics/static_perception.md"
SKIP_DIRS = frozenset({".git", ".venv", PARSER_ENV, "node_modules", "__pycache__"})


def log(message: str) -> None:
    print(message, file=sys.stderr)


class ConformanceError(SystemExit):
    """A conformance failure: message on stderr, exit status 1."""

    def __init__(self, message: str) -> None:
        log(f"gnn-conformance: {message}")
        super().__init__(1)


def run(args: list[str], *, cwd: Path) -> None:
    subprocess.run(
        args,
        check=True,
        text=True,
        cwd=cwd,
        stdout=subprocess.DEVNULL,
    )


def load_pin(repo_root: Path) -> dict:
    pin_file = repo_root / PIN_PATH
    try:
        pin = json.loads(pin_file.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConformanceError(f"cannot read {PIN_PATH}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ConformanceError(f"{PIN_PATH} is not valid JSON: {exc}") from exc

    required = ("schema", "repository", "revision", "tag")
    missing = [key for key in required if not pin.get(key)]
    if missing:
        raise ConformanceError(f"{PIN_PATH} is missing {', '.join(missing)}")
    if pin["schema"] != PIN_SCHEMA:
        raise ConformanceError(
            f"{PIN_PATH} schema is {pin['schema']!r}, expected {PIN_SCHEMA!r}"
        )
    if len(pin["revision"]) != 40 or any(
        c not in "0123456789abcdef" for c in pin["revision"]
    ):
        raise ConformanceError(
            f"{PIN_PATH} revision {pin['revision']!r} is not a 40-hex commit"
        )
    return pin


def parser_spec(pin: dict) -> str:
    return (
        f"generalized-notation-notation @ "
        f"git+https://github.com/{pin['repository']}.git@{pin['revision']}"
    )


def venv_python(repo_root: Path) -> Path:
    return repo_root / PARSER_ENV / "bin" / "python"


# Reads the installed parser's version and the git commit recorded by the
# installer, so reuse can be bound to the pin revision and not just the version.
PARSER_PROBE = """
import json
from importlib import metadata

try:
    dist = metadata.distribution("generalized-notation-notation")
except metadata.PackageNotFoundError:
    raise SystemExit(1)
raw = dist.read_text("direct_url.json")
info = json.loads(raw) if raw else {}
print(
    json.dumps(
        {
            "version": dist.version,
            "commit": (info.get("vcs_info") or {}).get("commit_id"),
        }
    )
)
"""


def installed_record(python: Path) -> dict | None:
    """The installed parser's `{version, commit}`, or `None` if absent."""
    probe = subprocess.run(
        [str(python), "-c", PARSER_PROBE],
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode != 0:
        return None
    try:
        return json.loads(probe.stdout)
    except json.JSONDecodeError:
        return None


def reuse_ok(record: dict | None, pin: dict) -> bool:
    """Reuse the parser env only for this exact pin: version *and* revision."""
    if not record:
        return False
    return record.get("version") == pin["tag"].removeprefix("v") and record.get(
        "commit"
    ) == pin["revision"]


def ensure_parser(repo_root: Path, pin: dict) -> Path:
    """Create/reuse the isolated parser environment and pin its parser."""
    python = venv_python(repo_root)

    if not python.exists():
        log(f"creating {PARSER_ENV} (python 3.12)")
        try:
            run(
                ["uv", "venv", "--python", "3.12", str(repo_root / PARSER_ENV)],
                cwd=repo_root,
            )
        except FileNotFoundError:
            raise ConformanceError("uv is not on PATH; install uv first") from None
        except subprocess.CalledProcessError as exc:
            raise ConformanceError(
                f"creating {PARSER_ENV} failed (exit {exc.returncode})"
            ) from exc

    record = installed_record(python)
    if reuse_ok(record, pin):
        log(
            f"reusing {PARSER_ENV}: gnn {record['version']} "
            f"at {record['commit']} matches the pin"
        )
        return python

    if record:
        log(
            f"reinstalling: {PARSER_ENV} holds gnn {record.get('version')} at "
            f"{record.get('commit')} but the pin is {pin['tag']} at "
            f"{pin['revision']}"
        )

    log(f"installing pinned parser {pin['repository']}@{pin['revision']}")
    try:
        run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(python),
                *PARSER_DEPS,
            ],
            cwd=repo_root,
        )
        run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(python),
                "--no-deps",
                parser_spec(pin),
            ],
            cwd=repo_root,
        )
    except FileNotFoundError:
        raise ConformanceError("uv is not on PATH; install uv first") from None
    except subprocess.CalledProcessError as exc:
        raise ConformanceError(
            f"installing the pinned parser failed (exit {exc.returncode}); "
            "the check needs network access to the pinned GitHub revision"
        ) from exc

    installed = installed_record(python)
    if not reuse_ok(installed, pin):
        raise ConformanceError(
            f"installed parser does not match the pin: got {installed!r}, "
            f"expected gnn {pin['tag']} at {pin['revision']}"
        )
    return python


def check_sketch(sketch: Path) -> None:
    from gnn.parsers.basic import parse_gnn_formal, validate_gnn_syntax

    parsed = parse_gnn_formal(sketch)
    parse_errors = getattr(parsed, "parse_errors", None)
    if parsed is None or parse_errors:
        raise ConformanceError(f"{sketch} did not parse cleanly: {parse_errors!r}")
    valid, syntax_errors = validate_gnn_syntax(sketch.read_text(encoding="utf-8"))
    if not valid:
        raise ConformanceError(f"{sketch} syntax errors: {syntax_errors!r}")


def typecheck_verdict(result: object) -> None:
    """Raise unless the checker returned its documented clean success.

    The clean contract is the boolean `True`. Equality-based checks are wrong
    here: `False == 0` and `1 == True`, so a membership test such as
    `result in (True, 0)` accepts a failure and a warning sentinel. `False`,
    `2` (warning sentinel), `1`, `0` and any non-boolean are failures.
    """
    if result is True:
        return
    raise ConformanceError(
        "GNN type checker did not return a clean result "
        f"(validate_gnn_files returned {result!r}, type "
        f"{type(result).__name__}); clean success is the boolean True"
    )


def typecheck(model_dir: Path, output_dir: Path) -> None:
    from gnn.type_checker import GNNTypeChecker

    typecheck_verdict(GNNTypeChecker().validate_gnn_files(model_dir, Path(output_dir)))


def rglob_sketches(root: Path) -> list[Path]:
    """Every `*.gnn.md` sketch, never one inside an environment or VCS dir."""
    sketches = []
    for path in root.rglob("*.gnn.md"):
        if SKIP_DIRS.intersection(path.relative_to(root).parts):
            continue
        sketches.append(path)
    return sketches


def check_repository_sketches(sketches: list[Path], output_dir: Path) -> dict:
    model_dir = Path(output_dir) / "models"
    model_dir.mkdir()
    for sketch in sketches:
        check_sketch(sketch)
    for index, sketch in enumerate(sketches):
        shutil.copy2(sketch, model_dir / f"{index:03d}_{sketch.name}")
    typecheck(model_dir, Path(output_dir))
    return {"models": len(sketches), "source": "repository"}


def check_pinned_exemplar(pin: dict, output_dir: Path) -> dict:
    url = (
        f"https://raw.githubusercontent.com/{pin['repository']}/"
        f"{pin['revision']}/{EXEMPLAR}"
    )
    exemplar = Path(output_dir) / Path(EXEMPLAR).name
    try:
        urllib.request.urlretrieve(url, exemplar)
    except OSError as exc:
        raise ConformanceError(f"cannot fetch pinned exemplar {url}: {exc}") from exc
    check_sketch(exemplar)
    model_dir = Path(output_dir) / "models"
    model_dir.mkdir()
    shutil.copy2(exemplar, model_dir / exemplar.name)
    typecheck(model_dir, Path(output_dir))
    return {"models": 1, "source": "pinned-exemplar"}


def verify(repo_root: Path, pin: dict) -> dict:
    from gnn import __version__

    expected = pin["tag"].removeprefix("v")
    if __version__ != expected:
        raise ConformanceError(
            f"installed GNN {__version__!r} does not match {pin['tag']!r}"
        )

    sketches = sorted(rglob_sketches(repo_root))
    with tempfile.TemporaryDirectory(prefix="gnn-conformance-") as output_dir:
        if sketches:
            extra = check_repository_sketches(sketches, output_dir)
        else:
            log("no repository sketch found; validating the pinned exemplar")
            extra = check_pinned_exemplar(pin, output_dir)

    return {
        "gnn-version": __version__,
        "pin-revision": pin["revision"],
        **extra,
        "result": "verified",
    }


def main(argv: list[str]) -> int:
    repo_root = REPO_ROOT
    os.chdir(repo_root)
    pin = load_pin(repo_root)

    if "--verify-only" in argv:
        # Second phase: this interpreter is the isolated parser environment.
        verdict = verify(repo_root, pin)
    else:
        python = ensure_parser(repo_root, pin)
        completed = subprocess.run(
            [str(python), str(Path(__file__).resolve()), "--verify-only"],
            text=True,
            capture_output=True,
            check=False,
        )
        sys.stderr.write(completed.stderr)
        if completed.returncode != 0:
            return completed.returncode
        verdict = json.loads(completed.stdout)

    print(json.dumps(verdict, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
