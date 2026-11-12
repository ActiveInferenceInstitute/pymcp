#!/usr/bin/env python3
"""Pinned GNN conformance check — the shared reader, run in the pinned parser.

This script is the launch shim for the `gnn conformance (pinned)` CI job:
`.github/workflows/ci.yml` runs `./run.sh gnn`, which runs this file. Nothing
about the check lives in YAML.

What it does, in order:

1. Read the revision `.github/gnn-pin.json` records, because the pinned parser
   has to be materialized before the reader that validates the pin can run.
   Only that one key is read here: every other pin question (schema, declared
   repository, tag, commit shape, specification list, pointer) belongs to
   `gnn.importers`, which is what step 3 runs.
2. Install that revision into an isolated `.gnn-venv/` (never the project
   environment, so `uv run --frozen` cannot prune it) and reuse it only when the
   installed distribution's recorded git commit equals the revision from step 1,
   so a same-version pin change cannot silently reuse older code.
3. Run the shared reader from that environment:

       python -m gnn.importers check --repo . --require-pointer \\
           --workflow .github/workflows/ci.yml --job gnn-conformance

   It owns pin discovery and shape, pin and pointer validation, and the syntax
   validation of every `*.gnn.md` specification with the pinned parser.
4. Type-check the specifications the reader discovers, with the pinned
   `GNNTypeChecker`: conformance for this member includes the semantic verdict
   the reader does not produce. The type check accepts one value, the boolean
   `True`. `GNNTypeChecker` returns `True` for a clean model and `2` for its
   warning sentinel, so equality tests are unsound here — `not 2` is `False`
   and `False == 0` — and the verdict is an identity check. A corpus with no
   specifications is refused rather than skipped, because `validate_gnn_files`
   reports "nothing to check" as that same `2` sentinel.

Exit status: 0 when the reader and the type check both agree; 1 otherwise, with
the reason on stderr and a JSON verdict on stdout.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PIN_PATH = Path(".github/gnn-pin.json")
PARSER_ENV = ".gnn-venv"
PARSER_DEPS = ("defusedxml", "psutil", "pyyaml")

#: The conformance job this repository's pointer must name, from
#: `.github/workflows/ci.yml` and the estate registry.
WORKFLOW = ".github/workflows/ci.yml"
JOB_KEY = "gnn-conformance"

#: The shared reader, called against this checkout.
READER_ARGS = (
    "-m",
    "gnn.importers",
    "check",
    "--repo",
    ".",
    "--require-pointer",
    "--workflow",
    WORKFLOW,
    "--job",
    JOB_KEY,
)


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


def pinned_revision(repo_root: Path) -> str:
    """The upstream revision recorded in `.github/gnn-pin.json`.

    This is the environment's input, not a second reader: the installed parser
    cannot validate a pin before it exists, so one key is taken from the file
    and the reader judges everything else about it afterwards.
    """
    pin_file = repo_root / PIN_PATH
    try:
        payload = json.loads(pin_file.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError) as exc:
        raise ConformanceError(f"cannot read {PIN_PATH}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ConformanceError(f"{PIN_PATH} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ConformanceError(f"{PIN_PATH} must be a JSON object")

    revision = payload.get("commit") or payload.get("revision")
    if not isinstance(revision, str) or not revision.strip():
        raise ConformanceError(f"{PIN_PATH} records no commit/revision")
    return revision.strip()


def parser_spec(revision: str) -> str:
    return (
        "generalized-notation-notation @ "
        "git+https://github.com/ActiveInferenceInstitute/"
        f"Generalized_Notation_Notation.git@{revision}"
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


def reuse_ok(record: dict | None, revision: str) -> bool:
    """Reuse the parser env only for this exact revision."""
    if not record:
        return False
    return record.get("commit") == revision


def ensure_parser(repo_root: Path, revision: str) -> Path:
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
    if reuse_ok(record, revision):
        log(
            f"reusing {PARSER_ENV}: gnn {record['version']} "
            f"at {record['commit']} matches the pin"
        )
        return python

    if record:
        log(
            f"reinstalling: {PARSER_ENV} holds gnn {record.get('version')} at "
            f"{record.get('commit')} but the pin is {revision}"
        )

    log(f"installing the pinned parser at {revision}")
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
                parser_spec(revision),
            ],
            cwd=repo_root,
        )
    except FileNotFoundError:
        raise ConformanceError("uv is not on PATH; install uv first") from None
    except subprocess.CalledProcessError as exc:
        raise ConformanceError(
            f"installing the pinned parser at {revision} failed "
            f"(exit {exc.returncode}); the check needs network access to that "
            "GitHub revision"
        ) from exc

    installed = installed_record(python)
    if not reuse_ok(installed, revision):
        raise ConformanceError(
            f"installed parser does not match the pin: got {installed!r}, "
            f"expected {revision}"
        )
    return python


def check_reader(python: Path) -> None:
    """Run the shared reader in the pinned environment, in place."""
    completed = subprocess.run(
        [str(python), *READER_ARGS],
        cwd=REPO_ROOT,
        check=False,
    )
    if completed.returncode != 0:
        raise ConformanceError(
            f"the shared reader rejected this checkout (exit {completed.returncode})"
        )


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


def require_specs(specs: list[Path]) -> None:
    """Raise unless there is at least one specification to type-check.

    `validate_gnn_files` reports an empty directory as its `2` warning sentinel
    ("nothing to do"), so skipping the call on an empty corpus would print
    `"result": "verified"` without validating anything. A conformance pass has
    to name the models it checked, so zero specifications is a failure here.
    """
    if specs:
        return
    raise ConformanceError(
        "no GNN specifications were discovered (*.gnn.md), so nothing was "
        "type-checked; the checker reports an empty corpus as its 2 sentinel, "
        "not as clean success"
    )


def typecheck_specs(repo_root: Path) -> dict:
    """Type-check the specifications the shared reader discovers.

    Runs inside the pinned parser environment, so `discover_specs` and the
    type checker come from the pinned revision and the file set is the reader's,
    not a second glob. An empty corpus is refused: a verified verdict means at
    least one specification passed the identity check on the boolean `True`.
    """
    from gnn import __version__
    from gnn.importers import discover_specs
    from gnn.type_checker import GNNTypeChecker

    specs = discover_specs(repo_root)
    require_specs(specs)
    with tempfile.TemporaryDirectory(prefix="gnn-conformance-") as output_dir:
        model_dir = Path(output_dir) / "models"
        model_dir.mkdir()
        for index, spec in enumerate(specs):
            shutil.copy2(spec, model_dir / f"{index:03d}_{spec.name}")
        typecheck_verdict(
            GNNTypeChecker().validate_gnn_files(model_dir, Path(output_dir))
        )

    return {
        "gnn-version": __version__,
        "models": len(specs),
        "files": [str(spec.relative_to(repo_root)) for spec in specs],
    }


def main(argv: list[str]) -> int:
    repo_root = REPO_ROOT
    os.chdir(repo_root)

    if "--typecheck" in argv:
        # Second phase: this interpreter is the isolated parser environment.
        report_path = Path(argv[argv.index("--typecheck") + 1])
        report_path.write_text(
            json.dumps(typecheck_specs(repo_root), sort_keys=True),
            encoding="utf-8",
        )
        return 0

    revision = pinned_revision(repo_root)
    python = ensure_parser(repo_root, revision)
    check_reader(python)

    with tempfile.TemporaryDirectory(prefix="gnn-conformance-report-") as report_dir:
        report_path = Path(report_dir) / "typecheck.json"
        phase = [
            str(python),
            str(Path(__file__).resolve()),
            "--typecheck",
            str(report_path),
        ]
        completed = subprocess.run(
            phase,
            text=True,
            capture_output=True,
            cwd=repo_root,
            check=False,
        )
        sys.stderr.write(completed.stderr)
        if completed.returncode != 0:
            return completed.returncode
        verdict = json.loads(report_path.read_text(encoding="utf-8"))

    report = {**verdict, "pin-revision": revision, "result": "verified"}
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
