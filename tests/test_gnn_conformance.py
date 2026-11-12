"""Tests for the pinned GNN conformance gate (`./run.sh gnn`).

The gate delegates pin, pointer and syntax validation to the shared reader
(`python -m gnn.importers check`), which the pinned revision ships. These tests
cover only what this repository still owns:

* the bootstrap revision read, which has to exist before the pinned reader can
  run at all;
* parser reuse, which must be bound to the pinned *revision* and not merely to
  the parser version, or a same-version pin change silently reuses old code;
* the type-check verdict, where equality tests accept a failure (`False == 0`)
  and a warning sentinel (`1 == True`), so only an identity check is correct;
* the corpus guard, because the checker reports "no files found" as its `2`
  sentinel and skipping the call would verify nothing;
* the reader invocation, which is the contract's arm and must not silently lose
  its pointer requirements.

The script is imported by path because it lives in `scripts/`, not in the
installed package. No `gnn` import happens at module scope, so these tests run
in the ordinary project environment.
"""

import importlib.util
import io
import json
import re
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "gnn_conformance", REPO_ROOT / "scripts" / "gnn_conformance.py"
)
gnn_conformance = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gnn_conformance)

REVISION = "75c834bd8519dc7b5adc4d24f16bf393cfd1ecdf"


class TestTypecheckVerdict(unittest.TestCase):
    def test_clean_boolean_is_accepted(self):
        gnn_conformance.typecheck_verdict(True)

    def test_false_is_rejected_even_though_it_equals_zero(self):
        # The regression: `False == 0`, so `False in (True, 0)` is True.
        self.assertTrue(False in (True, 0))
        with self.assertRaises(SystemExit) as raised:
            gnn_conformance.typecheck_verdict(False)
        self.assertEqual(raised.exception.code, 1)

    def test_warning_sentinel_two_is_rejected(self):
        # `not 2` is False, so a truthiness gate would pass a warning.
        self.assertFalse(not 2)
        with self.assertRaises(SystemExit) as raised:
            gnn_conformance.typecheck_verdict(2)
        self.assertEqual(raised.exception.code, 1)

    def test_one_is_rejected_even_though_it_equals_true(self):
        self.assertTrue(1 == True)  # noqa: E712 - the trap under test
        with self.assertRaises(SystemExit):
            gnn_conformance.typecheck_verdict(1)

    def test_zero_and_non_booleans_are_rejected(self):
        for result in (0, None, "true", [], object()):
            with self.subTest(result=result):
                with self.assertRaises(SystemExit):
                    gnn_conformance.typecheck_verdict(result)


class TestTypecheckCorpus(unittest.TestCase):
    def test_a_corpus_with_specifications_is_accepted(self):
        gnn_conformance.require_specs([Path("models/gridworld_3x3.gnn.md")])

    def test_no_specifications_is_refused(self):
        # The gate used to skip the checker when discovery found nothing, so an
        # empty corpus printed `"result": "verified"` with `"models": 0` while
        # `validate_gnn_files` would have returned its `2` "nothing to check"
        # sentinel. Nothing validated must not read as a conformance pass.
        buffer = io.StringIO()
        with self.assertRaises(SystemExit) as raised, redirect_stderr(buffer):
            gnn_conformance.require_specs([])
        self.assertEqual(raised.exception.code, 1)
        self.assertIn("no GNN specifications", buffer.getvalue())


class TestParserReuse(unittest.TestCase):
    def test_reuse_requires_the_pinned_revision(self):
        record = {"version": "5.1.0", "commit": REVISION}
        self.assertTrue(gnn_conformance.reuse_ok(record, REVISION))

    def test_different_commit_is_not_reused(self):
        record = {"version": "5.1.0", "commit": "0" * 40}
        self.assertFalse(gnn_conformance.reuse_ok(record, REVISION))

    def test_missing_commit_is_not_reused(self):
        record = {"version": "5.1.0", "commit": None}
        self.assertFalse(gnn_conformance.reuse_ok(record, REVISION))

    def test_absent_install_is_not_reused(self):
        self.assertFalse(gnn_conformance.reuse_ok(None, REVISION))

    def test_record_without_the_parser_is_none(self):
        # The project interpreter does not install the pinned parser.
        self.assertIsNone(gnn_conformance.installed_record(Path(sys.executable)))


class TestBootstrapRevision(unittest.TestCase):
    def _pin(self, payload: object) -> Path:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        (root / ".github").mkdir()
        (root / ".github" / "gnn-pin.json").write_text(json.dumps(payload))
        return root

    def test_recorded_revision_is_returned(self):
        root = self._pin({"revision": REVISION})
        self.assertEqual(gnn_conformance.pinned_revision(root), REVISION)

    def test_commit_key_is_accepted(self):
        root = self._pin({"commit": REVISION})
        self.assertEqual(gnn_conformance.pinned_revision(root), REVISION)

    def test_pin_without_a_revision_is_rejected(self):
        for payload in ({}, {"tag": "v5.1.0"}, {"revision": ""}, {"revision": 5}):
            with self.subTest(payload=payload):
                root = self._pin(payload)
                with self.assertRaises(SystemExit) as raised:
                    gnn_conformance.pinned_revision(root)
                self.assertEqual(raised.exception.code, 1)

    def test_non_object_pin_is_rejected(self):
        for payload in (None, False, 1, "pin", ["revision"]):
            with self.subTest(payload=payload):
                root = self._pin(payload)
                with self.assertRaises(SystemExit) as raised:
                    gnn_conformance.pinned_revision(root)
                self.assertEqual(raised.exception.code, 1)

    def test_absent_pin_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(SystemExit) as raised:
                gnn_conformance.pinned_revision(Path(directory))
            self.assertEqual(raised.exception.code, 1)

    def test_repository_pin_records_a_commit(self):
        revision = gnn_conformance.pinned_revision(REPO_ROOT)
        self.assertRegex(revision, r"^[0-9a-f]{40}$")

    def test_parser_spec_targets_the_revision(self):
        spec = gnn_conformance.parser_spec(REVISION)
        self.assertIn(REVISION, spec)
        self.assertIn(
            "git+https://github.com/"
            "ActiveInferenceInstitute/Generalized_Notation_Notation.git",
            spec,
        )


class TestReaderInvocation(unittest.TestCase):
    def test_arm_runs_the_shared_reader(self):
        self.assertEqual(
            gnn_conformance.READER_ARGS[:4], ("-m", "gnn.importers", "check", "--repo")
        )

    def test_arm_requires_the_pointer_with_workflow_and_job(self):
        args = gnn_conformance.READER_ARGS
        self.assertIn("--require-pointer", args)
        self.assertEqual(args[args.index("--workflow") + 1], ".github/workflows/ci.yml")
        self.assertEqual(args[args.index("--job") + 1], "gnn-conformance")

    def test_job_key_matches_the_committed_workflow(self):
        workflow = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text()
        self.assertRegex(workflow, rf"(?m)^\s+{re.escape(gnn_conformance.JOB_KEY)}:$")


if __name__ == "__main__":
    unittest.main()
