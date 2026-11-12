"""Tests for the pinned GNN conformance gate (`./run.sh gnn`).

These cover the two decisions that used to be unsound in the gate script:

* the type-check verdict, where equality tests accept a failure (`False == 0`)
  and a warning sentinel (`1 == True`), so only an identity check is correct;
* parser reuse, which must be bound to the pinned *revision* and not merely to
  the parser version, or a same-version pin change silently reuses old code.

The script is imported by path because it lives in `scripts/`, not in the
installed package. No `gnn` import happens at module scope, so these tests run
in the ordinary project environment.
"""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "gnn_conformance", REPO_ROOT / "scripts" / "gnn_conformance.py"
)
gnn_conformance = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gnn_conformance)

PIN = {
    "schema": "gnn-importer-pin/v1",
    "repository": "ActiveInferenceInstitute/Generalized_Notation_Notation",
    "revision": "8f293d9ba810e6eda5c09842d9c7ed9ffa5f92c1",
    "tag": "v5.0.3",
}


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


class TestParserReuse(unittest.TestCase):
    def test_reuse_requires_the_pinned_revision(self):
        record = {"version": "5.0.3", "commit": PIN["revision"]}
        self.assertTrue(gnn_conformance.reuse_ok(record, PIN))

    def test_same_version_different_commit_is_not_reused(self):
        record = {"version": "5.0.3", "commit": "0" * 40}
        self.assertFalse(gnn_conformance.reuse_ok(record, PIN))

    def test_different_version_is_not_reused(self):
        record = {"version": "5.0.2", "commit": PIN["revision"]}
        self.assertFalse(gnn_conformance.reuse_ok(record, PIN))

    def test_missing_commit_is_not_reused(self):
        record = {"version": "5.0.3", "commit": None}
        self.assertFalse(gnn_conformance.reuse_ok(record, PIN))

    def test_absent_install_is_not_reused(self):
        self.assertFalse(gnn_conformance.reuse_ok(None, PIN))

    def test_record_without_the_parser_is_none(self):
        # The project interpreter does not install the pinned parser.
        self.assertIsNone(gnn_conformance.installed_record(Path(sys.executable)))


class TestPinValidation(unittest.TestCase):
    def _load(self, pin: object) -> dict:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".github").mkdir()
            (root / ".github" / "gnn-pin.json").write_text(json.dumps(pin))
            return gnn_conformance.load_pin(root)

    def test_canonical_pin_loads(self):
        self.assertEqual(self._load(PIN), PIN)

    def test_non_object_pin_is_rejected(self):
        for pin in (None, False, 1, 1.5, "pin", [], ["schema"]):
            with self.subTest(pin=pin):
                with self.assertRaises(gnn_conformance.ConformanceError) as raised:
                    self._load(pin)
                self.assertEqual(raised.exception.code, 1)

    def test_non_string_fields_are_rejected(self):
        for field in PIN:
            for value in (None, False, 1, 1.5, [], {}, ["a"] * 40):
                with self.subTest(field=field, value=value):
                    with self.assertRaises(gnn_conformance.ConformanceError) as raised:
                        self._load({**PIN, field: value})
                    self.assertEqual(raised.exception.code, 1)

    def test_wrong_schema_is_rejected(self):
        with self.assertRaises(SystemExit) as raised:
            self._load({**PIN, "schema": "gnn-importer-pin/v0"})
        self.assertEqual(raised.exception.code, 1)

    def test_short_revision_is_rejected(self):
        with self.assertRaises(SystemExit):
            self._load({**PIN, "revision": "8f293d9b"})

    def test_missing_field_is_rejected(self):
        without_tag = {k: v for k, v in PIN.items() if k != "tag"}
        with self.assertRaises(SystemExit):
            self._load(without_tag)

    def test_repository_pin_is_valid(self):
        loaded = gnn_conformance.load_pin(REPO_ROOT)
        self.assertEqual(loaded["schema"], "gnn-importer-pin/v1")
        self.assertEqual(len(loaded["revision"]), 40)

    def test_parser_spec_targets_the_pin_revision(self):
        spec = gnn_conformance.parser_spec(PIN)
        self.assertIn(PIN["revision"], spec)
        self.assertIn(f"git+https://github.com/{PIN['repository']}.git", spec)


if __name__ == "__main__":
    unittest.main()
