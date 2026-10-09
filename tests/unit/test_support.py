"""Regression checks for test-only filesystem isolation."""

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tests import support


class TestSupportTests(unittest.TestCase):
    def test_optional_fixtures_skip_before_constructing_any_path(self):
        with patch.dict(os.environ, {"GRAPHDECISIONBENCH_TEST_FIXTURE_DIR": ""}), \
                patch.object(support, "Path", side_effect=AssertionError("unexpected filesystem access")):
            with self.assertRaises(unittest.SkipTest):
                support.external_fixture(self, "graphtext")

    def test_external_fixture_uses_only_explicit_root(self):
        with tempfile.TemporaryDirectory() as directory:
            expected = Path(directory) / "graphtext"
            expected.mkdir()
            with patch.dict(os.environ, {"GRAPHDECISIONBENCH_TEST_FIXTURE_DIR": directory}):
                self.assertEqual(support.external_fixture(self, "graphtext"), expected.resolve())

    def test_configured_missing_fixture_fails(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(os.environ, {"GRAPHDECISIONBENCH_TEST_FIXTURE_DIR": directory}):
            with self.assertRaisesRegex(AssertionError, "fixture directory is missing"):
                support.external_fixture(self, "graphtext")

    def test_checkout_and_relative_fixture_roots_are_rejected(self):
        for root in (str(support.REPO), "relative-fixtures"):
            with self.subTest(root=root), \
                    patch.dict(os.environ, {"GRAPHDECISIONBENCH_TEST_FIXTURE_DIR": root}):
                with self.assertRaises(AssertionError):
                    support.external_fixture(self, "graphtext")

    def test_repo_workspace_registers_cleanup(self):
        owner = unittest.TestCase()
        try:
            workspace = support.repo_workspace(owner, "support-")
            self.assertEqual(workspace.parent, support.REPO / ".test-work")
            self.assertTrue(workspace.is_dir())
            (workspace / "fixture.txt").write_text("temporary")
        finally:
            owner.doCleanups()
        self.assertFalse(workspace.exists())