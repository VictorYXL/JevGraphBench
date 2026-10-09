"""Filesystem helpers; never implicitly consume local experiment outputs."""

import os
from pathlib import Path
import tempfile


REPO = Path(__file__).resolve().parents[1]


def repo_workspace(testcase, prefix):
    """Create a cleaned-up repository descendant for path-constrained runners."""
    parent = REPO / ".test-work"
    parent.mkdir(exist_ok=True)
    temporary = tempfile.TemporaryDirectory(prefix=prefix, dir=parent)
    testcase.addCleanup(temporary.cleanup)
    return Path(temporary.name)


def external_fixture(testcase, name):
    """Skip before filesystem access unless an external fixture root is opted in."""
    configured = os.environ.get("GRAPHDECISIONBENCH_TEST_FIXTURE_DIR")
    if not configured:
        testcase.skipTest("Set GRAPHDECISIONBENCH_TEST_FIXTURE_DIR for optional offline fixtures")
    root = Path(configured).expanduser()
    testcase.assertTrue(root.is_absolute(), "Fixture root must be an absolute external path")
    root = root.resolve()
    testcase.assertFalse(root.is_relative_to(REPO), "Fixture root must be outside the checkout")
    fixture = root / name
    testcase.assertTrue(fixture.is_dir(), f"Configured fixture directory is missing: {fixture}")
    return fixture