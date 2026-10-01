"""Offline checks for the direct script entry point; never call the provider."""

import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import run_benchmark
from src.benchmark.config import load_config


SCRIPT = Path(__file__).resolve().parents[1] / "run_benchmark.py"
UTILITIES = (
    "audit_task_shortcuts", "extended_graph_suite", "paired_graph_ablation",
    "public_graph_suite", "tsp_abstraction_report", "tsp_abstraction_suite",
)


class ScriptEntryPointTests(unittest.TestCase):
    def test_legacy_module_delegates_to_root_cli(self):
        with patch("run_benchmark.main") as main:
            runpy.run_module("src.benchmark", run_name="__main__")
        main.assert_called_once_with()

    def test_import_does_not_start_evaluation(self):
        with patch("src.benchmark.runner.run_experiment") as run:
            runpy.run_path(str(SCRIPT), run_name="imported_entrypoint")
        run.assert_not_called()

    def test_root_cli_owns_config_output_override_run_and_summary(self):
        config = load_config(SCRIPT.parent / "configs" / "pilot.yaml")
        with patch.object(run_benchmark, "load_config", return_value=config) as load, \
                patch.object(run_benchmark, "run_experiment", return_value={"completed_calls": 3}) as run, \
                patch("builtins.print") as display:
            run_benchmark.main(["--config", "fixture.yaml", "--output", "custom-output", "--no-progress"])
        load.assert_called_once_with(Path("fixture.yaml"))
        executed = run.await_args.args[0]
        self.assertEqual(executed.run.output_dir, Path("custom-output").resolve())
        self.assertEqual(executed.data, config.data)
        self.assertNotEqual(executed.run.output_dir, config.run.output_dir)
        self.assertFalse(run.await_args.kwargs["progress"])
        display.assert_called_once_with('{\n  "completed_calls": 3\n}')

    def test_help_from_external_directory_without_key_or_pythonpath(self):
        env = {k: v for k, v in os.environ.items() if k not in {"TYPESAFE_API_KEY", "PYTHONPATH"}}
        with tempfile.TemporaryDirectory() as cwd:
            scripts = [SCRIPT, *(SCRIPT.parent / "src" / "utils" / f"{name}.py" for name in UTILITIES)]
            for script in scripts:
                with self.subTest(script=script.name):
                    result = subprocess.run(
                        [sys.executable, str(script), "--help"], cwd=cwd, env=env,
                        capture_output=True, text=True, timeout=15, check=False,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn(script.name, result.stdout)
                    if script == SCRIPT:
                        self.assertIn("--config", result.stdout)
                        self.assertIn("--no-progress", result.stdout)
                        self.assertNotIn("--execute", result.stdout)

    def test_module_help(self):
        for module in ("src.benchmark", *(f"src.utils.{name}" for name in UTILITIES)):
            with self.subTest(module=module):
                result = subprocess.run(
                    [sys.executable, "-m", module, "--help"], cwd=SCRIPT.parent,
                    capture_output=True, text=True, timeout=15, check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("usage:", result.stdout)


if __name__ == "__main__":
    unittest.main()