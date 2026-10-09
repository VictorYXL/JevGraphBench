"""Unified CLI contracts: offline previews, permission gates and real parsers.

Only results.check and a fresh synthetic development plan/report/analyze execute.
Every other real backend stops at the parser boundary, before artifact access.
No environment, authorization, historical inputs or provider services are used.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager, redirect_stderr, redirect_stdout
import importlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
import warnings
from unittest.mock import AsyncMock, patch

import yaml

import run_benchmark
from src.benchmark import workflows
from tests.support import REPO, repo_workspace


CONFIGS = REPO / "configs"
CONFIG_NAMES = {
    "adjacency", "development", "distance_threshold", "graph_text_arxiv",
    "graph_text_prime", "optimization", "pilot", "proposals", "results", "structural",
}
LEGACY_COUNTS = {"adjacency": (240, 960), "distance_threshold": (240, 3840),
                 "pilot": (60, 240)}
DOWNLOAD_ACTIONS = {("graph_text", "freeze"), ("optimization", "prepare"),
                    ("optimization", "source-evidence")}
PATH_PARAMETERS = {
    "prior_pool", "data_dir", "historical_root", "authorization", "pilot_model_config",
    "test_model_config", "output", "primary_root", "deployment_root", "approval",
    "outer_plan", "activation_proof", "tests_log", "cohort", "model_config", "instances",
}
# Independent examples cover every registry option, not just each required subset.
# These paths are tokens only: none of these input artifacts is created or read.
ACTION_PARAMETERS = {
    "development": {
        "plan": {"design": "structural_holdout", "seed": 0, "samples_per_size": 1,
                 "node_counts": [8, 12], "prior_pool": "inputs/prior.jsonl",
                 "no_think_max_tokens": 64, "models": ["jev", "qwen4"]},
        "run": {"models": ["jev", "qwen2"]},
        "report": {"models": ["jev", "qwen9"]},
        "analyze": {"models": ["jev", "gpt54"]},
    },
    "structural": {
        "freeze": {"data_dir": "inputs/data", "historical_root": "inputs/panel",
                   "authorization": "inputs/authorization.json",
                   "pilot_model_config": "inputs/pilot.json",
                   "test_model_config": "inputs/test.json"},
        "validate": {"data_dir": "inputs/data"},
        "run": {"model": "jev_action", "split": "public-pilot", "concurrency": 16,
                "base_url": "https://example.invalid/v1"},
        "report": {"output": "reports/structural.json"},
        "check-lane": {"model": "kev", "require_complete": True,
                       "output": "reports/lane.json"},
        "combine": {"primary_root": "inputs/primary", "deployment_root": "inputs/deployment",
                    "approval": "inputs/approval.json", "outer_plan": "inputs/plan.json",
                    "activation_proof": "inputs/proof.json"},
        "combined-report": {"output": "reports/combined.json"},
        "scoring-recovery-check": {"primary_root": "inputs/primary",
                                   "approval": "inputs/approval.json",
                                   "output": "reports/recovery.json"},
    },
    "graph_text": {
        "freeze": {}, "verify": {}, "report": {"output": "reports/text.json"},
        "run": {"model": "org/model-v1", "dataset": "prime",
                "approval": "inputs/approval.json"},
    },
    "optimization": {
        "prepare": {}, "source-evidence": {}, "validate": {},
        "freeze": {"tests_log": "inputs/tests.log", "cohort": "inputs/cohort"},
        "run": {"models": ["org/model-a", "org/model-b"],
                "model_config": "inputs/models.json", "concurrency": 2, "scope": "pilot"},
        "report": {"scope": "main"},
    },
    "proposals": {
        "freeze": {"instances": "inputs/instances.json", "model_config": "inputs/models.json",
                   "models": ["kev", "qwen2"]},
        "run": {"model_config": "inputs/models.json", "models": ["decider", "laya"],
                "concurrency": 4},
        "report": {}, "monitor": {"interval": 9},
    },
    "results": {"check": {}, "render": {}},
}


class ParserBoundary(BaseException):
    """Stop after successful real parsing, even in backends catching Exception."""


class WorkflowEntrypointTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="workflow-entrypoint-")
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name).resolve()
        # Fail closed if a regression tries a network connection or a subprocess.
        for target in ("socket.create_connection", "socket.socket.connect",
                       "socket.socket.connect_ex", "subprocess.Popen"):
            guard = patch(target, side_effect=AssertionError("Offline tests forbid live actions"))
            guard.start()
            self.addCleanup(guard.stop)

    def real_configs(self):
        paths = sorted(CONFIGS.glob("*.yaml"))
        self.assertEqual({p.stem for p in paths}, CONFIG_NAMES)
        self.assertEqual(len(paths), 10)
        return paths

    def write_config(self, workflow, action, parameters=None, *, actions=None):
        path = self.base / "nested configs" / "workflow.yaml"
        path.parent.mkdir(exist_ok=True)
        document = {"schema_version": 1, "workflow": workflow, "default_action": action,
                    "actions": actions if actions is not None else {
                        action: {} if parameters is None else parameters}}
        if workflow != "results":
            document["root"] = "artifacts/../planned root"
        path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
        return path

    def invoke(self, path, *args):
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            result = run_benchmark.main(["--config", str(path), *args])
        self.assertIsNone(result)
        self.assertEqual(stderr.getvalue(), "")
        return stdout.getvalue()

    def reject(self, path, *args, message):
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr), \
                self.assertRaises(SystemExit) as caught:
            run_benchmark.main(["--config", str(path), *args])
        self.assertEqual(caught.exception.code, 2)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn(message, stderr.getvalue())

    @contextmanager
    def no_dispatch(self):
        with patch.object(workflows.WorkflowInvocation, "dispatch", autospec=True,
                          side_effect=AssertionError("Unexpected workflow dispatch")) as dispatch, \
                patch.object(run_benchmark, "run_experiment", new_callable=AsyncMock,
                             side_effect=AssertionError("Unexpected legacy inference")) as run:
            yield
            dispatch.assert_not_called()
            run.assert_not_called()

    def assert_compilation(self, path, raw, invocation):
        """Compare compiled values and argv with the source YAML independently."""
        workflow, action = raw["workflow"], invocation.action
        expected = {name: (path.parent / value).resolve() if name in PATH_PARAMETERS else value
                    for name, value in raw["actions"][action].items()}
        self.assertEqual(invocation.parameters, expected)
        root = None if workflow == "results" else (path.parent / raw["root"]).resolve()
        self.assertEqual(invocation.root, root)
        if workflow == "results":
            argv = ["--check"] if action == "check" else []
        elif workflow == "graph_text":
            argv = ["--root", str(root), action]
        else:
            argv = [action, "--root", str(root)]
        for name in workflows._WORKFLOWS[workflow].actions[action].options:
            if name not in expected:
                continue
            value = expected[name]
            option = "--" + name.replace("_", "-")
            if type(value) is bool:
                if value:
                    argv.append(option)
            else:
                argv.extend([option, *(str(item) for item in (
                    value if isinstance(value, list) else [value]))])
        self.assertEqual(invocation.argv, argv)
        self.assertIs(invocation.requires_inference, action == "run")
        self.assertIs(invocation.requires_downloads, (workflow, action) in DOWNLOAD_ACTIONS)

    def assert_real_parser(self, invocation):
        """Dispatch to main(argv), run its parser, then stop before its first action."""
        # Importing ordinary public source is allowed; never mock a backend parser.
        importlib.import_module(invocation.module)
        original = argparse.ArgumentParser.parse_args
        original_argv, argv_values = sys.argv, list(sys.argv)
        invocation_argv = list(invocation.argv)
        captured = []

        def parse_and_stop(parser, args=None, namespace=None):
            self.assertIsNotNone(args, "Backend main must pass its explicit argv")
            self.assertEqual(args, invocation_argv)
            self.assertIsNot(args, invocation.argv, "Dispatch must pass a detached argv list")
            parsed = original(parser, args, namespace)
            captured.append(parsed)
            raise ParserBoundary()

        # No file reads/writes are needed between dispatch and the parser boundary.
        # This also catches accidental pre-parser authorization/data inspection.
        with patch.object(argparse.ArgumentParser, "parse_args", parse_and_stop), \
                patch("builtins.open", side_effect=AssertionError("Backend file access before parsing")), \
                patch("io.open", side_effect=AssertionError("Backend file access before parsing")), \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()), \
                self.assertRaises(ParserBoundary):
            invocation.dispatch()
        self.assertIs(sys.argv, original_argv)
        self.assertEqual(sys.argv, argv_values)
        self.assertEqual(invocation.argv, invocation_argv)
        self.assertEqual(len(captured), 1)
        parsed = captured[0]
        if invocation.workflow == "results":
            self.assertIs(parsed.check, invocation.action == "check")
        else:
            self.assertEqual(parsed.command, invocation.action)
            self.assertIsInstance(parsed.root, Path)
            self.assertEqual(parsed.root, invocation.root)
        for name, expected in invocation.parameters.items():
            with self.subTest(parameter=name):
                actual = getattr(parsed, name)
                self.assertEqual(actual, expected)
                self.assertIs(type(actual), type(expected))
                if isinstance(expected, list):
                    self.assertEqual([type(item) for item in actual],
                                     [type(item) for item in expected])

    def test_all_ten_real_config_default_previews_never_dispatch(self):
        with self.no_dispatch():
            for path in self.real_configs():
                with self.subTest(config=path.stem):
                    preview = json.loads(self.invoke(path, "--check"))
                    config = workflows.load_config(path)
                    if isinstance(config, workflows.WorkflowConfig):
                        self.assertEqual(preview, {**config.select().snapshot(), "action_invoked": False})
                    else:
                        graphs, questions = LEGACY_COUNTS[path.stem]
                        self.assertEqual(preview["workflow"], "topology")
                        self.assertEqual(preview["action"], "run")
                        self.assertIs(preview["action_invoked"], False)
                        self.assertIs(preview["requires_inference"], True)
                        self.assertIs(preview["requires_downloads"], False)
                        self.assertEqual(preview["planned_graphs"], graphs)
                        self.assertEqual(preview["planned_questions"], questions)
                        self.assertEqual(preview["config"], config.snapshot())

    def test_every_real_config_action_compiles_previews_and_reaches_real_parser(self):
        count = 0
        for path in self.real_configs():
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
            config = workflows.load_config(path)
            if not isinstance(config, workflows.WorkflowConfig):
                continue
            for action in raw["actions"]:
                with self.subTest(config=path.stem, action=action):
                    invocation = config.select(action)
                    self.assert_compilation(path, raw, invocation)
                    with self.no_dispatch():
                        preview = json.loads(self.invoke(path, "--action", action, "--check"))
                    self.assertEqual(preview, {**invocation.snapshot(), "action_invoked": False})
                    self.assert_real_parser(invocation)
                    count += 1
        self.assertEqual(count, 29)

    def test_all_28_allowlisted_actions_accept_all_parameters_in_real_backend_parsers(self):
        registry = {(workflow, action) for workflow, spec in workflows._WORKFLOWS.items()
                    for action in spec.actions}
        examples = {(workflow, action) for workflow, actions in ACTION_PARAMETERS.items()
                    for action in actions}
        self.assertEqual(examples, registry)
        self.assertEqual(len(examples), 28)
        for workflow, actions in ACTION_PARAMETERS.items():
            for action, parameters in actions.items():
                with self.subTest(workflow=workflow, action=action):
                    spec = workflows._WORKFLOWS[workflow].actions[action]
                    self.assertEqual(set(parameters), set(spec.options))
                    path = self.write_config(workflow, action, parameters)
                    invocation = workflows.load_config(path).select()
                    self.assert_real_parser(invocation)
                    self.assertFalse((path.parent / "inputs").exists())
                    if invocation.root is not None:
                        self.assertFalse(invocation.root.exists())

                    # Also exercise omitted options/defaults with only required inputs.
                    minimal = {name: parameters[name] for name in spec.required}
                    path = self.write_config(workflow, action, minimal)
                    self.assert_real_parser(workflows.load_config(path).select())

    def test_false_boolean_is_omitted_and_real_parser_defaults_to_false(self):
        path = self.write_config("structural", "check-lane", {
            "model": "kev", "require_complete": False})
        invocation = workflows.load_config(path).select()
        self.assertNotIn("--require-complete", invocation.argv)
        self.assert_real_parser(invocation)

    def test_all_four_modified_backend_mains_accept_explicit_help_argv(self):
        original_argv, argv_values = sys.argv, list(sys.argv)
        for name in ("public_structural", "public_graphtext", "public_optimization",
                     "graph_abstraction_suite"):
            with self.subTest(module=name):
                module = importlib.import_module("src.utils." + name)
                stdout = io.StringIO()
                with redirect_stdout(stdout), self.assertRaises(SystemExit) as caught:
                    module.main(["--help"])
                self.assertEqual(caught.exception.code, 0)
                self.assertIn("--root", stdout.getvalue())
                self.assertIs(sys.argv, original_argv)
                self.assertEqual(sys.argv, argv_values)

    def test_legacy_planned_counts_and_permitted_run_preserve_progress_and_output(self):
        for name, (graphs, questions) in LEGACY_COUNTS.items():
            path = CONFIGS / f"{name}.yaml"
            original = workflows.load_config(path)
            for no_progress in (False, True):
                with self.subTest(config=name, no_progress=no_progress):
                    output = self.base / "unused" / ".." / "legacy root"
                    args = ["--action", "run", "--allow-inference", "--output", str(output)]
                    if no_progress:
                        args.append("--no-progress")
                    with patch.object(run_benchmark, "run_experiment", new_callable=AsyncMock,
                                      return_value={"completed_calls": 0}) as run, \
                            patch.object(workflows.WorkflowInvocation, "dispatch") as dispatch:
                        result = json.loads(self.invoke(path, *args))
                    dispatch.assert_not_called()
                    run.assert_awaited_once()
                    executed = run.await_args.args[0]
                    self.assertEqual(executed.planned_graphs, graphs)
                    self.assertEqual(executed.planned_questions, questions)
                    self.assertEqual(executed.run.output_dir, output.resolve())
                    for field in ("data", "sampling", "tasks", "model"):
                        self.assertEqual(getattr(executed, field), getattr(original, field))
                    self.assertEqual(run.await_args.kwargs, {"progress": not no_progress})
                    self.assertEqual(result, {"completed_calls": 0})

    def test_inference_gate_rejects_topology_and_every_workflow_run(self):
        with self.no_dispatch():
            for path in self.real_configs():
                if path.stem == "results":
                    continue
                for flags in ([], ["--allow-downloads"]):
                    with self.subTest(config=path.stem, flags=flags):
                        self.reject(path, "--action", "run", *flags, message="--allow-inference")
                if path.stem in LEGACY_COUNTS:
                    self.reject(path, message="--allow-inference")

    def test_download_gate_rejects_all_download_actions_even_with_inference_permission(self):
        with self.no_dispatch():
            for workflow, action in sorted(DOWNLOAD_ACTIONS):
                path = self.write_config(workflow, action)
                for flags in ([], ["--allow-inference"]):
                    with self.subTest(workflow=workflow, action=action, flags=flags):
                        self.reject(path, *flags, message="--allow-downloads")

    def test_check_overrides_both_gates_without_dispatch(self):
        with self.no_dispatch():
            for path in self.real_configs():
                config = workflows.load_config(path)
                actions = config.actions if isinstance(config, workflows.WorkflowConfig) else ["run"]
                for action in actions:
                    for flags in ([], ["--allow-inference"], ["--allow-downloads"],
                                  ["--allow-inference", "--allow-downloads"]):
                        with self.subTest(config=path.stem, action=action, flags=flags):
                            preview = json.loads(self.invoke(path, "--action", action, "--check", *flags))
                            self.assertIs(preview["action_invoked"], False)
                            self.assertEqual(preview["action"], action)

    def test_correct_permissions_dispatch_exactly_one_selected_workflow(self):
        for path in self.real_configs():
            config = workflows.load_config(path)
            if not isinstance(config, workflows.WorkflowConfig):
                continue
            for action in config.actions:
                with self.subTest(config=path.stem, action=action):
                    invocation = config.select(action)
                    flags = []
                    if invocation.requires_inference:
                        flags.append("--allow-inference")
                    if invocation.requires_downloads:
                        flags.append("--allow-downloads")
                    with patch.object(workflows.WorkflowInvocation, "dispatch", autospec=True,
                                      return_value=0) as dispatch, \
                            patch.object(run_benchmark, "run_experiment", new_callable=AsyncMock) as run:
                        self.assertEqual(self.invoke(path, "--action", action, *flags), "")
                    dispatch.assert_called_once_with(invocation)
                    run.assert_not_called()

    def test_workflow_no_progress_is_rejected_including_previews(self):
        with self.no_dispatch():
            for path in self.real_configs():
                if path.stem in LEGACY_COUNTS:
                    continue
                for flags in ([], ["--check"]):
                    with self.subTest(config=path.stem, flags=flags):
                        self.reject(path, "--no-progress", *flags,
                                    message="--no-progress is supported only for schema 1")

    def test_results_output_is_rejected_for_both_actions_including_previews(self):
        with self.no_dispatch():
            for action in ("check", "render"):
                for flags in ([], ["--check"]):
                    with self.subTest(action=action, flags=flags):
                        self.reject(CONFIGS / "results.yaml", "--action", action,
                                    "--output", str(self.base / "unused"), *flags,
                                    message="results does not support an output root override")

    def test_unknown_and_unconfigured_actions_are_cli_errors_before_dispatch(self):
        with self.no_dispatch():
            for path in self.real_configs():
                for flags in ([], ["--check"]):
                    with self.subTest(config=path.stem, flags=flags):
                        message = ("support only the run action" if path.stem in LEGACY_COUNTS
                                   else "action is not configured or supported")
                        self.reject(path, "--action", "unknown-action", *flags, message=message)
            path = self.write_config("development", "report")
            self.reject(path, "--action", "run", "--allow-inference", message="action is not configured")

    def test_output_override_is_cwd_relative_and_does_not_redirect_yaml_action_paths(self):
        # Patch cwd lookup, not process state or sys.argv; YAML paths remain absolute.
        cwd = self.base / "separate working directory"
        cwd.mkdir()
        for workflow, actions in ACTION_PARAMETERS.items():
            if workflow == "results":
                continue
            for action, parameters in actions.items():
                with self.subTest(workflow=workflow, action=action):
                    path = self.write_config(workflow, action, parameters)
                    original = workflows.load_config(path).select()
                    with patch("os.getcwd", return_value=str(cwd)):
                        with self.no_dispatch():
                            preview = json.loads(self.invoke(path, "--check", "--output", "x/../new root"))
                        with patch.object(workflows.WorkflowInvocation, "dispatch", autospec=True,
                                          return_value=0) as dispatch:
                            self.invoke(path, "--output", "x/../new root",
                                        "--allow-inference", "--allow-downloads")
                    invoked = dispatch.call_args.args[0]
                    dispatch.assert_called_once()
                    self.assertEqual(invoked.root, cwd / "new root")
                    self.assertEqual(preview["root"], str(cwd / "new root"))
                    self.assertEqual(invoked.parameters, original.parameters)
                    self.assertEqual(preview["parameters"], original.snapshot()["parameters"])
                    expected_argv = list(original.argv)
                    expected_argv[expected_argv.index("--root") + 1] = str(cwd / "new root")
                    self.assertEqual(invoked.argv, expected_argv)
                    self.assertEqual(preview["argv"], expected_argv)
                    self.assertEqual(workflows.load_config(path).select(), original)
        with patch("os.getcwd", return_value=str(cwd)), self.no_dispatch():
            preview = json.loads(self.invoke(CONFIGS / "pilot.yaml", "--check", "--no-progress",
                                             "--output", "x/../legacy root"))
        self.assertEqual(preview["config"]["run"]["output_dir"], str(cwd / "legacy root"))

    def test_backend_return_status_propagates_through_real_dispatch_to_system_exit(self):
        path = self.write_config("development", "report", {"models": ["jev"]})
        invocation = workflows.load_config(path).select()
        backend = importlib.import_module(invocation.module)
        for status in (None, 0, 1, 2, 17, 130, 143):
            with self.subTest(status=status), patch.object(backend, "main", return_value=status) as main:
                if status:
                    with self.assertRaises(SystemExit) as caught:
                        self.invoke(path)
                    self.assertEqual(caught.exception.code, status)
                else:
                    self.assertEqual(self.invoke(path), "")
                main.assert_called_once_with(invocation.argv)

    def test_backend_failures_are_not_rewritten_as_configuration_errors(self):
        path = self.write_config("development", "report")
        backend = importlib.import_module(workflows.load_config(path).select().module)
        failure = ValueError("synthetic backend failure")
        with patch.object(backend, "main", side_effect=failure), \
                self.assertRaises(ValueError) as caught:
            self.invoke(path)
        self.assertIs(caught.exception, failure)

    def test_default_results_action_executes_real_offline_check_read_only(self):
        from src.utils import render_results

        path = CONFIGS / "results.yaml"
        assets = {REPO / "assets/benchmark" / name for name in (
            "overview.svg", "structural-queries.svg", "graph-text-decisions.svg",
            "optimization-trajectories.svg")}
        assets.add(REPO / "docs/results.md")
        allowed = assets | {path, REPO / "data/results/summary.json"}
        original_open = io.open
        reads = []

        def read_only(file, mode="r", *args, **kwargs):
            target = Path(file).resolve()
            self.assertIn(target, allowed, "Unexpected results input")
            self.assertEqual(mode, "r", "Results check must not write")
            reads.append(target)
            return original_open(file, mode, *args, **kwargs)

        with patch("io.open", read_only), \
                patch("builtins.open", side_effect=AssertionError("Unexpected results file access")), \
                patch.object(render_results, "main", wraps=render_results.main) as main, \
                patch.object(run_benchmark, "run_experiment", new_callable=AsyncMock) as run:
            output = self.invoke(path)
        main.assert_called_once_with(["--check"])
        run.assert_not_called()
        self.assertCountEqual(reads, allowed)
        self.assertCountEqual(output.splitlines(), [
            "Checked " + p.relative_to(REPO).as_posix() for p in assets])

    def test_real_offline_development_plan_report_analyze_all_ten_tasks(self):
        from src.utils import extended_graph_suite as suite

        workspace = repo_workspace(self, "workflow-entrypoint-")
        path = workspace / "development.yaml"
        document = yaml.safe_load((CONFIGS / "development.yaml").read_text(encoding="utf-8"))
        document["root"] = "unused configured root"
        document["actions"]["plan"].update(
            design="structural_dev", models=["jev"], node_counts=[8], samples_per_size=1)
        path.write_text(yaml.safe_dump(document), encoding="utf-8")
        root = workspace / "executed plan"
        override = str(workspace / "unused" / ".." / root.name)
        # Keep real task builders, validation, source freezing and backend dispatch.
        # Only forbid inference and avoid inspecting host package/environment metadata.
        with ExitStack() as stack:
            stack.enter_context(warnings.catch_warnings())
            # NetworkX 3.5 announces its directed-graph hash fix during real generation.
            warnings.filterwarnings(
                "ignore", message="The hashes produced for directed graphs changed in version v3.5.*",
                category=UserWarning, module=r"networkx\..*")
            guards = [stack.enter_context(patch.object(suite, name, side_effect=AssertionError(
                "Development integration must never execute inference")))
                for name in ("run_cli", "run_live", "create_client")]
            stack.enter_context(patch.object(suite, "environment_versions", return_value={
                "python": "offline-test", "packages": {}}))
            planned = json.loads(self.invoke(path, "--output", override))
            self.assertEqual(planned["status"], "planned")
            self.assertEqual(planned["design"], "structural_dev")
            self.assertEqual(planned["models"], ["jev"])
            expected_tasks = {"adjacency", "degree_exact", "cycle_detection", "pair_connectivity",
                              "distance_threshold", "articulation_point", "tsp_construct",
                              "maxcut_construct", "lt_influence_construct",
                              "community_bipartition_construct"}
            self.assertEqual(set(planned["budgets"]["by_task"]), expected_tasks)
            self.assertTrue(all(v["instances"] > 0 for v in planned["budgets"]["by_task"].values()))
            manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["node_counts"], [8])
            self.assertEqual(manifest["samples_per_size"], 1)
            self.assertEqual(set(manifest["models"]), {"jev"})
            instances = [json.loads(line) for line in (root / "instances.jsonl").read_text(
                encoding="utf-8").splitlines()]
            self.assertEqual({row["task"] for row in instances}, expected_tasks)
            self.assertTrue(all(len(row["state"]["nodes"]) == 8 for row in instances))
            self.assertEqual(len(instances), planned["budgets"]["instances_per_model"])
            frozen = root / manifest["source_snapshot"]
            self.assertTrue(frozen.is_dir())
            self.assertTrue(all((frozen / name).stat().st_mode & 0o222 == 0
                                for name in manifest["code_sha256"]))

            def artifacts():
                return {p.relative_to(root): (p.read_bytes(), p.stat().st_mtime_ns, p.stat().st_mode)
                        for p in root.rglob("*") if p.is_file()}

            before = artifacts()
            reported = json.loads(self.invoke(path, "--action", "report", "--output", override))
            analyzed = json.loads(self.invoke(path, "--action", "analyze", "--output", override))
            self.assertEqual(reported["models"], {"jev": {"status": "not_started"}})
            self.assertEqual(reported["budgets"], planned["budgets"])
            self.assertEqual(analyzed["models"]["jev"], {
                "status": "not_started", "trajectory_analysis": {"status": "not_started", "episodes": []}})
            self.assertEqual(analyzed["model_calls"], 0)
            self.assertEqual(before, artifacts(), "Report and analyze must leave the plan unchanged")
            self.assertFalse((root / "runs").exists())
            self.assertFalse((workspace / document["root"]).exists())
            for guard in guards:
                guard.assert_not_called()
        # tests.support registers TemporaryDirectory cleanup, including read-only snapshots.


if __name__ == "__main__":
    unittest.main()