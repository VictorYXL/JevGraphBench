"""Offline workflow contracts using synthetic YAML and mocked runner dispatch."""

from __future__ import annotations

import builtins
import copy
import importlib
import json
import sys
import tempfile
import unittest
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import yaml

from src.benchmark import workflows
from src.benchmark.config import BenchmarkConfig, load_config as load_legacy


E = [
    "jev", "qwen08", "qwen2", "qwen4", "qwen9", "gpt54", "gpt6astra",
    "qwen9think", "qwen9recommended", "qwen9thinkrecommended", "qwen9constrained",
]
P = [
    "jev_action", "qwen4_grammar", "qwen9_grammar", "qwen27_grammar",
    "qwen4_token_scores", "decider", "kev", "laya", "gpt54_default_reasoning",
    "gpt6astra_default_reasoning", "qwen38_27b_bf16", "qwen25_72b_bf16",
    "qwen08", "qwen2",
]
MODULES = {
    "development": "src.utils.extended_graph_suite",
    "structural": "src.utils.public_structural",
    "graph_text": "src.utils.public_graphtext",
    "optimization": "src.utils.public_optimization",
    "proposals": "src.utils.graph_abstraction_suite",
    "results": "src.utils.render_results",
}
SECRET = "injected-secret-do-not-echo"


class WorkflowConfigTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="workflow-config-")
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name).resolve()
        self.path = self.base / "workflow.yaml"

    def envelope(self, workflow="development", action="plan", parameters=None):
        document = {
            "schema_version": 1, "workflow": workflow, "default_action": action,
            "actions": {action: {} if parameters is None else parameters},
        }
        if workflow != "results":
            document["root"] = "artifacts"
        return document

    def load(self, document):
        self.path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
        return workflows.load_config(self.path)

    def invalid(self, document):
        with self.assertRaises(ValueError) as caught:
            self.load(document)
        self.assertNotIn(SECRET, str(caught.exception))

    def action_cases(self):
        """Every public action, every supported parameter, explicit expected argv."""
        def p(name):
            return str(self.base / name)

        return [
            ("development", "plan", {
                "design": "structural_holdout", "seed": 0, "samples_per_size": 1,
                "node_counts": [8, 12], "prior_pool": "prior.jsonl",
                "no_think_max_tokens": 64, "models": ["jev", "qwen4"],
            }, ["--design", "structural_holdout", "--seed", "0", "--samples-per-size", "1",
                "--node-counts", "8", "12", "--prior-pool", p("prior.jsonl"),
                "--no-think-max-tokens", "64", "--models", "jev", "qwen4"]),
            ("development", "run", {"models": ["jev"]}, ["--models", "jev"]),
            ("development", "report", {"models": ["qwen9"]}, ["--models", "qwen9"]),
            ("development", "analyze", {"models": ["gpt54"]}, ["--models", "gpt54"]),
            ("structural", "freeze", {
                "data_dir": "data", "historical_root": "history-placeholder",
                "authorization": "authorization.json", "pilot_model_config": "pilot.json",
                "test_model_config": "test.json",
            }, ["--data-dir", p("data"), "--historical-root", p("history-placeholder"),
                "--authorization", p("authorization.json"), "--pilot-model-config", p("pilot.json"),
                "--test-model-config", p("test.json")]),
            ("structural", "validate", {"data_dir": "data"}, ["--data-dir", p("data")]),
            ("structural", "run", {
                "model": "jev_action", "split": "public-pilot", "concurrency": 16,
                "base_url": "https://example.invalid/v1",
            }, ["--model", "jev_action", "--split", "public-pilot", "--concurrency", "16",
                "--base-url", "https://example.invalid/v1"]),
            ("structural", "report", {"output": "report.json"}, ["--output", p("report.json")]),
            ("structural", "check-lane", {
                "model": "kev", "require_complete": True, "output": "lane.json",
            }, ["--model", "kev", "--require-complete", "--output", p("lane.json")]),
            ("structural", "combine", {
                "primary_root": "primary", "deployment_root": "deployment", "approval": "ok.json",
                "outer_plan": "plan.json", "activation_proof": "proof.json",
            }, ["--primary-root", p("primary"), "--deployment-root", p("deployment"),
                "--approval", p("ok.json"), "--outer-plan", p("plan.json"),
                "--activation-proof", p("proof.json")]),
            ("structural", "combined-report", {"output": "combined.json"},
                ["--output", p("combined.json")]),
            ("structural", "scoring-recovery-check", {
                "primary_root": "primary", "approval": "ok.json", "output": "recovery.json",
            }, ["--primary-root", p("primary"), "--approval", p("ok.json"),
                "--output", p("recovery.json")]),
            ("graph_text", "freeze", {}, []),
            ("graph_text", "verify", {}, []),
            ("graph_text", "report", {"output": "text.json"}, ["--output", p("text.json")]),
            ("graph_text", "run", {
                "model": "org/model-v1", "dataset": "arxiv", "approval": "ok.json",
            }, ["--model", "org/model-v1", "--dataset", "arxiv", "--approval", p("ok.json")]),
            ("optimization", "prepare", {}, []),
            ("optimization", "source-evidence", {}, []),
            ("optimization", "validate", {}, []),
            ("optimization", "freeze", {"tests_log": "tests.log", "cohort": "cohort.json"},
                ["--tests-log", p("tests.log"), "--cohort", p("cohort.json")]),
            ("optimization", "run", {
                "models": ["org/a", "org/b"], "model_config": "models.json",
                "concurrency": 1, "scope": "pilot",
            }, ["--models", "org/a", "org/b", "--model-config", p("models.json"),
                "--concurrency", "1", "--scope", "pilot"]),
            ("optimization", "report", {"scope": "main"}, ["--scope", "main"]),
            ("proposals", "freeze", {
                "instances": "instances.json", "model_config": "models.json", "models": ["laya"],
            }, ["--instances", p("instances.json"), "--model-config", p("models.json"),
                "--models", "laya"]),
            ("proposals", "run", {
                "model_config": "models.json", "models": ["decider"], "concurrency": 4,
            }, ["--model-config", p("models.json"), "--models", "decider", "--concurrency", "4"]),
            ("proposals", "report", {}, []),
            ("proposals", "monitor", {"interval": 9}, ["--interval", "9"]),
            ("results", "check", {}, ["--check"]),
            ("results", "render", {}, []),
        ]

    def test_every_action_compiles_exact_argv_and_module(self):
        cases = self.action_cases()
        self.assertEqual(len(cases), 28)
        for workflow, action, parameters, options in cases:
            with self.subTest(workflow=workflow, action=action):
                config = self.load(self.envelope(workflow, action, parameters))
                invocation = config.select()
                if workflow == "results":
                    expected = options
                elif workflow == "graph_text":
                    expected = ["--root", str(self.base / "artifacts"), action] + options
                else:
                    expected = [action, "--root", str(self.base / "artifacts")] + options
                self.assertEqual(invocation.argv, expected)
                self.assertEqual(invocation.module, MODULES[workflow])
                self.assertEqual(invocation.workflow, workflow)
                self.assertEqual(invocation.action, action)

    def test_effects_for_every_action(self):
        downloads = {("graph_text", "freeze"), ("optimization", "prepare"),
                     ("optimization", "source-evidence")}
        for workflow, action, parameters, _ in self.action_cases():
            with self.subTest(workflow=workflow, action=action):
                invocation = self.load(self.envelope(workflow, action, parameters)).select()
                self.assertIs(invocation.requires_inference, action == "run")
                self.assertIs(invocation.requires_downloads, (workflow, action) in downloads)
                self.assertIs(invocation.snapshot()["requires_inference"], action == "run")
                self.assertIs(invocation.snapshot()["requires_downloads"], (workflow, action) in downloads)

    def test_optional_settings_leave_runner_defaults_unchanged(self):
        for workflow, action in [
            ("development", "plan"), ("development", "report"), ("development", "analyze"),
            ("structural", "freeze"), ("structural", "validate"), ("structural", "report"),
            ("structural", "combined-report"), ("graph_text", "report"),
            ("optimization", "report"), ("proposals", "monitor"),
        ]:
            with self.subTest(workflow=workflow, action=action):
                invocation = self.load(self.envelope(workflow, action)).select()
                self.assertEqual(invocation.parameters, {})
                self.assertEqual(len(invocation.argv), 3)

    def test_schema_one_delegates_to_existing_loader(self):
        pilot = Path(__file__).resolve().parents[2] / "configs" / "pilot.yaml"
        expected = load_legacy(pilot)
        with patch.object(workflows.legacy_config, "load_config", wraps=load_legacy) as delegate:
            actual = workflows.load_config(pilot)
        delegate.assert_called_once_with(pilot)
        self.assertIsInstance(actual, BenchmarkConfig)
        self.assertEqual(actual, expected)

    def test_schema_one_errors_are_sanitized(self):
        self.invalid({"schema_version": 1, SECRET: SECRET})

    def test_schema_version_requires_exact_integer(self):
        for value in [True, False, 2, 2.0, 1.0, "2", "1", 0, 3, None, [], {}, SECRET]:
            with self.subTest(value=value):
                document = self.envelope()
                document["schema_version"] = value
                self.invalid(document)

    def test_envelope_requires_exact_keys(self):
        valid = self.envelope()
        for key in valid:
            with self.subTest(missing=key):
                document = copy.deepcopy(valid)
                del document[key]
                self.invalid(document)
        for key in ["description", "descriptions", "module", "shell", "kwargs", "secrets", SECRET]:
            with self.subTest(extra=key):
                self.invalid(dict(valid, **{key: SECRET}))

    def test_results_forbids_root_even_null(self):
        for value in [None, "artifacts", "", SECRET]:
            with self.subTest(value=value):
                document = self.envelope("results", "check")
                document["root"] = value
                self.invalid(document)

    def test_results_forbids_all_output_overrides(self):
        config = self.load(self.envelope("results", "render"))
        self.assertIsNone(config.root)
        self.assertIsNone(config.select().snapshot()["root"])
        for value in ["", "output", self.base, False]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                config.select(output=value)

    def test_envelope_and_section_types(self):
        for value in [None, [], True, 2, SECRET]:
            with self.subTest(top_level=value):
                self.invalid(value)
        for field in ["workflow", "default_action", "actions"]:
            for value in [None, [], True, 2, {}, SECRET]:
                with self.subTest(field=field, value=value):
                    document = self.envelope()
                    document[field] = value
                    self.invalid(document)
        for value in [None, [], True, 2, SECRET]:
            with self.subTest(parameters=value):
                document = self.envelope()
                document["actions"]["plan"] = value
                self.invalid(document)

    def test_unknown_actions_options_and_unused_flags_rejected(self):
        document = self.envelope()
        document["actions"][SECRET] = {}
        self.invalid(document)
        for workflow, action, parameters, _ in self.action_cases():
            with self.subTest(workflow=workflow, action=action):
                self.invalid(self.envelope(workflow, action, dict(parameters, **{SECRET: SECRET})))
        for workflow, action, parameters in [
            ("development", "report", {"seed": 1}),
            ("development", "run", {"models": ["jev"], "design": "pilot"}),
            ("structural", "report", {"model": "kev"}),
            ("structural", "run", {"model": "kev", "approval": "ok.json"}),
            ("graph_text", "freeze", {"dataset": "arxiv"}),
            ("optimization", "report", {"models": ["a"]}),
            ("optimization", "freeze", {"tests_log": "t.log", "scope": "pilot"}),
            ("proposals", "freeze", {"instances": "i.json", "model_config": "m.json",
                                      "models": ["kev"], "concurrency": 1}),
            ("proposals", "monitor", {"models": ["kev"]}),
            ("results", "check", {"output": "o.json"}),
        ]:
            with self.subTest(workflow=workflow, action=action):
                self.invalid(self.envelope(workflow, action, parameters))

    def test_all_actions_validated_not_only_default(self):
        document = self.envelope()
        document["actions"]["run"] = {"models": [SECRET]}
        self.invalid(document)
        document["actions"]["run"] = {}
        self.invalid(document)

    def test_default_action_must_be_configured(self):
        document = self.envelope()
        document["default_action"] = "report"
        self.invalid(document)

    def test_selection_requires_configured_action(self):
        document = self.envelope()
        document["actions"]["report"] = {"models": ["jev"]}
        config = self.load(document)
        self.assertEqual(config.select("report").action, "report")
        self.assertEqual(config.select().action, "plan")
        for value in ["run", "", SECRET, [], {}, True, 1]:
            with self.subTest(value=value), self.assertRaises(ValueError) as caught:
                config.select(value)
            self.assertNotIn(SECRET, str(caught.exception))

    def test_required_parameters(self):
        required_cases = [
            ("development", "run", {"models": ["jev"]}),
            ("structural", "run", {"model": "kev"}),
            ("structural", "check-lane", {"model": "kev"}),
            ("structural", "combine", {"primary_root": "p", "deployment_root": "d", "approval": "a"}),
            ("structural", "scoring-recovery-check", {"primary_root": "p", "approval": "a"}),
            ("graph_text", "run", {"model": "m", "dataset": "prime", "approval": "a"}),
            ("optimization", "freeze", {"tests_log": "t"}),
            ("optimization", "run", {"models": ["m"]}),
            ("proposals", "freeze", {"instances": "i", "model_config": "m", "models": ["kev"]}),
            ("proposals", "run", {"model_config": "m", "models": ["kev"]}),
        ]
        for workflow, action, parameters in required_cases:
            for name in parameters:
                with self.subTest(workflow=workflow, action=action, missing=name):
                    reduced = dict(parameters)
                    del reduced[name]
                    self.invalid(self.envelope(workflow, action, reduced))

    def test_structural_parameter_pairs(self):
        for action, required, pair in [
            ("freeze", {}, ("pilot_model_config", "test_model_config")),
            ("combine", {"primary_root": "p", "deployment_root": "d", "approval": "a"},
                ("outer_plan", "activation_proof")),
        ]:
            self.load(self.envelope("structural", action, required))
            self.load(self.envelope("structural", action, dict(required, **{key: "x" for key in pair})))
            for key in pair:
                with self.subTest(action=action, only=key):
                    self.invalid(self.envelope("structural", action, dict(required, **{key: "x"})))

    def test_development_design_conditions(self):
        for parameters in [
            {"node_counts": [5]}, {"design": "pilot", "node_counts": [8]},
            {"design": "structural_holdout"},
            {"design": "structural_holdout", "prior_pool": "p", "node_counts": [7, 8]},
            {"prior_pool": "p"}, {"design": "curriculum", "prior_pool": "p"},
            {"design": "structural_dev", "prior_pool": "p"}, {"design": SECRET},
        ]:
            with self.subTest(parameters=parameters):
                self.invalid(self.envelope(parameters=parameters))
        for parameters in [
            {}, {"design": "pilot"}, {"design": "curriculum", "node_counts": [5, 12]},
            {"design": "structural_dev", "node_counts": [5, 12]},
            {"design": "structural_holdout", "prior_pool": "p"},
            {"design": "structural_holdout", "prior_pool": "p", "node_counts": [8, 12]},
        ]:
            with self.subTest(parameters=parameters):
                self.load(self.envelope(parameters=parameters))

    def test_integer_fields_reject_booleans_and_invalid_ranges(self):
        cases = [
            ("development", "plan", {}, "seed", 0, None),
            ("development", "plan", {}, "samples_per_size", 1, None),
            ("development", "plan", {}, "no_think_max_tokens", 1, None),
            ("structural", "run", {"model": "kev"}, "concurrency", 1, 16),
            ("optimization", "run", {"models": ["a"]}, "concurrency", 1, 16),
            ("proposals", "run", {"model_config": "m", "models": ["kev"]}, "concurrency", 1, 16),
            ("proposals", "monitor", {}, "interval", 1, None),
        ]
        for workflow, action, required, key, minimum, maximum in cases:
            bad = [True, False, None, "1", 1.0, [], {}, SECRET, minimum - 1]
            if maximum is not None:
                bad.append(maximum + 1)
            for value in bad:
                with self.subTest(workflow=workflow, key=key, value=value):
                    self.invalid(self.envelope(workflow, action, dict(required, **{key: value})))
            for value in [minimum, maximum if maximum is not None else 999999]:
                self.load(self.envelope(workflow, action, dict(required, **{key: value})))

    def test_node_counts_strict_unique_nonempty_integer_lists(self):
        for value in [None, [], "8", [8, 8], [True], [False], [8.0], ["8"], [4], [13], [[8]], [SECRET]]:
            with self.subTest(value=value):
                self.invalid(self.envelope(parameters={"design": "curriculum", "node_counts": value}))

    def test_model_lists_and_exact_alias_allowlists(self):
        cases = [
            ("development", "plan", {}, E), ("development", "run", {}, E),
            ("development", "report", {}, E), ("development", "analyze", {}, E),
            ("proposals", "freeze", {"instances": "i", "model_config": "m"}, P),
            ("proposals", "run", {"model_config": "m"}, P),
        ]
        for workflow, action, required, aliases in cases:
            with self.subTest(workflow=workflow, action=action):
                config = self.load(self.envelope(workflow, action, dict(required, models=aliases)))
                self.assertEqual(config.actions[action]["models"], aliases)
            for value in [[], None, "jev", [aliases[0], aliases[0]], [True], [1], [[]], [{}], [SECRET]]:
                with self.subTest(workflow=workflow, action=action, value=value):
                    self.invalid(self.envelope(workflow, action, dict(required, models=value)))

    def test_structural_models_use_exact_panel(self):
        for action in ["run", "check-lane"]:
            for model in P:
                self.load(self.envelope("structural", action, {"model": model}))
            for model in ["jev", "qwen4", "gpt54", SECRET, None, [], True]:
                with self.subTest(action=action, model=model):
                    self.invalid(self.envelope("structural", action, {"model": model}))

    def test_model_identifiers_prevent_option_injection(self):
        for value in ["", " ", "--help", "a b", "a\nb", "a;command", "$(command)", None, True, {}, []]:
            with self.subTest(value=value):
                self.invalid(self.envelope("graph_text", "run", {
                    "model": value, "dataset": "prime", "approval": "a",
                }))
                self.invalid(self.envelope("optimization", "run", {"models": [value]}))
        for value in [[], None, "model", ["a", "a"]]:
            self.invalid(self.envelope("optimization", "run", {"models": value}))
        for value in ["org/model", "model:version", "model-v1.2", "org/model@revision", "_local"]:
            self.load(self.envelope("optimization", "run", {"models": [value]}))

    def test_choice_fields_reject_unknown_values_and_wrong_types(self):
        cases = [
            ("structural", "run", {"model": "kev"}, "split", ["test", "public-pilot"]),
            ("graph_text", "run", {"model": "m", "approval": "a"}, "dataset", ["arxiv", "prime"]),
            ("optimization", "run", {"models": ["m"]}, "scope", ["main", "pilot"]),
            ("optimization", "report", {}, "scope", ["main", "pilot"]),
        ]
        for workflow, action, required, field, choices in cases:
            for value in [SECRET, "", None, True, 1, [], {}]:
                with self.subTest(workflow=workflow, field=field, value=value):
                    self.invalid(self.envelope(workflow, action, dict(required, **{field: value})))
            for value in choices:
                self.load(self.envelope(workflow, action, dict(required, **{field: value})))

    def test_boolean_flags_true_emits_false_omits(self):
        for value in [True, False]:
            invocation = self.load(self.envelope("structural", "check-lane", {
                "model": "kev", "require_complete": value,
            })).select()
            self.assertEqual("--require-complete" in invocation.argv, value)
            self.assertIs(invocation.parameters["require_complete"], value)
        for value in [0, 1, "true", "false", None, [], {}]:
            self.invalid(self.envelope("structural", "check-lane", {"model": "kev", "require_complete": value}))

    def test_base_url_validated_without_network(self):
        for value in [
            "https://example.invalid/v1", "http://127.0.0.1:8000/v1", "http://[::1]:8000/v1",
        ]:
            self.load(self.envelope("structural", "run", {"model": "kev", "base_url": value}))
        for value in [
            SECRET, f"https://{SECRET}@example.invalid", f"https://user:{SECRET}@example.invalid",
            f"https://example.invalid/?token={SECRET}", f"https://example.invalid/#{SECRET}",
            "https://@example.invalid", "https://example.invalid/?", "https://example.invalid/#",
            "ftp://example.invalid", "http:///missing", "http://[invalid", "http://host:99999",
            "http://host:port", "https://host/path\n", "https://host/has space", "https://host\\path",
            None, True, [], {}, 123,
        ]:
            with self.subTest(value=value):
                self.invalid(self.envelope("structural", "run", {"model": "kev", "base_url": value}))

    def test_all_path_fields_resolve_against_yaml_parent(self):
        path_names = {
            "prior_pool", "data_dir", "historical_root", "authorization", "pilot_model_config",
            "test_model_config", "output", "primary_root", "deployment_root", "approval",
            "outer_plan", "activation_proof", "tests_log", "cohort", "model_config", "instances",
        }
        seen = set()
        for workflow, action, parameters, _ in self.action_cases():
            config = self.load(self.envelope(workflow, action, parameters))
            if workflow != "results":
                self.assertEqual(config.root, self.base / "artifacts")
            for name in parameters.keys() & path_names:
                with self.subTest(workflow=workflow, action=action, name=name):
                    seen.add(name)
                    normalized = config.actions[action][name]
                    self.assertIsInstance(normalized, Path)
                    self.assertEqual(normalized, self.base / parameters[name])
                    self.assertTrue(normalized.is_absolute())
        self.assertEqual(seen, path_names)

    def test_invalid_paths_rejected_everywhere(self):
        path_names = {
            "prior_pool", "data_dir", "historical_root", "authorization", "pilot_model_config",
            "test_model_config", "output", "primary_root", "deployment_root", "approval",
            "outer_plan", "activation_proof", "tests_log", "cohort", "model_config", "instances",
        }
        for value in ["", "  ", None, True, 12, [], {}, "nul\0path"]:
            document = self.envelope()
            document["root"] = value
            self.invalid(document)
        for workflow, action, parameters, _ in self.action_cases():
            for name in parameters.keys() & path_names:
                for value in ["", " ", None, True, 12, [], {}]:
                    with self.subTest(workflow=workflow, action=action, name=name, value=value):
                        invalid = dict(parameters, **{name: value})
                        self.invalid(self.envelope(workflow, action, invalid))

    def test_absolute_paths_and_parent_traversal_normalize(self):
        document = self.envelope("structural", "report", {"output": str(self.base / "absolute.json")})
        document["root"] = "child/../normalized"
        config = self.load(document)
        self.assertEqual(config.root, self.base / "normalized")
        self.assertEqual(config.select().parameters["output"], self.base / "absolute.json")

    def test_home_expansion_without_reading_environment(self):
        original = Path.expanduser

        def expand(path):
            if str(path).startswith("~/"):
                return self.base / "synthetic-home" / str(path)[2:]
            return original(path)

        document = self.envelope("structural", "report", {"output": "~/report.json"})
        document["root"] = "~/artifacts"
        with patch.object(Path, "expanduser", expand):
            config = self.load(document)
            invocation = config.select(output="~/override")
        self.assertEqual(config.root, self.base / "synthetic-home/artifacts")
        self.assertEqual(invocation.root, self.base / "synthetic-home/override")
        self.assertEqual(invocation.parameters["output"], self.base / "synthetic-home/report.json")

    def test_output_override_uses_cwd_not_yaml_parent(self):
        config = self.load(self.envelope("structural", "report", {"output": "report.json"}))
        cwd = self.base / "synthetic-cwd"
        with patch.object(Path, "cwd", return_value=cwd):
            invocation = config.select(output="new-root")
        self.assertEqual(invocation.root, cwd / "new-root")
        self.assertEqual(invocation.argv, ["report", "--root", str(cwd / "new-root"),
                                           "--output", str(self.base / "report.json")])
        self.assertEqual(config.root, self.base / "artifacts")
        absolute = self.base / "absolute-root"
        self.assertEqual(config.select(output=absolute).root, absolute)
        for value in ["", " ", True, 1, [], {}]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                config.select(output=value)

    def test_no_artifact_existence_checks_or_artifact_reads(self):
        document = self.envelope("proposals", "freeze", {
            "instances": "missing/instances.json", "model_config": "missing/models.json", "models": ["kev"],
        })
        self.path.write_text(yaml.safe_dump(document), encoding="utf-8")
        original_open = Path.open

        def only_config(path, *args, **kwargs):
            self.assertEqual(path, self.path)
            return original_open(path, *args, **kwargs)

        with patch.object(Path, "open", only_config), \
                patch.object(Path, "exists", side_effect=AssertionError("existence check")), \
                patch.object(Path, "is_file", side_effect=AssertionError("existence check")), \
                patch.object(Path, "is_dir", side_effect=AssertionError("existence check")):
            invocation = workflows.load_config(self.path).select()
            invocation.snapshot()

    def test_yaml_rejects_duplicate_nonstring_merge_and_malicious_nodes(self):
        for text in [
            f"schema_version: 1\nschema_version: {SECRET}\n",
            f"schema_version: 1\nworkflow: results\ndefault_action: check\nactions:\n  check: {{}}\n  check: {SECRET}\n",
            f"schema_version: 1\nworkflow: structural\nroot: x\ndefault_action: report\nactions:\n  report:\n    output: a\n    output: {SECRET}\n",
            f"1: {SECRET}\n", f"true: {SECRET}\n", f"? [a, b]\n: {SECRET}\n",
            f"schema_version: 1\nworkflow: results\ndefault_action: check\nactions:\n  check:\n    1: {SECRET}\n",
            f"!!python/object/apply:os.system ['{SECRET}']\n",
            f"schema_version: 1\nworkflow: results\ndefault_action: check\nactions:\n  check: !!python/name:{SECRET} ''\n",
            f"x: &x {{}}\n<<: *x\nsecret: {SECRET}\n",
            f"schema_version: [{SECRET}\n", f"root: *{SECRET}\n",
            f"schema_version: 1\n---\n{SECRET}\n",
            f"schema_version: 1\nworkflow: results\ndefault_action: check\nactions: &a {{check: *a, {SECRET}: x}}\n",
        ]:
            with self.subTest(text=text):
                self.path.write_text(text, encoding="utf-8")
                with self.assertRaises(ValueError) as caught:
                    workflows.load_config(self.path)
                self.assertNotIn(SECRET, str(caught.exception))

    def test_unreadable_invalid_encoding_and_wrong_extension_sanitized(self):
        with self.assertRaises(ValueError):
            workflows.load_config(self.base / f"{SECRET}.yaml")
        with self.assertRaises(ValueError) as caught:
            workflows.load_config(self.base / f"{SECRET}.json")
        self.assertNotIn(SECRET, str(caught.exception))
        self.path.write_bytes(b"\xff" + SECRET.encode())
        with self.assertRaises(ValueError) as caught:
            workflows.load_config(self.path)
        self.assertNotIn(SECRET, str(caught.exception))

    def test_snapshot_is_json_safe_complete_and_detached(self):
        invocation = self.load(self.envelope("development", "plan", {
            "design": "structural_holdout", "prior_pool": "pool.json", "models": ["jev"],
        })).select(output=self.base / "overridden")
        snapshot = invocation.snapshot()
        self.assertEqual(json.loads(json.dumps(snapshot)), snapshot)
        self.assertEqual(set(snapshot), {"workflow", "action", "root", "parameters", "module", "argv",
                                         "requires_inference", "requires_downloads"})
        self.assertEqual(snapshot["root"], str(self.base / "overridden"))
        self.assertEqual(snapshot["parameters"]["prior_pool"], str(self.base / "pool.json"))
        self.assertEqual(snapshot["argv"], invocation.argv)
        snapshot["parameters"]["models"].append("qwen4")
        snapshot["argv"].clear()
        self.assertEqual(invocation.parameters["models"], ["jev"])
        self.assertTrue(invocation.argv)

    def test_config_is_frozen_and_selection_detaches_parameters(self):
        config = self.load(self.envelope("development", "run", {"models": ["jev"]}))
        self.assertIsInstance(config, workflows.WorkflowConfig)
        with self.assertRaises(FrozenInstanceError):
            config.workflow = "results"
        selected = config.select()
        selected.parameters["models"].append("qwen4")
        self.assertEqual(config.actions["run"]["models"], ["jev"])
        self.assertEqual(config.select().parameters["models"], ["jev"])

    def test_option_order_is_independent_of_yaml_key_order(self):
        parameters = {"model": "kev", "split": "test", "concurrency": 2}
        first = self.load(self.envelope("structural", "run", parameters)).select()
        second = self.load(self.envelope("structural", "run", dict(reversed(list(parameters.items()))))).select()
        self.assertEqual(first.argv, second.argv)

    def test_no_runner_import_during_load_select_or_snapshot(self):
        original_import = builtins.__import__
        # Execute a fresh module without reloading the shared module: reloading
        # would invalidate WorkflowConfig class references in other test suites.
        spec = importlib.util.spec_from_file_location("_workflow_import_check", workflows.__file__)
        probe = importlib.util.module_from_spec(spec)

        def guarded_import(name, *args, **kwargs):
            if name == "src.utils" or name.startswith("src.utils."):
                raise AssertionError("runner imported during validation")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=guarded_import), \
                patch.object(workflows.importlib, "import_module", side_effect=AssertionError("lazy import invoked")), \
                patch.dict(sys.modules, {spec.name: probe}):
            spec.loader.exec_module(probe)
            for workflow, action, parameters, _ in self.action_cases():
                with self.subTest(workflow=workflow, action=action):
                    document = self.envelope(workflow, action, parameters)
                    self.path.write_text(yaml.safe_dump(document), encoding="utf-8")
                    probe.load_config(self.path).select().snapshot()

    def test_dispatch_calls_main_with_argv_and_preserves_sys_argv(self):
        invocation = self.load(self.envelope("results", "check")).select()
        original_argv = sys.argv
        original_values = list(sys.argv)

        def main(argv):
            self.assertIs(sys.argv, original_argv)
            self.assertEqual(sys.argv, original_values)
            self.assertEqual(argv, ["--check"])
            self.assertIsNot(argv, invocation.argv)
            argv.clear()
            return 7

        runner = SimpleNamespace(main=Mock(side_effect=main))
        with patch.object(workflows.importlib, "import_module", return_value=runner) as importing:
            self.assertEqual(invocation.dispatch(), 7)
        importing.assert_called_once_with("src.utils.render_results")
        runner.main.assert_called_once()
        self.assertEqual(invocation.argv, ["--check"])
        self.assertIs(sys.argv, original_argv)
        self.assertEqual(sys.argv, original_values)

    def test_dispatch_none_becomes_zero_for_every_workflow(self):
        for workflow, action, parameters, _ in self.action_cases():
            with self.subTest(workflow=workflow, action=action):
                invocation = self.load(self.envelope(workflow, action, parameters)).select()
                runner = SimpleNamespace(main=Mock(return_value=None))
                with patch.object(workflows.importlib, "import_module", return_value=runner) as importing:
                    self.assertEqual(invocation.dispatch(), 0)
                importing.assert_called_once_with(MODULES[workflow])
                runner.main.assert_called_once_with(invocation.argv)

    def test_dispatch_rejects_nonallowlisted_module(self):
        invocation = self.load(self.envelope("results", "render")).select()
        with patch.object(workflows.importlib, "import_module") as importing:
            with self.assertRaises(ValueError) as caught:
                replace(invocation, module=SECRET).dispatch()
            self.assertNotIn(SECRET, str(caught.exception))
        importing.assert_not_called()


if __name__ == "__main__":
    unittest.main()