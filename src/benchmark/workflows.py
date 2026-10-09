"""Strict, offline YAML workflow validation and allowlisted lazy CLI dispatch.

Schema 1 supports topology configs and explicit workflow actions, not arbitrary
runner arguments. Configs without workflow delegate to the topology loader.
Omitted optional settings retain the
runner's defaults; paths are resolved without checking artifact existence.
Loading and selecting never import runners, read artifacts, or chain actions.
"""

from __future__ import annotations

import copy
import importlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit

import yaml

from src.benchmark import config as legacy_config
from src.benchmark.config import BenchmarkConfig, _ConfigLoader


_DEVELOPMENT_MODELS = (
    "jev", "qwen08", "qwen2", "qwen4", "qwen9", "gpt54", "gpt6astra",
    "qwen9think", "qwen9recommended", "qwen9thinkrecommended", "qwen9constrained",
)
_PANEL = (
    "jev_action", "qwen4_grammar", "qwen9_grammar", "qwen27_grammar",
    "qwen4_token_scores", "decider", "kev", "laya", "gpt54_default_reasoning",
    "gpt6astra_default_reasoning", "qwen38_27b_bf16", "qwen25_72b_bf16",
    "qwen08", "qwen2",
)

_Validator = Callable[[Any, Path], Any]
_Check = Callable[[dict[str, Any]], None]


def _path(value: Any, parent: Path) -> Path:
    if not isinstance(value, str) or not value.strip() or "\0" in value:
        raise ValueError("paths must be nonempty strings without NUL characters")
    try:
        return (parent / Path(value).expanduser()).resolve()
    except (OSError, RuntimeError, ValueError):
        raise ValueError("unable to resolve configured path") from None


def _integer(minimum: int, maximum: int | None = None) -> _Validator:
    def validate(value: Any, parent: Path) -> int:
        if type(value) is not int or value < minimum or (
                maximum is not None and value > maximum):
            raise ValueError("integer outside the supported range")
        return value
    return validate


def _enum(*choices: str) -> _Validator:
    def validate(value: Any, parent: Path) -> str:
        if not isinstance(value, str) or value not in choices:
            raise ValueError("unsupported choice")
        return value
    return validate


def _list_of(item_validator: _Validator) -> _Validator:
    def validate(value: Any, parent: Path) -> list:
        if not isinstance(value, list) or not value:
            raise ValueError("expected a nonempty list")
        items = [item_validator(item, parent) for item in value]
        if len(items) != len(set(items)):
            raise ValueError("list entries must be unique")
        return items
    return validate


def _boolean(value: Any, parent: Path) -> bool:
    if type(value) is not bool:
        raise ValueError("expected a boolean")
    return value


def _identifier(value: Any, parent: Path) -> str:
    # A single CLI token, never an option, whitespace, or shell expression.
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.:/@+\-]*", value):
        raise ValueError("expected a nonempty model identifier")
    return value


def _base_url(value: Any, parent: Path) -> str:
    try:
        if (not isinstance(value, str) or any(c.isspace() or ord(c) < 32 for c in value)
                or any(c in value for c in ("?", "#", "\\"))):
            raise ValueError()
        url = urlsplit(value)
        if (url.scheme not in ("http", "https") or not url.hostname
                or url.username is not None or url.password is not None):
            raise ValueError()
        _ = url.port  # Validate malformed/out-of-range ports without a network call.
    except ValueError:
        raise ValueError("base_url must be HTTP(S) without credentials/query/fragment") from None
    return value


def _paired(first: str, second: str) -> _Check:
    def check(parameters: dict[str, Any]) -> None:
        if (first in parameters) != (second in parameters):
            raise ValueError(f"{first} and {second} must be supplied together")
    return check


def _check_plan(parameters: dict[str, Any]) -> None:
    design = parameters.get("design", "pilot")
    if design == "pilot" and "node_counts" in parameters:
        raise ValueError("pilot does not support node_counts")
    if design == "structural_holdout":
        if "prior_pool" not in parameters:
            raise ValueError("structural_holdout requires prior_pool")
        if any(size < 8 for size in parameters.get("node_counts", [])):
            raise ValueError("structural_holdout requires node_counts >= 8")
    elif "prior_pool" in parameters:
        raise ValueError("prior_pool is supported only for structural_holdout")


@dataclass(frozen=True)
class _ActionSpec:
    options: dict[str, _Validator] = field(default_factory=dict)
    required: frozenset[str] = frozenset()
    check: _Check | None = None
    downloads: bool = False

    def normalize(self, raw: Any, parent: Path) -> dict[str, Any]:
        if (not isinstance(raw, dict) or not self.required <= raw.keys()
                or raw.keys() - self.options.keys()):
            raise ValueError("missing required parameters or unsupported options")
        parameters = {}
        # Registry order makes argv stable independently of YAML key order.
        for name, validator in self.options.items():
            if name in raw:
                try:
                    parameters[name] = validator(raw[name], parent)
                except ValueError as exc:
                    # Only trusted registry names and validator messages appear.
                    raise ValueError(f"{name}: {exc}") from None
        if self.check is not None:
            self.check(parameters)
        return parameters


@dataclass(frozen=True)
class _WorkflowSpec:
    module: str
    actions: dict[str, _ActionSpec]


_development_models = _list_of(_enum(*_DEVELOPMENT_MODELS))
_panel_models = _list_of(_enum(*_PANEL))
_concurrency = _integer(1, 16)
_scope = _enum("main", "pilot")
_report = _ActionSpec({"output": _path})

_WORKFLOWS = {
    "development": _WorkflowSpec("src.utils.extended_graph_suite", {
        "plan": _ActionSpec({
            "design": _enum("pilot", "curriculum", "structural_dev", "structural_holdout"),
            "seed": _integer(0), "samples_per_size": _integer(1),
            "node_counts": _list_of(_integer(5, 12)), "prior_pool": _path,
            "no_think_max_tokens": _integer(1), "models": _development_models,
        }, check=_check_plan),
        "run": _ActionSpec({"models": _development_models}, frozenset({"models"})),
        "report": _ActionSpec({"models": _development_models}),
        "analyze": _ActionSpec({"models": _development_models}),
    }),
    "structural": _WorkflowSpec("src.utils.public_structural", {
        "freeze": _ActionSpec({
            "data_dir": _path, "historical_root": _path, "authorization": _path,
            "pilot_model_config": _path, "test_model_config": _path,
        }, check=_paired("pilot_model_config", "test_model_config")),
        "validate": _ActionSpec({"data_dir": _path}),
        "run": _ActionSpec({
            "model": _enum(*_PANEL), "split": _enum("test", "public-pilot"),
            "concurrency": _concurrency, "base_url": _base_url,
        }, frozenset({"model"})),
        "report": _report,
        "check-lane": _ActionSpec({
            "model": _enum(*_PANEL), "require_complete": _boolean, "output": _path,
        }, frozenset({"model"})),
        "combine": _ActionSpec({
            "primary_root": _path, "deployment_root": _path, "approval": _path,
            "outer_plan": _path, "activation_proof": _path,
        }, frozenset({"primary_root", "deployment_root", "approval"}),
            _paired("outer_plan", "activation_proof")),
        "combined-report": _report,
        "scoring-recovery-check": _ActionSpec({
            "primary_root": _path, "approval": _path, "output": _path,
        }, frozenset({"primary_root", "approval"})),
    }),
    "graph_text": _WorkflowSpec("src.utils.public_graphtext", {
        "freeze": _ActionSpec(downloads=True),
        "verify": _ActionSpec(),
        "report": _report,
        "run": _ActionSpec({
            "model": _identifier, "dataset": _enum("arxiv", "prime"), "approval": _path,
        }, frozenset({"model", "dataset", "approval"})),
    }),
    "optimization": _WorkflowSpec("src.utils.public_optimization", {
        "prepare": _ActionSpec(downloads=True),
        "source-evidence": _ActionSpec(downloads=True),
        "validate": _ActionSpec(),
        "freeze": _ActionSpec({"tests_log": _path, "cohort": _path}, frozenset({"tests_log"})),
        "run": _ActionSpec({
            "models": _list_of(_identifier), "model_config": _path,
            "concurrency": _concurrency, "scope": _scope,
        }, frozenset({"models"})),
        "report": _ActionSpec({"scope": _scope}),
    }),
    "proposals": _WorkflowSpec("src.utils.graph_abstraction_suite", {
        "freeze": _ActionSpec({
            "instances": _path, "model_config": _path, "models": _panel_models,
        }, frozenset({"instances", "model_config", "models"})),
        "run": _ActionSpec({
            "model_config": _path, "models": _panel_models, "concurrency": _concurrency,
        }, frozenset({"model_config", "models"})),
        "report": _ActionSpec(),
        "monitor": _ActionSpec({"interval": _integer(1)}),
    }),
    "results": _WorkflowSpec("src.utils.render_results", {
        "check": _ActionSpec(), "render": _ActionSpec(),
    }),
}


@dataclass(frozen=True)
class WorkflowInvocation:
    """One effective CLI call; effects are declarations, not permission grants."""

    workflow: str
    action: str
    root: Path | None
    parameters: dict[str, Any]
    module: str
    argv: list[str]
    requires_inference: bool
    requires_downloads: bool

    def snapshot(self) -> dict[str, Any]:
        """Return a detached JSON-safe view, including resolved paths and effects."""
        return json.loads(json.dumps(asdict(self), default=str))

    def dispatch(self) -> int:
        """Import only the allowlisted runner and call main(argv) directly.

        The caller is responsible for obtaining any inference/download approval.
        No sys.argv mutation, shell, subprocess, or automatic follow-up is used.
        """
        spec = _WORKFLOWS.get(self.workflow)
        if spec is None or self.module != spec.module or self.action not in spec.actions:
            raise ValueError("unsupported workflow dispatch")
        status = importlib.import_module(self.module).main(list(self.argv))
        return 0 if status is None else int(status)


@dataclass(frozen=True)
class WorkflowConfig:
    workflow: str
    root: Path | None
    default_action: str
    actions: dict[str, dict[str, Any]]

    def select(self, action: str | None = None, output: str | Path | None = None) -> WorkflowInvocation:
        """Select exactly one configured action, optionally overriding its root.

        The output override is relative to cwd, not the YAML directory. It does
        not replace an action's own output parameter. Results has no root.
        """
        selected = self.default_action if action is None else action
        spec = _WORKFLOWS.get(self.workflow)
        if (spec is None or not isinstance(selected, str)
                or selected not in self.actions or selected not in spec.actions):
            raise ValueError("action is not configured or supported")
        root = self.root
        if output is not None:
            if self.workflow == "results":
                raise ValueError("results does not support an output root override")
            if not isinstance(output, (str, Path)):
                raise ValueError("output root must be a path")
            root = _path(str(output), Path.cwd())
        parameters = copy.deepcopy(self.actions[selected])
        if self.workflow == "results":
            argv = ["--check"] if selected == "check" else []
        elif self.workflow == "graph_text":
            argv = ["--root", str(root), selected]
        else:
            argv = [selected, "--root", str(root)]
        for name, value in parameters.items():
            option = "--" + name.replace("_", "-")
            if type(value) is bool:
                if value:
                    argv.append(option)
            else:
                argv.append(option)
                argv.extend(str(item) for item in (value if isinstance(value, list) else [value]))
        return WorkflowInvocation(
            self.workflow, selected, root, parameters, spec.module, argv,
            requires_inference=selected == "run",
            requires_downloads=spec.actions[selected].downloads,
        )


def load_config(path: str | Path) -> BenchmarkConfig | WorkflowConfig:
    """Load either schema-1 envelope; never disclose rejected source values."""
    path = _path(str(path), Path.cwd())
    if path.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError("Workflow config must be a .yaml or .yml file")
    try:
        with path.open("r", encoding="utf-8") as stream:
            data = yaml.load(stream, Loader=_ConfigLoader)
    except (yaml.YAMLError, ValueError, RecursionError):
        # PyYAML exceptions can include source text, including rejected secrets.
        raise ValueError("Invalid or unsupported YAML configuration") from None
    except OSError:
        raise ValueError("Unable to read YAML configuration") from None
    if (not isinstance(data, dict) or type(data.get("schema_version")) is not int
            or data["schema_version"] != 1):
        raise ValueError("schema_version must be the integer 1")
    if "workflow" not in data:
        # Preserve the original schema's defaults and semantics, including paths.
        try:
            return legacy_config.load_config(path)
        except (yaml.YAMLError, ValueError, TypeError, RecursionError):
            raise ValueError("Invalid schema_version 1 configuration") from None
    workflow = data.get("workflow")
    if not isinstance(workflow, str) or workflow not in _WORKFLOWS:
        raise ValueError("unsupported workflow")
    required = {"schema_version", "workflow", "default_action", "actions"}
    if workflow != "results":
        required.add("root")
    if data.keys() != required:
        raise ValueError("workflow envelope has missing or unsupported sections")
    raw_actions = data["actions"]
    spec = _WORKFLOWS[workflow]
    if (not isinstance(raw_actions, dict) or not raw_actions
            or raw_actions.keys() - spec.actions.keys()):
        raise ValueError("actions must be a nonempty mapping of supported actions")
    default = data["default_action"]
    if not isinstance(default, str) or default not in raw_actions:
        raise ValueError("default_action must be a configured action")
    actions = {}
    for name, raw in raw_actions.items():
        try:
            actions[name] = spec.actions[name].normalize(raw, path.parent)
        except ValueError as exc:
            raise ValueError(f"{workflow}.{name}: {exc}") from None
    root = None if workflow == "results" else _path(data["root"], path.parent)
    return WorkflowConfig(workflow, root, default, actions)