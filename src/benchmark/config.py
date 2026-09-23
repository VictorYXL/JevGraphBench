"""Strict safe YAML configuration; no secrets or arbitrary constructor kwargs."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

from src.datasets import available_datasets


class _ConfigLoader(yaml.SafeLoader):
    """SafeLoader with string-only, unique mapping keys (no silent overrides).

    Merge keys are intentionally unsupported: configurations are explicit rather
    than inherited. Standard safe YAML scalar/list/mapping values are supported.
    """

    def construct_mapping(self, node, deep=False):
        if not isinstance(node, yaml.MappingNode):
            raise ValueError("YAML configuration requires mappings")
        mapping = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, str):
                raise ValueError("YAML mapping keys must be strings")
            if key in mapping:
                raise ValueError(f"Duplicate YAML key at line {key_node.start_mark.line + 1}")
            mapping[key] = self.construct_object(value_node, deep=deep)
        return mapping


@dataclass(frozen=True)
class RunConfig:
    seed: int
    repetitions: int
    max_calls: int
    output_dir: Path


@dataclass(frozen=True)
class DataConfig:
    datasets: tuple[str, ...]
    data_dir: Path


@dataclass(frozen=True)
class SamplingConfig:
    node_counts: tuple[int, ...]
    samples_per_size: int
    max_attempts: int
    method: str = "connected_frontier"


@dataclass(frozen=True)
class TaskConfig:
    adjacency_questions: int
    distance_questions_per_threshold: int
    distance_thresholds: tuple[int, ...]


@dataclass(frozen=True)
class ModelConfig:
    provider: str
    model: str
    timeout_seconds: float


@dataclass(frozen=True)
class BenchmarkConfig:
    schema_version: int
    run: RunConfig
    data: DataConfig
    sampling: SamplingConfig
    tasks: TaskConfig
    model: ModelConfig

    @property
    def planned_graphs(self) -> int:
        return len(self.data.datasets) * len(self.sampling.node_counts) * self.sampling.samples_per_size

    @property
    def planned_questions(self) -> int:
        per_graph = self.tasks.adjacency_questions + (
            self.tasks.distance_questions_per_threshold * len(self.tasks.distance_thresholds)
        )
        return self.planned_graphs * per_graph

    def snapshot(self) -> dict:
        return json.loads(json.dumps(asdict(self), default=str))

    @property
    def sha256(self) -> str:
        body = json.dumps(self.snapshot(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(body.encode()).hexdigest()


def _keys(value: object, expected: set[str], name: str) -> dict:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{name}: expected exactly these keys: {', '.join(sorted(expected))}")
    return value


def _integer(value: object, name: str, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _integers(value: object, name: str, minimum: int, *, empty: bool = False) -> tuple[int, ...]:
    if not isinstance(value, list) or (not value and not empty):
        raise ValueError(f"{name} must be a nonempty array")
    result = tuple(_integer(v, name, minimum) for v in value)
    if len(result) != len(set(result)):
        raise ValueError(f"{name} must not contain duplicates")
    return result


def _path(value: object, parent: Path) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("paths must be nonempty strings")
    path = Path(value).expanduser()
    return (parent / path).resolve()


def load_config(path: str | Path) -> BenchmarkConfig:
    path = Path(path).resolve()
    if path.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError("Benchmark config must be a .yaml or .yml file")
    try:
        with path.open("r", encoding="utf-8") as stream:
            data = yaml.load(stream, Loader=_ConfigLoader)
    except yaml.YAMLError as exc:
        # PyYAML's usual exception text echoes input. Avoid logging config values.
        mark = getattr(exc, "problem_mark", None)
        location = f" at line {mark.line + 1}" if mark is not None else ""
        raise ValueError(f"Invalid or unsupported YAML configuration{location}") from None
    _keys(data, {"schema_version", "run", "data", "sampling", "tasks", "model"}, "config")
    if type(data["schema_version"]) is not int or data["schema_version"] != 1:
        raise ValueError("Only schema_version = 1 is supported")
    run = _keys(data["run"], {"seed", "repetitions", "max_calls", "output_dir"}, "run")
    source = _keys(data["data"], {"datasets", "data_dir"}, "data")
    sampling = _keys(data["sampling"], {"method", "node_counts", "samples_per_size", "max_attempts"}, "sampling")
    tasks = _keys(data["tasks"], {"adjacency_questions", "distance_questions_per_threshold", "distance_thresholds"}, "tasks")
    model = _keys(data["model"], {"provider", "model", "timeout_seconds"}, "model")
    names = source["datasets"]
    if not isinstance(names, list) or not names or any(
        not isinstance(n, str) or n not in available_datasets() for n in names
    ):
        raise ValueError("data.datasets must contain registered dataset IDs")
    if len(names) != len(set(names)):
        raise ValueError("data.datasets must not contain duplicates")
    if sampling["method"] != "connected_frontier":
        raise ValueError("Only connected_frontier sampling is supported")
    adjacency = _integer(tasks["adjacency_questions"], "adjacency_questions", 0)
    distance = _integer(tasks["distance_questions_per_threshold"], "distance_questions_per_threshold", 0)
    thresholds = _integers(tasks["distance_thresholds"], "distance_thresholds", 2, empty=True)
    if adjacency % 2 or distance % 2:
        raise ValueError("Question counts must be even for equal yes/no quotas")
    if bool(distance) != bool(thresholds) or not (adjacency or distance):
        raise ValueError("Enable at least one task; distance count and thresholds must both be enabled or disabled")
    if model["provider"] != "typesafe":
        raise ValueError("The CLI currently supports provider = 'typesafe' only")
    if not isinstance(model["model"], str) or not model["model"].strip():
        raise ValueError("model.model must be nonempty")
    timeout = model["timeout_seconds"]
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout_seconds must be finite and positive")
    config = BenchmarkConfig(
        schema_version=1,
        run=RunConfig(_integer(run["seed"], "seed", 0),
                      _integer(run["repetitions"], "repetitions"),
                      _integer(run["max_calls"], "max_calls"),
                      _path(run["output_dir"], path.parent)),
        data=DataConfig(tuple(names), _path(source["data_dir"], path.parent)),
        sampling=SamplingConfig(_integers(sampling["node_counts"], "node_counts", 3),
                                _integer(sampling["samples_per_size"], "samples_per_size"),
                                _integer(sampling["max_attempts"], "max_attempts")),
        tasks=TaskConfig(adjacency, distance, thresholds),
        model=ModelConfig(model["provider"], model["model"], float(timeout)),
    )
    if config.planned_questions * config.run.repetitions > config.run.max_calls:
        raise ValueError("Planned calls exceed run.max_calls (checked before sampling or network access)")
    return config