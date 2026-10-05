#!/usr/bin/env python3
"""Public RQ1: freeze induced queries, run once, and replay an all-scheduled report.

Example (freeze/report are offline; only run performs inference):
  python -m src.utils.public_structural_v32 freeze --root DIR
  python DIR/frozen-source/src/utils/public_structural_v32.py run \
      --root DIR --model qwen4_grammar --split test --concurrency 4
  python DIR/frozen-source/src/utils/public_structural_v32.py report --root DIR

The 800 test queries retain the historical quotas exactly. The 52-query public
pilot uses a separate seed and source node sets. Neither split is filtered using
model outcomes. Isomorphic samples are audited, not silently removed: requiring
non-isomorphism would change the degree-zero and sparse-graph target population.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
from collections import Counter, defaultdict
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import asdict, is_dataclass
import fcntl
import hashlib
from itertools import combinations
import json
import os
from pathlib import Path
import random
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import httpx
import networkx as nx

from src.benchmark import extended_tasks as tasks
from src.clients.base import (
    ClientTimeoutError, DecisionClientError, DecisionResponse, InvalidResponseError,
    UnsupportedRequestError,
)
from src.clients._llm_choice import ANSWER_ONLY_SYSTEM_MESSAGE, SYSTEM_MESSAGE, prompt_for
from src.datasets.real_graphs import load_graph
from src.utils import graph_abstraction_suite as prior

SEED = 20261003
SOURCES = ("facebook", "ca-grqc", "power", "human-ppi")
TASKS = ("degree_exact", "cycle_detection", "pair_connectivity", "adjacency",
         "distance_threshold", "articulation_point")
VERSION = "public-structural-v32-20261003"
HISTORICAL = REPO / "output/experiments/diagnostics/graph-proposals-20261003-v2"
AUTHORIZATION = REPO / "output/experiments/public/v32-20261003/authorization.json"
CAPTURE = ContextVar("rq1_capture", default=None)
PRIMARY_MODELS = (
    "jev_action", "qwen08", "qwen2", "qwen4_grammar", "qwen9_grammar",
    "gpt54_default_reasoning", "gpt6astra_default_reasoning",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write(path, value):
    path = Path(path)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(canonical(value) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    sync_dir(path.parent)


def read(path):
    return json.loads(Path(path).read_text())


def emit(value):
    stream = CAPTURE.get()
    require(stream is not None, "Raw capture outside a claimed query")
    require(not prior.contains_credential_material(value), "Credential material in raw capture")
    stream.write(canonical(value) + "\n")
    stream.flush()
    os.fsync(stream.fileno())


def source_key(source, ids):
    return sha([source, sorted(ids)])


def induced_state(graph, ids):
    inverse = {original: local for local, original in enumerate(ids)}
    edges = sorted(sorted((inverse[u], inverse[v])) for u, v in graph.subgraph(ids).edges())
    return {"nodes": list(range(len(ids))), "edges": edges, "directed": False}


def independent_answer(task, state):
    graph = nx.Graph()
    graph.add_nodes_from(state["nodes"])
    graph.add_edges_from(state["edges"])
    if task == "degree_exact":
        answer = graph.degree[state["vertex"]]
        require(tasks._exact_answer(task, state) == answer, "Independent oracle disagreement")
        return answer
    if task == "cycle_detection":
        positive = bool(nx.cycle_basis(graph))
    elif task == "adjacency":
        positive = graph.has_edge(*state["pair"])
    elif task == "pair_connectivity":
        positive = nx.has_path(graph, *state["pair"])
    elif task == "distance_threshold":
        positive = state["pair"][1] in nx.single_source_shortest_path_length(
            graph, state["pair"][0], cutoff=state["threshold"])
    elif task == "articulation_point":
        positive = state["vertex"] in set(nx.articulation_points(graph))
    else:
        raise ValueError("Unknown structural task")
    answer = "yes" if positive else "no"
    require(tasks._exact_answer(task, state) == answer, "Independent oracle disagreement")
    return answer


def quotas(split):
    require(split in ("test", "public-pilot"), "Unknown split")
    for source in SOURCES:
        for n in (range(8, 13) if split == "test" else (8,)):
            for degree in (range(n) if split == "test" else (0, 4, 7)):
                for repeat in range(2 if split == "test" else 1):
                    yield source, n, "degree_exact", degree, repeat
            for task in TASKS[1:]:
                for answer in ("yes", "no"):
                    for repeat in range(2 if split == "test" else 1):
                        yield source, n, task, answer, repeat


def sample_query(dataset, n, task, answer, rng, attempt):
    graph = dataset.graph
    nodes = list(graph)
    if task == "degree_exact":
        eligible = [v for v in nodes if graph.degree[v] >= answer and
                    len(nodes) - 1 - graph.degree[v] >= n - 1 - answer]
        require(bool(eligible), f"{dataset.source.name}: degree {answer} at n={n} infeasible")
        center = rng.choice(eligible)
        ids = [center, *rng.sample(sorted(graph[center]), answer)]
        while len(ids) < n:
            candidate = rng.choice(nodes)
            if candidate not in ids and candidate not in graph[center]:
                ids.append(candidate)
    else:
        # Fixed alternating uniform/frontier proposals; restarts permit disconnected samples.
        ids = [rng.choice(nodes)]
        while len(ids) < n:
            frontier = sorted({v for u in ids for v in graph[u]} - set(ids))
            local = attempt % 2 == 1 and frontier and rng.random() < 0.85
            candidate = rng.choice(frontier if local else nodes)
            if candidate not in ids:
                ids.append(candidate)
    rng.shuffle(ids)
    state = induced_state(graph, ids)
    if task == "degree_exact":
        state["vertex"] = ids.index(center)
        return ids, state
    if task == "cycle_detection":
        return (ids, state) if independent_answer(task, state) == answer else None
    candidates = list(range(n)) if task == "articulation_point" else list(combinations(range(n), 2))
    rng.shuffle(candidates)
    for candidate in candidates:
        query = deepcopy(state)
        if task == "articulation_point":
            query["vertex"] = candidate
        else:
            query["pair"] = list(candidate)
            if task == "distance_threshold":
                query["threshold"] = 2
        if independent_answer(task, query) == answer:
            return ids, query
    return None


def build_split(datasets, split, used=None, *, max_attempts=20000, excluded_queries=None):
    used = set() if used is None else used
    excluded_queries = set() if excluded_queries is None else excluded_queries
    instances, provenance = [], []
    for source, n, task, answer, repeat in quotas(split):
        cell = [split, source, n, task, answer, repeat]
        seed = sha([SEED, cell])
        rng = random.Random(seed)
        for attempt in range(1, max_attempts + 1):
            sampled = sample_query(datasets[source], n, task, answer, rng, attempt)
            if sampled is None:
                continue
            ids, state = sampled
            key = source_key(source, ids)
            if key not in used and sha([task, state]) not in excluded_queries:
                break
        else:
            raise ValueError(f"Sampling budget exhausted without topology edits: {cell}")
        used.add(key)
        require(independent_answer(task, state) == answer, "Quota oracle mismatch")
        identity = "rq1-" + sha([cell, ids, state])[:24]
        instances.append({"id": identity, "task": task, "kind": "exact", "state": state,
                          "query_budget": 1, "private": {"answer": answer}})
        provenance.append({
            "instance_id": identity, "split": split, "source": source, "n": n,
            "original_node_ids": ids, "source_node_set_sha256": key,
            "normalized_state_sha256": sha(state), "cell_seed": seed,
            "attempts": attempt, "quota": cell,
            "sampling": "degree-conditioned vertices" if task == "degree_exact"
                        else "alternating uniform/frontier induced proposals, answer-conditioned query",
        })
    random.Random(f"{SEED}:{split}:order").shuffle(instances)
    tasks.validate_instances(instances)
    return instances, provenance


def overlap_audit(instances, provenance):
    """Exact VF2 equivalence classes, with query marks for query-isomorphism."""
    meta = {p["instance_id"]: p for p in provenance}
    result = {}
    for query in (False, True):
        buckets = defaultdict(list)
        for instance in instances:
            state = instance["state"]
            graph = nx.Graph()
            marked = ({state["vertex"]} if "vertex" in state else set(state.get("pair", [])))
            graph.add_nodes_from((v, {"role": int(query and v in marked)}) for v in state["nodes"])
            graph.add_edges_from(state["edges"])
            key = (len(graph), graph.number_of_edges(),
                   tuple(sorted(dict(graph.degree()).values())),
                   instance["task"] if query else None, state.get("threshold") if query else None)
            groups = buckets[key]
            for representative, ids in groups:
                if nx.is_isomorphic(graph, representative,
                                    node_match=nx.algorithms.isomorphism.categorical_node_match("role", 0)):
                    ids.append(instance["id"])
                    break
            else:
                groups.append((graph, [instance["id"]]))
        groups = [ids for bucket in buckets.values() for _, ids in bucket]
        repeated = [ids for ids in groups if len(ids) > 1]
        result["query_isomorphism" if query else "graph_isomorphism"] = {
            "classes": len(groups), "repeated_classes": repeated,
            "duplicate_pairs": sum(len(ids) * (len(ids) - 1) // 2 for ids in groups),
            "cross_split_pairs": sum(
                sum(meta[a]["split"] != meta[b]["split"] for a, b in combinations(ids, 2))
                for ids in repeated),
        }
    keys = [p["source_node_set_sha256"] for p in provenance]
    result["duplicate_source_node_sets"] = len(keys) - len(set(keys))
    result["identical_requests_ignoring_id"] = len(instances) - len({
        sha([i["task"], i["state"]]) for i in instances})
    literal = defaultdict(list)
    for instance in instances:
        literal[sha([instance["task"], instance["state"]])].append(instance["id"])
    result["literal_query_overlap"] = {
        "repeated_groups": [ids for ids in literal.values() if len(ids) > 1],
        "cross_split_pairs": sum(
            meta[a]["split"] != meta[b]["split"]
            for ids in literal.values() for a, b in combinations(ids, 2)),
    }
    original_sets = {source: {split: set() for split in ("test", "public-pilot")} for source in SOURCES}
    for p in provenance:
        if "original_node_ids" in p:
            original_sets[p["source"]][p["split"]].update(p["original_node_ids"])
    result["source_vertices_shared_between_splits"] = {
        source: len(sets["test"] & sets["public-pilot"]) for source, sets in original_sets.items()}
    result["policy"] = ("Reject identical source node sets across both splits; retain and disclose "
                        "graph/query isomorphisms, including cross-split isomorphism. "
                        "Pilot additionally excludes literal test queries ignoring request IDs. "
                        "This is not an isomorphism-disjoint generalization split.")
    return result


def model_configs(root, historical=HISTORICAL):
    configs, history = {}, {}
    for name in prior.PANEL:
        path = historical / "runs" / name / "config.json"
        spec = read(path)
        history[name] = {"path": str(path), "sha256": file_sha(path)}
        if name in ("kev", "decider", "laya"):
            spec["adapter_kwargs"]["audit_path"] = str(root / "native-audit" / f"{name}.jsonl")
        configs[name] = spec
    validate_configs(configs)
    return configs, history


def validate_configs(configs):
    require(set(configs) == set(prior.PANEL), "Exactly all fourteen historical panel IDs required")
    for name, spec in configs.items():
        config = prior.validate_lane_config(name, spec)
        require(config.timeout_seconds == (30 if name == "jev_action" else 180), "Timeout drift")
        if name.startswith("gpt"):
            require(config.max_tokens == 4096 and config.reasoning_effort is None and
                    config.think is None, "GPT must retain default reasoning and 4096 tokens")


def scientific_spec(spec):
    result = deepcopy(spec)
    if result["provider"] == "vllm":
        result.pop("base_url", None)
    elif result["provider"] == "plugin" and "base_url" in result["adapter_kwargs"]:
        result["adapter_kwargs"].pop("base_url")
        result["adapter_kwargs"].pop("audit_path", None)
    return result


def bind_stage_configs(configs, paths, root):
    require(set(paths) == {"public-pilot", "test"}, "Both stage model configs are required")
    stage_configs, bindings = {}, {}
    for split, path in paths.items():
        path = Path(path).resolve()
        before = file_sha(path)
        supplied = read(path)
        validate_configs(supplied)
        for name, spec in supplied.items():
            require(scientific_spec(spec) == scientific_spec(configs[name]),
                    f"Canonical stage config changes scientific settings: {split}/{name}")
            if spec["provider"] == "plugin" and "audit_path" in spec["adapter_kwargs"]:
                audit_path = Path(spec["adapter_kwargs"]["audit_path"])
                require(audit_path.is_absolute() and audit_path.resolve().is_relative_to(root.parent),
                        "Native audit path must stay inside the RQ1 experiment tree")
        require(file_sha(path) == before, "Stage model config changed while binding")
        stage_configs[split] = supplied
        bindings[split] = {"path": str(path), "sha256": before}
    return stage_configs, bindings


def verify_stage_bindings(root):
    for split, binding in read(Path(root) / "deployment-config-bindings.json").items():
        require(file_sha(binding["path"]) == binding["sha256"], "Deployment config hash drift")
        require(read(binding["path"]) == read(Path(root) / "stage-models.json")[split],
                "Deployment config snapshot drift")


def code_paths():
    paths = [* (REPO / "src/clients").glob("*.py"),
             * (REPO / "src/benchmark").glob("*.py"),
             REPO / "src/__init__.py", REPO / "src/datasets/__init__.py",
             REPO / "src/datasets/sources.py", REPO / "src/datasets/real_graphs.py",
             REPO / "src/utils/__init__.py",
             * (REPO / "src/utils").glob("*graph*suite.py"),
             REPO / "src/utils/paired_graph_ablation.py",
             REPO / "tests/test_public_structural_v32.py", Path(__file__)]
    return sorted(set(p for p in paths if p.is_file()))


def authorization_binding(path):
    path = Path(path).resolve()
    document = read(path)
    require(document["authorization"] == "user_requested_execution" and
            document["new_local_thinking_lanes"] is False and
            set(document["models"]) == set(prior.PANEL), "Authorization model scope mismatch")
    rq = document["rq1"]
    require(set(rq["tasks"]) == set(TASKS) and rq["node_counts"] == list(range(8, 13)) and
            rq["target_queries"] == 800 and rq["no_topology_repair"] is True,
            "Authorization RQ1 scope mismatch")
    return {"path": str(path), "sha256": file_sha(path), "scope": "rq1"}, document


def verify_authorization(root):
    binding = read(Path(root) / "authorization-binding.json")
    require(file_sha(binding["path"]) == binding["sha256"], "Execution authorization hash drift")
    current, document = authorization_binding(binding["path"])
    require(current == binding and document == read(Path(root) / "authorization-snapshot.json"),
            "Execution authorization snapshot drift")
    return binding["sha256"]


def freeze(root, *, data_dir=REPO / "data", historical=HISTORICAL, authorization=AUTHORIZATION,
           pilot_model_config=None, test_model_config=None):
    root = Path(root).resolve()
    require(not root.exists(), "Freeze root exists; refuse overwrite")
    binding, authorization_document = authorization_binding(authorization)
    configs, history = model_configs(root, historical)
    require((pilot_model_config is None) == (test_model_config is None),
            "Supply both pilot and test model configuration files")
    stage_configs = {"public-pilot": deepcopy(configs), "test": deepcopy(configs)}
    deployment_bindings = {}
    if pilot_model_config is not None:
        stage_configs, deployment_bindings = bind_stage_configs(
            configs, {"public-pilot": pilot_model_config, "test": test_model_config}, root)
    datasets = {name: load_graph(name, data_dir=data_dir) for name in SOURCES}
    used = set()
    test, test_meta = build_split(datasets, "test", used)
    pilot, pilot_meta = build_split(
        datasets, "public-pilot", used,
        excluded_queries={sha([i["task"], i["state"]]) for i in test})
    provenance = test_meta + pilot_meta
    audit = overlap_audit(test + pilot, provenance)
    require(audit["duplicate_source_node_sets"] == 0, "Source overlap")
    require(audit["literal_query_overlap"]["cross_split_pairs"] == 0, "Literal pilot/test query overlap")
    root.mkdir(parents=True)
    hashes = {}
    payloads = {
        "authorization-binding.json": binding, "authorization-snapshot.json": authorization_document,
        "stage-models.json": stage_configs, "deployment-config-bindings.json": deployment_bindings,
        "instances.json": test, "public-pilot.json": pilot, "provenance.json": provenance,
        "overlap-audit.json": audit, "models.json": configs,
        "source-manifest.json": {
            name: {"source": asdict(data.source), "raw_sha256": data.raw_sha256,
                   "stats": asdict(data.stats),
                   "normalized_graph_sha256": sha({"nodes": list(data.graph),
                       "edges": sorted(sorted(e) for e in data.graph.edges())})}
            for name, data in datasets.items()},
        "freeze-config.json": {
            "seed": SEED, "sizes": list(range(8, 13)), "test_instances": 800,
            "degree_per_size_answer": 8, "binary_per_size": 16,
            "binary_yes_no_per_task": [40, 40], "per_source": 200,
            "distance_threshold": 2, "max_attempts_per_cell": 20000,
            "public_pilot_instances": 52, "connectedness_constraint": None,
            "pilot_exclusions": "test source-node sets and literal normalized test queries",
            "normalization": "Existing verified simple undirected loader; induced edges only",
            "model_outcome_selection": False, "quota_deviations": [],
            "historical_model_configs": history,
            "cache_manifest_sha256": file_sha(Path(data_dir) / "manifest.json"),
            "concurrency": {name: 1 if spec["provider"] == "plugin" else 4
                            for name, spec in configs.items()},
            "raw_capture": "fsynced intent, HTTP payload/response before parsing, SDK assistant events, "
                           "DecisionResponse; no headers/credentials; no local retries",
            "sdk_limit": "Observed provider retries rejected; provider-internal at-most-once "
                         "cannot be guaranteed, only one durable adapter send per query",
        },
    }
    for name, payload in payloads.items():
        write(root / name, payload)
        hashes[name] = file_sha(root / name)
    code = {}
    for path in code_paths():
        relative = path.relative_to(REPO)
        target = root / "frozen-source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(path.read_bytes())
            stream.flush()
            os.fsync(stream.fileno())
        target.chmod(0o444)
        code[str(relative)] = file_sha(target)
    write(root / "manifest.json", {"version": VERSION, "files": hashes, "code": code,
                                  "instances": 800, "public_pilot_instances": 52,
                                  "models": list(configs)})
    write(root / "manifest.sha256.json", {"sha256": file_sha(root / "manifest.json")})
    validate_frozen(root, data_dir=data_dir)
    return root


def load_frozen(root):
    root = Path(root).resolve()
    require(file_sha(root / "manifest.json") == read(root / "manifest.sha256.json")["sha256"],
            "Manifest hash drift")
    manifest = read(root / "manifest.json")
    require(manifest["version"] == VERSION, "Wrong experiment version")
    for name, expected in manifest["files"].items():
        require(file_sha(root / name) == expected, f"Frozen data drift: {name}")
    for name, expected in manifest["code"].items():
        require(file_sha(root / "frozen-source" / name) == expected, f"Snapshot drift: {name}")
        require(file_sha(REPO / name) == expected, f"Executing source drift: {name}; use frozen entrypoint")
    verify_authorization(root)
    verify_stage_bindings(root)
    return manifest


def validate_frozen(root, *, data_dir=None):
    root = Path(root)
    load_frozen(root)
    configs = read(root / "models.json")
    validate_configs(configs)
    for stage in read(root / "stage-models.json").values():
        validate_configs(stage)
        require(all(scientific_spec(stage[name]) == scientific_spec(configs[name]) for name in configs),
                "Stage scientific settings drift")
    metadata = read(root / "provenance.json")
    by_id = {p["instance_id"]: p for p in metadata}
    datasets = {s: load_graph(s, data_dir=data_dir) for s in SOURCES} if data_dir else None
    validation = {}
    all_instances = []
    for split, filename in (("test", "instances.json"), ("public-pilot", "public-pilot.json")):
        instances = read(root / filename)
        all_instances.extend(instances)
        validation[split] = tasks.validate_instances(instances)
        cells = Counter(tuple(p["quota"]) for p in metadata if p["split"] == split)
        require(cells == Counter((split, *cell) for cell in quotas(split)), "Quota mismatch")
        require(len(instances) == sum(cells.values()), "Quota count mismatch")
        for instance in instances:
            p = by_id[instance["id"]]
            state = instance["state"]
            require(p["split"] == split and sha(state) == p["normalized_state_sha256"], "Provenance drift")
            require(source_key(p["source"], p["original_node_ids"]) == p["source_node_set_sha256"],
                    "Original ID hash drift")
            if datasets:
                reconstructed = induced_state(datasets[p["source"]].graph, p["original_node_ids"])
                require(all(state[k] == v for k, v in reconstructed.items()), "Not source-induced")
    require(len(by_id) == len(all_instances) == len(metadata), "Provenance identity mismatch")
    require(overlap_audit(all_instances, metadata) == read(root / "overlap-audit.json"), "Audit drift")
    return validation


async def http_request_capture(request):
    if CAPTURE.get() is not None:
        emit({"event": "wire_request", "method": request.method,
              "path": request.url.path, "body": request.content.decode("utf-8"),
              "body_base64": base64.b64encode(request.content).decode("ascii")})


async def http_response_capture(response):
    if CAPTURE.get() is not None:
        await response.aread()
        emit({"event": "wire_response", "status": response.status_code,
              "body": response.content.decode("utf-8", errors="replace"),
              "body_base64": base64.b64encode(response.content).decode("ascii")})


class RuntimeCapture:
    """Observe SDK events before the existing client parses/rejects a completion."""
    def __init__(self, runtime):
        self.runtime = runtime

    def __getattr__(self, name):
        return getattr(self.runtime, name)

    async def create_session(self, **kwargs):
        callback = kwargs["on_event"]
        stream = CAPTURE.get()

        def on_event(event):
            kind = getattr(event.type, "value", event.type)
            if kind in ("assistant.message", "assistant.usage", "assistant.turn_retry"):
                data = event.data
                value = asdict(data) if is_dataclass(data) else data.model_dump(mode="json")
                token = CAPTURE.set(stream)
                try:
                    emit({"event": "sdk_event", "type": kind, "data": value})
                finally:
                    CAPTURE.reset(token)
            callback(event)

        return await self.runtime.create_session(**{**kwargs, "on_event": on_event})


def install_capture(client, spec):
    transport = getattr(client, "_http", getattr(client, "http", None))
    if isinstance(transport, httpx.AsyncClient):
        transport.event_hooks["request"].append(http_request_capture)
        transport.event_hooks["response"].append(http_response_capture)
    elif spec["provider"] == "github_copilot":
        client._runtime = RuntimeCapture(client._runtime)
    else:
        raise ValueError("Client lacks a supported pre-parse raw capture surface")


def failure_record(error):
    return {"status": "unsupported" if isinstance(error, UnsupportedRequestError) else "failed",
            "error_type": type(error).__name__,
            "diagnostic_code": getattr(error, "diagnostic_code", None)}


async def execute(directory, instance, client, spec, *, initialization_error=None,
                  authorization_sha256=None):
    directory.mkdir()
    sync_dir(directory.parent)
    request = tasks.next_request(instance, [])
    write(directory / "intent.json", {"request": asdict(request),
          "request_sha256": prior.request_digest(request), "time": time.time(),
          "authorization_sha256": authorization_sha256,
          "initialization_failed": initialization_error is not None})
    with (directory / "raw.jsonl").open("x") as stream:
        sync_dir(directory)
        token = CAPTURE.set(stream)
        start = time.perf_counter()
        decisions = []
        try:
            if initialization_error is not None:
                result = failure_record(initialization_error)
            else:
                try:
                    client.validate_request(request)
                    response = await asyncio.wait_for(client.predict(request), spec["timeout_seconds"] + 5)
                    require(isinstance(response, DecisionResponse), "Invalid response object")
                    emit({"event": "decision_response", "response": asdict(response)})
                    require(response.request_id == request.request_id and
                            response.resolved_model == spec["model"], "Response identity drift")
                    if response.selected_option_id not in {o.id for o in request.options}:
                        raise InvalidResponseError("Illegal choice", diagnostic_code="invalid_choice")
                    decisions = [response.selected_option_id]
                    result = {"status": "success", "selected_option_id": response.selected_option_id}
                except (DecisionClientError, TimeoutError) as error:
                    result = failure_record(ClientTimeoutError("Total timeout") if
                                            isinstance(error, TimeoutError) else error)
            result.update(request_sha256=prior.request_digest(request),
                          latency_seconds=time.perf_counter() - start,
                          score=tasks.score(instance, decisions))
            write(directory / "result.json", result)
            write(directory / "receipt.json", {
                name: file_sha(directory / name)
                for name in ("intent.json", "raw.jsonl", "result.json")})
        finally:
            CAPTURE.reset(token)
    return result


async def run(root, model, *, split="test", concurrency=None, base_url=None,
              factory=prior.make_client, capture=True):
    root = Path(root).resolve()
    load_frozen(root)
    authorization_sha256 = verify_authorization(root)
    require(model in prior.PANEL and split in ("test", "public-pilot"), "Unknown lane")
    configs = read(root / "stage-models.json")[split]
    spec = deepcopy(configs[model])
    maximum = read(root / "freeze-config.json")["concurrency"][model]
    concurrency = maximum if concurrency is None else concurrency
    require(type(concurrency) is int and 1 <= concurrency <= maximum, "Concurrency exceeds frozen panel")
    if base_url is not None:
        url = httpx.URL(base_url)
        require(url.scheme in ("http", "https") and url.host and not
                (url.username or url.password or url.query or url.fragment), "Unsafe endpoint")
        if spec["provider"] == "vllm":
            spec["base_url"] = base_url
        elif spec["provider"] == "plugin" and "base_url" in spec["adapter_kwargs"]:
            spec["adapter_kwargs"]["base_url"] = base_url
        else:
            raise ValueError("This adapter does not support endpoint relocation")
    prior.validate_lane_config(model, spec)
    lane = root / "runs" / split / model
    lane.mkdir(parents=True, exist_ok=True)
    with (lane / "lane.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (lane / "config.json").exists():
            require(read(lane / "config.json") == spec, "Effective run config changed")
        else:
            write(lane / "config.json", spec)
        write(lane / f"launch-{time.time_ns()}.json", {
            "concurrency": concurrency, "pid": os.getpid(), "split": split,
            "manifest_sha256": file_sha(root / "manifest.json"), "at_most_once": True,
            "authorization_sha256": authorization_sha256, "effective_config_sha256": sha(spec)})
        instances = read(root / ("instances.json" if split == "test" else "public-pilot.json"))
        queue = asyncio.Queue()
        for instance in instances:
            if not (lane / instance["id"]).exists():
                queue.put_nowait(instance)

        async def worker():
            if queue.empty():
                return
            client = factory(spec)
            initialization_error = None
            try:
                try:
                    await client.initialize()
                except (DecisionClientError, httpx.HTTPError, ImportError) as error:
                    initialization_error = error
                if capture and initialization_error is None:
                    install_capture(client, spec)
                while not queue.empty():
                    require(verify_authorization(root) == authorization_sha256,
                            "Authorization changed during lane")
                    verify_stage_bindings(root)
                    instance = queue.get_nowait()
                    await execute(lane / instance["id"], instance, client, spec,
                                  initialization_error=initialization_error,
                                  authorization_sha256=authorization_sha256)
            finally:
                await client.aclose()
        await asyncio.gather(*(worker() for _ in range(concurrency)))


def report(root):
    root = Path(root).resolve()
    load_frozen(root)
    authorization_sha256 = verify_authorization(root)
    output = {"version": VERSION, "manifest_sha256": file_sha(root / "manifest.json"), "rows": []}
    output["authorization_sha256"] = authorization_sha256
    for split, filename in (("test", "instances.json"), ("public-pilot", "public-pilot.json")):
        instances = read(root / filename)
        for model in read(root / "models.json"):
            details = []
            for instance in instances:
                directory = root / "runs" / split / model / instance["id"]
                status, result = "not_started", None
                if directory.exists():
                    status = "interrupted_unknown"
                    if (directory / "intent.json").exists():
                        request = tasks.next_request(instance, [])
                        intent = read(directory / "intent.json")
                        require(intent["request"] == json.loads(canonical(asdict(request))) and
                                intent["request_sha256"] == prior.request_digest(request), "Intent drift")
                        require(intent["authorization_sha256"] == authorization_sha256,
                                "Query authorization binding drift")
                    if (directory / "result.json").exists():
                        require((directory / "intent.json").is_file(), "Terminal result lacks intent")
                        if not (directory / "receipt.json").exists():
                            details.append({"instance_id": instance["id"], "task": instance["task"],
                                            "n": len(instance["state"]["nodes"]),
                                            "status": "interrupted_unsealed", "correct": False})
                            continue
                        require(read(directory / "receipt.json") == {
                            name: file_sha(directory / name)
                            for name in ("intent.json", "raw.jsonl", "result.json")},
                            "Terminal query receipt drift")
                        result = read(directory / "result.json")
                        require(result["request_sha256"] == prior.request_digest(request),
                                "Result request binding drift")
                        status = result["status"]
                        require(status in ("success", "failed", "unsupported"), "Invalid terminal status")
                        raw, torn = prior.read_events(directory / "raw.jsonl")
                        require(not torn, "Torn raw capture with terminal result")
                        choices = []
                        if status == "success":
                            responses = [r["response"] for r in raw if r["event"] == "decision_response"]
                            require(len(responses) == 1, "Missing/duplicate raw response")
                            response = responses[0]
                            require(response["selected_option_id"] == result["selected_option_id"] and
                                    response["request_id"] == request.request_id, "Raw result drift")
                            choices = [response["selected_option_id"]]
                        require(tasks.score(instance, choices) == result["score"], "Score replay drift")
                details.append({"instance_id": instance["id"], "task": instance["task"], "status": status,
                                "n": len(instance["state"]["nodes"]),
                                "correct": bool(result and result["score"].get("correct", False))})
            groups = {}
            for task in TASKS:
                subset = [d for d in details if d["task"] == task]
                by_size = {
                    str(n): {"scheduled": sum(d["n"] == n for d in subset),
                             "correct": sum(d["correct"] for d in subset if d["n"] == n),
                             "accuracy_all_scheduled":
                                 sum(d["correct"] for d in subset if d["n"] == n) /
                                 sum(d["n"] == n for d in subset)}
                    for n in sorted({d["n"] for d in subset})}
                groups[task] = {"scheduled": len(subset),
                                "statuses": dict(Counter(d["status"] for d in subset)),
                                "correct": sum(d["correct"] for d in subset),
                                "accuracy_all_scheduled": sum(d["correct"] for d in subset) / len(subset),
                                "by_size": by_size,
                                "size_macro_accuracy_all_scheduled":
                                    sum(v["accuracy_all_scheduled"] for v in by_size.values()) / len(by_size)}
            output["rows"].append({"split": split, "model": model, "scheduled": len(details),
                                   "statuses": dict(Counter(d["status"] for d in details)),
                                   "terminal": all(d["status"] in ("success", "failed", "unsupported")
                                                   for d in details),
                                   "by_task": groups, "instances": details})
    return output


def check_lane(root, model, split):
    require(model in prior.PANEL and split in ("test", "public-pilot"), "Unknown lane")
    full_report = report(root)
    row = next(r for r in full_report["rows"] if r["model"] == model and r["split"] == split)
    require(row["scheduled"] == (800 if split == "test" else 52), "Lane scheduled count mismatch")
    require(row["terminal"], f"Lane is not terminal: {row['statuses']}")
    return {
        "integrity_and_coverage_passed": True, "model": model, "split": split,
        "manifest_sha256": full_report["manifest_sha256"],
        "authorization_sha256": full_report["authorization_sha256"],
        "scheduled": row["scheduled"], "statuses": row["statuses"],
        "model_accuracy_gate": False,
        "infrastructure_review_required": bool(row["statuses"].get("failed")),
        "infrastructure_review_note": "Retain failures; scheduler must explicitly accept infrastructure "
                                      "readiness before full inference. No accuracy threshold.",
    }


def inspect_frozen_root(root):
    """Verify another runtime's snapshot without executing its runner or clients."""
    root = Path(root).resolve()
    manifest = read(root / "manifest.json")
    require(file_sha(root / "manifest.json") == read(root / "manifest.sha256.json")["sha256"],
            "Root manifest seal drift")
    require(manifest["version"] == VERSION, "Incompatible RQ1 root")
    for name, expected in manifest["files"].items():
        require(file_sha(root / name) == expected, f"Root data drift: {root}/{name}")
    for name, expected in manifest["code"].items():
        require(file_sha(root / "frozen-source" / name) == expected, f"Root runtime drift: {name}")
    if "authorization-binding.json" in manifest["files"]:
        verify_authorization(root)
    if "deployment-config-bindings.json" in manifest["files"]:
        verify_stage_bindings(root)
    return manifest


def root_stage_models(root, manifest):
    return (read(root / "stage-models.json") if "stage-models.json" in manifest["files"] else
            {split: read(root / "models.json") for split in ("test", "public-pilot")})


def approved_root_overrides(approved):
    primary_models = approved["existing_primary_lanes_must_not_be_repeated"]
    deployment_models = approved["untouched_deployment_lanes"]
    require(len(set(primary_models)) == len(primary_models) and
            len(set(deployment_models)) == len(deployment_models) and
            set(PRIMARY_MODELS) <= set(primary_models) and
            not set(primary_models) & set(deployment_models) and
            set(primary_models) | set(deployment_models) == set(prior.PANEL),
            "Supplemental root override scope mismatch")
    return {model: "primary" if model in primary_models else "deployment" for model in prior.PANEL}


def observed_outer_routes(overrides, primary_root, authorization_pin, plan_path, proof_path):
    plan_path, proof_path = Path(plan_path).resolve(), Path(proof_path).resolve()
    plan, proof = read(plan_path), read(proof_path)
    require(proof["plan_sha256"] == file_sha(plan_path) and
            proof["authorization_sha256"] == authorization_pin["sha256"] and
            plan["hashes"].get(authorization_pin["path"]) == authorization_pin["sha256"],
            "Outer activation authorization binding mismatch")
    runner = primary_root / "frozen-source/src/utils/public_structural_v32.py"
    require(plan["hashes"].get(str(primary_root / "manifest.json")) ==
            file_sha(primary_root / "manifest.json") and
            plan["hashes"].get(str(runner)) == file_sha(runner), "Outer plan runtime binding mismatch")
    result, adopted = dict(overrides), {}
    for model, assigned in overrides.items():
        counts = {split: sum(p.is_dir() for p in (primary_root / "runs" / split / model).iterdir())
                  if (primary_root / "runs" / split / model).exists() else 0
                  for split in ("public-pilot", "test")}
        if assigned != "deployment" or not any(counts.values()):
            continue
        jobs = []
        for split in ("public-pilot", "test"):
            matched = []
            for job in plan["jobs"]:
                command = job.get("command", [])
                if len(command) < 3 or command[1:3] != [str(runner), "run"]:
                    continue
                if all(command.count(flag) == 1 and command.index(flag) + 1 < len(command) and
                       command[command.index(flag) + 1] == expected for flag, expected in (
                           ("--root", str(primary_root)), ("--model", model), ("--split", split))):
                    matched.append(job["id"])
            require(len(matched) == 1, f"Existing primary route lacks one authorized job: {model}/{split}")
            jobs.extend(matched)
        result[model] = "primary"
        adopted[model] = {"authorized_job_ids": jobs, "primary_claims_at_binding": counts,
                          "reason": "Preserve actual first-root claims authorized by the activated outer plan"}
    return result, {
        "plan": {"path": str(plan_path), "sha256": file_sha(plan_path)},
        "activation_proof": {"path": str(proof_path), "sha256": file_sha(proof_path)},
        "adopted_existing_routes": adopted,
        "scope": "Source attribution for existing claims only; does not launch or authorize new attempts",
    }


def combine(root, primary_root, deployment_root, approval, authorization=AUTHORIZATION,
            outer_plan=None, activation_proof=None):
    """Freeze an explicit mixed-root report selection; never copy or execute raw runs."""
    root, primary_root, deployment_root = (
        Path(p).resolve() for p in (root, primary_root, deployment_root))
    require(not root.exists(), "Combined manifest root already exists")
    authorization_pin, _ = authorization_binding(authorization)
    approval = Path(approval).resolve()
    approved = read(approval)
    require(approved["status"] == "approved_under_user_requested_public_pilot_and_full_evaluation"
            and approved["authorization_sha256"] == authorization_pin["sha256"] and
            approved["pilot_queries_per_model"] == 52 and approved["models"] == 14 and
            approved["maximum_pilot_calls"] == 728 and
            approved["maximum_main_calls"] == 11200 and
            approved["existing_per_call_and_output_budgets_preserved"] is True and
            approved["automatic_retries"] is False, "Supplemental execution approval mismatch")
    overrides = approved_root_overrides(approved)
    declared_overrides = dict(overrides)
    require((outer_plan is None) == (activation_proof is None),
            "Outer plan and activation proof must be supplied together")
    outer_evidence = None
    if outer_plan is not None:
        overrides, outer_evidence = observed_outer_routes(
            overrides, primary_root, authorization_pin, outer_plan, activation_proof)
    roots = {"primary": primary_root, "deployment": deployment_root}
    manifests = {name: inspect_frozen_root(path) for name, path in roots.items()}
    data_files = ("instances.json", "public-pilot.json", "provenance.json",
                  "source-manifest.json", "overlap-audit.json")
    require(all(manifests["primary"]["files"][name] == manifests["deployment"]["files"][name]
                for name in data_files), "Mixed-root dataset/provenance identity mismatch")
    shared_code = {name: digest for name, digest in manifests["primary"]["code"].items()
                   if name not in ("src/utils/public_structural_v32.py",
                                   "tests/test_public_structural_v32.py")}
    require(all(manifests["deployment"]["code"].get(name) == digest and
                file_sha(REPO / name) == digest for name, digest in shared_code.items()),
            "Task rendering, oracle, client, or shared runtime source changed")
    stages = {name: root_stage_models(path, manifests[name]) for name, path in roots.items()}
    comparisons = {}
    for split in ("test", "public-pilot"):
        comparisons[split] = {}
        for model in prior.PANEL:
            first, second = stages["primary"][split][model], stages["deployment"][split][model]
            prior.validate_lane_config(model, first)
            prior.validate_lane_config(model, second)
            require(scientific_spec(first) == scientific_spec(second),
                    f"Model behavior changed across roots: {split}/{model}")
            comparisons[split][model] = {
                "primary_config_sha256": sha(first), "deployment_config_sha256": sha(second),
                "same_config_bytes": canonical(first) == canonical(second),
                "scientific_spec_sha256": sha(scientific_spec(first)),
                "allowed_transport_differences": {
                    "base_url": [first.get("base_url"), second.get("base_url")],
                    "native_endpoint": [first.get("adapter_kwargs", {}).get("base_url"),
                                        second.get("adapter_kwargs", {}).get("base_url")],
                    "native_audit_path": [first.get("adapter_kwargs", {}).get("audit_path"),
                                         second.get("adapter_kwargs", {}).get("audit_path")],
                },
            }
    request_hashes = {}
    for split, filename in (("test", "instances.json"), ("public-pilot", "public-pilot.json")):
        instances = read(primary_root / filename)
        require(len(instances) == (800 if split == "test" else 52), "Combined cohort count mismatch")
        tasks.validate_instances(instances)
        request_hashes[split] = sha([asdict(tasks.next_request(i, [])) for i in instances])
    code = {str(path.relative_to(REPO)): file_sha(path) for path in code_paths()}
    payload = {
        "version": "public-rq1-combined-v1", "created_at": time.time(),
        "roots": {name: {"path": str(path), "manifest_sha256": file_sha(path / "manifest.json")}
                  for name, path in roots.items()},
        "root_overrides": overrides, "supplemental_declared_root_overrides": declared_overrides,
        "outer_authorization_evidence": outer_evidence,
        "root_selection_policy": "Original seven primary models remain fixed. For the remaining "
            "models, attribute any existing primary query claims to primary only when both stage "
            "routes are bound by the supplied activated outer plan. Otherwise use the supplemental "
            "assignment. Apply this same frozen rule on each report; claims in both roots are an "
            "error. This is evidence attribution, not execution scheduling or outcome selection.",
        "authorization": authorization_pin,
        "supplemental_approval": {"path": str(approval), "sha256": file_sha(approval)},
        "data_sha256": {name: manifests["primary"]["files"][name] for name in data_files},
        "rendered_request_sha256": request_hashes, "shared_runtime_sha256": shared_code,
        "model_comparisons": comparisons, "code": code,
        "primary_authorization_evidence": "The older primary format has no per-query authorization hash. "
            "Supplemental approval and, when supplied, the activated outer plan provide authorization "
            "provenance. This is not a claim that primary execution lacked an outer authorization gate. "
            "No records are rewritten.",
        "raw_policy": "Read original roots in place; do not copy, rewrite, retry, or merge raw files.",
        "reporting_policy": "All 14 configurations and both complete scheduled cohorts remain visible. "
            "Active unsealed claims are in_progress, inactive ones interrupted; no accuracy selection.",
    }
    root.mkdir(parents=True)
    for name, expected in code.items():
        target = root / "frozen-source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write((REPO / name).read_bytes())
            stream.flush()
            os.fsync(stream.fileno())
        require(file_sha(target) == expected, "Reporter source changed during freeze")
        target.chmod(0o444)
    write(root / "combined-manifest.json", payload)
    write(root / "combined-manifest.sha256.json", {"sha256": file_sha(root / "combined-manifest.json")})
    return payload


def verify_combined(root):
    root = Path(root).resolve()
    manifest = read(root / "combined-manifest.json")
    require(file_sha(root / "combined-manifest.json") ==
            read(root / "combined-manifest.sha256.json")["sha256"], "Combined manifest drift")
    for name, digest in manifest["code"].items():
        require(file_sha(root / "frozen-source" / name) == digest and file_sha(REPO / name) == digest,
                "Use the sealed combined reporter; code drift")
    for key in ("authorization", "supplemental_approval"):
        pin = manifest[key]
        require(file_sha(pin["path"]) == pin["sha256"], f"{key} hash drift")
    if manifest.get("outer_authorization_evidence") is not None:
        for key in ("plan", "activation_proof"):
            pin = manifest["outer_authorization_evidence"][key]
            require(file_sha(pin["path"]) == pin["sha256"], f"Outer {key} hash drift")
    for entry in manifest["roots"].values():
        path = Path(entry["path"])
        require(file_sha(path / "manifest.json") == entry["manifest_sha256"], "Selected root drift")
        inspected = inspect_frozen_root(path)
        require(all(inspected["files"][name] == digest
                    for name, digest in manifest["data_sha256"].items()), "Selected dataset drift")
    return manifest


def lane_active(lane):
    path = lane / "lane.lock"
    if not path.exists():
        return False
    with path.open("r") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
    return False


def verify_wire(rows, request, spec, status):
    """Verify payload settings and raw returned identities without instantiating clients."""
    wires = [row for row in rows if row["event"] == "wire_request"]
    responses = [row for row in rows if row["event"] == "wire_response"]
    require(len(wires) <= 1 and len(responses) <= 1, "Multiple HTTP sends for one claimed query")
    for row in wires + responses:
        body = base64.b64decode(row["body_base64"], validate=True)
        require(body.decode("utf-8", errors="replace") == row["body"], "Raw byte/text mismatch")
    if wires:
        payload = json.loads(wires[0]["body"])
        if spec["provider"] == "vllm":
            expected = {
                "model": spec["model"],
                "messages": [{"role": "system", "content": ANSWER_ONLY_SYSTEM_MESSAGE
                              if spec["output_format"] == "answer_only" else SYSTEM_MESSAGE},
                             {"role": "user", "content": prompt_for(request)}],
                "temperature": spec["temperature"], "top_p": spec["top_p"],
                "max_tokens": spec["max_tokens"], "stream": False, "n": 1,
                "chat_template_kwargs": {"enable_thinking": spec["think"]},
                "structured_outputs": {"choice": [o.id for o in request.options]},
            }
            for name in ("seed", "top_k", "presence_penalty"):
                if spec.get(name) is not None:
                    expected[name] = spec[name]
            require(payload == expected, "Raw Qwen prompt/decoding payload drift")
        elif spec["provider"] == "typesafe":
            require(payload == {
                "state": request.state, "model": spec["model"], "questions": {
                    "decision": {"type": "choice", "instructions": request.question,
                                 "criteria": {o.id: o.description for o in request.options}}}},
                "Raw Jev prompt payload drift")
        elif spec["provider"] == "plugin" and "base_url" in spec["adapter_kwargs"]:
            require(payload == json.loads(canonical(asdict(request))), "Native request payload drift")
        else:
            require(payload.get("model") == spec["model"], "Scoring payload model drift")
    decisions = [row["response"] for row in rows if row["event"] == "decision_response"]
    require(len(decisions) <= 1, "Duplicate captured decision response")
    if status == "success":
        require(len(decisions) == 1, "Successful query lacks captured decision response")
        response = decisions[0]
        require(response["request_id"] == request.request_id and response["resolved_model"] == spec["model"]
                and response["selected_option_id"] in {o.id for o in request.options},
                "Captured response identity/choice drift")
        if spec["provider"] == "github_copilot":
            messages = [row for row in rows if row["event"] == "sdk_event"
                        and row["type"] == "assistant.message"]
            retries = [row for row in rows if row["event"] == "sdk_event"
                       and row["type"] == "assistant.turn_retry"]
            require(len(messages) == 1 and not retries and
                    messages[0]["data"]["content"] == response["raw_output"]["content"],
                    "SDK raw message/retry mismatch")
        else:
            require(len(wires) == len(responses) == 1, "Success lacks one raw request/response")
            wire = json.loads(responses[0]["body"])
            raw_output = response["raw_output"]
            if spec["provider"] == "plugin" and "base_url" not in spec["adapter_kwargs"]:
                raw_output = raw_output["provider"]
            require(wire == raw_output, "Captured provider body drift")
            if spec["provider"] == "plugin" and "base_url" in spec["adapter_kwargs"]:
                require(wire["status"] == "success" and wire["forward_count"] == 1,
                        "Native success must have exactly one forward")
    if status == "unsupported" and responses and spec["provider"] == "plugin":
        wire = json.loads(responses[0]["body"])
        require(wire["status"] == "unsupported" and wire["forward_count"] == 0,
                "Native capacity rejection must have zero forwards")
    return decisions


def replay_combined_query(directory, instance, spec, active, authorization_sha256=None):
    request = tasks.next_request(instance, [])
    base = {"instance_id": instance["id"], "task": instance["task"],
            "n": len(instance["state"]["nodes"]), "directory": str(directory),
            "status": "not_started", "correct": False, "receipt_sha256": None}
    if not directory.exists():
        return base
    receipt_path = directory / "receipt.json"
    if not receipt_path.exists():
        return {**base, "status": "in_progress" if active else "interrupted_unknown"}
    receipt_bytes = receipt_path.read_bytes()
    try:
        receipt = json.loads(receipt_bytes)
    except json.JSONDecodeError:
        if not active:
            raise
        return {**base, "status": "in_progress"}
    require(receipt == {name: file_sha(directory / name)
                        for name in ("intent.json", "raw.jsonl", "result.json")},
            "Combined replay receipt drift")
    intent, result = read(directory / "intent.json"), read(directory / "result.json")
    require(intent["request"] == json.loads(canonical(asdict(request))) and
            intent["request_sha256"] == result["request_sha256"] == prior.request_digest(request),
            "Combined replay request drift")
    if authorization_sha256 is not None:
        require(intent["authorization_sha256"] == authorization_sha256, "Intent authorization drift")
    raw, torn = prior.read_events(directory / "raw.jsonl")
    require(not torn, "Terminal raw capture is torn")
    status = result["status"]
    require(status in ("success", "failed", "unsupported"), "Unknown terminal query status")
    decisions = verify_wire(raw, request, spec, status)
    answer = independent_answer(instance["task"], instance["state"])
    require(answer == instance["private"]["answer"], "Combined independent oracle mismatch")
    correct = False
    if status == "success":
        choice = decisions[0]["selected_option_id"]
        require(choice == result["selected_option_id"], "Result/raw choice mismatch")
        correct = choice == str(answer)
    expected_score = {"correct": correct, "feasible": status == "success",
                      "status": "complete" if status == "success" else "incomplete"}
    require(canonical(result["score"]) == canonical(expected_score), "Independent score mismatch")
    receipt_digest = hashlib.sha256(receipt_bytes).hexdigest()
    require(file_sha(receipt_path) == receipt_digest, "Receipt changed during independent replay")
    return {**base, "status": status, "correct": correct, "receipt_sha256": receipt_digest,
            "diagnostic_code": result.get("diagnostic_code"), "error_type": result.get("error_type"),
            "latency_seconds": result["latency_seconds"],
            "raw_events": dict(Counter(row["event"] for row in raw))}


def combined_report(root, *, require_complete=False):
    root = Path(root).resolve()
    manifest = verify_combined(root)
    overrides = manifest["root_overrides"]
    outer = manifest.get("outer_authorization_evidence")
    route_evidence = None
    if outer is not None:
        overrides, route_evidence = observed_outer_routes(
            manifest["supplemental_declared_root_overrides"],
            Path(manifest["roots"]["primary"]["path"]), manifest["authorization"],
            outer["plan"]["path"], outer["activation_proof"]["path"])
    rows = []
    for split, filename in (("test", "instances.json"), ("public-pilot", "public-pilot.json")):
        for model in prior.PANEL:
            alias = overrides[model]
            selected = Path(manifest["roots"][alias]["path"])
            other_alias = "deployment" if alias == "primary" else "primary"
            other = Path(manifest["roots"][other_alias]["path"]) / "runs" / split / model
            require(not other.exists() or not any(p.is_dir() for p in other.iterdir()),
                    f"Cross-root duplicate model execution: {split}/{model}")
            source_manifest = read(selected / "manifest.json")
            spec = root_stage_models(selected, source_manifest)[split][model]
            lane = selected / "runs" / split / model
            active = lane_active(lane)
            observed_config, launches = None, []
            if (lane / "config.json").exists():
                observed_config = read(lane / "config.json")
                prior.validate_lane_config(model, observed_config)
                require(scientific_spec(observed_config) == scientific_spec(spec),
                        "Observed model behavior differs from frozen condition")
                spec = observed_config
                launches = [read(path) for path in sorted(lane.glob("launch-*.json"))]
                maximum = read(selected / "freeze-config.json")["concurrency"][model]
                require(all(launch["manifest_sha256"] == manifest["roots"][alias]["manifest_sha256"]
                            and launch["split"] == split and launch["at_most_once"] is True
                            and 1 <= launch["concurrency"] <= maximum for launch in launches),
                        "Launch identity/concurrency drift")
            if lane.exists() and any(p.is_dir() for p in lane.iterdir()):
                require(observed_config is not None and bool(launches),
                        "Claimed lane lacks configuration or launch provenance")
            instances = read(selected / filename)
            authorization = (manifest["authorization"]["sha256"]
                             if "authorization-binding.json" in source_manifest["files"] else None)
            details = [replay_combined_query(lane / i["id"], i, spec, active, authorization)
                       for i in instances]
            by_task = {}
            for task in TASKS:
                subset = [d for d in details if d["task"] == task]
                by_size = {}
                for n in sorted({d["n"] for d in subset}):
                    sized = [d for d in subset if d["n"] == n]
                    by_size[str(n)] = {"scheduled": len(sized), "correct": sum(d["correct"] for d in sized),
                                      "accuracy_all_scheduled": sum(d["correct"] for d in sized) / len(sized)}
                by_task[task] = {
                    "scheduled": len(subset), "correct": sum(d["correct"] for d in subset),
                    "statuses": dict(Counter(d["status"] for d in subset)), "by_size": by_size,
                    "accuracy_all_scheduled": sum(d["correct"] for d in subset) / len(subset),
                    "size_macro_accuracy_all_scheduled":
                        sum(v["accuracy_all_scheduled"] for v in by_size.values()) / len(by_size)}
            rows.append({
                "model": model, "split": split, "source_root": str(selected), "root_alias": alias,
                "source_manifest_sha256": manifest["roots"][alias]["manifest_sha256"],
                "observed_config_sha256": sha(observed_config) if observed_config else None,
                "scientific_spec_sha256": sha(scientific_spec(spec)),
                "observed_concurrency": sorted({launch["concurrency"] for launch in launches}),
                "lane_active_at_read": active, "scheduled": len(details),
                "statuses": dict(Counter(d["status"] for d in details)),
                "terminal": all(d["status"] in ("success", "failed", "unsupported") for d in details),
                "failure_codes": dict(Counter(d.get("diagnostic_code") or d.get("error_type") or "unspecified"
                                             for d in details if d["status"] == "failed")),
                "by_task": by_task, "instances": details,
            })
    complete = all(row["terminal"] for row in rows)
    require(not require_complete or complete, "Combined panel has nonterminal scheduled outcomes")
    return {"version": "public-rq1-combined-report-v1", "generated_at": time.time(),
            "combined_manifest_sha256": file_sha(root / "combined-manifest.json"),
            "authorization_sha256": manifest["authorization"]["sha256"],
            "supplemental_approval_sha256": manifest["supplemental_approval"]["sha256"],
            "root_overrides_at_read": overrides, "outer_route_evidence_at_read": route_evidence,
            "complete": complete, "scheduled_test": 11200, "scheduled_public_pilot": 728,
            "raw_copied": False, "inference_performed": False, "rows": rows}


def assert_unclaimed_scoring(primary_root, deployment_root):
    evidence = {}
    for alias, root in (("primary", Path(primary_root)), ("deployment", Path(deployment_root))):
        for split in ("public-pilot", "test"):
            lane = root / "runs" / split / "qwen4_token_scores"
            require(not lane_active(lane), f"Scoring worker remains active: {lane}")
            children = list(lane.iterdir()) if lane.exists() else []
            require(not any(p.is_dir() for p in children) and not list(lane.rglob("intent.json"))
                    and not list(lane.rglob("raw.jsonl")), f"Scoring query already claimed: {lane}")
            evidence[f"{alias}/{split}"] = {
                "path": str(lane.resolve()), "active": False, "query_directories": 0,
                "query_intents": 0, "raw_query_files": 0,
                "existing_files_sha256": {p.name: file_sha(p) for p in children if p.is_file()},
            }
    return evidence


def scoring_recovery_check(primary_root, deployment_root, approval):
    primary_root, deployment_root = Path(primary_root).resolve(), Path(deployment_root).resolve()
    primary_manifest, deployment_manifest = (
        inspect_frozen_root(p) for p in (primary_root, deployment_root))
    approved = read(approval)
    authorization_sha256 = verify_authorization(deployment_root)
    require(approved["authorization_sha256"] == authorization_sha256 and
            approved["automatic_retries"] is False and
            "qwen4_token_scores" in approved["untouched_deployment_lanes"],
            "Scoring recovery is not within the supplemental untouched-query scope")
    configs = root_stage_models(deployment_root, deployment_manifest)
    original = root_stage_models(primary_root, primary_manifest)
    for split in ("test", "public-pilot"):
        config = configs[split]["qwen4_token_scores"]
        prior.validate_lane_config("qwen4_token_scores", config)
        require(scientific_spec(config) == scientific_spec(original[split]["qwen4_token_scores"]),
                "Scoring recovery changes adapter/model behavior")
    evidence = assert_unclaimed_scoring(primary_root, deployment_root)
    return {
        "zero_query_recovery_check_passed": True, "observed_at": time.time(), "lanes": evidence,
        "checker_source": {"path": str(Path(__file__).resolve()), "sha256": file_sha(__file__)},
        "authorization_sha256": authorization_sha256,
        "supplemental_approval": {"path": str(Path(approval).resolve()), "sha256": file_sha(approval)},
        "primary_manifest_sha256": file_sha(primary_root / "manifest.json"),
        "deployment_manifest_sha256": file_sha(deployment_root / "manifest.json"),
        "historical_interpreter": "/fastdata/xianya/vllm-venv/bin/python",
        "execution_root": str(deployment_root), "model": "qwen4_token_scores", "concurrency": 1,
        "pilot_queries": 52, "test_queries": 800, "extra_retry_calls": 0,
        "failure_records_preserved": True, "adapter_or_frozen_source_edited": False,
        "inference_performed": False,
        "scheduler_requirement": "Run this check while holding the shared Qwen4 resource lease. "
            "Pin this fresh receipt in a separate recovery ledger before launching the historical "
            "interpreter on the deployment root. Do not reuse this observation after releasing "
            "the lease. Any existing/new query claim prohibits resubmission. Pilot gate remains required.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "validate", "run", "report", "check-lane",
                                          "combine", "combined-report", "scoring-recovery-check"))
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--historical-root", type=Path, default=HISTORICAL)
    parser.add_argument("--authorization", type=Path, default=AUTHORIZATION)
    parser.add_argument("--pilot-model-config", type=Path)
    parser.add_argument("--test-model-config", type=Path)
    parser.add_argument("--primary-root", type=Path)
    parser.add_argument("--deployment-root", type=Path)
    parser.add_argument("--approval", type=Path)
    parser.add_argument("--outer-plan", type=Path)
    parser.add_argument("--activation-proof", type=Path)
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--model", choices=prior.PANEL)
    parser.add_argument("--split", choices=("test", "public-pilot"), default="test")
    parser.add_argument("--concurrency", type=int)
    parser.add_argument("--base-url")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "combine":
        require(all(p is not None for p in (args.primary_root, args.deployment_root, args.approval)),
                "--primary-root, --deployment-root and --approval required")
        combine(args.root, args.primary_root, args.deployment_root, args.approval, args.authorization,
                args.outer_plan, args.activation_proof)
        print(canonical({"combined_manifest": str(args.root / "combined-manifest.json")}))
    elif args.command == "freeze":
        freeze(args.root, data_dir=args.data_dir or REPO / "data", historical=args.historical_root,
               authorization=args.authorization, pilot_model_config=args.pilot_model_config,
               test_model_config=args.test_model_config)
        print(canonical({"ready": True, "root": str(args.root)}))
    elif args.command == "validate":
        print(canonical(validate_frozen(args.root, data_dir=args.data_dir)))
    elif args.command == "run":
        require(args.model is not None, "--model required")
        asyncio.run(run(args.root, args.model, split=args.split, concurrency=args.concurrency,
                        base_url=args.base_url))
    else:
        if args.command == "scoring-recovery-check":
            require(args.primary_root is not None and args.approval is not None,
                    "--primary-root and --approval required")
        result = (check_lane(args.root, args.model, args.split) if args.command == "check-lane" else
                  combined_report(args.root, require_complete=args.require_complete)
                  if args.command == "combined-report" else
                  scoring_recovery_check(args.primary_root, args.root, args.approval)
                  if args.command == "scoring-recovery-check" else report(args.root))
        if args.output:
            write(args.output, result)
        else:
            print(canonical(result))


if __name__ == "__main__":
    main()
