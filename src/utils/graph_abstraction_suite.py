#!/usr/bin/env python3
"""Frozen four-task proposal experiment; fsynced at-most-once model attempts.

freeze --root DIR --instances JSON --model-config JSON --models NAME [NAME ...]
run --root DIR --model-config JSON --models NAME [NAME ...]
report --root DIR

Model JSON maps panel IDs to ModelConfig dictionaries, or to pinned plugin
specifications (see protocol.json). Explicit inputs accept a nonempty subset
of the four construction tasks and public TSP/MaxCut, with scoring references.
Construction records use extended_tasks.validate_instances; public records use
public_tasks.validate_instance. IDs must be safe single directory names.
Optional public dataset_id values must be nonempty strings; when omitted,
reporting groups by instance ID.
Persist a successful unittest log at DIR/tests.log before freezing. Run the
frozen-source copy after freezing. All supplied model configurations are bound.

Without --instances, freeze reproduces the historical 211-case cohort and
requires the original local input/ledger archives; it is not a clean-clone mode.
Explicit inputs are marked user-supplied, never historical-paper replication.
Scoring/native plugins are user-owned, SHA-256-pinned BaseDecisionClient
factories with explicit capability limits. No SemIf/upstream implementation,
weights, tokenizer, or local proposal_scoring_client adapter is distributed
or included in source snapshots. Supply licensed dependencies separately;
the adapter hash alone does not establish their provenance or equivalence.
For example, a scoring config entry (replace the path/hash/model) is:
{"qwen4_token_scores": {"provider": "plugin", "model": "exact-response-model",
 "timeout_seconds": 180, "adapter_file": "/absolute/path/client.py",
 "adapter_sha256": "<64 lowercase hex digits>", "factory": "create_client",
 "adapter_kwargs": {}}}
The factory receives adapter_kwargs; credentials belong in the environment.
Use --concurrency 1 for plugins. Plugin files/dependencies are external and
must remain available at their pinned paths for inference (not for reporting).

Original main-panel TSP records are Manhattan, not Euclidean. Their metric,
edges and references are preserved. The constructed adapter also explicitly
supports unrounded Euclidean distances; it never applies TSPLIB rounding.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import asdict, replace
import fcntl
from fractions import Fraction
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import re
import statistics
import sys
import time
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.dont_write_bytecode = True

from src.benchmark import extended_tasks as extended
from src.benchmark import graph_abstraction as graph
from src.benchmark import public_tasks
from src.benchmark import tsp_abstraction as tsp
from src.benchmark.config import ModelConfig
from src.clients.base import (
    BaseDecisionClient, ClientTimeoutError, DecisionOption, DecisionRequest,
    DecisionResponse, UnsupportedRequestError,
)
from src.utils import public_graph_suite as public

audit = public.audit
require, digest, read_json, write_json = (
    public.require, public.digest, public.read_json, public.write_json,
)
SEED = 20261003
RANDOM_SEEDS = (1101, 1102, 1103, 1104, 1105)
EPISODE_SECONDS = 900
ARMS = ("A", "B", "C")
TASKS = ("maxcut_construct", "tsp_construct", "lt_influence_construct",
         "community_bipartition_construct")
PANEL = (
    "jev_action", "qwen4_grammar", "qwen9_grammar", "qwen27_grammar",
    "qwen4_token_scores", "decider", "kev", "laya",
    "gpt54_default_reasoning", "gpt6astra_default_reasoning",
    "qwen38_27b_bf16", "qwen25_72b_bf16", "qwen08", "qwen2",
)
SYNTHETIC = Path("output/experiments/synthetic/development/"
                 "paper-structural-challenge-20260924-v1/instances.jsonl")
MAIN_SCORES = Path("output/experiments/synthetic/matrix/explicit-actions-matrix-20260924-v1/"
                   "structural_challenge.jev_action/scores.jsonl")
ORIGINAL_PUBLIC = Path("output/archive/portable/graphbench-portable-release-20260924-v2/"
                       "public/structural_challenge.jsonl")
ORIGINAL_REFERENCES = Path("output/archive/portable/graphbench-portable-release-20260924-v2/"
                           "references/structural_challenge.jsonl")
PUBLIC = Path("output/experiments/public/primary/public-graph-main-20260929-v2/instances.json")
MATRIX = Path("output/experiments/synthetic/matrix/"
              "twelve-model-publication-20260927-v2/benchmark.json")
VERSION = "graph-proposals-20261003-v2"
RAW_RESPONSE_VERSION = 1
CAPTURE_ERRORS = ("non_json_response", "invalid_response_type", "credential_material")
SECRET_FIELDS = frozenset({
    "apikey", "authorization", "proxyauthorization", "clientsecret", "accesstoken",
    "refreshtoken", "githubtoken", "password", "secretkey",
})
NATIVE_MODELS = ("Mapika/decider-2b", "jaredpalmer/kev-4b",
                 "convaiinnovations/laya-typed-decisions")
QWEN_MODELS = {
    "qwen08": "Qwen3.5-0.8B", "qwen2": "Qwen3.5-2B",
    "qwen4_grammar": "Qwen3.5-4B", "qwen9_grammar": "Qwen3.5-9B",
    "qwen27_grammar": "Qwen3.5-27B", "qwen38_27b_bf16": "Qwen3.8-27B",
    "qwen25_72b_bf16": "Qwen2.5-72B",
}


def atomic_json(path, value):
    """Publish only complete JSON and persist the directory entry as well."""
    path = Path(path)
    pending = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    write_json(pending, value)
    os.replace(pending, path)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def check_hold(root):
    path = Path(root) / "input-scope-hold.json"
    if path.exists():
        require(read_json(path).get("publication_allowed") is True,
                "Experiment input-scope hold is active.")


def original_instances():
    """Match both main-ledger identities and complete original private records."""
    original = [i for i in audit.read_rows(REPO / SYNTHETIC) if i["task"] in TASKS]
    require(Counter(i["task"] for i in original) == Counter({t: 40 for t in TASKS}),
            "Expected original 40 cases per construction task.")
    scores = [i for i in audit.read_rows(REPO / MAIN_SCORES) if i["task"] in TASKS]
    ids = {i["id"] for i in original}
    require(len(original) == len(ids) == len(scores) == 160 and
            ids == {i["instance_id"] for i in scores}, "Original main-panel IDs differ.")
    public_records = {i["id"]: i for i in audit.read_rows(REPO / ORIGINAL_PUBLIC)}
    private_records = {i["id"]: i["private"] for i in audit.read_rows(REPO / ORIGINAL_REFERENCES)}
    for instance in original:
        require(instance == {**public_records[instance["id"]],
                             "private": private_records[instance["id"]]},
                "Original public/private record mismatch.")
    validation = extended.validate_instances(original)
    return original, {
        **validation, "main_ledger_id_matches": len(ids),
        "original_portable_full_record_matches": len(ids),
        "private_reference_validation": "Independent exact optimization and baseline recomputation.",
        "tsp_metrics": dict(Counter(i["state"]["metric"] for i in original
                                   if i["task"] == "tsp_construct")),
        "instance_ids": sorted(ids),
    }


def budget(instance):
    if instance["task"].endswith("_public"):
        return public_tasks.query_budget(instance)
    return instance["query_budget"]


def rules(instance):
    return tsp.RULES if instance["task"].startswith("tsp_") else graph.rules(instance["task"])


def constructed_distances(instance):
    state = instance["state"]
    points, metric = state["points"], state["metric"]
    require(metric in ("Manhattan", "Euclidean", "euclidean_unrounded"),
            "Unsupported constructed TSP metric; no implicit rounding.")
    if metric == "Manhattan":
        extended._validate_state("tsp_construct", state)
        return extended._distances(state)
    require(len(points) == len(state["nodes"]) and all(
        len(p) == 2 and all(type(c) in (int, float) and math.isfinite(c) for c in p)
        for p in points), "Invalid Euclidean coordinates.")
    return [[math.hypot(x - a, y - b) for a, b in points] for x, y in points]


def prepare(instance):
    if instance["task"] == "tsp_public":
        return tsp.prepare(instance)
    if instance["task"] == "tsp_construct":
        return SimpleNamespace(instance={k: deepcopy(instance[k]) for k in ("id", "task", "state")},
                               distances=constructed_distances(instance))
    return graph.prepare(instance)


def step(prepared, decisions, arm, seed=SEED):
    instance = prepared.instance
    task = instance["task"]
    require(arm in ARMS, "Unknown arm.")
    if not task.startswith("tsp_"):
        return graph.step(prepared, decisions, arm, seed)
    if task == "tsp_public" and arm != "A":
        payload = tsp.step(prepared, decisions, arm, seed)
        if payload is None:
            return None
        return {
            **{k: v for k, v in payload.items() if k not in ("forced_city", "unvisited_count")},
            "forced_action": (None if payload["forced_city"] is None
                              else str(payload["forced_city"])),
            "legal_action_count": payload["unvisited_count"],
            "candidates": [{**row, "action": str(row["city"])} for row in payload["candidates"]],
            "rule_proposals": {k: str(v) for k, v in payload["rule_proposals"].items()},
        }
    request = (public_tasks.next_request(instance, decisions) if task == "tsp_public"
               else extended.next_request(instance, decisions))
    if request is None:
        return None
    if task == "tsp_construct" and instance["state"]["metric"] != "Manhattan":
        request = replace(request, question=request.question.replace("Manhattan", "unrounded Euclidean"))
    matrix = prepared.distances
    current = int(decisions[-1]) if decisions else 0
    legal = [o.id for o in request.options]
    unseen = list(map(int, legal))
    features = {v: (matrix[current][v], min(matrix[v][w] for w in unseen if w != v),
                    matrix[v][0]) for v in unseen}
    values = {v: (e, e + h, 2 * e - r, e - h) for v, (e, h, r) in features.items()}
    proposals = {rule: str(min(unseen, key=lambda v: (values[v][index], v)))
                 for index, rule in enumerate(tsp.RULES)}
    actions = legal.copy() if arm == "A" else sorted(set(proposals.values()), key=int)
    if arm != "A":
        order = hashlib.sha256(f"{seed}:{instance['id']}:{','.join(decisions)}".encode()).hexdigest()
        random.Random(order).shuffle(actions)
    candidates = [{
        "option_id": action if arm == "A" else str(index), "action": action,
        "edge": features[int(action)][0], "nearest_other_unvisited": features[int(action)][1],
        "return_to_start": features[int(action)][2],
        "rules": [r for r in tsp.RULES if proposals[r] == action],
    } for index, action in enumerate(actions)]
    payload = {
        "candidates": candidates, "rule_proposals": proposals,
        "legal_action_count": len(legal), "distinct_candidate_count": len(actions),
        "candidate_coverage": len(actions) / len(legal),
        "forced_action": actions[0] if len(actions) == 1 else None, "request": None,
    }
    if len(actions) == 1:
        return payload
    if arm == "A":
        payload["request"] = request
        return payload
    state = deepcopy(request.state)
    state["candidates"] = [{k: v for k, v in row.items() if k != "rules"} for row in candidates]
    state["feature_definitions"] = (
        "edge=d(current,v); nearest_other_unvisited=min(d(v,w) for other unvisited w); "
        "return_to_start=d(v,0). Distances follow the supplied metric without rounding."
    )
    question = request.question.rsplit("Return only", 1)[0] + (
        "Append exactly one displayed candidate action, without insertion, reordering "
        "or executing a completion. Return only its option_id, not its vertex ID."
    )
    if state["metric"] != "Manhattan":
        question = question.replace("Manhattan", "unrounded Euclidean")
    if arm == "C":
        state["rules"] = {"definitions": tsp.DEFINITIONS, "formulas": tsp.FORMULAS,
                          "tie_break": "lowest numeric vertex ID"}
        state["rule_groups"] = [{k: row[k] for k in ("option_id", "action", "rules")}
                                for row in candidates]
        state["rule_proposals"] = proposals.copy()
    payload["request"] = DecisionRequest(
        request_id=f"constructed_tsp:{arm}:{len(decisions)}", state=state, question=question,
        options=tuple(DecisionOption(
            row["option_id"], (", ".join(row["rules"]) + "; " if arm == "C" else "") +
            f"Append city {row['action']}") for row in candidates),
    )
    return payload


def independent_objective(instance, decisions):
    """Separate objective implementation, including rational modularity and LT."""
    task, state = instance["task"], instance["state"]
    require(len(decisions) == budget(instance), "Independent score needs a complete history.")
    if task.startswith("tsp_"):
        tour = [0, *map(int, decisions)]
        require(len(set(tour)) == len(tour), "Repeated city.")
        tour += [v for v in state["nodes"] if v not in tour] + [0]
        if task == "tsp_public":
            points = state["coordinates"]
            def distance(u, v):
                dx, dy = float(points[u][0]) - float(points[v][0]), float(points[u][1]) - float(points[v][1])
                return int(math.sqrt(dx * dx + dy * dy) + 0.5)
        else:
            points = state["points"]
            def distance(u, v):
                x, y = points[u]
                a, b = points[v]
                return (abs(x - a) + abs(y - b) if state["metric"] == "Manhattan"
                        else math.sqrt((x - a) ** 2 + (y - b) ** 2))
        return sum(distance(u, v) for u, v in zip(tour, tour[1:]))
    if task == "lt_influence_construct":
        active = set(map(int, decisions))
        while True:
            nxt = active.copy()
            for v in state["nodes"]:
                weights = [w for w in state["weights"] if w["target"] == v]
                mass = sum((Fraction(w["numerator"], w["denominator"])
                            for w in weights if w["source"] in active), Fraction(0))
                if mass >= Fraction(1, 2):
                    nxt.add(v)
            if nxt == active:
                return len(active)
            active = nxt
    sides = ["A", *decisions]
    if task != "community_bipartition_construct":
        return sum((e[2] if len(e) == 3 else 1) for e in state["edges"] if sides[e[0]] != sides[e[1]])
    m = len(state["edges"])
    result = Fraction(0)
    for side in ("A", "B"):
        internal = sum(sides[u] == sides[v] == side for u, v in state["edges"])
        volume = sum((sides[u] == side) + (sides[v] == side) for u, v in state["edges"])
        result += Fraction(internal, m) - Fraction(volume, 2 * m) ** 2
    return float(result)


def score(instance, decisions):
    if (instance["task"] == "tsp_construct" and instance["state"]["metric"] != "Manhattan"):
        # Use the original history validator, but never its Manhattan-edge scorer.
        extended._check_history(instance, decisions)
        feasible = len(decisions) == budget(instance)
        objective = independent_objective(instance, decisions) if feasible else None
        optimum = instance["private"]["objective"]
        gap = abs(objective - optimum) if feasible else None
        result = {"feasible": feasible, "objective": objective, "absolute_gap": gap,
                  "optimum": optimum, "status": "complete" if feasible else "incomplete"}
    else:
        result = (public_tasks.score(instance, decisions) if instance["task"].endswith("_public")
                  else extended.score(instance, decisions))
    if result["feasible"]:
        expected = independent_objective(instance, decisions)
        require(math.isclose(expected, result["objective"], rel_tol=1e-12, abs_tol=1e-12),
                "Independent objective mismatch.")
    else:
        require(result["objective"] is None, "Incomplete episode has an objective.")
    return result


def source_hashes():
    paths = [*(REPO / "src").rglob("*.py"), REPO / "tests/test_graph_abstraction_suite.py",
             REPO / "tests/test_graph_abstraction.py", REPO / "tests/test_tsp_abstraction.py"]
    return {str(p.relative_to(REPO)): digest(p) for p in sorted(paths)
            if p != REPO / "src/utils/proposal_scoring_client.py"}


def supplied_inputs(instances_path, config_path, models):
    """Validate explicit inputs without reading any historical experiment files."""
    require(config_path is not None and models, "Explicit inputs require --model-config and --models.")
    require(len(set(models)) == len(models) and set(models) <= set(PANEL),
            "Choose distinct supported panel IDs.")
    instances_path, config_path = Path(instances_path).resolve(), Path(config_path).resolve()
    before = {str(p): digest(p) for p in (instances_path, config_path)}
    instances, configs = read_json(instances_path), read_json(config_path)
    require(type(instances) is list and bool(instances), "Instances must be a nonempty JSON list.")
    require(type(configs) is dict and set(configs) == set(models),
            "Model config must contain exactly the selected panel IDs.")
    ids = set()
    for instance in instances:
        require(type(instance) is dict and type(instance.get("id")) is str and
                re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", instance["id"]) is not None,
                "Instance IDs must be safe single directory names.")
        require(instance["id"] not in ids, "Duplicate instance ID.")
        ids.add(instance["id"])
        require(instance.get("task") in (*TASKS, "tsp_public", "maxcut_public"),
                "Unsupported proposal task.")
    constructed = [i for i in instances if i["task"] in TASKS]
    validation = extended.validate_instances(constructed)
    for instance in instances:
        if instance["task"].endswith("_public"):
            public_tasks.validate_instance(instance)
            require("dataset_id" not in instance or
                    (type(instance["dataset_id"]) is str and bool(instance["dataset_id"].strip())),
                    "Public dataset_id must be a nonempty string when present.")
    for name, spec in configs.items():
        validate_lane_config(name, spec)
    require(not contains_credential_material(configs), "Credentials must not be embedded in model config.")
    require(before == {str(p): digest(p) for p in (instances_path, config_path)},
            "Supplied inputs changed during validation.")
    validation.update(
        input_scope="user_supplied_not_historical_replication",
        public_conditions=len(instances) - len(constructed),
        instance_ids=sorted(ids),
    )
    return instances, validation, [{"id": name} for name in models], configs, before


def baselines(instances):
    rows = []
    for instance in instances:
        prepared = prepare(instance)
        for method in (*rules(instance), "random_candidate", "classical"):
            for seed in (RANDOM_SEEDS if method == "random_candidate" else (SEED,)):
                started = time.perf_counter()
                decisions, forced, counts = [], 0, []
                if method == "classical":
                    if instance["task"].endswith("_public"):
                        classical = ("nearest_neighbor_two_opt" if instance["task"] == "tsp_public"
                                     else "greedy_single_flip")
                        decisions = public_tasks.baseline(instance, classical, seed=seed)
                    else:
                        classical = instance["private"]["baseline"]["method"]
                        solution = extended._baseline(instance["task"], instance["state"])
                        decisions = ([str(v) for v in solution[1:-2]]
                                     if instance["task"] == "tsp_construct" else
                                     [str(v) for v in solution] if instance["task"] == "lt_influence_construct"
                                     else solution[1:])
                else:
                    rng = random.Random(seed)
                    while (payload := step(prepared, decisions, "B")) is not None:
                        action = (rng.choice(payload["candidates"])["action"] if method == "random_candidate"
                                  else payload["rule_proposals"][method])
                        counts.append(payload["distinct_candidate_count"])
                        forced += payload["forced_action"] is not None
                        decisions.append(action)
                rows.append({
                    "instance_id": instance["id"], "task": instance["task"], "method": method,
                    "classical_method": classical if method == "classical" else None,
                    "seed": seed, "decisions": decisions, "forced_steps": forced,
                    "candidate_counts": counts, "wall_seconds": time.perf_counter() - started,
                    **score(instance, decisions),
                })
                require(rows[-1]["feasible"], "Incomplete baseline.")
        print(json.dumps({"baseline_instance": instance["id"], "completed": len(rows)}), flush=True)
    return rows


def freeze(root, *, instances_path=None, config_path=None, models=None):
    root = Path(root).resolve()
    check_hold(root)
    require(not (root / "protocol.json").exists(), "Protocol already frozen.")
    root.mkdir(parents=True, exist_ok=True)
    require(not (root / "instances.json").exists(), "Partial freeze exists; do not overwrite.")
    tests = root / "tests.log"
    require(tests.is_file() and re.search(r"Ran \d+ tests in .*?\n\nOK\s*$", tests.read_text()) is not None,
            "Persist the successful unittest log before freezing.")
    if instances_path is not None:
        instances, validation, panel, cloud_configs, provenance = supplied_inputs(
            instances_path, config_path, models)
        version = VERSION + "-user-inputs"
    else:
        require(config_path is None and models is None,
                "--model-config/--models at freeze require --instances.")
        original, validation = original_instances()
        public_instances = read_json(REPO / PUBLIC)
        require(Counter(i["task"] for i in public_instances) ==
                Counter(tsp_public=21, maxcut_public=30), "Expected public 7+10 graphs x3.")
        for instance in public_instances:
            public_tasks.validate_instance(instance)
        instances = [*original, *public_instances]
        require(len({i["id"] for i in instances}) == 211, "Duplicate or missing cases.")
        panel = read_json(REPO / MATRIX)["models"]
        require(tuple(m["id"] for m in panel) == PANEL[:12], "Original model panel drift.")
        panel = [*panel, {"id": "qwen08"}, {"id": "qwen2"}]
        cloud = public.shared.model_configs(models=("jev", "gpt54", "gpt6astra"))
        cloud_configs = {
            name: asdict(cloud[old]) for name, old in
            (("jev_action", "jev"), ("gpt54_default_reasoning", "gpt54"),
             ("gpt6astra_default_reasoning", "gpt6astra"))
        }
        provenance = {str(p): digest(REPO / p) for p in
                      (SYNTHETIC, MAIN_SCORES, ORIGINAL_PUBLIC, ORIGINAL_REFERENCES, PUBLIC, MATRIX)}
        version = VERSION
    model_names = [m["id"] for m in panel]
    synthetic_count = sum(i["task"] in TASKS for i in instances)
    jobs = [{"instance_id": i["id"], "arm": arm} for i in instances for arm in ARMS]
    random.Random(SEED).shuffle(jobs)
    write_json(root / "instances.json", instances)
    write_json(root / "input-validation.json", validation)
    write_json(root / "schedule.json", jobs)
    write_json(root / "baselines.json", baselines(instances))
    write_json(root / "cloud-models.json", cloud_configs)
    hashes = source_hashes()
    snapshot = root / "frozen-source"
    snapshot.mkdir()
    for name, expected in hashes.items():
        target = snapshot / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write((REPO / name).read_bytes())
            stream.flush()
            os.fsync(stream.fileno())
        require(digest(target) == expected, "Source drift during freeze.")
        target.chmod(0o444)
    protocol = {
        "version": version, "created_at": public.now(),
        "input_scope": ("user_supplied_not_historical_replication" if instances_path is not None
                        else "historical_main_panel"),
        "seed": SEED, "arms": list(ARMS), "panel": panel,
        "models": model_names, "cases": len(instances), "episodes_per_model": len(jobs),
        "synthetic_cases": synthetic_count, "public_conditions": len(instances) - synthetic_count,
        "max_calls_per_model": 3 * sum(budget(i) for i in instances),
        "episode_seconds": EPISODE_SECONDS, "random_seeds": list(RANDOM_SEEDS),
        "arm_definitions": {
            "A": "All original legal actions, original task request; no candidate restriction.",
            "B": "Anonymous deduplicated rule proposals, numeric features.",
            "C": "Same proposals/order/features at matched histories, plus rule information.",
        },
        "partition_contract": "Fixed next vertex, exactly A/B; report singleton collapse, never reorder vertices.",
        "tsp_metric": ("Supplied construction records require validated Manhattan distances; "
                       "public TSP uses EUC_2D." if instances_path is not None else
                       "Original main-panel 40-case TSP bank is Manhattan. Preserve it exactly; "
                       "Euclidean-unrounded supported explicitly, not substituted. Public TSP uses EUC_2D."),
        "fresh_run": ("Fresh user-supplied cohort, not a replication of the paper cohort."
                      if instances_path is not None else
                      "All model episodes, including A, are fresh. No v1 episodes or scores reused. "
                      "v1 remains a quarantined confirmatory-bank pilot. Pre-raw-capture v2 episodes "
                      "are also quarantined and never reused or retrofitted."),
        "failures": "No retries. Durable call intent before transport. On resume seal started episodes "
                    "as interrupted, never repeat a call. Untouched episodes may start. Fatal errors stop lane.",
        "support": "Unsupported capacities are separate terminal outcomes, not zero performance; "
                   "every scheduled episode remains in coverage denominator. Native adapter enforces "
                   "full rendered context/head/option caps without truncation.",
        "call_accounting": "model_calls counts entered client.predict invocations, not native forwards; "
                           "renderer/context rejection can precede any forward. A pending durable intent "
                           "has model_calls=null, confirmed_model_calls is a lower bound. No false zero.",
        "model_binding": "Each lane config is fsynced and hash-bound before initialize; immutable on resume. "
                         "Explicit-input mode binds every supplied config; historical mode binds cloud "
                         "configs and accepts endpoint configs from the deployment owner.",
        "plugin_interface": {
            "provider": "plugin", "model": "exact response model identity", "timeout_seconds": 180,
            "adapter_file": "absolute path to audited BaseDecisionClient factory module",
            "adapter_sha256": "sha256 of module", "factory": "create_client",
            "adapter_kwargs": "public settings only; credentials via existing environment",
            "contract": "factory(**adapter_kwargs) returns BaseDecisionClient; model is exact response identity "
                        "and timeout_seconds is the runner watchdog (native adapter keeps its own timeout); "
                        "capabilities.max_options/max_context_tokens mandatory. predict must reject all "
                        "native truncation and unsupported subcomponent caps with UnsupportedRequestError.",
        },
        "native_audit": "Decider/Kev/Laya require adapter_kwargs.audit_path under this root. The adapter "
                        "fsyncs request intent and full raw_wire with actual forward_count before returning. "
                        "The runner copies those exact rows into the fsynced step ledger and replays "
                        "request hashes, native actions and forward counts. Missing evidence fails closed "
                        "without retry. Transport failures retain unknown forwards, never false zero.",
        "raw_response_capture": {
            "schema_version": RAW_RESPONSE_VERSION, "file": "raw-responses.jsonl",
            "policy": "Every returned DecisionResponse is JSON-serialized and fsynced BEFORE allowlisted "
                      "response_record parsing. Preserve raw_output, probabilities, probability_kind, "
                      "confidence, usage, request_id, selected_option_id, resolved_model and status. "
                      "No client configuration, headers or credentials are read into this sidecar. "
                      "Non-JSON/non-DecisionResponse returns are explicit fatal capture failures, never "
                      "silently omitted. Transport exceptions have no returned response. All episode "
                      "seals include the raw sidecar hash; replay binds responses to request hashes and "
                      "successful selected/model/usage identities. Pending orphan responses remain "
                      "unexecuted and are not retried.",
        },
        "analysis": "No imputation. Report coverage, support, failure/timeout, singleton frequency, "
                    "candidate coverage and feasible-only performance. Public inference unit is original "
                    "graph, average its supplied conditions (three in the historical cohort); "
                    "synthetic unit is original case.",
        "source_provenance": provenance,
        "source_sha256": hashes,
        "inputs_sha256": {name: digest(root / name) for name in
                         ("instances.json", "input-validation.json", "schedule.json",
                          "baselines.json", "cloud-models.json")},
        "environment": audit.environment_versions(),
    }
    for name in ("launch-cloud.sh", "launch-monitor.sh", "tests.log"):
        if (root / name).exists():
            protocol["inputs_sha256"][name] = digest(root / name)
    if (root / "superseded-protocols.json").exists():
        protocol["inputs_sha256"]["superseded-protocols.json"] = digest(root / "superseded-protocols.json")
    require(source_hashes() == hashes, "Source changed while freezing.")
    write_json(root / "protocol.json", protocol)
    write_json(root / "protocol.sha256", digest(root / "protocol.json"))
    for name in (*protocol["inputs_sha256"], "protocol.json", "protocol.sha256"):
        (root / name).chmod(0o444)
    load(root)
    atomic_json(root / "freeze-ready.json", {
        "freeze_test_ready": True, "created_at": public.now(),
        "protocol_sha256": digest(root / "protocol.json"), "version": version,
        "input_validation_sha256": digest(root / "input-validation.json"),
        "tests_sha256": digest(root / "tests.log") if (root / "tests.log").exists() else None,
        "raw_response_capture_schema_version": RAW_RESPONSE_VERSION,
        "cases": len(instances), "episodes_per_model": len(jobs),
        "scheduled_episodes": len(model_names) * len(jobs),
    })
    print(json.dumps({"status": "frozen", "root": str(root), "models": len(model_names),
                      "cases": len(instances), "episodes_per_model": len(jobs)}), flush=True)


def load(root):
    root = Path(root).resolve()
    check_hold(root)
    require(digest(root / "protocol.json") == read_json(root / "protocol.sha256"), "Protocol seal mismatch.")
    protocol = read_json(root / "protocol.json")
    for name, expected in protocol["inputs_sha256"].items():
        require(digest(root / name) == expected, "Frozen input drift.")
    for name, expected in protocol["source_sha256"].items():
        require(digest(root / "frozen-source" / name) == expected, "Frozen code drift.")
        require(digest(REPO / name) == expected, "Run the frozen-source entrypoint; executing code drift.")
    return root, protocol, {i["id"]: i for i in read_json(root / "instances.json")}


def validate_config(spec, *, verify_adapter=True):
    require(type(spec) is dict, "Model config must be an object.")
    if spec.get("provider") != "plugin":
        return ModelConfig(**spec)
    require(set(spec) == {"provider", "model", "timeout_seconds", "adapter_file",
                         "adapter_sha256", "factory", "adapter_kwargs"}, "Invalid plugin specification.")
    require(isinstance(spec["model"], str) and spec["model"].strip(), "Missing plugin model ID.")
    require(type(spec["timeout_seconds"]) in (int, float) and
            math.isfinite(spec["timeout_seconds"]) and spec["timeout_seconds"] > 0, "Invalid timeout.")
    require(type(spec["adapter_kwargs"]) is dict, "Invalid adapter kwargs.")
    require(not re.search(r'(secret|password|api[_-]?key|authorization|access_token)',
                          json.dumps(spec["adapter_kwargs"]), re.I), "Secrets forbidden in model JSON.")
    require(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", spec["factory"]) is not None, "Invalid factory.")
    path = Path(spec["adapter_file"])
    require(path.is_absolute() and re.fullmatch(r"[0-9a-f]{64}", spec["adapter_sha256"]) is not None,
            "Invalid plugin path/hash.")
    if verify_adapter:
        require(path.is_file() and not path.is_symlink(), "Invalid plugin path.")
        require(digest(path) == spec["adapter_sha256"], "Plugin hash mismatch.")
    return SimpleNamespace(**spec)


def make_client(spec):
    config = validate_config(spec)
    if config.provider != "plugin":
        return public.create_public_client(config.provider, **config.client_kwargs())
    module_spec = importlib.util.spec_from_file_location("graph_proposal_plugin", config.adapter_file)
    require(module_spec is not None and module_spec.loader is not None, "Cannot import plugin.")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    client = getattr(module, config.factory)(**config.adapter_kwargs)
    require(isinstance(client, BaseDecisionClient), "Plugin must implement BaseDecisionClient.")
    require(client.capabilities.max_options is not None and
            client.capabilities.max_context_tokens is not None, "Native caps must be explicit.")
    return client


def validate_lane_config(name, spec):
    config = validate_config(spec)
    if name in ("qwen4_token_scores", "decider", "kev", "laya"):
        require(config.provider == "plugin", "Native/candidate scoring lanes require a pinned native adapter.")
        if name in ("decider", "kev", "laya"):
            require(config.model == dict(zip(("decider", "kev", "laya"), NATIVE_MODELS))[name] and
                    isinstance(config.adapter_kwargs.get("audit_path"), str) and
                    Path(config.adapter_kwargs["audit_path"]).is_absolute(),
                    "Native v2 lanes require a pinned audited adapter and absolute audit_path.")
    elif name.startswith("qwen"):
        require(config.provider == "vllm" and config.think is False and
                config.constrain_choices is True and config.output_format == "answer_only" and
                config.max_tokens == 64 and config.temperature == 0,
                "Qwen generation panel requires no-thinking greedy choice grammar and 64 tokens.")
        require(config.model in (QWEN_MODELS[name], "Qwen/" + QWEN_MODELS[name]),
                "Served model identity must match its frozen panel lane.")
    return config


def read_events(path):
    """A torn final record is not executed/retried; preserve original bytes."""
    if not path.exists():
        return [], False
    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    torn = bool(lines and not lines[-1].endswith(b"\n"))
    if torn:
        lines = lines[:-1]
    return [json.loads(line) for line in lines], torn


def request_digest(request):
    encoded = json.dumps(asdict(request), sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def contains_credential_material(value):
    if isinstance(value, dict):
        return any(
            re.sub(r"[-_]", "", key).lower() in SECRET_FIELDS and bool(item)
            or contains_credential_material(item) for key, item in value.items())
    if isinstance(value, list):
        return any(contains_credential_material(item) for item in value)
    if isinstance(value, str):
        return any(secret in value for name in (
            "TYPESAFE_API_KEY", "COPILOT_GITHUB_TOKEN", "VLLM_API_KEY", "GITHUB_TOKEN")
            if (secret := os.environ.get(name)) and len(secret) >= 8)
    return False


def capture_response(stream, response, request, instance, arm, step_index, response_index):
    row = {
        "schema_version": RAW_RESPONSE_VERSION, "instance_id": instance["id"], "arm": arm,
        "step": step_index, "response_index": response_index,
        "request_sha256": request_digest(request),
    }
    error = None
    if not isinstance(response, DecisionResponse):
        error = "invalid_response_type"
    else:
        try:
            value = audit.json_value(asdict(response))
            if contains_credential_material(value):
                error = "credential_material"
            else:
                row["response"] = value
        except (ValueError, TypeError, OverflowError, RecursionError):
            error = "non_json_response"
    row.update(capture_status="failure" if error else "captured", capture_error=error)
    audit.append_row(stream, row)
    return error


def verify_raw_response(row, request, instance, arm, step_index, attempt=None):
    require(row["schema_version"] == RAW_RESPONSE_VERSION and
            row["instance_id"] == instance["id"] and row["arm"] == arm and
            row["step"] == step_index and row["request_sha256"] == request_digest(request),
            "Raw response request binding mismatch.")
    if row["capture_status"] == "failure":
        require("response" not in row and
                row["capture_error"] in CAPTURE_ERRORS,
                "Invalid raw capture failure.")
        if attempt is not None:
            require(attempt["status"] == "failure" and attempt["fatal"] and
                    attempt["raw_response_error"] == row["capture_error"],
                    "Capture failure must terminate the lane explicitly.")
        return
    require(row["capture_status"] == "captured" and row["capture_error"] is None,
            "Invalid raw capture status.")
    value = row["response"]
    require(type(value) is dict and set(value) == {
        "request_id", "selected_option_id", "resolved_model", "probabilities",
        "probability_kind", "confidence", "usage", "raw_output", "status",
    } and value["status"] == "success", "Invalid captured DecisionResponse shape.")
    if attempt is not None:
        require(attempt["raw_response_error"] is None, "Unexpected capture error.")
        if attempt["status"] == "success":
            require(value["request_id"] == request.request_id and
                    value["selected_option_id"] == attempt["selected_option_id"] and
                    value["resolved_model"] == attempt["resolved_model"] and
                    value["usage"] == attempt["usage"], "Raw/allowlisted response identity mismatch.")


def native_config(config):
    return config.provider == "plugin" and config.model in NATIVE_MODELS


def native_rows(config, offset):
    path = Path(config.adapter_kwargs["audit_path"])
    with path.open("rb") as stream:
        stream.seek(offset)
        raw = stream.read()
    require(raw.endswith(b"\n"), "Missing complete durable native evidence.")
    return [json.loads(line) for line in raw.splitlines()]


def verify_native_evidence(attempt, request, config):
    evidence, error = attempt["native_audit"], attempt["native_audit_error"]
    if not native_config(config):
        require(evidence is None and error is None, "Native evidence attached to non-native client.")
        return None
    if error is not None:
        require(error in ("missing_native_audit", "invalid_native_audit") and
                attempt["status"] == "failure" and attempt["fatal"],
                "Native evidence failure must terminate the lane.")
        return None
    require(type(evidence) is list, "Missing native audit list.")
    if not attempt["transport_entered"]:
        require(not evidence and attempt["status"] == "failure", "Unexpected uncalled native evidence.")
        return 0
    expected = request_digest(request)
    require(bool(evidence) and all(
        type(row) is dict and row.get("request_sha256") == expected and
        row.get("model") == config.model and type(row.get("call_index")) is int and row["call_index"] > 0
        for row in evidence), "Native evidence request/model mismatch.")
    require(len({row["call_index"] for row in evidence}) == 1 and
            evidence[0].get("event") == "intent", "Native evidence must describe exactly one attempt.")
    wires = [row["raw_wire"] for row in evidence if "raw_wire" in row]
    require(len(wires) <= 1 and all(type(w) is dict for w in wires), "Invalid raw native wire records.")
    if attempt["status"] == "success":
        require(len(wires) == 1 and wires[0].get("status") == "success" and
                type(wires[0].get("forward_count")) is int and wires[0]["forward_count"] == 1 and
                wires[0].get("result", {}).get("selected_option_id") == attempt["selected_option_id"],
                "Successful native call requires its exact raw wire and one forward.")
        require(evidence[-1].get("event") == "success", "Native response was not durably completed.")
        return 1
    if attempt["diagnostic_code"] == "unsupported":
        require(len(wires) == 1 and wires[0].get("status") == "unsupported" and
                type(wires[0].get("forward_count")) is int and wires[0]["forward_count"] == 0,
                "Native capacity rejection requires its raw zero-forward wire.")
        return 0
    if wires:
        count = wires[0].get("forward_count")
        require(count is None or type(count) is int and count in (0, 1), "Invalid native forward count.")
        return count
    require(evidence[-1].get("event") in ("failure", "http_failure"),
            "Transport failure lacks durable native failure evidence.")
    return None


def replay(instance, events, config, arm, raw_responses=None):
    raw_responses = [] if raw_responses is None else raw_responses
    prepared = prepare(instance)
    decisions, pending, stopped = [], None, False
    calls = forced = intents = 0
    candidate_counts, coverage = [], []
    failure, fatal = None, False
    candidate_seconds = model_seconds = 0.0
    probability_failures = 0
    native_forwards = native_unknown = 0
    raw_cursor = raw_failures = 0
    for event in events:
        require(not stopped, "Events after terminal failure.")
        if event["event"] == "call_started":
            require(pending is None and event["step"] == len(decisions), "Invalid call intent.")
            payload = step(prepared, decisions, arm)
            require(payload is not None and payload["forced_action"] is None, "Invalid call intent step.")
            require(event["request"] == audit.json_value(asdict(payload["request"])), "Intent request drift.")
            pending = event
            intents += 1
            continue
        require(event["event"] == "step" and event["step"] == len(decisions), "Invalid step ordering.")
        payload = step(prepared, decisions, arm)
        require(payload is not None, "Step after completion.")
        require(event["proposal"] == {k: v for k, v in payload.items() if k != "request"},
                "Proposal/order/feature replay mismatch.")
        require(type(event["candidate_seconds"]) in (int, float) and
                math.isfinite(event["candidate_seconds"]) and event["candidate_seconds"] >= 0,
                "Invalid candidate timing.")
        candidate_seconds += event["candidate_seconds"]
        candidate_counts.append(payload["distinct_candidate_count"])
        coverage.append(payload["candidate_coverage"])
        action = event["selected_action"]
        if payload["forced_action"] is not None:
            require(pending is None and event["attempt"] is None and action == payload["forced_action"],
                    "Invalid forced transition.")
            forced += 1
        else:
            require(pending is not None, "Missing fsynced pre-call intent.")
            pending = None
            attempt = event["attempt"]
            require(type(attempt["transport_entered"]) is bool, "Missing transport accounting.")
            calls += attempt["transport_entered"]
            base = {k: v for k, v in attempt.items()
                    if k not in ("probability_audit", "transport_entered",
                                 "native_audit", "native_audit_error",
                                 "raw_response_index", "raw_response_error")}
            unsupported = base["diagnostic_code"] == "unsupported"
            if unsupported:
                base = {**base, "diagnostic_code": "client_error"}
            audit.validate_attempt(base, payload["request"], config)
            require(attempt["instance_id"] == instance["id"] and attempt["step"] == len(decisions),
                    "Attempt identity mismatch.")
            index = attempt["raw_response_index"]
            if index is not None:
                require(type(index) is int and index == raw_cursor and index < len(raw_responses) and
                        raw_responses[index]["response_index"] == index,
                        "Raw response ordering mismatch.")
                verify_raw_response(raw_responses[index], payload["request"], instance, arm,
                                    len(decisions), attempt)
                raw_failures += raw_responses[index]["capture_status"] == "failure"
                raw_cursor += 1
            else:
                require(attempt["status"] == "failure" and attempt["raw_response_error"] is None,
                        "A successful returned response must have durable raw evidence.")
            forward_count = verify_native_evidence(attempt, payload["request"], config)
            if native_config(config):
                native_forwards += forward_count or 0
                native_unknown += forward_count is None
            model_seconds += attempt["latency_seconds"]
            vector = attempt["probability_audit"]
            if config.provider == "typesafe" and attempt["status"] == "success":
                require(isinstance(vector, dict), "Missing Jev probability audit.")
                body = {"model": config.model, "usage": attempt["usage"], "answers": {"decision": {
                    "type": "choice", "choice": attempt["selected_option_id"],
                    "probabilities": vector["probabilities"], "confidence": vector["confidence"]}}}
                response = public.TypeSafeClient._parse_response(payload["request"], body,
                                                                 require_probabilities=False)
                require(public.probability_audit(response, payload["request"], config) == vector,
                        "Probability replay mismatch.")
                probability_failures += vector["diagnostic_code"] is not None
            else:
                require(vector is None, "Unexpected probability audit.")
            if attempt["status"] == "failure":
                require(action is None, "Failed action executed.")
                failure, fatal, stopped = attempt["diagnostic_code"], attempt["fatal"], True
                continue
            require(action == graph.selected_action(payload, attempt["selected_option_id"]),
                    "Option/action mapping mismatch.")
        decisions.append(action)
    if pending is not None and raw_cursor < len(raw_responses):
        require(len(raw_responses) == raw_cursor + 1 and
                raw_responses[raw_cursor]["response_index"] == raw_cursor,
                "Unexpected orphan response count.")
        verify_raw_response(raw_responses[raw_cursor], step(prepared, decisions, arm)["request"],
                            instance, arm, len(decisions))
    else:
        require(len(raw_responses) == raw_cursor, "Unbound raw responses after terminal step.")
    result = score(instance, decisions)
    return {
        **result, "instance_id": instance["id"], "task": instance["task"],
        "dataset_id": instance.get("dataset_id", instance["id"]),
        "replicate": instance.get("replicate"), "arm": arm, "decisions": decisions,
        "model_calls": None if pending is not None else calls,
        "confirmed_model_calls": calls, "call_intents": intents,
        "model_calls_unknown": pending is not None,
        "forced_candidate_steps": forced,
        "candidate_counts": candidate_counts, "candidate_coverage": coverage,
        "candidate_seconds": candidate_seconds, "model_call_seconds": model_seconds,
        "probability_audit_failures": probability_failures,
        "failure": failure, "fatal": fatal, "pending_call": pending is not None,
        "native_forward_count": (native_forwards if native_config(config) and
                                 not native_unknown and pending is None else None),
        "confirmed_native_forwards": native_forwards if native_config(config) else None,
        "native_forward_unknown_attempts": (native_unknown + int(pending is not None)
                                           if native_config(config) else None),
        "captured_response_records": len(raw_responses),
        "raw_response_capture_failures": raw_failures,
        "orphan_raw_responses": len(raw_responses) - raw_cursor,
    }


def seal_episode(directory, instance, config, arm, reason=None, wall=None):
    started = time.perf_counter()
    events, torn = read_events(directory / "events.jsonl")
    raw_responses, raw_torn = read_events(directory / "raw-responses.jsonl")
    result = replay(instance, events, config, arm, raw_responses)
    result.update(
        terminal_reason=result["failure"] or reason or (
            None if result["feasible"] else "interrupted"),
        solve_wall_seconds=wall, verification_seconds=time.perf_counter() - started,
        torn_final_event=torn,
        torn_final_raw_response=raw_torn,
    )
    atomic_json(directory / "score.json", result)
    atomic_json(directory / "status.json", {
        "status": "completed" if result["feasible"] else "incomplete",
        "finished_at": public.now(), "artifact_sha256": {
            name: digest(directory / name) for name in ("events.jsonl", "raw-responses.jsonl", "score.json")},
    })
    return result


async def execute_episode(directory, instance, config, arm, client, abort, *, root=None):
    if root is not None:
        check_hold(root)
    directory.mkdir(parents=True, exist_ok=False)
    boundary = root if root is not None else directory.parent
    for path in (directory, *directory.parents):
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        if path == boundary:
            break
    started = time.perf_counter()
    atomic_json(directory / "status.json", {"status": "running", "pid": os.getpid(), "at": public.now()})
    prepared, decisions, reason = prepare(instance), [], None
    response_count = 0
    with (directory / "events.jsonl").open("x") as stream, \
            (directory / "raw-responses.jsonl").open("x") as raw_stream:
        fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        while not abort.is_set():
            if root is not None:
                check_hold(root)
            candidate_start = time.perf_counter()
            payload = step(prepared, decisions, arm)
            candidate_seconds = time.perf_counter() - candidate_start
            if payload is None:
                break
            remaining = EPISODE_SECONDS - (time.perf_counter() - started)
            if remaining <= 0:
                reason = "episode_timeout"
                break
            row = {"event": "step", "step": len(decisions),
                   "proposal": {k: v for k, v in payload.items() if k != "request"},
                   "candidate_seconds": candidate_seconds, "attempt": None,
                   "selected_action": payload["forced_action"]}
            if payload["forced_action"] is None:
                request = payload["request"]
                audit.append_row(stream, {
                    "event": "call_started", "step": len(decisions), "at": public.now(),
                    "request": audit.json_value(asdict(request)),
                })
                call_start = time.perf_counter()
                transport_entered = False
                raw_index = raw_error = None
                audit_path = Path(config.adapter_kwargs["audit_path"]) if native_config(config) else None
                audit_offset = audit_path.stat().st_size if audit_path and audit_path.exists() else 0
                try:
                    client.validate_request(request)
                    transport_entered = True
                    response = await asyncio.wait_for(client.predict(request),
                        timeout=min(config.timeout_seconds + 5, remaining))
                    raw_index = response_count
                    raw_error = capture_response(raw_stream, response, request, instance, arm,
                                                 len(decisions), response_count)
                    response_count += 1
                    require(raw_error is None, "Non-JSON or invalid raw DecisionResponse capture.")
                    attempt = audit.response_record(response, request, config)
                    attempt["probability_audit"] = public.probability_audit(response, request, config)
                    row["selected_action"] = graph.selected_action(payload, response.selected_option_id)
                except Exception as exc:
                    # Fail closed, record only allowlisted diagnostics, never provider text/secrets.
                    if isinstance(exc, TimeoutError):
                        exc = ClientTimeoutError("Total call deadline exhausted.")
                    diagnostic = ({"diagnostic_code": "unsupported", "http_status": None, "fatal": False}
                                  if isinstance(exc, UnsupportedRequestError) else audit.failure(exc))
                    attempt = {"status": "failure", "selected_option_id": None, "resolved_model": None,
                               "usage": audit.safe_usage(getattr(exc, "usage", None)),
                               "probability_audit": None, **diagnostic}
                    if attempt["fatal"]:
                        abort.set()
                attempt.update(instance_id=instance["id"], step=len(decisions),
                               request=audit.json_value(asdict(request)),
                               transport_entered=transport_entered,
                               native_audit=[] if audit_path else None, native_audit_error=None,
                               raw_response_index=raw_index, raw_response_error=raw_error,
                               latency_seconds=time.perf_counter() - call_start)
                if audit_path and transport_entered:
                    try:
                        attempt["native_audit"] = native_rows(config, audit_offset)
                        verify_native_evidence(attempt, request, config)
                    except (OSError, ValueError, public.shared.Refusal, KeyError, TypeError):
                        attempt.update(
                            status="failure", selected_option_id=None, resolved_model=None,
                            diagnostic_code="internal_error", http_status=None, fatal=True,
                            probability_audit=None,
                            native_audit_error=("missing_native_audit" if not attempt["native_audit"]
                                                else "invalid_native_audit"),
                        )
                        row["selected_action"] = None
                        abort.set()
                row["attempt"] = attempt
            audit.append_row(stream, row)
            if row["selected_action"] is None:
                break
            decisions.append(row["selected_action"])
    result = seal_episode(directory, instance, config, arm, reason or (
        "fatal_lane_abort" if abort.is_set() else None), time.perf_counter() - started)
    print(json.dumps({"episode": str(directory), "feasible": result["feasible"],
                      "calls": result["model_calls"], "terminal_reason": result["terminal_reason"]}), flush=True)
    return result


async def run_lane(root, instances, name, spec, concurrency, factory=make_client):
    check_hold(root)
    lane = root / "runs" / name
    lane.mkdir(parents=True, exist_ok=True)
    with (lane / "lane.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = validate_config(spec)
        if native_config(config):
            require(Path(config.adapter_kwargs["audit_path"]).resolve().is_relative_to(root.resolve()),
                    "Native audit_path must belong to this fresh experiment root.")
        if (lane / "config.json").exists():
            require(read_json(lane / "config.json") == spec, "Cannot change an existing lane config.")
        else:
            write_json(lane / "config.json", spec)
            write_json(lane / "config.sha256", digest(lane / "config.json"))
        require(digest(lane / "config.json") == read_json(lane / "config.sha256"), "Lane config drift.")
        jobs = read_json(root / "schedule.json")
        abort = asyncio.Event()
        queue = asyncio.Queue()
        for job in jobs:
            directory = lane / job["arm"] / job["instance_id"]
            if directory.exists():
                if not (directory / "events.jsonl").exists():
                    with (directory / "events.jsonl").open("x"):
                        pass
                if not (directory / "raw-responses.jsonl").exists():
                    with (directory / "raw-responses.jsonl").open("x"):
                        pass
                if (directory / "score.json").exists() and (directory / "status.json").exists() and (
                        read_json(directory / "status.json").get("status") in ("completed", "incomplete")):
                    verify_episode(directory, instances[job["instance_id"]], config, job["arm"])
                    result = read_json(directory / "score.json")
                else:
                    result = seal_episode(directory, instances[job["instance_id"]], config, job["arm"],
                                          "interrupted_no_retry")
                if result["fatal"]:
                    abort.set()
            else:
                queue.put_nowait(job)
        previous = read_json(lane / "status.json") if (lane / "status.json").exists() else {}
        if previous.get("status") == "aborted":
            abort.set()
        atomic_json(lane / "status.json", {"status": "running", "pid": os.getpid(),
                                          "started_at": public.now(), "concurrency": concurrency})
        errors = []

        async def worker():
            client = None
            try:
                if abort.is_set() or queue.empty():
                    return
                client = factory(spec)
                await asyncio.wait_for(client.initialize(), timeout=config.timeout_seconds + 5)
                capabilities = asdict(client.capabilities)
                if (lane / "capabilities.json").exists():
                    require(read_json(lane / "capabilities.json") == capabilities, "Client capability drift.")
                else:
                    atomic_json(lane / "capabilities.json", capabilities)
                while not abort.is_set():
                    check_hold(root)
                    try:
                        job = queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                    await execute_episode(lane / job["arm"] / job["instance_id"],
                                          instances[job["instance_id"]], config, job["arm"], client, abort,
                                          root=root)
            except Exception as exc:
                errors.append({"phase": "worker", "type": type(exc).__name__, **audit.failure(exc)})
                abort.set()
                print(json.dumps({"lane": name, "failure": errors[-1]}), flush=True)
            finally:
                if client is not None:
                    try:
                        await asyncio.wait_for(client.aclose(), timeout=30)
                    except Exception as exc:
                        errors.append({"phase": "shutdown", "type": type(exc).__name__, **audit.failure(exc)})
                        abort.set()

        await asyncio.gather(*(worker() for _ in range(concurrency)))
        check_hold(root)
        for job in jobs:
            directory = lane / job["arm"] / job["instance_id"]
            if not directory.exists():
                directory.mkdir(parents=True)
                with (directory / "events.jsonl").open("x"):
                    pass
                with (directory / "raw-responses.jsonl").open("x"):
                    pass
                seal_episode(directory, instances[job["instance_id"]], config, job["arm"], "fatal_lane_abort")
            elif not (directory / "score.json").exists():
                seal_episode(directory, instances[job["instance_id"]], config, job["arm"], "worker_failure")
        atomic_json(lane / "status.json", {"status": "aborted" if abort.is_set() else "completed",
                                          "finished_at": public.now(), "errors": errors})


def verify_episode(directory, instance, config, arm):
    status = read_json(directory / "status.json")
    require(set(status["artifact_sha256"]) == {"events.jsonl", "raw-responses.jsonl", "score.json"},
            "Episode seal must include every required sidecar.")
    require(all(digest(directory / name) == expected for name, expected in status["artifact_sha256"].items()),
            "Episode seal mismatch.")
    events, torn = read_events(directory / "events.jsonl")
    raw_responses, raw_torn = read_events(directory / "raw-responses.jsonl")
    expected = replay(instance, events, config, arm, raw_responses)
    result = read_json(directory / "score.json")
    require(all(result[k] == v for k, v in expected.items()) and result["torn_final_event"] == torn and
            result["torn_final_raw_response"] == raw_torn,
            "Episode independent replay mismatch.")
    return result


def report(root):
    root, protocol, instances = load(root)
    condition_counts = Counter((i["task"], i.get("dataset_id", i["id"])) for i in instances.values())
    rows = []
    for name in protocol["models"]:
        lane = root / "runs" / name
        config = (validate_config(read_json(lane / "config.json"), verify_adapter=False)
                  if (lane / "config.json").exists() else None)
        if config is not None:
            require(digest(lane / "config.json") == read_json(lane / "config.sha256"),
                    "Lane configuration seal mismatch.")
        for job in read_json(root / "schedule.json"):
            instance = instances[job["instance_id"]]
            directory = lane / job["arm"] / job["instance_id"]
            if (directory / "score.json").exists() and (directory / "status.json").exists() and (
                    read_json(directory / "status.json").get("status") in ("completed", "incomplete")):
                result = verify_episode(directory, instance, config, job["arm"])
            else:
                result = {
                    "instance_id": instance["id"], "task": instance["task"], "arm": job["arm"],
                    "dataset_id": instance.get("dataset_id", instance["id"]),
                    "feasible": False, "objective": None,
                    "terminal_reason": "in_progress" if directory.exists() else "unattempted",
                    "model_calls": None, "candidate_counts": [], "candidate_coverage": [],
                }
            rows.append({"model": name, **result})
    groups = defaultdict(list)
    for row in rows:
        groups[(row["model"], row["task"], row["arm"])].append(row)
    summaries = []
    for (name, task, arm), own in groups.items():
        complete = [r for r in own if r["feasible"]]
        per_graph = defaultdict(list)
        for r in complete:
            per_graph[r["dataset_id"]].append(r.get("percentage_gap", r.get("absolute_gap")))
        counts = [c for r in own for c in r["candidate_counts"]]
        summaries.append({
            "model": name, "task": task, "arm": arm, "scheduled": len(own),
            "completed": len(complete), "coverage": len(complete) / len(own),
            "support_outcomes": dict(Counter(r["terminal_reason"] or "complete" for r in own)),
            "objective_feasible_mean": statistics.mean(r["objective"] for r in complete) if complete else None,
            "gap_graph_macro": statistics.mean(statistics.mean(v) for v in per_graph.values()) if per_graph else None,
            "gap_units": "percentage_points" if task.endswith("_public") else "absolute_objective_gap",
            "graphs_with_complete_conditions": sum(len(v) == condition_counts[task, graph_id]
                                                  for graph_id, v in per_graph.items()),
            "singleton_fraction_observed": counts.count(1) / len(counts) if counts else None,
            "mean_candidate_count_observed": statistics.mean(counts) if counts else None,
            "mean_candidate_coverage_observed": public.average(
                [v for r in own for v in r["candidate_coverage"]]),
            "observed_steps": len(counts),
            "model_calls": sum(r["model_calls"] or 0 for r in own),
            "model_call_count_known_episodes": sum(r["model_calls"] is not None for r in own),
        })
    indexed = {(r["model"], r["instance_id"], r["arm"]): r for r in rows}
    paired = []
    for name in protocol["models"]:
        for task in sorted({i["task"] for i in instances.values()}):
            own_instances = [i for i in instances.values() if i["task"] == task]
            for left, right in (("C", "B"), ("B", "A"), ("C", "A")):
                deltas = defaultdict(list)
                for instance in own_instances:
                    a, b = (indexed[name, instance["id"], arm] for arm in (left, right))
                    if a["feasible"] and b["feasible"]:
                        key = "percentage_gap" if task.endswith("_public") else "absolute_gap"
                        deltas[instance.get("dataset_id", instance["id"])].append(a[key] - b[key])
                paired.append({
                    "model": name, "task": task, "contrast": f"{left}-{right}",
                    "scheduled_pairs": len(own_instances),
                    "completed_pairs": sum(map(len, deltas.values())),
                    "graph_macro_gap_delta": public.average([statistics.mean(v) for v in deltas.values()]),
                    "graphs_with_all_conditions": sum(len(v) == condition_counts[task, graph_id]
                                                       for graph_id, v in deltas.items()),
                    "negative_favors": left,
                })
    atomic_json(root / "report.json", {"at": public.now(), "protocol_sha256": digest(root / "protocol.json"),
                                      "episodes": rows, "summaries": summaries, "paired": paired})
    return summaries


def process_state(pid):
    if type(pid) is not int or pid <= 0:
        return None
    try:
        return Path(f"/proc/{pid}/stat").read_text().rpartition(")")[2].split()[0]
    except FileNotFoundError:
        return "missing"


def monitor_snapshot(root, protocol, jobs, *, now=None, stall_seconds=1800):
    now = time.time() if now is None else now
    lanes, alerts = {}, []
    for name in protocol["models"]:
        lane = root / "runs" / name
        status = read_json(lane / "status.json") if (lane / "status.json").exists() else {"status": "unstarted"}
        counts = Counter()
        latest = None
        for job in jobs:
            directory = lane / job["arm"] / job["instance_id"]
            path = directory / "status.json"
            counts[read_json(path)["status"] if path.exists() else "unstarted"] += 1
            events = directory / "events.jsonl"
            if events.exists():
                stamp = events.stat().st_mtime
                latest = max(latest, stamp) if latest is not None else stamp
        state = process_state(status.get("pid"))
        terminal = status["status"] in ("completed", "aborted")
        if not terminal and status["status"] == "running":
            if state in ("missing", "Z", "T", "t"):
                alerts.append({"model": name, "code": "controller_inactive", "process_state": state})
            if latest is not None and now - latest > stall_seconds:
                alerts.append({"model": name, "code": "no_recent_episode_progress",
                               "seconds_since_event": now - latest})
        terminal_count = counts["completed"] + counts["incomplete"]
        if terminal and terminal_count != len(jobs):
            alerts.append({"model": name, "code": "terminal_lane_missing_episode_seals"})
        lanes[name] = {
            "status": status["status"], "pid": status.get("pid"), "process_state": state,
            "scheduled": len(jobs), "episodes": dict(counts), "terminal_episodes": terminal_count,
            "last_event_age_seconds": None if latest is None else max(0, now - latest),
        }
    terminal_count = sum(v["terminal_episodes"] for v in lanes.values())
    ready = all(v["status"] in ("completed", "aborted") for v in lanes.values())
    scheduled = len(jobs) * len(protocol["models"])
    return {
        "at": public.now(), "monitor_pid": os.getpid(), "version": protocol["version"],
        "scheduled_episodes": scheduled, "terminal_episodes": terminal_count,
        "all_terminal": ready and terminal_count == scheduled,
        "status": "all_terminal_unverified" if ready and terminal_count == scheduled else
                  "attention_required" if alerts else "progressing",
        "lanes": lanes, "alerts": alerts,
    }


def verify_terminal(root, protocol, instances, jobs):
    check_hold(root)
    hashes, verified = {}, 0
    outcomes = Counter()
    for name in protocol["models"]:
        lane = root / "runs" / name
        require(read_json(lane / "status.json")["status"] in ("completed", "aborted"),
                "Not every lane is terminal.")
        require(digest(lane / "config.json") == read_json(lane / "config.sha256"), "Lane config drift.")
        config = validate_config(read_json(lane / "config.json"), verify_adapter=False)
        for file in ("status.json", "config.json", "config.sha256"):
            hashes[str((lane / file).relative_to(root))] = digest(lane / file)
        for job in jobs:
            directory = lane / job["arm"] / job["instance_id"]
            require(read_json(directory / "status.json")["status"] in ("completed", "incomplete"),
                    "Not every episode is terminal.")
            result = verify_episode(directory, instances[job["instance_id"]], config, job["arm"])
            verified += 1
            outcomes[result["terminal_reason"] or "complete"] += 1
            hashes[str((directory / "status.json").relative_to(root))] = digest(directory / "status.json")
    expected = len(jobs) * len(protocol["models"])
    require(verified == expected, "Missing terminal verification.")
    check_hold(root)
    return {
        "at": public.now(), "protocol_sha256": digest(root / "protocol.json"),
        "scheduled_episodes": expected, "terminal_episodes": expected, "verified_episodes": verified,
        "terminal_lanes": len(protocol["models"]), "outcomes": dict(outcomes),
        "inputs_sha256": protocol["inputs_sha256"], "ledger_seals_sha256": hashes,
        "status": "all_terminal_independently_replayed",
        "publication": "Scientific/paper validation remains separate; failures are not successes.",
    }


def monitor(root, interval):
    require(type(interval) is int and interval >= 1, "Monitor interval must be a positive integer.")
    root, protocol, instances = load(root)
    jobs = read_json(root / "schedule.json")
    with (root / "monitor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (root / "monitor-events.jsonl").open("a") as stream:
            while True:
                try:
                    check_hold(root)
                    snapshot = monitor_snapshot(root, protocol, jobs)
                    snapshot["protocol_sha256"] = digest(root / "protocol.json")
                    audit.append_row(stream, snapshot)
                    atomic_json(root / "monitor-status.json", snapshot)
                    print(json.dumps({"at": snapshot["at"], "status": snapshot["status"],
                                      "terminal": snapshot["terminal_episodes"],
                                      "scheduled": snapshot["scheduled_episodes"],
                                      "alerts": snapshot["alerts"]}), flush=True)
                    if snapshot["all_terminal"]:
                        receipt = verify_terminal(root, protocol, instances, jobs)
                        atomic_json(root / "terminal-receipt.json", receipt)
                        atomic_json(root / "monitor-status.json", {
                            **snapshot, "status": "all_terminal_verified",
                            "terminal_receipt_sha256": digest(root / "terminal-receipt.json"),
                        })
                        return
                except Exception as exc:
                    failure = {"at": public.now(), "status": "verification_or_monitor_failed",
                               "error_type": type(exc).__name__, "pid": os.getpid()}
                    audit.append_row(stream, failure)
                    atomic_json(root / "monitor-status.json", failure)
                    raise
                time.sleep(interval)


async def run(root, config_path, models, concurrency):
    root, protocol, instances = load(root)
    specs = read_json(Path(config_path))
    require(models and len(set(models)) == len(models) and set(models) <= set(protocol["models"]),
            "Choose distinct frozen panel IDs.")
    require(type(concurrency) is int and 1 <= concurrency <= 16, "Concurrency must be 1..16.")
    for name in models:
        config = validate_lane_config(name, specs[name])
        if config.provider == "plugin":
            require(concurrency == 1, "Native/plugin lanes require concurrency=1 to avoid duplicate model loads.")
        if config.provider == "typesafe":
            public.shared.credential_preflight(("jev",))
        if name in read_json(root / "cloud-models.json"):
            require(specs[name] == read_json(root / "cloud-models.json")[name], "Frozen cloud config drift.")
    await asyncio.gather(*(run_lane(root, instances, name, specs[name], concurrency) for name in models))
    report(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "run", "report", "monitor"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model-config", type=Path)
    parser.add_argument("--instances", type=Path, help="Explicit JSON record list for a portable freeze.")
    parser.add_argument("--models", nargs="+", choices=PANEL)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--interval", type=int, default=60)
    args = parser.parse_args()
    try:
        require(args.instances is None or args.command == "freeze", "--instances is only valid for freeze.")
        if args.command == "freeze":
            freeze(args.root, instances_path=args.instances, config_path=args.model_config, models=args.models)
        elif args.command == "report":
            report(args.root)
        elif args.command == "monitor":
            monitor(args.root, args.interval)
        else:
            require(args.model_config is not None, "--model-config required.")
            asyncio.run(run(args.root, args.model_config, args.models, args.concurrency))
    except Exception as exc:
        print(json.dumps({"status": "failed", "type": type(exc).__name__,
                          "diagnostic": "refused_or_failed_no_retry",
                          "details": "Exception text suppressed; inspect sealed status artifacts."}), flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
