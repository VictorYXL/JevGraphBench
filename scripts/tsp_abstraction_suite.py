#!/usr/bin/env python3
"""Frozen, isolated TSPLIB action-abstraction supplement; no main-run mutation."""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from copy import deepcopy
import csv
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
import sys
import time
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts import public_graph_suite as public
from src.benchmark import public_tasks as tasks
from src.benchmark import tsp_abstraction as abstraction
from src.benchmark.config import ModelConfig
from src.clients.base import DecisionClientError, ClientTimeoutError

audit, shared = public.audit, public.shared
require, digest, read_json, write_json = public.require, public.digest, public.read_json, public.write_json
MODELS = ("jev", "qwen4", "qwen9")
ARMS = ("B", "C")
SEED = 20260929
PROTOCOL = "tsplib-append-abstraction-v1"
RANDOM_SEEDS = (1101, 1102, 1103, 1104, 1105)
EPISODE_SECONDS = 900


def code_hashes():
    hashes = audit.code_hashes()
    for name in ("scripts/public_graph_suite.py", "scripts/tsp_abstraction_suite.py",
                 "tests/test_tsp_abstraction.py"):
        hashes[name] = digest(REPO / name)
    return dict(sorted(hashes.items()))


def freeze(root, hashes):
    destination = root / "frozen-source"
    destination.mkdir()
    for name, expected in hashes.items():
        source, target = REPO / name, destination / name
        require(not source.is_symlink(), "Refuse symlinked source.")
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(source.read_bytes())
            stream.flush()
            os.fsync(stream.fileno())
        require(digest(target) == expected, "Source changed during freeze.")
        target.chmod(0o444)
    audit.verify_source_snapshot(root, hashes)
    require(code_hashes() == hashes, "Source drift during freeze.")


def independent_matrix(instance):
    points = instance["state"]["coordinates"]
    matrix = []
    for x, y in points:
        row = []
        for a, b in points:
            dx, dy = float(x) - float(a), float(y) - float(b)
            row.append(int(math.sqrt(dx * dx + dy * dy) + 0.5))
        matrix.append(row)
    return matrix


def independent_proposals(matrix, decisions):
    current = int(decisions[-1]) if decisions else 0
    unseen = set(range(1, len(matrix))) - set(map(int, decisions))
    values = {}
    for vertex in sorted(unseen):
        e = matrix[current][vertex]
        h = min(matrix[vertex][other] for other in unseen if other != vertex)
        r = matrix[vertex][0]
        values[vertex] = (e, e + h, 2 * e - r, e - h)
    return {rule: min(unseen, key=lambda city: (values[city][index], city))
            for index, rule in enumerate(abstraction.RULES)}


def independent_objective(instance, decisions):
    n = len(instance["state"]["nodes"])
    require(len(decisions) == n - 2 and all(type(v) is str for v in decisions),
            "Independent objective requires completed exact-string history.")
    tour = [0, *map(int, decisions)]
    require(len(set(tour)) == n - 1 and set(tour) <= set(range(n)), "Invalid tour prefix.")
    last = next(v for v in range(n) if v not in tour)
    tour.extend([last, 0])
    matrix = independent_matrix(instance)
    return sum(matrix[a][b] for a, b in zip(tour, tour[1:]))


def verify_score(instance, result):
    if result["feasible"]:
        objective = independent_objective(instance, result["decisions"])
        require(objective == result["objective"], "Independent objective mismatch.")
        reference = instance["private"]["reference"]["value"]
        require(result["additive_gap"] == objective - reference, "Gap mismatch.")
        require(abs(result["percentage_gap"] - 100 * (objective - reference) / reference) < 1e-9,
                "Percentage gap mismatch.")
    else:
        require(result["objective"] is None and result["percentage_gap"] is None,
                "Incomplete episode must not receive an objective.")


def baseline_episode(instance, method, seed):
    started = time.perf_counter()
    prep = abstraction.prepare(instance)
    setup_seconds = time.perf_counter() - started
    rng = random.Random(seed)
    decisions, steps = [], []
    candidate_seconds = 0.0
    if method in ("nearest_neighbor", "nearest_neighbor_two_opt"):
        decisions = tasks.baseline(instance, method, seed=seed)
    else:
        while len(decisions) < len(prep.distances) - 2:
            start = time.perf_counter()
            payload = abstraction.step(prep, decisions, "B", SEED)
            candidate_seconds += time.perf_counter() - start
            city = abstraction.baseline_choice(payload, method, rng)
            steps.append({
                "step": len(decisions), "candidates": payload["candidates"],
                "rule_proposals": payload["rule_proposals"], "selected_city": city,
                "distinct_candidate_count": payload["distinct_candidate_count"],
                "candidate_coverage": payload["candidate_coverage"],
                "forced": payload["forced_city"] is not None,
            })
            decisions.append(str(city))
    result = {
        "instance_id": instance["id"], "dataset_id": instance["dataset_id"],
        "replicate": instance["replicate"], "method": method, "seed": seed,
        "decisions": decisions, "setup_seconds": setup_seconds,
        "candidate_seconds": candidate_seconds, "model_call_seconds": 0.0,
        "model_calls": 0, "forced_candidate_steps": sum(row["forced"] for row in steps),
        "steps": steps, **tasks.score(instance, decisions),
    }
    result["solve_wall_seconds"] = time.perf_counter() - started
    replay_started = time.perf_counter()
    verify_score(instance, result)
    matrix = independent_matrix(instance)
    for row in steps:
        expected = independent_proposals(matrix, decisions[:row["step"]])
        require(row["rule_proposals"] == expected, "Baseline independent proposal mismatch.")
        require(row["selected_city"] in expected.values(), "Baseline chose outside candidates.")
    result["verification_seconds"] = time.perf_counter() - replay_started
    return result


def prior_art():
    return {
        "consulted": [
            {
                "document": "Local OR survey: 03-or-and-recommender.md",
                "sha256": "7b18a98008aa1679abf36088ff65b7bef065c5430259b35d9cd66b6bfcfa66f1",
            },
            {
                "document": "HeurAgenix: src/problems/tsp/heuristics/basic_heuristics/nearest_neighbor_f91d.py",
                "sha256": "f067b0e8facfa5252f7e49ecc59f43b4c817637088e91909caf008781cde164e",
            },
            {
                "document": "HeurAgenix: src/problems/tsp/heuristics/evolved_heuristics/nearest_neighbor_e8a4.py",
                "sha256": "ad05530b0ba5497b6ab0f2ac8ef9f5b841d375bce65b42e2cdf85f250dc038cc",
            },
        ],
        "provenance": "Historical consulted-content hashes, not runtime file dependencies or a fresh source verification.",
        "heuragenix_paper": "https://arxiv.org/abs/2506.15196",
        "scope": "Local OR survey establishes prior Jev TSP/search/heuristic-selection examples; no first-use claim.",
        "nearest_neighbor": "HeurAgenix contains append-nearest-neighbor; reused concept, independently implemented.",
        "exclusions": "HeurAgenix evolved heuristic mixes insertion, adaptive filtering and 2-opt. None is allowed as a model action here.",
        "pool_rationale": {
            "nearest": "Immediate-cost anchor and explicit deterministic selector comparator.",
            "one_step_lookahead": "Charge an optimistic onward edge; discourages expensive-to-leave append moves.",
            "return_reserve": "Trade twice immediate cost against distance from start, reserving near-start vertices for closure.",
            "isolation_priority": "Subtract nearest-other distance to prioritize vertices otherwise expensive to connect.",
        },
        "status": "Four fixed hand-designed scores, not claimed named standard algorithms; no learned/tuned weights.",
    }


def plan(root, verified):
    root, verified = Path(root).resolve(), Path(verified).resolve()
    require(not root.exists(), "Supplement root must be new.")
    report = read_json(verified / "report.json")
    summary = read_json(verified / "summary.json")
    sources = report["curation"]["source_runs"]
    require(len(sources) == 2, "Expected original and complete tsp225 replacement sources.")
    source_roots = [Path(entry["root"]) for entry in sources]
    for entry, directory in zip(sources, source_roots):
        require(digest(directory / "manifest.json") == entry["manifest_sha256"], "A source manifest drift.")
    all_instances = read_json(source_roots[0] / "instances.json")
    instances = [instance for instance in all_instances if instance["task"] == "tsp_public"]
    require(len(instances) == 21 and len({i["dataset_id"] for i in instances}) == 7,
            "Require exactly seven graphs and their three original conditions.")
    replacements = {i["id"]: i for i in read_json(source_roots[1] / "instances.json")}
    for instance in instances:
        tasks.validate_instance(instance)
        if instance["dataset_id"] == "tsp225":
            require(instance == replacements[instance["id"]], "Replacement condition mismatch.")
    configs = public.configurations(MODELS)
    a_rows, provenance, a_candidate_coverage = [], [], []
    for name in MODELS:
        require(report["models"][name]["config"] == asdict(configs[name]), "A/model settings differ.")
        indexed = {row["instance_id"]: row for row in report["models"][name]["episodes"]}
        for instance in instances:
            directory = source_roots[1 if instance["dataset_id"] == "tsp225" else 0]
            path = directory / "runs" / name / instance["id"]
            attempts = audit.read_rows(path / "attempts.jsonl")
            replay_start = time.perf_counter()
            replay = public.replay_episode(instance, attempts, configs[name])
            require(replay == read_json(path / "score.json"), "A original score replay mismatch.")
            verified_row = indexed[instance["id"]]
            require(all(verified_row[k] == value for k, value in replay.items()), "A curated report mismatch.")
            require(replay["feasible"], "A reuse requires a complete independently verified trajectory.")
            verify_score(instance, replay)
            status = read_json(path / "status.json")
            require(all(digest(path / file) == value
                        for file, value in status["artifact_sha256"].items()), "A episode seal drift.")
            a_rows.append({
                **verified_row, "model": name, "arm": "A",
                "model_calls": replay["query_count"], "forced_candidate_steps": 0,
                "solve_wall_seconds": None,
                "legacy_wall_seconds_including_replay": status["wall_seconds"],
                "verification_seconds": time.perf_counter() - replay_start,
                "source_directory": str(path),
                "timing_caveat": "Legacy A wall includes verification; true A solve-only wall unavailable. Do not subtract a new-machine replay duration.",
            })
            prepared = abstraction.prepare(instance)
            for step_index, city in enumerate(replay["decisions"]):
                payload = abstraction.step(prepared, replay["decisions"][:step_index], "B", SEED)
                a_candidate_coverage.append({
                    "model": name, "instance_id": instance["id"], "dataset_id": instance["dataset_id"],
                    "step": step_index, "selected_city": int(city),
                    "candidate_cities": sorted(set(payload["rule_proposals"].values())),
                    "selected_city_in_candidates": int(city) in payload["rule_proposals"].values(),
                    "distinct_candidates": payload["distinct_candidate_count"],
                    "legal_cities": payload["unvisited_count"],
                    "rule_proposals": payload["rule_proposals"],
                })
            provenance.append({
                "model": name, "instance_id": instance["id"], "root": str(directory),
                "sha256": {file: digest(path / file) for file in
                           ("attempts.jsonl", "score.json", "status.json")},
            })
    # Baselines are executed before model outcomes, and cannot alter the fixed pool.
    baselines = []
    for instance in instances:
        for method in (*abstraction.RULES, "shortest_edge_selector",
                       "nearest_neighbor", "nearest_neighbor_two_opt", "random_candidate"):
            seeds = RANDOM_SEEDS if method == "random_candidate" else (SEED,)
            for seed in seeds:
                baselines.append(baseline_episode(instance, method, seed))
        rows = [row for row in baselines if row["instance_id"] == instance["id"]]
        nn = next(row for row in rows if row["method"] == "nearest_neighbor")
        selector = next(row for row in rows if row["method"] == "shortest_edge_selector")
        require(nn["decisions"] == selector["decisions"], "Shortest-edge selector must equal NN.")
    root.mkdir(parents=True)
    write_json(root / "instances.json", instances)
    write_json(root / "a-reuse.json", a_rows)
    write_json(root / "a-provenance.json", provenance)
    write_json(root / "a-candidate-coverage.json", a_candidate_coverage)
    write_json(root / "baselines.json", baselines)
    write_json(root / "prior-art.json", prior_art())
    jobs = [{"instance_id": instance["id"], "arm": arm} for instance in instances for arm in ARMS]
    random.Random(SEED).shuffle(jobs)
    write_json(root / "schedule.json", jobs)
    hashes = code_hashes()
    freeze(root, hashes)
    manifest = {
        "protocol": PROTOCOL, "root": str(root), "created_at": public.now(),
        "frozen_before_any_new_model_call": True, "seed": SEED,
        "model_configs": {name: asdict(configs[name]) for name in MODELS},
        "arms": {"A": "Reused verified-main-v3 all legal cities; complete tsp225 replacements only.",
                 "B": "Anonymous deduplicated proposal cities; full graph and shared candidate numeric features.",
                 "C": "Same deduplicated proposals/order/features at same history, plus rule semantics and mapping."},
        "rules": {"order": abstraction.RULES, "formulas": abstraction.FORMULAS,
                  "definitions": abstraction.DEFINITIONS, "tie_break": "minimum new city ID",
                  "hyperparameters": "Integer coefficients fixed at e, e+h, 2e-r, e-h; no fitting."},
        "deduplication": "One option per proposed CITY, irrespective of rule multiplicity.",
        "option_order": "Shuffle sorted unique cities using local Random(sha256(seed:instance_id:comma_joined_history)); same at matched B/C histories.",
        "forced": "One proposed city: log deterministic step, no call. Final unvisited city + return use public core deterministic closure.",
        "matching": "B/C action sets and visible numeric graph features match CONDITIONAL ON THE SAME HISTORY. Policies may diverge; this is not pure causal identification of planning ability.",
        "decision_mapping": "Every log saves proposals, group membership, numeric candidates, option IDs and selected city; no repair.",
        "baselines": {"fixed_rules": abstraction.RULES, "random_candidate_seeds": RANDOM_SEEDS,
                      "random_selector": "Uniform distinct city/group; B and C random baselines identical by construction.",
                      "deterministic_selector": "minimum immediate edge; NN is in pool, so exactly NN.",
                      "classical_references": ["nearest_neighbor", "nearest_neighbor_two_opt"],
                      "oracle_policy": "None; no posthoc best-rule executable selector."},
        "budgets": {"new_episodes": 126, "episodes_per_model": 42,
                    "max_calls_per_model": 2 * sum(tasks.query_budget(i) for i in instances),
                    "max_calls_total": 6 * sum(tasks.query_budget(i) for i in instances),
                    "concurrency_per_model_across_both_arms": 4, "episode_wall_limit_seconds": EPISODE_SECONDS},
        "probability_policy": "Jev native action, independent strict raw probability audit; no renormalization, no argmax substitution, never retry a failed call.",
        "failures": "First failure terminates episode, retains denominator and None gap. Fatal client errors stop lane and mark remaining scheduled episodes unattempted. No retries or replacement seeds.",
        "timeout": "Adapter timeouts unchanged from A; additional total per-call watchdog timeout_seconds+5, plus 900s total solve deadline per episode.",
        "timing": "B/C solve wall includes preparation, candidate generation, model calls, log fsync and scoring, excludes independent replay. Candidate/setup/API/replay separate. A legacy wall includes replay; do not pretend it is solve-only.",
        "analysis": {
            "unit": "Seven original graphs; average 3 relabel/start conditions within graph. Random seeds average within condition first.",
            "primary": "Per-model C-B paired percentage-point gap delta, graph-macro; negative is better. All scheduled coverage alongside conditional feasible gaps.",
            "secondary": "B-A, C-A paired deltas; comparisons against fixed rules, random candidate, NN, NN+2opt; heterogeneous graph results and choice/candidate collapse.",
            "missing": "No imputation or success-only denominator. Paired gaps use jointly completed same conditions and report counts; graph summaries flag missing conditions.",
            "uncertainty": "Descriptive seven-graph results; no treating 63 trajectories as independent samples and no significance cherry-picking.",
            "inclusion": "Recommend appendix diagnostic if independent verification passes and every model/arm coverage>=95%; otherwise archive as incomplete. Performance-improvement language additionally requires >=1pp graph-macro gain and >=5/7 graph directions, with all models/contrasts disclosed. No automatic main-paper edits.",
        },
        "environment": {"python": platform.python_version(), "executable": sys.executable,
                        "versions": audit.environment_versions()},
        "verified_A_source": {"path": str(verified), "report_sha256": digest(verified / "report.json"),
                              "summary_sha256": digest(verified / "summary.json"),
                              "independent_verification": summary["independent_verification"]},
        "input_sha256": {name: digest(root / name) for name in
                         ("instances.json", "a-reuse.json", "a-provenance.json", "a-candidate-coverage.json",
                          "baselines.json", "prior-art.json", "schedule.json")},
        "code_sha256": hashes,
    }
    write_json(root / "protocol.json", manifest)
    (root / "protocol.sha256").write_text(digest(root / "protocol.json") + "\n")
    return manifest


def load(root):
    root = Path(root).resolve()
    manifest = read_json(root / "protocol.json")
    require(manifest["root"] == str(root) and manifest["protocol"] == PROTOCOL, "Wrong supplement.")
    require(digest(root / "protocol.json") == (root / "protocol.sha256").read_text().strip(),
            "Protocol checksum drift.")
    require(code_hashes() == manifest["code_sha256"], "Execute matching frozen source.")
    audit.verify_source_snapshot(root, manifest["code_sha256"])
    require(all(digest(root / name) == value for name, value in manifest["input_sha256"].items()),
            "Supplement input drift.")
    return root, manifest, {i["id"]: i for i in read_json(root / "instances.json")}


def replay_episode(instance, rows, config, arm):
    started = time.perf_counter()
    prepared = abstraction.prepare(instance)
    matrix = independent_matrix(instance)
    decisions, calls, forced = [], 0, 0
    failed = False
    probability_failures = 0
    for index, row in enumerate(rows):
        require(not failed and row["step"] == index, "Steps after failure or wrong ordering.")
        payload = abstraction.step(prepared, decisions, arm, SEED)
        require(payload is not None, "Unexpected step after completion.")
        require(payload["rule_proposals"] == independent_proposals(matrix, decisions),
                "Independent rule proposals differ.")
        expected = {key: value for key, value in payload.items() if key != "request"}
        require(row["proposal"] == expected, "Proposal/candidate/group replay mismatch.")
        for metric in ("candidate_seconds",):
            require(type(row[metric]) in (int, float) and math.isfinite(row[metric]) and row[metric] >= 0,
                    "Invalid timing.")
        if payload["forced_city"] is not None:
            require(row["kind"] == "forced" and row["attempt"] is None
                    and row["selected_city"] == payload["forced_city"], "Invalid forced transition.")
            forced += 1
        else:
            require(row["kind"] == "model" and type(row["attempt"]) is dict, "Missing model attempt.")
            attempt = row["attempt"]
            base = {k: v for k, v in attempt.items() if k != "probability_audit"}
            audit.validate_attempt(base, payload["request"], config)
            require(attempt["instance_id"] == instance["id"] and attempt["step"] == index,
                    "Attempt identity mismatch.")
            vector = attempt["probability_audit"]
            if config.provider == "typesafe" and attempt["status"] == "success":
                require(type(vector) is dict, "Missing probability audit.")
                body = {"model": config.model, "usage": attempt["usage"], "answers": {"decision": {
                    "type": "choice", "choice": attempt["selected_option_id"],
                    "probabilities": vector["probabilities"], "confidence": vector["confidence"]}}}
                response = public.TypeSafeClient._parse_response(
                    payload["request"], body, require_probabilities=False)
                require(public.probability_audit(response, payload["request"], config) == vector,
                        "Probability audit mismatch.")
                probability_failures += bool(vector["diagnostic_code"])
            else:
                require(vector is None, "Unexpected probability audit.")
            calls += 1
            if attempt["status"] == "failure":
                require(row["selected_city"] is None, "Failed call must not execute a city.")
                failed = True
                continue
            city = abstraction.selected_city(payload, attempt["selected_option_id"])
            require(row["selected_city"] == city, "Native action/city mapping mismatch.")
        decisions.append(str(row["selected_city"]))
    result = {
        "instance_id": instance["id"], "dataset_id": instance["dataset_id"],
        "replicate": instance["replicate"], "arm": arm, "decisions": decisions,
        "model_calls": calls, "forced_candidate_steps": forced,
        "failure_count": int(failed), "probability_audit_failures": probability_failures,
        **tasks.score(instance, decisions),
    }
    verify_score(instance, result)
    return result, time.perf_counter() - started


async def execute_episode(directory, instance, config, arm, client, abort):
    directory.mkdir(parents=True, exist_ok=False)
    audit.set_status(directory, {"status": "running", "started_at": public.now()})
    start = time.perf_counter()
    prepared = abstraction.prepare(instance)
    setup_seconds = time.perf_counter() - start
    decisions, rows = [], []
    terminal_reason = None
    with (directory / "steps.jsonl").open("x") as stream:
        while not abort.is_set():
            started = time.perf_counter()
            payload = abstraction.step(prepared, decisions, arm, SEED)
            candidate_seconds = time.perf_counter() - started
            if payload is None:
                break
            row = {
                "step": len(decisions), "proposal": {k: v for k, v in payload.items() if k != "request"},
                "candidate_seconds": candidate_seconds, "attempt": None,
                "kind": "forced" if payload["forced_city"] is not None else "model",
                "selected_city": payload["forced_city"],
            }
            if payload["forced_city"] is None:
                request = payload["request"]
                call_start = time.perf_counter()
                try:
                    remaining = EPISODE_SECONDS - (time.perf_counter() - start)
                    if remaining <= 0:
                        raise ClientTimeoutError("Episode solve deadline exhausted.")
                    try:
                        response = await asyncio.wait_for(
                            client.predict(request), timeout=min(config.timeout_seconds + 5, remaining))
                    except TimeoutError:
                        raise ClientTimeoutError("Total call/episode deadline exhausted.") from None
                    attempt = audit.response_record(response, request, config)
                    attempt["probability_audit"] = public.probability_audit(response, request, config)
                    row["selected_city"] = abstraction.selected_city(payload, response.selected_option_id)
                except DecisionClientError as exc:
                    attempt = {
                        "status": "failure", "selected_option_id": None, "resolved_model": None,
                        "usage": audit.safe_usage(getattr(exc, "usage", None)),
                        "probability_audit": None, **audit.failure(exc),
                    }
                    terminal_reason = attempt["diagnostic_code"]
                    if attempt["fatal"]:
                        abort.set()
                attempt.update(
                    instance_id=instance["id"], step=len(decisions),
                    request=audit.json_value(asdict(request)), latency_seconds=time.perf_counter() - call_start,
                )
                row["attempt"] = attempt
            audit.append_row(stream, row)
            rows.append(row)
            if row["selected_city"] is None:
                break
            decisions.append(str(row["selected_city"]))
    scored = tasks.score(instance, decisions)
    solve_seconds = time.perf_counter() - start
    result, verification_seconds = replay_episode(instance, rows, config, arm)
    require(all(result[key] == value for key, value in scored.items()), "Scoring mismatch.")
    result.update(
        setup_seconds=setup_seconds, candidate_seconds=sum(r["candidate_seconds"] for r in rows),
        model_call_seconds=sum(r["attempt"]["latency_seconds"] for r in rows if r["attempt"]),
        solve_wall_seconds=solve_seconds, verification_seconds=verification_seconds,
        terminal_reason=terminal_reason or (None if result["feasible"] else "lane_aborted"),
    )
    write_json(directory / "score.json", result)
    audit.set_status(directory, {
        "status": "completed" if result["feasible"] else "incomplete",
        "finished_at": public.now(), "solve_wall_seconds": solve_seconds,
        "verification_seconds": verification_seconds,
        "artifact_sha256": {name: digest(directory / name) for name in ("steps.jsonl", "score.json")},
    })
    print(json.dumps({"episode": str(directory), "status": result["status"],
                      "calls": result["model_calls"], "forced": result["forced_candidate_steps"]}),
          flush=True)


async def run_lane(root, manifest, instances, name):
    lane = root / "runs" / name
    lane.mkdir(parents=True, exist_ok=False)
    config = ModelConfig(**manifest["model_configs"][name])
    audit.set_status(lane, {"status": "running", "started_at": public.now(), "model": name})
    abort = asyncio.Event()
    jobs = read_json(root / "schedule.json")
    queue = asyncio.Queue()
    for job in jobs:
        queue.put_nowait(job)
    errors = []

    async def worker():
        kwargs = config.client_kwargs()
        if config.provider == "vllm":
            kwargs["api_key"] = ""
        client = None
        try:
            client = public.create_public_client(config.provider, **kwargs)
            await client.initialize()
            while not abort.is_set():
                try:
                    job = queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                await execute_episode(lane / job["arm"] / job["instance_id"],
                                      instances[job["instance_id"]], config, job["arm"], client, abort)
        except Exception as exc:
            abort.set()
            errors.append({"type": type(exc).__name__, "diagnostic_code": audit.failure(exc)["diagnostic_code"]})
            print(json.dumps({"lane": name, "fatal_worker_error": type(exc).__name__}), flush=True)
        finally:
            if client is not None:
                try:
                    await client.aclose()
                except Exception as exc:
                    abort.set()
                    errors.append({"type": type(exc).__name__, "diagnostic_code": "shutdown_failed"})
                    print(json.dumps({"lane": name, "shutdown_error": type(exc).__name__}), flush=True)

    await asyncio.gather(*(worker() for _ in range(4)))
    # Never silently drop queued episodes after a fatal failure.
    for job in jobs:
        directory = lane / job["arm"] / job["instance_id"]
        if not directory.exists():
            directory.mkdir(parents=True)
            with (directory / "steps.jsonl").open("x"):
                pass
        status = read_json(directory / "status.json") if (directory / "status.json").exists() else {}
        if status.get("status") not in ("completed", "incomplete"):
            rows = audit.read_rows(directory / "steps.jsonl")
            result, elapsed = replay_episode(instances[job["instance_id"]], rows, config, job["arm"])
            result.update(setup_seconds=None, candidate_seconds=sum(r["candidate_seconds"] for r in rows),
                          model_call_seconds=sum(r["attempt"]["latency_seconds"] for r in rows if r["attempt"]),
                          solve_wall_seconds=None, verification_seconds=elapsed,
                          terminal_reason="fatal_lane_abort")
            if not (directory / "score.json").exists():
                write_json(directory / "score.json", result)
            audit.set_status(directory, {
                "status": "incomplete", "finished_at": public.now(),
                "artifact_sha256": {file: digest(directory / file) for file in ("steps.jsonl", "score.json")},
            })
    audit.set_status(lane, {"status": "aborted" if abort.is_set() else "completed",
                            "finished_at": public.now(), "errors": errors})


async def run(root):
    root, manifest, instances = load(root)
    require(not (root / "runs").exists(), "No overwrite/resampling of existing supplement lanes.")
    shared.credential_preflight(MODELS)
    with shared.lane_locks(MODELS):
        configs = [SimpleNamespace(model=ModelConfig(**manifest["model_configs"][name])) for name in MODELS]
        await shared.vllm_preflight(configs)
        audit.set_status(root, {"status": "running", "pid": os.getpid(), "started_at": public.now(),
                                "protocol_sha256": digest(root / "protocol.json")})
        await asyncio.gather(*(run_lane(root, manifest, instances, name) for name in MODELS))
    audit.set_status(root, {"status": "verifying", "pid": os.getpid(), "at": public.now()})
    analyze(root)
    audit.set_status(root, {"status": "analysis_ready", "finished_at": public.now(),
                            "protocol_sha256": digest(root / "protocol.json")})


def mean(values):
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else None


def write_csv(path, rows, fields=None):
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(root):
    root, manifest, instances = load(root)
    destination = root / "analysis"
    require(not destination.exists(), "Never overwrite analysis.")
    rows = deepcopy(read_json(root / "a-reuse.json"))
    audits, choice_stats, candidate_stats, verification = [], [], [], []
    for name in MODELS:
        config = ModelConfig(**manifest["model_configs"][name])
        for job in read_json(root / "schedule.json"):
            arm, identity = job["arm"], job["instance_id"]
            directory = root / "runs" / name / arm / identity
            status = read_json(directory / "status.json")
            require(status["status"] in ("completed", "incomplete"), "Every scheduled episode must be terminal.")
            require(all(digest(directory / file) == value for file, value in status["artifact_sha256"].items()),
                    "Episode seal drift.")
            log = audit.read_rows(directory / "steps.jsonl")
            expected, elapsed = replay_episode(instances[identity], log, config, arm)
            result = read_json(directory / "score.json")
            require(all(result[key] == value for key, value in expected.items()), "Score replay failed.")
            rows.append({**result, "model": name})
            for step in log:
                proposals = step["proposal"]
                selected = step["selected_city"]
                counts = Counter(proposals["rule_proposals"].values())
                candidate_stats.append({
                    "model": name, "arm": arm, "instance_id": identity,
                    "dataset_id": instances[identity]["dataset_id"], "step": step["step"],
                    "unvisited": proposals["unvisited_count"], "candidates": proposals["distinct_candidate_count"],
                    "candidate_coverage": proposals["candidate_coverage"],
                    "all_rules_agree": proposals["distinct_candidate_count"] == 1,
                    "pairwise_rule_agreement": sum(n * (n - 1) / 2 for n in counts.values()) / 6,
                    "forced": step["kind"] == "forced",
                })
                if selected is not None:
                    chosen = next(row for row in proposals["candidates"] if row["city"] == selected)
                    for rule in chosen["rules"]:
                        choice_stats.append({
                            "model": name, "arm": arm, "instance_id": identity,
                            "dataset_id": instances[identity]["dataset_id"], "step": step["step"],
                            "rule": rule, "fractional_credit": 1 / len(chosen["rules"]),
                            "forced": step["kind"] == "forced",
                            "option_id": chosen["option_id"], "group": "|".join(chosen["rules"]),
                        })
                attempt = step["attempt"]
                if attempt:
                    audits.append({"model": name, "arm": arm, "instance_id": identity, "step": step["step"],
                                   "status": attempt["status"], "diagnostic": attempt["diagnostic_code"],
                                   "probability_diagnostic": (
                                       attempt["probability_audit"]["diagnostic_code"]
                                       if attempt["probability_audit"] else None)})
            verification.append({
                "model": name, "arm": arm, "instance_id": identity,
                "independent_replay_seconds": elapsed, "status": expected["status"],
                "artifact_sha256": status["artifact_sha256"],
            })
    # Recheck A seals and objective provenance without borrowing the wrong tsp225 history.
    for entry in read_json(root / "a-provenance.json"):
        path = Path(entry["root"]) / "runs" / entry["model"] / entry["instance_id"]
        require(all(digest(path / file) == expected for file, expected in entry["sha256"].items()),
                "Reused A source drift.")
    for row in rows:
        verify_score(instances[row["instance_id"]], row)
    baseline_rows = read_json(root / "baselines.json")
    for row in baseline_rows:
        verify_score(instances[row["instance_id"]], row)
    flat = [{
        "model": row["model"], "arm": row["arm"], "instance_id": row["instance_id"],
        "dataset_id": row["dataset_id"], "replicate": row["replicate"],
        "status": row["status"], "feasible": row["feasible"], "objective": row["objective"],
        "gap_percent": row["percentage_gap"], "calls": row["model_calls"],
        "forced_candidate_steps": row["forced_candidate_steps"],
        "candidate_seconds": row.get("candidate_seconds"), "setup_seconds": row.get("setup_seconds"),
        "model_call_seconds": row["model_call_seconds"], "solve_wall_seconds": row["solve_wall_seconds"],
        "legacy_wall_including_replay": row.get("legacy_wall_seconds_including_replay"),
        "verification_seconds": row["verification_seconds"],
        "failure_count": row["failure_count"], "probability_audit_failures": row["probability_audit_failures"],
    } for row in rows]
    by_graph = []
    datasets = sorted({instance["dataset_id"] for instance in instances.values()})
    for name in MODELS:
        for arm in ("A", *ARMS):
            for dataset in datasets:
                own = [row for row in flat if (row["model"], row["arm"], row["dataset_id"]) == (name, arm, dataset)]
                require(len(own) == 3, "Scheduled graph denominator mismatch.")
                by_graph.append({
                    "model": name, "arm": arm, "dataset_id": dataset,
                    "scheduled": 3, "completed": sum(row["feasible"] for row in own),
                    "gap_percent": mean([row["gap_percent"] for row in own]),
                    "calls": mean([row["calls"] for row in own]),
                    "solve_wall_seconds": mean([row["solve_wall_seconds"] for row in own]),
                    "legacy_wall_including_replay": mean([row["legacy_wall_including_replay"] for row in own]),
                    "model_call_seconds": mean([row["model_call_seconds"] for row in own]),
                })
    macro = []
    for name in MODELS:
        for arm in ("A", *ARMS):
            graphs = [r for r in by_graph if (r["model"], r["arm"]) == (name, arm)]
            own = [r for r in flat if (r["model"], r["arm"]) == (name, arm)]
            macro.append({
                "model": name, "arm": arm, "graphs": 7, "scheduled": 21,
                "completed": sum(r["completed"] for r in graphs),
                "coverage": sum(r["completed"] for r in graphs) / 21,
                "gap_percent_graph_macro": mean([r["gap_percent"] for r in graphs]),
                "calls_graph_macro": mean([r["calls"] for r in graphs]),
                "solve_wall_seconds_graph_macro": mean([r["solve_wall_seconds"] for r in graphs]),
                "legacy_wall_including_replay_graph_macro": mean([r["legacy_wall_including_replay"] for r in graphs]),
                "model_call_seconds_graph_macro": mean([r["model_call_seconds"] for r in graphs]),
                "probability_audit_failures": sum(r["probability_audit_failures"] for r in own),
            })
    paired = []
    for name in MODELS:
        for left, right in (("C", "B"), ("B", "A"), ("C", "A")):
            for dataset in datasets:
                a = {r["replicate"]: r for r in flat
                     if (r["model"], r["arm"], r["dataset_id"]) == (name, left, dataset)}
                b = {r["replicate"]: r for r in flat
                     if (r["model"], r["arm"], r["dataset_id"]) == (name, right, dataset)}
                complete = [rep for rep in range(3) if a[rep]["feasible"] and b[rep]["feasible"]]
                paired.append({
                    "model": name, "contrast": f"{left}-{right}", "dataset_id": dataset,
                    "scheduled_pairs": 3, "complete_pairs": len(complete),
                    "gap_delta_pp": mean([a[rep]["gap_percent"] - b[rep]["gap_percent"] for rep in complete]),
                    "calls_delta": mean([a[rep]["calls"] - b[rep]["calls"] for rep in complete]),
                })
    contrast_macro = []
    for name in MODELS:
        for contrast in ("C-B", "B-A", "C-A"):
            own = [r for r in paired if (r["model"], r["contrast"]) == (name, contrast)]
            contrast_macro.append({
                "model": name, "contrast": contrast,
                "graph_macro_delta_pp": mean([r["gap_delta_pp"] for r in own]),
                "improved_graphs": sum(r["gap_delta_pp"] is not None and r["gap_delta_pp"] < 0 for r in own),
                "worse_graphs": sum(r["gap_delta_pp"] is not None and r["gap_delta_pp"] > 0 for r in own),
                "jointly_complete_conditions": sum(r["complete_pairs"] for r in own),
                "graph_count": sum(r["gap_delta_pp"] is not None for r in own),
            })
    baseline_summary = []
    for method in sorted({row["method"] for row in baseline_rows}):
        graph_gaps, graph_times = [], []
        for dataset in datasets:
            # Seeds first, then relabel/start conditions, then equal graph weight.
            condition_gaps, condition_times = [], []
            for replicate in range(3):
                own = [r for r in baseline_rows if (r["method"], r["dataset_id"], r["replicate"])
                       == (method, dataset, replicate)]
                condition_gaps.append(mean([r["percentage_gap"] for r in own]))
                condition_times.append(mean([r["solve_wall_seconds"] for r in own]))
            graph_gaps.append(mean(condition_gaps))
            graph_times.append(mean(condition_times))
        baseline_summary.append({"method": method, "graphs": 7, "coverage": 1.0,
                                 "gap_percent_graph_macro": mean(graph_gaps),
                                 "solve_wall_seconds_graph_macro": mean(graph_times)})
    behavior = []
    for name in MODELS:
        for arm in ARMS:
            own = [row for row in candidate_stats if (row["model"], row["arm"]) == (name, arm)]
            choices = [r for r in choice_stats if (r["model"], r["arm"]) == (name, arm) and not r["forced"]]
            credit = {rule: sum(r["fractional_credit"] for r in choices if r["rule"] == rule)
                      for rule in abstraction.RULES}
            total = sum(credit.values())
            distribution = {rule: value / total if total else None for rule, value in credit.items()}
            behavior.append({
                "model": name, "arm": arm, "observed_steps": len(own),
                "candidate_count_histogram": dict(Counter(r["candidates"] for r in own)),
                "all_rules_agree_rate": mean([r["all_rules_agree"] for r in own]),
                "pairwise_rule_agreement_rate": mean([r["pairwise_rule_agreement"] for r in own]),
                "mean_candidate_fraction_of_legal": mean([r["candidate_coverage"] for r in own]),
                "nonforced_rule_fractional_distribution": distribution,
                "maximum_rule_share": max(distribution.values()) if total else None,
                "interpretation": "B credits are retrospective rule membership, not explicit semantic rule choices. Forced steps excluded from selection distribution.",
            })
    sufficient = all(row["coverage"] >= .95 for row in macro)
    recommendation = {
        "recommendation": "appendix_diagnostic" if sufficient else "archive_incomplete",
        "independent_verification": "passed", "coverage_threshold_satisfied": sufficient,
        "main_paper_modified": False,
        "claims": [
            "This is a matched-action-set conditional-on-history comparison, not isolated planning ability.",
            "Seven original graphs, three correlated relabel/start conditions; no independent n=63 claim.",
            "A/B and A/C combine candidate restriction and additional arithmetic assistance; C/B adds semantics.",
            "One-candidate agreement saves real calls; equal API-call counts are not claimed.",
            "A true solve-only wall cannot be recovered from legacy timing; its total includes replay.",
        ],
        "positive_direction_flags_not_significance_tests": [
            {**row, "meets_frozen_direction_threshold":
             row["graph_macro_delta_pp"] is not None and row["graph_macro_delta_pp"] <= -1
             and row["improved_graphs"] >= 5 and row["jointly_complete_conditions"] == 21}
            for row in contrast_macro
        ],
        "include_all_outcomes": True, "no_rule_pool_or_model_changes_after_results": True,
    }
    destination.mkdir()
    write_csv(destination / "episodes.csv", flat)
    write_csv(destination / "per-graph.csv", by_graph)
    write_csv(destination / "summary.csv", macro)
    write_csv(destination / "paired-deltas.csv", paired)
    write_csv(destination / "candidate-steps.csv", candidate_stats, [
        "model", "arm", "instance_id", "dataset_id", "step", "unvisited", "candidates",
        "candidate_coverage", "all_rules_agree", "pairwise_rule_agreement", "forced"])
    write_csv(destination / "rule-selections.csv", choice_stats, [
        "model", "arm", "instance_id", "dataset_id", "step", "rule",
        "fractional_credit", "forced", "option_id", "group"])
    write_csv(destination / "call-audits.csv", audits, [
        "model", "arm", "instance_id", "step", "status", "diagnostic", "probability_diagnostic"])
    a_coverage = read_json(root / "a-candidate-coverage.json")
    a_candidate_summary = []
    for name in MODELS:
        graph_rates = [
            mean([row["selected_city_in_candidates"] for row in a_coverage
                  if (row["model"], row["dataset_id"]) == (name, dataset)])
            for dataset in datasets
        ]
        a_candidate_summary.append({"model": name, "graphs": 7,
                                    "a_chosen_city_in_candidates_graph_macro": mean(graph_rates)})
    write_json(destination / "summary.json", {"models": macro, "contrasts": contrast_macro,
                                            "baselines": baseline_summary, "behavior": behavior,
                                            "A_candidate_coverage": a_candidate_summary})
    write_json(destination / "paper-recommendation.json", recommendation)
    write_json(destination / "verification-receipt.json", {
        "status": "passed", "protocol_sha256": digest(root / "protocol.json"),
        "input_sha256": manifest["input_sha256"], "code_sha256": manifest["code_sha256"],
        "new_scheduled_terminal_episodes": len(verification), "reused_A_episodes": len(rows) - len(verification),
        "baseline_episodes": len(baseline_rows),
        "independence": "Separate direct-double-sqrt matrix and exhaustive rule-score minima; full option/request/mapping replay, independent completed-tour objective and signed gap; artifact seals rechecked.",
        "episodes": verification,
        "analysis_sha256": {path.name: digest(path) for path in destination.iterdir() if path.is_file()},
    })
    text = [
        "TSPLIB append-only action abstraction supplement; descriptive seven-graph macro analysis.",
        "All contrasts negative = better; every model and arm retained.",
        *[json.dumps(row) for row in contrast_macro],
        "Recommendation: " + recommendation["recommendation"],
        *recommendation["claims"],
    ]
    (destination / "interpretation.txt").write_text("\n".join(text) + "\n")
    print(json.dumps({"status": "analysis_ready", "directory": str(destination),
                      "terminal_episodes": len(verification)}), flush=True)
    return recommendation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "run", "analyze"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--verified", type=Path)
    args = parser.parse_args()
    if args.command == "plan":
        require(args.verified is not None, "plan requires verified A directory.")
        result = plan(args.root, args.verified)
        print(json.dumps({"root": result["root"], "budgets": result["budgets"],
                          "protocol_sha256": digest(args.root / "protocol.json")}), flush=True)
    elif args.command == "run":
        asyncio.run(run(args.root))
    else:
        analyze(args.root)


if __name__ == "__main__":
    main()
