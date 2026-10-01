#!/usr/bin/env python3
"""Full-input public graph experiments with pinned inputs and replayable episodes."""

from __future__ import annotations

import argparse
import asyncio
from copy import deepcopy
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.utils import extended_graph_suite as audit
from src.utils import paired_graph_ablation as shared
from src.benchmark import public_tasks as tasks
from src.benchmark.config import ModelConfig
from src.clients.base import DecisionClientError, InvalidResponseError
from src.clients.registry import create_client
from src.clients.typesafe import TypeSafeClient

DEFAULT_MODELS = ("jev", "qwen08", "qwen2", "qwen4", "qwen9")
PROTOCOL = "public-full-input-v3"
SEED = 20260929
require = shared.require
digest = shared.digest
write_json = audit.write_json
read_json = audit.read_json
PROBABILITY_DIAGNOSTICS = {
    "invalid_probabilities", "invalid_probability_sum",
    "choice_probability_mismatch", "invalid_confidence",
}


def create_public_client(provider, **kwargs):
    if provider == "typesafe":
        kwargs["require_probabilities"] = False
    return create_client(provider, **kwargs)


def probability_audit(response, request, config):
    if config.provider != "typesafe":
        return None
    code = None
    try:
        TypeSafeClient._parse_response(request, response.raw_output)
    except InvalidResponseError as exc:
        require(exc.diagnostic_code in PROBABILITY_DIAGNOSTICS,
                "Native-action response violates non-probability fields.")
        code = exc.diagnostic_code
    answer = response.raw_output["answers"]["decision"]
    probabilities = answer.get("probabilities")
    if not (type(probabilities) is dict and set(probabilities) == {o.id for o in request.options}
            and all(type(p) in (int, float) and math.isfinite(p) and 0 <= p <= 1
                    for p in probabilities.values())):
        probabilities = None
    confidence = answer.get("confidence")
    if not (type(confidence) in (int, float) and math.isfinite(confidence) and 0 <= confidence <= 1):
        confidence = None
    return {
        "diagnostic_code": code, "probabilities": probabilities, "confidence": confidence,
        "sum": math.fsum(probabilities.values()) if probabilities is not None else None,
    }


def now():
    return datetime.now(timezone.utc).isoformat()


def code_hashes():
    return audit.code_hashes()


def freeze_source(root, hashes):
    snapshot = root / audit.SOURCE_SNAPSHOT
    snapshot.mkdir()
    for name, expected in hashes.items():
        source, target = REPO / name, snapshot / name
        require(not source.is_symlink() and source.resolve().is_relative_to(REPO),
                "Unsafe source snapshot path.")
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(source.read_bytes())
            stream.flush()
            os.fsync(stream.fileno())
        require(digest(target) == expected, "Source changed while freezing.")
        target.chmod(0o444)
    audit.verify_source_snapshot(root, hashes)
    require(code_hashes() == hashes, "Source changed while freezing.")


def configurations(names):
    require(bool(names) and len(set(names)) == len(names) and set(names) <= set(shared.MODELS),
            "Choose distinct supported model names.")
    configs = shared.model_configs("answer_only", names)
    return {name: replace(config, constrain_choices=True, max_tokens=16)
            if config.provider == "vllm" else config for name, config in configs.items()}


def relabel(instance, replicate, seed):
    result = deepcopy(instance)
    n = len(instance["state"]["nodes"])
    original_ids = list(range(n))
    random.Random(f"{seed}:{instance['id']}:{replicate}").shuffle(original_ids)
    inverse = {old: new for new, old in enumerate(original_ids)}
    state = result["state"]
    if instance["task"] == "tsp_public":
        state["coordinates"] = [instance["state"]["coordinates"][old] for old in original_ids]
    else:
        state["edges"] = sorted(
            [min(inverse[u], inverse[v]), max(inverse[u], inverse[v]), weight]
            for u, v, weight in instance["state"]["edges"])
    result.update(dataset_id=instance["id"], replicate=replicate,
                  original_ids=original_ids,
                  id=hashlib.sha256(
                      f"{instance['id']}:{replicate}:{seed}".encode()).hexdigest()[:20])
    tasks.validate_instance(result)
    return result


def baseline_rows(instances, seed):
    result = []
    for instance in instances:
        for method in tasks.BASELINES[instance["task"]]:
            start = time.perf_counter()
            decisions = tasks.baseline(instance, method, seed=seed + instance["replicate"])
            scored = tasks.score(instance, decisions)
            elapsed = time.perf_counter() - start
            require(scored["feasible"], "Classical baseline returned an infeasible solution.")
            result.append({"instance_id": instance["id"], "method": method,
                           "decisions": decisions, "wall_seconds": elapsed, **scored})
    return result


def plan(root, inputs, *, models=DEFAULT_MODELS, repeats=3, concurrency=4,
         instance_ids=None, seed=SEED):
    root, inputs = Path(root).resolve(), Path(inputs).resolve()
    require(type(repeats) is int and repeats >= 1, "Replicates must be positive.")
    require(type(concurrency) is int and 1 <= concurrency <= 16,
            "Episode concurrency must be in 1..16.")
    require(type(seed) is int and seed >= 0, "Seed must be a nonnegative integer.")
    configs = configurations(tuple(models))
    source = read_json(inputs)
    require(type(source) is list and bool(source), "Inputs must be a nonempty JSON list.")
    require(len({i["id"] for i in source}) == len(source), "Duplicate source instance IDs.")
    for instance in source:
        tasks.validate_instance(instance)
    if instance_ids is not None:
        require(bool(instance_ids) and len(set(instance_ids)) == len(instance_ids)
                and set(instance_ids) <= {i["id"] for i in source},
                "Select distinct existing dataset IDs.")
        source = [i for i in source if i["id"] in instance_ids]
    instances = [relabel(i, repeat, seed) for repeat in range(repeats) for i in source]
    random.Random(seed).shuffle(instances)
    baselines = baseline_rows(instances, seed)
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "source-instances.json", source)
    write_json(root / "instances.json", instances)
    write_json(root / "baselines.json", baselines)
    hashes = code_hashes()
    freeze_source(root, hashes)
    manifest = {
        "schema_version": 1, "protocol": PROTOCOL, "root": str(root), "created_at": now(),
        "typesafe_readout": "native_action_with_independent_probability_audit",
        "tsp_distance_formula": "double_sqrt_dx_squared_plus_dy_squared_then_nint",
        "input_source": {"path": str(inputs), "sha256": digest(inputs)},
        "seed": seed, "repeats": repeats, "concurrency_per_model": concurrency,
        "model_configs": {name: asdict(config) for name, config in configs.items()},
        "input_sha256": {name: digest(root / name) for name in
                         ("source-instances.json", "instances.json", "baselines.json")},
        "code_sha256": hashes,
        "budgets": {"distinct_instances": len(source), "episodes_per_model": len(instances),
                    "max_calls_per_model": sum(tasks.query_budget(i) for i in instances)},
        "protocol_notes": [
            "Complete graph input and all legal actions; no candidate filtering.",
            "TSPLIB EUC_2D coordinates and exact current-city distances; integer rounding.",
            "TSP n<255; forced last city and return are not model calls.",
            "Three replicates, when requested, are relabel/start conditions, not independent graphs.",
            "No retry, output repair, reference disclosure, or model-guided postprocessing.",
            "Jev legal native actions are scored independently of probability conformance; "
            "probability vectors are audited and never renormalized or used to replace actions.",
            "Qwen uses grammar-constrained answer-only decoding, thinking disabled.",
            "Concurrent episodes share an endpoint; latency is observed service time, not intrinsic speed.",
            "GPT references retain their separate reasoning budget, when included.",
        ],
    }
    write_json(root / "manifest.json", manifest)
    (root / "manifest.sha256").write_text(digest(root / "manifest.json") + "\n", encoding="ascii")
    return manifest


def load_plan(root):
    root = Path(root).resolve()
    require(digest(root / "manifest.json") == (root / "manifest.sha256").read_text().strip(),
            "Manifest checksum mismatch.")
    manifest = read_json(root / "manifest.json")
    require(manifest["protocol"] == PROTOCOL and manifest["root"] == str(root),
            "Wrong protocol or relocated plan.")
    require(manifest["typesafe_readout"] == "native_action_with_independent_probability_audit",
            "Native-action probability policy drift.")
    require(manifest["tsp_distance_formula"]
            == "double_sqrt_dx_squared_plus_dy_squared_then_nint", "TSP formula drift.")
    require(code_hashes() == manifest["code_sha256"],
            "Source drift: execute the matching frozen source, never edit the manifest.")
    audit.verify_source_snapshot(root, manifest["code_sha256"])
    require(all(digest(root / name) == value
                for name, value in manifest["input_sha256"].items()), "Input checksum mismatch.")
    instances = read_json(root / "instances.json")
    for instance in instances:
        tasks.validate_instance(instance)
    source = read_json(root / "source-instances.json")
    expected = [relabel(i, repeat, manifest["seed"])
                for repeat in range(manifest["repeats"]) for i in source]
    random.Random(manifest["seed"]).shuffle(expected)
    require(instances == expected, "Relabel/start conditions differ from source records.")
    configs = {name: ModelConfig(**value) for name, value in manifest["model_configs"].items()}
    require(configs == configurations(tuple(configs)), "Model settings drift.")
    return root, manifest, instances, configs


def replay_episode(instance, rows, config):
    decisions, failed = [], False
    for step, row in enumerate(rows):
        require(not failed and row["instance_id"] == instance["id"] and row["step"] == step,
                "Episode history/order is invalid.")
        request = tasks.next_request(instance, decisions)
        require(request is not None, "Attempt after completed episode.")
        require("probability_audit" in row, "Missing explicit probability audit.")
        base = {key: value for key, value in row.items() if key != "probability_audit"}
        audit.validate_attempt(base, request, config)
        vector = row["probability_audit"]
        if config.provider == "typesafe" and row["status"] == "success":
            require(type(vector) is dict and set(vector) ==
                    {"diagnostic_code", "probabilities", "confidence", "sum"}
                    and vector["diagnostic_code"] in PROBABILITY_DIAGNOSTICS | {None},
                    "Invalid probability audit fields.")
            body = {"model": config.model, "usage": row["usage"], "answers": {"decision": {
                "type": "choice", "choice": row["selected_option_id"],
                "probabilities": vector["probabilities"], "confidence": vector["confidence"]}}}
            response = TypeSafeClient._parse_response(request, body, require_probabilities=False)
            require(probability_audit(response, request, config) == vector,
                    "Probability audit does not replay.")
        else:
            require(vector is None, "Unexpected probability audit.")
        if row["status"] == "success":
            decisions.append(row["selected_option_id"])
        else:
            failed = True
    return {
        "instance_id": instance["id"], "dataset_id": instance["dataset_id"],
        "replicate": instance["replicate"], "task": instance["task"],
        "node_count": len(instance["state"]["nodes"]),
        "decisions": decisions, "query_count": len(rows),
        "failure_count": sum(r["status"] == "failure" for r in rows),
        "probability_audit_failures": sum(
            bool(r["probability_audit"] and r["probability_audit"]["diagnostic_code"])
            for r in rows),
        "model_call_seconds": sum(r["latency_seconds"] for r in rows),
        **tasks.score(instance, decisions),
    }


async def run_episode(directory, instance, config, client, abort):
    directory.mkdir(exist_ok=False)
    start = time.perf_counter()
    status = {"status": "running", "started_at": now()}
    audit.set_status(directory, status)
    rows, decisions = [], []
    try:
        with (directory / "attempts.jsonl").open("x", encoding="utf-8") as stream:
            while not abort.is_set():
                request = tasks.next_request(instance, decisions)
                if request is None:
                    break
                public = audit.json_value(asdict(request))
                started = time.perf_counter()
                try:
                    response = await client.predict(request)
                    record = audit.response_record(response, request, config)
                    record["probability_audit"] = probability_audit(response, request, config)
                except DecisionClientError as exc:
                    record = {"status": "failure", "selected_option_id": None,
                              "resolved_model": None, "usage": audit.safe_usage(
                                  getattr(exc, "usage", None)), "probability_audit": None,
                              **audit.failure(exc)}
                record.update(instance_id=instance["id"], step=len(decisions),
                              request=public, latency_seconds=time.perf_counter() - started)
                audit.append_row(stream, record)
                rows.append(record)
                if record["status"] == "failure":
                    if record["fatal"]:
                        abort.set()
                    break
                decisions.append(record["selected_option_id"])
        result = replay_episode(instance, rows, config)
        write_json(directory / "score.json", result)
        status.update(status="completed" if result["feasible"] else "incomplete",
                      wall_seconds=time.perf_counter() - start, finished_at=now(),
                      artifact_sha256={name: digest(directory / name)
                                       for name in ("attempts.jsonl", "score.json")})
    except BaseException:
        status.update(status="interrupted", finished_at=now(),
                      wall_seconds=time.perf_counter() - start)
        abort.set()
        raise
    finally:
        audit.set_status(directory, status)


async def run_lane(root, manifest, instances, name, config, *, client_factory, preflight):
    lane = root / "runs" / name
    lane.mkdir(parents=True, exist_ok=False)
    status = {"status": "running", "model": name, "started_at": now(),
              "manifest_sha256": digest(root / "manifest.json"),
              "injected_client": client_factory is not create_public_client}
    audit.set_status(lane, status)
    start = time.perf_counter()
    abort = asyncio.Event()
    queue = asyncio.Queue()
    for instance in instances:
        queue.put_nowait(instance)

    async def worker():
        client = client_factory(config.provider, **config.client_kwargs())
        try:
            await client.initialize()
            while not abort.is_set():
                try:
                    instance = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                await run_episode(lane / instance["id"], instance, config, client, abort)
        finally:
            await client.aclose()

    try:
        await preflight([SimpleNamespace(model=config)])
        async with asyncio.TaskGroup() as group:
            for _ in range(min(manifest["concurrency_per_model"], len(instances))):
                group.create_task(worker())
        episode_statuses = [read_json(lane / instance["id"] / "status.json")["status"]
                            for instance in instances if (lane / instance["id"]).exists()]
        status["status"] = (
            "aborted" if abort.is_set() else
            "completed_with_failures" if any(s != "completed" for s in episode_statuses)
            else "completed")
    except BaseException:
        status["status"] = "aborted"
        raise
    finally:
        status.update(finished_at=now(), wall_seconds=time.perf_counter() - start)
        audit.set_status(lane, status)


async def run_live(root, models=None, *, client_factory=create_public_client,
                   preflight=shared.vllm_preflight):
    root, manifest, instances, configs = load_plan(root)
    names = tuple(configs) if models is None else tuple(models)
    require(bool(names) and len(set(names)) == len(names) and set(names) <= set(configs),
            "Models were not frozen in the plan.")
    with shared.lane_locks(names):
        require(all(not (root / "runs" / name).exists() for name in names),
                "A selected lane already exists; no overwrite or implicit rerun.")
        async with asyncio.TaskGroup() as group:
            for name in names:
                group.create_task(run_lane(root, manifest, instances, name, configs[name],
                                           client_factory=client_factory, preflight=preflight))
    return report(root, names)


def average(values):
    return sum(values) / len(values) if values else None


def metrics(rows):
    complete = [row for row in rows if row["feasible"]]
    by_dataset = {}
    for row in complete:
        by_dataset.setdefault(row["dataset_id"], []).append(row["percentage_gap"])
    return {
        "scheduled": len(rows), "completed": len(complete),
        "coverage": len(complete) / len(rows), "distinct_completed_instances": len(by_dataset),
        "mean_gap_percent_feasible": average([row["percentage_gap"] for row in complete]),
        "instance_macro_gap_percent_feasible": average(
            [average(values) for values in by_dataset.values()]),
        "mean_call_seconds_feasible": average([row["model_call_seconds"] for row in complete]),
        "mean_wall_seconds_feasible": average([row["wall_seconds"] for row in complete]),
        "attempts": sum(row["query_count"] for row in rows),
        "explicit_failures": sum(row["failure_count"] for row in rows),
        "probability_audit_failures": sum(row["probability_audit_failures"] for row in rows),
    }


def report(root, models=None):
    root, manifest, instances, configs = load_plan(root)
    names = tuple(configs) if models is None else tuple(models)
    require(bool(names) and len(set(names)) == len(names) and set(names) <= set(configs),
            "Models were not frozen in the plan.")
    result = {"protocol": PROTOCOL, "budgets": manifest["budgets"],
              "manifest_sha256": digest(root / "manifest.json"),
              "aggregation": "Per-task only. Gaps conditional on feasible completion; all planned "
                             "episodes in coverage. Replicates are correlated conditions, not new graphs.",
              "models": {}, "baselines": read_json(root / "baselines.json")}
    for name in names:
        lane = root / "runs" / name
        lane_status = read_json(lane / "status.json") if lane.exists() else {"status": "not_started"}
        rows = []
        for instance in instances:
            directory = lane / instance["id"]
            attempts, elapsed, state = [], None, "not_started"
            if directory.exists():
                status = read_json(directory / "status.json")
                state, elapsed = status["status"], status.get("wall_seconds")
                require(state in ("running", "completed", "incomplete", "interrupted"),
                        "Unknown episode status.")
                if state in ("completed", "incomplete"):
                    require(set(status.get("artifact_sha256", {}))
                            == {"attempts.jsonl", "score.json"},
                            "Terminal episode is missing its artifact seal.")
                    require(type(elapsed) in (int, float) and math.isfinite(elapsed)
                            and elapsed >= 0, "Invalid episode wall time.")
                if "artifact_sha256" in status:
                    require(all(digest(directory / path) == value
                                for path, value in status["artifact_sha256"].items()),
                            "Sealed episode checksum mismatch.")
                path = directory / "attempts.jsonl"
                attempts = audit.read_rows(path) if path.exists() else []
            row = replay_episode(instance, attempts, configs[name])
            if directory.exists() and "artifact_sha256" in status:
                require(row == read_json(directory / "score.json"), "Sealed score replay mismatch.")
                require((state == "completed") == row["feasible"], "Episode status mismatch.")
            # An unsealed log is auditable progress, not a completed evaluation.
            if state not in ("completed", "incomplete"):
                row.update(feasible=False, completed=False, objective=None,
                           additive_gap=None, percentage_gap=None)
            rows.append({**row, "execution_status": state, "wall_seconds": elapsed})
        result["models"][name] = {
            "status": lane_status["status"], "config": asdict(configs[name]), "episodes": rows,
            "injected_client": lane_status.get("injected_client", False),
            "by_task": {task: metrics([row for row in rows if row["task"] == task])
                        for task in sorted({i["task"] for i in instances})},
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preparing = commands.add_parser("prepare", help="Explicitly download and verify the public catalog")
    preparing.add_argument("--root", type=Path, required=True)
    preparing.add_argument("--include-calibration", action="store_true")
    preparing.add_argument("--dataset-ids", nargs="+")
    planning = commands.add_parser("plan")
    planning.add_argument("--root", type=Path, required=True)
    planning.add_argument("--inputs", type=Path, required=True)
    planning.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    planning.add_argument("--instance-ids", nargs="+")
    planning.add_argument("--repeats", type=int, default=3)
    planning.add_argument("--concurrency", type=int, default=4)
    for name in ("run", "report"):
        command = commands.add_parser(name)
        command.add_argument("--root", type=Path, required=True)
        command.add_argument("--models", nargs="+")
    args = parser.parse_args()
    if args.command == "prepare":
        from src.benchmark.public_data import prepare
        result = prepare(args.root, include_calibration=args.include_calibration,
                         dataset_ids=args.dataset_ids)
    elif args.command == "plan":
        result = plan(args.root, args.inputs, models=args.models, repeats=args.repeats,
                      concurrency=args.concurrency, instance_ids=args.instance_ids)
    elif args.command == "run":
        result = asyncio.run(run_live(args.root, args.models))
    else:
        result = report(args.root, args.models)
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
