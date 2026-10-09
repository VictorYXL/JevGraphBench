"""Optional bounded live interface smoke; synthetic data, NOT primary reproduction.

No inference on import, no downloads, no credential/config-file discovery here.
The shared provider adapter uses separately configured authentication when opted in.
One instance per type, A for optimization and TG for graph-text. No accuracy
guarantee, retries, repair, fallback models, or forced-only support claims.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.benchmark import extended_tasks
from src.benchmark.config import ModelConfig
from src.clients.base import BaseDecisionClient, ClientTimeoutError, InvalidResponseError
from src.utils import extended_graph_suite as audit
from src.utils import graph_abstraction_suite as episodes
from src.utils import public_graphtext as graphtext

CALL_SECONDS = 45.0
OVERALL_SECONDS = 600.0
STRUCTURAL = ("adjacency", "degree_exact", "cycle_detection", "pair_connectivity",
              "distance_threshold", "articulation_point")
OPTIMIZATION = ("tsp_construct", "maxcut_construct", "lt_influence_construct",
                "community_bipartition_construct")
TASKS = (*STRUCTURAL, "arxiv", "prime", *OPTIMIZATION)


def fixtures():
    """Build only synthetic, small model-visible inputs using production builders."""
    result = []
    for task in STRUCTURAL:
        state = {"nodes": list(range(4)), "edges": [[0, 1], [1, 2], [2, 3]],
                 "directed": False}
        if task in ("degree_exact", "articulation_point"):
            state["vertex"] = 1
        if task in ("adjacency", "pair_connectivity", "distance_threshold"):
            state["pair"] = [0, 2]
        if task == "distance_threshold":
            state["threshold"] = 2
        instance = extended_tasks._record(task, 0, state)
        result.append((task, instance, extended_tasks.next_request(instance, []), "A"))
    record = {"target": "synthetic-arxiv", "anchors": [{"node": "anchor", "class": "0"}],
              "texts": [{"node": "target", "text": "Synthetic graph algorithms paper."},
                        {"node": "anchor", "text": "Synthetic graph context."}],
              "edges": [["target", "anchor"]]}
    classes = {str(i): f"Synthetic class {i}" for i in range(40)}
    result.append(("arxiv", None, graphtext.arxiv_request(record, "TG", classes), "TG"))
    nodes = {i: {"name": f"Synthetic entity {i}", "type": "synthetic"} for i in range(43)}
    documents = {i: f"Short synthetic description {i}." for i in nodes}
    adjacency = {i: [(str(i), "synthetic_relation", str(40 + i))] for i in range(3)}
    record = graphtext.evidence_record(
        {"id": 0, "query": "Select a synthetic entity related to entity 40."},
        list(range(40)), [0.0] * 40, nodes, documents, adjacency)
    result.append(("prime", None, graphtext.prime_request(record, "TG"), "TG"))
    bank = extended_tasks.build_instances()
    for task in OPTIMIZATION:
        instance = next(item for item in bank if item["task"] == task)
        result.append((task, instance, None, "A"))
    return result


def diagnostic(exc):
    """Never persist exception messages, response bodies, URLs or headers."""
    if isinstance(exc, TimeoutError):
        exc = ClientTimeoutError("Deadline")
    return {"error_type": type(exc).__name__, **audit.failure(exc)}


class Observations:
    """Transport evidence, independent of predict entry and semantic correctness.

    HTTP counts are outbound request attempts, not confirmed server forwards.
    Copilot counts are observed SDK turns (usage is a lower-bound fallback).
    Hidden server/SDK work cannot be proved absent; observed retries fail closed.
    """

    def __init__(self, root):
        self.root = root
        self.row = None
        self.seen = set()
        self.turns = self.usages = self.http = 0
        self.retried = False

    def begin(self, request):
        if request.request_id in self.seen:
            raise InvalidResponseError("Duplicate request; retries forbidden")
        self.seen.add(request.request_id)
        self.turns = self.usages = self.http = 0
        self.retried = False
        self.row["predict_calls"] += 1

    def event(self, kind, status_code=None):
        if self.row is None:
            return
        before = max(self.turns, self.usages, self.http)
        if kind == "http_request":
            self.http += 1
        elif kind == "assistant.turn_start":
            self.turns += 1
        elif kind == "assistant.usage":
            self.usages += 1
        elif kind == "assistant.turn_retry":
            self.retried = True
        elif kind == "http_response" and type(status_code) is int:
            self.row["http_statuses"].append(status_code)
        after = max(self.turns, self.usages, self.http)
        self.row["actual_calls"] += after - before
        self.row["retry_observed"] |= self.retried or after > 1
        # Metadata only: no error body, event payload, headers, or URL is stored.
        payload = {"event": kind, "http_status": status_code}
        with (self.root / "transport.jsonl").open("a", encoding="utf-8") as stream:
            audit.append_row(stream, {"task": self.row["task"], **payload})
        graphtext.capture_wire("smoke_transport_metadata", payload)


def make_client(config, observations):
    """Create one existing-endpoint adapter, lazily and only after opt-in."""
    if config.provider == "vllm":
        from src.clients.vllm_client import VLLMClient

        client = VLLMClient(**config.client_kwargs())

        async def request_hook(request):
            observations.event("http_request")

        async def response_hook(response):
            observations.event("http_response", response.status_code)

        client._http.event_hooks["request"].append(request_hook)
        client._http.event_hooks["response"].append(response_hook)
        return client
    from src.clients.github_copilot_client import GitHubCopilotClient

    def event_hook(request, event):
        kind = event.get("type")
        if kind in {"assistant.turn_start", "assistant.usage", "assistant.turn_retry",
                    "assistant.message", "session.error"}:
            observations.event(kind)

    return GitHubCopilotClient(**config.client_kwargs(), on_decision_event=event_hook)


class BoundedClient(BaseDecisionClient):
    """One delegation per unique request; normalized response trace without free text."""

    def __init__(self, client, config, observations):
        self.client, self.config, self.observations = client, config, observations

    @property
    def capabilities(self):
        return self.client.capabilities

    def validate_request(self, request):
        self.client.validate_request(request)

    async def predict(self, request):
        self.observations.begin(request)
        try:
            response = await asyncio.wait_for(self.client.predict(request), CALL_SECONDS)
            if self.observations.row["retry_observed"]:
                raise InvalidResponseError("Observed provider retry")
            audit.response_record(response, request, self.config)
            self.observations.row["responses"] += 1
            # Production executors capture this validated normalized response, not
            # arbitrary provider text (including malformed responses with secrets).
            return replace(response, raw_output=None)
        except Exception as exc:
            self.observations.row["errors"].append(diagnostic(exc))
            raise


def model_config(provider, model, base_url=None):
    if provider not in ("vllm", "github_copilot"):
        raise ValueError("Unsupported smoke provider")
    return ModelConfig(
        provider=provider, model=model, base_url=base_url, timeout_seconds=CALL_SECONDS,
        think=False, max_tokens=128, output_format="answer_only",
        reasoning_effort="none" if provider == "github_copilot" else None,
        constrain_choices=True if provider == "vllm" else None)


async def run_smoke(*, provider, model, output, base_url=None, allow_inference=False,
                    client_factory=None):
    """Return a JSON-safe report. Injected factories are explicitly marked offline.

    Exit success concerns complete transport/protocol coverage, NOT accuracy.
    The CLI requires a fresh directory; existing traces are never resumed/retried.
    """
    if allow_inference is not True:
        raise PermissionError("Explicit --allow-inference is required")
    config = model_config(provider, model, base_url)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    rows = [{"task": task, "arm": "TG" if task in ("arxiv", "prime") else "A",
             "transport_status": "not_attempted", "protocol_status": "not_attempted",
             "predict_calls": 0, "actual_calls": 0, "responses": 0,
             "http_statuses": [], "retry_observed": False, "errors": [],
             "semantic": None} for task in TASKS]
    report = {"schema_version": 1, "scope": "synthetic_live_smoke_not_primary_reproduction",
              "accuracy_guarantee": False, "injected_client": client_factory is not None,
              "call_timeout_seconds": CALL_SECONDS, "overall_timeout_seconds": OVERALL_SECONDS,
              "runner_retries": 0, "sdk_hidden_retries": "not_controllable; observed retries rejected",
              "call_count_basis": "outbound HTTP attempts or observed SDK turns/usage; not server forwards",
              "trace_scope": "validated normalized responses and transport metadata; no free provider text",
              "tasks": rows, "errors": []}
    settings = {key: value for key, value in asdict(config).items() if value is not None}
    graphtext.save(output / "settings.json", settings)
    episodes.atomic_json(output / "summary.json", report)
    observations = Observations(output)
    client = None
    try:
        async with asyncio.timeout(OVERALL_SECONDS):
            jobs = fixtures()
            # Retain synthetic source states/references separately from requests.
            graphtext.save(output / "fixtures.json", [
                {"task": task, "instance": instance,
                 "request": asdict(request) if request else None, "arm": arm}
                for task, instance, request, arm in jobs])
            client = (client_factory or make_client)(config, observations)
            await asyncio.wait_for(client.initialize(), CALL_SECONDS)
            bounded = BoundedClient(client, config, observations)
            for index, (task, instance, request, arm) in enumerate(jobs):
                row = rows[index]
                observations.row = row
                # Reserve time for every remaining type instead of letting the
                # first slow construction consume the whole 600-second budget.
                remaining = OVERALL_SECONDS - (time.monotonic() - started)
                task_seconds = max(0.001, (remaining - 5) / (len(jobs) - index))
                try:
                    async with asyncio.timeout(task_seconds):
                        if request is not None:
                            public = asdict(request)
                            job = {"dataset": task, "split": "smoke", "target": task,
                                   "arm": arm, "request": public,
                                   "request_sha256": graphtext.value_hash(public)}
                            outcome = await graphtext.execute_one(
                                output, output / "decisions", job, settings, bounded)
                            row["protocol_status"] = outcome["status"]
                            if outcome["status"] != "success":
                                row["errors"].append({key: outcome[key] for key in
                                    ("error_type", "diagnostic_code", "http_status") if key in outcome})
                            if instance is not None and outcome["status"] == "success":
                                row["semantic"] = extended_tasks.score(
                                    instance, [outcome["selected_option_id"]])
                        else:
                            directory = output / "episodes" / task
                            await episodes.execute_episode(
                                directory, instance, config, arm, bounded, asyncio.Event())
                            score = episodes.verify_episode(directory, instance, config, arm)
                            row["protocol_status"] = "success" if score["feasible"] else "incomplete"
                            row["semantic"] = {key: score.get(key) for key in
                                               ("feasible", "objective", "absolute_gap", "optimal")}
                            if score["failure"]:
                                row["errors"].append({"diagnostic_code": score["failure"]})
                except Exception as exc:
                    row["protocol_status"] = "failure"
                    row["errors"].append(diagnostic(exc))
                    # Cancellation can leave a durable intent without a final
                    # step/outcome. Seal it as incomplete, never resume the call.
                    try:
                        if request is None:
                            directory = output / "episodes" / task
                            if (directory / "events.jsonl").exists():
                                episodes.seal_episode(directory, instance, config, arm,
                                                      "smoke_interrupted_no_retry")
                                score = episodes.verify_episode(directory, instance, config, arm)
                                row["semantic"] = {key: score.get(key) for key in
                                                   ("feasible", "objective", "absolute_gap", "optimal")}
                        else:
                            directory = output / "decisions" / "smoke" / f"{task}.{arm}"
                            if directory.exists() and not (directory / "outcome.json").exists():
                                graphtext.save(directory / "smoke-interruption.json", diagnostic(exc))
                    except Exception as seal_error:
                        row["errors"].append(diagnostic(seal_error))
                finally:
                    row["transport_status"] = (
                        "response_received" if row["responses"] else
                        "attempted_no_valid_response" if row["actual_calls"] else "no_call_observed")
                    if not row["actual_calls"] or row["retry_observed"]:
                        row["protocol_status"] = "failure"
                    episodes.atomic_json(output / "summary.json", report)
    except Exception as exc:
        report["errors"].append(diagnostic(exc))
    finally:
        observations.row = None
        if client is not None:
            try:
                await asyncio.wait_for(client.aclose(), 5)
            except Exception as exc:
                report["errors"].append(diagnostic(exc))
        report["coverage_complete"] = all(row["actual_calls"] >= 1 for row in rows)
        report["live_coverage_complete"] = report["coverage_complete"] and client_factory is None
        report["protocol_complete"] = all(row["protocol_status"] == "success" for row in rows)
        report["ok"] = report["coverage_complete"] and report["protocol_complete"] and not report["errors"]
        report["elapsed_seconds"] = time.monotonic() - started
        episodes.atomic_json(output / "summary.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", required=True, choices=("vllm", "github_copilot"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", help="Existing vLLM endpoint only")
    parser.add_argument("--output", type=Path, required=True, help="Fresh output/runtime/... directory")
    parser.add_argument("--allow-inference", action="store_true", help="Explicitly authorize live calls")
    args = parser.parse_args(argv)
    if not args.allow_inference:
        parser.error("--allow-inference is required; nothing was run")
    try:
        report = asyncio.run(run_smoke(**vars(args)))
    except Exception as exc:
        print(json.dumps({"ok": False, "error": diagnostic(exc)}))
        return 2
    print(json.dumps({"ok": report["ok"], "coverage_complete": report["coverage_complete"],
                      "protocol_complete": report["protocol_complete"]}))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())