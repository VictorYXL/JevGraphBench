"""Focused offline RQ1 tests; no model inference or public-data downloads."""

import asyncio
from copy import deepcopy
from dataclasses import asdict
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import httpx
import networkx as nx

from src.clients.base import (
    BaseDecisionClient, ClientCapabilities, ClientTimeoutError, DecisionResponse,
    InvalidResponseError, UnsupportedRequestError,
)
from src.clients.vllm_client import VLLMClient
from src.utils import public_structural as suite


def instance(task="degree_exact"):
    state = {"nodes": list(range(8)), "edges": [[0, 1], [1, 2]], "directed": False}
    if task == "degree_exact":
        state["vertex"] = 0
    return {"id": "fixture", "task": task, "kind": "exact", "query_budget": 1,
            "state": state, "private": {"answer": suite.independent_answer(task, state)}}


class FakeClient(BaseDecisionClient):
    def __init__(self, error=None):
        self.calls = 0
        self.error = error

    @property
    def capabilities(self):
        return ClientCapabilities()

    async def predict(self, request):
        self.calls += 1
        if self.error:
            raise self.error
        return DecisionResponse(request.request_id, "1", "fixture", raw_output={"answer": "1"})


class SamplingTests(unittest.TestCase):
    def test_outer_authorized_existing_claims_are_attributed_without_resubmission(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            primary = root / "primary"
            runner = primary / "frozen-source/src/utils/public_structural.py"
            runner.parent.mkdir(parents=True)
            runner.write_text("fixture")
            suite.write(primary / "manifest.json", {"fixture": True})
            (primary / "runs/public-pilot/decider/claimed").mkdir(parents=True)
            authorization = {"path": str(root / "authorization.json"), "sha256": "auth"}
            plan = {
                "hashes": {authorization["path"]: "auth", str(runner): suite.file_sha(runner),
                           str(primary / "manifest.json"): suite.file_sha(primary / "manifest.json")},
                "jobs": [{"id": model + "-" + split,
                          "command": ["python", str(runner), "run", "--root", str(primary),
                                      "--model", model, "--split", split]}
                         for model in ("decider", "kev") for split in ("public-pilot", "test")]}
            suite.write(root / "plan.json", plan)
            suite.write(root / "proof.json", {
                "plan_sha256": suite.file_sha(root / "plan.json"), "authorization_sha256": "auth"})
            selected, evidence = suite.observed_outer_routes(
                {"decider": "deployment", "kev": "deployment"}, primary, authorization,
                root / "plan.json", root / "proof.json")
            self.assertEqual(selected, {"decider": "primary", "kev": "deployment"})
            self.assertEqual(evidence["adopted_existing_routes"]["decider"]
                             ["primary_claims_at_binding"]["public-pilot"], 1)
            (primary / "runs/public-pilot/kev/later-claim").mkdir(parents=True)
            late_selected, _ = suite.observed_outer_routes(
                {"decider": "deployment", "kev": "deployment"}, primary, authorization,
                root / "plan.json", root / "proof.json")
            self.assertEqual(late_selected, {"decider": "primary", "kev": "primary"})
            with self.assertRaisesRegex(ValueError, "authorization binding"):
                suite.observed_outer_routes(
                    selected, primary, {**authorization, "sha256": "changed"},
                    root / "plan.json", root / "proof.json")

    def test_scoring_recovery_refuses_any_claim_or_active_preflight(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            primary, deployment = root / "primary", root / "deployment"
            evidence = suite.assert_unclaimed_scoring(primary, deployment)
            self.assertEqual(len(evidence), 4)
            with patch.object(suite, "lane_active", return_value=True):
                with self.assertRaisesRegex(ValueError, "remains active"):
                    suite.assert_unclaimed_scoring(primary, deployment)
            claimed = primary / "runs/public-pilot/qwen4_token_scores/claim"
            claimed.mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, "already claimed"):
                suite.assert_unclaimed_scoring(primary, deployment)

    def test_scoring_recovery_preserves_zero_query_failed_launch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            lane = root / "primary/runs/public-pilot/qwen4_token_scores"
            lane.mkdir(parents=True)
            suite.write(lane / "launch-failed.json", {"initialization_error": "missing transformers"})
            before = suite.file_sha(lane / "launch-failed.json")
            evidence = suite.assert_unclaimed_scoring(root / "primary", root / "deployment")
            self.assertEqual(evidence["primary/public-pilot"]["existing_files_sha256"]
                             ["launch-failed.json"], before)
            self.assertEqual(suite.file_sha(lane / "launch-failed.json"), before)

    def test_explicit_parent_root_overrides_never_drop_or_duplicate_models(self):
        primary = list(suite.PRIMARY_MODELS)
        deployment = [m for m in suite.prior.PANEL if m not in primary]
        approval = {"existing_primary_lanes_must_not_be_repeated": primary,
                    "untouched_deployment_lanes": deployment}
        self.assertEqual(suite.approved_root_overrides(approval)["decider"], "deployment")
        primary.append("decider")
        deployment.remove("decider")
        self.assertEqual(suite.approved_root_overrides(approval)["decider"], "primary")
        deployment.append("decider")
        with self.assertRaisesRegex(ValueError, "scope mismatch"):
            suite.approved_root_overrides(approval)
        deployment.remove("decider")
        primary.remove("jev_action")
        with self.assertRaisesRegex(ValueError, "scope mismatch"):
            suite.approved_root_overrides(approval)

    def test_stage_config_binding_allows_transport_not_scientific_changes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            reference = {"qwen4_grammar": {"provider": "vllm", "model": "Qwen3.5-4B",
                         "base_url": "http://old/v1", "max_tokens": 64}}
            incoming = deepcopy(reference)
            incoming["qwen4_grammar"]["base_url"] = "http://new/v1"
            suite.write(root / "pilot.json", incoming)
            suite.write(root / "test.json", incoming)
            paths = {"public-pilot": root / "pilot.json", "test": root / "test.json"}
            with patch.object(suite, "validate_configs"):
                stages, bindings = suite.bind_stage_configs(reference, paths, root / "condition")
                self.assertEqual(stages["test"], incoming)
                self.assertEqual(bindings["test"]["sha256"], suite.file_sha(paths["test"]))
                incoming["qwen4_grammar"]["max_tokens"] = 65
                suite.write(root / "changed.json", incoming)
                with self.assertRaisesRegex(ValueError, "scientific settings"):
                    suite.bind_stage_configs(reference, {**paths, "test": root / "changed.json"},
                                             root / "condition")

    def test_stage_binding_drift_is_explicit(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            suite.write(root / "stage.json", {"qwen": {"model": "fixture"}})
            suite.write(root / "deployment-config-bindings.json", {
                "test": {"path": str(root / "stage.json"), "sha256": suite.file_sha(root / "stage.json")}})
            suite.write(root / "stage-models.json", {"test": suite.read(root / "stage.json")})
            suite.verify_stage_bindings(root)
            with (root / "stage.json").open("a") as stream:
                stream.write("\n")
            with self.assertRaisesRegex(ValueError, "Deployment config hash drift"):
                suite.verify_stage_bindings(root)

    def test_lane_check_requires_terminal_exact_scheduled_coverage(self):
        row = {"split": "public-pilot", "model": "jev_action", "scheduled": 52,
               "terminal": True, "statuses": {"success": 50, "unsupported": 2}}
        report = {"rows": [row], "manifest_sha256": "manifest", "authorization_sha256": "authorization"}
        with patch.object(suite, "report", return_value=report):
            receipt = suite.check_lane(Path("."), "jev_action", "public-pilot")
            self.assertTrue(receipt["integrity_and_coverage_passed"])
            row["terminal"] = False
            with self.assertRaisesRegex(ValueError, "not terminal"):
                suite.check_lane(Path("."), "jev_action", "public-pilot")
            row["scheduled"] = 51
            with self.assertRaisesRegex(ValueError, "scheduled count"):
                suite.check_lane(Path("."), "jev_action", "public-pilot")

    def test_exact_quota_matrix(self):
        from collections import Counter
        rows = list(suite.quotas("test"))
        self.assertEqual(len(rows), 800)
        self.assertEqual(Counter(r[0] for r in rows), dict.fromkeys(suite.SOURCES, 200))
        self.assertEqual(Counter(r[2] for r in rows),
                         {"degree_exact": 400, **dict.fromkeys(suite.TASKS[1:], 80)})
        for n in range(8, 13):
            for answer in range(n):
                self.assertEqual(sum(r[1:4] == (n, "degree_exact", answer) for r in rows), 8)
        for task in suite.TASKS[1:]:
            for answer in ("yes", "no"):
                self.assertEqual(sum(r[2:4] == (task, answer) for r in rows), 40)
        self.assertEqual(len(list(suite.quotas("public-pilot"))), 52)

    def test_reproducible_induced_pilot_with_original_ids(self):
        original = nx.barabasi_albert_graph(80, 3, seed=41)
        graph = nx.relabel_nodes(original, {v: 1000 + v * 11 for v in original})
        datasets = {name: SimpleNamespace(graph=graph, source=SimpleNamespace(name=name))
                    for name in suite.SOURCES}
        first, meta = suite.build_split(datasets, "public-pilot")
        second, other = suite.build_split(datasets, "public-pilot")
        self.assertEqual((first, meta), (second, other))
        self.assertEqual(len(first), 52)
        by_id = {i["id"]: i for i in first}
        for p in meta:
            self.assertTrue(all(v >= 1000 for v in p["original_node_ids"]))
            state = by_id[p["instance_id"]]["state"]
            for key, value in suite.induced_state(graph, p["original_node_ids"]).items():
                self.assertEqual(state[key], value)
        used = {p["source_node_set_sha256"] for p in meta}
        literal = {suite.sha([i["task"], i["state"]]) for i in first}
        next_instances, next_meta = suite.build_split(
            datasets, "public-pilot", used, excluded_queries=literal)
        self.assertTrue({p["source_node_set_sha256"] for p in meta}.isdisjoint(
            p["source_node_set_sha256"] for p in next_meta))
        self.assertTrue(literal.isdisjoint(suite.sha([i["task"], i["state"]]) for i in next_instances))
        self.assertTrue(any(nx.number_connected_components(nx.Graph(
            [tuple(e) for e in i["state"]["edges"]])) > 1 for i in first if i["state"]["edges"]))

    def test_independent_oracles_cover_all_tasks_and_disconnected_articulation(self):
        state = {"nodes": list(range(8)), "edges": [[0, 1], [1, 2], [3, 4], [4, 5], [3, 5]],
                 "directed": False}
        variants = {
            "degree_exact": ({**state, "vertex": 1}, 2),
            "cycle_detection": (state, "yes"),
            "pair_connectivity": ({**state, "pair": [0, 5]}, "no"),
            "adjacency": ({**state, "pair": [0, 2]}, "no"),
            "distance_threshold": ({**state, "pair": [0, 2], "threshold": 2}, "yes"),
            "articulation_point": ({**state, "vertex": 1}, "yes"),
        }
        for task, (query, answer) in variants.items():
            self.assertEqual(suite.independent_answer(task, query), answer)

    def test_impossible_degree_explicitly_fails(self):
        data = SimpleNamespace(graph=nx.path_graph(40), source=SimpleNamespace(name="path"))
        with self.assertRaisesRegex(ValueError, "infeasible"):
            suite.sample_query(data, 8, "degree_exact", 7, __import__("random").Random(1), 1)

    def test_isomorphism_audit_separates_query_roles_and_discloses_overlap(self):
        first, second = instance(), instance()
        second["id"] = "second"
        second["state"]["vertex"] = 1
        metadata = [{"instance_id": i["id"], "split": split, "source_node_set_sha256": str(n)}
                    for n, (i, split) in enumerate(((first, "test"), (second, "public-pilot")))]
        audit = suite.overlap_audit([first, second], metadata)
        self.assertEqual(audit["graph_isomorphism"]["duplicate_pairs"], 1)
        self.assertEqual(audit["graph_isomorphism"]["cross_split_pairs"], 1)
        self.assertEqual(audit["query_isomorphism"]["duplicate_pairs"], 0)


class ExecutionTests(unittest.IsolatedAsyncioTestCase):
    async def test_independent_primary_replay_uses_original_raw_and_oracle(self):
        class CapturedSDK(FakeClient):
            async def predict(self, request):
                suite.emit({"event": "sdk_event", "type": "assistant.message",
                            "data": {"content": "1"}})
                return DecisionResponse(request.request_id, "1", "fixture",
                                        raw_output={"content": "1"})
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp) / "query"
            spec = {"provider": "github_copilot", "model": "fixture", "timeout_seconds": 1}
            await suite.execute(directory, instance(), CapturedSDK(), spec)
            before = {p.name: suite.file_sha(p) for p in directory.iterdir()}
            row = suite.replay_combined_query(directory, instance(), spec, False)
            self.assertTrue(row["correct"])
            self.assertEqual(row["status"], "success")
            self.assertEqual(before, {p.name: suite.file_sha(p) for p in directory.iterdir()})
            corrupted = deepcopy(instance())
            corrupted["private"]["answer"] = 7
            with self.assertRaisesRegex(ValueError, "oracle mismatch"):
                suite.replay_combined_query(directory, corrupted, spec, False)
            with self.assertRaisesRegex(ValueError, "authorization"):
                suite.replay_combined_query(directory, instance(), spec, False, "missing-binding")

    async def test_independent_replay_retains_failures_and_active_claims(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            spec = {"provider": "github_copilot", "model": "fixture", "timeout_seconds": 1}
            await suite.execute(root / "failed", instance(), FakeClient(ClientTimeoutError("timeout")), spec)
            row = suite.replay_combined_query(root / "failed", instance(), spec, False)
            self.assertEqual(row["status"], "failed")
            self.assertFalse(row["correct"])
            (root / "active").mkdir()
            self.assertEqual(suite.replay_combined_query(root / "active", instance(), spec, True)["status"],
                             "in_progress")
            self.assertEqual(suite.replay_combined_query(root / "active", instance(), spec, False)["status"],
                             "interrupted_unknown")
            (root / "active/receipt.json").write_text('{"')
            self.assertEqual(suite.replay_combined_query(root / "active", instance(), spec, True)["status"],
                             "in_progress")
            with self.assertRaises(json.JSONDecodeError):
                suite.replay_combined_query(root / "active", instance(), spec, False)

    async def test_independent_wire_replay_rejects_changed_qwen_budget(self):
        import base64
        request = suite.tasks.next_request(instance(), [])
        body = {
            "model": "fixture", "messages": [
                {"role": "system", "content": suite.ANSWER_ONLY_SYSTEM_MESSAGE},
                {"role": "user", "content": suite.prompt_for(request)}],
            "temperature": 0, "top_p": 1, "max_tokens": 65, "stream": False, "n": 1,
            "chat_template_kwargs": {"enable_thinking": False},
            "structured_outputs": {"choice": [o.id for o in request.options]}}
        encoded = suite.canonical(body)
        rows = [{"event": "wire_request", "body": encoded,
                 "body_base64": base64.b64encode(encoded.encode()).decode()}]
        spec = {"provider": "vllm", "model": "fixture", "temperature": 0, "top_p": 1,
                "max_tokens": 64, "think": False, "output_format": "answer_only"}
        with self.assertRaisesRegex(ValueError, "decoding payload"):
            suite.verify_wire(rows, request, spec, "failed")

    def runner_fixture(self, root):
        authorization = {
            "authorization": "user_requested_execution", "new_local_thinking_lanes": False,
            "models": list(suite.prior.PANEL),
            "rq1": {"tasks": list(suite.TASKS), "node_counts": list(range(8, 13)),
                    "target_queries": 800, "no_topology_repair": True}}
        suite.write(root / "permission.json", authorization)
        binding, snapshot = suite.authorization_binding(root / "permission.json")
        suite.write(root / "authorization-binding.json", binding)
        suite.write(root / "authorization-snapshot.json", snapshot)
        records = []
        for task in suite.TASKS:
            row = instance()
            row.update(id=task, task=task)
            state = row["state"]
            if task not in ("degree_exact", "articulation_point"):
                state.pop("vertex")
            if task in ("pair_connectivity", "adjacency", "distance_threshold"):
                state["pair"] = [0, 1]
            if task == "distance_threshold":
                state["threshold"] = 2
            row["private"]["answer"] = suite.independent_answer(task, state)
            records.append(row)
        spec = {"provider": "vllm", "model": "fixture", "timeout_seconds": 1}
        suite.write(root / "models.json", dict.fromkeys(suite.prior.PANEL, spec))
        suite.write(root / "stage-models.json", {
            split: dict.fromkeys(suite.prior.PANEL, spec) for split in ("test", "public-pilot")})
        suite.write(root / "deployment-config-bindings.json", {})
        suite.write(root / "freeze-config.json", {"concurrency": dict.fromkeys(suite.prior.PANEL, 4)})
        suite.write(root / "manifest.json", {"fixture": True})
        suite.write(root / "instances.json", records)
        suite.write(root / "public-pilot.json", records)
        return records

    async def test_run_resume_retains_claims_and_report_scheduled_failures(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            records = self.runner_fixture(root)
            clients = []
            def factory(spec):
                client = FakeClient(UnsupportedRequestError("capacity"))
                clients.append(client)
                return client
            lane = root / "runs/test/qwen4_grammar"
            lane.mkdir(parents=True)
            (lane / records[0]["id"]).mkdir()
            with patch.object(suite, "load_frozen"), patch.object(suite.prior, "validate_lane_config"):
                await suite.run(root, "qwen4_grammar", concurrency=2, factory=factory, capture=False)
                await suite.run(root, "qwen4_grammar", concurrency=2, factory=factory, capture=False)
                result = suite.report(root)
            self.assertEqual(sum(c.calls for c in clients), 5)
            row = next(r for r in result["rows"] if r["split"] == "test" and r["model"] == "qwen4_grammar")
            self.assertEqual(row["statuses"], {"interrupted_unknown": 1, "unsupported": 5})
            self.assertEqual(row["scheduled"], 6)
            self.assertFalse(row["terminal"])
            self.assertEqual(len(result["rows"]), 28)

    async def test_report_detects_modified_raw_and_unsealed_result(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            records = self.runner_fixture(root)
            directory = root / "runs/test/qwen4_grammar" / records[0]["id"]
            directory.parent.mkdir(parents=True)
            await suite.execute(directory, records[0], FakeClient(),
                                {"model": "fixture", "timeout_seconds": 1},
                                authorization_sha256=suite.verify_authorization(root))
            with patch.object(suite, "load_frozen"):
                suite.report(root)
                with (directory / "raw.jsonl").open("a") as stream:
                    stream.write("{}\n")
                with self.assertRaisesRegex(ValueError, "receipt drift"):
                    suite.report(root)
                (directory / "receipt.json").unlink()
                report = suite.report(root)
            row = next(r for r in report["rows"] if r["split"] == "test" and r["model"] == "qwen4_grammar")
            self.assertEqual(row["statuses"]["interrupted_unsealed"], 1)

    async def test_frozen_concurrency_cap_is_enforced(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.runner_fixture(root)
            with patch.object(suite, "load_frozen"):
                with self.assertRaisesRegex(ValueError, "Concurrency"):
                    await suite.run(root, "qwen4_grammar", concurrency=5, factory=lambda _: self.fail("network"))

    async def test_authorization_drift_blocks_before_client_creation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.runner_fixture(root)
            with (root / "permission.json").open("a") as stream:
                stream.write("\n")
            with patch.object(suite, "load_frozen"):
                with self.assertRaisesRegex(ValueError, "authorization hash drift"):
                    await suite.run(root, "qwen4_grammar", factory=lambda _: self.fail("network"))

    async def test_success_durable_capture_and_exact_score(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "query"
            client = FakeClient()
            result = await suite.execute(path, instance(), client, {"model": "fixture", "timeout_seconds": 1})
            self.assertEqual(client.calls, 1)
            self.assertTrue(result["score"]["correct"])
            self.assertEqual(suite.read(path / "intent.json")["request"]["state"]["vertex"], 0)
            raw = [json.loads(line) for line in (path / "raw.jsonl").read_text().splitlines()]
            self.assertEqual(raw[0]["response"]["raw_output"], {"answer": "1"})
            with self.assertRaises(FileExistsError):
                await suite.execute(path, instance(), client, {"model": "fixture", "timeout_seconds": 1})
            self.assertEqual(client.calls, 1)

    async def test_failure_unsupported_timeout_and_initialization_retained(self):
        for error in (UnsupportedRequestError("capacity"), ClientTimeoutError("timeout"),
                      InvalidResponseError("bad", diagnostic_code="invalid_choice")):
            with tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / "query"
                result = await suite.execute(path, instance(), FakeClient(error),
                                             {"model": "fixture", "timeout_seconds": 1})
                self.assertFalse(result["score"]["correct"])
                self.assertIn(result["status"], ("failed", "unsupported"))
                self.assertTrue((path / "result.json").is_file())
        with tempfile.TemporaryDirectory() as temp:
            client = FakeClient()
            await suite.execute(Path(temp) / "query", instance(), client,
                                {"model": "fixture", "timeout_seconds": 1},
                                initialization_error=UnsupportedRequestError("unavailable"))
            self.assertEqual(client.calls, 0)

    async def test_preparse_http_raw_retained_for_invalid_response(self):
        async def handler(request):
            return httpx.Response(200, json={"model": "fixture", "choices": [{
                "finish_reason": "length", "message": {"role": "assistant", "content": "bad"}}]})
        client = VLLMClient(model="fixture", base_url="http://offline/v1", think=False,
                            max_tokens=64, output_format="answer_only",
                            transport=httpx.MockTransport(handler))
        suite.install_capture(client, {"provider": "vllm"})
        try:
            with tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / "query"
                result = await suite.execute(path, instance(), client,
                                             {"model": "fixture", "timeout_seconds": 1})
                raw = [json.loads(line) for line in (path / "raw.jsonl").read_text().splitlines()]
                self.assertEqual([r["event"] for r in raw], ["wire_request", "wire_response"])
                self.assertIn('"max_tokens":64', raw[0]["body"])
                self.assertIn('"finish_reason":"length"', raw[1]["body"])
                self.assertEqual(result["diagnostic_code"], "finish_length")
        finally:
            await client.aclose()

    async def test_unexpected_error_preserves_claim_without_success_fallback(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "query"
            with self.assertRaisesRegex(ValueError, "fatal"):
                await suite.execute(path, instance(), FakeClient(ValueError("fatal")),
                                    {"model": "fixture", "timeout_seconds": 1})
            self.assertTrue((path / "intent.json").exists())
            self.assertFalse((path / "result.json").exists())

    async def test_sdk_capture_before_existing_callback(self):
        from dataclasses import dataclass
        @dataclass
        class Data:
            content: str
        seen = []
        class Runtime:
            async def create_session(self, **kwargs):
                kwargs["on_event"](SimpleNamespace(type="assistant.message", data=Data("bad choice")))
                return "session"
        stream = io.StringIO()
        with patch.object(suite.os, "fsync"), patch.object(stream, "fileno", return_value=1):
            token = suite.CAPTURE.set(stream)
            try:
                await suite.RuntimeCapture(Runtime()).create_session(on_event=lambda _: seen.append(stream.getvalue()))
            finally:
                suite.CAPTURE.reset(token)
        self.assertIn("bad choice", seen[0])


if __name__ == "__main__":
    unittest.main()
