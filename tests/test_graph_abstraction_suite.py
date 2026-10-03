"""Four-task adapter, at-most-once WAL, scoring and support regression tests."""

import asyncio
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from src.utils import graph_abstraction_suite as suite
from src.clients.base import (
    BaseDecisionClient, ClientCapabilities, ClientTimeoutError, DecisionResponse,
    ProviderHTTPError, TokenUsage, UnsupportedRequestError,
    InvalidResponseError,
)


def fixture(task):
    return deepcopy(next(i for i in suite.extended.build_instances() if i["task"] == task))


def public_fixture(task):
    if task == "tsp_public":
        state = {"nodes": list(range(5)), "coordinates": [[0, 0], [1, 1], [7, 2], [1, 5], [4, 9]],
                 "edge_weight_type": "EUC_2D"}
    else:
        state = {"nodes": list(range(5)), "edges": [[0, 1, 3], [0, 4, -1], [1, 2, 2], [2, 3, 4]]}
    return {"id": task, "dataset_id": task, "replicate": 0, "task": task, "state": state,
            "private": {"reference": {"value": 1, "status": "best_known",
                                       "source": "https://example.invalid"}}}


SPEC = {"provider": "vllm", "model": "fake", "timeout_seconds": 0.1}
PORTABLE_SPEC = {
    "provider": "vllm", "model": "Qwen3.5-0.8B", "timeout_seconds": 180,
    "base_url": "http://127.0.0.1:8000/v1", "think": False, "max_tokens": 64,
    "constrain_choices": True, "output_format": "answer_only", "temperature": 0,
}


def historical_rows():
    """Self-contained historical-format records; no ignored experiment artifacts."""
    originals = []
    fixtures = suite.extended.build_instances()
    for task in suite.TASKS:
        template = next(i for i in fixtures if i["task"] == task)
        for index in range(40):
            originals.append({**deepcopy(template), "id": f"{task}-{index}"})
    return {
        suite.REPO / suite.SYNTHETIC: originals,
        suite.REPO / suite.MAIN_SCORES: [
            {"task": i["task"], "instance_id": i["id"]} for i in originals],
        suite.REPO / suite.ORIGINAL_PUBLIC: [
            {k: v for k, v in i.items() if k != "private"} for i in originals],
        suite.REPO / suite.ORIGINAL_REFERENCES: [
            {"id": i["id"], "private": i["private"]} for i in originals],
    }


class FakeClient(BaseDecisionClient):
    def __init__(self, error=None, limit=None):
        self.calls = 0
        self.error, self.limit = error, limit

    @property
    def capabilities(self):
        return ClientCapabilities(max_options=self.limit)

    async def predict(self, request):
        self.calls += 1
        if self.error:
            raise self.error
        return DecisionResponse(request.request_id, request.options[0].id, "fake", usage=TokenUsage())


class AuditedNativeClient(FakeClient):
    def __init__(self, path, *, missing=False, wrong_hash=False, unsupported=False):
        super().__init__()
        self.path = path
        self.missing, self.wrong_hash, self.unsupported = missing, wrong_hash, unsupported

    async def predict(self, request):
        self.calls += 1
        encoded = json.dumps(asdict(request), sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False, allow_nan=False).encode()
        identity = {"model": suite.NATIVE_MODELS[0], "call_index": self.calls,
                    "request_sha256": "0" * 64 if self.wrong_hash else hashlib.sha256(encoded).hexdigest()}
        wire = {"status": "unsupported" if self.unsupported else "success",
                "forward_count": 0 if self.unsupported else 1,
                "result": {"selected_option_id": request.options[0].id}}
        if not self.missing:
            with self.path.open("a") as stream:
                for row in (
                    {**identity, "event": "intent"},
                    {**identity, "event": "response", "raw_wire": wire},
                    {**identity, "event": wire["status"], "forward_count": wire["forward_count"]},
                ):
                    suite.audit.append_row(stream, row)
        if self.unsupported:
            raise UnsupportedRequestError("Native lossless capacity")
        return DecisionResponse(request.request_id, request.options[0].id,
                                suite.NATIVE_MODELS[0], usage=TokenUsage())


class RawClient(FakeClient):
    def __init__(self, raw=None):
        super().__init__()
        self.raw = {"vendor": {"retained": [1, 2, 3]}} if raw is None else raw

    async def predict(self, request):
        self.calls += 1
        probabilities = {o.id: float(index == 0) for index, o in enumerate(request.options)}
        return DecisionResponse(
            request.request_id, request.options[0].id, "fake",
            probabilities=probabilities, probability_kind="option_logits", confidence=1.0,
            usage=TokenUsage(12, 1, None), raw_output=self.raw,
        )


class AdapterTests(unittest.TestCase):
    def test_all_tasks_all_arms_complete_and_replay_objectives(self):
        for instance in [*(fixture(t) for t in suite.TASKS),
                         public_fixture("tsp_public"), public_fixture("maxcut_public")]:
            for arm in suite.ARMS:
                with self.subTest(task=instance["task"], arm=arm):
                    prep, decisions = suite.prepare(instance), []
                    while (payload := suite.step(prep, decisions, arm)) is not None:
                        self.assertNotIn("private", json.dumps(asdict(payload["request"]))
                                         if payload["request"] else "")
                        decisions.append(payload["candidates"][0]["action"])
                    result = suite.score(instance, decisions)
                    self.assertTrue(result["feasible"])
                    self.assertEqual(len(decisions), suite.budget(instance))

    def test_matched_bc_history_identical_actions_order_features(self):
        for instance in [*(fixture(t) for t in suite.TASKS), public_fixture("tsp_public")]:
            prepared, history = suite.prepare(instance), []
            while (b := suite.step(prepared, history, "B")) is not None:
                c = suite.step(prepared, history, "C")
                self.assertEqual({k: v for k, v in b.items() if k != "request"},
                                 {k: v for k, v in c.items() if k != "request"})
                if b["request"]:
                    self.assertNotIn("rules", b["request"].state)
                    self.assertIn("rules", c["request"].state)
                    self.assertEqual(b["request"].state["candidates"], c["request"].state["candidates"])
                history.append(b["candidates"][0]["action"])

    def test_arm_a_keeps_original_request_and_binary_partition(self):
        for task in suite.TASKS:
            instance = fixture(task)
            p = suite.step(suite.prepare(instance), [], "A")
            self.assertEqual(p["request"], suite.extended.next_request(instance, []))
            if task in ("maxcut_construct", "community_bipartition_construct"):
                self.assertEqual([o.id for o in p["request"].options], ["A", "B"])

    def test_euclidean_unrounded_not_public_distance(self):
        instance = fixture("tsp_construct")
        instance["state"]["metric"] = "euclidean_unrounded"
        instance["state"]["points"][0:2] = [[0, 0], [1, 1]]
        prepared = suite.prepare(instance)
        self.assertEqual(prepared.distances[0][1], math.sqrt(2))
        self.assertNotEqual(prepared.distances[0][1], 1)
        for arm in suite.ARMS:
            payload = suite.step(prepared, [], arm)
            if payload["request"]:
                self.assertNotIn("Manhattan", payload["request"].question)

    def test_original_manhattan_metric_not_reinterpreted(self):
        instance = fixture("tsp_construct")
        self.assertEqual(instance["state"]["metric"], "Manhattan")
        self.assertEqual(suite.prepare(instance).distances, suite.extended._distances(instance["state"]))

    def test_unknown_metric_refused(self):
        instance = fixture("tsp_construct")
        instance["state"]["metric"] = "rounded_unspecified"
        with self.assertRaises(suite.public.shared.Refusal):
            suite.prepare(instance)

    def test_incomplete_scores_are_null_not_zero(self):
        for instance in [*(fixture(t) for t in suite.TASKS), public_fixture("tsp_public")]:
            result = suite.score(instance, [])
            self.assertFalse(result["feasible"])
            self.assertIsNone(result["objective"])

    def test_all_baseline_controls_and_five_seeds(self):
        records = suite.baselines([fixture(t) for t in suite.TASKS])
        self.assertEqual(len(records), 40)
        for task in suite.TASKS:
            own = [r for r in records if r["task"] == task]
            self.assertTrue(all(r["feasible"] for r in own))
            self.assertEqual([r["seed"] for r in own if r["method"] == "random_candidate"],
                             list(suite.RANDOM_SEEDS))

    def test_reference_does_not_change_proposals(self):
        for task in suite.TASKS:
            instance = fixture(task)
            other = deepcopy(instance)
            other["private"] = {"secret": "DO_NOT_USE"}
            a = suite.step(suite.prepare(instance), [], "C")
            b = suite.step(suite.prepare(other), [], "C")
            self.assertEqual(a, b)


class DurabilityTests(unittest.IsolatedAsyncioTestCase):
    async def test_generic_raw_response_and_probabilities_are_sealed_for_every_success(self):
        instance, client = fixture("maxcut_construct"), RawClient()
        config = suite.validate_config(SPEC)
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp) / "episode"
            result = await suite.execute_episode(directory, instance, config, "A", client, asyncio.Event())
            rows = suite.audit.read_rows(directory / "raw-responses.jsonl")
            self.assertEqual(len(rows), suite.budget(instance))
            response = rows[0]["response"]
            self.assertEqual(response["raw_output"], client.raw)
            self.assertEqual(response["probabilities"], {"A": 1.0, "B": 0.0})
            self.assertEqual(response["probability_kind"], "option_logits")
            self.assertEqual(response["usage"]["input_tokens"], 12)
            self.assertEqual(result, suite.verify_episode(directory, instance, config, "A"))
            events, _ = suite.read_events(directory / "events.jsonl")
            changed = deepcopy(rows)
            changed[0]["response"]["selected_option_id"] = "B"
            with self.assertRaises(suite.public.shared.Refusal):
                suite.replay(instance, events, config, "A", changed)
            with (directory / "raw-responses.jsonl").open("a") as stream:
                stream.write("{}\n")
            with self.assertRaises(suite.public.shared.Refusal):
                suite.verify_episode(directory, instance, config, "A")

    async def test_raw_response_fsynced_before_allowlisted_parser_failure(self):
        instance, client = fixture("maxcut_construct"), RawClient()
        config = suite.validate_config(SPEC)
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp) / "episode"
            raw_path = directory / "raw-responses.jsonl"
            fsynced = []
            real_fsync = os.fsync

            def fsync(fd):
                fsynced.append(os.readlink(f"/proc/self/fd/{fd}"))
                return real_fsync(fd)

            def fail_parse(response, request, settings):
                self.assertIn(str(raw_path), fsynced)
                self.assertEqual(suite.audit.read_rows(raw_path)[0]["response"]["raw_output"], client.raw)
                raise InvalidResponseError("Parsing failed", diagnostic_code="invalid_choice")

            with patch.object(suite.os, "fsync", side_effect=fsync), \
                    patch.object(suite.audit, "response_record", side_effect=fail_parse):
                result = await suite.execute_episode(
                    directory, instance, config, "A", client, asyncio.Event())
            self.assertFalse(result["feasible"])
            self.assertEqual(client.calls, 1)
            self.assertEqual(result["captured_response_records"], 1)
            self.assertEqual(result, suite.verify_episode(directory, instance, config, "A"))

    async def test_non_json_raw_response_is_explicit_fatal_capture_failure(self):
        instance, client = fixture("maxcut_construct"), RawClient({"invalid": {1, 2}})
        config = suite.validate_config(SPEC)
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp) / "episode"
            abort = asyncio.Event()
            with patch.object(suite.audit, "response_record") as parser:
                result = await suite.execute_episode(directory, instance, config, "A", client, abort)
            parser.assert_not_called()
            self.assertTrue(abort.is_set())
            self.assertFalse(result["feasible"])
            self.assertIsNone(result["objective"])
            self.assertEqual(result["raw_response_capture_failures"], 1)
            row = suite.audit.read_rows(directory / "raw-responses.jsonl")[0]
            self.assertEqual(row["capture_error"], "non_json_response")
            self.assertNotIn("response", row)

    async def test_credential_material_is_never_written_to_raw_sidecar(self):
        secret = "synthetic-test-secret-not-a-real-credential"
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {"VLLM_API_KEY": secret}):
            directory = Path(temp) / "episode"
            client = RawClient({"vendor": {"echo": secret}})
            result = await suite.execute_episode(directory, fixture("maxcut_construct"),
                suite.validate_config(SPEC), "A", client, asyncio.Event())
            raw = (directory / "raw-responses.jsonl").read_text()
            self.assertNotIn(secret, raw)
            self.assertIn("credential_material", raw)
            self.assertTrue(result["fatal"])
            self.assertIsNone(result["objective"])
            self.assertTrue(suite.contains_credential_material({"headers": {"Authorization": "Bearer test"}}))

    async def test_orphan_raw_response_is_preserved_and_never_executed_or_retried(self):
        instance, client = fixture("maxcut_construct"), RawClient()
        config = suite.validate_config(SPEC)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            suite.write_json(root / "schedule.json", [{"instance_id": instance["id"], "arm": "A"}])
            directory = root / "runs/test/A" / instance["id"]
            directory.mkdir(parents=True)
            request = suite.step(suite.prepare(instance), [], "A")["request"]
            response = await client.predict(request)
            with (directory / "events.jsonl").open("x") as stream:
                suite.audit.append_row(stream, {"event": "call_started", "step": 0,
                                                "request": suite.audit.json_value(asdict(request))})
            with (directory / "raw-responses.jsonl").open("x") as stream:
                suite.capture_response(stream, response, request, instance, "A", 0, 0)
            await suite.run_lane(root, {instance["id"]: instance}, "test", SPEC, 1,
                                 factory=lambda spec: client)
            result = suite.verify_episode(directory, instance, config, "A")
            self.assertEqual(client.calls, 1)
            self.assertEqual(result["orphan_raw_responses"], 1)
            self.assertEqual(result["decisions"], [])
            self.assertFalse(result["feasible"])

    async def test_native_raw_audit_copied_and_replayed_before_action(self):
        instance = fixture("lt_influence_construct")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            client = AuditedNativeClient(root / "native.jsonl")
            config = suite.SimpleNamespace(provider="plugin", model=suite.NATIVE_MODELS[0],
                                           timeout_seconds=1, adapter_kwargs={"audit_path": str(client.path)})
            directory = root / "episode"
            result = await suite.execute_episode(directory, instance, config, "A", client, asyncio.Event())
            self.assertTrue(result["feasible"])
            self.assertEqual(result["native_forward_count"], suite.budget(instance))
            self.assertEqual(result, suite.verify_episode(directory, instance, config, "A"))
            events, _ = suite.read_events(directory / "events.jsonl")
            attempt = next(row["attempt"] for row in events if row["event"] == "step")
            self.assertIn("raw_wire", attempt["native_audit"][1])
            attempt["native_audit"][1]["raw_wire"]["forward_count"] = 0
            with self.assertRaises(suite.public.shared.Refusal):
                suite.replay(instance, events, config, "A")

    async def test_missing_or_mismatched_native_audit_fails_closed_without_retry(self):
        instance = fixture("maxcut_construct")
        for kwargs in ({"missing": True}, {"wrong_hash": True}):
            with self.subTest(kwargs=kwargs), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                client = AuditedNativeClient(root / "native.jsonl", **kwargs)
                config = suite.SimpleNamespace(provider="plugin", model=suite.NATIVE_MODELS[0],
                    timeout_seconds=1, adapter_kwargs={"audit_path": str(client.path)})
                abort = asyncio.Event()
                result = await suite.execute_episode(root / "episode", instance, config, "A", client, abort)
                self.assertEqual(client.calls, 1)
                self.assertTrue(abort.is_set())
                self.assertFalse(result["feasible"])
                self.assertIsNone(result["objective"])
                self.assertIsNone(result["native_forward_count"])
                self.assertEqual(result["native_forward_unknown_attempts"], 1)

    async def test_native_unsupported_keeps_raw_zero_forward_proof(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            client = AuditedNativeClient(root / "native.jsonl", unsupported=True)
            config = suite.SimpleNamespace(provider="plugin", model=suite.NATIVE_MODELS[0],
                timeout_seconds=1, adapter_kwargs={"audit_path": str(client.path)})
            result = await suite.execute_episode(root / "episode", fixture("maxcut_construct"),
                                                  config, "A", client, asyncio.Event())
            self.assertEqual(result["native_forward_count"], 0)
            self.assertEqual(result["terminal_reason"], "unsupported")
            self.assertIsNone(result["objective"])

    async def test_hold_prevents_any_new_episode_or_factory_initialization(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            suite.write_json(root / "input-scope-hold.json", {"publication_allowed": False})
            client = FakeClient()
            with self.assertRaises(suite.public.shared.Refusal):
                await suite.execute_episode(root / "episode", fixture("maxcut_construct"),
                    suite.validate_config(SPEC), "A", client, asyncio.Event(), root=root)
            self.assertFalse((root / "episode").exists())
            with self.assertRaises(suite.public.shared.Refusal):
                await suite.run_lane(root, {}, "test", SPEC, 1,
                                     factory=lambda spec: self.fail("Held factory initialized"))
            self.assertEqual(client.calls, 0)

    async def test_success_sealed_and_tampering_rejected(self):
        instance, client = fixture("maxcut_construct"), FakeClient()
        config = suite.validate_config(SPEC)
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp) / "episode"
            result = await suite.execute_episode(directory, instance, config, "A", client, asyncio.Event())
            self.assertTrue(result["feasible"])
            self.assertEqual(client.calls, suite.budget(instance))
            self.assertEqual(result, suite.verify_episode(directory, instance, config, "A"))
            with (directory / "events.jsonl").open("a") as stream:
                stream.write("{}\n")
            with self.assertRaises(suite.public.shared.Refusal):
                suite.verify_episode(directory, instance, config, "A")

    async def test_timeout_is_one_failed_attempt_not_zero_quality(self):
        instance, client = fixture("maxcut_construct"), FakeClient(ClientTimeoutError("hidden"))
        config = suite.validate_config(SPEC)
        with tempfile.TemporaryDirectory() as temp:
            result = await suite.execute_episode(Path(temp) / "episode", instance, config,
                                                  "A", client, asyncio.Event())
            self.assertEqual(client.calls, 1)
            self.assertEqual(result["terminal_reason"], "timeout")
            self.assertIsNone(result["objective"])
            self.assertFalse(result["fatal"])

    async def test_native_option_cap_is_unsupported_not_zero_score(self):
        instance, client = fixture("lt_influence_construct"), FakeClient(limit=2)
        config = suite.validate_config(SPEC)
        with tempfile.TemporaryDirectory() as temp:
            result = await suite.execute_episode(Path(temp) / "episode", instance, config,
                                                  "A", client, asyncio.Event())
            self.assertEqual(client.calls, 0)
            self.assertEqual(result["terminal_reason"], "unsupported")
            self.assertIsNone(result["objective"])
            self.assertFalse(result["fatal"])

    async def test_singletons_do_not_call_client(self):
        instance = fixture("maxcut_construct")
        instance["state"]["edges"] = []
        instance["private"]["objective"] = 0
        client = FakeClient()
        with tempfile.TemporaryDirectory() as temp:
            result = await suite.execute_episode(Path(temp) / "episode", instance,
                suite.validate_config(SPEC), "B", client, asyncio.Event())
            self.assertTrue(result["feasible"])
            self.assertEqual(client.calls, 0)
            self.assertEqual(result["forced_candidate_steps"], suite.budget(instance))
            self.assertEqual(result["objective"], 0)  # A genuinely completed zero cut.

    async def test_lane_resume_never_retries_failures(self):
        instance, client = fixture("maxcut_construct"), FakeClient(ClientTimeoutError("hidden"))
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            suite.write_json(root / "schedule.json", [{"instance_id": instance["id"], "arm": "A"}])
            for _ in range(2):
                await suite.run_lane(root, {instance["id"]: instance}, "test", SPEC, 1,
                                     factory=lambda spec: client)
            self.assertEqual(client.calls, 1)

    async def test_durable_pending_intent_not_reissued(self):
        instance, client = fixture("maxcut_construct"), FakeClient()
        config = suite.validate_config(SPEC)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            suite.write_json(root / "schedule.json", [{"instance_id": instance["id"], "arm": "A"}])
            directory = root / "runs/test/A" / instance["id"]
            directory.mkdir(parents=True)
            request = suite.step(suite.prepare(instance), [], "A")["request"]
            with (directory / "events.jsonl").open("x") as stream:
                suite.audit.append_row(stream, {"event": "call_started", "step": 0,
                                                "request": suite.audit.json_value(asdict(request))})
            await suite.run_lane(root, {instance["id"]: instance}, "test", SPEC, 1,
                                 factory=lambda spec: client)
            result = suite.verify_episode(directory, instance, config, "A")
            self.assertEqual(client.calls, 0)
            self.assertTrue(result["pending_call"])
            self.assertEqual(result["terminal_reason"], "interrupted_no_retry")
            self.assertIsNone(result["objective"])

    async def test_fatal_error_retains_all_scheduled_denominators(self):
        instance, client = fixture("maxcut_construct"), FakeClient(ProviderHTTPError(401))
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            suite.write_json(root / "schedule.json", [
                {"instance_id": instance["id"], "arm": arm} for arm in suite.ARMS])
            await suite.run_lane(root, {instance["id"]: instance}, "test", SPEC, 1,
                                 factory=lambda spec: client)
            self.assertEqual(client.calls, 1)
            self.assertEqual(len(list((root / "runs/test").glob("*/*/score.json"))), 3)
            self.assertEqual(suite.read_json(root / "runs/test/status.json")["status"], "aborted")
            protocol = {"models": ["test"], "inputs_sha256": {}}
            suite.write_json(root / "protocol.json", protocol)
            receipt = suite.verify_terminal(root, protocol, {instance["id"]: instance},
                                            suite.read_json(root / "schedule.json"))
            self.assertEqual(receipt["verified_episodes"], 3)
            self.assertEqual(receipt["terminal_lanes"], 1)
            self.assertNotIn("complete", receipt["outcomes"])

    async def test_expired_episode_terminates_even_for_forced_steps(self):
        instance, client = fixture("maxcut_construct"), FakeClient()
        instance["state"]["edges"] = []
        with tempfile.TemporaryDirectory() as temp, patch.object(suite, "EPISODE_SECONDS", 0):
            result = await suite.execute_episode(Path(temp) / "episode", instance,
                suite.validate_config(SPEC), "B", client, asyncio.Event())
            self.assertEqual(result["terminal_reason"], "episode_timeout")
            self.assertFalse(result["feasible"])

    def test_torn_event_is_preserved_not_reparsed_or_retried(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "events.jsonl"
            path.write_bytes(b'{"event":"call_started","step":0}\n{"event":')
            events, torn = suite.read_events(path)
            self.assertTrue(torn)
            self.assertEqual(len(events), 1)
            self.assertTrue(path.read_bytes().endswith(b'{"event":'))

    def test_plugin_caps_and_hash_required(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "plugin.py"
            path.write_text("def create_client(**kwargs):\n    return object()\n")
            spec = {"provider": "plugin", "model": "native", "timeout_seconds": 1,
                    "adapter_file": str(path), "adapter_sha256": suite.digest(path),
                    "factory": "create_client", "adapter_kwargs": {}}
            suite.validate_config(spec)
            with self.assertRaises(suite.public.shared.Refusal):
                suite.make_client(spec)
            spec["adapter_sha256"] = "0" * 64
            with self.assertRaises(suite.public.shared.Refusal):
                suite.validate_config(spec)

    def test_native_lane_cannot_silently_become_text_generation(self):
        for name in ("qwen4_token_scores", "decider", "kev", "laya"):
            with self.assertRaises(suite.public.shared.Refusal):
                suite.validate_lane_config(name, SPEC)

    def test_qwen_panel_requires_original_grammar_budget(self):
        with self.assertRaises(suite.public.shared.Refusal):
            suite.validate_lane_config("qwen4_grammar", SPEC)
        config = suite.validate_lane_config("qwen4_grammar", {
            **SPEC, "model": "Qwen3.5-4B", "think": False, "max_tokens": 64, "constrain_choices": True,
            "output_format": "answer_only", "temperature": 0,
        })
        self.assertEqual(config.max_tokens, 64)

    def test_report_contains_every_unattempted_lane_and_null_performance(self):
        instance = fixture("maxcut_construct")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            suite.write_json(root / "protocol.json", {})
            suite.write_json(root / "schedule.json", [
                {"instance_id": instance["id"], "arm": arm} for arm in suite.ARMS])
            with patch.object(suite, "load", return_value=(
                    root, {"models": list(suite.PANEL)}, {instance["id"]: instance})):
                suite.report(root)
            report = suite.read_json(root / "report.json")
            self.assertEqual(len(report["episodes"]), 14 * 3)
            self.assertTrue(all(r["objective_feasible_mean"] is None for r in report["summaries"]))
            self.assertTrue(all(r["model_calls"] is None for r in report["episodes"]))
            self.assertTrue(all(r["graph_macro_gap_delta"] is None for r in report["paired"]))


class V2FreezeMonitorTests(unittest.TestCase):
    def test_qwen38_cannot_be_silently_replaced_with_classic_qwen3(self):
        with self.assertRaises(suite.public.shared.Refusal):
            suite.validate_lane_config("qwen38_27b_bf16", {
                **SPEC, "model": "Qwen3-8B", "think": False, "max_tokens": 64,
                "constrain_choices": True, "output_format": "answer_only", "temperature": 0,
            })

    def test_original_main_bank_ids_and_private_references_are_exact(self):
        with patch.object(suite.audit, "read_rows", side_effect=historical_rows().__getitem__):
            original, validation = suite.original_instances()
        self.assertEqual(len(original), 160)
        self.assertEqual(validation["main_ledger_id_matches"], 160)
        self.assertEqual(validation["original_portable_full_record_matches"], 160)
        self.assertEqual(validation["tsp_metrics"], {"Manhattan": 40})
        self.assertNotIn("confirmation", str(suite.SYNTHETIC))

    def test_wrong_bank_is_rejected_even_when_quotas_match(self):
        rows = historical_rows()
        rows[suite.REPO / suite.SYNTHETIC][0]["id"] = "not-in-main-ledger"
        with patch.object(suite.audit, "read_rows", side_effect=rows.__getitem__):
            with self.assertRaises(suite.public.shared.Refusal):
                suite.original_instances()

    def test_private_reference_corruption_is_rejected_independently(self):
        record = fixture("maxcut_construct")
        record["private"]["objective"] += 1
        with self.assertRaises(ValueError):
            suite.extended.validate_instances([record])

    def test_monitor_distinguishes_paused_process_and_terminal_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            lane = root / "runs/test"
            lane.mkdir(parents=True)
            suite.atomic_json(lane / "status.json", {"status": "running", "pid": 123})
            protocol = {"models": ["test"], "version": suite.VERSION}
            jobs = [{"instance_id": "case", "arm": "A"}]
            with patch.object(suite, "process_state", return_value="T"):
                result = suite.monitor_snapshot(root, protocol, jobs)
            self.assertEqual(result["status"], "attention_required")
            self.assertEqual(result["alerts"][0]["code"], "controller_inactive")
            self.assertFalse(result["all_terminal"])
            directory = lane / "A/case"
            directory.mkdir(parents=True)
            suite.atomic_json(directory / "status.json", {"status": "incomplete"})
            suite.atomic_json(lane / "status.json", {"status": "aborted"})
            result = suite.monitor_snapshot(root, protocol, jobs)
            self.assertTrue(result["all_terminal"])
            self.assertEqual(result["terminal_episodes"], 1)

    def test_monitor_does_not_promote_terminal_lane_with_missing_records(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            lane = root / "runs/test"
            lane.mkdir(parents=True)
            suite.atomic_json(lane / "status.json", {"status": "aborted"})
            result = suite.monitor_snapshot(
                root, {"models": ["test"], "version": suite.VERSION},
                [{"instance_id": "missing", "arm": "A"}])
            self.assertFalse(result["all_terminal"])
            self.assertEqual(result["alerts"][0]["code"], "terminal_lane_missing_episode_seals")

    def test_monitor_completes_and_persists_verified_receipt_without_inference(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            suite.write_json(root / "schedule.json", [])
            suite.write_json(root / "protocol.json", {})
            snapshot = {"at": suite.public.now(), "status": "all_terminal_unverified",
                        "scheduled_episodes": 8862, "terminal_episodes": 8862,
                        "all_terminal": True, "alerts": []}
            receipt = {"verified_episodes": 8862, "terminal_episodes": 8862}
            with patch.object(suite, "load", return_value=(root, {}, {})), \
                    patch.object(suite, "monitor_snapshot", return_value=snapshot), \
                    patch.object(suite, "verify_terminal", return_value=receipt) as verify:
                suite.monitor(root, 1)
            verify.assert_called_once()
            self.assertEqual(suite.read_json(root / "terminal-receipt.json"), receipt)
            self.assertEqual(suite.read_json(root / "monitor-status.json")["status"],
                             "all_terminal_verified")

    def test_terminal_receipt_rejects_a_held_complete_schedule(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            suite.write_json(root / "input-scope-hold.json", {"publication_allowed": False})
            with self.assertRaises(suite.public.shared.Refusal):
                suite.verify_terminal(root, {"models": []}, {}, [])


class PortableInputsTests(unittest.TestCase):
    def write_inputs(self, root, instances):
        records, configs = root / "records.json", root / "models.json"
        suite.write_json(records, instances)
        suite.write_json(configs, {"qwen08": PORTABLE_SPEC})
        return records, configs

    def test_freeze_and_frozen_report_without_historical_artifacts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            instances = [fixture(t) for t in suite.TASKS]
            instances += [public_fixture("maxcut_public"), public_fixture("tsp_public")]
            del instances[-1]["dataset_id"]
            records, configs = self.write_inputs(root, instances)
            output = root / "run"
            output.mkdir()
            (output / "tests.log").write_text("Ran 1 tests in 0.01s\n\nOK\n")
            with patch.object(suite, "original_instances", side_effect=AssertionError("historical input")), \
                    patch.object(suite.public.shared, "model_configs",
                                 side_effect=AssertionError("local deployment config")):
                suite.freeze(output, instances_path=records, config_path=configs, models=["qwen08"])
            protocol = suite.read_json(output / "protocol.json")
            self.assertEqual(protocol["input_scope"], "user_supplied_not_historical_replication")
            self.assertEqual(protocol["models"], ["qwen08"])
            self.assertEqual(protocol["cases"], 6)
            self.assertEqual(protocol["episodes_per_model"], 18)
            self.assertEqual(protocol["synthetic_cases"], 4)
            self.assertEqual(protocol["public_conditions"], 2)
            self.assertEqual(suite.read_json(output / "freeze-ready.json")["scheduled_episodes"], 18)
            self.assertEqual(protocol["source_provenance"], {
                str(records): suite.digest(records), str(configs): suite.digest(configs)})
            self.assertNotIn("src/utils/proposal_scoring_client.py", protocol["source_sha256"])
            self.assertEqual(suite.read_json(output / "cloud-models.json"), {"qwen08": PORTABLE_SPEC})
            self.assertEqual(len(suite.read_json(output / "baselines.json")), 60)
            records.unlink()
            configs.unlink()
            result = subprocess.run([
                sys.executable, str(output / "frozen-source/src/utils/graph_abstraction_suite.py"),
                "report", "--root", str(output),
            ], cwd=root, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            episodes = suite.read_json(output / "report.json")["episodes"]
            self.assertEqual(len(episodes), 18)
            self.assertTrue(all(row["dataset_id"] == row["instance_id"]
                                for row in episodes if row["task"] == "tsp_public"))

    def test_invalid_inputs_fail_before_freeze(self):
        record = fixture("maxcut_construct")
        invalid = [
            [], [record, record], [{**record, "id": "../escape"}],
            [{**record, "task": "unknown"}],
            [{**record, "private": {**record["private"], "objective": -100}}],
        ]
        for instances in invalid:
            with self.subTest(instances=instances), tempfile.TemporaryDirectory() as temp:
                records, configs = self.write_inputs(Path(temp), instances)
                with self.assertRaises((ValueError, suite.public.shared.Refusal)):
                    suite.supplied_inputs(records, configs, ["qwen08"])

    def test_explicit_configs_require_exact_model_set_and_lane_contract(self):
        for models, config in [
            (["qwen08", "qwen08"], PORTABLE_SPEC),
            (["unknown"], PORTABLE_SPEC),
            (["qwen2"], PORTABLE_SPEC),
            (["qwen08"], SPEC),
        ]:
            with self.subTest(models=models, config=config), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                records = root / "records.json"
                configs = root / "models.json"
                suite.write_json(records, [fixture("maxcut_construct")])
                suite.write_json(configs, {"qwen08": config})
                with self.assertRaises(suite.public.shared.Refusal):
                    suite.supplied_inputs(records, configs, models)

    def test_invalid_public_dataset_id_rejected_before_freeze_artifact_writes(self):
        for task in ("maxcut_public", "tsp_public"):
            for dataset_id in ([], {}, None, "", " \t\n", 1, False):
                with self.subTest(task=task, dataset_id=dataset_id), \
                        tempfile.TemporaryDirectory() as temp:
                    root = Path(temp)
                    record = {**public_fixture(task), "dataset_id": dataset_id}
                    records, configs = self.write_inputs(root, [record])
                    output = root / "run"
                    output.mkdir()
                    (output / "tests.log").write_text("Ran 1 tests in 0.01s\n\nOK\n")
                    with patch.object(suite, "baselines") as baselines, \
                            self.assertRaisesRegex(suite.public.shared.Refusal, "dataset_id"):
                        suite.freeze(output, instances_path=records, config_path=configs,
                                     models=["qwen08"])
                    baselines.assert_not_called()
                    self.assertEqual({p.name for p in output.iterdir()}, {"tests.log"})

    def test_cli_plugin_run_and_single_condition_reporting(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plugin = root / "client.py"
            plugin.write_text(
                "from src.clients.base import BaseDecisionClient, ClientCapabilities, DecisionResponse\n"
                "class Client(BaseDecisionClient):\n"
                "    @property\n"
                "    def capabilities(self):\n"
                "        return ClientCapabilities(max_options=254, max_context_tokens=100000)\n"
                "    async def predict(self, request):\n"
                "        return DecisionResponse(request.request_id, request.options[0].id, 'offline-test')\n"
                "def create_client():\n"
                "    return Client()\n"
            )
            name = "qwen4_token_scores"
            records, configs = root / "records.json", root / "models.json"
            explicit = public_fixture("maxcut_public")
            explicit["dataset_id"] = "explicit-graph"
            missing = {**public_fixture("maxcut_public"), "id": "missing-dataset-id"}
            del missing["dataset_id"]
            suite.write_json(records, [explicit, missing])
            suite.write_json(configs, {name: {
                "provider": "plugin", "model": "offline-test", "timeout_seconds": 10,
                "adapter_file": str(plugin), "adapter_sha256": suite.digest(plugin),
                "factory": "create_client", "adapter_kwargs": {},
            }})
            output = root / "run"
            output.mkdir()
            (output / "tests.log").write_text("Ran 1 tests in 0.01s\n\nOK\n")
            commands = [
                [str(suite.REPO / "src/utils/graph_abstraction_suite.py"), "freeze",
                 "--instances", str(records), "--model-config", str(configs), "--models", name],
                [str(output / "frozen-source/src/utils/graph_abstraction_suite.py"), "run",
                 "--model-config", str(configs), "--models", name, "--concurrency", "1"],
            ]
            for command in commands:
                result = subprocess.run([sys.executable, *command, "--root", str(output)],
                                        cwd=root, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            report = suite.read_json(output / "report.json")
            self.assertEqual(len(report["episodes"]), 6)
            self.assertTrue(all(row["feasible"] for row in report["episodes"]))
            self.assertEqual({row["instance_id"]: row["dataset_id"] for row in report["episodes"]},
                             {explicit["id"]: "explicit-graph", missing["id"]: missing["id"]})
            self.assertTrue(all(row["graphs_with_complete_conditions"] == 2
                                for row in report["summaries"]))
            self.assertTrue(all(row["graphs_with_all_conditions"] == 2 for row in report["paired"]))
            plugin.unlink()
            result = subprocess.run([
                sys.executable, str(output / "frozen-source/src/utils/graph_abstraction_suite.py"),
                "report", "--root", str(output),
            ], cwd=root, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
