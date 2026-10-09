"""Offline live-smoke coverage, permissions, transport accounting and hygiene."""

import asyncio
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from dataclasses import asdict
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import httpx

from src.clients.base import (
    BaseDecisionClient, ClientCapabilities, DecisionResponse, InvalidResponseError,
    ProviderHTTPError,
)
from src.utils import live_smoke as smoke


class MockClient(BaseDecisionClient):
    def __init__(self, observations, *, error=None, evidence=True, limit=None):
        self.observations = observations
        self.error, self.evidence, self.limit = error, evidence, limit
        self.requests = []
        self.closed = False
        self.initializations = 0

    @property
    def capabilities(self):
        return ClientCapabilities(max_options=self.limit)

    async def initialize(self):
        self.initializations += 1

    async def predict(self, request):
        self.requests.append(request)
        if self.evidence:
            self.observations.event("http_request")
        if self.error:
            raise self.error
        if self.evidence:
            self.observations.event("http_response", 200)
        return DecisionResponse(request.request_id, request.options[0].id, "mock-model",
                                raw_output={"content": "UNTRUSTED_PROVIDER_TEXT"})

    async def aclose(self):
        self.closed = True


class LiveSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.jobs = smoke.fixtures()

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "smoke"
        self.created = []

    def run_mock(self, **kwargs):
        def factory(config, observations):
            client = MockClient(observations, **kwargs)
            self.created.append(client)
            return client

        with (patch.object(smoke, "fixtures", return_value=deepcopy(self.jobs)),
              patch("socket.create_connection", side_effect=AssertionError("Network forbidden")),
              patch("httpx.AsyncClient.send", side_effect=AssertionError("Network forbidden")),
              redirect_stdout(io.StringIO())):
            return asyncio.run(smoke.run_smoke(
                provider="vllm", model="mock-model", output=self.root,
                allow_inference=True, client_factory=factory))

    def artifacts(self):
        return "\n".join(path.read_text() for path in self.root.rglob("*") if path.is_file())

    def test_synthetic_fixtures_cover_exactly_twelve_types(self):
        self.assertEqual(tuple(job[0] for job in self.jobs), smoke.TASKS)
        self.assertEqual(len(set(smoke.TASKS)), 12)
        for task, instance, request, arm in self.jobs[:6]:
            self.assertEqual(instance["state"]["nodes"], [0, 1, 2, 3])
            self.assertEqual(instance["state"]["edges"], [[0, 1], [1, 2], [2, 3]])
            self.assertEqual(request, smoke.extended_tasks.next_request(instance, []))
            self.assertNotIn("private", asdict(request))
            self.assertEqual(arm, "A")
        arxiv, prime = self.jobs[6][2], self.jobs[7][2]
        self.assertEqual(len(arxiv.options), 40)
        self.assertEqual(len(prime.options), 40)
        self.assertEqual(len(prime.state["candidate_nodes"]), 40)
        self.assertEqual(len(prime.state["context_nodes"]), 3)
        self.assertIn("citation_edges", arxiv.state)
        self.assertIn("relations", prime.state)
        self.assertLess(len(json.dumps(asdict(prime))), smoke.graphtext.REQUEST_CHAR_CAP)
        bank = smoke.extended_tasks.build_instances()
        for task, instance, request, arm in self.jobs[8:]:
            self.assertEqual(instance, next(i for i in bank if i["task"] == task))
            self.assertLessEqual(len(instance["state"]["nodes"]), 10)
            self.assertIsNone(request)
            self.assertEqual(arm, "A")

    def test_full_mock_coverage_uses_one_shared_client_and_real_executors(self):
        report = self.run_mock()
        self.assertTrue(report["ok"])
        self.assertTrue(report["coverage_complete"])
        self.assertFalse(report["live_coverage_complete"])
        self.assertTrue(report["injected_client"])
        self.assertEqual(report["runner_retries"], 0)
        self.assertEqual(report["call_timeout_seconds"], 45)
        self.assertEqual(report["overall_timeout_seconds"], 600)
        self.assertEqual(len(self.created), 1)
        client = self.created[0]
        self.assertTrue(client.closed)
        self.assertEqual(client.initializations, 1)
        expected_calls = [1] * 8 + [6, 9, 2, 9]
        self.assertEqual([row["actual_calls"] for row in report["tasks"]], expected_calls)
        self.assertEqual([row["predict_calls"] for row in report["tasks"]], expected_calls)
        self.assertEqual(len(client.requests), sum(expected_calls))
        self.assertEqual(len({request.request_id for request in client.requests}), len(client.requests))
        degree = next(row for row in report["tasks"] if row["task"] == "degree_exact")
        self.assertFalse(degree["semantic"]["correct"])
        self.assertEqual(degree["protocol_status"], "success")
        self.assertIsNone(report["tasks"][6]["semantic"])
        config = smoke.model_config("vllm", "mock-model")
        for task, instance, _, arm in self.jobs[8:]:
            score = smoke.episodes.verify_episode(self.root / "episodes" / task, instance, config, arm)
            self.assertTrue(score["feasible"])
        self.assertEqual(len(list(self.root.rglob("intent.json"))), 8)
        self.assertEqual(len(list(self.root.rglob("outcome.json"))), 8)
        self.assertEqual(json.loads((self.root / "summary.json").read_text()), report)
        self.assertNotIn("UNTRUSTED_PROVIDER_TEXT", self.artifacts())

    def test_opt_in_refused_before_factory_fixtures_or_output(self):
        with (patch.object(smoke, "make_client") as factory,
              patch.object(smoke, "fixtures") as fixtures):
            for opt_in in (False, None, 1):
                with self.assertRaises(PermissionError):
                    asyncio.run(smoke.run_smoke(provider="vllm", model="mock-model",
                                              output=self.root, allow_inference=opt_in))
            factory.assert_not_called()
            fixtures.assert_not_called()
        self.assertFalse(self.root.exists())

    def test_cli_default_refuses_and_help_is_offline(self):
        with patch.object(smoke, "run_smoke") as run, redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                smoke.main(["--provider", "vllm", "--model", "mock-model",
                            "--output", str(self.root)])
            self.assertEqual(raised.exception.code, 2)
            run.assert_not_called()
        code = (
            "import socket, sys; from unittest.mock import patch; "
            "guard=patch('socket.create_connection',side_effect=AssertionError('network')); "
            "guard.start(); from src.utils import live_smoke; live_smoke.main(['--help'])")
        result = subprocess.run([sys.executable, "-B", "-c", code], cwd=smoke.REPO,
                                capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--allow-inference", result.stdout)
        self.assertFalse(self.root.exists())

    def test_failure_has_no_retry_and_errors_are_sanitized(self):
        report = self.run_mock(error=ProviderHTTPError(503, "SECRET_HEADER"))
        self.assertFalse(report["ok"])
        self.assertTrue(report["coverage_complete"])
        self.assertEqual([r["actual_calls"] for r in report["tasks"]], [1] * 12)
        self.assertEqual(len(self.created[0].requests), 12)
        self.assertNotIn("SECRET_HEADER", self.artifacts())
        for row in report["tasks"]:
            self.assertEqual(row["errors"][0]["diagnostic_code"], "http_error")
            self.assertEqual(row["errors"][0]["http_status"], 503)

    def test_exception_text_and_raw_payload_never_persist(self):
        report = self.run_mock(error=InvalidResponseError(
            "SECRET_ERROR_MESSAGE", raw_output={"token": "SECRET_ERROR_BODY"}))
        self.assertFalse(report["ok"])
        self.assertNotIn("SECRET_ERROR", self.artifacts())

    def test_valid_response_without_transport_evidence_cannot_claim_coverage(self):
        report = self.run_mock(evidence=False)
        self.assertFalse(report["ok"])
        self.assertFalse(report["coverage_complete"])
        self.assertTrue(all(r["actual_calls"] == 0 for r in report["tasks"]))
        self.assertTrue(all(r["protocol_status"] == "failure" for r in report["tasks"]))

    def test_unsupported_validation_is_not_counted_as_real_call(self):
        report = self.run_mock(limit=2)
        self.assertFalse(report["coverage_complete"])
        for row in report["tasks"]:
            if row["task"] in ("degree_exact", "arxiv", "prime", "tsp_construct", "lt_influence_construct"):
                self.assertEqual(row["actual_calls"], 0)

    def test_existing_output_cannot_resume_or_retry(self):
        self.root.mkdir()
        with patch.object(smoke, "make_client") as factory:
            with self.assertRaises(FileExistsError):
                asyncio.run(smoke.run_smoke(provider="vllm", model="mock-model", output=self.root,
                                          allow_inference=True))
            factory.assert_not_called()

    def test_base_url_validation_is_offline_and_copilot_rejects_override(self):
        for provider, url in (("vllm", "http://user:secret@localhost/v1"),
                              ("github_copilot", "http://localhost/v1")):
            with self.assertRaises(ValueError):
                asyncio.run(smoke.run_smoke(provider=provider, model="mock-model", output=self.root,
                                          base_url=url, allow_inference=True))
        self.assertFalse(self.root.exists())

    def test_vllm_real_adapter_with_mock_http_has_one_attempt_no_body_leak(self):
        from src.clients.vllm_client import VLLMClient

        requests = []

        def handler(request):
            requests.append(request)
            return httpx.Response(503, json={"error": "SECRET_HTTP_BODY"})

        def adapter(**kwargs):
            return VLLMClient(**kwargs, api_key="", transport=httpx.MockTransport(handler))

        with (patch("src.clients.vllm_client.VLLMClient", side_effect=adapter),
              patch.object(smoke, "fixtures", return_value=deepcopy(self.jobs)),
              redirect_stdout(io.StringIO())):
            report = asyncio.run(smoke.run_smoke(
                provider="vllm", model="mock-model", output=self.root, allow_inference=True,
                client_factory=smoke.make_client))
        self.assertEqual(len(requests), 12)
        self.assertEqual([r["actual_calls"] for r in report["tasks"]], [1] * 12)
        self.assertTrue(all(r["http_statuses"] == [503] for r in report["tasks"]))
        self.assertNotIn("SECRET_HTTP_BODY", self.artifacts())
        self.assertFalse(report["live_coverage_complete"])

    def test_sdk_observed_retry_is_rejected_and_turns_not_double_counted(self):
        def factory(config, observations):
            class RetryClient(MockClient):
                async def predict(self, request):
                    self.requests.append(request)
                    observations.event("assistant.turn_start")
                    observations.event("assistant.usage")
                    observations.event("assistant.turn_retry")
                    observations.event("assistant.turn_start")
                    observations.event("assistant.usage")
                    return DecisionResponse(request.request_id, request.options[0].id, "mock-model")
            return RetryClient(observations)

        with (patch.object(smoke, "fixtures", return_value=deepcopy(self.jobs)),
              redirect_stdout(io.StringIO())):
            report = asyncio.run(smoke.run_smoke(
                provider="github_copilot", model="mock-model", output=self.root,
                allow_inference=True, client_factory=factory))
        self.assertFalse(report["ok"])
        self.assertTrue(all(row["retry_observed"] for row in report["tasks"]))
        self.assertEqual([row["actual_calls"] for row in report["tasks"]], [2] * 12)

    def test_call_timeout_does_not_retry_and_later_types_are_attempted(self):
        async def slow_predict(client, request):
            client.requests.append(request)
            client.observations.event("http_request")
            await asyncio.Future()

        with (patch.object(MockClient, "predict", slow_predict),
              patch.object(smoke, "CALL_SECONDS", 0.01)):
            report = self.run_mock()
        self.assertFalse(report["ok"])
        self.assertTrue(report["coverage_complete"])
        self.assertEqual([row["actual_calls"] for row in report["tasks"]], [1] * 12)
        self.assertTrue(self.created[0].closed)
        self.assertLess(report["elapsed_seconds"], 3)

    def test_task_deadline_seals_partial_episode_without_retry(self):
        async def slow_predict(client, request):
            client.requests.append(request)
            client.observations.event("http_request")
            await asyncio.Future()

        with (patch.object(MockClient, "predict", slow_predict),
              patch.object(smoke, "OVERALL_SECONDS", 0.3)):
            report = self.run_mock()
        self.assertFalse(report["ok"])
        self.assertTrue(report["coverage_complete"])
        self.assertEqual([row["actual_calls"] for row in report["tasks"]], [1] * 12)
        for task, instance, _, arm in self.jobs[8:]:
            score = smoke.episodes.verify_episode(
                self.root / "episodes" / task, instance, smoke.model_config("vllm", "mock-model"), arm)
            self.assertFalse(score["feasible"])
            self.assertTrue(score["pending_call"])
            self.assertEqual(score["terminal_reason"], "smoke_interrupted_no_retry")
        self.assertEqual(len(list(self.root.rglob("smoke-interruption.json"))), 8)
        self.assertTrue(self.created[0].closed)

    def test_initialization_error_is_sanitized_and_client_closed(self):
        async def bad_initialize(client):
            raise RuntimeError("SECRET_INITIALIZATION_ERROR")

        with patch.object(MockClient, "initialize", bad_initialize):
            report = self.run_mock()
        self.assertFalse(report["ok"])
        self.assertTrue(all(row["actual_calls"] == 0 for row in report["tasks"]))
        self.assertTrue(self.created[0].closed)
        self.assertNotIn("SECRET_INITIALIZATION_ERROR", self.artifacts())

    def test_duplicate_request_is_blocked_before_second_delegation(self):
        async def exercise():
            self.root.mkdir()
            observations = smoke.Observations(self.root)
            observations.row = {"task": "adjacency", "predict_calls": 0, "actual_calls": 0,
                                "retry_observed": False, "http_statuses": [], "responses": 0,
                                "errors": []}
            client = MockClient(observations)
            bounded = smoke.BoundedClient(client, smoke.model_config("vllm", "mock-model"), observations)
            request = self.jobs[0][2]
            await bounded.predict(request)
            with self.assertRaises(InvalidResponseError):
                await bounded.predict(request)
            self.assertEqual(len(client.requests), 1)
        asyncio.run(exercise())


if __name__ == "__main__":
    unittest.main()