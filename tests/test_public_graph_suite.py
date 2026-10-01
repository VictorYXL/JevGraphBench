"""Offline orchestration tests; no provider requests or credentials required."""

import asyncio
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from src.utils import public_graph_suite as suite
from src.clients.base import (
    BaseDecisionClient, ClientCapabilities, ClientTimeoutError, DecisionResponse,
    ProviderHTTPError, TokenUsage,
)


def instances():
    return [
        {"id": "square", "task": "tsp_public",
         "state": {"nodes": list(range(4)), "coordinates": [[0, 0], [3, 0], [3, 3], [0, 3]],
                   "edge_weight_type": "EUC_2D"},
         "private": {"reference": {"value": 12, "status": "proven_optimum",
                                   "source": "https://example.org/fixture"}}},
        {"id": "triangle", "task": "maxcut_public",
         "state": {"nodes": list(range(3)), "edges": [[0, 1, 2], [0, 2, 2], [1, 2, 2]]},
         "private": {"reference": {"value": 4, "status": "proven_optimum",
                                   "source": "https://example.org/fixture"}}},
    ]


class FakeClient(BaseDecisionClient):
    def __init__(self, model, behavior=None):
        self.model, self.behavior = model, behavior
        self.closed = False

    @property
    def capabilities(self):
        return ClientCapabilities()

    async def predict(self, request):
        await asyncio.sleep(0)
        if self.behavior:
            raise self.behavior
        return DecisionResponse(request.request_id, request.options[0].id, self.model,
                                usage=TokenUsage(12, 1, None),
                                raw_output={
                                    "model": self.model, "usage": {"input_tokens": 12, "output_tokens": 1},
                                    "answers": {"decision": {
                                        "type": "choice", "choice": request.options[0].id,
                                        "probabilities": {o.id: float(index == 0)
                                                          for index, o in enumerate(request.options)},
                                        "confidence": 1.0}},
                                    "secret": "never-persist-provider-body"})

    async def aclose(self):
        self.closed = True


class PublicSuiteTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.workspace = Path(temporary.name)
        self.root = self.workspace / "plan"
        self.inputs = self.workspace / "input.json"
        self.inputs.write_text(json.dumps(instances()))
        self.clients = []
        lock = patch.object(suite.shared.tempfile, "gettempdir", return_value=str(self.workspace))
        lock.start()
        self.addCleanup(lock.stop)

    def factory(self, provider, **kwargs):
        client = FakeClient(kwargs["model"])
        self.clients.append(client)
        return client

    async def preflight(self, configs):
        self.assertEqual(len(configs), 1)

    def plan(self, **kwargs):
        return suite.plan(self.root, self.inputs, models=("jev", "qwen4"),
                          repeats=2, concurrency=2, **kwargs)

    def evaluate(self, factory=None):
        return asyncio.run(suite.run_live(
            self.root, client_factory=factory or self.factory, preflight=self.preflight))

    def test_plan_and_full_replay(self):
        manifest = self.plan()
        self.assertEqual(manifest["budgets"]["episodes_per_model"], 4)
        untouched = suite.report(self.root)
        self.assertEqual(untouched["models"]["jev"]["status"], "not_started")
        self.assertEqual(untouched["models"]["jev"]["by_task"]["tsp_public"]["completed"], 0)
        report = self.evaluate()
        for lane in report["models"].values():
            for row in lane["episodes"]:
                self.assertTrue(row["feasible"])
                self.assertEqual(row["query_count"], 2)
                self.assertIsNotNone(row["objective"])
        self.assertTrue(all(client.closed for client in self.clients))
        self.assertEqual(report, suite.report(self.root))
        frozen = self.root / suite.audit.SOURCE_SNAPSHOT / "src" / "utils" / "public_graph_suite.py"
        replay = subprocess.run([sys.executable, str(frozen), "report", "--root", str(self.root)],
                                capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(replay.stdout), report)
        for path in (self.root / "runs").rglob("*.jsonl"):
            self.assertNotIn("never-persist-provider-body", path.read_text())
        with self.assertRaises(suite.shared.Refusal):
            self.evaluate()

    def test_nonfatal_failures_remain_in_denominators(self):
        self.plan()
        report = self.evaluate(lambda provider, **kwargs:
                               FakeClient(kwargs["model"], ClientTimeoutError("fixture")))
        for lane in report["models"].values():
            self.assertEqual(lane["status"], "completed_with_failures")
            for metrics in lane["by_task"].values():
                self.assertEqual(metrics["scheduled"], 2)
                self.assertEqual(metrics["completed"], 0)
                self.assertEqual(metrics["explicit_failures"], 2)
                self.assertIsNone(metrics["mean_gap_percent_feasible"])

    def test_auth_failure_aborts_without_repair(self):
        self.plan()
        report = self.evaluate(lambda provider, **kwargs:
                               FakeClient(kwargs["model"], ProviderHTTPError(401)))
        for lane in report["models"].values():
            self.assertEqual(lane["status"], "aborted")
            self.assertFalse(any(row["feasible"] for row in lane["episodes"]))
            self.assertLessEqual(sum(row["query_count"] for row in lane["episodes"]), 2)

    def test_probability_audit_does_not_replace_or_reject_native_action(self):
        class NoisyClient(FakeClient):
            async def predict(self, request):
                response = await super().predict(request)
                response.raw_output["answers"]["decision"]["probabilities"] = {
                    option.id: 0.39 if index == 0 else 0.60 / (len(request.options) - 1)
                    for index, option in enumerate(request.options)}
                return response

        self.plan()
        report = self.evaluate(lambda provider, **kwargs: NoisyClient(kwargs["model"]))
        lane = report["models"]["jev"]
        self.assertTrue(all(row["feasible"] for row in lane["episodes"]))
        self.assertEqual(sum(row["probability_audit_failures"] for row in lane["episodes"]), 8)
        for path in (self.root / "runs" / "jev").glob("*/attempts.jsonl"):
            for row in suite.audit.read_rows(path):
                vector = row["probability_audit"]
                self.assertEqual(vector["diagnostic_code"], "invalid_probability_sum")
                self.assertAlmostEqual(vector["sum"], 0.99)
                self.assertEqual(row["selected_option_id"], row["request"]["options"][0]["id"])
        self.assertEqual(report, suite.report(self.root))

    def test_checksum_and_score_tampering_refused(self):
        self.plan()
        self.evaluate()
        path = next((self.root / "runs" / "jev").rglob("score.json"))
        original = path.read_bytes()
        path.write_text("{}")
        with self.assertRaises(suite.shared.Refusal):
            suite.report(self.root)
        path.write_bytes(original)
        path = self.root / "instances.json"
        path.write_text("[]")
        with self.assertRaises(suite.shared.Refusal):
            suite.report(self.root)

    def test_permutation_preserves_full_graph(self):
        for instance in instances():
            original = deepcopy(instance)
            result = suite.relabel(instance, 2, 7)
            permutation = result["original_ids"]
            if instance["task"] == "tsp_public":
                self.assertEqual(result["state"]["coordinates"],
                                 [instance["state"]["coordinates"][i] for i in permutation])
            else:
                mapped = sorted([min(permutation[u], permutation[v]),
                                 max(permutation[u], permutation[v]), weight]
                                for u, v, weight in result["state"]["edges"])
                self.assertEqual(mapped, instance["state"]["edges"])
            self.assertEqual(instance, original)
            self.assertEqual(result["private"], instance["private"])

    def test_bad_selection_does_not_create_plan(self):
        with self.assertRaises(suite.shared.Refusal):
            self.plan(instance_ids=["absent"])
        self.assertFalse(self.root.exists())

    def test_prepare_cli_is_explicit_and_forwards_selection(self):
        root = self.workspace / "prepared"
        with patch("src.benchmark.public_data.prepare", return_value={"verified": True}) as prepare:
            with patch.object(sys, "argv", [
                "public_graph_suite.py", "prepare", "--root", str(root),
                "--include-calibration", "--dataset-ids", "kroA100",
            ]), redirect_stdout(io.StringIO()) as output:
                suite.main()
        prepare.assert_called_once_with(root, include_calibration=True, dataset_ids=["kroA100"])
        self.assertEqual(json.loads(output.getvalue()), {"verified": True})


if __name__ == "__main__":
    unittest.main()
