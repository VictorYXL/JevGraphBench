"""Offline tests for matched candidate/rule actions and durable episode audit."""

import asyncio
from copy import deepcopy
from dataclasses import asdict
from itertools import permutations
import json
from pathlib import Path
from unittest.mock import patch
import random
import subprocess
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.utils import tsp_abstraction_suite as suite
from src.benchmark import tsp_abstraction as abstraction
from src.benchmark import public_tasks
from src.clients.base import DecisionResponse, TokenUsage, ClientTimeoutError, ProviderHTTPError


def fixture(points=None):
    points = points or [[0, 0], [7, 0], [2, 3], [20, 8], [2, 10], [4, 4]]
    return {
        "id": "PRIVATE_ID", "dataset_id": "PRIVATE_DATASET", "replicate": 0,
        "original_ids": list(range(len(points))), "task": "tsp_public",
        "state": {"nodes": list(range(len(points))), "coordinates": points, "edge_weight_type": "EUC_2D"},
        "private": {"reference": {"value": 1, "status": "best_known", "source": "https://secret.invalid"}},
        "metadata": {"secret": 93847853},
    }


class FakeClient:
    def __init__(self, config, error=None):
        self.config, self.error, self.calls = config, error, 0

    async def initialize(self):
        pass

    async def aclose(self):
        pass

    async def predict(self, request):
        self.calls += 1
        if self.error:
            raise self.error
        option = request.options[0].id
        raw = None
        if self.config.provider == "typesafe":
            raw = {"model": self.config.model, "usage": None, "answers": {"decision": {
                "type": "choice", "choice": option,
                "probabilities": {o.id: 0.8 for o in request.options}, "confidence": 0.8}}}
        return DecisionResponse(request.request_id, option, self.config.model,
                                usage=TokenUsage(), raw_output=raw)


class AbstractionTests(unittest.TestCase):
    def test_source_snapshot_keeps_utilities_and_pinned_regression(self):
        hashes = suite.code_hashes()
        self.assertEqual(set(hashes), {*suite.audit.code_hashes(), "tests/integration/test_tsp_abstraction.py"})
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            suite.freeze(root, hashes)
            snapshot = root / suite.audit.SOURCE_SNAPSHOT
            script = snapshot / "src" / "utils" / "tsp_abstraction_suite.py"
            probe = (
                "import json,runpy,sys;"
                "module=runpy.run_path(sys.argv[1],run_name='snapshot_probe');"
                "print(json.dumps({'hashes':module['code_hashes'](),"
                "'root':str(module['REPO']),"
                "'public':module['public'].__file__}))")
            result = subprocess.run(
                [sys.executable, "-B", "-I", "-c", probe, str(script)],
                cwd=root, capture_output=True, text=True, timeout=30, check=True)
            restored = json.loads(result.stdout)
            self.assertEqual(restored["hashes"], hashes)
            self.assertEqual(Path(restored["root"]), snapshot)
            self.assertTrue(Path(restored["public"]).is_relative_to(snapshot))

    def test_prior_art_has_no_local_file_dependency(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "unavailable-checkout"
            with patch.object(suite, "REPO", root), \
                    patch.object(suite, "digest", side_effect=AssertionError("unexpected file read")), \
                    patch("builtins.open", side_effect=AssertionError("unexpected file read")), \
                    patch("io.open", side_effect=AssertionError("unexpected file read")):
                record = suite.prior_art()
            self.assertNotIn(str(root), json.dumps(record))
            self.assertFalse(root.exists())
        self.assertEqual(len(record["consulted"]), 3)
        for source in record["consulted"]:
            self.assertEqual(len(source["sha256"]), 64)
            self.assertFalse(source["document"].startswith("/"))

    def test_rules_and_matched_visibility_exhaustive_histories(self):
        record = fixture()
        prep = abstraction.prepare(record)
        matrix = suite.independent_matrix(record)
        for length in range(4):
            for history in permutations(range(1, 6), length):
                decisions = list(map(str, history))
                b = abstraction.step(prep, decisions, "B", 42)
                c = abstraction.step(prep, decisions, "C", 42)
                self.assertEqual(b["rule_proposals"], suite.independent_proposals(matrix, decisions))
                self.assertEqual(b["candidates"], c["candidates"])
                self.assertEqual(b["forced_city"], c["forced_city"])
                self.assertEqual(len({r["city"] for r in b["candidates"]}), len(b["candidates"]))
                if b["request"]:
                    bs, cs = b["request"].state, c["request"].state
                    self.assertEqual(bs, {k: v for k, v in cs.items() if k not in ("rules", "rule_groups")})
                    self.assertEqual(bs["coordinates"], record["state"]["coordinates"])
                    self.assertEqual(len(bs["current_city_distances"]), 5 - length)
                    self.assertNotIn("rules", bs)
                    self.assertNotIn("PRIVATE", json.dumps(asdict(b["request"])))
                    self.assertNotIn("93847853", json.dumps(asdict(c["request"])))
                    self.assertEqual([o.id for o in b["request"].options],
                                     [o.id for o in c["request"].options])

    def test_singleton_forced_and_final_closure(self):
        prep = abstraction.prepare(fixture([[0, 0]] * 4))
        first = abstraction.step(prep, [], "B", 1)
        self.assertIsNone(first["request"])
        self.assertEqual(first["forced_city"], 1)
        self.assertEqual(first["distinct_candidate_count"], 1)
        self.assertIsNone(abstraction.step(prep, ["1", "2"], "C", 1))
        score = public_tasks.score(prep.instance, ["1", "2"])
        self.assertEqual(score["solution"], [0, 1, 2, 3, 0])
        with self.assertRaises(ValueError):
            abstraction.selected_city(first, "0")

    def test_inputs_rng_and_exact_ids(self):
        record = fixture()
        before = deepcopy(record)
        rng = random.getstate()
        prep = abstraction.prepare(record)
        for arm in ("B", "C"):
            payload = abstraction.step(prep, [], arm, 41)
            self.assertEqual(payload["candidates"], abstraction.step(prep, [], arm, 41)["candidates"])
            if payload["request"]:
                for option in payload["request"].options:
                    city = abstraction.selected_city(payload, option.id)
                    self.assertIn(city, payload["rule_proposals"].values())
                with self.assertRaises(ValueError):
                    abstraction.selected_city(payload, " 0")
        self.assertEqual(record, before)
        self.assertEqual(random.getstate(), rng)
        for bad in (["0"], ["01"], ["1", "1"], [1], ["1"] * 5):
            with self.assertRaises(ValueError):
                abstraction.step(prep, bad, "B", 1)

    def test_fixed_rule_and_random_baselines(self):
        record = fixture()
        rows = {}
        for method in (*abstraction.RULES, "shortest_edge_selector", "random_candidate",
                       "nearest_neighbor", "nearest_neighbor_two_opt"):
            rows[method] = row = suite.baseline_episode(record, method, 17)
            self.assertTrue(row["feasible"])
            self.assertEqual(row["objective"], suite.independent_objective(record, row["decisions"]))
            self.assertEqual(row["decisions"], suite.baseline_episode(record, method, 17)["decisions"])
        self.assertEqual(rows["nearest_neighbor"]["decisions"],
                         rows["shortest_edge_selector"]["decisions"])

    def test_documented_source_real_distance(self):
        prep = abstraction.prepare(fixture([[347.42, 278.65], [461.42, 193.15], [0, 0]]))
        self.assertEqual(prep.distances[0][1], 143)

    def test_baseline_replay_detects_wrong_solution(self):
        record = fixture()
        row = suite.baseline_episode(record, "nearest", 1)
        row["objective"] += 1
        with self.assertRaises(suite.shared.Refusal):
            suite.verify_score(record, row)


class EpisodeTests(unittest.IsolatedAsyncioTestCase):
    async def execute(self, provider="qwen4", error=None, points=None):
        record = fixture(points)
        config = suite.public.configurations((provider,))[provider]
        client = FakeClient(config, error)
        abort = asyncio.Event()
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "episode"
            await suite.execute_episode(directory, record, config, "B", client, abort)
            rows = suite.audit.read_rows(directory / "steps.jsonl")
            result = suite.read_json(directory / "score.json")
            status = suite.read_json(directory / "status.json")
            replay, _ = suite.replay_episode(record, rows, config, "B")
            self.assertTrue(all(result[k] == v for k, v in replay.items()))
            self.assertEqual(client.calls, result["model_calls"])
            self.assertTrue(all(suite.digest(directory / f) == h for f, h in status["artifact_sha256"].items()))
            return record, config, rows, result, abort

    async def test_success_and_tamper_detection(self):
        record, config, rows, result, _ = await self.execute()
        self.assertTrue(result["feasible"])
        self.assertGreater(result["solve_wall_seconds"], 0)
        self.assertGreater(result["verification_seconds"], 0)
        damaged = deepcopy(rows)
        damaged[0]["proposal"]["candidates"][0]["city"] = 999
        with self.assertRaises(suite.shared.Refusal):
            suite.replay_episode(record, damaged, config, "B")

    async def test_failures_retained_without_retry_and_only_fatal_aborts_lane(self):
        for error, fatal, diagnostic in (
            (ClientTimeoutError("test"), False, "timeout"),
            (ProviderHTTPError(401), True, "http_error"),
        ):
            with self.subTest(error=type(error).__name__):
                _, _, rows, result, abort = await self.execute(error=error)
                self.assertFalse(result["feasible"])
                self.assertEqual(result["model_calls"], 1)
                self.assertIsNone(result["objective"])
                self.assertEqual(abort.is_set(), fatal)
                self.assertEqual(rows[-1]["attempt"]["diagnostic_code"], diagnostic)

    async def test_invalid_probability_preserves_native_action(self):
        _, _, rows, result, _ = await self.execute(provider="jev")
        self.assertTrue(result["feasible"])
        self.assertGreater(result["probability_audit_failures"], 0)
        for row in rows:
            if row["attempt"]:
                self.assertEqual(row["attempt"]["selected_option_id"], "0")
                self.assertGreater(row["attempt"]["probability_audit"]["sum"], 1)

    async def test_forced_steps_make_no_model_calls(self):
        _, _, rows, result, _ = await self.execute(points=[[0, 0]] * 5)
        self.assertTrue(result["feasible"])
        self.assertEqual(result["model_calls"], 0)
        self.assertEqual(result["forced_candidate_steps"], 3)
        self.assertTrue(all(row["attempt"] is None for row in rows))

    async def test_126_episode_controller_and_terminal_analysis_offline(self):
        instances = {}
        a_rows, baselines, jobs, coverage = [], [], [], []
        for graph in range(7):
            for repeat in range(3):
                record = fixture()
                record.update(id=f"graph{graph}-r{repeat}", dataset_id=f"graph{graph}", replicate=repeat)
                instances[record["id"]] = record
                baseline = suite.baseline_episode(record, "nearest_neighbor", 1)
                baselines.append(baseline)
                for arm in suite.ARMS:
                    jobs.append({"arm": arm, "instance_id": record["id"]})
                for model in suite.MODELS:
                    a_rows.append({
                        **baseline, "model": model, "arm": "A", "model_calls": 4,
                        "probability_audit_failures": 0, "failure_count": 0,
                        "legacy_wall_seconds_including_replay": .1,
                        "solve_wall_seconds": None,
                    })
                    coverage.append({"model": model, "dataset_id": record["dataset_id"],
                                     "selected_city_in_candidates": True})
        configs = suite.public.configurations(suite.MODELS)
        manifest = {"model_configs": {k: asdict(v) for k, v in configs.items()},
                    "input_sha256": {}, "code_sha256": {}}

        def factory(provider, **kwargs):
            config = next(c for c in configs.values() if c.model == kwargs["model"])
            return FakeClient(config)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for file, data in (
                ("schedule.json", jobs), ("protocol.json", manifest), ("a-reuse.json", a_rows),
                ("a-provenance.json", []), ("a-candidate-coverage.json", coverage),
                ("baselines.json", baselines),
            ):
                suite.write_json(root / file, data)
            with patch.object(suite.public, "create_public_client", factory), \
                    patch.object(suite, "load", return_value=(root, manifest, instances)), \
                    patch("builtins.print"):
                await asyncio.gather(*(suite.run_lane(root, manifest, instances, name)
                                       for name in suite.MODELS))
                result = suite.analyze(root)
            receipt = suite.read_json(root / "analysis/verification-receipt.json")
            self.assertEqual(receipt["new_scheduled_terminal_episodes"], 126)
            self.assertEqual(receipt["reused_A_episodes"], 63)
            self.assertEqual(result["recommendation"], "appendix_diagnostic")
            self.assertTrue((root / "analysis/episodes.csv").exists())


if __name__ == "__main__":
    unittest.main()
