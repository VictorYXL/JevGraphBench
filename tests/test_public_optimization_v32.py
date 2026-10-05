import asyncio
from copy import deepcopy
from fractions import Fraction
from itertools import combinations, product
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from src.clients.base import BaseDecisionClient, ClientCapabilities, DecisionResponse, TokenUsage
from src.utils import public_optimization_v32 as suite

Refusal = suite.shared.public.shared.Refusal


def fixture(task, n=6, edges=None):
    lt = task == suite.TASKS[2]
    edges = edges if edges is not None else [[v, v + 1] for v in range(n - 1)]
    state = {"nodes": list(range(n)), "edges": edges, "directed": lt}
    if lt:
        degree = suite.Counter(v for _, v in edges)
        state.update(weights=[{"source": u, "target": v, "numerator": 1, "denominator": degree[v]}
                              for u, v in edges],
                     threshold={"numerator": 1, "denominator": 2}, seed_count=2)
    record = {"id": "fixture", "task": task, "state": state, "query_budget": 2 if lt else n - 1,
              "state_sha256": suite.canonical_hash(state)}
    record["private"] = {"reference": suite.lt_reference(state) if lt else suite.modularity_reference(state)}
    return record


class FakeClient(BaseDecisionClient):
    def __init__(self, limit=None):
        self.calls, self.limit = 0, limit

    @property
    def capabilities(self):
        return ClientCapabilities(max_options=self.limit)

    async def predict(self, request):
        self.calls += 1
        return DecisionResponse(request.request_id, request.options[0].id, "fake",
                                usage=TokenUsage(), raw_output={"selected": request.options[0].id})


class ReferenceTests(unittest.TestCase):
    def test_lt_all_pairs_independent_fraction_diffusion(self):
        rng = random.Random(4)
        for _ in range(5):
            edges = [[u, v] for u in range(8) for v in range(8) if u != v and rng.random() < 0.25]
            instance = fixture(suite.TASKS[2], 8, edges)
            state = instance["state"]
            objectives = []
            for pair in combinations(range(8), 2):
                expected = suite.extended_tasks._independent_lt(state, pair)
                self.assertEqual(suite.lt_queue(suite.lt_structures(state), pair), expected)
                objectives.append(len(expected))
            self.assertEqual(instance["private"]["reference"]["value"], max(objectives))
            self.assertEqual(instance["private"]["reference"]["pairs_checked"], 28)

    def test_lt_equality_simultaneous_seeds_and_indegree_zero(self):
        instance = fixture(suite.TASKS[2], 6, [[0, 2], [1, 2], [2, 3], [4, 3]])
        state = instance["state"]
        self.assertEqual(suite.lt_queue(suite.lt_structures(state), [0, 5]), {0, 2, 3, 5})
        self.assertNotIn(1, suite.lt_queue(suite.lt_structures(state), [0, 5]))
        self.assertEqual(suite.lt_queue(suite.lt_structures(state), [0, 1]),
                         suite.lt_synchronous(suite.lt_structures(state), [0, 1]))

    def test_modularity_bound_all_a_and_no_false_optimum(self):
        instance = fixture(suite.TASKS[3])
        state, reference = instance["state"], instance["private"]["reference"]
        self.assertEqual(suite.modularity(state, ["A"] * 6), 0)
        exact = max(suite.modularity(state, ["A", *tail]) for tail in product(("A", "B"), repeat=5))
        self.assertLessEqual(reference["value"], float(exact))
        self.assertLessEqual(exact, Fraction(1, 2))
        scored = suite.score(instance, reference["solution"][1:])
        self.assertTrue(scored["feasible"])
        self.assertIsNone(scored["optimum"])
        self.assertIsNone(scored["absolute_gap"])
        self.assertFalse(scored["optimal"])
        self.assertEqual(scored["signed_reference_gap"], 0)

    def test_modularity_integer_energy_identity_and_label_symmetry(self):
        rng = random.Random(33)
        graphs = [
            (5, [[v, v + 1] for v in range(4)]),
            (5, [[0, 1], [0, 4], [1, 2], [2, 3], [3, 4]]),
            (5, [list(edge) for edge in combinations(range(5), 2)]),
            (6, [[0, 1], [1, 2], [3, 4]]),
            (6, [[0, 1]] + [
                [u, v] for u, v in combinations(range(6), 2)
                if (u, v) != (0, 1) and rng.random() < 0.4]),
        ]
        for n, edges in graphs:
            with self.subTest(n=n, edges=edges):
                state = fixture(suite.TASKS[3], n, edges)["state"]
                m = len(edges)
                for tail in product(("A", "B"), repeat=n - 1):
                    sides = ["A", *tail]
                    volume = sum(int(sides[u] == "A") + int(sides[v] == "A")
                                 for u, v in edges)
                    cut = sum(sides[u] != sides[v] for u, v in edges)
                    energy = (volume - m) ** 2 + 2 * m * cut
                    objective = suite.modularity(state, sides)
                    self.assertEqual(objective, Fraction(1, 2) - Fraction(energy, 2 * m * m))
                    self.assertEqual(objective, suite.extended_tasks._objective(
                        suite.TASKS[3], state, sides))
                    self.assertEqual(objective, suite.modularity(
                        state, ["B" if side == "A" else "A" for side in sides]))
                self.assertEqual(suite.modularity(state, ["A"] * n), 0)

    def test_feasible_reference_cannot_be_relabelled_as_certified(self):
        instance = fixture(suite.TASKS[3])
        instance["private"]["reference"]["status"] = "proven_optimum"
        with self.assertRaisesRegex(Refusal, "Feasible modularity reference drift"):
            suite.validate_reference(instance)

    def test_150_node_states_never_invoke_exponential_validation(self):
        with patch.object(suite.extended_tasks, "validate_instances", side_effect=AssertionError("exponential")), \
             patch.object(suite.extended_tasks, "_optimize", side_effect=AssertionError("exponential")):
            for task in suite.TASKS[2:]:
                instance = fixture(task, 150)
                suite.prepare_instance(instance)
                suite.validate_reference(instance)
                if task == suite.TASKS[2]:
                    self.assertEqual(instance["private"]["reference"]["pairs_checked"], 11175)

    def test_invalid_parameters_and_histories_rejected(self):
        instance = fixture(suite.TASKS[2])
        bad = deepcopy(instance)
        bad["state"]["weights"][0]["denominator"] = 8
        with self.assertRaises(Refusal):
            suite.prepare_instance(bad)
        for history in (["0", "0"], ["6"], [" 0"], [0], ["0", "1", "2"]):
            self.assertFalse(suite.score(instance, history)["feasible"])
        bad = fixture(suite.TASKS[3])
        bad["state"]["edges"] = []
        with self.assertRaises(Refusal):
            suite.validate_state(bad)


class ProposalTests(unittest.TestCase):
    def test_already_active_unselected_lt_vertex_remains_legal(self):
        instance = fixture(suite.TASKS[2], 4, [[0, 1], [1, 2]])
        prepared = suite.prepare_instance(instance)
        payload = suite.shared.step(prepared, ["0"], "A")
        self.assertEqual([option.id for option in payload["request"].options], ["1", "2", "3"])
        self.assertEqual(suite.score(instance, ["0", "1"])["objective"], 3)

    def test_original_a_and_named_c_are_next_actions(self):
        for task in suite.TASKS[2:]:
            instance = fixture(task, 12)
            prepared = suite.prepare_instance(instance)
            a = suite.shared.step(prepared, [], "A")
            self.assertEqual(a["request"], suite.extended_tasks.next_request(instance, []))
            c = suite.shared.step(prepared, [], "C")
            self.assertEqual(len(c["rule_proposals"]), 4)
            self.assertLessEqual(c["distinct_candidate_count"], 4 if task == suite.TASKS[2] else 2)
            self.assertTrue(all(type(row["action"]) is str for row in c["candidates"]))
            if c["request"]:
                self.assertIn("rule_proposals", c["request"].state)
                self.assertNotIn("private", c["request"].state)

    def test_partition_rules_exact_increment_and_fixed_order(self):
        instance = fixture(suite.TASKS[3])
        prepared = suite.prepare_instance(instance)
        history = ["B"]
        a = suite.shared.step(prepared, history, "A")
        self.assertEqual(a["request"].state["current_vertex"], 2)
        m, d = 5, 2
        for row in a["candidates"]:
            side = row["action"]
            internal = int(side == "B")
            volume = 1 if side == "A" else 2
            expected = 4 * m * internal - 2 * d * volume - d * d
            self.assertEqual(row["features"]["assigned_score"], expected)
        self.assertEqual(suite.score(instance, ["A"] * 5)["objective"], 0)

    def test_ten_controls_and_replay(self):
        instance = fixture(suite.TASKS[2])
        rows = suite.controls([instance])
        self.assertEqual(len(rows), 10)
        random_rows = [r for r in rows if r["method"] == "random_candidate"]
        self.assertEqual([r["seed"] for r in random_rows], list(suite.shared.RANDOM_SEEDS))
        for row in rows:
            self.assertTrue(row["feasible"])
            self.assertEqual(row["objective"], suite.score(instance, row["decisions"])["objective"])

    def test_sampler_is_fixed_and_only_induced_edges(self):
        nodes, edges = list(range(20)), [(v, v + 1) for v in range(19)]
        chosen = suite.sample_nodes(nodes, edges, 10, "fixed")
        self.assertEqual(chosen, suite.sample_nodes(list(reversed(nodes)), list(reversed(edges)), 10, "fixed"))
        self.assertEqual(len(set(chosen)), 10)
        raw = b"0 0\n0 1\n1 0\n1 2\n2 1\n"
        nodes, edges, note = suite.snap_graph(raw, False, 3)
        self.assertEqual(edges, [(0, 1), (1, 2)])
        self.assertEqual(note["self_loop_rows_removed"], 1)
        self.assertEqual(note["duplicate_or_reciprocal_rows_collapsed"], 2)


class DurabilityTests(unittest.TestCase):
    def test_approval_is_scope_and_protocol_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            suite.write_json(root / "protocol.json", {"test": True})
            with self.assertRaises(Refusal):
                suite.check_approval(root, "main")
            suite.write_json(root / "deployment-approval.json", {
                "protocol_sha256": suite.digest(root / "protocol.json"),
                "authorized": True, "scope": "pilot"})
            suite.check_approval(root, "pilot")
            with self.assertRaises(Refusal):
                suite.check_approval(root, "main")
            suite.write_json(root / "deployment-approval.json", {
                "protocol_sha256": suite.digest(root / "protocol.json"),
                "authorized": True, "scope": "main", "pilot_validated": True})
            suite.check_approval(root, "main")
            suite.write_json(root / "protocol.json", {"changed": True})
            with self.assertRaises(Refusal):
                suite.check_approval(root, "main")

    def test_run_cannot_create_client_before_approval(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(suite, "load", return_value=(root, {"models": ["jev_action"]}, {})), \
                 patch.object(suite, "runtime", side_effect=AssertionError("created client")):
                with self.assertRaises(Refusal):
                    asyncio.run(suite.run(root, ["jev_action"]))

    def test_report_preserves_all_unattempted_cells_and_a_c_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            instances = {}
            for task in suite.TASKS:
                for index in range(10):
                    identifier = f"{task}-{index}"
                    instances[identifier] = {"id": identifier, "task": task}
            suite.write_json(root / "protocol.json", {"test": True})
            suite.write_json(root / "controls.json", [])
            suite.write_json(root / "schedule.json", [
                {"instance_id": i, "arm": arm} for i in instances for arm in suite.ARMS])
            with patch.object(suite, "load", return_value=(
                    root, {"models": list(suite.shared.PANEL)}, instances)):
                result = suite.report(root)
            self.assertEqual(result["scheduled"], 1120)
            self.assertEqual(result["excluded"], 0)
            self.assertEqual(len(result["summaries"]), 112)
            self.assertEqual(len(result["paired_contrasts"]), 56)
            self.assertEqual({r["arm"] for r in result["rows"]}, {"A", "C"})
            self.assertTrue(all(r["terminal_reason"] == "unattempted" for r in result["rows"]))
            self.assertFalse(result["inference_complete"])

    def test_runtime_isolation(self):
        before = suite.shared.prepare
        runner = suite.runtime()
        self.assertIs(suite.shared.prepare, before)
        self.assertIs(runner.prepare, suite.prepare_instance)
        self.assertEqual(runner.ARMS, ("A", "C"))
        with self.assertRaises(Refusal):
            runner.step(suite.prepare_instance(fixture(suite.TASKS[2])), [], "B")

    def test_full_capture_replay_and_tamper(self):
        runner = suite.runtime()
        instance = fixture(suite.TASKS[2])
        config = runner.validate_config({"provider": "vllm", "model": "fake", "timeout_seconds": 1})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episode"
            client = FakeClient()
            asyncio.run(runner.execute_episode(path, instance, config, "A", client, asyncio.Event()))
            result = runner.verify_episode(path, instance, config, "A")
            self.assertTrue(result["feasible"])
            self.assertEqual(result["captured_response_records"], 2)
            self.assertEqual(client.calls, 2)
            with (path / "events.jsonl").open("a") as stream:
                stream.write("{}\n")
            with self.assertRaises(Refusal):
                runner.verify_episode(path, instance, config, "A")

    def test_interrupted_intent_is_never_retried(self):
        runner = suite.runtime()
        instance = fixture(suite.TASKS[2])
        config = runner.validate_config({"provider": "vllm", "model": "fake", "timeout_seconds": 1})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            request = runner.step(runner.prepare(instance), [], "A")["request"]
            with (path / "events.jsonl").open("w") as stream:
                runner.audit.append_row(stream, {"event": "call_started", "step": 0,
                                                "request": runner.audit.json_value(suite.asdict(request))})
            (path / "raw-responses.jsonl").touch()
            result = runner.seal_episode(path, instance, config, "A", "interrupted_no_retry")
            self.assertTrue(result["pending_call"])
            self.assertIsNone(result["model_calls"])
            self.assertFalse(result["feasible"])
            runner.verify_episode(path, instance, config, "A")

    def test_unsupported_retained_and_zero_transport(self):
        runner = suite.runtime()
        instance = fixture(suite.TASKS[2])
        config = runner.validate_config({"provider": "vllm", "model": "fake", "timeout_seconds": 1})
        with tempfile.TemporaryDirectory() as directory:
            path, client = Path(directory) / "episode", FakeClient(limit=2)
            asyncio.run(runner.execute_episode(path, instance, config, "A", client, asyncio.Event()))
            result = runner.verify_episode(path, instance, config, "A")
            self.assertFalse(result["feasible"])
            self.assertEqual(client.calls, 0)
            self.assertEqual(result["model_calls"], 0)
            self.assertEqual(result["failure"], "unsupported")

    def test_semantic_overrides_cannot_change_sampling_or_caps(self):
        a = {"provider": "vllm", "model": "Qwen3.5-4B", "think": False, "base_url": "http://a"}
        b = {**a, "base_url": "http://b"}
        self.assertEqual(suite.semantic_spec(a), suite.semantic_spec(b))
        b["think"] = True
        self.assertNotEqual(suite.semantic_spec(a), suite.semantic_spec(b))


if __name__ == "__main__":
    unittest.main()
