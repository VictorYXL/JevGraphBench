"""Exact offline trajectory diagnostics; no model clients or inference."""

from copy import deepcopy
from itertools import permutations, product
import json
import math
from pathlib import Path
import unittest
from unittest.mock import patch

from src.benchmark import extended_tasks as tasks
from src.benchmark import trajectory


class TrajectoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.instances = [i for i in tasks.build_curriculum(node_counts=(5,), samples_per_size=1)
                         if i["kind"] == "optimization"]

    def histories(self, instance):
        if instance["task"] == "tsp_construct":
            return [list(map(str, h)) for h in permutations(range(1, 5), 3)]
        if instance["task"] == "lt_influence_construct":
            return [list(map(str, h)) for h in permutations(range(5), 2)]
        return [list(h) for h in product(("A", "B"), repeat=4)]

    def test_every_small_history_and_prefix_matches_exhaustive_completion(self):
        for instance in self.instances:
            histories = self.histories(instance)
            scores = [(h, tasks.score(instance, h)) for h in histories]
            minimize = instance["task"] == "tsp_construct"
            best = min if minimize else max
            prefixes = {tuple(h[:n]) for h in histories for n in range(len(h) + 1)}
            for prefix in prefixes:
                history = list(prefix)
                with self.subTest(task=instance["task"], history=history):
                    report = trajectory.analyze_trajectory(instance, history)
                    reachable = best(s["objective"] for h, s in scores if h[:len(history)] == history)
                    self.assertAlmostEqual(report["best_reachable_objective"], reachable)
                    self.assertAlmostEqual(
                        math.fsum(s["incremental_regret"] for s in report["steps"]),
                        report["unavoidable_gap"])
                    self.assertEqual(report["global_optimum_reachable"],
                                     math.isclose(reachable, report["optimum"], abs_tol=1e-12))
                    self.assertEqual(report["first_loss_step"], next(
                        (s["step"] for s in report["steps"] if not s["locally_optimal"]), None))
                    if len(history) == instance["query_budget"]:
                        scored = tasks.score(instance, history)
                        self.assertTrue(report["feasible"])
                        self.assertAlmostEqual(report["absolute_gap"], scored["absolute_gap"])
                        self.assertAlmostEqual(report["unavoidable_gap"], scored["absolute_gap"])
                    else:
                        self.assertFalse(report["feasible"])
                        self.assertIsNone(report["absolute_gap"])
                        self.assertIsNone(report["objective"])
                        values = trajectory.continuation_values(instance, history)
                        options = tasks.next_request(instance, history).options
                        self.assertEqual(set(values), {option.id for option in options})
                        for action, value in values.items():
                            expected = best(s["objective"] for h, s in scores
                                            if h[:len(history) + 1] == history + [action])
                            self.assertAlmostEqual(value, expected)
                    json.dumps(report, allow_nan=False)

    def test_later_locally_optimal_action_does_not_restore_global_optimality(self):
        instance = next(i for i in self.instances if i["task"] == "maxcut_construct")
        reports = [trajectory.analyze_trajectory(instance, h) for h in self.histories(instance)]
        witness = next(r for r in reports if any(
            s["locally_optimal"] and not s["global_optimum_reachable"] for s in r["steps"]))
        first = witness["first_loss_step"]
        self.assertIsNotNone(first)
        self.assertTrue(all(not s["global_optimum_reachable"] for s in witness["steps"][first - 1:]))

    def test_validation_rejects_invalid_histories_and_corrupted_records(self):
        instance = next(i for i in self.instances if i["task"] == "tsp_construct")
        for history in (None, (), [1], [" 1"], ["1", "1"], ["0"], ["1"] * 4):
            with self.subTest(history=history), self.assertRaises(ValueError):
                trajectory.analyze_trajectory(instance, history)
        with self.assertRaises(ValueError):
            trajectory.continuation_values(instance, self.histories(instance)[0])
        exact = next(i for i in tasks.build_curriculum(node_counts=(5,), samples_per_size=1)
                     if i["kind"] == "exact")
        with self.assertRaises(ValueError):
            trajectory.analyze_trajectory(exact, [])
        broken = deepcopy(instance)
        broken["private"]["objective"] += 1
        with self.assertRaises(ValueError):
            trajectory.analyze_trajectory(broken, [])

    def test_analysis_is_read_only_and_oracle_data_never_enters_requests(self):
        for instance in self.instances:
            before = deepcopy(instance)
            history = self.histories(instance)[0]
            public = tasks.next_request(instance, [])
            trajectory.analyze_trajectory(instance, history)
            self.assertEqual(instance, before)
            self.assertEqual(tasks.next_request(instance, []), public)
            encoded = json.dumps(public.state)
            for key in ("private", "action_values", "best_reachable", "optimum", "reference"):
                self.assertNotIn(key, encoded)

    def test_corrupted_oracle_is_not_a_success_shaped_fallback(self):
        instance = self.instances[0]
        values = trajectory._values

        def corrupt(record, history):
            return {key: value + 1 for key, value in values(record, history).items()}

        with patch.object(trajectory, "_values", side_effect=corrupt):
            with self.assertRaisesRegex(ValueError, "oracle disagrees"):
                trajectory.analyze_trajectory(instance, [])

    def test_published_example_replays_without_archived_experiments(self):
        path = Path(__file__).resolve().parents[1] / "assets/benchmark/trajectory-example.json"
        example = json.loads(path.read_text())
        self.assertEqual(example["model"], "Jev-1.13.0")
        self.assertEqual(example["provenance"]["model_calls"], 0)
        report = trajectory.analyze_trajectory(example["instance"], example["decisions"])
        self.assertEqual(report, example["analysis"])
        self.assertEqual(report["optimum"], 192)
        self.assertEqual(report["objective"], 216)
        self.assertEqual(report["first_loss_step"], 2)
        self.assertTrue(report["steps"][3]["locally_optimal"])
        self.assertFalse(report["steps"][3]["global_optimum_reachable"])


if __name__ == "__main__":
    unittest.main()
