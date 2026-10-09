"""Offline, independent checks of sequential graph candidate abstractions."""

from copy import deepcopy
from dataclasses import asdict
from fractions import Fraction
from itertools import combinations, product
import json
from pathlib import Path
import random
import sys
import time
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.benchmark import extended_tasks, graph_abstraction as abstraction, public_tasks


TASKS = (
    "maxcut_construct", "community_bipartition_construct",
    "maxcut_public", "lt_influence_construct",
)


def fixture(task, n=5, edges=None):
    if edges is None:
        edges = ([[0, 1], [0, 4], [1, 2], [2, 3], [3, 1], [4, 3]]
                 if task == "lt_influence_construct" else
                 [[0, 2], [0, 4], [1, 2], [1, 3], [2, 3]])
        if task == "maxcut_public":
            edges = [[u, v, w] for (u, v), w in zip(edges, (3, -2, 0, 5, -4))]
    state = {"nodes": list(range(n)), "edges": sorted(deepcopy(edges))}
    if task != "maxcut_public":
        state["directed"] = task == "lt_influence_construct"
    if task == "lt_influence_construct":
        state.update(
            weights=[{"source": u, "target": v, "numerator": 1,
                      "denominator": sum(t == v for _, t in edges)} for u, v in state["edges"]],
            threshold={"numerator": 1, "denominator": 2}, seed_count=2,
        )
    return {
        "id": f"graph-{task}", "task": task, "state": state,
        "private": ({"reference": {"value": 1, "status": "best_known",
                                   "source": "https://PRIVATE_SENTINEL.invalid"}}
                    if task == "maxcut_public" else
                    {"objective": 0, "reference": "PRIVATE_SENTINEL", "baseline": {}}),
    }


def independent_closure(state, seeds):
    active = set(seeds)
    while True:
        totals = {v: sum(
            (Fraction(edge["numerator"], edge["denominator"])
             for edge in state["weights"] if edge["target"] == v and edge["source"] in active),
            Fraction(0)) for v in state["nodes"]}
        updated = active | {v for v, weight in totals.items() if weight >= Fraction(1, 2)}
        if updated == active:
            return active
        active = updated


def partition_objective(record, prefix):
    """Adjacency-matrix modularity also defines the partial-prefix potential."""
    edges = record["state"]["edges"]
    if record["task"] != "community_bipartition_construct":
        return sum((e[2] if len(e) == 3 else 1) for e in edges
                   if max(e[:2]) < len(prefix) and prefix[e[0]] != prefix[e[1]])
    m = len(edges)
    degree = {v: sum(v in e for e in edges) for v in record["state"]["nodes"]}
    adjacent = {tuple(e) for e in edges}
    return sum(
        (Fraction(int(tuple(sorted((u, v))) in adjacent)) -
         Fraction(degree[u] * degree[v], 2 * m))
        for u in range(len(prefix)) for v in range(len(prefix)) if prefix[u] == prefix[v]
    ) / (2 * m)


def independent_features(record, history):
    state, task = record["state"], record["task"]
    n, edges = len(state["nodes"]), state["edges"]
    if task == "lt_influence_construct":
        seeds = list(map(int, history))
        active = independent_closure(state, seeds)
        result = {}
        for v in state["nodes"]:
            if v in seeds:
                continue
            targets = {b for a, b in edges if a == v}
            reach = targets | {b for a, b in edges if a in targets}
            relief = Fraction(0)
            if v not in active:
                for t in targets - active:
                    incoming = [a for a, b in edges if b == t]
                    needed = len(incoming) // 2 + len(incoming) % 2
                    relief += Fraction(1, needed - sum(a in active for a in incoming))
            result[str(v)] = {
                "marginal_active": len(independent_closure(state, seeds + [v])) - len(active),
                "outdegree": len(targets), "two_hop_inactive": len(reach - active - {v}),
                "threshold_relief": {"numerator": relief.numerator, "denominator": relief.denominator},
            }
        return result
    prefix = ["A", *history]
    volume = [sum(abs(e[2]) if len(e) == 3 else 1 for e in edges if v in e[:2])
              for v in state["nodes"]]
    scale = 4 * len(edges) ** 2 if task == "community_bipartition_construct" else 1
    result = {}
    for side in ("A", "B"):
        assigned = [*prefix, side]
        rollout = assigned.copy()
        while len(rollout) < n:
            chosen = min(("A", "B"), key=lambda s: (-partition_objective(record, rollout + [s]), s))
            rollout.append(chosen)
        opposite = assigned + [("B" if side == "A" else "A")] * (n - len(assigned))
        result[side] = {
            "assigned_score": int(scale * (
                partition_objective(record, assigned) - partition_objective(record, prefix))),
            "rollout_score": int(scale * partition_objective(record, rollout)),
            "opposite_suffix_score": int(scale * partition_objective(record, opposite)),
            "negative_volume_imbalance": -abs(sum(
                volume[u] * (1 if s == "A" else -1) for u, s in enumerate(assigned))),
            "vertex_volume": volume[len(prefix)],
        }
    return result


def independent_proposals(record, features):
    lt = record["task"] == "lt_influence_construct"
    pairs = (
        (("marginal_cascade", "marginal_active"), ("outdegree", "outdegree"),
         ("two_hop_reach", "two_hop_inactive"), ("threshold_deficit", "threshold_relief"))
        if lt else
        (("assigned_gain", "assigned_score"), ("greedy_completion", "rollout_score"),
         ("opposite_completion", "opposite_suffix_score"),
         ("degree_balance", "negative_volume_imbalance"))
    )

    def value(action, feature):
        score = features[action][feature]
        return Fraction(score["numerator"], score["denominator"]) if isinstance(score, dict) else score

    return {rule: min(features, key=lambda a: (-value(a, feature), int(a) if lt else a))
            for rule, feature in pairs}


class GraphAbstractionTests(unittest.TestCase):
    def assert_step(self, record, history, seed):
        prepared = abstraction.prepare(record)
        expected_features = independent_features(record, history)
        expected = independent_proposals(record, expected_features)
        payloads = {arm: abstraction.step(prepared, history, arm, seed) for arm in ("A", "B", "C")}
        self.assertEqual(payloads["B"]["candidates"], payloads["C"]["candidates"])
        self.assertEqual(payloads["B"]["rule_proposals"], payloads["C"]["rule_proposals"])
        original = (public_tasks if record["task"] == "maxcut_public" else extended_tasks)
        self.assertEqual(payloads["A"]["request"], original.next_request(record, history))
        legal = {o.id for o in payloads["A"]["request"].options}
        for arm, payload in payloads.items():
            self.assertEqual(payload["rule_proposals"], expected)
            self.assertEqual(tuple(expected), abstraction.rules(record["task"]))
            candidates = payload["candidates"]
            actions = {row["action"] for row in candidates}
            self.assertEqual(actions, legal if arm == "A" else set(expected.values()))
            self.assertEqual(len(candidates), len(actions))
            self.assertEqual(len({row["option_id"] for row in candidates}), len(actions))
            self.assertTrue(actions <= legal)
            self.assertEqual(payload["legal_action_count"], len(legal))
            self.assertEqual(payload["distinct_candidate_count"], len(actions))
            self.assertEqual(payload["candidate_coverage"], len(actions) / len(legal))
            for row in candidates:
                self.assertEqual(row["features"], expected_features[row["action"]])
                self.assertEqual(row["rules"], [r for r, a in expected.items() if a == row["action"]])
            if len(actions) == 1:
                self.assertEqual(payload["forced_action"], next(iter(actions)))
                self.assertIsNone(payload["request"])
            else:
                self.assertIsNone(payload["forced_action"])
                self.assertEqual([row["option_id"] for row in candidates],
                                 [o.id for o in payload["request"].options])
                for row in candidates:
                    self.assertEqual(abstraction.selected_action(payload, row["option_id"]), row["action"])
            if arm != "A":
                self.assertEqual(len(legal), len(record["state"]["nodes"]) - len(history)
                                 if record["task"] == "lt_influence_construct" else 2)
                if payload["request"]:
                    self.assertEqual(payload["request"].state["edges"], record["state"]["edges"])
        b, c = payloads["B"]["request"], payloads["C"]["request"]
        if b is not None:
            self.assertEqual(b.state, {k: v for k, v in c.state.items()
                                      if k not in ("rules", "rule_groups", "rule_proposals")})
            self.assertNotIn("rules", b.state)
            self.assertTrue(all("rules" not in row for row in b.state["candidates"]))
            for rule in expected:
                self.assertNotIn(rule, " ".join(o.description for o in b.options))
            self.assertEqual([o.id for o in b.options], [o.id for o in c.options])
            self.assertEqual(c.state["rule_proposals"], expected)
            self.assertEqual(set(c.state["rules"]["formulas"]), set(expected))
        for payload in payloads.values():
            if payload["request"]:
                encoded = json.dumps(asdict(payload["request"]), allow_nan=False)
                self.assertNotIn("PRIVATE_SENTINEL", encoded)
                self.assertNotIn('"private"', encoded)
                self.assertNotIn('"reference"', encoded)
        return payloads

    def test_independent_candidates_all_small_histories(self):
        for task in TASKS:
            disagreements = set()
            for graph_seed in range(8):
                rng = random.Random(graph_seed)
                pairs = ([(u, v) for u in range(5) for v in range(5) if u != v]
                         if task == "lt_influence_construct" else list(combinations(range(5), 2)))
                edges = [list(e) for e in pairs if rng.random() < 0.4] or [[0, 1]]
                if task == "maxcut_public":
                    edges = [e + [rng.randint(-5, 7)] for e in edges]
                record = fixture(task, edges=edges)
                histories = ([[], *[[str(v)] for v in range(5)]]
                             if task == "lt_influence_construct" else
                             [list(h) for length in range(4) for h in product(("A", "B"), repeat=length)])
                for history in histories:
                    with self.subTest(task=task, graph_seed=graph_seed, history=history):
                        payloads = self.assert_step(record, history, 41)
                        proposals = payloads["B"]["rule_proposals"]
                        disagreements.update((a, b) for a, b in combinations(proposals, 2)
                                             if proposals[a] != proposals[b])
            self.assertEqual(disagreements, set(combinations(abstraction.rules(task), 2)),
                             f"{task}: each pair of rules should have a disagreement witness")

    def test_forced_steps_are_counted_and_a_is_not_reduced(self):
        for task in TASKS:
            edges = [[3, 4, 0]] if task == "maxcut_public" else (
                [[3, 4]] if task == "community_bipartition_construct" else [])
            record = fixture(task, edges=edges)
            prepared = abstraction.prepare(record)
            b = abstraction.step(prepared, [], "B", 7)
            self.assertEqual(b["distinct_candidate_count"], 1)
            self.assertEqual(b["candidate_coverage"], 1 / b["legal_action_count"])
            self.assertEqual(b["forced_action"], "0" if task == "lt_influence_construct" else "A")
            self.assertIsNone(b["request"])
            self.assertIsNotNone(abstraction.step(prepared, [], "A", 7)["request"])
            with self.assertRaises(ValueError):
                abstraction.selected_action(b, "0")
            if task == "lt_influence_construct" or task == "maxcut_public":
                history, transitions, calls = [], 0, 0
                while (payload := abstraction.step(prepared, history, "C", 7)) is not None:
                    transitions += 1
                    calls += payload["request"] is not None
                    history.append(payload["forced_action"])
                self.assertEqual(transitions, 2 if task == "lt_influence_construct" else 4)
                self.assertEqual(calls, 0)

    def test_lt_threshold_equality_cycles_zero_indegree_and_active_seed(self):
        record = fixture("lt_influence_construct", n=6,
                         edges=[[0, 2], [1, 2], [2, 3], [3, 2], [3, 4], [4, 3]])
        self.assertEqual(independent_closure(record["state"], [0]), {0})
        self.assertEqual(independent_closure(record["state"], [0, 1]), {0, 1, 2, 3, 4})
        payloads = self.assert_step(record, ["2"], 0)
        features = {row["action"]: row["features"] for row in payloads["A"]["candidates"]}
        self.assertEqual(features["3"]["marginal_active"], 0)
        self.assertEqual(features["3"]["threshold_relief"], {"numerator": 0, "denominator": 1})
        self.assertEqual(features["0"]["marginal_active"], 1)
        self.assertEqual(features["5"]["outdegree"], 0)
        self.assertIn("3", features)  # Already active is still a legal unselected seed.
        equality = fixture("lt_influence_construct", edges=[[0, 2], [1, 2], [2, 3]])
        self.assertEqual(independent_closure(equality["state"], [0]), {0, 2, 3})
        self.assert_step(equality, [], 12)
        self.assert_step(record, ["0"], 12)

    def test_signed_large_integer_cut_and_all_a_modularity(self):
        record = fixture("maxcut_public", edges=[
            [0, 1, -(2 ** 60)], [0, 2, 2 ** 60 + 1], [1, 2, 0], [3, 4, -7],
        ])
        self.assert_step(record, [], 2)
        self.assert_step(record, ["B", "A"], 2)
        community = fixture("community_bipartition_construct", edges=[[0, 1], [1, 2], [2, 3], [3, 4]])
        self.assertEqual(partition_objective(community, ["A"] * 5), 0)
        self.assert_step(community, ["A", "A", "A"], 2)

    def test_numeric_lt_tie_break_and_exact_anonymous_ids(self):
        record = fixture("lt_influence_construct", n=12, edges=[[2, 0], [10, 1]])
        payload = abstraction.step(abstraction.prepare(record), [], "B", 10)
        self.assertEqual(set(payload["rule_proposals"].values()), {"2"})
        self.assertEqual(payload["forced_action"], "2")
        for arm in ("B", "C"):
            payload = abstraction.step(abstraction.prepare(fixture("maxcut_construct")), [], arm, 0)
            self.assertEqual({row["action"] for row in payload["candidates"]}, {"A", "B"})
            for invalid in (0, True, "00", " 0", "0 ", "A", "B", "2", None):
                with self.assertRaises(ValueError):
                    abstraction.selected_action(payload, invalid)
            for rule, action in payload["rule_proposals"].items():
                self.assertEqual(abstraction.baseline_choice(payload, rule, random.Random(0)), action)

    def test_complete_episodes_preserve_original_objectives(self):
        for task in TASKS:
            record = fixture(task)
            for method in (*abstraction.rules(task), "random_candidate"):
                histories = []
                for arm in ("B", "C"):
                    prepared, history, rng = abstraction.prepare(record), [], random.Random(34)
                    while (payload := abstraction.step(prepared, history, arm, 81)) is not None:
                        action = abstraction.baseline_choice(payload, method, rng)
                        history.append(action)
                    histories.append(history)
                    scorer = public_tasks if task == "maxcut_public" else extended_tasks
                    result = scorer.score(record, history)
                    self.assertTrue(result["feasible"])
                    expected = (len(independent_closure(record["state"], list(map(int, history))))
                                if task == "lt_influence_construct"
                                else partition_objective(record, ["A", *history]))
                    self.assertAlmostEqual(result["objective"], float(expected))
                    self.assertIsNone(scorer.next_request(record, history))
                self.assertEqual(*histories)

    def test_rejects_invalid_history_arm_seed_task_and_option(self):
        for task in TASKS:
            prepared = abstraction.prepare(fixture(task))
            bad_histories = [None, (), "A", [True], [1], [" 0"], ["01"]]
            bad_histories += ([["0", "0"], ["5"], ["-1"], ["A"], ["0", "1", "2"]]
                              if task == "lt_influence_construct" else
                              [["a"], [" A"], ["B "], ["0"], ["A"] * 5])
            for arm in ("A", "B", "C"):
                for history in bad_histories:
                    with self.subTest(task=task, history=history, arm=arm), self.assertRaises(ValueError):
                        abstraction.step(prepared, history, arm, 0)
                for seed in (True, None, "0", 1.0):
                    with self.assertRaises(ValueError):
                        abstraction.step(prepared, [], arm, seed)
            for arm in (None, 0, "b", "D", [], True):
                with self.assertRaises(ValueError):
                    abstraction.step(prepared, [], arm, 0)
            payload = abstraction.step(prepared, [], "A", 0)
            for option in (None, 0, True, " 0", "01", "-1", " A", "missing"):
                with self.assertRaises(ValueError):
                    abstraction.selected_action(payload, option)
            for method in (None, [], "random", "unknown"):
                with self.assertRaises(ValueError):
                    abstraction.baseline_choice(payload, method, random.Random(0))
        for task in ("tsp_public", "degree_exact", None, [], True):
            with self.assertRaises(ValueError):
                abstraction.rules(task)
            with self.assertRaises(ValueError):
                abstraction.prepare({"id": "bad", "task": task, "state": {}})

    def test_completion_and_validation_are_not_bypassed(self):
        for task in TASKS:
            prepared = abstraction.prepare(fixture(task))
            history = ["0", "1"] if task == "lt_influence_construct" else ["A"] * 4
            for arm in ("A", "B", "C"):
                self.assertIsNone(abstraction.step(prepared, history, arm, 0))
                with self.assertRaises(ValueError):
                    abstraction.step(prepared, history + ["A"], arm, 0)
            with self.assertRaises(ValueError):
                abstraction.step(prepared, history, "D", 0)
            with self.assertRaises(ValueError):
                abstraction.step(prepared, history, "B", True)
            for key, value in (("nodes", [0, 2]), ("edges", [[0, 0]]), ("unexpected", "secret")):
                invalid = fixture(task)
                invalid["state"][key] = value
                with self.assertRaises(ValueError):
                    abstraction.prepare(invalid)
        lt = fixture("lt_influence_construct")
        lt["state"]["threshold"] = {"numerator": 1, "denominator": 3}
        with self.assertRaises(ValueError):
            abstraction.prepare(lt)

    def test_no_private_access_metadata_leak_or_global_rng_change(self):
        class Unreadable:
            def __deepcopy__(self, memo):
                raise AssertionError("private data was copied")

        before = random.getstate()
        with patch.object(extended_tasks, "_optimize", side_effect=AssertionError("oracle")), \
                patch.object(extended_tasks, "validate_instances", side_effect=AssertionError("oracle")), \
                patch.object(extended_tasks, "score", side_effect=AssertionError("score")), \
                patch.object(public_tasks, "score", side_effect=AssertionError("score")), \
                patch("socket.socket.connect", side_effect=AssertionError("offline")):
            for task in TASKS:
                record = fixture(task)
                clean = abstraction.prepare(record)
                expected = [abstraction.step(clean, [], arm, 14) for arm in ("A", "B", "C")]
                record.update(private=Unreadable(), metadata=Unreadable(), dataset_id=Unreadable(),
                              original_ids=Unreadable(), replicate=Unreadable())
                prepared = abstraction.prepare(record)
                self.assertEqual(set(prepared.instance), {"id", "task", "state"})
                actual = [abstraction.step(prepared, [], arm, 14) for arm in ("A", "B", "C")]
                self.assertEqual(actual, expected)
                record.pop("private")
                self.assertEqual(abstraction.prepare(record), prepared)
                record["state"]["edges"].clear()
                self.assertEqual(prepared.instance, clean.instance)
                if actual[0]["request"]:
                    actual[0]["request"].state["edges"].clear()
                    self.assertEqual(prepared.instance, clean.instance)
                self.assertEqual([abstraction.step(prepared, [], arm, 14) for arm in ("A", "B", "C")],
                                 expected)
        self.assertEqual(random.getstate(), before)

    def test_seed_only_changes_candidate_order_and_random_baseline_is_uniform_over_actions(self):
        record = fixture("maxcut_construct")
        prepared = abstraction.prepare(record)
        orders = set()
        for seed in range(16):
            payload = abstraction.step(prepared, [], "B", seed)
            orders.add(tuple(row["action"] for row in payload["candidates"]))
            self.assertEqual(payload["rule_proposals"],
                             abstraction.step(prepared, [], "C", seed)["rule_proposals"])
            self.assertEqual(payload["candidates"],
                             abstraction.step(prepared, [], "C", seed)["candidates"])
        self.assertEqual(orders, {("A", "B"), ("B", "A")})
        payload = abstraction.step(prepared, [], "B", 1)
        a, b = random.Random(42), random.Random(42)
        for _ in range(20):
            self.assertEqual(abstraction.baseline_choice(payload, "random_candidate", a),
                             b.choice(payload["candidates"])["action"])

    def test_public125_full_trajectory_without_exact_search(self):
        edges = [[u, v, (u + 3 * v) % 9 - 4] for u in range(125)
                 for v in range(u + 1, 125) if (u * 7 + v) % 5 == 0]
        record = fixture("maxcut_public", n=125, edges=edges)
        start = time.monotonic()
        with patch.object(extended_tasks, "_optimize", side_effect=AssertionError("exponential")), \
                patch.object(extended_tasks, "_objective", side_effect=AssertionError("wrong scorer")):
            prepared = abstraction.prepare(record)
            history, count = [], 0
            while (payload := abstraction.step(prepared, history, "B", 3)) is not None:
                self.assertEqual(payload["legal_action_count"], 2)
                self.assertIn(payload["candidate_coverage"], (0.5, 1))
                history.append(abstraction.baseline_choice(payload, "greedy_completion", random.Random(0)))
                count += 1
        self.assertEqual(count, 124)
        self.assertEqual(public_tasks.score(record, history)["objective"],
                         partition_objective(record, ["A", *history]))
        self.assertLess(time.monotonic() - start, 15, "public125 must not use exhaustive completion")


if __name__ == "__main__":
    unittest.main()
