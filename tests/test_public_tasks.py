"""Focused offline tests with independent exhaustive tiny-instance objectives."""

from copy import deepcopy
from dataclasses import asdict
from decimal import Decimal, localcontext, ROUND_HALF_UP
from itertools import permutations, product
import json
import math
from pathlib import Path
import random
import sys
import unittest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.benchmark import public_tasks as tasks


def instance(task, state, value=100, status="best_known"):
    return {
        "id": "SECRET_DATASET_IDENTIFIER", "task": task, "state": state,
        "private": {"reference": {
            "value": value, "status": status, "source": "https://private.example/answer",
        }},
        "metadata": {"name": "SECRET_METADATA", "answer": 987654321},
    }


def tsp(coordinates=None, **kwargs):
    coordinates = coordinates if coordinates is not None else [[0, 0], [3, 0], [3, 4], [0, 4]]
    return instance("tsp_public", {
        "nodes": list(range(len(coordinates))), "coordinates": coordinates,
        "edge_weight_type": "EUC_2D",
    }, **kwargs)


def cut(n=4, edges=None, **kwargs):
    edges = edges if edges is not None else [[0, 1, 3], [1, 2, -2], [2, 3, 4], [0, 3, 1]]
    return instance("maxcut_public", {"nodes": list(range(n)), "edges": edges}, **kwargs)


def tour_length(coordinates, tour):
    return sum(int(math.sqrt((coordinates[u][0] - coordinates[v][0]) ** 2
                             + (coordinates[u][1] - coordinates[v][1]) ** 2) + 0.5)
               for u, v in zip(tour, tour[1:]))


def cut_weight(edges, sides):
    return sum(weight for u, v, weight in edges if sides[u] != sides[v])


class PublicTaskTests(unittest.TestCase):
    def test_standardized_api_metadata_and_input_rng_preservation(self):
        global_rng_before = random.getstate()
        for record in (tsp(), cut()):
            record.update(dataset_id="SECRET_DATASET", replicate=2,
                          original_ids=["SECRET_ORIGINAL_IDS"] * len(record["state"]["nodes"]))
            before = deepcopy(record)
            tasks.validate_instance(record)
            self.assertEqual(tasks.query_budget(record), tasks.decision_budget(record))
            self.assertEqual(tasks.allowed_baselines(record["task"]), tasks.BASELINES[record["task"]])
            for method in tasks.allowed_baselines(record["task"]):
                history = tasks.baseline(record, method, seed=5)
                self.assertEqual(len(history), tasks.query_budget(record))
                result = tasks.score(record, history)
                self.assertTrue(result["completed"])
                if record["task"] == "tsp_public":
                    self.assertEqual(result["solution"][0], 0)
                    self.assertEqual(result["solution"][-1], 0)
                for count in range(len(history)):
                    request = tasks.next_request(record, history[:count])
                    self.assertNotIn("SECRET", json.dumps(asdict(request)))
            self.assertEqual(record, before)
        self.assertEqual(random.getstate(), global_rng_before)
        with self.assertRaises(ValueError):
            tasks.allowed_baselines("unknown")

    def test_full_graph_all_actions_large_request(self):
        record = tsp([[i, i % 7] for i in range(254)])
        request = tasks.next_request(record, [])
        self.assertEqual(len(request.options), 253)
        self.assertEqual([o.id for o in request.options], [str(i) for i in range(1, 254)])
        self.assertEqual(request.state["coordinates"], record["state"]["coordinates"])
        self.assertEqual(len(request.state["current_city_distances"]), 253)
        self.assertEqual(tasks.decision_budget(record), 252)
        self.assertEqual(tasks.decision_budget(cut(n=254)), 253)

    def test_rounding_is_tsplib_not_bankers(self):
        record = tsp([[0, 0], [2.5, 0], [0, 1]])
        request = tasks.next_request(record, [])
        self.assertEqual(request.state["current_city_distances"], [[1, 3], [2, 1]])
        self.assertEqual(tasks.score(record, ["1"])["objective"], 7)
        self.assertIn("not bankers rounding", request.question)

    def test_source_real_tsp225_binary_float_convention(self):
        # Raw tsp225 decimal coordinates have half-integer cancellation cases.
        # Its official tour totals 3916 with float nint, 3921 with exact decimals;
        # silently changing to Decimal would invalidate the existing reference.
        pairs = [
            ([575.42, 250.15], [575.42, 271.65], 21),
            ([254.92, 271.65], [254.92, 243.15], 28),
            ([489.92, 136.15], [532.42, 136.15], 42),
            ([254.92, 335.65], [254.92, 136.15], 199),
            ([525.42, 356.65], [496.92, 356.65], 28),
            ([496.92, 271.65], [496.92, 243.15], 28),
            ([226.42, 264.15], [226.42, 235.65], 28),
        ]
        for a, b, expected in pairs:
            with self.subTest(a=a, b=b):
                record = tsp([a, b, a.copy()], value=1000)
                independent_float = int(math.sqrt(
                    (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) + 0.5)
                self.assertEqual(independent_float, expected)
                distances = dict(tasks.next_request(record, []).state["current_city_distances"])
                self.assertEqual(distances[1], expected)
                self.assertEqual(tasks.score(record, ["1"])["objective"], 2 * expected)

    def test_source_real_tsp225_edge_75_111_uses_double_sqrt_not_hypot(self):
        a, b = [347.42, 278.65], [461.42, 193.15]
        dx, dy = a[0] - b[0], a[1] - b[1]
        self.assertEqual(int(math.hypot(dx, dy) + 0.5), 142)
        self.assertEqual(int(math.sqrt(dx * dx + dy * dy) + 0.5), 143)
        record = tsp([a, b, a.copy()], value=1000)
        distances = dict(tasks.next_request(record, []).state["current_city_distances"])
        self.assertEqual(distances[1], 143)
        self.assertEqual(tasks.score(record, ["1"])["objective"], 286)

    def test_source_real_tsp225_float_half_edge_is_not_exact_decimal(self):
        a, b = [254.92, 271.65], [254.92, 243.15]
        exact_difference = abs(Decimal(str(a[1])) - Decimal(str(b[1])))
        self.assertEqual(exact_difference, Decimal("28.50"))
        self.assertEqual(int(exact_difference.to_integral_value(rounding=ROUND_HALF_UP)), 29)
        record = tsp([a, b, a.copy()], value=1000)
        self.assertEqual(dict(tasks.next_request(record, []).state["current_city_distances"])[1], 28)

    def test_available_official_tsp225_tour_preserves_3916_reference(self):
        data = REPO / "output/public-graph-data-20260929-v1"
        record_path = data / "task-instances/tsp225.json"
        tour_path = data / "raw/tsp225.opt.tour"
        if not record_path.is_file() or not tour_path.is_file():
            self.skipTest("optional acquired official tsp225 offline fixture is unavailable")
        record = json.loads(record_path.read_text())
        tokens = tour_path.read_text().split("TOUR_SECTION", 1)[1].split()
        tour = []
        for token in tokens:
            if token in ("-1", "EOF"):
                break
            tour.append(int(token) - 1)
        self.assertEqual(sorted(tour), list(range(225)))
        self.assertEqual(tour[0], 0)
        result = tasks.score(record, [str(v) for v in tour[1:-1]])
        self.assertEqual(result["solution"], [*tour, 0])
        self.assertEqual(result["objective"], 3916)
        self.assertEqual(result["reference"]["value"], 3916)
        self.assertEqual(result["additive_gap"], 0)
        with localcontext() as context:
            context.prec = 80
            coordinates = [[Decimal(str(x)) for x in point]
                           for point in record["state"]["coordinates"]]
            decimal_objective = sum(
                int(sum((a - b) ** 2 for a, b in zip(coordinates[u], coordinates[v]))
                    .sqrt().to_integral_value(rounding=ROUND_HALF_UP))
                for u, v in zip(result["solution"], result["solution"][1:])
            )
        self.assertEqual(decimal_objective, 3921)

    def test_all_tsp_histories_and_prefixes(self):
        record = tsp([[0, 0], [3, 0], [4, 4], [0, 4], [1, 2]])
        coordinates = record["state"]["coordinates"]
        values = [tour_length(coordinates, [0, *p, 0]) for p in permutations(range(1, 5))]
        record["private"]["reference"].update(value=min(values), status="proven_optimum")
        for history in permutations(range(1, 5), 3):
            decisions = [str(v) for v in history]
            for length in range(3):
                prefix = decisions[:length]
                request = tasks.next_request(record, prefix)
                remaining = [v for v in range(1, 5) if str(v) not in prefix]
                self.assertEqual([o.id for o in request.options], list(map(str, remaining)))
                self.assertEqual(request.state["partial_tour"], [0, *history[:length]])
                self.assertEqual([v for v, _ in request.state["current_city_distances"]], remaining)
                result = tasks.score(record, prefix)
                self.assertEqual(result["status"], "incomplete")
                self.assertIsNone(result["objective"])
                self.assertIsNone(result["additive_gap"])
                self.assertIsNone(result["percentage_gap"])
                self.assertEqual(result["deterministic_steps"], [])
            self.assertIsNone(tasks.next_request(record, decisions))
            result = tasks.score(record, decisions)
            self.assertEqual(result["status"], "feasible")
            self.assertTrue(result["feasible"])
            self.assertTrue(result["completed"])
            self.assertEqual(result["objective"], tour_length(coordinates, result["solution"]))
            self.assertEqual(sorted(result["solution"][:-1]), list(range(5)))
            self.assertEqual(result["solution"][0], result["solution"][-1])
            self.assertEqual(result["additive_gap"], result["objective"] - min(values))
            self.assertEqual(len(result["deterministic_steps"]), 2)
            self.assertTrue(all(not step["model_decision"] for step in result["deterministic_steps"]))
        self.assertEqual(tasks.score(tsp([[0, 0], [1, 0], [0, 1]]), ["1"])["solution"],
                         [0, 1, 2, 0])

    def test_all_cut_histories_and_prefixes_signed_weights(self):
        record = cut(edges=[[0, 1, -3], [1, 2, 7], [2, 3, -5], [0, 3, 4], [0, 2, 0]])
        values = [cut_weight(record["state"]["edges"], ["A", *p])
                  for p in product(("A", "B"), repeat=3)]
        record["private"]["reference"].update(value=max(values), status="proven_optimum")
        for choices in product(("A", "B"), repeat=3):
            for length in range(3):
                request = tasks.next_request(record, list(choices[:length]))
                self.assertEqual(request.state["edges"], record["state"]["edges"])
                self.assertEqual(request.state["partial_partition"], ["A", *choices[:length]])
                self.assertEqual(request.state["current_vertex"], length + 1)
                self.assertEqual([o.id for o in request.options], ["A", "B"])
            decisions = list(choices)
            self.assertIsNone(tasks.next_request(record, decisions))
            result = tasks.score(record, decisions)
            expected = cut_weight(record["state"]["edges"], ["A", *choices])
            self.assertEqual(result["objective"], expected)
            self.assertEqual(result["additive_gap"], max(values) - expected)
            self.assertEqual(result["deterministic_steps"], [])
        self.assertEqual(tasks.score(record, ["A"] * 3)["objective"], 0)
        self.assertLess(tasks.score(cut(), ["A", "B", "B"])["objective"], 0)

    def test_signed_best_known_gaps_and_corrupt_proven_reference(self):
        for record, decisions, objective in [
            (tsp(value=20), ["1", "2"], 14),
            (cut(n=2, edges=[[0, 1, 10]], value=8), ["B"], 10),
        ]:
            result = tasks.score(record, decisions)
            self.assertEqual(result["objective"], objective)
            self.assertLess(result["additive_gap"], 0)
            self.assertEqual(result["percentage_gap"],
                             100 * result["additive_gap"] / result["reference"]["value"])
            record["private"]["reference"]["status"] = "proven_optimum"
            with self.assertRaisesRegex(tasks.ReferenceCorruptionError, "proven optimum"):
                tasks.score(record, decisions)

    def test_reference_metadata_and_identifier_secrecy_and_copying(self):
        for record in (tsp(value=987654321), cut(value=987654321)):
            before = deepcopy(record)
            request = tasks.next_request(record, [])
            serialized = json.dumps(asdict(request))
            for secret in ("SECRET", "private", "987654321", "best_known", "reference"):
                self.assertNotIn(secret, serialized)
            changed = deepcopy(record)
            changed["id"] = "different"
            changed["metadata"] = {"different": "value"}
            changed["private"]["reference"].update(value=111, status="proven_optimum")
            self.assertEqual(asdict(request), asdict(tasks.next_request(changed, [])))
            request.state["nodes"].append(999)
            self.assertEqual(record, before)
            result = tasks.score(record, tasks.baseline_decisions(record, "random", seed=9))
            result["reference"]["value"] = 1
            self.assertEqual(record, before)

    def test_illegal_histories_are_not_repaired(self):
        for record, histories in [
            (tsp(), [None, (), {}, "1", [1], [True], ["0"], ["01"], [" 1"], ["1 "],
                     ["1", "1"], ["4"], ["-1"], ["1", "2", "3"], ["NaN"]]),
            (cut(), [None, ["a"], [" A"], ["B "], ["C"], [0], ["A"] * 4]),
        ]:
            for history in histories:
                with self.subTest(history=history, task=record["task"]):
                    with self.assertRaises(ValueError):
                        tasks.next_request(record, history)
                    result = tasks.score(record, history)
                    self.assertEqual(result["status"], "invalid")
                    self.assertFalse(result["feasible"])
                    self.assertFalse(result["completed"])
                    self.assertTrue(result["reason"])
                    for key in ("objective", "solution", "additive_gap", "percentage_gap"):
                        self.assertIsNone(result[key])

    def test_bad_instances_rejected(self):
        invalid = [None, {}, [], tsp([]), tsp([[0, 0]]), tsp([[0, 0]] * 255),
                   cut(n=1), cut(n=255), cut(edges=[])]
        changes = [
            ("id", ""), ("id", 1), ("task", "tsp_construct"), ("task", []),
            ("state", {}), ("private", {}), ("metadata", {"x": float("nan")}),
        ]
        for key, value in changes:
            record = tsp()
            record[key] = value
            invalid.append(record)
        for nodes in ([0, 1, 1, 3], [1, 2, 3, 4], [False, 1, 2, 3], [0., 1, 2, 3]):
            record = tsp()
            record["state"]["nodes"] = nodes
            invalid.append(record)
        for coordinates in (
            [[0, 0]], [[0, 0], [1, 0], [0, 1], [True, 0]],
            [[0, 0], [1, 0], [0, 1], [float("nan"), 0]],
            [[0, 0], [1, 0], [0, 1], [float("inf"), 0]],
            [[0, 0], [1, 0], [0, 1], ["2", 0]],
            [[0, 0], [1, 0], [0, 1], [1, 2, 3]],
            [[1e308, 0], [-1e308, 0], [0, 1]],
            [[10 ** 308, 0], [-10 ** 308, 0], [0, 1]],
        ):
            record = tsp()
            record["state"]["coordinates"] = coordinates
            if len(coordinates) == 3:
                record["state"]["nodes"] = [0, 1, 2]
            invalid.append(record)
        for metric in ("ATT", "GEO", "CEIL_2D", None):
            record = tsp()
            record["state"]["edge_weight_type"] = metric
            invalid.append(record)
        for edges in ([[0, 0, 1]], [[0, 4, 1]], [[-1, 2, 1]], [[0, 1]],
                      [[0, 1, 1.0]], [[0, True, 1]], [[0, 1, True]],
                      [[0, 1, float("inf")]], [[0, 1, 1], [1, 0, 2]]):
            invalid.append(cut(edges=edges))
        for key, value in (("value", 0), ("value", -1), ("value", True),
                           ("value", float("nan")), ("value", float("inf")),
                           ("status", "optimum"), ("source", "file:///secret"),
                           ("source", "https://"), ("source", 1)):
            record = tsp()
            record["private"]["reference"][key] = value
            invalid.append(record)
        for record in (tsp(), cut()):
            record["state"]["reference"] = 4
            invalid.append(record)
        for record in invalid:
            with self.subTest(record=record):
                with self.assertRaises(ValueError):
                    tasks.validate_instance(record)

    def test_baselines_deterministic_complete_and_nonworsening(self):
        rng = random.Random(391)
        for _ in range(8):
            records = [
                (tsp([[rng.randrange(30), rng.randrange(30)] for _ in range(7)]),
                 ("nearest_neighbor", "nearest_neighbor_two_opt")),
                (cut(n=7, edges=[[u, v, rng.randrange(-5, 10)]
                                for u in range(7) for v in range(u + 1, 7)]),
                 ("greedy", "greedy_single_flip")),
            ]
            for record, methods in records:
                objectives = []
                for method in (*methods, "random"):
                    history = tasks.baseline_decisions(record, method, seed=192)
                    self.assertEqual(history, tasks.baseline_decisions(record, method, seed=192))
                    self.assertEqual(len(history), tasks.decision_budget(record))
                    for index, choice in enumerate(history):
                        options = tasks.next_request(record, history[:index]).options
                        self.assertIn(choice, [option.id for option in options])
                    result = tasks.score(record, history)
                    self.assertTrue(result["feasible"])
                    objectives.append(result["objective"])
                if record["task"] == "tsp_public":
                    self.assertLessEqual(objectives[1], objectives[0])
                    tour = tasks.score(record, tasks.baseline_decisions(record, methods[1]))["solution"]
                    for start in range(1, 6):
                        for end in range(start + 1, 7):
                            candidate = tour[:start] + list(reversed(tour[start:end + 1])) + tour[end + 1:]
                            self.assertGreaterEqual(tour_length(record["state"]["coordinates"], candidate),
                                                    objectives[1])
                else:
                    self.assertGreaterEqual(objectives[1], objectives[0])
                    sides = tasks.score(record, tasks.baseline_decisions(record, methods[1]))["solution"]
                    for vertex in range(7):
                        flipped = sides.copy()
                        flipped[vertex] = "B" if flipped[vertex] == "A" else "A"
                        self.assertLessEqual(cut_weight(record["state"]["edges"], flipped), objectives[1])
                with self.assertRaises(ValueError):
                    tasks.baseline_decisions(record, "unknown")
                with self.assertRaises(ValueError):
                    tasks.baseline_decisions(record, "random", seed=True)

    def test_ties_zero_distances_and_zero_weights(self):
        record = tsp([[0, 0]] * 4, value=1)
        for method in ("nearest_neighbor", "nearest_neighbor_two_opt"):
            self.assertEqual(tasks.baseline_decisions(record, method), ["1", "2"])
        self.assertEqual(tasks.score(record, ["1", "2"])["objective"], 0)
        self.assertEqual(tasks.next_request(record, []).state["current_city_distances"],
                         [[1, 0], [2, 0], [3, 0]])
        record = cut(n=4, edges=[[0, 1, 0]], value=1)
        for method in ("greedy", "greedy_single_flip"):
            self.assertEqual(tasks.baseline_decisions(record, method), ["A"] * 3)
        self.assertEqual(tasks.score(record, ["B"] * 3)["objective"], 0)

    def test_uniform_random_history_matches_seeded_sampling(self):
        rng = random.Random(71)
        shuffled = [1, 2, 3]
        rng.shuffle(shuffled)
        self.assertEqual(tasks.baseline_decisions(tsp(), "random", seed=71),
                         list(map(str, shuffled[:-1])))
        rng = random.Random(71)
        self.assertEqual(tasks.baseline_decisions(cut(), "random", seed=71),
                         [rng.choice(("A", "B")) for _ in range(3)])

    def test_objective_and_requests_equivariant_under_relabeling(self):
        permutation = [0, 3, 1, 2]
        record = tsp()
        relabeled = deepcopy(record)
        for old, new in enumerate(permutation):
            relabeled["state"]["coordinates"][new] = record["state"]["coordinates"][old]
        original = tasks.score(record, ["1", "2"])
        mapped = tasks.score(relabeled, [str(permutation[v]) for v in (1, 2)])
        self.assertEqual(original["objective"], mapped["objective"])
        self.assertEqual([permutation[v] for v in original["solution"]], mapped["solution"])
        old_dist = dict(tasks.next_request(record, ["1"]).state["current_city_distances"])
        new_dist = dict(tasks.next_request(relabeled, ["3"]).state["current_city_distances"])
        self.assertEqual({permutation[v]: d for v, d in old_dist.items()}, new_dist)
        record = cut()
        relabeled = deepcopy(record)
        relabeled["state"]["edges"] = [[permutation[u], permutation[v], w]
                                      for u, v, w in record["state"]["edges"]]
        for choices in product(("A", "B"), repeat=3):
            sides = ["A", *choices]
            mapped = ["A"] * 4
            for old, new in enumerate(permutation):
                mapped[new] = sides[old]
            self.assertEqual(tasks.score(record, list(choices))["objective"],
                             tasks.score(relabeled, mapped[1:])["objective"])


if __name__ == "__main__":
    unittest.main()
