"""Matched append-only TSP candidate/rule abstractions; no reference access.

Four hand-designed integer scores propose one city each. Equal scores break by
city ID. Proposals are deduplicated by city, so rule multiplicity never changes
the number of options. B hides rule identities; C adds identities, formulas and
the rule-to-city mapping to otherwise identical graph and numeric features.
Singleton sets are deterministic transitions, not model calls. The public TSP
core still handles the final unvisited city and the return to city zero.
"""

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import random

from src.benchmark import public_tasks
from src.clients.base import DecisionOption, DecisionRequest

RULES = ("nearest", "one_step_lookahead", "return_reserve", "isolation_priority")
FORMULAS = {
    "nearest": "minimize e",
    "one_step_lookahead": "minimize e + h",
    "return_reserve": "minimize 2*e - r",
    "isolation_priority": "minimize e - h",
}
DEFINITIONS = (
    "c is the current city, U is ALL unvisited cities, s=0. "
    "e=d(c,v), h=min(d(v,w) for w in U excluding v), r=d(v,s). "
    "Each rule proposes its minimum-score city over all U; ties use lowest city ID. "
    "Rules with identical proposed cities are grouped into a single action."
)


@dataclass
class Prepared:
    instance: dict
    distances: list[list[int]]
    neighbor_order: list[list[int]]


def prepare(instance: dict) -> Prepared:
    public_tasks.validate_instance(instance)
    if instance["task"] != "tsp_public":
        raise ValueError("append-only abstraction supports only tsp_public")
    instance = deepcopy(instance)
    points = instance["state"]["coordinates"]
    n = len(points)
    matrix = [[public_tasks._distance(points, u, v) for v in range(n)] for u in range(n)]
    order = [sorted((v for v in range(n) if v != u), key=lambda v: (matrix[u][v], v))
             for u in range(n)]
    return Prepared(instance, matrix, order)


def step(prepared: Prepared, decisions: list[str], arm: str, seed: int) -> dict | None:
    if arm not in ("B", "C") or type(seed) is not int:
        raise ValueError("arm must be B or C and seed must be an integer")
    error = public_tasks._history_error(prepared.instance, decisions)
    if error:
        raise ValueError(error)
    n = len(prepared.distances)
    if len(decisions) == n - 2:
        return None
    tour = [0, *map(int, decisions)]
    unvisited = set(range(n)) - set(tour)
    current = tour[-1]
    features = {}
    for v in sorted(unvisited):
        neighbor = next(w for w in prepared.neighbor_order[v] if w in unvisited)
        features[v] = (prepared.distances[current][v],
                       prepared.distances[v][neighbor], prepared.distances[v][0])
    scores = {v: (e, e + h, 2 * e - r, e - h) for v, (e, h, r) in features.items()}
    proposals = {name: min(unvisited, key=lambda v: (scores[v][index], v))
                 for index, name in enumerate(RULES)}
    cities = sorted(set(proposals.values()))
    order_seed = hashlib.sha256(
        f"{seed}:{prepared.instance['id']}:{','.join(decisions)}".encode()).hexdigest()
    random.Random(order_seed).shuffle(cities)
    candidates = []
    for index, city in enumerate(cities):
        e, h, r = features[city]
        candidates.append({
            "option_id": str(index), "city": city, "edge": e,
            "nearest_other_unvisited": h, "return_to_start": r,
            "rules": [rule for rule in RULES if proposals[rule] == city],
        })
    payload = {
        "candidates": candidates, "rule_proposals": proposals,
        "unvisited_count": len(unvisited), "distinct_candidate_count": len(cities),
        "candidate_coverage": len(cities) / len(unvisited),
        "forced_city": cities[0] if len(cities) == 1 else None,
        "request": None,
    }
    if len(cities) == 1:
        return payload
    state = deepcopy(prepared.instance["state"])
    state.update(
        partial_tour=tour,
        current_city_distances=[[v, prepared.distances[current][v]] for v in sorted(unvisited)],
        candidates=[{k: value for k, value in row.items() if k != "rules"} for row in candidates],
    )
    question = (
        "Construct a minimum-length Hamiltonian cycle starting at city 0. The full "
        "coordinate graph and distances to ALL unvisited cities are supplied. "
        "Distances use TSPLIB double sqrt(dx*dx+dy*dy), then int(distance+0.5). "
        "Preserve partial_tour; the ONLY operation is append the selected candidate city. "
        "No insertion, reordering, lookahead execution, or local search is performed. "
        "Candidate numeric features are distance from current city (edge), minimum "
        "distance to another unvisited city (nearest_other_unvisited), and distance "
        "to city 0 (return_to_start). Once one city remains it and the return to 0 "
        "are forced without a model call. "
    )
    if arm == "B":
        question += "Select an anonymous candidate city; return only its option_id."
        options = tuple(DecisionOption(row["option_id"], f"Append city {row['city']}")
                        for row in candidates)
    else:
        state["rules"] = {"definitions": DEFINITIONS, "formulas": FORMULAS.copy(),
                          "tie_break": "lowest city ID"}
        state["rule_groups"] = [
            {"option_id": row["option_id"], "city": row["city"], "rules": row["rules"][:]}
            for row in candidates
        ]
        question += (
            "Select a rule-equivalence group. Executing the selected group appends "
            "its displayed city, exactly one step, then recomputes rules. Return only its option_id."
        )
        options = tuple(DecisionOption(
            row["option_id"], f"{', '.join(row['rules'])}; append city {row['city']}")
            for row in candidates)
    payload["request"] = DecisionRequest(
        request_id=f"tsp_abstraction:{arm}:{len(decisions)}",
        state=state, question=question, options=options,
    )
    return payload


def selected_city(payload: dict, option_id: str) -> int:
    if type(option_id) is not str or payload["forced_city"] is not None:
        raise ValueError("model selection requires an exact option ID and multiple candidates")
    matches = [row["city"] for row in payload["candidates"] if row["option_id"] == option_id]
    if len(matches) != 1:
        raise ValueError("illegal model option; no action repair")
    return matches[0]


def baseline_choice(payload: dict, method: str, rng: random.Random) -> int:
    if method in RULES:
        return payload["rule_proposals"][method]
    if method == "random_candidate":
        return rng.choice(payload["candidates"])["city"]
    if method == "shortest_edge_selector":
        return min(payload["candidates"], key=lambda row: (row["edge"], row["city"]))["city"]
    raise ValueError("unknown append-only baseline")
