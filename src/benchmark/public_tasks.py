"""Independent, full-graph public-instance construction and scoring.

Instances contain ``id``, ``task``, ``state``, ``private`` and optional JSON
``metadata``, ``dataset_id``, ``replicate`` and ``original_ids`` outside state.
``private`` is exactly ``{"reference": {"value": positive_number,
"status": "proven_optimum" | "best_known", "source": http_or_https_url}}``.
TSP state contains nodes, coordinates and edge_weight_type="EUC_2D"; MaxCut
state contains nodes and undirected integer-weighted edges. No private fields,
metadata, or instance identifiers are copied into DecisionRequest.

Histories are lists of exact option IDs, without normalization or repair.
TSP starts at 0 and has n-2 model decisions; its last city and return are
deterministic steps recorded by score. MaxCut fixes vertex 0 in A and has n-1
decisions. next_request returns None only for a valid completed history.
Every request exposes the entire graph and every legal action in numeric ID
order (A, B for cuts). TSP's current_city_distances contains [city, distance]
for EVERY unvisited city, calculated uniformly from the supplied coordinates;
it is arithmetic assistance, not a candidate filter or reference-derived hint.

score reports feasible/incomplete/invalid, never an objective for an unfinished
or invalid history. Directional gaps are objective-reference for minimization
and reference-objective for maximization; percentage_gap is 100*gap/reference.
Negative best-known gaps are retained. Improving a proven optimum raises
ReferenceCorruptionError. No exact optimization or trajectory oracle is used.

baseline returns a complete model-decision history compatible with
next_request and score. Methods are random (explicit seed), nearest_neighbor,
nearest_neighbor_two_opt (TSP), greedy and greedy_single_flip (MaxCut).
All deterministic ties use increasing vertex IDs; local search accepts strict
improvements only. Runtime measurement belongs to the caller.
"""

from __future__ import annotations

from copy import deepcopy
import json
import math
import random
from urllib.parse import urlsplit

from src.clients.base import DecisionOption, DecisionRequest

__all__ = [
    "BASELINES", "ReferenceCorruptionError", "validate_instance", "query_budget",
    "next_request", "score", "baseline", "allowed_baselines",
    "decision_budget", "baseline_decisions",
]

BASELINES = {
    "tsp_public": ("random", "nearest_neighbor", "nearest_neighbor_two_opt"),
    "maxcut_public": ("random", "greedy", "greedy_single_flip"),
}


def allowed_baselines(task: str) -> tuple[str, ...]:
    """Supported classical baseline names for a public task."""
    if type(task) is not str or task not in BASELINES:
        raise ValueError("unsupported public task")
    return BASELINES[task]


class ReferenceCorruptionError(ValueError):
    """A feasible solution strictly improved a declared proven optimum."""


def _finite_number(value: object) -> bool:
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _distance(coordinates: list, u: int, v: int) -> int:
    x, y = coordinates[u]
    a, b = coordinates[v]
    try:
        dx, dy = float(x) - float(a), float(y) - float(b)
        # Match TSPLIB's double sqrt operation order; hypot differs at some ties.
        distance = math.sqrt(dx * dx + dy * dy)
    except OverflowError as exc:
        raise ValueError("coordinate separation must have a finite EUC_2D distance") from exc
    if not math.isfinite(distance):
        raise ValueError("coordinate separation must have a finite EUC_2D distance")
    return int(distance + 0.5)


def validate_instance(instance: dict) -> None:
    """Validate the bounded simple-graph contract, not reference optimality."""
    if type(instance) is not dict:
        raise ValueError("instance must be a dict")
    required = {"id", "task", "state", "private"}
    optional = {"metadata", "dataset_id", "replicate", "original_ids"}
    if not required <= instance.keys() or instance.keys() - required - optional:
        raise ValueError("instance requires id, task, state, private; only declared metadata fields are optional")
    if type(instance["id"]) is not str or not instance["id"].strip():
        raise ValueError("id must be a nonempty string")
    task = instance["task"]
    if type(task) is not str or task not in ("tsp_public", "maxcut_public"):
        raise ValueError("unsupported public task")
    state = instance["state"]
    keys = ({"nodes", "coordinates", "edge_weight_type"} if task == "tsp_public"
            else {"nodes", "edges"})
    if type(state) is not dict or state.keys() != keys:
        raise ValueError("public state has missing or unknown fields")
    nodes = state["nodes"]
    minimum = 3 if task == "tsp_public" else 2
    if (type(nodes) is not list or not minimum <= len(nodes) <= 254
            or any(type(v) is not int for v in nodes)
            or nodes != list(range(len(nodes)))):
        raise ValueError(f"nodes must be consecutive integer IDs 0..n-1, {minimum} <= n <= 254")
    n = len(nodes)
    if task == "tsp_public":
        if state["edge_weight_type"] != "EUC_2D":
            raise ValueError("only TSPLIB EUC_2D is supported")
        coordinates = state["coordinates"]
        if (type(coordinates) is not list or len(coordinates) != n
                or any(type(p) is not list or len(p) != 2
                       or not all(_finite_number(x) for x in p) for p in coordinates)):
            raise ValueError("coordinates must be n finite numeric [x,y] pairs")
        for u in nodes:
            for v in range(u):
                _distance(coordinates, u, v)
    else:
        edges = state["edges"]
        if type(edges) is not list or not edges:
            raise ValueError("edges must be a nonempty list of [u,v,integer_weight]")
        seen = set()
        for edge in edges:
            if (type(edge) is not list or len(edge) != 3
                    or any(type(x) is not int for x in edge)):
                raise ValueError("each edge must be [u,v,integer_weight]")
            u, v, _ = edge
            if not 0 <= u < n or not 0 <= v < n or u == v:
                raise ValueError("edge endpoints must be distinct valid node IDs")
            key = (min(u, v), max(u, v))
            if key in seen:
                raise ValueError("duplicate undirected edge")
            seen.add(key)
    private = instance["private"]
    if type(private) is not dict or private.keys() != {"reference"}:
        raise ValueError("private must contain exactly reference")
    reference = private["reference"]
    if (type(reference) is not dict
            or reference.keys() != {"value", "status", "source"}):
        raise ValueError("reference must contain exactly value, status, source")
    if not _finite_number(reference["value"]) or reference["value"] <= 0:
        raise ValueError("reference value must be positive and finite")
    if reference["status"] not in ("proven_optimum", "best_known"):
        raise ValueError("reference status must be proven_optimum or best_known")
    source = reference["source"]
    if type(source) is not str:
        raise ValueError("reference source must be an HTTP(S) URL")
    try:
        parsed = urlsplit(source)
        valid_source = parsed.scheme in ("http", "https") and bool(parsed.netloc)
    except ValueError:
        valid_source = False
    if not valid_source:
        raise ValueError("reference source must be an HTTP(S) URL")
    try:
        json.dumps(instance, allow_nan=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("instance must contain finite JSON-serializable data") from exc


def _budget(instance: dict) -> int:
    return len(instance["state"]["nodes"]) - (2 if instance["task"] == "tsp_public" else 1)


def query_budget(instance: dict) -> int:
    """Number of scheduled model decisions, excluding deterministic closure."""
    validate_instance(instance)
    return _budget(instance)


def _history_error(instance: dict, decisions: list[str]) -> str | None:
    if type(decisions) is not list or any(type(d) is not str for d in decisions):
        return "decisions must be a list of exact string option IDs"
    if len(decisions) > _budget(instance):
        return "too many model decisions"
    if instance["task"] == "maxcut_public":
        if any(d not in ("A", "B") for d in decisions):
            return "illegal partition option ID"
    else:
        available = {str(v) for v in instance["state"]["nodes"][1:]}
        for d in decisions:
            if d not in available:
                return "illegal or repeated city option ID"
            available.remove(d)
    return None


def next_request(instance: dict, decisions: list[str]) -> DecisionRequest | None:
    """Build the next full-graph/all-actions request; reject invalid histories."""
    validate_instance(instance)
    error = _history_error(instance, decisions)
    if error:
        raise ValueError(error)
    if len(decisions) == _budget(instance):
        return None
    state = deepcopy(instance["state"])
    if instance["task"] == "tsp_public":
        tour = [0, *map(int, decisions)]
        unvisited = [v for v in state["nodes"] if v not in tour]
        state["partial_tour"] = tour
        state["current_city_distances"] = [
            [v, _distance(state["coordinates"], tour[-1], v)] for v in unvisited
        ]
        options = tuple(DecisionOption(str(v), f"Visit city {v}") for v in unvisited)
        question = (
            "Construct a minimum-length Hamiltonian cycle on the COMPLETE graph "
            "defined by all supplied coordinates. Each undirected edge uses "
            "TSPLIB EUC_2D distance floor(sqrt((x_u-x_v)^2+(y_u-y_v)^2)+0.5), "
            "not bankers rounding. current_city_distances gives this exact integer "
            "distance from the current city to EVERY unvisited city, uniformly "
            "computed without filtering. Start at 0 and preserve partial_tour. "
            "Choose any unvisited city. Once one city remains, it and the return "
            "to 0 are appended deterministically without a model decision. "
            "Return only the selected decimal city option ID."
        )
    else:
        state["partial_partition"] = ["A", *decisions]
        state["current_vertex"] = len(decisions) + 1
        options = (DecisionOption("A", "Assign to side A"),
                   DecisionOption("B", "Assign to side B"))
        question = (
            "Construct a maximum-weight cut of the FULL undirected graph. "
            "Each edge is [u,v,integer_weight]; signed and zero weights are allowed. "
            "Maximize the SUM OF WEIGHTS of edges whose endpoints have different "
            "sides. Vertex 0 is fixed in A. Assign current_vertex to A or B, "
            "preserving partial_partition; remaining vertices will be assigned "
            "in increasing ID order. Return only A or B."
        )
    return DecisionRequest(
        request_id=f"{instance['task']}:decision:{len(decisions)}",
        state=state, question=question, options=options,
    )


def score(instance: dict, decisions: list[str]) -> dict:
    """Score a construction independently; do not repair or complete model errors."""
    validate_instance(instance)
    error = _history_error(instance, decisions)
    complete = error is None and len(decisions) == _budget(instance)
    reference = deepcopy(instance["private"]["reference"])
    result = {
        "status": "invalid" if error else ("feasible" if complete else "incomplete"),
        "feasible": complete, "completed": complete,
        "reason": error if error else (None if complete else "incomplete decision history"),
        "objective": None, "solution": None,
        "direction": "min" if instance["task"] == "tsp_public" else "max",
        "reference": reference, "additive_gap": None, "percentage_gap": None,
        "decision_count": len(decisions) if type(decisions) is list else None,
        "decision_budget": _budget(instance), "deterministic_steps": [],
    }
    if not complete:
        return result
    state = instance["state"]
    if instance["task"] == "tsp_public":
        tour = [0, *map(int, decisions)]
        last = next(v for v in state["nodes"] if v not in tour)
        solution = [*tour, last, 0]
        objective = sum(_distance(state["coordinates"], u, v)
                        for u, v in zip(solution, solution[1:]))
        result["deterministic_steps"] = [
            {"kind": "forced_last_city", "vertex": last, "model_decision": False},
            {"kind": "return_to_start", "vertex": 0, "model_decision": False},
        ]
        gap = objective - reference["value"]
    else:
        solution = ["A", *decisions]
        objective = sum(w for u, v, w in state["edges"] if solution[u] != solution[v])
        gap = reference["value"] - objective
    if reference["status"] == "proven_optimum" and gap < 0:
        raise ReferenceCorruptionError(
            f"feasible {instance['task']} objective {objective} improves declared "
            f"proven optimum {reference['value']}; check instance/reference integrity"
        )
    result.update(objective=objective, solution=solution, additive_gap=gap,
                  percentage_gap=100 * (gap / reference["value"]))
    return result


def baseline(instance: dict, method: str, *, seed: int = 0) -> list[str]:
    """Return a complete construction history; seed is required to be an integer."""
    validate_instance(instance)
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    state = instance["state"]
    n = len(state["nodes"])
    rng = random.Random(seed)
    if method not in allowed_baselines(instance["task"]):
        raise ValueError("unsupported baseline for public task")
    if instance["task"] == "tsp_public":
        if method == "random":
            rest = list(range(1, n))
            rng.shuffle(rest)
            return [str(v) for v in rest[:-1]]
        distance = [[_distance(state["coordinates"], u, v) for v in range(n)]
                    for u in range(n)]
        tour = [0]
        unseen = set(range(1, n))
        while unseen:
            city = min(unseen, key=lambda v: (distance[tour[-1]][v], v))
            tour.append(city)
            unseen.remove(city)
        if method == "nearest_neighbor_two_opt":
            improved = True
            while improved:
                improved = False
                for i in range(1, n - 1):
                    for j in range(i + 1, n):
                        a, b, c, d = tour[i - 1], tour[i], tour[j], tour[(j + 1) % n]
                        if distance[a][c] + distance[b][d] < distance[a][b] + distance[c][d]:
                            tour[i:j + 1] = reversed(tour[i:j + 1])
                            improved = True
                            break
                    if improved:
                        break
        return [str(v) for v in tour[1:-1]]
    if method == "random":
        return [rng.choice(("A", "B")) for _ in range(n - 1)]
    neighbors = [[] for _ in range(n)]
    for u, v, w in state["edges"]:
        neighbors[u].append((v, w))
        neighbors[v].append((u, w))
    partition = ["A"]
    for v in range(1, n):
        gains = {side: sum(w for u, w in neighbors[v]
                           if u < v and partition[u] != side) for side in ("A", "B")}
        partition.append("B" if gains["B"] > gains["A"] else "A")
    if method == "greedy_single_flip":
        improved = True
        while improved:
            improved = False
            # Fixing 0 removes global complement symmetry, not possible flips:
            # a beneficial flip of 0 is represented by complementing every side.
            for v in range(n):
                gain = sum(w if partition[u] == partition[v] else -w
                           for u, w in neighbors[v])
                if gain > 0:
                    partition[v] = "B" if partition[v] == "A" else "A"
                    if v == 0:
                        partition = ["B" if side == "A" else "A" for side in partition]
                    improved = True
                    break
    return partition[1:]


# Preserve the original names for callers written before API standardization.
decision_budget = query_budget
baseline_decisions = baseline
