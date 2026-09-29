"""Offline exact continuation analysis for recorded construction decisions.

These privileged diagnostics must never be included in model requests. They
reuse the continuation oracle from the historical construction-prefix analysis,
but accept incomplete legal histories without treating them as completed runs.
"""

from __future__ import annotations

from fractions import Fraction
from functools import lru_cache
from itertools import product
import math

from src.benchmark import extended_tasks as tasks

__all__ = ["continuation_values", "analyze_trajectory"]


def _validate(instance: dict, decisions: list[str]) -> None:
    tasks.validate_instances([instance])
    if instance["kind"] != "optimization":
        raise ValueError("continuation analysis requires a construction task")
    tasks._check_history(instance, decisions)


def _values(instance: dict, history: list[str]) -> dict[str, int | Fraction]:
    request = tasks.next_request(instance, history)
    if request is None:
        raise ValueError("continuation values require an unfinished history")
    task, state = instance["task"], instance["state"]
    if task == "tsp_construct":
        distances = tasks._distances(state)
        path = [0, *map(int, history)]
        remaining = tuple(v for v in state["nodes"] if v not in path)
        prefix_cost = sum(distances[u][v] for u, v in zip(path, path[1:]))

        @lru_cache(None)
        def finish(last: int, unvisited: tuple[int, ...]) -> int:
            if not unvisited:
                return distances[last][0]
            return min(distances[last][v] + finish(
                v, tuple(w for w in unvisited if w != v)) for v in unvisited)

        return {str(v): prefix_cost + distances[path[-1]][v] + finish(
            v, tuple(w for w in remaining if w != v)) for v in remaining}
    values = {}
    for option in request.options:
        if task == "lt_influence_construct":
            selected = [*map(int, history), int(option.id)]
            solutions = ([selected] if len(selected) == 2 else
                         [selected + [v] for v in state["nodes"] if v not in selected])
        else:
            length = instance["query_budget"] - len(history) - 1
            solutions = (["A", *history, option.id, *tail]
                         for tail in product(("A", "B"), repeat=length))
        values[option.id] = max(tasks._objective(task, state, solution) for solution in solutions)
    return values


def continuation_values(instance: dict, decisions: list[str]) -> dict[str, int | float]:
    """Return the best full objective reachable after each currently legal action.

    Validate the record and exact string IDs; refuse terminal or invalid histories.
    The TSP value includes prefix cost, the forced last city and the return edge.
    Values are JSON-safe; the internal modularity calculations use exact fractions.
    """
    _validate(instance, decisions)
    return {key: tasks._number(value) for key, value in _values(instance, decisions).items()}


def analyze_trajectory(instance: dict, decisions: list[str]) -> dict:
    """Decompose a legal recorded history into irreversible objective losses.

    For a complete history, step losses sum to its final additive objective gap.
    For an incomplete history, they sum only to the unavoidable gap of the prefix;
    final objective/gap remain None. Empty histories are incomplete, not successes.
    Invalid histories and corrupt records raise ValueError without repair.
    """
    _validate(instance, decisions)
    minimize = instance["task"] == "tsp_construct"
    best = min if minimize else max
    values = _values(instance, [])
    optimum = best(values.values())
    if not math.isclose(float(optimum), instance["private"]["objective"],
                        rel_tol=0, abs_tol=1e-12):
        raise ValueError("continuation oracle disagrees with the recorded optimum")
    previous = optimum
    steps = []
    losses = []
    for step, action in enumerate(decisions):
        if step:
            values = _values(instance, decisions[:step])
        before = best(values.values())
        if before != previous:
            raise ValueError("consecutive continuation values disagree")
        after = values[action]
        loss = after - before if minimize else before - after
        if loss < 0:
            raise ValueError("negative irreversible loss")
        steps.append({
            "step": step + 1,
            "action": action,
            "best_before": tasks._number(before),
            "best_after": tasks._number(after),
            "action_values": {key: tasks._number(value) for key, value in values.items()},
            "incremental_regret": tasks._number(loss),
            "locally_optimal": loss == 0,
            "global_optimum_reachable": after == optimum,
            "informative": len(set(values.values())) > 1,
        })
        losses.append(loss)
        previous = after
    unavoidable = previous - optimum if minimize else optimum - previous
    if sum(losses) != unavoidable:
        raise ValueError("step losses do not telescope to the prefix gap")
    scored = tasks.score(instance, decisions)
    if scored["feasible"] and not (
        math.isclose(float(previous), scored["objective"], rel_tol=0, abs_tol=1e-12)
        and math.isclose(float(unavoidable), scored["absolute_gap"], rel_tol=0, abs_tol=1e-12)
    ):
        raise ValueError("continuation loss disagrees with the scored final solution")
    return {
        "instance_id": instance["id"],
        "task": instance["task"],
        "node_count": len(instance["state"]["nodes"]),
        "status": scored["status"],
        "feasible": scored["feasible"],
        "decisions": list(decisions),
        "direction": scored["direction"],
        "optimum": tasks._number(optimum),
        "best_reachable_objective": tasks._number(previous),
        "unavoidable_gap": tasks._number(unavoidable),
        "objective": scored["objective"],
        "absolute_gap": scored["absolute_gap"],
        "first_loss_step": next(
            (row["step"] for row in steps if not row["locally_optimal"]), None),
        "global_optimum_reachable": previous == optimum,
        "steps": steps,
    }
