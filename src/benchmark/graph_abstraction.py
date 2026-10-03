"""Matched sequential graph A/B/C abstractions, using public state only.

API: prepare(instance), step(prepared, decisions, arm, seed),
selected_action(payload, option_id), baseline_choice(payload, method, rng),
rules(task). Histories and returned actions are original exact string IDs.
A exposes all legal actions and the existing request unchanged. B exposes
anonymous deduplicated proposals; C adds rule descriptions/mappings to the
same candidates, numeric features and option ordering. Only ``request`` is
model-facing: payload rules/proposals are audit data, including for B.

Every nonterminal step returns a payload, even a forced B/C singleton. Append
its forced_action to history and count that transition/coverage, but make no
model call. None means the ORIGINAL decision budget is complete. In particular,
partitions still have just two legal actions, not four: candidate coverage is
1/2 or 1. No vertex reordering, partition repair or rollout execution occurs.
The caller scores the resulting history with the original task scorer.

Four fixed rules maximize the explicitly documented features below. Ties are
A before B, or lowest numeric vertex ID. All scoring comparisons are exact.
Completion estimates are heuristics, never optimum/reference values. Each cut
step uses two linear greedy rollouts, O(n+m), not exponential search; LT uses
four proposals over all remaining seeds and exact deterministic diffusion.

Preparation validates public state, NOT private record/reference integrity.
The public task API requires a reference even to build a reference-independent
request. A fresh constant schema placeholder is used only at that API boundary;
the supplied private field and all metadata are never read, copied or retained.
Extended arm A retains its original instance-ID-based request_id, as required
by unchanged-request semantics. B/C request IDs contain no instance identifier.
"""

from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import random

from src.benchmark import extended_tasks, public_tasks
from src.clients.base import DecisionOption, DecisionRequest

_PARTITION_RULES = (
    "assigned_gain", "greedy_completion", "opposite_completion", "degree_balance",
)
_LT_RULES = ("marginal_cascade", "outdegree", "two_hop_reach", "threshold_deficit")
_TASKS = (
    "maxcut_construct", "lt_influence_construct",
    "community_bipartition_construct", "maxcut_public",
)
_PARTITION_FEATURES = (
    "assigned_score", "rollout_score", "opposite_suffix_score", "negative_volume_imbalance",
)
_LT_FEATURES = ("marginal_active", "outdegree", "two_hop_inactive", "threshold_relief")
_PARTITION_DEFINITIONS = (
    "P is the fixed assigned prefix (vertex 0 in A), v=len(P), s is candidate side, "
    "and U contains vertices greater than v. Unit weights apply to constructed graphs; "
    "public cut weights are signed integers. For cuts, volume is sum of absolute "
    "incident weights; for modularity it is full-graph degree. D_A,D_B are volumes "
    "of assigned vertices AFTER assigning v to s. negative_volume_imbalance=-abs(D_A-D_B). "
    "assigned_score is the incremental objective contribution against P, as defined below. "
    "rollout_score evaluates a full partition obtained by fixing P+[s], then assigning "
    "each vertex in U in increasing ID order to maximize assigned_score (ties A). "
    "opposite_suffix_score evaluates P+[s] with EVERY vertex in U on the side opposite s. "
    "These completions only estimate scores; executing an option assigns v alone. "
)
_CUT_DEFINITIONS = (
    "assigned_score=sum(w(u,v) for u in P with P[u]!=s). "
    "Full-partition score=sum(w(u,v) for edges with different endpoint sides), "
    "with signed weights unchanged. vertex_volume=sum(abs(w(u,v))). "
)
_COMMUNITY_DEFINITIONS = (
    "m is the FULL graph edge count, d is v's FULL degree, e_s is the number of edges "
    "from v to assigned vertices on s, and V_s is their full-degree sum BEFORE assignment. "
    "assigned_score=4*m*e_s-2*d*V_s-d*d, the increment of partial modularity scaled "
    "by 4*m*m. Full-partition score=4*m*(internal_A+internal_B)-D_A*D_A-D_B*D_B "
    "=4*m*m*Q, where Q=sum_C[internal_C/m-(D_C/(2*m))^2]. "
    "The positive scaling preserves standard modularity over AT MOST TWO communities; "
    "all A remains legal. vertex_volume=d. "
)
_LT_DEFINITIONS = (
    "S is the selected seed set, F(S) its deterministic normalized LT closure, X=F(S). "
    "Both final seeds start simultaneously: synchronous irreversible rounds activate "
    "an inactive t with positive indegree when 2*active_inneighbors(t)>=indegree(t); "
    "equality activates, unseeded indegree-zero vertices never activate. "
    "Every v not in S is legal, even if already in X. "
    "marginal_active=|F(S union {v})|-|X|. outdegree=|N_out(v)|. "
    "two_hop_inactive=|(N_out(v) union union_{u in N_out(v)} N_out(u)) minus (X union {v})|. "
    "For t not in X with positive indegree, deficit(t)=ceil(indegree(t)/2)-|N_in(t) intersect X|. "
    "threshold_relief=0 if v in X, otherwise sum(1/deficit(t) for t in N_out(v) minus X). "
    "It measures one active-neighbor contribution relative to each target's residual "
    "threshold, NOT an activation probability. threshold_relief is an exact rational "
    "{numerator,denominator}; comparisons use rational arithmetic. "
)


def rules(task: str) -> tuple[str, ...]:
    """Return the four fixed rule names in proposal order; reject unknown tasks."""
    if type(task) is not str or task not in _TASKS:
        raise ValueError("unsupported graph abstraction task")
    return _LT_RULES if task == "lt_influence_construct" else _PARTITION_RULES


@dataclass
class Prepared:
    instance: dict
    neighbors: list[dict[int, int]]
    volumes: list[int]


def _public_record(instance: dict) -> dict:
    # The existing request builder validates this field but never uses its value.
    return {**instance, "private": {"reference": {
        "value": 1, "status": "best_known", "source": "https://unused.invalid",
    }}}


def prepare(instance: dict) -> Prepared:
    """Copy only id/task/state; validate graph semantics without accessing private."""
    if type(instance) is not dict or not {"id", "task", "state"} <= instance.keys():
        raise ValueError("instance requires id, task and state")
    task = instance["task"]
    rules(task)
    if type(instance["id"]) is not str or not instance["id"].strip():
        raise ValueError("id must be a nonempty string")
    clean = {key: deepcopy(instance[key]) for key in ("id", "task", "state")}
    if task == "maxcut_public":
        public_tasks.validate_instance(_public_record(clean))
    else:
        extended_tasks._validate_state(task, clean["state"])
    neighbors = [{} for _ in clean["state"]["nodes"]]
    for edge in clean["state"]["edges"]:
        u, v = edge[:2]
        w = edge[2] if task == "maxcut_public" else 1
        neighbors[u][v] = w
        if task != "lt_influence_construct":
            neighbors[v][u] = w
    return Prepared(clean, neighbors, [sum(map(abs, row.values())) for row in neighbors])


def _assigned_score(prepared: Prepared, vertex: int, side: str,
                    prefix: list[str], volumes: dict[str, int]) -> int:
    if prepared.instance["task"] == "community_bipartition_construct":
        d = prepared.volumes[vertex]
        internal = sum(prefix[u] == side for u in prepared.neighbors[vertex] if u < vertex)
        return 4 * len(prepared.instance["state"]["edges"]) * internal - 2 * d * volumes[side] - d * d
    return sum(w for u, w in prepared.neighbors[vertex].items()
               if u < vertex and prefix[u] != side)


def _partition_score(prepared: Prepared, sides: list[str]) -> int:
    edges = prepared.instance["state"]["edges"]
    if prepared.instance["task"] == "community_bipartition_construct":
        internal = sum(sides[u] == sides[v] for u, v in edges)
        volumes = [sum(d for v, d in enumerate(prepared.volumes) if sides[v] == s)
                   for s in ("A", "B")]
        return 4 * len(edges) * internal - sum(d * d for d in volumes)
    return sum((edge[2] if len(edge) == 3 else 1)
               for edge in edges if sides[edge[0]] != sides[edge[1]])


def _partition_features(prepared: Prepared, decisions: list[str]) -> dict:
    prefix = ["A", *decisions]
    vertex, n = len(prefix), len(prepared.neighbors)
    volumes = {s: sum(prepared.volumes[u] for u in range(vertex) if prefix[u] == s)
               for s in ("A", "B")}
    features = {}
    for side in ("A", "B"):
        assigned = _assigned_score(prepared, vertex, side, prefix, volumes)
        loads = volumes.copy()
        loads[side] += prepared.volumes[vertex]
        imbalance = -abs(loads["A"] - loads["B"])
        rollout = [*prefix, side]
        for u in range(vertex + 1, n):
            chosen = min(("A", "B"), key=lambda s: (
                -_assigned_score(prepared, u, s, rollout, loads), s))
            rollout.append(chosen)
            loads[chosen] += prepared.volumes[u]
        opposite = "B" if side == "A" else "A"
        features[side] = dict(
            assigned_score=assigned, rollout_score=_partition_score(prepared, rollout),
            opposite_suffix_score=_partition_score(
                prepared, [*prefix, side, *([opposite] * (n - vertex - 1))]),
            negative_volume_imbalance=imbalance, vertex_volume=prepared.volumes[vertex],
        )
    return features


def _lt_features(prepared: Prepared, decisions: list[str]) -> dict:
    state = prepared.instance["state"]
    seeds = list(map(int, decisions))
    active = extended_tasks._lt_active(state, seeds)
    incoming = [set() for _ in state["nodes"]]
    for u, v in state["edges"]:
        incoming[v].add(u)
    deficit = {v: (len(row) + 1) // 2 - len(row & active)
               for v, row in enumerate(incoming) if row and v not in active}
    features = {}
    for v in state["nodes"]:
        if v in seeds:
            continue
        outgoing = set(prepared.neighbors[v])
        reach = outgoing | {w for u in outgoing for w in prepared.neighbors[u]}
        relief = (Fraction(0) if v in active else
                  sum((Fraction(1, deficit[t]) for t in outgoing - active), Fraction(0)))
        features[str(v)] = dict(
            marginal_active=len(extended_tasks._lt_active(state, [*seeds, v])) - len(active),
            outdegree=len(outgoing), two_hop_inactive=len(reach - active - {v}),
            threshold_relief={"numerator": relief.numerator, "denominator": relief.denominator},
        )
    return features


def _rule_info(task: str) -> dict:
    lt = task == "lt_influence_construct"
    return {
        "definitions": (_LT_DEFINITIONS if lt else _PARTITION_DEFINITIONS +
                        (_COMMUNITY_DEFINITIONS if task == "community_bipartition_construct"
                         else _CUT_DEFINITIONS)),
        "formulas": dict(zip(rules(task), (
            f"maximize {feature}" for feature in (_LT_FEATURES if lt else _PARTITION_FEATURES)))),
        "tie_break": "lowest numeric vertex ID" if lt else "A before B",
    }


def step(prepared: Prepared, decisions: list[str], arm: str, seed: int) -> dict | None:
    """Produce one scheduled transition; candidate singletons still return counts.

    ``candidates`` rows contain option_id/action/rules/features. In A their IDs
    are original action IDs, with all legal actions in original order; B/C use
    anonymous decimal IDs after a local seed/id/history shuffle. rule_proposals
    maps all four rules to original actions, independent of arm and shuffle.
    legal_action_count counts ALL original choices; distinct_candidate_count
    counts displayed deduplicated actions (all legal choices in A), and
    candidate_coverage is their ratio. forced_action is an original action
    exactly when that displayed set is a singleton, otherwise None. request is
    None for singletons, and a DecisionRequest otherwise. Append original
    actions, never anonymous option IDs, to decisions, including forced steps.
    """
    if type(arm) is not str or arm not in ("A", "B", "C") or type(seed) is not int:
        raise ValueError("arm must be A, B or C and seed must be an integer")
    instance = prepared.instance
    task = instance["task"]
    if task == "maxcut_public":
        request = public_tasks.next_request(_public_record(instance), decisions)
    else:
        request = extended_tasks.next_request(instance, decisions)
    if request is None:
        return None
    lt = task == "lt_influence_construct"
    features = _lt_features(prepared, decisions) if lt else _partition_features(prepared, decisions)
    legal = [option.id for option in request.options]

    def score(action, feature):
        value = features[action][feature]
        return Fraction(value["numerator"], value["denominator"]) if isinstance(value, dict) else value

    proposals = {
        rule: min(legal, key=lambda action: (
            -score(action, feature), int(action) if lt else action))
        for rule, feature in zip(rules(task), _LT_FEATURES if lt else _PARTITION_FEATURES)
    }
    actions = legal.copy() if arm == "A" else [a for a in legal if a in proposals.values()]
    if arm != "A":
        order_seed = hashlib.sha256(
            f"{seed}:{instance['id']}:{','.join(decisions)}".encode()).hexdigest()
        random.Random(order_seed).shuffle(actions)
    candidates = [{
        "option_id": action if arm == "A" else str(index), "action": action,
        "rules": [rule for rule in rules(task) if proposals[rule] == action],
        "features": features[action],
    } for index, action in enumerate(actions)]
    payload = {
        "candidates": candidates, "rule_proposals": proposals,
        "legal_action_count": len(legal), "distinct_candidate_count": len(actions),
        "candidate_coverage": len(actions) / len(legal),
        "forced_action": actions[0] if len(actions) == 1 else None,
        "request": None,
    }
    if len(actions) == 1:
        return payload
    if arm == "A":
        payload["request"] = request
        return payload
    state = deepcopy(request.state)
    state["candidates"] = [
        {key: deepcopy(value) for key, value in row.items() if key != "rules"}
        for row in candidates
    ]
    info = _rule_info(task)
    state["feature_definitions"] = info["definitions"]
    question = request.question.rsplit("Return only", 1)[0] + (
        "Only the displayed candidate actions are selectable. Execute exactly one "
        "original action; completion estimates do not execute future assignments. "
        "Return only the selected option_id, not the action."
    )
    if arm == "C":
        state["rules"] = info
        state["rule_groups"] = [
            {key: deepcopy(row[key]) for key in ("option_id", "action", "rules")}
            for row in candidates
        ]
        state["rule_proposals"] = proposals.copy()
        question += " Rule-equivalence groups name the rules proposing each candidate action."
    options = tuple(DecisionOption(
        row["option_id"], (f"{', '.join(row['rules'])}; " if arm == "C" else "") +
        (f"Select seed {row['action']}" if lt else f"Assign current vertex to {row['action']}"),
    ) for row in candidates)
    payload["request"] = DecisionRequest(
        request_id=f"graph_abstraction:{arm}:{len(decisions)}",
        state=state, question=question, options=options,
    )
    return payload


def selected_action(payload: dict, option_id: str) -> str:
    """Resolve an exact model option ID to an original action, with no repair."""
    if type(option_id) is not str or payload["forced_action"] is not None:
        raise ValueError("model selection requires an exact option ID and multiple candidates")
    matches = [row["action"] for row in payload["candidates"] if row["option_id"] == option_id]
    if len(matches) != 1:
        raise ValueError("illegal model option; no action repair")
    return matches[0]


def baseline_choice(payload: dict, method: str, rng: random.Random) -> str:
    """Select a fixed rule's proposal or a uniform distinct candidate, not a rule."""
    if type(method) is not str:
        raise ValueError("unknown graph abstraction baseline")
    if method in payload["rule_proposals"]:
        return payload["rule_proposals"][method]
    if method == "random_candidate":
        return rng.choice(payload["candidates"])["action"]
    raise ValueError("unknown graph abstraction baseline")
