#!/usr/bin/env python3
"""Public RQ3: prepare, validate, freeze, run, report (no inference on freeze).

Main: four datasets, ten distinct graphs each, one deterministic condition,
fourteen historical model configurations, A/C only. ``run`` requires a separate
protocol-bound deployment approval. Existing episodes are never imported.

The public large-graph adapter reuses unchanged proposal rules and the durable
runner in an isolated module namespace. It replaces only prepare/score/load:
the small constructed-bank exponential validator is never invoked. The imported
shared modules and their globals are not modified.

Use run_benchmark.py --config configs/optimization.yaml --action ACTION for the
configured public workflow. Download-capable actions require download opt-in;
run requires inference opt-in as well as deployment approval. The supplied
configuration uses a supplementary heuristic modularity reference, not the
primary certified reference. Direct module and frozen-source CLIs remain
compatibility interfaces.

Raw public sources are local audit artifacts, NOT licensed for redistribution
by this repository. No department/community labels are downloaded or read.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter, deque
from copy import deepcopy
from dataclasses import asdict
from fractions import Fraction
import gzip
import hashlib
import importlib.util
from itertools import combinations
import json
import math
from pathlib import Path
import random
import re
import statistics
import sys
import time

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.benchmark import extended_tasks, graph_abstraction, public_data, public_tasks
from src.utils import graph_abstraction_suite as shared

DEFAULT_ROOT = REPO / "output/experiments/public/primary/rq3"
COHORT = REPO / "output/reference-inputs/model-panel"
PROTOCOL = "GraphDecisionBench-public-optimization"
ARMS = ("A", "C")
TASKS = ("tsp_public", "maxcut_public", "lt_influence_construct",
         "community_bipartition_construct")
SIZES = (50, 75, 100, 125, 150)
SEED = 20261003
TSPLIB = "https://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/"
MIRROR = ("https://raw.githubusercontent.com/mdenhoedt/MaxCut-Instances/"
          "9c9c1a84774ac14b7c161d5d5ea1f4f554a39a76/set2/")
SNAP = {
    "email-Eu-core": ("https://snap.stanford.edu/data/email-Eu-core.txt.gz", True, 1005),
    "ca-GrQc": ("https://snap.stanford.edu/data/ca-GrQc.txt.gz", False, 5242),
}
require, digest, read_json = shared.require, shared.digest, shared.read_json
write_json = shared.atomic_json


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def public_view(instance):
    return {**{k: instance[k] for k in ("id", "task", "state")},
            "private": {"reference": instance["private"]["reference"]}}


def budget(instance):
    n = len(instance["state"]["nodes"])
    return 2 if instance["task"] == TASKS[2] else n - (2 if instance["task"] == TASKS[0] else 1)


def validate_state(instance):
    task, state = instance["task"], instance["state"]
    require(task in TASKS, "Unsupported public RQ3 task.")
    if task.endswith("_public"):
        public_tasks.validate_instance(public_view(instance))
        return
    keys = {"nodes", "edges", "directed"}
    lt = task == "lt_influence_construct"
    if lt:
        keys |= {"weights", "threshold", "seed_count"}
    require(type(state) is dict and set(state) == keys, "Invalid public state keys.")
    n = len(state["nodes"])
    require(2 <= n <= 254 and state["nodes"] == list(range(n)) and
            all(type(v) is int for v in state["nodes"]), "Invalid public node IDs.")
    require(type(state["directed"]) is bool and state["directed"] == lt,
            "Direction must be preserved.")
    edges = state["edges"]
    require(type(edges) is list and all(type(e) is list and len(e) == 2 and
            all(type(v) is int and 0 <= v < n for v in e) and e[0] != e[1] and
            (lt or e[0] < e[1]) for e in edges), "Invalid simple edges.")
    require(edges == sorted(edges) and len({tuple(e) for e in edges}) == len(edges),
            "Edges must be sorted, unique, and unmodified.")
    if lt:
        indegree = Counter(v for _, v in edges)
        require(state["weights"] == [
            {"source": u, "target": v, "numerator": 1, "denominator": indegree[v]}
            for u, v in edges] and state["threshold"] == {"numerator": 1, "denominator": 2}
            and type(state["seed_count"]) is int and state["seed_count"] == 2,
            "LT must use normalized indegree weights, half threshold, two seeds.")
    else:
        require(bool(edges), "Standard modularity is undefined on an edgeless graph.")


def prepare_instance(instance):
    validate_state(instance)
    if instance["task"].endswith("_public"):
        return shared.prepare(public_view(instance))
    clean = {k: deepcopy(instance[k]) for k in ("id", "task", "state")}
    neighbors = [{} for _ in clean["state"]["nodes"]]
    for u, v in clean["state"]["edges"]:
        neighbors[u][v] = 1
        if not clean["state"]["directed"]:
            neighbors[v][u] = 1
    return graph_abstraction.Prepared(clean, neighbors, [len(row) for row in neighbors])


def lt_structures(state):
    outgoing = [[] for _ in state["nodes"]]
    incoming = [0] * len(outgoing)
    degree = [0] * len(outgoing)
    for u, v in state["edges"]:
        outgoing[u].append(v)
        incoming[v] |= 1 << u
        degree[v] += 1
    return outgoing, incoming, degree


def lt_queue(structures, seeds):
    """Monotone event queue; closure equals synchronous irreversible diffusion."""
    outgoing, _, degree = structures
    active = set(seeds)
    counts = [0] * len(outgoing)
    pending = deque(seeds)
    while pending:
        for v in outgoing[pending.popleft()]:
            counts[v] += 1
            if v not in active and 2 * counts[v] >= degree[v]:
                active.add(v)
                pending.append(v)
    return active


def lt_synchronous(structures, seeds):
    """Independent synchronous bitset implementation, checking every seed pair."""
    _, incoming, degree = structures
    active = sum(1 << v for v in seeds)
    while True:
        updated = active
        for v, mask in enumerate(incoming):
            if degree[v] and 2 * (mask & active).bit_count() >= degree[v]:
                updated |= 1 << v
        if updated == active:
            return {v for v in range(len(incoming)) if active & (1 << v)}
        active = updated


def lt_reference(state):
    structures = lt_structures(state)
    best, witness, checked = -1, None, 0
    for pair in combinations(state["nodes"], 2):
        fast = lt_queue(structures, pair)
        require(fast == lt_synchronous(structures, pair), "Independent LT closure mismatch.")
        checked += 1
        if len(fast) > best:
            best, witness = len(fast), list(pair)
    return {"value": best, "status": "proven_optimum", "lower_bound": best,
            "upper_bound": best, "solution": witness, "pairs_checked": checked,
            "method": "all unordered two-seed pairs; event queue + synchronous bitset",
            "source": "independent_exact_enumeration"}


def modularity(state, sides):
    m = len(state["edges"])
    cut = sum(sides[u] != sides[v] for u, v in state["edges"])
    volume = sum((sides[u] == "A") + (sides[v] == "A") for u, v in state["edges"])
    return Fraction(volume * (2 * m - volume) - 2 * m * cut, 2 * m * m)


def modularity_reference(state):
    """Deterministic feasible local search, NOT a certifying optimization solver."""
    import networkx as nx

    n, m = len(state["nodes"]), len(state["edges"])
    graph = nx.Graph()
    graph.add_nodes_from(state["nodes"])
    graph.add_edges_from(state["edges"])
    # A one-community start and a fixed balanced start, not data-tuned restarts.
    starts = [["A"] * n, ["A" if v < (n + 1) // 2 else "B" for v in range(n)]]
    communities = sorted(nx.community.greedy_modularity_communities(graph),
                         key=lambda group: (-len(group), min(group)))
    merged, volumes = ["A"] * n, {"A": 0, "B": 0}
    for group in communities:
        side = min(volumes, key=lambda s: (volumes[s], s))
        for v in group:
            merged[v] = side
        volumes[side] += sum(graph.degree(v) for v in group)
    starts.append(merged)
    candidates = []
    for sides in starts:
        if sides[0] != "A":
            sides = ["B" if s == "A" else "A" for s in sides]
        current = modularity(state, sides)
        while True:
            changed = False
            for v in range(1, n):
                sides[v] = "B" if sides[v] == "A" else "A"
                value = modularity(state, sides)
                if value > current:
                    current, changed = value, True
                else:
                    sides[v] = "B" if sides[v] == "A" else "A"
            if not changed:
                break
        groups = [{v for v in range(n) if sides[v] == s} for s in ("A", "B")]
        independent = nx.community.modularity(graph, [g for g in groups if g], weight=None)
        require(math.isclose(float(current), independent, abs_tol=1e-12),
                "Independent NetworkX modularity mismatch.")
        candidates.append((current, sides.copy()))
    value, witness = max(candidates, key=lambda pair: pair[0])
    require(0 <= value <= Fraction(1, 2) and m > 0, "Invalid modularity reference.")
    return {
        "value": float(value), "value_fraction": [value.numerator, value.denominator],
        "status": "feasible_best_reference", "lower_bound": float(value), "upper_bound": 0.5,
        "solution": witness, "method": "best of three fixed starts + strict single-vertex ascent",
        "starts": ["all_A", "fixed_half", "degree-balanced merge of greedy communities"],
        "bound_proof": "Q=2*a*(1-a)-cut/m <= 1/2 for at most two groups; all A gives Q=0.",
        "source": "independently_checked_feasible_search_not_optimum",
    }


def score(instance, decisions):
    task = instance["task"]
    if task.endswith("_public"):
        result = shared.score(public_view(instance), decisions)
        result["reference_status"] = instance["private"]["reference"]["status"]
        return result
    status = "complete"
    try:
        extended_tasks._check_history(instance, decisions)
    except ValueError:
        status = "invalid"
    else:
        if len(decisions) != budget(instance):
            status = "incomplete"
    feasible = status == "complete"
    reference = instance["private"]["reference"]
    objective = None
    if feasible:
        if task == "lt_influence_construct":
            seeds = list(map(int, decisions))
            structures = lt_structures(instance["state"])
            active = lt_queue(structures, seeds)
            require(active == lt_synchronous(structures, seeds), "LT score cross-check failed.")
            objective = len(active)
        else:
            objective = float(modularity(instance["state"], ["A", *decisions]))
            require(math.isclose(objective, shared.independent_objective(instance, decisions),
                                 abs_tol=1e-12), "Independent modularity score mismatch.")
    signed_gap = reference["value"] - objective if feasible else None
    exact = reference["status"] == "proven_optimum"
    return {
        "feasible": feasible, "status": status, "objective": objective,
        "direction": "maximize", "reference": reference["value"],
        "reference_status": reference["status"],
        "optimum": reference["value"] if exact else None,
        "signed_reference_gap": signed_gap,
        "absolute_gap": abs(signed_gap) if feasible and exact else None,
        "optimality_gap_upper_bound": max(0, reference["upper_bound"] - objective) if feasible else None,
        "optimal": bool(feasible and exact and signed_gap == 0),
        "normalized_quality": 1 / (1 + abs(signed_gap)) if feasible and exact else None,
    }


def download(root, name, url, expected=None):
    directory = root / "sources"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    if path.exists():
        data = path.read_bytes()
    else:
        data = public_data.fetch_public_bytes(url)
        with path.open("xb") as stream:
            stream.write(data)
    sha = hashlib.sha256(data).hexdigest()
    require(expected is None or sha == expected, f"Source checksum mismatch: {name}")
    return data, {"file": str(path.relative_to(root)), "url": url, "sha256": sha}


def snap_graph(raw, directed, expected_nodes):
    nodes, edges, loops, duplicates = set(), set(), 0, 0
    for line in raw.decode("ascii").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split()
        require(len(parts) == 2, "Invalid SNAP edge row.")
        u, v = map(int, parts)
        nodes.update((u, v))
        if u == v:
            loops += 1
            continue
        edge = (u, v) if directed else tuple(sorted((u, v)))
        if edge in edges:
            duplicates += 1
        edges.add(edge)
    require(len(nodes) == expected_nodes, "SNAP node count mismatch.")
    return sorted(nodes), sorted(edges), {
        "self_loop_rows_removed": loops, "duplicate_or_reciprocal_rows_collapsed": duplicates,
        "normalized_edges": len(edges), "source_nodes": len(nodes),
        "policy": "remove self loops; collapse duplicate arcs; for ca-GrQc only collapse reciprocal rows",
    }


def sample_nodes(nodes, edges, n, key):
    """Weak-neighbor expansion with hash-priority frontier; no labels/objectives."""
    ranks = {v: hashlib.sha256(f"{PROTOCOL}:{key}:{v}".encode()).hexdigest() for v in nodes}
    adjacency = {v: set() for v in nodes}
    for u, v in edges:
        adjacency[u].add(v)
        adjacency[v].add(u)
    selected, frontier = set(), set()
    while len(selected) < n:
        pool = frontier - selected
        if not pool:
            pool = set(nodes) - selected
        v = min(pool, key=lambda u: (ranks[u], u))
        selected.add(v)
        frontier.discard(v)
        frontier.update(adjacency[v] - selected)
    return sorted(selected, key=lambda v: (ranks[v], v))


def induced_record(dataset, nodes, edges, n, replicate, *, pilot=False):
    directed = SNAP[dataset][1]
    key = f"{dataset}:{'pilot' if pilot else 'main'}:{n}:{replicate}"
    original = sample_nodes(nodes, edges, n, key)
    mapping = {v: i for i, v in enumerate(original)}
    sampled = sorted([mapping[u], mapping[v]] if directed else sorted((mapping[u], mapping[v]))
                     for u, v in edges if u in mapping and v in mapping)
    state = {"nodes": list(range(n)), "edges": sampled, "directed": directed}
    if directed:
        indegree = Counter(v for _, v in sampled)
        state.update(weights=[{"source": u, "target": v, "numerator": 1,
                               "denominator": indegree[v]} for u, v in sampled],
                     threshold={"numerator": 1, "denominator": 2}, seed_count=2)
    name = f"{dataset}-{'pilot' if pilot else 'main'}-n{n}-s{replicate}"
    instance = {
        "id": name, "task": TASKS[2 if directed else 3], "dataset_id": name,
        "source_dataset": dataset, "replicate": 0, "state": state,
        "query_budget": 2 if directed else n - 1,
        "metadata": {"role": "pilot_validation" if pilot else "main",
                     "sample_key": key, "original_node_ids_in_local_order": original,
                     "sampler": "fixed hash-priority weak-neighbor expansion; induced edges only",
                     "labels_used": False, "added_edges": 0},
    }
    validate_state(instance)
    instance["private"] = {"reference": lt_reference(state) if directed else modularity_reference(state)}
    return instance


def public_record(name, task, state, reference, source, role="main"):
    return {"id": name, "dataset_id": name, "task": task, "state": state,
            "source_dataset": "TSPLIB" if task == "tsp_public" else "OPTSICOM-Set2",
            "replicate": 0, "private": {"reference": reference},
            "metadata": {"role": role, "source": source, "complete_original_graph": True,
                         "original_node_ids_in_local_order": list(range(1, len(state["nodes"]) + 1)),
                         "condition": "native order; fixed city/vertex 0; no relabel repeats"}}


def prepare_data(root):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    require(not (root / "instances.json").exists(), "Prepared data already exists; no overwrite.")
    catalog = read_json(REPO / "data/optimization-catalog.json")
    instances, sources = [], []
    for row in catalog["instances"]:
        if row["role"] != "main":
            continue
        wire, provenance = download(root, row["id"] + ".wire", row["download_url"], row["wire_sha256"])
        raw = gzip.decompress(wire) if row["compression"] == "gzip" else wire
        require(hashlib.sha256(raw).hexdigest() == row["raw_sha256"], "Raw catalog checksum drift.")
        state = (public_data.parse_tsplib_euc_2d(raw, row["dimension"])
                 if row["task"] == "tsp_public" else
                 public_data.parse_maxcut(raw, row["dimension"], row["edge_count"]))
        if row["task"] == "maxcut_public":
            require(all(e[2] in (-1, 1) for e in state["edges"]), "Signed +/-1 source changed.")
        sources.append({**provenance, "raw_sha256": row["raw_sha256"], "catalog_entry": row})
        instances.append(public_record(row["id"], row["task"], state, row["reference"], provenance))
    optimum_html, provenance = download(root, "TSPLIB-optima.html", TSPLIB + "TSP-BEST.html")
    sources.append(provenance)
    pilots = []
    for name, n, value, pilot in (("eil76", 76, 538, False), ("rat99", 99, 1211, False),
                                   ("kroD100", 100, 21294, False), ("berlin52", 52, 7542, True)):
        text = re.sub("<[^>]+>", " ", optimum_html.decode("latin1"))
        require(re.search(rf"\b{name}\s*:\s*{value}\b", text), "Official TSPLIB optimum not verified.")
        wire, provenance = download(root, name + ".tsp.gz", TSPLIB + name + ".tsp.gz")
        raw = gzip.decompress(wire)
        state = public_data.parse_tsplib_euc_2d(raw, n)
        sources.append({**provenance, "raw_sha256": hashlib.sha256(raw).hexdigest(),
                        "dimension": n, "edge_weight_type": "EUC_2D"})
        record = public_record(name, "tsp_public", state,
                               {"value": value, "status": "proven_optimum",
                                "source": TSPLIB + "TSP-BEST.html"}, provenance,
                               "pilot_validation" if pilot else "main")
        (pilots if pilot else instances).append(record)
    for dataset, (url, directed, expected_nodes) in SNAP.items():
        wire, provenance = download(root, dataset + ".txt.gz", url)
        raw = gzip.decompress(wire)
        nodes, edges, normalization = snap_graph(raw, directed, expected_nodes)
        sources.append({**provenance, "raw_sha256": hashlib.sha256(raw).hexdigest(),
                        "dataset_id": dataset, "normalization": normalization})
        for n in SIZES:
            for replicate in range(2):
                instance = induced_record(dataset, nodes, edges, n, replicate)
                instances.append(instance)
                print(json.dumps({"prepared": instance["id"],
                                  "reference": instance["private"]["reference"]["status"]}), flush=True)
        pilots.append(induced_record(dataset, nodes, edges, 50, 0, pilot=True))
    # Set2 has exactly ten n=125 instances, all reserved for main. The public
    # MaxCut pilot is an explicit induced sample of a different n=1000 member.
    wire, provenance = download(root, "sg3dl101000.mc", MIRROR + "instances/sg3dl101000.mc")
    lines = [line.split() for line in wire.decode("ascii").splitlines() if line.strip()]
    n, m = map(int, lines[0])
    require(n == 1000 and len(lines) == m + 1, "Unexpected MaxCut pilot source dimensions.")
    weighted = [list(map(int, line)) for line in lines[1:]]
    require(all(len(e) == 3 and e[2] in (-1, 1) and e[0] != e[1] for e in weighted),
            "Invalid MaxCut pilot source.")
    original = sample_nodes(list(range(1, n + 1)), [(u, v) for u, v, _ in weighted],
                            50, "OPTSICOM-Set2:pilot:50")
    mapping = {v: i for i, v in enumerate(original)}
    state = {"nodes": list(range(50)), "edges": sorted(
        [*sorted((mapping[u], mapping[v])), w] for u, v, w in weighted if u in mapping and v in mapping)}
    pilot = public_record("sg3dl101000-induced50-pilot", "maxcut_public", state,
                          {"value": 1, "status": "best_known", "source": provenance["url"]},
                          provenance, "pilot_validation")
    decisions = public_tasks.baseline(public_view(pilot), "greedy_single_flip", seed=SEED)
    value = shared.independent_objective(public_view(pilot), decisions)
    require(value > 0, "Nonpositive MaxCut pilot feasible reference.")
    pilot["private"]["reference"]["value"] = value
    pilot["private"]["reference_caveat"] = "feasible pilot baseline, not source BKS or optimum"
    pilot["private"]["witness"] = ["A", *decisions]
    pilot["metadata"].update(complete_original_graph=False,
                             original_node_ids_in_local_order=original,
                             normalization="explicit induced sample; signed weights unchanged")
    sources.append(provenance)
    pilots.append(pilot)
    for instance in [*instances, *pilots]:
        validate_state(instance)
        instance["state_sha256"] = canonical_hash(instance["state"])
    manifest = {
        "version": PROTOCOL, "created_at": shared.public.now(), "sources": sources,
        "license": {"TSPLIB": "No explicit redistribution license verified; source attribution only.",
                    "OPTSICOM-Set2": catalog["license_note"],
                    "SNAP": "Public research datasets; no explicit redistribution license verified on source pages."},
        "source_pages": ["https://snap.stanford.edu/data/email-Eu-core.html",
                         "https://snap.stanford.edu/data/ca-GrQc.html"],
        "normalization": "No edge editing except explicit loop/dedup normalization and induced subgraph restriction.",
        "sampling": "Frozen label-blind hash-priority weak-neighbor expansion; no performance-based filtering.",
        "pilot": "Four separate public validation graphs, never main performance observations.",
        "maxcut_reference_provenance": catalog["maxcut_reference_provenance"],
        "historical_reuse": {
            "episode_reuse": 0, "classification": "new_protocol_not_historical_replication",
            "reason": "A/C only, one native condition, 40-graph public cohort, independent large-graph references; "
                      "old successes/failures remain unchanged and are not rerun as historical cells.",
            "reused": "Only checksum-verified original data and source-reported references; no inference records.",
        },
    }
    write_json(root / "instances.json", instances)
    write_json(root / "pilot-instances.json", pilots)
    write_json(root / "source-manifest.json", manifest)
    return instances, pilots


def controls(instances):
    rows = []
    for instance in instances:
        prepared = prepare_instance(instance)
        for method in (*shared.rules(instance), "random_candidate", "classical"):
            seeds = shared.RANDOM_SEEDS if method == "random_candidate" else (SEED,)
            for seed in seeds:
                decisions, trace = [], []
                started = time.perf_counter()
                if method == "classical":
                    if instance["task"].endswith("_public"):
                        classical = ("nearest_neighbor_two_opt" if instance["task"] == "tsp_public"
                                     else "greedy_single_flip")
                        decisions = public_tasks.baseline(public_view(instance), classical, seed=seed)
                    else:
                        classical = ("marginal-greedy" if instance["task"] == TASKS[2]
                                     else "single-sweep-greedy-modularity")
                        solution = extended_tasks._baseline(instance["task"], instance["state"])
                        decisions = ([str(v) for v in solution] if instance["task"] == TASKS[2]
                                     else solution[1:])
                else:
                    classical = None
                    rng = random.Random(seed)
                    while (payload := shared.step(prepared, decisions, "C")) is not None:
                        action = (rng.choice(payload["candidates"])["action"]
                                  if method == "random_candidate" else payload["rule_proposals"][method])
                        trace.append({"selected_action": action,
                                      "proposal_sha256": canonical_hash(
                                          {k: v for k, v in payload.items() if k != "request"}),
                                      "candidate_count": payload["distinct_candidate_count"],
                                      "legal_count": payload["legal_action_count"],
                                      "forced": payload["forced_action"] is not None})
                        decisions.append(action)
                result = score(instance, decisions)
                require(result["feasible"], "Control did not complete.")
                rows.append({"instance_id": instance["id"], "state_sha256": canonical_hash(instance["state"]),
                             "method": method, "classical_method": classical, "seed": seed,
                             "decisions": decisions, "trace": trace,
                             "wall_seconds": time.perf_counter() - started, **result})
        print(json.dumps({"control_graph": instance["id"], "rows": len(rows)}), flush=True)
    return rows


def acquire_reference_evidence(root):
    root = Path(root).resolve()
    require(not (root / "protocol.json").exists(), "Reference evidence must precede freezing.")
    require(not (root / "reference-evidence.json").exists(), "Reference evidence already captured.")
    instances = read_json(root / "instances.json")
    sources, witnesses = [], {}
    for instance in instances:
        if instance["task"] != "maxcut_public":
            continue
        name = instance["dataset_id"]
        raw, source = download(root, name + ".bksol", MIRROR + "bksol/" + name + ".bksol")
        sources.append(source)
        bits = raw.decode("ascii").strip()
        require(len(bits) == len(instance["state"]["nodes"]) and set(bits) <= {"0", "1"},
                "Malformed historical BKS witness.")
        value_raw, source = download(root, name + ".bkvl", MIRROR + "bkvl/" + name + ".bkvl")
        sources.append(source)
        value = int(value_raw.decode("ascii").strip())
        sides = ["A" if bit == bits[0] else "B" for bit in bits]
        evaluated = sum(w for u, v, w in instance["state"]["edges"] if sides[u] != sides[v])
        require(value == evaluated == instance["private"]["reference"]["value"],
                "Historical source BKS witness/value mismatch.")
        witnesses[name] = {"solution": sides, "objective": evaluated, "source_reported_value": value,
                           "state_sha256": instance["state_sha256"]}
    readme, source = download(root, "MaxCut-mirror-README.md", MIRROR.split("/set2/")[0] + "/README.md")
    sources.append(source)
    require("best known" in readme.decode("utf8").lower(), "Missing historical BKS provenance statement.")
    for dataset in SNAP:
        _, source = download(root, dataset + ".html", f"https://snap.stanford.edu/data/{dataset}.html")
        sources.append(source)
    write_json(root / "reference-evidence.json", {"sources": sources, "maxcut_witnesses": witnesses,
                                                "historical_reference_not_current_global_BKS": True})


def validate_reference(instance):
    validate_state(instance)
    require(instance["state_sha256"] == canonical_hash(instance["state"]), "State seal mismatch.")
    task, reference = instance["task"], instance["private"]["reference"]
    if task == TASKS[2]:
        require(reference == lt_reference(instance["state"]), "LT exhaustive reference drift.")
    elif task == TASKS[3]:
        require(reference == modularity_reference(instance["state"]), "Feasible modularity reference drift.")
        require(reference["status"] == "feasible_best_reference" and reference["upper_bound"] == 0.5,
                "Modularity must not masquerade as an optimum.")
    if task in TASKS[2:]:
        solution = reference["solution"]
        decisions = list(map(str, solution)) if task == TASKS[2] else solution[1:]
        require(score(instance, decisions)["objective"] == reference["value"], "Reference witness mismatch.")


def validate_sources(root, instances, pilots):
    manifest = read_json(root / "source-manifest.json")
    evidence = read_json(root / "reference-evidence.json")
    for source in [*manifest["sources"], *evidence["sources"]]:
        require(digest(root / source["file"]) == source["sha256"], "Raw source drift.")
        if "catalog_entry" in source:
            entry = source["catalog_entry"]
            wire = (root / source["file"]).read_bytes()
            raw = gzip.decompress(wire) if entry["compression"] == "gzip" else wire
            require(hashlib.sha256(raw).hexdigest() == entry["raw_sha256"], "Catalog raw drift.")
            expected = (public_data.parse_tsplib_euc_2d(raw, entry["dimension"])
                        if entry["task"] == "tsp_public" else
                        public_data.parse_maxcut(raw, entry["dimension"], entry["edge_count"]))
            record = next(i for i in instances if i["dataset_id"] == entry["id"])
            require(record["state"] == expected and record["private"]["reference"] == entry["reference"],
                    "Original full graph/reference changed.")
        elif source.get("edge_weight_type") == "EUC_2D":
            raw = gzip.decompress((root / source["file"]).read_bytes())
            name = Path(source["file"]).name.removesuffix(".tsp.gz")
            record = next(i for i in [*instances, *pilots] if i["dataset_id"] == name)
            require(record["state"] == public_data.parse_tsplib_euc_2d(raw, source["dimension"]),
                    "New native TSPLIB graph drift.")
        elif source.get("dataset_id") in SNAP:
            dataset = source["dataset_id"]
            raw = gzip.decompress((root / source["file"]).read_bytes())
            nodes, edges, normalization = snap_graph(raw, SNAP[dataset][1], SNAP[dataset][2])
            require(source["normalization"] == normalization, "Normalization counts drift.")
            for record in [i for i in [*instances, *pilots] if i["source_dataset"] == dataset]:
                metadata = record["metadata"]
                selected = sample_nodes(nodes, edges, len(record["state"]["nodes"]), metadata["sample_key"])
                require(selected == metadata["original_node_ids_in_local_order"], "Sampler/ID mapping drift.")
                index = {v: i for i, v in enumerate(selected)}
                expected = sorted([index[u], index[v]] if SNAP[dataset][1] else sorted((index[u], index[v]))
                                  for u, v in edges if u in index and v in index)
                require(record["state"]["edges"] == expected, "Induced topology was edited.")
    optimum_html = (root / "sources/TSPLIB-optima.html").read_bytes().decode("latin1")
    optimum_text = re.sub("<[^>]+>", " ", optimum_html)
    for instance in [*instances, *pilots]:
        if instance["task"] == "tsp_public":
            name, value = instance["dataset_id"], instance["private"]["reference"]["value"]
            require(re.search(rf"\b{re.escape(name)}\s*:\s*{value}\b", optimum_text),
                    "TSPLIB reference differs from captured official optimum.")
    require(set(evidence["maxcut_witnesses"]) ==
            {i["dataset_id"] for i in instances if i["task"] == "maxcut_public"},
            "Missing main MaxCut BKS evidence.")
    for instance in (i for i in instances if i["task"] == "maxcut_public"):
        witness = evidence["maxcut_witnesses"][instance["dataset_id"]]
        sides = witness["solution"]
        require(len(sides) == 125 and sides[0] == "A" and set(sides) <= {"A", "B"},
                "Invalid MaxCut BKS partition.")
        require(witness["state_sha256"] == instance["state_sha256"] and
                witness["objective"] == witness["source_reported_value"] ==
                instance["private"]["reference"]["value"] ==
                sum(w for u, v, w in instance["state"]["edges"] if sides[u] != sides[v]),
                "Independent signed MaxCut BKS replay mismatch.")
    return manifest


def validate_data(root, *, check_controls=True):
    root = Path(root).resolve()
    instances, pilots = read_json(root / "instances.json"), read_json(root / "pilot-instances.json")
    require(Counter(i["task"] for i in instances) == Counter({task: 10 for task in TASKS}),
            "Expected four tasks, ten graphs per task.")
    require(Counter(i["task"] for i in pilots) == Counter({task: 1 for task in TASKS}),
            "Expected four separate public pilot-validation graphs.")
    all_instances = [*instances, *pilots]
    require(len({i["id"] for i in all_instances}) == 44, "Duplicate graph IDs.")
    require(len({i["state_sha256"] for i in all_instances}) == 44, "Repeated graph states.")
    require(all("eil51" not in i["id"] for i in instances), "Calibration eil51 in main.")
    for task, dataset in zip(TASKS, ("TSPLIB", "OPTSICOM-Set2", "email-Eu-core", "ca-GrQc")):
        require({i["source_dataset"] for i in instances if i["task"] == task} == {dataset},
                "Expected one source dataset per task.")
    for task in TASKS[2:]:
        own = [i for i in instances if i["task"] == task]
        require(Counter(len(i["state"]["nodes"]) for i in own) == Counter({n: 2 for n in SIZES}),
                "SNAP size strata drift.")
        require(len({tuple(sorted(i["metadata"]["original_node_ids_in_local_order"])) for i in own}) == 10,
                "Duplicate original-node samples.")
    manifest = validate_sources(root, instances, pilots)
    for instance in all_instances:
        validate_reference(instance)
        require(instance["replicate"] == 0, "Exactly one condition per graph is required.")
        prepared = prepare_instance(instance)
        for arm in ARMS:
            payload = shared.step(prepared, [], arm)
            require(payload is not None and len(payload["rule_proposals"]) == 4,
                    "Four fixed action rules required.")
            if payload["request"] is not None:
                public = asdict(payload["request"])
                require("private" not in public["state"] and "reference" not in public["state"],
                        "Private reference leaked to request.")
    control_checks = 0
    if check_controls:
        for filename, own in (("controls.json", instances), ("pilot-controls.json", pilots)):
            stored = read_json(root / filename)
            expected = controls(own)
            require(len(stored) == len(expected) == len(own) * 10, "Incomplete controls.")
            for a, b in zip(stored, expected):
                require({k: v for k, v in a.items() if k != "wall_seconds"} ==
                        {k: v for k, v in b.items() if k != "wall_seconds"},
                        "Independent control replay differs.")
                control_checks += 1
    return {
        "version": PROTOCOL, "main_graphs": 40, "pilot_graphs": 4,
        "source_files_checked": len(manifest["sources"]) +
                                len(read_json(root / "reference-evidence.json")["sources"]),
        "control_rows_replayed": control_checks,
        "exact_lt_pairs_checked": sum(i["private"]["reference"]["pairs_checked"] for i in all_instances
                                     if i["task"] == TASKS[2]),
        "modularity_reference_status": "feasible_best_reference_with_analytic_upper_bound",
        "modularity_optimum_claimed": False, "offline_only": True,
        "validated_at": shared.public.now(),
    }


def model_specs(root, cohort=COHORT):
    cohort = Path(cohort).resolve()
    specs, provenance = {}, {}
    protocol = read_json(cohort / "protocol.json")
    require(digest(cohort / "protocol.json") == read_json(cohort / "protocol.sha256") and
            protocol["models"] == list(shared.PANEL), "Historical cohort identity drift.")
    provenance["protocol_sha256"] = digest(cohort / "protocol.json")
    for name in shared.PANEL:
        path = cohort / "runs" / name / "config.json"
        require(digest(path) == read_json(path.parent / "config.sha256"), "Historical model seal drift.")
        original = read_json(path)
        config = shared.validate_lane_config(name, original)
        require(config.timeout_seconds == (30 if name == "jev_action" else 180),
                "Historical request timeout differs from authorized scope.")
        spec = deepcopy(original)
        if shared.native_config(config):
            spec["adapter_kwargs"]["audit_path"] = str(root / "native-audit" / f"{name}.jsonl")
        specs[name] = spec
        provenance[name] = {"config_sha256": digest(path), "original_config": original,
                            "changes": "native audit_path only" if spec != original else "none"}
    return specs, provenance


def source_hashes():
    utils = ("__init__.py", "public_optimization.py", "graph_abstraction_suite.py",
             "public_graph_suite.py", "extended_graph_suite.py", "paired_graph_ablation.py")
    paths = [REPO / "src/__init__.py", *(REPO / "src/benchmark").rglob("*.py"),
             *(REPO / "src/clients").rglob("*.py"),
             *(REPO / "src/datasets").rglob("*.py"),
             *(REPO / "src/utils" / name for name in utils),
             REPO / "tests/integration/test_public_optimization.py"]
    return {str(p.relative_to(REPO)): digest(p) for p in sorted(set(paths))}


def freeze(root, tests_log, cohort=COHORT):
    root = Path(root).resolve()
    require(not (root / "protocol.json").exists(), "Already frozen; never overwrite old protocol.")
    tests = Path(tests_log).read_text()
    require(re.search(r"Ran \d+ tests in .*?\n\nOK\s*$", tests), "Successful persisted unittest log required.")
    validation = validate_data(root)
    specs, cohort_provenance = model_specs(root, cohort)
    instances = read_json(root / "instances.json")
    jobs = [{"instance_id": i["id"], "arm": arm} for i in instances for arm in ARMS]
    random.Random(SEED).shuffle(jobs)
    pilot_jobs = [{"instance_id": i["id"], "arm": arm}
                  for i in read_json(root / "pilot-instances.json") for arm in ARMS]
    for name, value in (("validation.json", validation), ("models.json", specs),
                        ("cohort-provenance.json", cohort_provenance), ("schedule.json", jobs),
                        ("pilot-schedule.json", pilot_jobs)):
        write_json(root / name, value)
    if Path(tests_log).resolve() != root / "tests.log":
        with (root / "tests.log").open("x") as stream:
            stream.write(tests)
    hashes = source_hashes()
    snapshot = root / "frozen-source"
    snapshot.mkdir()
    for name, sha in hashes.items():
        target = snapshot / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write((REPO / name).read_bytes())
        require(digest(target) == sha, "Snapshot source drift.")
        target.chmod(0o444)
    inputs = ("instances.json", "pilot-instances.json", "source-manifest.json", "controls.json",
              "pilot-controls.json", "validation.json", "models.json", "cohort-provenance.json",
              "schedule.json", "pilot-schedule.json", "reference-evidence.json", "tests.log")
    protocol = {
        "version": PROTOCOL, "models": list(shared.PANEL), "arms": list(ARMS),
        "cases": 40, "episodes_per_model": 80, "scheduled_episodes": 1120,
        "episode_seconds": 900, "request_seconds": {"jev_action": 30, "all_others": 180},
        "max_calls_per_model": 2 * sum(budget(i) for i in instances),
        "source_sha256": hashes, "inputs_sha256": {name: digest(root / name) for name in inputs},
        "source_file_sha256": {s["file"]: s["sha256"] for s in [
            *read_json(root / "source-manifest.json")["sources"],
            *read_json(root / "reference-evidence.json")["sources"]]},
        "input_scope": "new_public_protocol_not_historical_replication",
        "historical_episode_reuse": 0, "start_conditions_per_graph": 1,
        "environment": shared.audit.environment_versions(),
        "arms_definition": {"A": "all legal next actions",
                            "C": "deduplicated named fixed-heuristic next ACTION proposals, never full solutions"},
        "controls_per_graph": {"classical": 1, "fixed_rules": 4, "within_pool_random_seeds": list(shared.RANDOM_SEEDS)},
        "lt": "two simultaneous seeds; 1/indegree; threshold 1/2; exact all-pair independent validation",
        "modularity": "standard gamma=1; at most two communities; all A legal; fixed 0 and ordered assignments; "
                      "feasible best reference + Q<=1/2 bound, NOT a fabricated optimum",
        "support": "no support caps changed; all scheduled failures/unsupported/unattempted remain in denominators",
        "durability": "unchanged graph_abstraction_suite fsynced intent/raw capture/replay; no retries",
        "pilot_gate": "offline validation is not model pilot completion; parent approval required before any inference",
        "created_at": shared.public.now(),
    }
    require(source_hashes() == hashes, "Source changed during freeze.")
    write_json(root / "protocol.json", protocol)
    write_json(root / "protocol.sha256", digest(root / "protocol.json"))
    for name in (*inputs, "protocol.json", "protocol.sha256"):
        (root / name).chmod(0o444)
    load(root)
    receipt = {
        "version": PROTOCOL, "offline_ready": True, "inference_launched": False,
        "main_inference_authorized": False, "pilot_inference_completed": False,
        "protocol_sha256": digest(root / "protocol.json"),
        "validation_sha256": digest(root / "validation.json"), "tests_sha256": digest(root / "tests.log"),
        "main_graphs": 40, "pilot_graphs": 4, "scheduled_episodes": 1120,
        "entrypoint": str(snapshot / "src/utils/public_optimization.py"),
        "working_directory": str(snapshot),
        "run": "python -m src.utils.public_optimization run --root ROOT --models MODEL",
        "pilot_run": "python -m src.utils.public_optimization run --root ROOT --scope pilot --models MODEL",
        "gate": "deployment-approval.json must bind protocol_sha256, authorized=true, "
                "scope=pilot for pilot; pilot_validated=true, scope=main for main; "
                "absence blocks before client creation",
        "limitations": ["Modularity references are feasible lower bounds, not proven optima.",
                        "Dataset redistribution license not verified; retain source attribution/cached data locally.",
                        "No old public episodes reused or overwritten."],
    }
    write_json(root / "ready-receipt.json", receipt)
    return receipt


def load(root):
    root = Path(root).resolve()
    protocol = read_json(root / "protocol.json")
    require(protocol["version"] == PROTOCOL and protocol["arms"] == list(ARMS) and
            protocol["models"] == list(shared.PANEL), "RQ3 protocol identity drift.")
    require(digest(root / "protocol.json") == read_json(root / "protocol.sha256"), "Protocol seal mismatch.")
    for name, expected in {**protocol["inputs_sha256"], **protocol["source_file_sha256"]}.items():
        require(digest(root / name) == expected, f"Frozen input drift: {name}")
    for name, expected in protocol["source_sha256"].items():
        require(digest(root / "frozen-source" / name) == expected and digest(REPO / name) == expected,
                f"Run the frozen-source entrypoint; code drift: {name}")
    return root, protocol, {i["id"]: i for i in read_json(root / "instances.json")}


def runtime():
    """Private module instance: reuse audited durability without global monkeypatches."""
    name = "_public_optimization_durable"
    spec = importlib.util.spec_from_file_location(name, Path(shared.__file__))
    require(spec is not None and spec.loader is not None, "Cannot load durable runner.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    module.prepare = prepare_instance
    module.score = score
    module.budget = budget
    module.load = load
    module.ARMS = ARMS
    return module


def semantic_spec(spec):
    clean = deepcopy(spec)
    clean.pop("base_url", None)
    if clean.get("provider") == "plugin":
        clean.pop("adapter_file", None)
        clean["adapter_kwargs"].pop("audit_path", None)
        clean["adapter_kwargs"].pop("base_url", None)
    return clean


def check_approval(root, scope):
    require(scope in ("main", "pilot"), "Unknown inference scope.")
    gate = root / "deployment-approval.json"
    require(gate.exists(), "No deployment approval; inference is blocked.")
    approval = read_json(gate)
    require(approval.get("protocol_sha256") == digest(root / "protocol.json") and
            approval.get("authorized") is True and approval.get("scope") == scope,
            "Inference approval does not bind this protocol and scope.")
    if scope == "main":
        require(approval.get("pilot_validated") is True, "Main inference requires validated public pilots.")
    shared.check_hold(root)


async def run(root, models, overrides=None, concurrency=1, scope="main"):
    root, protocol, instances = load(root)
    require(models and len(set(models)) == len(models) and set(models) <= set(protocol["models"]),
            "Choose distinct frozen model IDs.")
    check_approval(root, scope)
    specs = read_json(root / "models.json")
    if overrides:
        supplied = read_json(overrides)
        require(set(supplied) == set(models), "Transport overrides must match selected models.")
        for name in models:
            require(semantic_spec(supplied[name]) == semantic_spec(specs[name]),
                    "Only transport locations/audit paths may change; no model/sampling/cap changes.")
            specs[name] = supplied[name]
    require(type(concurrency) is int and 1 <= concurrency <= 16, "Invalid concurrency.")
    runner = runtime()
    runner.check_hold = lambda unused: check_approval(root, scope)
    for name in models:
        config = runner.validate_lane_config(name, specs[name])
        require(config.timeout_seconds == (30 if name == "jev_action" else 180), "Request timeout changed.")
        require(config.provider != "plugin" or concurrency == 1, "Native/plugin lanes require concurrency 1.")
    execution_root = root
    if scope == "pilot":
        instances = {i["id"]: i for i in read_json(root / "pilot-instances.json")}
        execution_root = root / "pilot"
        execution_root.mkdir(exist_ok=True)
        jobs = read_json(root / "pilot-schedule.json")
        schedule = execution_root / "schedule.json"
        if schedule.exists():
            require(read_json(schedule) == jobs, "Pilot schedule drift.")
        else:
            write_json(schedule, jobs)
        for name in models:
            config = runner.validate_config(specs[name])
            if runner.native_config(config):
                specs[name]["adapter_kwargs"]["audit_path"] = str(execution_root / "native-audit" / f"{name}.jsonl")
    for name in models:
        await runner.run_lane(execution_root, instances, name, specs[name], concurrency)


def report(root, scope="main"):
    root, protocol, instances = load(root)
    require(scope in ("main", "pilot"), "Unknown reporting scope.")
    execution_root = root
    schedule = read_json(root / "schedule.json")
    if scope == "pilot":
        execution_root = root / "pilot"
        instances = {i["id"]: i for i in read_json(root / "pilot-instances.json")}
        schedule = read_json(root / "pilot-schedule.json")
    runner, rows = runtime(), []
    for name in protocol["models"]:
        lane = execution_root / "runs" / name
        config = None
        if (lane / "config.json").exists():
            require(digest(lane / "config.json") == read_json(lane / "config.sha256"), "Lane config drift.")
            spec = read_json(lane / "config.json")
            require(semantic_spec(spec) == semantic_spec(read_json(root / "models.json")[name]),
                    "Lane changed frozen model semantics.")
            config = runner.validate_config(spec, verify_adapter=False)
        for job in schedule:
            instance = instances[job["instance_id"]]
            directory = lane / job["arm"] / instance["id"]
            if (directory / "score.json").exists():
                require(config is not None, "Completed episode without lane configuration.")
                result = runner.verify_episode(directory, instance, config, job["arm"])
            else:
                result = {"instance_id": instance["id"], "task": instance["task"], "arm": job["arm"],
                          "feasible": False, "objective": None, "model_calls": None,
                          "terminal_reason": "in_progress" if directory.exists() else "unattempted"}
            rows.append({"model": name, **result})
    summaries = []
    for name in protocol["models"]:
        for task in TASKS:
            for arm in ARMS:
                own = [r for r in rows if (r["model"], r["task"], r["arm"]) == (name, task, arm)]
                feasible = [r for r in own if r["feasible"]]
                values = [r["objective"] for r in feasible]
                summaries.append({
                    "model": name, "task": task, "arm": arm, "scheduled": len(own),
                    "feasible": len(feasible), "feasible_rate": len(feasible) / len(own),
                    "terminal_reasons": dict(Counter(r.get("terminal_reason") or "complete" for r in own)),
                    "graph_macro_objective_feasible_only": statistics.mean(values) if values else None,
                    "confirmed_model_calls": sum(r.get("confirmed_model_calls", 0) for r in own),
                    "unknown_call_episodes": sum(r.get("model_calls_unknown", False) for r in own),
                    "forced_candidate_steps": sum(r.get("forced_candidate_steps", 0) for r in own),
                    "mean_candidate_coverage": (statistics.mean(
                        [v for r in own for v in r.get("candidate_coverage", [])])
                        if any(r.get("candidate_coverage") for r in own) else None),
                    "reference_status": ("feasible_best_reference_not_optimum" if task == TASKS[3] else
                                         "historical_source_BKS_not_optimum" if task == TASKS[1] else "proven_optimum"),
                })
    paired = []
    for name in protocol["models"]:
        for task in TASKS:
            differences = []
            for instance in (i for i in instances.values() if i["task"] == task):
                arms = {r["arm"]: r for r in rows if r["model"] == name and r["instance_id"] == instance["id"]}
                if all(arms[arm]["feasible"] for arm in ARMS):
                    differences.append(arms["C"]["objective"] - arms["A"]["objective"])
            paired.append({"model": name, "task": task, "paired_feasible_graphs": len(differences),
                           "mean_C_minus_A_objective": statistics.mean(differences) if differences else None,
                           "favorable_sign": "negative" if task == "tsp_public" else "positive"})
    result = {"protocol_sha256": digest(root / "protocol.json"), "scope": scope, "scheduled": len(rows),
              "rows": rows, "summaries": summaries, "excluded": 0,
              "paired_contrasts": paired,
              "controls_sha256": digest(root / ("controls.json" if scope == "main" else "pilot-controls.json")),
              "analysis_unit": "one original/induced graph; no imputation or cross-task objective averaging",
              "inference_complete": all(r.get("terminal_reason") not in ("unattempted", "in_progress") for r in rows)}
    write_json(root / ("report.json" if scope == "main" else "pilot-report.json"), result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "source-evidence", "validate", "freeze", "run", "report"))
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--tests-log", type=Path)
    parser.add_argument("--cohort", type=Path, default=COHORT)
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--model-config", type=Path)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--scope", choices=("main", "pilot"), default="main")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        instances, pilots = prepare_data(args.root)
        acquire_reference_evidence(args.root)
        write_json(args.root / "controls.json", controls(instances))
        write_json(args.root / "pilot-controls.json", controls(pilots))
    elif args.command == "source-evidence":
        acquire_reference_evidence(args.root)
    elif args.command == "validate":
        result = validate_data(args.root)
        write_json(args.root / "offline-validation.json", result)
        print(json.dumps(result, indent=2))
    elif args.command == "freeze":
        require(args.tests_log is not None, "--tests-log is required.")
        print(json.dumps(freeze(args.root, args.tests_log, args.cohort), indent=2))
    elif args.command == "run":
        asyncio.run(run(args.root, args.models, args.model_config, args.concurrency, args.scope))
    else:
        result = report(args.root, args.scope)
        print(json.dumps({k: v for k, v in result.items() if k not in ("rows", "summaries")}, indent=2))


if __name__ == "__main__":
    main()
