"""Offline v33 summaries from final-result CSVs; no downloads or model calls.

Usage: python -m src.utils.public_results_v33 --data-dir assets/benchmark/v33/gpt54
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
import random
from statistics import mean


TASKS = ("adjacency", "degree_exact", "cycle_detection", "pair_connectivity",
         "distance_threshold", "articulation_point")
SOURCES = ("facebook", "ca-grqc", "power", "human-ppi")
SIZES = tuple(range(8, 13))
ARMS = ("T", "G", "TG", "BAG", "A")
OPTIMIZATION_TASKS = ("tsp_public", "maxcut_public", "lt_influence_construct",
                      "community_bipartition_construct")
CONTRASTS = (("TG", "BAG"), ("G", "A"), ("TG", "T"), ("TG", "G"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames and len(reader.fieldnames) == len(set(reader.fieldnames)),
                f"Missing or duplicate CSV columns: {path}")
        rows = list(reader)
    require(rows and all(None not in row and None not in row.values() for row in rows),
            f"Empty or malformed CSV: {path}")
    return rows


def bit(row, key):
    require(row[key] in ("0", "1"), f"{key} must be 0 or 1")
    return int(row[key])


def correctness(rows):
    for row in rows:
        require(not bit(row, "correct") or bit(row, "output_legal"),
                "An illegal output cannot be correct")


def rq1_summary(rows):
    require(len(rows) == 800 and len({r["query_id"] for r in rows}) == 800,
            "RQ1 requires 800 unique query IDs")
    correctness(rows)
    cells = defaultdict(list)
    for row in rows:
        source, task, n = row["data_source"], row["task"], int(row["node_count"])
        require(source in SOURCES and task in TASKS and n in SIZES, "Unknown RQ1 stratum")
        cells[source, task, n].append(bit(row, "correct"))
    for source in SOURCES:
        for task in TASKS:
            for n in SIZES:
                require(len(cells[source, task, n]) == (2 * n if task == "degree_exact" else 4),
                        f"RQ1 quota mismatch: {source}/{task}/{n}")
    panels = {}
    for label, sources in (("all", SOURCES), *((s, (s,)) for s in SOURCES)):
        by_task = {}
        for task in TASKS:
            sizes = {n: [x for source in sources for x in cells[source, task, n]] for n in SIZES}
            values = [x for v in sizes.values() for x in v]
            by_task[task] = {
                "correct": sum(values), "scheduled": len(values),
                "query_weighted_percent": 100 * mean(values),
                "size_equal_percent": 100 * mean(mean(v) for v in sizes.values()),
                "by_size": {str(n): {"correct": sum(v), "scheduled": len(v),
                                     "accuracy_percent": 100 * mean(v)} for n, v in sizes.items()},
            }
        own = [r for r in rows if r["data_source"] in sources]
        panels[label] = {
            "scheduled": len(own), "correct": sum(bit(r, "correct") for r in own),
            "valid": sum(bit(r, "output_legal") for r in own),
            "query_weighted_percent": 100 * mean(bit(r, "correct") for r in own),
            "task_macro_percent": mean(t["query_weighted_percent"] for t in by_task.values()),
            "task_size_macro_percent": mean(t["size_equal_percent"] for t in by_task.values()),
            "by_task": by_task,
        }
    return panels


def paired_intervals(by_query):
    """The manuscript's GPT-5.4 bootstrap: numeric ID order, common index draws."""
    ids = sorted(by_query, key=int)
    require(bool(ids), "Cannot bootstrap an empty paired cohort")
    differences = {f"{left}-{right}": [by_query[q][left] - by_query[q][right] for q in ids]
                   for left, right in CONTRASTS}
    bootstrap = {key: [] for key in differences}
    rng = random.Random(20261001)
    for _ in range(1000):
        indices = [rng.randrange(len(ids)) for _ in ids]
        for key, values in differences.items():
            bootstrap[key].append(100 * sum(values[i] for i in indices) / len(ids))
    return {
        key: {"difference_pp": 100 * mean(values),
              "ci95_pp": [sorted(bootstrap[key])[24], sorted(bootstrap[key])[974]],
              "paired_queries": len(ids)}
        for key, values in differences.items()
    }


def rq2_summary(rows):
    require(len(rows) == 1000, "RQ2 requires 1000 query-condition rows")
    correctness(rows)
    datasets = defaultdict(dict)
    validity = defaultdict(Counter)
    for row in rows:
        dataset, query, arm = row["dataset"], row["query_id"], row["condition"]
        require(dataset in ("arxiv", "prime") and arm in ARMS, "Unknown RQ2 dataset/condition")
        require(query.isdecimal() and str(int(query)) == query, "Noncanonical numeric query ID")
        views = datasets[dataset].setdefault(query, {})
        require(arm not in views, f"Duplicate RQ2 pair: {dataset}/{query}/{arm}")
        views[arm] = bit(row, "correct")
        validity[dataset][arm] += bit(row, "output_legal")
    result = {}
    for dataset in ("arxiv", "prime"):
        by_query = datasets[dataset]
        require(len(by_query) == 100 and all(set(v) == set(ARMS) for v in by_query.values()),
                f"RQ2 needs 100 queries with all five paired conditions: {dataset}")
        result[dataset] = {
            "by_condition": {
                arm: {"correct": sum(v[arm] for v in by_query.values()), "scheduled": 100,
                      "score_percent": 100 * mean(v[arm] for v in by_query.values()),
                      "valid": validity[dataset][arm], "invalid": 100 - validity[dataset][arm]}
                for arm in ARMS},
            "paired": paired_intervals(by_query),
        }
    return result


def finite(row, key):
    value = float(row[key])
    require(math.isfinite(value), f"Nonfinite {key}")
    return value


def rq3_summary(rows):
    require(len(rows) == 80, "RQ3 requires 80 scheduled trajectories")
    groups, pairs = defaultdict(list), defaultdict(dict)
    references = {}
    for row in rows:
        task, graph, mode = row["task"], row["graph_id"], row["mode"]
        require(task in OPTIMIZATION_TASKS and mode in ("A", "C"), "Unknown RQ3 task/mode")
        require(mode not in pairs[task, graph], "Duplicate RQ3 graph/mode")
        calls = int(row["model_calls"])
        require(calls >= 0, "Negative model-call count")
        feasible = bit(row, "feasible")
        output_legal = bit(row, "output_legal")
        reference = finite(row, "reference")
        reference_identity = (reference, row["reference_status"], row["gap_unit"])
        require((task, graph) not in references or references[task, graph] == reference_identity,
                "Paired RQ3 references differ")
        references[task, graph] = reference_identity
        gap = None
        if feasible:
            require(output_legal and row["final_status"] in ("complete", "feasible"),
                    "Inconsistent feasible trajectory")
            objective = finite(row, "objective")
            if task == "community_bipartition_construct":
                require(row["gap_unit"] == "absolute_Q"
                        and row["reference_status"] == "numerically_certified_MILP_optimum",
                        "Modularity requires the v33 exact reference and absolute gap")
                expected = reference - objective
            else:
                require(reference > 0 and row["gap_unit"] == "percent", "Invalid percentage reference")
                expected = 100 * (objective - reference if task == "tsp_public"
                                  else reference - objective) / reference
            gap = finite(row, "gap")
            require(math.isclose(gap, expected, rel_tol=1e-9, abs_tol=1e-9), "RQ3 gap mismatch")
        else:
            require(row["final_status"] not in ("complete", "feasible"),
                    "Inconsistent incomplete trajectory")
            require(row["objective"] == row["gap"] == "", "Failed trajectory has imputed quality")
        groups[task, mode].append((gap, calls))
        pairs[task, graph][mode] = gap
    require(len(pairs) == 40 and all(set(v) == {"A", "C"} for v in pairs.values()),
            "Incomplete RQ3 A/C graph pairs")
    result = {}
    for task in OPTIMIZATION_TASKS:
        modes = {}
        for mode in ("A", "C"):
            values = groups[task, mode]
            require(len(values) == 10, f"Expected ten RQ3 graphs: {task}/{mode}")
            gaps = [gap for gap, _ in values if gap is not None]
            modes[mode] = {"scheduled": 10, "feasible": len(gaps),
                           "mean_gap": mean(gaps) if gaps else None,
                           "model_calls": sum(calls for _, calls in values)}
        differences = [v["C"] - v["A"] for (t, _), v in pairs.items()
                       if t == task and v["A"] is not None and v["C"] is not None]
        result[task] = {"by_mode": modes, "gap_unit": "absolute_Q" if task ==
                       "community_bipartition_construct" else "percent",
                       "paired_C_minus_A": mean(differences) if differences else None,
                       "paired_graphs": len(differences)}
    return result


def report(directory):
    directory = Path(directory)
    return {
        "protocol": "GraphDecide-v33-final-selected-results",
        "inference_performed": False,
        "rq1": rq1_summary(read_csv(directory / "rq1.csv")),
        "rq2": rq2_summary(read_csv(directory / "rq2.csv")),
        "rq3": rq3_summary(read_csv(directory / "rq3.csv")),
        "notes": [
            "Final selected results are not claims of single-attempt performance.",
            "Prime A is displayed as A* in the paper; invalid answers count as incorrect.",
            "Macro-F1 cannot be reconstructed from binary correctness; requires labels and predictions.",
            "Bootstrap: 1000 resamples, seed 20261001, numeric IDs, endpoints 24/974; pointwise.",
            "RQ3 calls count selected trajectories only; retry costs are separate.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(report(args.data_dir), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
