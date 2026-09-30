#!/usr/bin/env python3
"""Render predeclared supplemental baseline/heterogeneity outputs, without inference.

This is a reporting-only extension of the frozen controller's verified analysis.
It does not change the protocol, candidate pool, runs, scores, or its initial
analysis files. Its own source hash and all output hashes are sealed separately.
"""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import statistics


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def mean(values):
    present = [value for value in values if value is not None]
    return statistics.mean(present) if present else None


def write_json(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def write_csv(path, rows):
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def baseline_tables(rows):
    episodes = [{
        "method": row["method"], "instance_id": row["instance_id"],
        "dataset_id": row["dataset_id"], "replicate": row["replicate"], "seed": row["seed"],
        "feasible": row["feasible"], "objective": row["objective"], "gap_percent": row["percentage_gap"],
        "calls": row["model_calls"], "solve_wall_seconds": row["solve_wall_seconds"],
        "setup_seconds": row["setup_seconds"], "candidate_seconds": row["candidate_seconds"],
        "verification_seconds": row["verification_seconds"],
        "forced_candidate_steps": row["forced_candidate_steps"],
    } for row in rows]
    conditions = []
    for method, identity in sorted({(row["method"], row["instance_id"]) for row in episodes}):
        own = [row for row in episodes if (row["method"], row["instance_id"]) == (method, identity)]
        conditions.append({
            "method": method, "instance_id": identity, "dataset_id": own[0]["dataset_id"],
            "replicate": own[0]["replicate"], "seed_count": len(own),
            "gap_percent": mean([row["gap_percent"] for row in own]),
            "gap_percent_seed_min": min(row["gap_percent"] for row in own),
            "gap_percent_seed_max": max(row["gap_percent"] for row in own),
            "solve_wall_seconds": mean([row["solve_wall_seconds"] for row in own]),
        })
    graphs = []
    for method, dataset in sorted({(row["method"], row["dataset_id"]) for row in conditions}):
        own = [row for row in conditions if (row["method"], row["dataset_id"]) == (method, dataset)]
        graphs.append({
            "method": method, "dataset_id": dataset, "conditions": len(own),
            "gap_percent": mean([row["gap_percent"] for row in own]),
            "solve_wall_seconds": mean([row["solve_wall_seconds"] for row in own]),
        })
    return episodes, conditions, graphs


def main(root):
    root = Path(root).resolve()
    if read(root / "status.json")["status"] != "analysis_ready":
        raise ValueError("Frozen controller analysis must be terminal and independently verified.")
    base = root / "analysis"
    receipt = read(base / "verification-receipt.json")
    if receipt["status"] != "passed" or receipt["new_scheduled_terminal_episodes"] != 126:
        raise ValueError("Expected all 126 independently verified terminal episodes.")
    for name, expected in receipt["analysis_sha256"].items():
        if digest(base / name) != expected:
            raise ValueError("Frozen analysis artifact hash mismatch.")
    if digest(root / "protocol.json") != receipt["protocol_sha256"]:
        raise ValueError("Protocol drift.")
    out = root / "report-tables"
    out.mkdir(exist_ok=False)
    episodes, conditions, graphs = baseline_tables(read(root / "baselines.json"))
    write_csv(out / "baseline-episodes.csv", episodes)
    write_csv(out / "baseline-conditions.csv", conditions)
    write_csv(out / "baseline-per-graph.csv", graphs)
    models = list(csv.DictReader((base / "episodes.csv").open()))
    for row in models:
        if row["arm"] == "A":
            row["terminal_reason"] = ""
        else:
            result = read(root / "runs" / row["model"] / row["arm"] / row["instance_id"] / "score.json")
            row["terminal_reason"] = result["terminal_reason"] or ""
    write_csv(out / "model-episodes-with-terminal-reasons.csv", models)
    methods = sorted({row["method"] for row in conditions})
    lookup = {(row["method"], row["instance_id"]): row for row in conditions}
    paired = []
    for model, arm, dataset in sorted({(row["model"], row["arm"], row["dataset_id"]) for row in models}):
        own = [row for row in models if (row["model"], row["arm"], row["dataset_id"]) == (model, arm, dataset)]
        for method in methods:
            complete = [row for row in own if row["feasible"] == "True"]
            paired.append({
                "model": model, "arm": arm, "baseline": method, "dataset_id": dataset,
                "scheduled": len(own), "completed": len(complete),
                "model_minus_baseline_gap_pp": mean([
                    float(row["gap_percent"]) - lookup[method, row["instance_id"]]["gap_percent"]
                    for row in complete]),
            })
    write_csv(out / "paired-model-vs-baseline.csv", paired)
    comparative = []
    for model, arm, method in sorted({(r["model"], r["arm"], r["baseline"]) for r in paired}):
        own = [r for r in paired if (r["model"], r["arm"], r["baseline"]) == (model, arm, method)]
        comparative.append({
            "model": model, "arm": arm, "baseline": method,
            "graph_macro_model_minus_baseline_pp": mean([r["model_minus_baseline_gap_pp"] for r in own]),
            "graphs_model_better": sum(r["model_minus_baseline_gap_pp"] is not None
                                      and r["model_minus_baseline_gap_pp"] < 0 for r in own),
            "paired_conditions": sum(r["completed"] for r in own),
        })
    write_csv(out / "model-vs-baseline-summary.csv", comparative)
    candidate_rows = list(csv.DictReader((base / "candidate-steps.csv").open()))
    selection_rows = list(csv.DictReader((base / "rule-selections.csv").open()))
    behavior = []
    for model, arm, dataset in sorted({(r["model"], r["arm"], r["dataset_id"]) for r in candidate_rows}):
        own = [r for r in candidate_rows if (r["model"], r["arm"], r["dataset_id"]) == (model, arm, dataset)]
        selections = [r for r in selection_rows
                      if (r["model"], r["arm"], r["dataset_id"]) == (model, arm, dataset) and r["forced"] == "False"]
        credits = Counter()
        positions = Counter()
        groups = Counter()
        seen = set()
        for row in selections:
            credits[row["rule"]] += float(row["fractional_credit"])
            key = row["instance_id"], row["step"]
            if key not in seen:
                seen.add(key)
                positions[row["option_id"]] += 1
                groups[row["group"]] += 1
        total = sum(credits.values())
        probabilities = [value / total for value in credits.values()] if total else []
        behavior.append({
            "model": model, "arm": arm, "dataset_id": dataset,
            "observed_append_steps": len(own),
            "all_rules_agree_rate": mean([r["all_rules_agree"] == "True" for r in own]),
            "pairwise_rule_agreement_rate": mean([float(r["pairwise_rule_agreement"]) for r in own]),
            "mean_candidate_count": mean([int(r["candidates"]) for r in own]),
            "mean_candidate_fraction": mean([float(r["candidate_coverage"]) for r in own]),
            "nonforced_selections": len(seen),
            "maximum_fractional_rule_share": max(probabilities) if probabilities else None,
            "rule_fractional_distribution": {rule: value / total for rule, value in credits.items()} if total else {},
            "option_position_counts": dict(positions), "chosen_rule_group_counts": dict(groups),
        })
    write_json(out / "per-graph-choice-behavior.json", behavior)
    macro_behavior = []
    for model, arm in sorted({(row["model"], row["arm"]) for row in behavior}):
        own = [r for r in behavior if (r["model"], r["arm"]) == (model, arm)]
        macro_behavior.append({
            "model": model, "arm": arm, "observed_graphs": len(own),
            "graph_macro_all_rules_agree_rate": mean([r["all_rules_agree_rate"] for r in own]),
            "graph_macro_candidate_count": mean([r["mean_candidate_count"] for r in own]),
            "graph_macro_candidate_fraction": mean([r["mean_candidate_fraction"] for r in own]),
            "graph_macro_maximum_rule_share": mean([r["maximum_fractional_rule_share"] for r in own]),
        })
    write_csv(out / "graph-macro-choice-behavior.csv", macro_behavior)
    timing = []
    for model, arm in sorted({(r["model"], r["arm"]) for r in models}):
        own = [r for r in models if (r["model"], r["arm"]) == (model, arm)]
        values = {}
        for metric in ("candidate_seconds", "setup_seconds", "model_call_seconds",
                       "solve_wall_seconds", "verification_seconds", "calls", "forced_candidate_steps"):
            graph_values = []
            for dataset in sorted({r["dataset_id"] for r in own}):
                graph_values.append(mean([
                    float(r[metric]) for r in own if r["dataset_id"] == dataset and r[metric] != ""]))
            values[f"graph_macro_{metric}"] = mean(graph_values)
        timing.append({
            "model": model, "arm": arm, "scheduled": len(own),
            "completed": sum(r["feasible"] == "True" for r in own), **values,
            "timing_population": "all scheduled conditions; an incomplete episode is not a solved-tour timing",
        })
    write_csv(out / "timing-and-call-accounting.csv", timing)
    summary = read(base / "summary.json")
    recommendation = read(base / "paper-recommendation.json")
    interpretation = [
        "No paper or README was edited. All findings are descriptive, based on seven original graphs.",
        "Gap differences below are percentage points, negative means the first method is better.",
        "Main paired graph contrasts (all models and directions):",
        *[json.dumps(row) for row in summary["contrasts"]],
        "Comparisons to random candidates and NN+2opt (all model arms):",
        *[json.dumps(row) for row in comparative
          if row["baseline"] in ("random_candidate", "nearest_neighbor_two_opt")],
        "Candidate agreement and selection concentration are trajectory-conditional, not fixed treatment attributes.",
        "B retrospective rule memberships must not be interpreted as explicit semantic choices.",
        "A timings include legacy replay and cannot support a clean solve-only speed comparison.",
        "Frozen paper inclusion recommendation: " + recommendation["recommendation"],
    ]
    (out / "interpretation.txt").write_text("\n".join(interpretation) + "\n")
    write_json(out / "reporting-receipt.json", {
        "reporting_only": True,
        "protocol_sha256": digest(root / "protocol.json"),
        "controller_verification_receipt_sha256": digest(base / "verification-receipt.json"),
        "source": {"path": str(Path(__file__).resolve()), "sha256": digest(Path(__file__))},
        "method": "Render already-frozen metrics; random seeds averaged within condition before equal graph weighting. No new model selection, rule tuning, inference, rescoring or historical file edits.",
        "output_sha256": {path.name: digest(path) for path in out.iterdir() if path.is_file()},
    })
    print(json.dumps({"status": "report_tables_ready", "directory": str(out)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    main(parser.parse_args().root)
