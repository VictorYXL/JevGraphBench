"""Export portable metrics from supplied, hash-bound independent replay artifacts."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
from collections import defaultdict
from statistics import mean


TASKS = {
    "maxcut_construct": "cut edges",
    "tsp_construct": "distance units",
    "lt_influence_construct": "active vertices",
    "community_bipartition_construct": "modularity units",
    "tsp_public": "percentage points",
    "maxcut_public": "percentage points",
}
TASK_TITLES = {
    "maxcut_construct": "Synthetic MaxCut",
    "tsp_construct": "Synthetic Manhattan TSP",
    "lt_influence_construct": "Synthetic deterministic LT",
    "community_bipartition_construct": "Synthetic binary modularity",
    "tsp_public": "Public TSP",
    "maxcut_public": "Public signed MaxCut",
}
COUNTS = ("scheduled", "complete", "unsupported", "failed", "not_started")
ARM_METRICS = (*COUNTS, "gap", "model_calls", "confirmed_model_calls",
               "unknown_call_episodes", "forced_steps", "singleton_steps",
               "observed_steps", "graphs_with_completion")
EPISODE_METRICS = (
    "objective", "absolute_gap", "percentage_gap", "model_calls",
    "confirmed_model_calls", "forced_candidate_steps", "call_intents",
    "confirmed_native_forwards", "native_forward_count", "native_forward_unknown_attempts",
    "captured_response_records", "raw_response_capture_failures", "orphan_raw_responses",
    "probability_audit_failures", "candidate_seconds", "model_call_seconds",
    "solve_wall_seconds", "verification_seconds",
)
TERMINAL_REASONS = frozenset({
    "unsupported", "timeout", "http_error", "client_error", "internal_error",
    "interrupted", "episode_timeout", "fatal_lane_abort", "interrupted_no_retry", "worker_failure",
    "invalid_response", "missing_answer", "invalid_json", "invalid_choice",
    "incomplete_reasoning", "unexpected_reasoning", "invalid_usage", "invalid_finish_reason",
    "finish_length", "finish_tool_calls", "finish_function_call", "finish_content_filter",
    "finish_error", "finish_abort", "invalid_model", "invalid_native_answer",
    "invalid_probabilities", "invalid_probability_sum", "choice_probability_mismatch", "invalid_confidence",
})
BEGIN = "<!-- proposal-results:start -->"
END = "<!-- proposal-results:end -->"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def number(value):
    require(value is None or (type(value) in (int, float) and math.isfinite(value)),
            "Public metric must be finite numeric or null")
    return value


def metrics(row, keys):
    return {key: number(row[key]) for key in keys}


def identifier(value):
    require(isinstance(value, str) and
            re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", value) is not None,
            "Unsafe public identifier")
    return value


def counts(row):
    result = metrics(row, COUNTS)
    require(all(type(v) is int and v >= 0 for v in result.values()),
            "Invalid coverage counts")
    require(result["scheduled"] == sum(result[k] for k in COUNTS[1:]),
            "Coverage categories do not partition scheduled episodes")
    return result


def episode(row, cohort):
    require(row["task"] in TASKS and row["arm"] in "ABC"
            and len(row["arm"]) == 1, "Unknown episode configuration")
    require(type(row["feasible"]) is bool, "Missing explicit feasibility")
    reason = row["terminal_reason"]
    require(reason not in ("in_progress", "unattempted"), "Nonterminal episode")
    require(reason is None or type(reason) is str and reason in TERMINAL_REASONS,
            "Unrecognized terminal diagnostic")
    require(row["feasible"] == (reason is None), "Terminal reason/feasibility mismatch")
    category = ("complete" if row["feasible"] else
                "unsupported" if reason == "unsupported" else
                "fatal_lane_abort" if reason == "fatal_lane_abort" else "failed")
    identity = identifier(row["instance_id"])
    graph = identifier(row.get("dataset_id") or identity)
    result = {
        "cohort": cohort, "model": identifier(row["model"]), "task": row["task"], "arm": row["arm"],
        "instance_id": identity, "graph_id": graph,
        "feasible": row["feasible"], "terminal_reason": reason, "terminal_category": category,
        "model_calls_unknown": row["model_calls_unknown"],
        **{key: number(row.get(key)) for key in EPISODE_METRICS},
    }
    for key in EPISODE_METRICS:
        value = result[key]
        if value is not None and key not in ("objective", "absolute_gap", "percentage_gap"):
            require(value >= 0 and (key.endswith("_seconds") or type(value) is int),
                    "Invalid timing or counter metric")
    gap_key = "percentage_gap" if row["task"].endswith("_public") else "absolute_gap"
    for key in ("objective", gap_key, "model_calls", "confirmed_model_calls", "forced_candidate_steps"):
        require(key in row, f"Missing episode metric: {key}")
    for key in ("model_calls", "confirmed_model_calls", "forced_candidate_steps"):
        value = result[key]
        require((key == "model_calls" and value is None) or type(value) is int and value >= 0,
                "Invalid call/step count")
    require(type(row["model_calls_unknown"]) is bool
            and row["model_calls_unknown"] == (result["model_calls"] is None),
            "Unknown call flag mismatch")
    require(result["model_calls"] is None or
            result["model_calls"] == result["confirmed_model_calls"], "Call count mismatch")
    require(not row["feasible"] or result["objective"] is not None and
            result[gap_key] is not None, "Completed episode lacks objective/gap")
    require(row["feasible"] or result["objective"] is None and
            result[gap_key] is None, "Incomplete objective must not be imputed")
    require(row["task"].endswith("_public") or result[gap_key] is None or result[gap_key] >= -1e-12,
            "Negative exact-optimum gap")
    return result


def load_verified(summary, report, controls, replay, summary_sha256, replay_sha256, supplement):
    """Validate supplied evidence, not production publication authorization."""
    require(digest(summary) == summary_sha256, "Summary binding mismatch")
    require(digest(replay) == replay_sha256, "Independent replay binding mismatch")
    receipt, data, primary, baseline = read(replay), read(summary), read(report), read(controls)
    protocol = receipt["protocol_sha256"]
    require(isinstance(protocol, str) and re.fullmatch(r"[a-f0-9]{64}", protocol) is not None,
            "Invalid protocol digest")
    require(receipt["status"] == "passed"
            and receipt["episodes_replayed"] == len(primary["episodes"])
            and receipt["controls_replayed"] == len(baseline)
            and receipt["new_model_calls"] == receipt["model_initializations"] == 0,
            "Independent replay accounting mismatch")
    require(digest(report) == receipt["report_sha256"] and
            digest(controls) == receipt["baseline_sha256"], "Report/control binding mismatch")
    require(data.get("purpose") != "results_display" and primary.get("purpose") != "results_display"
            and data["protocol_sha256"] == primary["protocol_sha256"] == protocol,
            "Display or foreign results cannot be released")
    require(data["scheduled"] == len(primary["episodes"]) > 0 and baseline,
            "Empty or mismatched scheduled panel")
    require(data["complete"] == sum(r["feasible"] for r in primary["episodes"])
            and data["noncomplete"] == data["scheduled"] - data["complete"],
            "Primary coverage mismatch")
    supplements = {}
    keys = set()
    for model, key, path in supplement:
        identifier(model)
        identifier(key)
        require(model not in supplements and key not in keys, "Duplicate supplement specification")
        keys.add(key)
        verification = receipt[f"{key}_verification"]
        total = verification["scheduled"]
        require(verification["status"] == "passed"
                and verification["primary_rows_unchanged"] is True
                and verification["terminal_verified"] == verification["exercised"] == total
                and digest(path) == verification["report_sha256"], "Supplement replay binding mismatch")
        own = read(path)
        require(own["status"] == "independently_replayed"
                and own["primary_rows_unchanged"] is True
                and own["primary_protocol_sha256"] == protocol
                and own["scheduled"] == own["terminal_verified"] ==
                own["exercised"] == len(own["rows"]) == total,
                "Supplement must be separately exercised and verified")
        require(all(r["model"] == model and r["exercised"] is True for r in own["rows"]),
                "Supplement row identity mismatch")
        supplements[model] = (key, own["rows"])
    require(keys == {key for key, value in data.items()
                     if isinstance(value, dict) and "supplement" in value and "panels" in value},
            "Missing or unexpected supplemental report")
    return data, primary["episodes"], baseline, supplements, {
        "summary_sha256": digest(summary), "report_sha256": digest(report),
        "controls_sha256": digest(controls), "independent_replay_sha256": digest(replay),
    }


def tables(summary, primary, controls, supplements):
    tasks = set(summary["panels"])
    require(tasks and tasks <= set(TASKS), "Unsupported task panels")
    models = {m["model"] for m in next(iter(summary["panels"].values()))["models"]}
    require(models, "Empty model panel")
    episodes = [episode(r, "primary") for r in primary]
    raw_by_identity = {(r["model"], r["arm"], r["instance_id"]): r for r in primary}
    identities = {(r["model"], r["arm"], r["instance_id"]) for r in episodes}
    reference = {(r["task"], r["instance_id"]) for r in episodes
                 if r["model"] == sorted(models)[0] and r["arm"] == "A"}
    require(reference and len(identities) == len(episodes),
            "Duplicate or missing primary identities")
    require({r["model"] for r in episodes} == models and {r["task"] for r in episodes} == tasks,
            "Report model/task mismatch")
    for model in models:
        for arm in "ABC":
            require({(r["task"], r["instance_id"]) for r in episodes
                     if r["model"] == model and r["arm"] == arm} == reference,
                    "Primary schedules differ")
    lookup = {}
    for row in episodes:
        key = row["task"], row["instance_id"]
        require(key not in lookup or lookup[key] == row["graph_id"], "Graph identity mismatch")
        lookup[key] = row["graph_id"]
    control_rows = []
    for row in controls:
        key = row["task"], row["instance_id"]
        require(key in lookup and row["feasible"] is True, "Invalid control row")
        require(type(row["seed"]) is int, "Control seed must be an integer")
        value = gap(row)
        require(value is not None and number(row["objective"]) is not None, "Missing control objective/gap")
        require(row["task"].endswith("_public") or value >= -1e-12, "Negative exact control gap")
        control_rows.append({
            "task": row["task"], "instance_id": row["instance_id"], "graph_id": lookup[key],
            "method": identifier(row["method"]), "seed": row["seed"],
            **{k: number(row.get(k)) for k in
               ("objective", "absolute_gap", "percentage_gap", "forced_steps", "wall_seconds")},
        })
    require(len({(r["task"], r["instance_id"], r["method"], r["seed"]) for r in control_rows})
            == len(control_rows), "Duplicate controls")
    summaries, paired, control_summary, supplement_summary, supplement_episodes = [], [], [], [], []
    for task in sorted(tasks):
        units = TASKS[task]
        panel = summary["panels"][task]
        require({m["model"] for m in panel["models"]} == models
                and len(panel["models"]) == len(models), "Incomplete model panel")
        expected_instances = {identity for t, identity in reference if t == task}
        methods = {r["method"] for r in control_rows if r["task"] == task}
        require(methods and methods == set(panel["controls"]), "Control method mismatch")
        reduced = {}
        for method in sorted(methods):
            own = [r for r in control_rows if r["task"] == task and r["method"] == method]
            by_instance = defaultdict(list)
            for row in own:
                by_instance[row["instance_id"]].append(row)
            require(set(by_instance) == expected_instances, "Missing control instances")
            require(len({tuple(sorted(r["seed"] for r in rows)) for rows in by_instance.values()}) == 1,
                    "Control seed sets differ between instances")
            reduced[method] = {identity: {"graph_id": rows[0]["graph_id"], "value": mean(gap(r) for r in rows)}
                               for identity, rows in by_instance.items()}
            row = panel["controls"][method]
            equal(row["gap"], macro(reduced[method].values()), "Control gap mismatch")
            require(row["trajectories"] == len(own) and row["instances"] == len(by_instance),
                    "Control coverage mismatch")
            control_summary.append({"task": task, "units": units, "method": method,
                                    **metrics(row, ("gap", "trajectories", "instances"))})
        for model in panel["models"]:
            require(set(model["arms"]) == set("ABC"), "Missing arm")
            arms = {}
            for arm in "ABC":
                row = model["arms"][arm]
                own = [r for r in episodes if r["task"] == task and r["model"] == model["model"] and r["arm"] == arm]
                arms[arm] = {r["instance_id"]: r for r in own}
                validate_coverage(row, own)
                expected_calls = None if any(r["model_calls"] is None for r in own) else sum(r["model_calls"] for r in own)
                equal(row["model_calls"], expected_calls, "Model call total mismatch")
                equal(row["confirmed_model_calls"], sum(r["confirmed_model_calls"] for r in own), "Confirmed call mismatch")
                equal(row["forced_steps"], sum(r["forced_candidate_steps"] for r in own), "Forced step mismatch")
                equal(row["unknown_call_episodes"], sum(r["model_calls"] is None for r in own), "Unknown call count mismatch")
                candidate_counts = [n for r in own for n in
                                    raw_by_identity[r["model"], r["arm"], r["instance_id"]]["candidate_counts"]]
                require(all(type(n) is int and n >= 1 for n in candidate_counts), "Invalid candidate count")
                equal(row["singleton_steps"], candidate_counts.count(1), "Singleton count mismatch")
                equal(row["observed_steps"], len(candidate_counts), "Observed step mismatch")
                require(row["graphs_with_completion"] == len({r["graph_id"] for r in own if r["feasible"]}),
                        "Completed graph count mismatch")
                summaries.append({"task": task, "units": units, "model": model["model"],
                                  "arm": arm, **metrics(row, ARM_METRICS)})
                require(set(row["matched_controls"]) == methods, "Missing matched controls")
                for method, comparison in sorted(row["matched_controls"].items()):
                    complete = [r for r in own if r["feasible"]]
                    matched = [reduced[method][r["instance_id"]] for r in complete]
                    differences = [{"graph_id": r["graph_id"],
                                    "value": gap(r) - reduced[method][r["instance_id"]]["value"]} for r in complete]
                    validate_pairs(comparison, differences, "model_minus_control")
                    equal(comparison["matched_control_gap"], macro(matched), "Matched control gap mismatch")
                    paired.append({"task": task, "units": units, "model": model["model"],
                                   "contrast": f"{arm}-minus-control", "control": method,
                                   **metrics(comparison, ("model_minus_control", "matched_control_gap",
                                                          "pairs", "graphs")),
                                   "C_minus_B": None})
            comparison = model["paired_C_minus_B"]
            differences = [{"graph_id": b["graph_id"], "value": gap(arms["C"][identity]) - gap(b)}
                           for identity, b in arms["B"].items() if b["feasible"] and arms["C"][identity]["feasible"]]
            validate_pairs(comparison, differences, "delta")
            paired.append({"task": task, "units": units, "model": model["model"],
                           "contrast": "C-minus-B", "control": "",
                           "model_minus_control": None, "matched_control_gap": None,
                           **metrics(comparison, ("pairs", "graphs")),
                           "C_minus_B": number(comparison["delta"])})
        for model, (key, rows) in supplements.items():
            require(counts(summary[key]["supplement"])["scheduled"] == len(rows),
                    "Supplement total mismatch")
            for arm in "ABC":
                row = summary[key]["panels"][task]["arms"][arm]
                own = [episode(r, "zero_attempt_supplement") for r in rows if r["task"] == task and r["arm"] == arm]
                validate_coverage(row, own)
                supplement_summary.append({"model": model, "task": task, "units": units,
                                           "arm": arm, **counts(row), "gap": number(row["gap"])})
    for model, (key, rows) in supplements.items():
        own = [episode(r, "zero_attempt_supplement") for r in rows]
        keys = {(r["model"], r["arm"], r["instance_id"]) for r in own}
        require(len(keys) == len(own) and keys <= identities, "Invalid supplement identities")
        require(all(lookup.get((r["task"], r["instance_id"])) == r["graph_id"] for r in own),
                "Supplement task/graph identity mismatch")
        total = summary[key]["supplement"]
        require(total["complete"] == sum(r["feasible"] for r in own), "Supplement coverage mismatch")
        for field in COUNTS:
            require(total[field] == sum(summary[key]["panels"][task]["arms"][arm][field]
                                       for task in tasks for arm in "ABC"), "Supplement counts mismatch")
        supplement_episodes.extend(own)
    return {
        "proposal-summary.csv": summaries, "proposal-episodes.csv": episodes,
        "proposal-controls.csv": control_rows, "proposal-control-summary.csv": control_summary,
        "proposal-paired.csv": paired, "proposal-supplement-summary.csv": supplement_summary,
        "proposal-supplement-episodes.csv": supplement_episodes,
    }


def gap(row):
    return number(row["percentage_gap" if row["task"].endswith("_public") else "absolute_gap"])


def macro(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["graph_id"]].append(row["value"])
    return mean(mean(values) for values in groups.values()) if groups else None


def equal(actual, expected, message):
    number(actual)
    require(actual is None if expected is None else
            actual is not None and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10), message)


def validate_coverage(summary, rows):
    counts(summary)
    require(summary["scheduled"] == len(rows) and summary["complete"] == sum(r["feasible"] for r in rows)
            and summary["unsupported"] == sum(r["terminal_category"] == "unsupported" for r in rows),
            "Episode coverage mismatch")
    equal(summary["gap"], macro({"graph_id": r["graph_id"], "value": gap(r)}
                               for r in rows if r["feasible"]), "Graph-macro gap mismatch")


def validate_pairs(summary, rows, field):
    require(summary["pairs"] == len(rows) and summary["graphs"] == len({r["graph_id"] for r in rows}),
            "Paired coverage mismatch")
    equal(summary[field], macro(rows), "Paired delta mismatch")


def format_gap(value):
    number(value)
    return "n/a" if value is None else f"{value:.3f}"


def render_readme(summary, control_count, supplements):
    lines = [BEGIN, "### Verified proposal results", "",
             f"The separate extension contains **{summary['complete']:,}/{summary['scheduled']:,} complete primary "
             f"episodes** and **{summary['noncomplete']:,} noncomplete primary episodes**, "
             f"with {control_count:,} independently replayed controls. All scheduled outcomes remain in the denominator.",
             "", "| Model | Separate zero-attempt supplement completed / scheduled |",
             "| --- | ---: |"]
    for model, (key, rows) in supplements.items():
        lines.append(f"| `{model}` | {summary[key]['supplement']['complete']}/{len(rows)} |")
    lines += ["", "Supplement episodes never replace or merge into primary results. "
              "Gaps are conditional on feasible completions; public percentage-point gaps and "
              "synthetic additive gaps must not be pooled. Graph-macro means average conditions "
              "within graphs first; random-control seeds are averaged before conditions. "
              "Paired contrasts use only matched feasible episodes, not differences of marginal means.",
              "", "**Primary taskwise results; no pooled ranking.** Each A/B/C cell shows the "
              "conditional graph-macro gap followed by complete/scheduled coverage. Lower gaps "
              "are better; a negative C-minus-B delta favors C. Matched coverage is paired "
              "conditions / scheduled conditions, followed by the number of represented graphs. "
              "`n/a` means no eligible completion or pair, never a zero gap."]
    for task in TASKS:
        if task not in summary["panels"]:
            continue
        lines += ["", "<details>",
                  f"<summary><strong>{TASK_TITLES[task]} - gaps and deltas in {TASKS[task]}</strong></summary>",
                  "", "| Model | A gap (complete/scheduled) | B gap (complete/scheduled) | "
                  "C gap (complete/scheduled) | C-minus-B | Matched conditions; graphs |",
                  "| --- | ---: | ---: | ---: | ---: | --- |"]
        for model in summary["panels"][task]["models"]:
            cells = [f"`{identifier(model['model'])}`"]
            for arm in "ABC":
                row = model["arms"][arm]
                cells.append(f"{format_gap(row['gap'])} ({row['complete']}/{row['scheduled']})")
            paired = model["paired_C_minus_B"]
            cells += [format_gap(paired["delta"]),
                      f"{paired['pairs']}/{model['arms']['B']['scheduled']}; {paired['graphs']} graphs"]
            lines.append("| " + " | ".join(cells) + " |")
        lines += ["", "</details>"]
    lines += ["", "[Primary summary](assets/benchmark/proposal-summary.csv) · "
              "[Primary episodes](assets/benchmark/proposal-episodes.csv) · "
              "[Controls](assets/benchmark/proposal-controls.csv) · "
              "[Control summary](assets/benchmark/proposal-control-summary.csv) · "
              "[Paired contrasts](assets/benchmark/proposal-paired.csv) · "
              "[Supplement summary](assets/benchmark/proposal-supplement-summary.csv) · "
              "[Supplement episodes](assets/benchmark/proposal-supplement-episodes.csv) · "
              "[Export provenance](assets/benchmark/proposal-results.json)", END]
    return "\n".join(lines)


def update_readme(text, block):
    require(text.count(BEGIN) == text.count(END) == 1, "README needs one proposal results marker pair")
    start, end = text.index(BEGIN), text.index(END) + len(END)
    require(start < end - len(END), "README markers out of order")
    return text[:start] + block + text[end:]


def export(summary, report, controls, replay, summary_sha256, replay_sha256,
           output, readme=None, supplement=()):
    data, primary, baseline, supplements, hashes = load_verified(
        summary, report, controls, replay, summary_sha256, replay_sha256, supplement)
    result = tables(data, primary, baseline, supplements)
    payloads = {}
    for name, rows in result.items():
        stream = io.StringIO(newline="")
        fields = (list(rows[0]) if rows else
                  list(result["proposal-episodes.csv"][0]) if name == "proposal-supplement-episodes.csv"
                  else ["model", "task", "units", "arm", *COUNTS, "gap"])
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: tuple(str(v) for v in row.values())))
        payloads[name] = stream.getvalue()
    manifest = {
        "schema_version": 1, "protocol_sha256": data["protocol_sha256"], "status": "verified_metrics_export",
        "primary_scheduled": data["scheduled"], "primary_complete": data["complete"],
        "primary_noncomplete": data["noncomplete"], "controls": len(baseline),
        "supplement_scheduled": {model: len(rows) for model, (_, rows) in supplements.items()},
        "supplements_merged_into_primary": False, "source_sha256": hashes,
        "files_sha256": {name: hashlib.sha256(text.encode()).hexdigest()
                         for name, text in payloads.items()},
        "interpretation": {
            "gaps": "Graph-macro conditional on feasible completion; units remain task-specific.",
            "paired": "Matched feasible conditions only; random-control seeds averaged first.",
            "missing": "CSV empty cells and JSON null mean unavailable, never zero.",
            "timing": "Observed durations, not matched-hardware speed comparisons.",
            "privacy": "Allowlisted metrics only; no requests, responses, configs or error details.",
            "historical": "Separate from the earlier TSP-only action-abstraction panel.",
        },
    }
    payloads["proposal-results.json"] = json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    readme_path = Path(readme) if readme else None
    readme_text = update_readme(readme_path.read_text(), render_readme(data, len(baseline), supplements)) if readme_path else None
    output = Path(output)
    require(output.is_dir(), "Output directory must already exist")
    for name, text in payloads.items():
        target = output / name
        require(not target.is_symlink() and (not target.exists() or target.read_text() == text),
                "Refusing to replace an existing different proposal asset")
    for name, text in payloads.items():
        target = output / name
        if not target.exists():
            with target.open("x") as stream:
                stream.write(text)
    if readme_path:
        readme_path.write_text(readme_text)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("summary", "report", "controls", "replay", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--summary-sha256", required=True)
    parser.add_argument("--replay-sha256", required=True)
    parser.add_argument("--supplement", nargs=3, action="append", default=[],
                        metavar=("MODEL", "SUMMARY_KEY", "REPORT"))
    parser.add_argument("--readme", type=Path)
    args = parser.parse_args()
    export(**vars(args))
    print("Verified proposal metrics exported; historical assets unchanged.")


if __name__ == "__main__":
    main()
