import copy
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from src.utils import proposal_export as exporter


class ProposalExportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.output = self.root / "public"
        self.output.mkdir()
        self.secret = "/private/operator/token=do-not-export"
        self.protocol = "a" * 64
        self.rows = []
        for arm, offset in (("A", 0), ("B", 1), ("C", -1)):
            for index, gap in enumerate((2, 4, 10)):
                self.rows.append({
                    "model": "example_model", "task": "tsp_public", "arm": arm,
                    "instance_id": f"case-{index}", "dataset_id": f"graph-{index // 2}",
                    "feasible": True, "terminal_reason": None, "objective": 100 + gap + offset,
                    "percentage_gap": gap + offset, "absolute_gap": None,
                    "model_calls": 1, "model_calls_unknown": False, "confirmed_model_calls": 1,
                    "forced_candidate_steps": 1, "candidate_counts": [2, 1],
                    "raw_request": {"authorization": self.secret}, "failure": self.secret,
                })
        self.rows[1].update(feasible=False, terminal_reason="fatal_lane_abort", objective=None,
                            percentage_gap=None, model_calls=0, confirmed_model_calls=0,
                            forced_candidate_steps=0, candidate_counts=[])
        self.controls = []
        for index in range(3):
            for method, seed, gap in (("classical", 7, 1), ("random_candidate", 8, 0),
                                      ("random_candidate", 9, 2)):
                self.controls.append({
                    "task": "tsp_public", "instance_id": f"case-{index}", "method": method,
                    "seed": seed, "feasible": True, "objective": 100 + gap,
                    "percentage_gap": gap, "private_path": self.secret,
                })
        arms = {}
        for arm, gap, complete, graphs in (("A", 6, 2, 2), ("B", 7.5, 3, 2), ("C", 5.5, 3, 2)):
            arms[arm] = {
                "scheduled": 3, "complete": complete, "unsupported": 0, "failed": 0,
                "not_started": 3 - complete, "gap": gap, "model_calls": complete,
                "confirmed_model_calls": complete, "unknown_call_episodes": 0,
                "forced_steps": complete, "singleton_steps": complete, "observed_steps": complete * 2,
                "graphs_with_completion": graphs,
                "matched_controls": {
                    method: {"model_minus_control": gap - 1, "matched_control_gap": 1,
                             "pairs": complete, "graphs": graphs}
                    for method in ("classical", "random_candidate")},
            }
        self.summary = {
            "protocol_sha256": self.protocol, "scheduled": 9, "complete": 8, "noncomplete": 1,
            "panels": {"tsp_public": {
                "models": [{"model": "example_model", "arms": arms,
                            "paired_C_minus_B": {"delta": -2, "pairs": 3, "graphs": 2}}],
                "controls": {"classical": {"gap": 1, "trajectories": 3, "instances": 3},
                             "random_candidate": {"gap": 1, "trajectories": 6, "instances": 3}},
                "private_path": self.secret,
            }},
            "recovery": {
                "supplement": {"scheduled": 1, "complete": 1, "unsupported": 0, "failed": 0, "not_started": 0},
                "panels": {"tsp_public": {"arms": {
                    arm: {"scheduled": int(arm == "A"), "complete": int(arm == "A"), "unsupported": 0,
                          "failed": 0, "not_started": 0, "gap": 4 if arm == "A" else None} for arm in "ABC"}}},
            },
        }
        self.supplement = {
            "status": "independently_replayed", "primary_rows_unchanged": True,
            "primary_protocol_sha256": self.protocol, "scheduled": 1,
            "terminal_verified": 1, "exercised": 1,
            "rows": [{**self.rows[4], "arm": "A", "percentage_gap": 4, "objective": 104, "exercised": True}],
        }
        self.receipt = {
            "status": "passed", "protocol_sha256": self.protocol,
            "episodes_replayed": 9, "controls_replayed": 9,
            "new_model_calls": 0, "model_initializations": 0,
            "recovery_verification": {
                "status": "passed", "primary_rows_unchanged": True,
                "scheduled": 1, "terminal_verified": 1, "exercised": 1,
            },
        }
        self.readme = self.root / "README.md"
        self.readme.write_text(f"Historical TSP values unchanged\n{exporter.BEGIN}\n{exporter.END}\nExisting edits\n")
        self.sync()

    def write(self, name, value):
        (self.root / name).write_text(json.dumps(value))

    def sync(self):
        self.write("summary.json", self.summary)
        self.write("report.json", {"protocol_sha256": self.protocol, "episodes": self.rows})
        self.write("baselines.json", self.controls)
        self.write("supplement.json", self.supplement)
        self.receipt["report_sha256"] = exporter.digest(self.root / "report.json")
        self.receipt["baseline_sha256"] = exporter.digest(self.root / "baselines.json")
        self.receipt["recovery_verification"]["report_sha256"] = exporter.digest(self.root / "supplement.json")
        self.write("replay.json", self.receipt)
        self.kwargs = {
            "summary": self.root / "summary.json", "report": self.root / "report.json",
            "controls": self.root / "baselines.json", "replay": self.root / "replay.json",
            "summary_sha256": exporter.digest(self.root / "summary.json"),
            "replay_sha256": exporter.digest(self.root / "replay.json"),
            "output": self.output, "readme": self.readme,
            "supplement": [("example_model", "recovery", self.root / "supplement.json")],
        }

    def test_generic_export_deterministic_private_and_separate(self):
        manifest = exporter.export(**self.kwargs)
        before = {p.name: p.read_bytes() for p in self.output.iterdir()}
        exporter.export(**self.kwargs)
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.output.iterdir()})
        self.assertEqual(len(before), 8)
        self.assertEqual(manifest["primary_scheduled"], 9)
        self.assertEqual(manifest["primary_complete"], 8)
        self.assertFalse(manifest["supplements_merged_into_primary"])
        for content in before.values():
            self.assertNotIn(self.secret.encode(), content)
            self.assertNotIn(str(self.root).encode(), content)
            self.assertNotIn(b"raw_request", content)
        for name, sha in manifest["files_sha256"].items():
            self.assertEqual(hashlib.sha256(before[name]).hexdigest(), sha)
        self.assertEqual(len(before["proposal-episodes.csv"].splitlines()), 10)
        self.assertEqual(len(before["proposal-controls.csv"].splitlines()), 10)
        self.assertEqual(len(before["proposal-supplement-episodes.csv"].splitlines()), 2)
        text = self.readme.read_text()
        self.assertTrue(text.startswith("Historical TSP values unchanged\n"))
        self.assertTrue(text.endswith("\nExisting edits\n"))
        self.assertIn("8/9 complete", text)
        self.assertIn("| `example_model` | 6.000 (2/3) | 7.500 (3/3) | 5.500 (3/3) | -2.000 | 3/3; 2 graphs |", text)
        with (self.output / "proposal-summary.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual([float(r["gap"]) for r in rows], [6, 7.5, 5.5])
        with (self.output / "proposal-episodes.csv").open() as stream:
            episodes = list(csv.DictReader(stream))
        self.assertEqual(episodes[1]["terminal_reason"], "fatal_lane_abort")
        self.assertEqual(episodes[0]["terminal_reason"], "")
        self.assertEqual(episodes[0]["model_calls_unknown"], "False")

    def test_cli_updates_assets_and_readme(self):
        args = [sys.executable, "-m", "src.utils.proposal_export"]
        for name, value in self.kwargs.items():
            if name != "supplement":
                args.extend(["--" + name.replace("_", "-"), str(value)])
        args.extend(["--supplement", "example_model", "recovery", str(self.root / "supplement.json")])
        result = subprocess.run(args, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(list(self.output.iterdir())), 8)
        self.assertIn("8/9 complete", self.readme.read_text())

    def test_readme_has_six_taskwise_panels_with_units_and_missing_values(self):
        data = copy.deepcopy(self.summary)
        panel = data["panels"]["tsp_public"]
        data["panels"] = {task: copy.deepcopy(panel) for task in exporter.TASKS}
        model = data["panels"]["maxcut_construct"]["models"][0]
        for arm in model["arms"].values():
            arm.update(gap=None, complete=0, unsupported=3, not_started=0)
        model["paired_C_minus_B"] = {"delta": None, "pairs": 0, "graphs": 0}
        text = exporter.render_readme(data, 9, {})
        self.assertEqual(text.count("<details>"), 6)
        self.assertEqual(text.count("</details>"), 6)
        self.assertEqual(text.count("| Model | A gap"), 6)
        for task, units in exporter.TASKS.items():
            self.assertIn(f"{exporter.TASK_TITLES[task]} - gaps and deltas in {units}", text)
        self.assertIn("| `example_model` | n/a (0/3) | n/a (0/3) | n/a (0/3) | n/a | 0/3; 0 graphs |", text)
        self.assertIn("no pooled ranking", text)
        self.assertNotIn(self.secret, text)

    def test_drift_rejected_before_writing(self):
        for name in ("summary.json", "report.json", "baselines.json", "replay.json", "supplement.json"):
            with self.subTest(name=name):
                path = self.root / name
                original = path.read_bytes()
                path.write_bytes(original + b" ")
                with self.assertRaises(ValueError):
                    exporter.export(**self.kwargs)
                self.assertFalse(list(self.output.iterdir()))
                path.write_bytes(original)

    def test_invalid_missing_nonfinite_and_mismatched_rows(self):
        original = copy.deepcopy(self.rows)
        mutations = [
            lambda: self.rows.pop(),
            lambda: self.rows.append(copy.deepcopy(self.rows[0])),
            lambda: self.rows[0].pop("objective"),
            lambda: self.rows[0].update(percentage_gap=float("nan")),
            lambda: self.rows[0].update(solve_wall_seconds=float("inf")),
            lambda: self.rows[0].update(solve_wall_seconds=-1),
            lambda: self.rows[0].update(candidate_counts=[-1, 1]),
            lambda: self.rows[0].update(model_calls=-1),
            lambda: self.rows[0].update(model_calls=True),
            lambda: self.rows[0].update(model_calls_unknown=True),
            lambda: self.rows[0].update(percentage_gap=99),
            lambda: self.rows[0].update(instance_id=self.secret),
            lambda: self.rows[0].update(dataset_id=self.secret),
            lambda: self.rows[0].update(terminal_reason="in_progress"),
            lambda: self.rows[0].update(terminal_reason="failure"),
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutations.index(mutation)):
                self.rows = copy.deepcopy(original)
                mutation()
                self.sync()
                with self.assertRaises((ValueError, KeyError)):
                    exporter.export(**self.kwargs)
                self.assertFalse(list(self.output.iterdir()))

    def test_mismatched_controls_and_pairs(self):
        self.controls[0]["percentage_gap"] = 50
        self.sync()
        with self.assertRaisesRegex(ValueError, "Control gap mismatch"):
            exporter.export(**self.kwargs)
        self.controls[0]["percentage_gap"] = 1
        self.summary["panels"]["tsp_public"]["models"][0]["paired_C_minus_B"]["delta"] = 0
        self.sync()
        with self.assertRaisesRegex(ValueError, "Paired delta mismatch"):
            exporter.export(**self.kwargs)
        self.assertFalse(list(self.output.iterdir()))

    def test_missing_supplement_and_nonpassed_replay(self):
        with self.assertRaisesRegex(ValueError, "Missing or unexpected supplemental"):
            exporter.export(**{**self.kwargs, "supplement": []})
        self.receipt["status"] = "running"
        self.sync()
        with self.assertRaises(ValueError):
            exporter.export(**self.kwargs)
        self.assertFalse(list(self.output.iterdir()))

    def test_display_and_mismatched_supplements_rejected(self):
        self.summary["purpose"] = "results_display"
        self.sync()
        with self.assertRaisesRegex(ValueError, "Display or foreign"):
            exporter.export(**self.kwargs)
        del self.summary["purpose"]
        self.supplement["rows"][0]["dataset_id"] = "different-graph"
        self.sync()
        with self.assertRaisesRegex(ValueError, "Supplement task/graph"):
            exporter.export(**self.kwargs)
        self.assertFalse(list(self.output.iterdir()))

    def test_refuses_overwrite_and_invalid_readme(self):
        target = self.output / "proposal-summary.csv"
        target.write_text("historical data")
        with self.assertRaises(ValueError):
            exporter.export(**self.kwargs)
        self.assertEqual(target.read_text(), "historical data")
        target.unlink()
        self.readme.write_text("missing markers")
        with self.assertRaises(ValueError):
            exporter.export(**self.kwargs)
        self.assertFalse(list(self.output.iterdir()))

    def test_no_supplements_supported(self):
        del self.summary["recovery"]
        self.sync()
        manifest = exporter.export(**{**self.kwargs, "supplement": []})
        self.assertEqual(manifest["supplement_scheduled"], {})
        self.assertEqual(len((self.output / "proposal-supplement-episodes.csv").read_text().splitlines()), 1)

    def test_terminal_diagnostics_preserve_full_values(self):
        for reason in exporter.TERMINAL_REASONS:
            with self.subTest(reason=reason):
                clean = exporter.episode({**self.rows[1], "terminal_reason": reason}, "primary")
                self.assertEqual(clean["terminal_reason"], reason)
        self.assertIsNone(exporter.episode(self.rows[0], "primary")["terminal_reason"])

    def test_unknown_and_nonterminal_reasons_fail_with_sanitized_errors(self):
        for reason in (self.secret, "transport_failure", "preflight_failed", "complete", "",
                       "in_progress", "unattempted", None, {"private": self.secret}):
            with self.subTest(reason=reason), self.assertRaises(ValueError) as caught:
                exporter.episode({**self.rows[1], "terminal_reason": reason}, "primary")
            self.assertNotIn(self.secret, str(caught.exception))

    def test_unknown_calls_forwards_and_transport_without_raw_response_are_preserved(self):
        row = {**self.rows[1], "terminal_reason": "http_error",
               "model_calls": None, "model_calls_unknown": True, "confirmed_model_calls": 1,
               "native_forward_count": None, "confirmed_native_forwards": 0,
               "native_forward_unknown_attempts": 1, "captured_response_records": 0}
        clean = exporter.episode(row, "primary")
        for key in ("model_calls", "model_calls_unknown", "confirmed_model_calls", "native_forward_count",
                    "confirmed_native_forwards", "native_forward_unknown_attempts", "captured_response_records"):
            self.assertEqual(clean[key], row[key])
        self.assertIsNone(clean["objective"])
        self.assertIsNone(clean["percentage_gap"])


if __name__ == "__main__":
    unittest.main()
