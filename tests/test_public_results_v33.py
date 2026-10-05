"""Offline manuscript-regression tests using the distributed final-result CSVs."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from src.utils import public_results_v33 as results


DATA = Path(__file__).resolve().parents[1] / "assets/benchmark/v33/gpt54"


class PublicResultsV33Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rq1 = results.read_csv(DATA / "rq1.csv")
        cls.rq2 = results.read_csv(DATA / "rq2.csv")
        cls.rq3 = results.read_csv(DATA / "rq3.csv")

    def test_rq1_manuscript_aggregations_are_distinct(self):
        report = results.rq1_summary(self.rq1)
        all_sources = report["all"]
        self.assertEqual((all_sources["correct"], all_sources["valid"]), (733, 800))
        self.assertEqual(all_sources["query_weighted_percent"], 91.625)
        self.assertAlmostEqual(all_sources["task_macro_percent"], 92.04166666666667)
        self.assertAlmostEqual(all_sources["task_size_macro_percent"], 92.11090067340068)
        degree = all_sources["by_task"]["degree_exact"]
        self.assertEqual(degree["query_weighted_percent"], 91)
        self.assertAlmostEqual(degree["size_equal_percent"], 91.41540404040404)
        self.assertEqual([report[s]["scheduled"] for s in results.SOURCES], [200] * 4)

    def test_rq1_rejects_duplicate_wrong_quota_and_illegal_correct(self):
        for change, error in (
            (lambda rows: rows.__setitem__(0, rows[1]), "unique"),
            (lambda rows: rows[0].update(node_count="13"), "stratum"),
            (lambda rows: rows[0].update(correct="1", output_legal="0"), "illegal"),
            (lambda rows: rows[0].update(correct="True"), "0 or 1"),
        ):
            rows = deepcopy(self.rq1)
            change(rows)
            with self.subTest(error=error), self.assertRaisesRegex(ValueError, error):
                results.rq1_summary(rows)

    def test_rq2_paired_intervals_match_paper_not_marginal_counts(self):
        report = results.rq2_summary(self.rq2)
        expected = {
            "arxiv": [(1, [-2, 4]), (13, [3, 22]), (0, [-6, 5]), (12, [4, 20])],
            "prime": [(-3, [-11, 4]), (-5, [-13, 3]), (-7, [-15, 1]), (2, [-4, 8])],
        }
        for dataset, pairs in expected.items():
            for (left, right), (difference, interval) in zip(results.CONTRASTS, pairs):
                value = report[dataset]["paired"][f"{left}-{right}"]
                self.assertAlmostEqual(value["difference_pp"], difference)
                self.assertEqual(value["ci95_pp"], interval)
                self.assertEqual(value["paired_queries"], 100)
        self.assertEqual(report["prime"]["by_condition"]["TG"]["invalid"], 2)
        self.assertEqual(report["prime"]["by_condition"]["TG"]["correct"], 53)
        self.assertEqual(report, results.rq2_summary(list(reversed(self.rq2))))

    def test_rq2_rejects_duplicate_missing_and_noncanonical_pairs(self):
        rows = deepcopy(self.rq2)
        rows[0] = rows[1]
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            results.rq2_summary(rows)
        with self.assertRaisesRegex(ValueError, "1000"):
            results.rq2_summary(self.rq2[:-1])
        rows = deepcopy(self.rq2)
        rows[0]["query_id"] = "0" + rows[0]["query_id"]
        with self.assertRaisesRegex(ValueError, "numeric"):
            results.rq2_summary(rows)

    def test_rq3_references_calls_and_full_ten_graph_tsp_mean(self):
        report = results.rq3_summary(self.rq3)
        self.assertAlmostEqual(report["tsp_public"]["by_mode"]["A"]["mean_gap"], 29.48587045552236)
        self.assertAlmostEqual(report["community_bipartition_construct"]["by_mode"]["A"]["mean_gap"],
                               0.40423984558168824)
        self.assertAlmostEqual(report["tsp_public"]["paired_C_minus_A"], 2.758173943251803)
        self.assertEqual(sum(m["model_calls"] for t in report.values() for m in t["by_mode"].values()),
                         5878)
        self.assertTrue(all(t["paired_graphs"] == 10 for t in report.values()))

    def test_rq3_failures_keep_denominator_without_imputed_gap(self):
        rows = deepcopy(self.rq3)
        row = next(r for r in rows if r["task"] == "tsp_public" and r["mode"] == "A")
        row.update(feasible="0", output_legal="0", objective="", gap="", final_status="incomplete")
        report = results.rq3_summary(rows)["tsp_public"]
        self.assertEqual(report["by_mode"]["A"]["scheduled"], 10)
        self.assertEqual(report["by_mode"]["A"]["feasible"], 9)
        self.assertEqual(report["paired_graphs"], 9)
        row["objective"] = "0"
        with self.assertRaisesRegex(ValueError, "imputed"):
            results.rq3_summary(rows)

    def test_rq3_rejects_wrong_gap_units_stale_reference_and_duplicates(self):
        for field, value, error in (("gap", "NaN", "Nonfinite"),
                                    ("gap", "123", "gap mismatch"),
                                    ("gap_unit", "percent", "Modularity"),
                                    ("reference_status", "heuristic", "Modularity"),
                                    ("model_calls", "-1", "Negative")):
            rows = deepcopy(self.rq3)
            rows[0][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, error):
                results.rq3_summary(rows)
        rows = deepcopy(self.rq3)
        rows[0] = rows[1]
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            results.rq3_summary(rows)
        rows = deepcopy(self.rq3)
        rows[1]["reference"] = str(float(rows[1]["reference"]) + 0.01)
        rows[1]["gap"] = str(float(rows[1]["gap"]) + 0.01)
        with self.assertRaisesRegex(ValueError, "references differ"):
            results.rq3_summary(rows)

    def test_csv_and_cli_are_offline_and_strict(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "bad.csv"
            for content in ("a,a\n1,2\n", "a,b\n1\n", "a,b\n1,2,3\n", ""):
                path.write_text(content)
                with self.assertRaises(ValueError):
                    results.read_csv(path)
        process = subprocess.run(
            [sys.executable, "-B", "-m", "src.utils.public_results_v33", "--data-dir", str(DATA)],
            check=True, capture_output=True, text=True, cwd=DATA.parents[3])
        report = json.loads(process.stdout)
        self.assertFalse(report["inference_performed"])
        self.assertEqual(report["rq1"]["all"]["correct"], 733)


if __name__ == "__main__":
    unittest.main()
