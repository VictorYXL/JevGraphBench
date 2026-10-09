"""Reporting-only baseline aggregation does not inflate random-seed sample size."""

from pathlib import Path
import sys
import unittest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.utils.tsp_abstraction_report import baseline_tables


class ReportingTests(unittest.TestCase):
    def test_seeds_average_before_conditions_and_graphs(self):
        rows = []
        for replicate, values in enumerate(([2, 8], [20, 20], [30])):
            for seed, value in enumerate(values):
                rows.append({
                    "method": "random_candidate", "instance_id": f"i{replicate}", "dataset_id": "graph",
                    "replicate": replicate, "seed": seed, "feasible": True, "objective": value,
                    "percentage_gap": value, "model_calls": 0, "solve_wall_seconds": 1,
                    "setup_seconds": 0, "candidate_seconds": .1, "verification_seconds": .2,
                    "forced_candidate_steps": 1,
                })
        episodes, conditions, graphs = baseline_tables(rows)
        self.assertEqual(len(episodes), 5)
        self.assertEqual(len(conditions), 3)
        self.assertEqual([row["gap_percent"] for row in conditions], [5, 20, 30])
        self.assertEqual(graphs[0]["gap_percent"], 55 / 3)
        self.assertEqual(graphs[0]["conditions"], 3)


if __name__ == "__main__":
    unittest.main()
