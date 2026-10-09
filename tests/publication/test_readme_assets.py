"""Check reproducible README assets against the frozen fourteen-model aggregates."""

import hashlib
import json
import math
from pathlib import Path
import re
import unittest
import xml.etree.ElementTree as ET

from src.utils import render_results as renderer


REPO = Path(__file__).resolve().parents[2]
ASSETS = REPO / "assets/benchmark"
DATA = REPO / "data/results"


class ReadmeAssetsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = json.loads((DATA / "summary.json").read_text())

    def test_generated_files_match_and_svgs_are_self_contained(self):
        generated = renderer.build(self.data)
        self.assertEqual(len(generated), 5)
        self.assertEqual(set(generated), {
            ASSETS / "overview.svg", ASSETS / "structural-queries.svg",
            ASSETS / "graph-text-decisions.svg", ASSETS / "optimization-trajectories.svg",
            REPO / "docs/results.md",
        })
        for path, content in generated.items():
            with self.subTest(path=path.name):
                self.assertEqual(path.read_text(), content)
                if path.suffix != ".svg":
                    continue
                root = ET.fromstring(content)
                ns = {"s": "http://www.w3.org/2000/svg"}
                self.assertTrue(root.find("s:title", ns).text)
                self.assertTrue(root.find("s:desc", ns).text)
                self.assertFalse(root.findall(".//s:script", ns))
                self.assertFalse(root.findall(".//s:foreignObject", ns))
                self.assertFalse(root.findall(".//s:image", ns))
                self.assertNotIn("http", content.split(">", 1)[1])

    def test_all_model_aggregates_match_frozen_publication_snapshot(self):
        # This is a preservation check, not independent raw-output recomputation.
        encoded = json.dumps(self.data["models"], sort_keys=True, separators=(",", ":")).encode()
        self.assertEqual(hashlib.sha256(encoded).hexdigest(),
                         "e68145619b535824b0f9a164f52508894053b9ad2295ab5f7974ca366b1dd18c")

    def test_aggregate_shapes_ranges_and_completion_semantics(self):
        models = self.data["models"]
        self.assertEqual(len({model["id"] for model in models}), 14)
        self.assertEqual(len(self.data["rq1_tasks"]), 6)
        self.assertEqual(self.data["rq2_conditions"], ["T", "G", "TG", "BAG", "A"])
        self.assertEqual(len(self.data["rq3_tasks"]), 4)
        for model in models:
            with self.subTest(model=model["id"]):
                for field, count in (("rq1", 6), ("arxiv", 5), ("prime", 5)):
                    self.assertEqual(len(model[field]), count)
                    for value in model[field]:
                        if value is None:
                            self.assertNotEqual(field, "rq1")
                        else:
                            self.assertTrue(math.isfinite(value) and 0 <= value <= 100)
                self.assertEqual(len(model["rq3"]), 4)
                for row in model["rq3"]:
                    self.assertEqual(len(row), 4)
                    for gap, completed in zip(row[:2], row[2:]):
                        self.assertIs(type(completed), int)
                        self.assertTrue(0 <= completed <= 10)
                        if completed == 0:
                            self.assertIsNone(gap)
                        else:
                            self.assertIsNotNone(gap)
                            self.assertTrue(math.isfinite(gap))

    def test_public_bundle_documents_aggregate_only_scope(self):
        self.assertEqual({path.name for path in DATA.iterdir()}, {"README.md", "summary.json"})
        for relative in ("README.md", "data/README.md", "data/results/README.md",
                         "docs/reproduction.md", "docs/results.md"):
            text = (REPO / relative).read_text()
            with self.subTest(document=relative):
                self.assertNotIn("src.utils.public_results", text)
                self.assertNotIn("tests.publication.test_public_results", text)
                for name in ("structural-queries.csv", "graph-text-decisions.csv",
                             "optimization-trajectories.csv"):
                    self.assertNotIn(name, text)
        self.assertNotIn("distributed final rows", self.data["sources"]["note"])

    def test_all_models_coverage_and_missing_values_are_preserved(self):
        models = self.data["models"]
        self.assertEqual(len(models), 14)
        self.assertEqual(sum(row[2] + row[3] for m in models for row in m["rq3"]), 951)
        self.assertEqual([m["id"] for m in models if m["arxiv"][0] is None], ["laya", "qwen4score"])
        jev = next(m for m in models if m["id"] == "jev")
        self.assertEqual(jev["rq3"][2][2:], [8, 8])
        laya = next(m for m in models if m["id"] == "laya")
        self.assertEqual(laya["rq3"][2], [None, 0, 0, 1])
        chart = (ASSETS / "optimization-trajectories.svg").read_text()
        self.assertIn("stroke-dasharray", chart)
        self.assertIn("0.00 [1]", chart)

    def test_relative_gap_definition_and_modularity_units_are_explicit(self):
        for relative in ("README.md", "data/results/README.md", "docs/results.md"):
            text = (REPO / relative).read_text()
            with self.subTest(document=relative):
                self.assertIn("|f_i-R_i|", text)
                self.assertIn(r"\times100", text)
                self.assertIn("Q_i^*-Q_i", text)
                self.assertNotIn("Signed gaps are not clamped", text)
        for model in self.data["models"]:
            for row in model["rq3"][:3]:
                for gap in row[:2]:
                    if gap is not None:
                        self.assertGreaterEqual(gap, 0)

    def test_document_links(self):
        for relative in ("README.md", "data/README.md", "docs/results.md", "docs/reproduction.md",
                 "data/results/README.md", "configs/README.md", "tests/README.md"):
            path = REPO / relative
            content = path.read_text()
            links = re.findall(r'\]\(([^)]+)\)|(?:href|src)="([^"]+)"', content)
            for pair in links:
                link = next(v for v in pair if v)
                if link.startswith(("https://", "http://")):
                    continue
                target, _, anchor = link.partition("#")
                target_path = path.parent / target if target else path
                self.assertTrue(target_path.exists(), (relative, link))
                if anchor:
                    headings = re.findall(r"^#+ (.+)$", target_path.read_text(), re.M)
                    slugs = [re.sub(r"[^\w -]", "", heading.lower()).replace(" ", "-")
                             for heading in headings]
                    self.assertIn(anchor, slugs, (relative, link))


if __name__ == "__main__":
    unittest.main()
