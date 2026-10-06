"""Check reproducible README assets and their link to the final v33 result rows."""

import importlib.util
import json
from pathlib import Path
import re
import unittest
import xml.etree.ElementTree as ET

import yaml

from src.utils.public_results_v33 import report


REPO = Path(__file__).resolve().parents[1]
ASSETS = REPO / "assets/benchmark/v33"
spec = importlib.util.spec_from_file_location("readme_renderer", ASSETS / "render.py")
renderer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(renderer)


class ReadmeAssetsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = json.loads((ASSETS / "display-data.json").read_text())

    def test_generated_files_match_and_svgs_are_self_contained(self):
        generated = renderer.build(self.data)
        self.assertEqual(len(generated), 5)
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

    def test_displayed_gpt54_values_match_independent_row_report(self):
        results = report(ASSETS / "gpt54")
        model = next(m for m in self.data["models"] if m["id"] == "gpt54")
        tasks = ["adjacency", "degree_exact", "cycle_detection", "pair_connectivity",
                 "distance_threshold", "articulation_point"]
        for task, value in zip(tasks, model["rq1"]):
            self.assertAlmostEqual(value, results["rq1"]["all"]["by_task"][task]["size_equal_percent"],
                                   delta=0.00005)
        for dataset in ("arxiv", "prime"):
            for arm, value in zip(self.data["rq2_conditions"], model[dataset]):
                self.assertAlmostEqual(value, results["rq2"][dataset]["by_condition"][arm]["score_percent"])
        tasks = ["tsp_public", "maxcut_public", "lt_influence_construct", "community_bipartition_construct"]
        for i, (task, row) in enumerate(zip(tasks, model["rq3"])):
            for j, arm in enumerate(("A", "C")):
                values = results["rq3"][task]["by_mode"][arm]
                self.assertAlmostEqual(row[j], values["mean_gap"], delta=5e-8 if i == 3 else 0.00005)
                self.assertEqual(row[j + 2], values["feasible"])

    def test_all_models_coverage_and_missing_values_are_preserved(self):
        models = self.data["models"]
        self.assertEqual(len(models), 14)
        self.assertEqual(sum(row[2] + row[3] for m in models for row in m["rq3"]), 951)
        self.assertEqual([m["id"] for m in models if m["arxiv"][0] is None], ["laya", "qwen4score"])
        jev = next(m for m in models if m["id"] == "jev")
        self.assertEqual(jev["rq3"][2][2:], [8, 8])
        laya = next(m for m in models if m["id"] == "laya")
        self.assertEqual(laya["rq3"][2], [None, 0, 0, 1])
        chart = (ASSETS / "rq3.svg").read_text()
        self.assertIn("stroke-dasharray", chart)
        self.assertIn("0.00 [1]", chart)

    def test_document_links_and_citation_metadata(self):
        for relative in ("README.md", "docs/results.md", "docs/reproduction.md"):
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
        citation = yaml.safe_load((REPO / "CITATION.cff").read_text())
        self.assertEqual(citation["cff-version"], "1.2.0")
        preferred = citation["preferred-citation"]
        self.assertEqual(preferred["type"], "unpublished")
        self.assertEqual(preferred["year"], 2026)
        self.assertEqual([a["family-names"] for a in preferred["authors"]], ["Yang", "Zhang", "Zhao"])
        self.assertNotIn("doi", preferred)
        self.assertIn(preferred["title"], (REPO / "README.md").read_text())


if __name__ == "__main__":
    unittest.main()
