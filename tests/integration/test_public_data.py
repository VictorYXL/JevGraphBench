"""Offline public-data preparation tests: synthetic byte fixtures, no network."""

from dataclasses import asdict
import gzip
import hashlib
import importlib
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import httpx

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tests.support import repo_workspace
from src.benchmark import public_data as data
from src.benchmark import public_tasks as core


TSP = b"""NAME : tiny
TYPE : TSP
COMMENT : synthetic offline fixture
DIMENSION : 3
EDGE_WEIGHT_TYPE : EUC_2D
NODE_COORD_SECTION
3 0 4
1 0 0
2 2.5 0
EOF
"""
CUT = b"3 3\n3 1 -2\n1 2 1\n2 3 4\n"


def digest(content):
    return hashlib.sha256(content).hexdigest()


def entry(name, raw, task="tsp_public", role="main", compressed=True):
    wire = gzip.compress(raw, mtime=0) if compressed else raw
    row = {
        "id": name, "task": task, "role": role, "dimension": 3, "edge_count": 3,
        "source_url": f"https://public.example/{name}",
        "download_url": f"https://public.example/{name}.data",
        "compression": "gzip" if compressed else "none",
        "wire_sha256": digest(wire), "raw_sha256": digest(raw),
        "reference": {"value": 123456789, "status": "best_known",
                      "source": "https://private.example/reference-secret"},
        "reference_as_of": "2016-08-16",
        "reference_note": "historical reference-secret, not proven optimal",
    }
    if task == "tsp_public":
        row["edge_weight_type"] = "EUC_2D"
    return row, wire


class PublicDataTests(unittest.TestCase):
    def setUp(self):
        self.root = repo_workspace(self, "public-data-")
        self.catalog = self.root / "catalog.json"
        self.output = self.root / "prepared"
        self.entries, self.downloads = [], {}
        for args in [
            ("tiny", TSP),
            ("signed", CUT, "maxcut_public", "main", False),
            ("calibration", TSP, "tsp_public", "calibration"),
        ]:
            row, wire = entry(*args)
            self.entries.append(row)
            self.downloads[row["download_url"]] = wire
        self.save_catalog()
        self.no_network = patch.object(httpx.Client, "send", side_effect=AssertionError("network forbidden"))
        self.no_network.start()
        self.addCleanup(self.no_network.stop)

    def save_catalog(self):
        self.catalog.write_text(json.dumps({
            "schema_version": 1, "instances": self.entries,
            "license_note": "No inferred redistribution license.",
        }))

    def prepare(self, **kwargs):
        return data.prepare(self.output, self.catalog, fetch=self.downloads.__getitem__, **kwargs)

    def replace_tsp(self, raw):
        row, wire = entry("tiny", raw)
        self.entries[0] = row
        self.downloads[row["download_url"]] = wire
        self.save_catalog()

    def test_catalog_has_only_verified_bounded_selection(self):
        catalog, _ = data._catalog(data.DEFAULT_CATALOG)
        rows = catalog["instances"]
        self.assertEqual(len(rows), 18)
        self.assertEqual(sum(row["role"] == "main" for row in rows), 17)
        self.assertEqual([row["id"] for row in rows if row["role"] == "calibration"], ["eil51"])
        tsps = [row for row in rows if row["task"] == "tsp_public"]
        cuts = [row for row in rows if row["task"] == "maxcut_public"]
        self.assertEqual(len(tsps), 8)
        self.assertEqual(len(cuts), 10)
        self.assertTrue(all(row["dimension"] < 255 and row["edge_weight_type"] == "EUC_2D" for row in tsps))
        for row in cuts:
            self.assertEqual((row["dimension"], row["edge_count"]), (125, 375))
            self.assertIn("/9c9c1a84774ac14b7c161d5d5ea1f4f554a39a76/", row["download_url"])
            self.assertFalse(row["download_url"].endswith(".zip"))
            self.assertEqual(row["reference"]["status"], "best_known")
            self.assertEqual(row["reference_as_of"], "2016-08-16")
        self.assertIn("best known", catalog["maxcut_reference_provenance"]["readme_quote"])

    def test_import_does_not_download(self):
        with patch.object(httpx, "Client", side_effect=AssertionError("implicit client")):
            importlib.reload(data)

    def test_default_main_calibration_separate_and_manifest_hashes(self):
        manifest = self.prepare(include_calibration=True)
        main = json.loads((self.output / "instances.json").read_text())
        calibration = json.loads((self.output / "calibration-instances.json").read_text())
        self.assertEqual([r["id"] for r in main], ["tiny", "signed"])
        self.assertEqual([r["id"] for r in calibration], ["calibration"])
        self.assertEqual(manifest, json.loads((self.output / "provenance.json").read_text()))
        for name, details in manifest["files"].items():
            content = (self.output / name).read_bytes()
            self.assertEqual(digest(content), details["sha256"])
            self.assertEqual(len(content), details["bytes"])
        self.assertEqual((self.output / "raw/tiny.tsp").read_bytes(), TSP)
        self.assertEqual(main[1]["state"]["edges"], [[0, 1, 1], [0, 2, -2], [1, 2, 4]])
        self.assertEqual(main[0]["state"]["coordinates"], [[0.0, 0.0], [2.5, 0.0], [0.0, 4.0]])
        for instance in main + calibration:
            core.validate_instance(instance)
            request = core.next_request(instance, [])
            text = json.dumps(asdict(request))
            self.assertNotIn("reference-secret", text)
            self.assertNotIn("123456789", text)
            self.assertNotIn("license_note", text)
        request = core.next_request(main[0], [])
        self.assertEqual([option.id for option in request.options], ["1", "2"])
        self.assertEqual(request.state["current_city_distances"], [[1, 3], [2, 4]])

    def test_only_selected_files_downloaded_and_no_default_calibration(self):
        calls = []
        def fetch(url):
            calls.append(url)
            return self.downloads[url]
        data.prepare(self.output, self.catalog, dataset_ids=["signed"], fetch=fetch)
        self.assertEqual(calls, [self.entries[1]["download_url"]])
        self.assertFalse((self.output / "calibration-instances.json").exists())
        self.assertEqual([r["id"] for r in json.loads((self.output / "instances.json").read_text())], ["signed"])

    def test_existing_output_even_empty_is_never_overwritten(self):
        self.output.mkdir()
        with self.assertRaises(FileExistsError), patch.object(data, "fetch_public_bytes", side_effect=AssertionError):
            data.prepare(self.output, self.catalog)
        self.assertEqual(list(self.output.iterdir()), [])

    def test_no_implicit_overwrite_on_second_call(self):
        self.prepare()
        before = (self.output / "instances.json").read_bytes()
        with self.assertRaises(FileExistsError):
            self.prepare()
        self.assertEqual(before, (self.output / "instances.json").read_bytes())

    def test_symlink_output_ancestor_and_dangling_link_rejected(self):
        target = self.root / "target"
        target.mkdir()
        for name, destination in [("link", target), ("dangling", self.root / "missing")]:
            link = self.root / name
            link.symlink_to(destination, target_is_directory=True)
            for output in [link, link / "new"]:
                with self.subTest(output=output), self.assertRaises(ValueError):
                    data.prepare(output, self.catalog, fetch=self.downloads.__getitem__)
        with self.assertRaises(ValueError):
            data.prepare(self.root / "target" / ".." / "prepared", self.catalog)

    def test_wire_hash_mismatch_and_raw_hash_mismatch_leave_no_output(self):
        for field in ["wire_sha256", "raw_sha256"]:
            with self.subTest(field=field):
                old = self.entries[0][field]
                self.entries[0][field] = "0" * 64
                self.save_catalog()
                with self.assertRaisesRegex(ValueError, "SHA256"):
                    self.prepare()
                self.assertFalse(self.output.exists())
                self.entries[0][field] = old

    def test_verified_gzip_bytes_still_must_be_valid_gzip(self):
        bad = b"not gzip"
        self.entries[0]["wire_sha256"] = digest(bad)
        self.downloads[self.entries[0]["download_url"]] = bad
        self.save_catalog()
        with self.assertRaisesRegex(ValueError, "gzip"):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_decompressed_size_is_bounded(self):
        huge = b"x" * (data.MAX_FILE_BYTES + 1)
        row, wire = entry("tiny", huge)
        with self.assertRaisesRegex(ValueError, "size limit"):
            data._unpack(wire, row)

    def test_tsp_corruption_rejected(self):
        bad = [
            TSP.replace(b"DIMENSION : 3", b"DIMENSION : 4"),
            TSP.replace(b"EUC_2D", b"GEO"),
            TSP.replace(b"TYPE : TSP", b"TYPE : ATSP"),
            TSP.replace(b"2 2.5 0", b"1 2.5 0"),
            TSP.replace(b"3 0 4", b"4 0 4"),
            TSP.replace(b"3 0 4\n", b""),
            TSP.replace(b"3 0 4", b"3 nan 4"),
            TSP.replace(b"3 0 4", b"3 inf 4"),
            TSP.replace(b"3 0 4", b"3 0"),
            TSP.replace(b"EOF", b"EOF\n4 0 1"),
            TSP.replace(b"TYPE : TSP", b"TYPE : TSP\nTYPE : TSP"),
            TSP.replace(b"NODE_COORD_SECTION", b"NODE_COORD_SECTION\nNODE_COORD_SECTION"),
        ]
        for raw in bad:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                data.parse_tsplib_euc_2d(raw, 3)
        for n in [2, 255, True]:
            with self.assertRaises(ValueError):
                data.parse_tsplib_euc_2d(TSP, n)

    def test_corrupt_but_correctly_hashed_graph_never_freezes(self):
        self.replace_tsp(TSP.replace(b"2 2.5 0", b"1 2.5 0"))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_coordinate_export_preserves_double_sqrt_rounding_convention(self):
        raw = (TSP.replace(b"1 0 0", b"1 347.42 278.65")
               .replace(b"2 2.5 0", b"2 461.42 193.15")
               .replace(b"3 0 4", b"3 347.42 250.15"))
        self.replace_tsp(raw)
        self.prepare(dataset_ids=["tiny"])
        instance = json.loads((self.output / "instances.json").read_text())[0]
        self.assertEqual(instance["state"]["coordinates"],
                         [[347.42, 278.65], [461.42, 193.15], [347.42, 250.15]])
        core.validate_instance(instance)
        request = core.next_request(instance, [])
        # The first pair is original tsp225 75--111: hypot would give 142.
        # The second pair rounds to 28 using source doubles, not Decimal's 29.
        self.assertEqual(request.state["current_city_distances"], [[1, 143], [2, 28]])

    def test_maxcut_corruption_never_repaired(self):
        bad = [
            b"3 2\n1 2 1\n2 3 -1\n",
            b"4 3\n1 2 1\n2 3 1\n1 3 -1\n",
            b"3 3\n1 2 1\n2 1 -1\n2 3 1\n",
            b"3 3\n1 1 1\n1 2 -1\n2 3 1\n",
            b"3 3\n0 1 1\n1 2 -1\n2 3 1\n",
            b"3 3\n1 4 1\n1 2 -1\n2 3 1\n",
            b"3 3\n1 3 0.5\n1 2 -1\n2 3 1\n",
            b"3 3\n1 3 1\n1 2 -1\n",
            b"3 3\n1 3 1\n1 2 -1\n2 3 1\n1 2 1\n",
        ]
        for raw in bad:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                data.parse_maxcut(raw, 3, 3)
        result = data.parse_maxcut(b"3 1\n1 2 -7\n", 3, 1)
        self.assertEqual(result["nodes"], [0, 1, 2])
        self.assertEqual(result["edges"], [[0, 1, -7]])

    def test_bad_catalog_selection_or_id_never_downloads(self):
        for selection in [[], ["unknown"], ["calibration"], ["tiny", "tiny"], "tiny"]:
            with self.subTest(selection=selection), self.assertRaises(ValueError):
                self.prepare(dataset_ids=selection)
        self.entries[0]["id"] = "../escape"
        self.save_catalog()
        with self.assertRaisesRegex(ValueError, "catalog ID"):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_bad_source_url_and_reference_rejected_before_fetch(self):
        for field, value in [("download_url", "http://example.org/data"),
                             ("download_url", "https://user:secret@example.org/data"),
                             ("source_url", "file:///etc/passwd")]:
            original = self.entries[0][field]
            self.entries[0][field] = value
            self.save_catalog()
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.prepare()
            self.entries[0][field] = original
        self.entries[0]["reference"]["status"] = "proven"
        self.save_catalog()
        with self.assertRaises(ValueError):
            self.prepare()

    def test_network_failure_does_not_create_partial_output(self):
        def fail(url):
            raise httpx.ConnectError("offline")
        with self.assertRaises(httpx.ConnectError):
            data.prepare(self.output, self.catalog, fetch=fail)
        self.assertFalse(self.output.exists())

    def test_https_downloader_redirects_and_limits_offline(self):
        self.no_network.stop()
        original_client = httpx.Client
        def run(handler, limit=data.MAX_FILE_BYTES):
            transport = httpx.MockTransport(handler)
            def factory(**kwargs):
                self.assertFalse(kwargs["trust_env"])
                self.assertFalse(kwargs["follow_redirects"])
                return original_client(transport=transport, **kwargs)
            with patch.object(data.httpx, "Client", side_effect=factory), patch.object(data, "MAX_FILE_BYTES", limit):
                return data.fetch_public_bytes("https://public.example/start")
        def redirect(request):
            if request.url.path == "/start":
                return httpx.Response(302, headers={"location": "/finish"})
            return httpx.Response(200, content=b"verified")
        self.assertEqual(run(redirect), b"verified")
        with self.assertRaisesRegex(ValueError, "HTTPS"):
            run(lambda request: httpx.Response(302, headers={"location": "http://public.example/file"}))
        with self.assertRaisesRegex(ValueError, "size limit"):
            run(lambda request: httpx.Response(200, content=b"12345"), limit=4)
        with self.assertRaisesRegex(ValueError, "too many"):
            run(lambda request: httpx.Response(302, headers={"location": "/start"}))


if __name__ == "__main__":
    unittest.main()
