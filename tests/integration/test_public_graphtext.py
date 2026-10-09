"""Independent, offline invariants for the public RQ2 migration."""

import asyncio
from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.clients.base import BaseDecisionClient, ClientCapabilities, DecisionResponse, InvalidResponseError
from src.utils import public_graphtext as suite
from tests.support import external_fixture


class FakeClient(BaseDecisionClient):
    def __init__(self):
        self.calls = 0

    @property
    def capabilities(self):
        return ClientCapabilities()

    async def predict(self, request):
        self.calls += 1
        return DecisionResponse(request.request_id, request.options[0].id, "fixture",
                                raw_output={"selected": request.options[0].id})


class PublicGraphtextTests(unittest.TestCase):
    def fixture(self):
        nodes = {n: {"name": f"entity {n}", "type": "gene/protein",
                     "details": {"summary": "long intrinsic words " * 1000,
                                 "neighbors": {"named_group": ["SECRET_EDGE"]}}}
                 for n in range(44)}
        documents = {n: suite.intrinsic(info) for n, info in nodes.items()}
        adjacency = {n: [(str(n), "ppi", str(40 + n % 4))] for n in range(40)}
        record = suite.evidence_record({"id": 5, "query": "native query"},
                                       list(range(40)), [1.0] * 40, nodes, documents, adjacency)
        return record

    def test_bm25_full_corpus_zero_hits_and_tie_order(self):
        index = suite.BM25({n: "alpha beta" if n % 2 else "alpha" for n in range(50)})
        ids, scores = index.top("unseen")
        self.assertEqual(ids, list(range(40)))
        self.assertEqual(scores, [0] * 40)
        ids, _ = index.top("beta")
        self.assertEqual(ids[:25], list(range(1, 50, 2)))
        self.assertEqual(ids[25:], list(range(0, 30, 2)))
        self.assertEqual(index.top("beta beta"), index.top("beta"))

    def test_bm25_formula_independent_reference(self):
        import math
        docs = {0: "cat cat dog", 1: "cat", 2: "dog"}
        ids, scores = suite.BM25(docs).top("cat", 3)
        n, average = 3, 5 / 3
        idf = math.log(1 + (n - 2 + .5) / (2 + .5))
        reference = {
            0: idf * 2 * 2.2 / (2 + 1.2 * (.25 + .75 * 3 / average)),
            1: idf * 2.2 / (1 + 1.2 * (.25 + .75 / average)),
            2: 0,
        }
        self.assertEqual(ids, sorted(docs, key=lambda node: (-reference[node], node)))
        for node, score in zip(ids, scores):
            self.assertAlmostEqual(score, reference[node])

    def test_flat_bag_removes_only_explicit_edges(self):
        record = self.fixture()
        requests = {a: suite.prime_request(record, a) for a in suite.ARMS}
        tg = requests["TG"].state
        self.assertEqual(requests["BAG"].state, {k: v for k, v in tg.items() if k != "relations"})
        self.assertTrue(tg["relations"])
        for request in requests.values():
            self.assertEqual(request.options, requests["TG"].options)
            self.assertEqual(request.state["candidate_nodes"], tg["candidate_nodes"])
            self.assertLessEqual(len(suite.canonical(asdict(request))), suite.REQUEST_CHAR_CAP)
            self.assertNotIn("SECRET_EDGE", suite.canonical(asdict(request)))
        self.assertEqual(set(requests["A"].state), {"query", "candidate_ids", "candidate_nodes"})
        self.assertNotIn("texts", requests["G"].state)
        self.assertNotIn("context_nodes", requests["T"].state)
        self.assertNotIn("relations", requests["T"].state)
        self.assertEqual(requests["T"].state["texts"], record["candidate_texts"])
        self.assertTrue(all(set(node) == {"id", "name", "type"} for node in tg["context_nodes"]))

    def test_budget_is_overall_not_forty_per_node_limits(self):
        record = self.fixture()
        self.assertLess(record["effective_text_prefix_chars"], 400)
        self.assertLessEqual(len(suite.canonical(asdict(suite.prime_request(record, "TG")))), 18000)
        self.assertEqual(record, self.fixture())

    def test_request_json_roundtrip_preserves_identity(self):
        request = suite.prime_request(self.fixture(), "TG")
        stored = json.loads(json.dumps(asdict(request)))
        self.assertEqual(suite.value_hash(stored), suite.value_hash(asdict(request)))
        self.assertEqual(suite.request_from_dict(stored), request)

    def test_selection_is_order_independent_and_label_blind(self):
        ids = list(range(200))
        selected = suite.choose_ids(ids, 100, "arxiv-test")
        self.assertEqual(selected, suite.choose_ids(list(reversed(ids)), 100, "arxiv-test"))
        self.assertEqual(len(set(selected)), 100)
        with self.assertRaises(ValueError):
            suite.choose_ids([1, 1, 2], 2, "invalid")

    def test_native_selection_preserves_query_and_answers_but_separates_labels(self):
        import csv
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "sources").mkdir()
            for filename, ids in (("test.index", range(100)), ("val.index", range(100, 112))):
                (root / "sources" / filename).write_text("\n".join(map(str, ids)))
            with (root / "sources/queries.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["id", "query", "answer_ids"])
                writer.writeheader()
                for i in range(112):
                    writer.writerow({"id": i, "query": f"exact native {i}", "answer_ids": "[123, 456]"})
            queries, labels, counts = suite.native_queries(root)
            self.assertEqual(counts, {"test": 100, "valid": 12})
            self.assertFalse({q["id"] for q in queries["test"]} & {q["id"] for q in queries["valid"]})
            self.assertTrue(all(set(q) == {"id", "query"} for qs in queries.values() for q in qs))
            self.assertEqual(labels["test"]["0"], [123, 456])
            self.assertEqual(next(q["query"] for q in queries["test"] if q["id"] == 0), "exact native 0")

    def test_unconditional_and_pool_conditional_use_scheduled_denominators(self):
        rows = [{"target": n, "request": {"options": [{"id": "1"}, {"id": "2"}]}}
                for n in range(4)]
        labels = {"0": [1, 3], "1": [9], "2": [2], "3": [2]}
        outcomes = [{"status": "success", "selected_option_id": "1"},
                    {"status": "success", "selected_option_id": "1"},
                    {"status": "failure"}, {"status": "not_attempted"}]
        result = suite.metric_rows(rows, labels, outcomes, "prime")
        self.assertEqual(result["scheduled"], 4)
        self.assertEqual(result["Hit@1_unconditional"], .25)
        self.assertEqual(result["Hit@1_pool_hit_conditional"], 1 / 3)
        self.assertEqual(result["candidate_Hit@40"], .75)
        self.assertEqual(result["candidate_Recall@40"], .625)
        self.assertEqual(result["attempted"], 3)
        self.assertEqual(result["states"]["not_attempted"], 1)
        miss = suite.metric_rows(rows[:1], {"0": [9]}, outcomes[:1], "prime")
        self.assertIsNone(miss["Hit@1_pool_hit_conditional"])

    def test_at_most_once_and_raw_capture(self):
        request = suite.prime_request(self.fixture(), "A")
        row = {"split": "valid", "target": 5, "arm": "A", "request": asdict(request),
               "request_sha256": suite.value_hash(asdict(request))}
        client = FakeClient()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings = {"model": "fixture", "timeout_seconds": 1}
            result = asyncio.run(suite.execute_one(root, root, row, settings, client))
            self.assertEqual(result["status"], "success")
            self.assertEqual(client.calls, 1)
            raw = suite.read(root / "valid/5.A/raw-response.json")
            self.assertEqual(raw["raw_output"], {"selected": "0"})
            self.assertTrue((root / "valid/5.A/intent.json").exists())
            with self.assertRaises(FileExistsError):
                asyncio.run(suite.execute_one(root, root, row, settings, client))
            self.assertEqual(client.calls, 1)

    def test_http_wire_survives_invalid_json_before_parser(self):
        import httpx

        class BadJSONClient(FakeClient):
            async def predict(self, request):
                async with httpx.AsyncClient(
                    transport=httpx.MockTransport(lambda _: httpx.Response(200, content=b"not-json")),
                    event_hooks={"response": [suite.capture_http_response]},
                ) as http:
                    response = await http.post("http://localhost/predict", content="public fixture")
                    try:
                        response.json()
                    except ValueError as error:
                        raise InvalidResponseError("invalid JSON", diagnostic_code="invalid_json") from error

        request = suite.prime_request(self.fixture(), "A")
        row = {"split": "valid", "target": 5, "arm": "A", "request": asdict(request),
               "request_sha256": suite.value_hash(asdict(request))}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = asyncio.run(suite.execute_one(
                root, root, row, {"model": "fixture", "timeout_seconds": 1}, BadJSONClient()))
            self.assertEqual(result["status"], "failure")
            wire = suite.read(root / "valid/5.A/wire-0000.json")
            self.assertEqual(wire["payload"]["body_base64"], "bm90LWpzb24=")
            self.assertEqual(wire["request_sha256"], row["request_sha256"])
            self.assertEqual(len(result["wire_capture_sha256"]), 1)
            self.assertFalse((root / "valid/5.A/raw-response.json").exists())

    def test_sdk_returned_raw_event_saved_before_any_answer_parser(self):
        @dataclass
        class EventData:
            content: str

        @dataclass
        class Event:
            type: str
            data: EventData

        class Session:
            async def send_and_wait(self, *args, **kwargs):
                return Event("assistant.message", EventData("not a legal choice"))

        with tempfile.TemporaryDirectory() as tmp:
            context = {"directory": Path(tmp), "request_sha256": "fixture", "wire_index": 0}
            token = suite.CALL_CONTEXT.set(context)
            try:
                returned = asyncio.run(suite.CapturedSession(Session()).send_and_wait("query"))
            finally:
                suite.CALL_CONTEXT.reset(token)
            self.assertEqual(returned.data.content, "not a legal choice")
            wire = suite.read(Path(tmp) / "wire-0000.json")
            self.assertEqual(wire["payload"]["data"]["content"], "not a legal choice")

    @unittest.skipUnless(suite.importlib.util.find_spec("copilot"), "Optional pinned SDK unavailable")
    def test_actual_sdk_message_and_usage_schema_is_losslessly_serializable(self):
        from copilot.generated.session_events import AssistantMessageData, AssistantUsageData
        from datetime import timedelta

        message = AssistantMessageData(content="illegal choice is still captured",
                                       message_id="fixture", model="gpt-5.4")
        usage = AssistantUsageData(model="gpt-5.4", duration=timedelta(microseconds=1234567),
                                   input_tokens=17, reasoning_effort=None)
        self.assertEqual(suite.event_value(message)["content"], message.content)
        encoded = suite.event_value(usage)
        self.assertEqual(encoded["duration"], {"timedelta_microseconds": 1234567})
        self.assertIsNone(encoded["reasoning_effort"])
        json.dumps(encoded, allow_nan=False)

    def test_pinned_native_bridge_preserves_caps_and_separate_stage_wire(self):
        adapter = external_fixture(self, "graphtext") / "native_endpoint_client.py"
        self.assertTrue(adapter.is_file(), "Configured pinned native adapter is missing")
        import httpx

        async def scenario(root):
            settings = {"provider": "native_local", "model": "Mapika/decider-2b",
                        "timeout_seconds": 180}
            client = suite.StageNativeClient(settings, {
                "native_adapter_sha256": suite.NATIVE_ADAPTER_SHA,
                "base_url": "http://localhost:18120",
            }, root)
            self.assertEqual(client.capabilities.max_options, 255)
            self.assertEqual(client.capabilities.max_context_tokens, 32768)

            def handle(request):
                if request.url.path == "/health":
                    return httpx.Response(200, json={"model": settings["model"],
                                                     "model_loaded": True, "forward_count": 0})
                body = json.loads(request.content)
                if body["request_id"].endswith(".G"):
                    return httpx.Response(200, json={"status": "unsupported", "forward_count": 0,
                                                     "coverage": {"full_input": False}})
                ids = [option["id"] for option in body["options"]]
                return httpx.Response(200, json={"status": "success", "forward_count": 1,
                    "result": {"selected_option_id": ids[0], "input_tokens": 2000,
                               "probabilities": {node: int(node == ids[0]) for node in ids}}})

            for plugin in client.clients.values():
                await plugin.http.aclose()
                plugin.http = httpx.AsyncClient(transport=httpx.MockTransport(handle),
                    event_hooks={"response": [suite.capture_http_response]})
            try:
                await client.initialize()
                for split in ("valid", "test"):
                    request = suite.prime_request(self.fixture(), "A")
                    row = {"split": split, "target": 5, "arm": "A", "request": asdict(request),
                           "request_sha256": suite.value_hash(asdict(request))}
                    result = await suite.execute_one(root, root, row, settings, client)
                    self.assertEqual(result["status"], "success")
                    self.assertTrue(result["wire_capture_sha256"])
                    audit = suite.read_rows(root / split / "native-audit.jsonl")
                    self.assertEqual(audit[-1]["forward_count"], 1)
                    self.assertEqual(audit[-1]["event"], "success")
                    self.assertIn("raw_wire", audit[-1])
                request = suite.prime_request(self.fixture(), "G")
                row = {"split": "valid", "target": 5, "arm": "G", "request": asdict(request),
                       "request_sha256": suite.value_hash(asdict(request))}
                result = await suite.execute_one(root, root, row, settings, client)
                self.assertEqual(result["status"], "unsupported")
                self.assertTrue(result["wire_capture_sha256"])
                self.assertEqual(suite.read_rows(root / "valid/native-audit.jsonl")[-1]["forward_count"], 0)
            finally:
                await client.aclose()

        with tempfile.TemporaryDirectory() as tmp, patch.object(suite, "NATIVE_ADAPTER", adapter):
            asyncio.run(scenario(Path(tmp)))

    def test_endpoint_translation_cannot_change_scientific_budget(self):
        async def scenario():
            settings = {"provider": "vllm", "model": "Qwen2.5-72B-Instruct",
                        "timeout_seconds": 180, "max_tokens": 16, "seed": 20260929,
                        "temperature": 0, "top_p": 1, "think": False,
                        "output_format": "answer_only", "constrain_choices": True}
            with self.assertRaises(ValueError):
                await suite.create_client(settings, {"deployment": {
                    "base_url": "http://localhost:19112/v1", "max_tokens": 64}}, Path("/unused"))
            client = await suite.create_client(settings, {"deployment": {
                "base_url": "http://localhost:19112/v1", "served_model": "Qwen2.5-72B",
                "checkpoint_identity_verified": True}}, Path("/unused"))
            try:
                self.assertEqual(client.max_tokens, 16)
                self.assertEqual(client.model, "Qwen2.5-72B")
                self.assertTrue(client._http.event_hooks["response"])
            finally:
                await client.aclose()

        asyncio.run(scenario())

    def test_approval_gates_and_binding(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "rq2"
            suite.save(root / "manifest.json", {})
            suite.save(root.parent / "authorization.json", {"fixture": "authorization"})
            auth_hash = suite.digest(root.parent / "authorization.json")
            manifest = {"models": {"fixture": {"settings": {"model": "fixture"}}},
                        "parent_authorization": {"sha256": auth_hash}}
            approval = {"parent_approved": True, "deployment_ready": True,
                        "manifest_sha256": suite.digest(root / "manifest.json"),
                        "authorization_sha256": auth_hash,
                        "settings_sha256": suite.value_hash({"model": "fixture"}),
                        "model": "fixture", "dataset": "prime", "expires_unix": 1e12,
                        "max_calls": 560}
            with patch.object(suite, "PARENT_AUTH_SHA", auth_hash):
                suite.approval_check(approval, root, manifest, "fixture", "prime")
                for field, value in (("parent_approved", False), ("deployment_ready", False),
                                     ("manifest_sha256", "wrong"), ("settings_sha256", "wrong"),
                                     ("authorization_sha256", "wrong"),
                                     ("expires_unix", 0), ("max_calls", 559)):
                    with self.subTest(field=field), self.assertRaises(ValueError):
                        suite.approval_check({**approval, field: value}, root, manifest, "fixture", "prime")
                (root.parent / "authorization.json").write_text("{}")
                with self.assertRaises(ValueError):
                    suite.approval_check(approval, root, manifest, "fixture", "prime")

    def test_every_old_prompt_hash_is_exact(self):
        fixture = external_fixture(self, "graphtext")
        records = suite.read(fixture / "records.json")
        classes = suite.read(fixture / "classes.json")
        expected = suite.read(fixture / "request-hashes.json")
        for split, values in records.items():
            got = [{"target": record["target"], "arm": arm,
                    "sha256": suite.value_hash(asdict(suite.arxiv_request(record, arm, classes)))}
                   for record in values for arm in suite.ARMS]
            self.assertEqual(got, expected[split])

    def test_all_fourteen_configurations_and_inherited_settings(self):
        old = suite.read(external_fixture(self, "graphtext") / "manifest.json")
        configs = suite.configurations(old)
        self.assertEqual(len(configs), 14)
        for name, original in old["models"].items():
            self.assertEqual(configs[name]["settings"], original)
        for name in ("qwen08", "qwen2", "qwen38"):
            config = configs[name]["settings"]
            for field in ("max_tokens", "temperature", "top_p", "think", "timeout_seconds", "seed"):
                self.assertEqual(config[field], old["models"]["qwen4"][field])
        self.assertIn("unsupported", configs["laya"])
        self.assertIn("unsupported", configs["qwen4_scoring"])
        self.assertEqual(configs["qwen38"]["settings"]["model"], "Qwen3.8-27B")
        self.assertEqual(configs["qwen38"]["execution_precision"], "bfloat16")

    def test_reuse_rejects_changed_request_and_checkpoint(self):
        request = asdict(suite.prime_request(self.fixture(), "T"))
        key = ("test", 5, "T")
        hashed = suite.value_hash(request)
        start = {"split": "test", "target": 5, "arm": "T", "request": request, "sha256": hashed}
        result = {"split": "test", "target": 5, "arm": "T", "request_sha256": hashed,
                  "status": "success", "selected_option_id": "1", "resolved_model": "fixture",
                  "response": {"selected_option_id": "1"}}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            suite.save(root / "status.json", {"manifest_sha256": "manifest"})
            (root / "requests.jsonl").write_text(json.dumps(start) + "\n")
            (root / "attempts.jsonl").write_text(json.dumps(result) + "\n")
            outcomes, _ = suite.validate_reuse(root, {key: hashed}, "manifest", "fixture")
            self.assertEqual(outcomes[key]["selected_option_id"], "1")
            with self.assertRaises(ValueError):
                suite.validate_reuse(root, {key: hashed}, "manifest", "another")
            start["request"]["question"] += " changed"
            (root / "requests.jsonl").write_text(json.dumps(start) + "\n")
            with self.assertRaises(ValueError):
                suite.validate_reuse(root, {key: hashed}, "manifest", "fixture")


class PortableReuseIndexTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.remote = self.root / "remote"
        self.old = self.root / "arxiv-not-opened"
        self.entries = {}
        for name in ("qwen4", "decider"):
            directory = self.remote / "runs" / name
            directory.mkdir(parents=True)
            # Deliberately not JSON: discovery must hash bytes, not reinterpret them.
            (directory / "requests.jsonl").write_bytes(b"unchanged requests\r\n")
            (directory / "attempts.jsonl").write_bytes(b"unchanged attempts\n")
            (directory / "status.json").write_text('{"manifest_sha256": "manifest"}')
            self.entries[name] = {
                "directory": f"runs/{name}",
                "requests_sha256": suite.digest(directory / "requests.jsonl"),
                "attempts_sha256": suite.digest(directory / "attempts.jsonl"),
            }
        self.index = self.remote / "reuse-index.json"
        self.write_index()
        self.default_remote = suite.REMOTE
        patcher = patch.object(suite, "REMOTE", self.remote)
        patcher.start()
        self.addCleanup(patcher.stop)

    def write_index(self):
        self.index.write_text(json.dumps({"models": self.entries}))

    def test_semantic_defaults_and_deterministic_map_without_local_reads(self):
        self.assertEqual(suite.OLD, suite.REPO / "output/config-inputs/graph-text/arxiv")
        self.assertEqual(self.default_remote, suite.REPO / "output/config-inputs/graph-text/remote")
        self.assertEqual(suite.NATIVE_ADAPTER,
                         suite.REPO / "output/config-inputs/graph-text/native_endpoint_client.py")
        original_open = Path.open

        def guarded_open(path, *args, **kwargs):
            self.assertTrue(path.is_relative_to(self.remote), "Opened a local reuse lane")
            return original_open(path, *args, **kwargs)

        with patch.object(Path, "open", guarded_open):
            first = suite.old_directories(self.old)
            self.entries = dict(reversed(list(self.entries.items())))
            self.write_index()
            second = suite.old_directories(self.old)
        self.assertEqual(list(first.items()), list(second.items()))
        self.assertEqual(list(first), ["jev", "gpt54", "gpt6astra", "decider", "qwen4"])
        for name in ("jev", "gpt54", "gpt6astra"):
            self.assertEqual(first[name], self.old / "runs" / name)
        self.assertEqual(first["decider"], self.remote / "runs/decider")
        self.assertEqual(first["qwen4"], self.remote / "runs/qwen4")
        self.assertFalse(self.old.exists())

    def test_every_supported_remote_alias_is_preserved(self):
        entry = self.entries["qwen4"]
        for name in ("qwen9", "qwen27", "qwen72", "kev", "laya"):
            self.entries[name] = dict(entry)
        self.write_index()
        result = suite.old_directories(self.old)
        self.assertEqual(set(result), set(suite.AUTHORIZED_NAMES) - set(suite.NEW))

    def test_changed_bytes_fail_for_each_model_and_hash(self):
        for name in self.entries:
            for filename in ("requests.jsonl", "attempts.jsonl"):
                with self.subTest(model=name, filename=filename):
                    path = self.remote / self.entries[name]["directory"] / filename
                    original = path.read_bytes()
                    path.write_bytes(original + b" ")
                    with self.assertRaisesRegex(ValueError, "digest mismatch"):
                        suite.old_directories(self.old)
                    path.write_bytes(original)

    def test_invalid_model_identifiers_and_reserved_aliases(self):
        for name in ("jev", "gpt54", "gpt6astra", *suite.NEW,
                     "unknown", "Qwen4", "../qwen4", "/qwen4", "", "qwen4\n"):
            with self.subTest(name=name):
                self.entries[name] = dict(self.entries["qwen4"])
                self.write_index()
                with patch.object(suite, "digest", side_effect=AssertionError("Premature blob read")):
                    with self.assertRaisesRegex(ValueError, "model identifier"):
                        suite.old_directories(self.old)
                del self.entries[name]

    def test_exact_schema_and_hash_format_before_any_blob_reads(self):
        valid = json.loads(self.index.read_text())
        invalid = [None, [], {}, {"models": []}, {**valid, "extra": True}]
        for entry in (None, [], {}, {**self.entries["qwen4"], "raw_results": "runs/qwen4"}):
            invalid.append({"models": {**self.entries, "qwen4": entry}})
        for field in ("requests_sha256", "attempts_sha256"):
            entry = dict(self.entries["qwen4"])
            del entry[field]
            invalid.append({"models": {**self.entries, "qwen4": entry}})
            for value in (None, 1, [], "", "a" * 63, "a" * 65, "A" * 64, "g" * 64, "a" * 64 + "\n"):
                invalid.append({"models": {**self.entries, "qwen4": {
                    **self.entries["qwen4"], field: value}}})
        for document in invalid:
            with self.subTest(document=document):
                self.index.write_text(json.dumps(document))
                with patch.object(suite, "digest", side_effect=AssertionError("Premature blob read")):
                    with self.assertRaises(ValueError):
                        suite.old_directories(self.old)

    def test_duplicate_json_keys_are_not_silently_overwritten(self):
        entry = json.dumps(self.entries["decider"])
        documents = [
            '{"models": {}, "models": {"decider": ' + entry + '}}',
            '{"models": {"decider": ' + entry + ', "decider": ' + entry + '}}',
            '{"models": {"decider": ' + entry.replace(
                '"directory":', '"directory": "runs/other", "directory":') + '}}',
        ]
        for document in documents:
            with self.subTest(document=document):
                self.index.write_text(document)
                with self.assertRaisesRegex(ValueError, "Duplicate reuse index key"):
                    suite.old_directories(self.old)

    def test_decider_is_required_and_no_missing_index_fallback(self):
        del self.entries["decider"]
        self.write_index()
        with patch.object(suite, "digest", side_effect=AssertionError("Premature blob read")):
            with self.assertRaisesRegex(ValueError, "explicit decider"):
                suite.old_directories(self.old)
        self.index.unlink()
        with self.assertRaisesRegex(ValueError, "regular file"):
            suite.old_directories(self.old)

    def test_unsafe_and_nonportable_directory_paths(self):
        for relative in (None, 1, [], "", ".", "..", "../outside", "runs/../runs/qwen4",
                         str(self.remote / "runs/qwen4"), "/outside", "//outside", "C:/runs/qwen4",
                         "runs\\qwen4", "runs//qwen4", "./runs/qwen4", "runs/qwen4/", "runs/\x00"):
            with self.subTest(relative=relative):
                self.entries["qwen4"]["directory"] = relative
                self.write_index()
                with patch.object(suite, "digest", side_effect=AssertionError("Premature blob read")):
                    with self.assertRaises(ValueError):
                        suite.old_directories(self.old)

    def test_symlink_directories_inside_and_outside_root_are_rejected(self):
        outside = self.root / "outside"
        outside.mkdir()
        for target in (outside, self.remote / "runs/qwen4"):
            with self.subTest(target=target):
                link = self.remote / "linked"
                link.symlink_to(target, target_is_directory=True)
                self.entries["qwen4"]["directory"] = "linked"
                self.write_index()
                with self.assertRaises(ValueError):
                    suite.old_directories(self.old)
                link.unlink()
        (self.remote / "linked").symlink_to(self.remote / "runs", target_is_directory=True)
        self.entries["qwen4"]["directory"] = "linked/qwen4"
        self.write_index()
        with self.assertRaisesRegex(ValueError, "symlinks"):
            suite.old_directories(self.old)

    def test_symlink_root_is_rejected(self):
        link = self.root / "linked-root"
        link.symlink_to(self.remote, target_is_directory=True)
        with patch.object(suite, "REMOTE", link), self.assertRaisesRegex(ValueError, "reuse root"):
            suite.old_directories(self.old)

    def test_index_and_result_files_must_not_be_symlinks(self):
        paths = [self.index] + [self.remote / "runs/qwen4" / name for name in
                               ("requests.jsonl", "attempts.jsonl", "status.json")]
        for path in paths:
            original = path.read_bytes()
            for parent in (self.root, self.remote):
                with self.subTest(path=path, parent=parent):
                    target = parent / "target-blob"
                    target.write_bytes(original)
                    path.unlink()
                    path.symlink_to(target)
                    with self.assertRaises(ValueError):
                        suite.old_directories(self.old)
                    path.unlink()
                    path.write_bytes(original)
                    target.unlink()

    def test_missing_and_nonregular_inputs_fail_before_hashing(self):
        paths = [self.index] + [self.remote / "runs/qwen4" / name for name in
                               ("requests.jsonl", "attempts.jsonl", "status.json")]
        for path in paths:
            original = path.read_bytes()
            path.unlink()
            for kind in ("missing", "directory", "fifo"):
                with self.subTest(path=path, kind=kind):
                    if kind == "directory":
                        path.mkdir()
                    elif kind == "fifo":
                        suite.os.mkfifo(path)
                    with patch.object(suite, "digest", side_effect=AssertionError("Premature blob read")):
                        with self.assertRaisesRegex(ValueError, "regular file"):
                            suite.old_directories(self.old)
                    if kind == "directory":
                        path.rmdir()
                    elif kind == "fifo":
                        path.unlink()
            path.write_bytes(original)

    def test_missing_result_directory_is_not_manufactured(self):
        self.entries["decider"]["directory"] = "runs/missing"
        self.write_index()
        with self.assertRaisesRegex(ValueError, "directory missing"):
            suite.old_directories(self.old)
        self.assertFalse((self.remote / "runs/missing").exists())

    def test_portable_locator_does_not_bypass_existing_reuse_audit(self):
        directory = self.remote / "runs/qwen4"
        request = {"question": "synthetic exact prompt"}
        hashed = suite.value_hash(request)
        key = ("test", 5, "T")
        start = {"split": "test", "target": 5, "arm": "T", "request": request, "sha256": hashed}
        outcome = {"split": "test", "target": 5, "arm": "T", "request_sha256": hashed,
                   "status": "success", "selected_option_id": "1", "resolved_model": "fixture",
                   "response": {"selected_option_id": "1"}}
        for filename, row, field in (("requests.jsonl", start, "requests_sha256"),
                                     ("attempts.jsonl", outcome, "attempts_sha256")):
            path = directory / filename
            path.write_text(json.dumps(row) + "\n")
            self.entries["qwen4"][field] = suite.digest(path)
        self.write_index()
        before = {path.name: path.read_bytes() for path in directory.iterdir()}
        located = suite.old_directories(self.old)["qwen4"]
        outcomes, _ = suite.validate_reuse(located, {key: hashed}, "manifest", "fixture")
        self.assertEqual(outcomes[key], outcome)
        for expected, manifest, model in (({key: "wrong"}, "manifest", "fixture"),
                                          ({key: hashed}, "wrong", "fixture"),
                                          ({key: hashed}, "manifest", "wrong")):
            with self.subTest(manifest=manifest, model=model), self.assertRaises(ValueError):
                suite.validate_reuse(located, expected, manifest, model)
        self.assertEqual(before, {path.name: path.read_bytes() for path in directory.iterdir()})


if __name__ == "__main__":
    unittest.main()
