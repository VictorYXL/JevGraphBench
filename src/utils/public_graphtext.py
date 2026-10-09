"""RQ2: immutable arXiv reuse and label-blind STaRK-Prime one-choice evaluation.

Use run_benchmark.py --config with configs/graph_text_arxiv.yaml or
configs/graph_text_prime.yaml and an explicitly configured action. The Prime
configuration is supplementary label-blind evaluation, not the primary
gold-containing candidate protocol. Download-capable actions require download
opt-in; running requires inference opt-in and a hash-bound deployment approval.
The direct module CLI remains a compatibility interface.
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import base64
from collections import Counter, defaultdict
from contextvars import ContextVar
from dataclasses import asdict, is_dataclass, replace
from datetime import date, datetime, timedelta
from enum import Enum
import hashlib
import heapq
import importlib
import io
import json
import math
import os
from pathlib import Path
import pickle
import re
import shutil
import sys
import time
import urllib.request
from uuid import UUID
import zipfile

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.clients.base import BaseDecisionClient, DecisionOption, DecisionRequest, UnsupportedRequestError

PROTOCOL = "GraphDecisionBench-public-graphtext"
ROOT = REPO / "output/experiments/public/primary/rq2"
OLD = REPO / "output/config-inputs/graph-text/arxiv"
REMOTE = REPO / "output/config-inputs/graph-text/remote"
DATA_REV = "88269e23e90587f99476c5dd74e235a0877e69be"
CODE_REV = "b246b9490b1dc216892ce78303140014506932c1"
ZIP_SHA = "f6f265f60b761784fb7c2f052359f9cd3bf9ea20c4a307865f62ce173b294dfc"
PARENT_AUTH_SHA = "9eb49b9a3c25a73c830db29ed867e65cdada9e6ca7fdeba58465e6fdf29587f6"
NATIVE_ADAPTER = REPO / "output/config-inputs/graph-text/native_endpoint_client.py"
NATIVE_ADAPTER_SHA = "f9c7c572057cb30ea5ce687f97528a56f9c3d77df0cfdf39efdd2c856f6d0550"
CALL_CONTEXT = ContextVar("rq2_call_capture", default=None)
SEED = 20261003
ARMS = ("T", "G", "TG", "BAG", "A")
NEW = ("qwen08", "qwen2", "qwen38", "qwen4_scoring")
AUTHORIZED_NAMES = {
    "jev": "jev_action", "qwen08": "qwen08", "qwen2": "qwen2",
    "qwen4": "qwen4_grammar", "qwen9": "qwen9_grammar", "qwen27": "qwen27_grammar",
    "qwen38": "qwen38_27b_bf16", "qwen72": "qwen25_72b_bf16",
    "qwen4_scoring": "qwen4_token_scores", "decider": "decider", "kev": "kev", "laya": "laya",
    "gpt54": "gpt54_default_reasoning", "gpt6astra": "gpt6astra_default_reasoning",
}
REQUEST_CHAR_CAP = 18000
QUESTION = (
    "Predict the primary arXiv computer-science subject category of target_node. "
    "Choose exactly one of the 40 candidate class IDs. Known classes refer only "
    "to training anchors, never the target. Directed [u,v] means paper u cites v. "
    "Omitted text or edges are unavailable, not evidence of absence."
)
PRIME_QUESTION = (
    "Answer the native query by choosing exactly one candidate entity ID. "
    "Only candidate_ids are legal answers; other nodes are context only. "
    "Relations are directed [source, relation_type, destination] facts from the "
    "knowledge base. Omitted text or relations are unavailable, not evidence of absence."
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical(value):
    # Identical to the frozen arXiv request hash encoding, including whitespace.
    return json.dumps(value, sort_keys=True, allow_nan=False)


def value_hash(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(canonical(value) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def read_rows(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def choose_ids(ids, count, namespace):
    ids = list(ids)
    require(len(set(ids)) == len(ids) and len(ids) >= count, "Invalid selection pool")
    return sorted(ids, key=lambda n: (value_hash([SEED, namespace, n]), n))[:count]


def parent_authorization(root):
    path = root.parent / "authorization.json"
    require(digest(path) == PARENT_AUTH_SHA, "Parent authorization changed or differs")
    authorization = read(path)
    require(set(authorization["models"]) == set(AUTHORIZED_NAMES.values()),
            "Authorized model roster differs")
    rq2 = authorization["rq2"]
    require(rq2["candidate_count"] == 40 and rq2["gold_injection"] is False and
            rq2["test_answer_dependent_selection"] is False and
            rq2["proposed_test_targets"] == {"ogbn-arxiv": 100, "STaRK-Prime": 100} and
            authorization["new_local_thinking_lanes"] is False,
            "Parent authorization does not match frozen RQ2 semantics")
    return {"relative_path": "../authorization.json", "sha256": PARENT_AUTH_SHA,
            "model_aliases": AUTHORIZED_NAMES}


def arxiv_request(record, arm, classes):
    require(arm in ARMS, "Unknown arm")
    state = {"target_node": "target", "training_anchors": record["anchors"]}
    if arm in ("T", "TG", "BAG"):
        state["texts"] = record["texts"][:1] if arm == "T" else record["texts"]
    if arm in ("G", "TG"):
        state["citation_edges"] = record["edges"]
    return DecisionRequest(
        f'{record["target"]}.{arm}', state, QUESTION,
        tuple(DecisionOption(str(i), classes[str(i)]) for i in range(40)),
    )


def configurations(old):
    result = {name: {"settings": settings, "origin": "exact_old_graphtext"}
              for name, settings in old["models"].items()}
    for name, model, port in (
        ("qwen08", "Qwen3.5-0.8B", 8000),
        ("qwen2", "Qwen3.5-2B", 8001),
        ("qwen38", "Qwen3.8-27B", 18113),
    ):
        settings = dict(old["models"]["qwen4"])
        settings.update(model=model, base_url=f"http://127.0.0.1:{port}/v1")
        result[name] = {"settings": settings, "origin": "new_inherited_qwen4_graphtext"}
    result["qwen38"]["execution_precision"] = "bfloat16"
    result["qwen38"]["authorized_configuration"] = "qwen38_27b_bf16"
    result["qwen4_scoring"] = {
        "settings": {"provider": "semif_scoring", "model": "Qwen3.5-4B",
                     "timeout_seconds": 180, "max_prefix_tokens": 2048,
                     "scoring_mode": "requested"},
        "origin": "new_historical_scoring_not_generation",
        "unsupported": "Historical scoring encoding restricts candidates/context; "
                       "no validated lossless 40-choice mapping. No surrogate.",
    }
    result["laya"]["unsupported"] = "Native maximum 12 choices is below required 40."
    require(len(result) == 14, "Expected all 14 manuscript configurations")
    return result


def acquire(root):
    sources = {
        "processed.zip": "skb/prime/processed.zip",
        "queries.csv": "qa/prime/stark_qa/stark_qa.csv",
        "test.index": "qa/prime/split/test.index",
        "val.index": "qa/prime/split/val.index",
        "dataset-card.txt": "README.md",
    }
    provenance = {}
    for local, remote in sources.items():
        url = f"https://huggingface.co/datasets/snap-stanford/stark/resolve/{DATA_REV}/{remote}"
        path = root / "sources" / local
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            with urllib.request.urlopen(url, timeout=120) as response, path.open("xb") as out:
                while chunk := response.read(1024 * 1024):
                    out.write(chunk)
        provenance[local] = {"url": url, "sha256": digest(path), "bytes": path.stat().st_size}
    require(provenance["processed.zip"]["sha256"] == ZIP_SHA, "Pinned Prime ZIP hash mismatch")
    return provenance


class PrimitiveUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError("Only primitive official metadata is accepted")


def load_prime(root):
    import torch

    path = root / "sources/processed.zip"
    require(digest(path) == ZIP_SHA, "Prime archive differs from official LFS digest")
    with zipfile.ZipFile(path) as archive:
        nodes = PrimitiveUnpickler(io.BytesIO(archive.read("processed/node_info.pkl"))).load()
        relations = PrimitiveUnpickler(io.BytesIO(archive.read("processed/edge_type_dict.pkl"))).load()
        edge_index = torch.load(io.BytesIO(archive.read("processed/edge_index.pt")),
                                weights_only=True, map_location="cpu").numpy()
        edge_types = torch.load(io.BytesIO(archive.read("processed/edge_types.pt")),
                                weights_only=True, map_location="cpu").numpy()
    require(set(nodes) == set(range(129375)), "Expected 129375 native entity IDs")
    require(edge_index.shape == (2, len(edge_types)), "Invalid typed edge dimensions")
    require(int(edge_index.min()) >= 0 and int(edge_index.max()) < len(nodes),
            "Invalid native edge endpoints")
    return nodes, relations, edge_index, edge_types


def intrinsic(node):
    # Explicit metadata allowlist: never render grouped neighbors or edge incidence.
    details = node.get("details", {})
    fields = [str(node["name"]), str(node["type"])]
    if isinstance(details, dict):
        for key in ("name", "alias", "summary", "description", "definition"):
            value = details.get(key)
            if isinstance(value, str):
                fields.append(value)
            elif isinstance(value, list):
                fields.extend(item for item in value if isinstance(item, str))
    return " ".join(" ".join(fields).split())


def tokens(text):
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25:
    """Full-corpus BM25, k1=1.2, b=.75, positive Robertson IDF; ID tie break."""

    def __init__(self, documents):
        self.ids = sorted(documents)
        self.lengths = {}
        self.postings = defaultdict(list)
        for node in self.ids:
            counts = Counter(tokens(documents[node]))
            self.lengths[node] = sum(counts.values())
            for term, frequency in counts.items():
                self.postings[term].append((node, frequency))
        self.average = sum(self.lengths.values()) / len(self.ids)
        require(self.average > 0, "Empty retrieval corpus")

    def top(self, query, count=40):
        require(len(self.ids) >= count, "Insufficient full candidate corpus")
        scores = defaultdict(float)
        n = len(self.ids)
        for term in sorted(set(tokens(query))):
            posting = self.postings.get(term, ())
            idf = math.log(1 + (n - len(posting) + .5) / (len(posting) + .5))
            for node, frequency in posting:
                norm = 1.2 * (.25 + .75 * self.lengths[node] / self.average)
                scores[node] += idf * frequency * 2.2 / (frequency + norm)
        # Include all zero-score entities rather than filtering/resampling queries.
        ranked = heapq.nsmallest(count, self.ids, key=lambda node: (-scores.get(node, 0), node))
        return ranked, [scores.get(node, 0.0) for node in ranked]


def native_queries(root):
    import csv

    with (root / "sources/queries.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    by_id = {int(row["id"]): row for row in rows}
    require(len(by_id) == len(rows), "Duplicate native query IDs")
    splits = {name: [int(x) for x in (root / "sources" / filename).read_text().split()]
              for name, filename in (("test", "test.index"), ("valid", "val.index"))}
    require(not set(splits["valid"]) & set(splits["test"]), "Validation/test overlap")
    chosen = {split: choose_ids(ids, 100 if split == "test" else 12, f"prime-{split}")
              for split, ids in splits.items()}
    # Selection above cannot inspect answer_ids.
    queries = {split: [{"id": qid, "query": by_id[qid]["query"]} for qid in ids]
               for split, ids in chosen.items()}
    labels = {}
    for split, ids in chosen.items():
        labels[split] = {}
        for qid in ids:
            answers = ast.literal_eval(by_id[qid]["answer_ids"])
            require(isinstance(answers, list) and answers and
                    all(type(x) is int and 0 <= x < 129375 for x in answers),
                    "Invalid native answer IDs; never silently discard a query")
            labels[split][str(qid)] = sorted(set(answers))
    return queries, labels, {split: len(ids) for split, ids in splits.items()}


def prime_request(record, arm):
    require(arm in ARMS, "Unknown arm")
    state = {"query": record["query"], "candidate_ids": record["candidate_ids"],
             "candidate_nodes": record["candidate_nodes"]}
    if arm in ("G", "TG", "BAG"):
        state["context_nodes"] = record["context_nodes"]
    if arm in ("T", "TG", "BAG"):
        state["texts"] = record["candidate_texts"]
        if arm != "T":
            state["texts"] = state["texts"] + record["context_texts"]
    if arm in ("G", "TG"):
        state["relations"] = record["relations"]
    return DecisionRequest(f'prime.{record["id"]}.{arm}', state, PRIME_QUESTION,
                           tuple(DecisionOption(str(node)) for node in record["candidate_ids"]))


def grounding(node_id, nodes):
    node = nodes[node_id]
    return {"id": str(node_id), "name": str(node["name"]), "type": str(node["type"])}


def evidence_record(query, candidates, scores, nodes, documents, adjacency):
    candidate_set = set(candidates)
    # Two hash-ordered incident facts per candidate; no query answers in this path.
    edges = {edge for node in candidates for edge in adjacency.get(node, ())[:2]}
    ordered = sorted(edges, key=lambda e: (value_hash([SEED, "edge", query["id"], e]), e))
    record = {"id": query["id"], "query": query["query"],
              "candidate_ids": [str(node) for node in candidates],
              "candidate_nodes": [grounding(node, nodes) for node in candidates],
              "candidate_texts": [], "context_nodes": [], "context_texts": [],
              "relations": [], "retrieval_scores": scores}
    # Reserve about half the overall request budget for intrinsic descriptions.
    while True:
        context = {int(e[0]) for e in ordered} | {int(e[2]) for e in ordered}
        context -= candidate_set
        context = sorted(context, key=lambda n: (value_hash([SEED, "context", query["id"], n]), n))
        record["relations"] = [list(edge) for edge in ordered]
        record["context_nodes"] = [grounding(node, nodes) for node in context]
        if len(canonical(asdict(prime_request(record, "TG")))) <= REQUEST_CHAR_CAP - 6000:
            break
        require(bool(ordered), "Mandatory native query/candidate grounding exceeds common cap")
        ordered.pop()
    text_ids = candidates + context
    # One global water-filled character budget, not 40 independent per-node caps.
    low, high = 0, REQUEST_CHAR_CAP
    while low < high:
        cap = (low + high + 1) // 2
        texts = [{"id": str(node), "text": documents[node][:cap]} for node in text_ids]
        record["candidate_texts"], record["context_texts"] = texts[:40], texts[40:]
        if len(canonical(asdict(prime_request(record, "TG")))) <= REQUEST_CHAR_CAP:
            low = cap
        else:
            high = cap - 1
    texts = [{"id": str(node), "text": documents[node][:low]} for node in text_ids]
    record["candidate_texts"], record["context_texts"] = texts[:40], texts[40:]
    record["effective_text_prefix_chars"] = low
    require(all(len(canonical(asdict(prime_request(record, arm)))) <= REQUEST_CHAR_CAP
                for arm in ARMS), "Common request cap exceeded")
    return record


def build_prime(root):
    import numpy as np

    nodes, relations, edges, types = load_prime(root)
    queries, labels, split_counts = native_queries(root)
    documents = {node: intrinsic(info) for node, info in nodes.items()}
    index = BM25(documents)
    pools = {q["id"]: index.top(q["query"]) for split in queries.values() for q in split}
    needed = {node for candidates, _ in pools.values() for node in candidates}
    mask = np.isin(edges[0], sorted(needed)) | np.isin(edges[1], sorted(needed))
    positions = np.flatnonzero(mask)
    # Keep only two facts per entity while scanning all its native incident edges.
    best = defaultdict(list)
    for pos in positions:
        u, v, rel = int(edges[0, pos]), int(edges[1, pos]), relations[int(types[pos])]
        edge = (str(u), rel, str(v))
        key = value_hash([SEED, "incident", edge])
        for node in {u, v} & needed:
            entries = best[node]
            if edge not in [item[1] for item in entries]:
                entries.append((key, edge))
                entries.sort()
                del entries[2:]
    adjacency = {node: [edge for _, edge in entries] for node, entries in best.items()}
    records = {split: [evidence_record(q, *pools[q["id"]], nodes, documents, adjacency)
                       for q in selected] for split, selected in queries.items()}
    audit = {"candidate_corpus_size": len(nodes), "native_directed_edges": len(types),
             "official_split_counts": split_counts, "corpus_sha256": value_hash(documents),
             "retrieval": "BM25 intrinsic name/type/name/alias/summary/description/definition; "
                          "all 129375 entities; k1=1.2 b=.75; unique query terms; ID tie break",
             "evidence": "Up to 2 hash-ordered incident typed facts per candidate; flat "
                         "hash-mixed context nodes; no per-candidate neighbor groups",
             "budget": {"all_models_serialized_request_characters": REQUEST_CHAR_CAP,
                        "encoding": "json.dumps(sort_keys=True, ensure_ascii=True)",
                        "allocation": "common skeleton then global max-min text-prefix water filling",
                        "per_model_truncation": False},
             "selection_reads_answers": False, "gold_injection": False,
             "zero_hit_filter": False, "resample_on_answers": False}
    return records, labels, audit


def old_directories(old):
    """Locate unchanged reuse blobs via an explicitly supplied, unsealed index.

    The locator is not scientific provenance: byte hashes and the subsequent
    validate_reuse prompt/manifest audit remain mandatory. Never discover or
    translate machine-specific result locations from historical artifacts.
    """
    roots = {name: old / "runs" / name for name in ("jev", "gpt54", "gpt6astra")}
    remote = REMOTE.resolve()
    require(REMOTE.is_dir() and not REMOTE.is_symlink(), "Invalid reuse root")

    def local_path(relative, *, directory=False):
        require(isinstance(relative, str) and bool(relative) and
                not any(char in relative for char in ("\\", ":", "\x00")) and
                all(part not in ("", ".", "..") for part in relative.split("/")),
                "Reuse paths must be nonempty portable root-relative paths")
        path = remote / relative
        require(not Path(relative).is_absolute() and path.resolve().is_relative_to(remote),
                "Reuse path escapes its root")
        cursor = remote
        for part in Path(relative).parts:
            cursor = cursor / part
            require(not cursor.is_symlink(), "Reuse paths must not contain symlinks")
        require(path.is_dir() if directory else path.is_file(),
                "Reuse directory missing" if directory else "Reuse input must be a regular file")
        return path

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "Duplicate reuse index key")
            result[key] = value
        return result

    index = json.loads(local_path("reuse-index.json").read_text(), object_pairs_hook=unique_object)
    require(isinstance(index, dict) and set(index) == {"models"} and
            isinstance(index["models"], dict), "Invalid reuse index schema")
    entries = index["models"]
    require("decider" in entries, "Reuse index requires an explicit decider entry")
    # Validate the entire locator before opening any result blob. Local lanes
    # are only returned, not opened or verified by this location resolver.
    for name, entry in entries.items():
        require(re.fullmatch(r"[a-z][a-z0-9_]*", name) is not None and
                name in AUTHORIZED_NAMES and name not in roots and name not in NEW,
                "Invalid or reserved reuse model identifier")
        require(isinstance(entry, dict) and
                set(entry) == {"directory", "requests_sha256", "attempts_sha256"},
                "Invalid reuse model entry schema")
        for field in ("requests_sha256", "attempts_sha256"):
            require(isinstance(entry[field], str) and
                    re.fullmatch(r"[0-9a-f]{64}", entry[field]) is not None,
                    "Reuse hashes must be 64 lowercase hexadecimal characters")
    located = {}
    for name in sorted(entries):
        relative = entries[name]["directory"]
        located[name] = local_path(relative, directory=True)
        for filename in ("requests.jsonl", "attempts.jsonl", "status.json"):
            local_path(f"{relative}/{filename}")
    for name, directory in located.items():
        for filename, field in (("requests.jsonl", "requests_sha256"),
                                ("attempts.jsonl", "attempts_sha256")):
            require(digest(directory / filename) == entries[name][field], "Remote reuse digest mismatch")
    roots.update(located)
    return roots


def validate_reuse(directory, expected, manifest_sha, model):
    status = read(directory / "status.json")
    require(status["manifest_sha256"] == manifest_sha, "Old result manifest mismatch")
    requests = read_rows(directory / "requests.jsonl")
    attempts = read_rows(directory / "attempts.jsonl")
    seen = {}
    for row in requests:
        key = (row["split"], row["target"], row["arm"])
        require(key not in seen and key in expected, "Duplicate/foreign old request")
        require(value_hash(row["request"]) == expected[key], "Old prompt differs")
        require(row["sha256"] == expected[key], "Old request digest differs")
        seen[key] = row
    outcomes = {}
    for row in attempts:
        key = (row["split"], row["target"], row["arm"])
        require(key in seen and key not in outcomes, "Unbound/duplicate old outcome")
        require(row["request_sha256"] == expected[key], "Old result request digest differs")
        if row["status"] == "success":
            require(row["resolved_model"] == model, "Old exact model differs")
            require(row["selected_option_id"] in {str(i) for i in range(40)}, "Invalid old choice")
            require(row["response"]["selected_option_id"] == row["selected_option_id"],
                    "Old response identity mismatch")
        outcomes[key] = row
    return outcomes, seen


def freeze(root=ROOT, old=OLD):
    require(not (root / "manifest.json").exists(), "Never overwrite a frozen RQ2 protocol")
    authorization = parent_authorization(root)
    old_manifest = read(old / "manifest.json")
    for name, expected in old_manifest["inputs_sha256"].items():
        require(digest(old / name) == expected, f"Old input changed: {name}")
    for name, expected in old_manifest["source_sha256"].items():
        require(digest(old / "frozen-source" / name) == expected, f"Old source changed: {name}")
    old_records, old_labels, classes = (read(old / name) for name in
                                      ("records.json", "private-labels.json", "classes.json"))
    frozen_hashes = read(old / "request-hashes.json")
    expected = {(split, row["target"], row["arm"]): row["sha256"]
                for split, rows in frozen_hashes.items() for row in rows}
    for split, records in old_records.items():
        for record in records:
            for arm in ARMS:
                require(value_hash(asdict(arxiv_request(record, arm, classes))) ==
                        expected[(split, record["target"], arm)], "arXiv exact prompt replay failed")
    selected = choose_ids([r["target"] for r in old_records["test"]], 100, "arxiv-test")
    arxiv = {"test": [next(r for r in old_records["test"] if r["target"] == node)
                      for node in selected], "valid": old_records["valid"]}
    provenance = acquire(root)
    prime, prime_labels, prime_audit = build_prime(root)
    configs = configurations(old_manifest)
    rows = []
    for dataset, records in (("arxiv", arxiv), ("prime", prime)):
        for split, values in records.items():
            for record in values:
                for arm in ARMS:
                    req = arxiv_request(record, arm, classes) if dataset == "arxiv" else prime_request(record, arm)
                    rows.append({"dataset": dataset, "split": split,
                                 "target": record["target"] if dataset == "arxiv" else record["id"],
                                 "arm": arm, "request": asdict(req),
                                 "request_sha256": value_hash(asdict(req))})
    save(root / "requests.json", rows)
    save(root / "prime/records.json", prime)
    save(root / "prime/retrieval-audit.json", prime_audit)
    save(root / "arxiv/records.json", arxiv)
    save(root / "labels.json", {"prime": prime_labels, "arxiv": {
        split: {str(r["target"]): [old_labels[split][str(r["target"])]] for r in records}
        for split, records in arxiv.items()}})
    sources = {"old_manifest": {"path": str(old / "manifest.json"), "sha256": digest(old / "manifest.json")}}
    reused = {}
    all200 = {}
    for model, directory in old_directories(old).items():
        outcomes, starts = validate_reuse(directory, expected, digest(old / "manifest.json"),
                                         configs[model]["settings"]["model"])
        for name in ("attempts.jsonl", "requests.jsonl", "status.json"):
            path = directory / name
            sources[f"{model}/{name}"] = {"path": str(path), "sha256": digest(path)}
        all200[model] = list(outcomes.values())
        for row in rows:
            key = (row["split"], row["target"], row["arm"])
            if row["dataset"] != "arxiv":
                continue
            outcome = outcomes.get(key)
            state = ("success" if outcome and outcome["status"] == "success" else
                     "failure" if outcome else "attempted_unknown" if key in starts else "not_attempted")
            reused[f'{model}/{row["split"]}/{row["target"]}/{row["arm"]}'] = {
                "status": state, "outcome": outcome, "source": str(directory),
                "rerun_allowed": False, "validated_exact_request": key in starts}
    save(root / "arxiv/original200-all-results.json", all200)
    save(root / "arxiv/original200-labels.json", old_labels)
    save(root / "arxiv/reuse.json", reused)
    owned_source = {str(Path(__file__).relative_to(REPO)): digest(__file__)}
    for file in (REPO / "src/clients").glob("*.py"):
        owned_source[str(file.relative_to(REPO))] = digest(file)
    owned_source["src/benchmark/config.py"] = digest(REPO / "src/benchmark/config.py")
    for name in ("tests/integration/test_public_graphtext.py", "src/utils/extended_graph_suite.py",
                 "src/utils/paired_graph_ablation.py", "src/utils/__init__.py",
                 "src/benchmark/__init__.py", "src/__init__.py"):
        owned_source[name] = digest(REPO / name)
    require(digest(NATIVE_ADAPTER) == NATIVE_ADAPTER_SHA, "Native adapter changed before freeze")
    owned_source[str(NATIVE_ADAPTER.relative_to(REPO))] = NATIVE_ADAPTER_SHA
    for name, expected_hash in owned_source.items():
        target = root / "frozen-source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        require(not target.exists(), "Frozen source already exists")
        shutil.copyfile(REPO / name, target)
        require(digest(target) == expected_hash, "Source changed while freezing")
    old_sizes = sorted(len(canonical(asdict(arxiv_request(record, "TG", classes))))
                       for record in old_records["test"])
    artifact_paths = [p for p in root.rglob("*") if p.is_file()
                      and "offline-preflight-archive" not in p.relative_to(root).parts]
    manifest = {
        "protocol": PROTOCOL, "created_unix": time.time(), "seed": SEED,
        "parent_authorization": authorization,
        "models": configs, "arms": list(ARMS), "prime_control_display_name": "A*",
        "source_sha256": owned_source, "legacy_sources": sources,
        "data_revision": DATA_REV, "code_reference_revision": CODE_REV,
        "upstream_sources": provenance, "upstream_license": "STaRK dataset CC-BY-4.0",
        "attribution": "Wu et al. (2024), STaRK; PrimeKG (Chandak et al.); OGB/MAG/arXiv authors.",
        "third_party_code_copied": False,
        "selection": "SHA256([20261003, namespace, native_id]) sorted; first 100 test / "
                     "12 validation; original arXiv 12 validation unchanged; outcome-independent",
        "settings": "Old arXiv exact requests/configs/results only; new Qwens explicitly inherit "
                    "16 tokens, temperature 0, top_p 1, no-thinking, 180s. GPT default 4096/180s; Jev 30s.",
        "interpretation": "All Prime views are text-conditioned by full-corpus BM25 retrieval. "
                          "TG-BAG removes explicit edges with identical flat nodes/texts; "
                          "A* is query plus minimal entity grounding, not training labels.",
        "original_arxiv_TG_serialized_chars": {
            "min": old_sizes[0], "median": old_sizes[len(old_sizes) // 2], "max": old_sizes[-1]},
        "metrics": "Single-choice Hit@1 unconditional and pool-hit conditional; candidate "
                   "Hit@40/Recall@40. Not full-corpus model ranking or MRR. arXiv accuracy/macro-F1.",
        "pilot": "12 native validation queries x5 arms x14 configs, no outcome tuning; "
                 "added arXiv four configs use old 12 validation then selected 100 test. "
                 "Only structural/protocol success gates test, never accuracy or response diversity.",
        "attempt_policy": "At most once; exclusive durable intent before transport; no retries, "
                          "repair or rerun of any old arXiv scheduled state.",
        "files": {str(p.relative_to(root)): digest(p) for p in artifact_paths},
    }
    save(root / "manifest.json", manifest)
    save(root / "readiness.json", {
        "offline_freeze": "complete", "manifest_sha256": digest(root / "manifest.json"),
        "authorization_sha256": PARENT_AUTH_SHA,
        "parent_authorization_verified": True,
        "scientific_execution_authorized": True, "deployment_attestation_required": True,
        "inference_authorized": False, "scheduled_per_dataset_per_model": 560,
        "new_scheduled_slots": 10080, "new_requests_max": 8400,
        "blockers": ["Deployment health/lease attestation required; parent authorization already granted",
                     "Laya unsupported 40 choices; scoring unsupported lossless mapping"],
        "native_adapter": {"path": str(NATIVE_ADAPTER), "sha256": NATIVE_ADAPTER_SHA,
                           "bridge": "built in; unchanged historical caps/decoding, stage-separated audit"},
        "raw_capture": "HTTP response bytes before parsing; SDK returned assistant/usage/retry events "
                       "before parsing; full returned DecisionResponse; no credentials/headers captured",
        "run_command": "python -m src.utils.public_graphtext run --dataset prime "
                       "--model MODEL --approval APPROVAL.json",
    })
    return manifest


def verify(root, sources=True):
    manifest = read(root / "manifest.json")
    require(manifest["parent_authorization"] == parent_authorization(root),
            "Frozen parent authorization differs")
    for name, expected in manifest["files"].items():
        require(digest(root / name) == expected, f"Frozen RQ2 artifact changed: {name}")
    if sources:
        for name, expected in manifest["source_sha256"].items():
            require(digest(REPO / name) == expected, f"Runtime source changed: {name}")
    rows = read(root / "requests.json")
    require(len(rows) == 1120, "Expected 2 datasets x112 queries x5 arms")
    for row in rows:
        require(value_hash(row["request"]) == row["request_sha256"], "Frozen request mismatch")
        require(len(row["request"]["options"]) == 40, "All requests must have 40 choices")
    identities = [(r["dataset"], r["split"], r["target"], r["arm"]) for r in rows]
    require(len(set(identities)) == len(identities), "Duplicate scheduled request")
    prime = read(root / "prime/records.json")
    labels = read(root / "labels.json")
    for split, records in prime.items():
        require(len(records) == (100 if split == "test" else 12), "Invalid Prime split size")
        for record in records:
            require(len(set(record["candidate_ids"])) == 40, "Prime pool is not 40 unique entities")
            views = {arm: asdict(prime_request(record, arm)) for arm in ARMS}
            require(views["BAG"]["state"] == {
                k: v for k, v in views["TG"]["state"].items() if k != "relations"},
                "BAG contains edge incidence or differs from TG")
            for arm, view in views.items():
                require(len(canonical(view)) <= REQUEST_CHAR_CAP, "Prime budget exceeded")
                matches = [r for r in rows if r["dataset"] == "prime" and r["split"] == split
                           and r["target"] == record["id"] and r["arm"] == arm]
                require(len(matches) == 1 and canonical(matches[0]["request"]) == canonical(view),
                        "Prime frozen evidence/request mismatch")
            require(str(record["id"]) in labels["prime"][split], "Missing reporting labels")
    require(not {r["id"] for r in prime["test"]} & {r["id"] for r in prime["valid"]},
            "Prime split overlap")
    return manifest, rows


def request_from_dict(value):
    return DecisionRequest(**{**value, "options": tuple(DecisionOption(**o) for o in value["options"])})


def approval_check(approval, root, manifest, model, dataset):
    require(approval.get("parent_approved") is True and approval.get("deployment_ready") is True,
            "Parent and deployment readiness approvals both required")
    require(approval["manifest_sha256"] == digest(root / "manifest.json"),
            "Approval not bound to this manifest")
    require(approval["authorization_sha256"] == manifest["parent_authorization"]["sha256"] ==
            PARENT_AUTH_SHA == digest(root.parent / "authorization.json"),
            "Deployment approval is not bound to the unchanged parent authorization")
    require(approval["model"] == model and approval["dataset"] == dataset,
            "Approval is for another lane")
    require(approval["settings_sha256"] == value_hash(manifest["models"][model]["settings"]),
            "Approved model settings differ")
    require(approval["expires_unix"] > time.time(), "Deployment approval expired")
    require(approval.get("max_calls", 0) >= 560, "Explicit 560-call lane budget required")
    precision = manifest["models"][model].get("execution_precision")
    if precision is not None:
        require(approval.get("execution_precision") == precision, "Deployment precision differs")


def capture_wire(kind, payload, context=None):
    context = CALL_CONTEXT.get() if context is None else context
    if context is None:
        return
    path = context["directory"] / f'wire-{context["wire_index"]:04d}.json'
    save(path, {"kind": kind, "request_sha256": context["request_sha256"], "payload": payload})
    context["wire_index"] += 1


async def capture_http_response(response):
    if CALL_CONTEXT.get() is None:
        return
    body = await response.aread()
    capture_wire("http_response", {
        "method": response.request.method, "path": response.request.url.path,
        "status_code": response.status_code, "body_base64": base64.b64encode(body).decode("ascii"),
    })


def event_value(value):
    if is_dataclass(value):
        return event_value(asdict(value))
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, Enum):
        return event_value(value.value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, timedelta):
        return {"timedelta_microseconds":
                (value.days * 86400 + value.seconds) * 1000000 + value.microseconds}
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, dict):
        require(all(isinstance(key, str) for key in value), "Invalid SDK event field names")
        return {key: event_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [event_value(item) for item in value]
    require(value is None or isinstance(value, (str, int, float, bool)),
            "SDK event data has no lossless JSON representation")
    return value


def event_payload(event):
    data = event_value(event.data)
    require(isinstance(data, dict), "SDK event data must be an object")
    kind = getattr(event.type, "value", event.type)
    return {"type": kind, "data": data}


class CapturedSession:
    def __init__(self, session):
        self.session = session

    def __getattr__(self, name):
        return getattr(self.session, name)

    async def send_and_wait(self, *args, **kwargs):
        event = await self.session.send_and_wait(*args, **kwargs)
        if event is not None:
            capture_wire("sdk_returned_event", event_payload(event))
        return event


class CapturedRuntime:
    def __init__(self, runtime):
        self.runtime = runtime

    def __getattr__(self, name):
        return getattr(self.runtime, name)

    async def create_session(self, **kwargs):
        original = kwargs["on_event"]
        context = CALL_CONTEXT.get()

        def on_event(event):
            kind = getattr(event.type, "value", event.type)
            if kind in ("assistant.message", "assistant.usage", "assistant.turn_retry"):
                capture_wire("sdk_event", event_payload(event), context=context)
            original(event)

        return CapturedSession(await self.runtime.create_session(**{**kwargs, "on_event": on_event}))


def native_factory():
    require(digest(NATIVE_ADAPTER) == NATIVE_ADAPTER_SHA, "Pinned native adapter changed")
    spec = importlib.util.spec_from_file_location("rq2_pinned_native_endpoint", NATIVE_ADAPTER)
    require(spec is not None and spec.loader is not None, "Native adapter cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.create_client


class StageNativeClient(BaseDecisionClient):
    """Use the unchanged pinned native plugin with distinct validation/test audit files."""

    def __init__(self, settings, deployment, directory):
        require(set(deployment) == {"native_adapter_sha256", "base_url"},
                "Native deployment may only bind the adapter hash and endpoint")
        require(deployment["native_adapter_sha256"] == NATIVE_ADAPTER_SHA,
                "Deployment native adapter hash differs")
        factory = native_factory()
        self.clients = {split: factory(
            model=settings["model"], timeout_seconds=settings["timeout_seconds"],
            base_url=deployment["base_url"],
            audit_path=str(directory / split / "native-audit.jsonl"),
        ) for split in ("valid", "test")}
        for client in self.clients.values():
            client.http.event_hooks["response"].append(capture_http_response)
        self.initialized = set()

    @property
    def capabilities(self):
        return self.clients["valid"].capabilities

    async def initialize(self):
        await self.clients["valid"].initialize()
        self.initialized.add("valid")

    async def predict(self, request):
        context = CALL_CONTEXT.get()
        require(context is not None, "Native prediction requires durable request context")
        split = context["split"]
        if split not in self.initialized:
            await self.clients[split].initialize()
            self.initialized.add(split)
        return await self.clients[split].predict(request)

    async def aclose(self):
        try:
            await self.clients["valid"].aclose()
        finally:
            await self.clients["test"].aclose()


async def create_client(settings, approval, directory):
    deployment = approval.get("deployment", {})
    if settings["provider"] == "native_local":
        return StageNativeClient(settings, deployment, directory)
    if "adapter_factory" in approval:
        module_name, function_name = approval["adapter_factory"].split(":")
        spec = importlib.util.find_spec(module_name)
        require(spec is not None and spec.origin is not None, "Adapter module unavailable")
        require(digest(spec.origin) == approval["adapter_sha256"], "Adapter source differs")
        factory = getattr(importlib.import_module(module_name), function_name)
        return factory(settings=settings, audit_dir=directory)
    require(settings["provider"] in ("typesafe", "vllm", "github_copilot"),
            "Native/scoring lane requires an approved adapter; no surrogate")
    from src.benchmark.config import ModelConfig
    from src.clients.registry import create_client as registry_client

    config = ModelConfig(**settings)
    if deployment:
        require(config.provider == "vllm", "Endpoint relocation is only for local vLLM/native")
        require(set(deployment) <= {"base_url", "served_model", "checkpoint_identity_verified"},
                "Deployment cannot override scientific settings")
        served_model = deployment.get("served_model", config.model)
        if served_model != config.model:
            require(deployment.get("checkpoint_identity_verified") is True,
                    "Served alias needs verified same-checkpoint identity")
        config = replace(config, base_url=deployment.get("base_url", config.base_url), model=served_model)
    kwargs = config.client_kwargs()
    if config.provider == "typesafe":
        kwargs["require_probabilities"] = False
    if config.provider == "github_copilot":
        from src.clients.github_copilot_client import GitHubCopilotClient

        class CapturedCopilotClient(GitHubCopilotClient):
            async def _ensure_runtime(self):
                await super()._ensure_runtime()
                if not isinstance(self._runtime, CapturedRuntime):
                    self._runtime = CapturedRuntime(self._runtime)

        return CapturedCopilotClient(**kwargs)
    client = registry_client(config.provider, **kwargs)
    client._http.event_hooks["response"].append(capture_http_response)
    return client


async def execute_one(root, directory, row, settings, client, deployment=None):
    from src.utils import extended_graph_suite as audit

    request = request_from_dict(row["request"])
    destination = directory / row["split"] / f'{row["target"]}.{row["arm"]}'
    destination.mkdir(parents=True, exist_ok=False)
    save(destination / "intent.json", {**row, "settings_sha256": value_hash(settings),
                                     "approval_sha256": digest(directory / "approval.json")
                                     if (directory / "approval.json").exists() else None,
                                     "started_unix": time.time()})
    started = time.perf_counter()
    entered = False
    context = {"directory": destination, "request_sha256": row["request_sha256"],
               "wire_index": 0, "split": row["split"]}
    context_token = CALL_CONTEXT.set(context)
    try:
        client.validate_request(request)
        entered = True
        response = await asyncio.wait_for(client.predict(request), settings["timeout_seconds"])
        # Capture the entire returned response before parsing/scoring, as in the durable suite.
        save(destination / "raw-response.json", asdict(response))
        if settings.get("provider") in ("native_local", "vllm", "typesafe", "github_copilot"):
            require(context["wire_index"] > 0, "Missing pre-parse provider capture")
        require(response.request_id == request.request_id, "Response/request ID mismatch")
        require(response.selected_option_id in {o.id for o in request.options}, "Invalid response choice")
        expected_model = (deployment or {}).get("served_model", settings["model"])
        require(response.resolved_model == expected_model, "Resolved checkpoint differs")
        outcome = {"status": "success", "selected_option_id": response.selected_option_id,
                   "resolved_model": response.resolved_model, "scientific_model": settings["model"],
                   "usage": asdict(response.usage)}
    except Exception as exc:
        diagnostic = audit.failure(exc)
        outcome = {"status": "unsupported" if isinstance(exc, UnsupportedRequestError) else "failure",
                   "selected_option_id": None, "usage": audit.safe_usage(getattr(exc, "usage", None)),
                   "error_type": type(exc).__name__, **diagnostic}
    finally:
        CALL_CONTEXT.reset(context_token)
    outcome.update(request_sha256=row["request_sha256"], transport_entered=entered,
                   settings_sha256=value_hash(settings),
                   raw_response_sha256=digest(destination / "raw-response.json")
                   if (destination / "raw-response.json").exists() else None,
                   wire_capture_sha256={path.name: digest(path)
                                        for path in sorted(destination.glob("wire-*.json"))},
                   latency_seconds=time.perf_counter() - started, monetary_cost_usd=None)
    save(destination / "outcome.json", outcome)
    return outcome


async def run(root, model, dataset, approval_path):
    manifest, rows = verify(root)
    require(model in manifest["models"] and dataset in ("arxiv", "prime"), "Unknown lane")
    require(dataset != "arxiv" or model in NEW, "Old arXiv lanes are reuse-only, including failures")
    approval = read(approval_path)
    approval_check(approval, root, manifest, model, dataset)
    lane = root / "runs" / dataset / model
    lane.mkdir(parents=True, exist_ok=False)
    save(lane / "approval.json", approval)
    config = manifest["models"][model]
    if "unsupported" in config:
        save(lane / "terminal.json", {"status": "unsupported", "reason": config["unsupported"],
                                      "scheduled": 560, "transport_calls": 0})
        return
    client = None
    try:
        client = await create_client(config["settings"], approval, lane)
        await client.initialize()
        selected = [r for r in rows if r["dataset"] == dataset]
        for split in ("valid", "test"):
            jobs = [r for r in selected if r["split"] == split]
            jobs.sort(key=lambda row: value_hash([SEED, "run-order", row["request_sha256"]]))
            for row in jobs:
                approval_check(approval, root, manifest, model, dataset)
                outcome = await execute_one(root, lane, row, config["settings"], client,
                                            approval.get("deployment"))
                if outcome["status"] != "success":
                    save(lane / "terminal.json", {"status": "halted", "phase": split,
                                                  "reason": "protocol_failure_no_retry"})
                    return
        save(lane / "terminal.json", {"status": "completed", "scheduled": 560})
    except BaseException as exc:
        if not (lane / "terminal.json").exists():
            save(lane / "terminal.json", {"status": "blocked", "error_type": type(exc).__name__})
        raise
    finally:
        if client is not None:
            await client.aclose()


def metric_rows(rows, labels, outcomes, dataset):
    n = len(rows)
    counts = Counter({state: 0 for state in
                      ("success", "failure", "unsupported", "not_attempted", "attempted_unknown")})
    counts.update(outcome["status"] for outcome in outcomes)
    hit = pool_hit = 0
    recall = 0.0
    truth, predictions = [], []
    for row, outcome in zip(rows, outcomes):
        answers = set(map(str, labels[str(row["target"])]))
        candidates = {option["id"] for option in row["request"]["options"]}
        overlap = answers & candidates
        pool_hit += bool(overlap)
        recall += len(overlap) / len(answers)
        prediction = outcome.get("selected_option_id") if outcome["status"] == "success" else None
        hit += prediction in answers
        truth.append(next(iter(answers)))
        predictions.append(prediction)
    result = {"scheduled": n, "states": dict(counts), "successful": counts["success"],
              "attempted": sum(o.get("transport_entered", o["status"] in
                                    ("success", "failure", "attempted_unknown")) for o in outcomes),
              "Hit@1_unconditional": hit / n if n else None,
              "Hit@1_pool_hit_conditional": hit / pool_hit if pool_hit else None,
              "pool_hit_queries": pool_hit, "candidate_Hit@40": pool_hit / n if n else None,
              "candidate_Recall@40": recall / n if n else None}
    if dataset == "arxiv":
        f1 = []
        for label in map(str, range(40)):
            tp = sum(t == p == label for t, p in zip(truth, predictions))
            denominator = sum(t == label for t in truth) + sum(p == label for p in predictions)
            f1.append(2 * tp / denominator if denominator else 0)
        result.update(accuracy=result["Hit@1_unconditional"], macro_f1=sum(f1) / 40)
    return result


def report(root):
    manifest, requests = verify(root)
    labels, reused = read(root / "labels.json"), read(root / "arxiv/reuse.json")
    report_data = {"manifest_sha256": digest(root / "manifest.json"), "models": {},
                   "full200_retained": "arxiv/original200-all-results.json"}
    for model, config in manifest["models"].items():
        report_data["models"][model] = {}
        for dataset in ("arxiv", "prime"):
            scores = {}
            for split in ("valid", "test"):
                for arm in ARMS:
                    rows = [r for r in requests if r["dataset"] == dataset and
                            r["split"] == split and r["arm"] == arm]
                    outcomes = []
                    for row in rows:
                        key = f'{model}/{split}/{row["target"]}/{arm}'
                        lane = root / "runs" / dataset / model / split / f'{row["target"]}.{arm}'
                        if "unsupported" in config:
                            outcome = {"status": "unsupported"}
                        elif dataset == "arxiv" and model not in NEW:
                            old = reused[key]
                            outcome = old["outcome"] or {"status": old["status"]}
                        elif (lane / "outcome.json").exists():
                            outcome = read(lane / "outcome.json")
                            require(outcome["request_sha256"] == row["request_sha256"],
                                    "Unbound new result")
                            intent = read(lane / "intent.json")
                            require(intent["request"] == row["request"] and
                                    intent["request_sha256"] == row["request_sha256"] and
                                    intent["settings_sha256"] == value_hash(config["settings"]) ==
                                    outcome["settings_sha256"], "Unbound new intent/config")
                            approval = read(lane.parent.parent / "approval.json")
                            require(intent["approval_sha256"] == digest(lane.parent.parent / "approval.json"),
                                    "Deployment approval changed after request intent")
                            require(approval["manifest_sha256"] == digest(root / "manifest.json") and
                                    approval["settings_sha256"] == value_hash(config["settings"]) and
                                    approval["authorization_sha256"] == PARENT_AUTH_SHA,
                                    "Result approval binding differs")
                            for filename, expected_hash in outcome["wire_capture_sha256"].items():
                                require(Path(filename).name == filename and filename.startswith("wire-"),
                                        "Invalid wire capture path")
                                require(digest(lane / filename) == expected_hash and
                                        read(lane / filename)["request_sha256"] == row["request_sha256"],
                                        "Wire capture binding differs")
                            if outcome["status"] == "success":
                                raw = read(lane / "raw-response.json")
                                require(digest(lane / "raw-response.json") ==
                                        outcome["raw_response_sha256"], "Raw response changed")
                                require(raw["selected_option_id"] == outcome["selected_option_id"] and
                                        raw["request_id"] == row["request"]["request_id"] and
                                        raw["resolved_model"] == approval.get("deployment", {}).get(
                                            "served_model", config["settings"]["model"]) and
                                        outcome["scientific_model"] == config["settings"]["model"],
                                        "Raw response/result identity mismatch")
                        elif (lane / "intent.json").exists():
                            outcome = {"status": "attempted_unknown"}
                        else:
                            outcome = {"status": "not_attempted"}
                        outcomes.append(outcome)
                    scores[f"{split}/{arm}"] = metric_rows(rows, labels[dataset][split], outcomes, dataset)
            report_data["models"][model][dataset] = scores
    full200 = read(root / "arxiv/original200-all-results.json")
    old_labels = read(root / "arxiv/original200-labels.json")
    report_data["original_arxiv_200_separate"] = {}
    for model in (name for name in manifest["models"] if name not in NEW):
        by_key = {(r["split"], str(r["target"]), r["arm"]): r for r in full200.get(model, [])}
        scores = {}
        for arm in ARMS:
            rows = [{"target": target, "request": {"options": [{"id": str(i)} for i in range(40)]}}
                    for target in sorted(old_labels["test"], key=int)]
            outcomes = [by_key.get(("test", row["target"], arm), {
                "status": "unsupported" if "unsupported" in manifest["models"][model] else "not_attempted"})
                        for row in rows]
            scores[arm] = metric_rows(rows, {k: [v] for k, v in old_labels["test"].items()},
                                     outcomes, "arxiv")
        report_data["original_arxiv_200_separate"][model] = scores
    return report_data


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("freeze")
    commands.add_parser("verify")
    reporting = commands.add_parser("report")
    reporting.add_argument("--output", type=Path)
    running = commands.add_parser("run")
    running.add_argument("--model", required=True)
    running.add_argument("--dataset", choices=("arxiv", "prime"), required=True)
    running.add_argument("--approval", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "freeze":
        freeze(args.root)
        print(canonical({"frozen": str(args.root), "manifest_sha256": digest(args.root / "manifest.json")}))
    elif args.command == "verify":
        manifest, rows = verify(args.root)
        print(canonical({"verified_requests": len(rows), "models": len(manifest["models"])}))
    elif args.command == "run":
        asyncio.run(run(args.root, args.model, args.dataset, args.approval))
    else:
        result = report(args.root)
        if args.output:
            save(args.output, result)
        else:
            print(canonical(result))


if __name__ == "__main__":
    main()
