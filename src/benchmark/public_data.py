"""Explicit, checksum-pinned preparation of complete public optimization graphs.

No downloads occur on import. ``prepare`` writes only into a new directory and
never repairs, crops, or deduplicates input graphs. ``fetch(url) -> bytes`` can be
injected for offline tests. References and provenance remain outside public state.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import re
from urllib.parse import urlsplit

import httpx

from .public_tasks import validate_instance

DEFAULT_CATALOG = Path(__file__).resolve().parents[2] / "data/optimization-catalog.json"
MAX_FILE_BYTES = 2 * 1024 * 1024
MAX_CATALOG_ENTRIES = 64


def _https(url: str) -> None:
    parsed = urlsplit(url)
    if (parsed.scheme != "https" or not parsed.hostname
            or parsed.username is not None or parsed.password is not None):
        raise ValueError("public source must be a credential-free HTTPS URL")


def fetch_public_bytes(url: str) -> bytes:
    """Bounded HTTPS download, without environment credentials or automatic retries."""
    with httpx.Client(timeout=60, trust_env=False, follow_redirects=False) as client:
        for _ in range(6):
            _https(url)
            with client.stream("GET", url) as response:
                if response.is_redirect:
                    if "location" not in response.headers:
                        raise ValueError("public-data redirect has no Location")
                    url = str(response.url.join(response.headers["location"]))
                    continue
                response.raise_for_status()
                content = bytearray()
                for chunk in response.iter_bytes():
                    content.extend(chunk)
                    if len(content) > MAX_FILE_BYTES:
                        raise ValueError("public file exceeds size limit")
                return bytes(content)
    raise ValueError("too many public-data redirects")


def _lines(data: bytes) -> list[str]:
    if not isinstance(data, bytes) or len(data) > MAX_FILE_BYTES:
        raise ValueError("expected bounded source bytes")
    try:
        return [line.strip() for line in data.decode("ascii").splitlines() if line.strip()]
    except UnicodeDecodeError as exc:
        raise ValueError("public graph must be ASCII text") from exc


def parse_tsplib_euc_2d(data: bytes, expected_n: int) -> dict:
    """Parse a full 1-based TSPLIB EUC_2D instance; preserve binary-double coordinates."""
    if type(expected_n) is not int or not 3 <= expected_n <= 254:
        raise ValueError("TSP dimension must be 3..254")
    lines = _lines(data)
    if lines.count("NODE_COORD_SECTION") != 1:
        raise ValueError("expected exactly one NODE_COORD_SECTION")
    split = lines.index("NODE_COORD_SECTION")
    headers = {}
    for line in lines[:split]:
        if ":" not in line:
            raise ValueError("invalid TSPLIB header")
        key, value = (part.strip() for part in line.split(":", 1))
        if key in headers and key != "COMMENT":
            raise ValueError("duplicate TSPLIB header")
        headers[key] = value
    if headers.get("TYPE") != "TSP" or headers.get("EDGE_WEIGHT_TYPE") != "EUC_2D":
        raise ValueError("only TYPE TSP and EDGE_WEIGHT_TYPE EUC_2D are supported")
    if int(headers.get("DIMENSION", "0")) != expected_n:
        raise ValueError("TSPLIB dimension disagrees with catalog")
    coordinates = {}
    rows = lines[split + 1:]
    if rows and rows[-1] == "EOF":
        rows = rows[:-1]
    for line in rows:
        parts = line.split()
        if len(parts) != 3:
            raise ValueError("invalid TSPLIB coordinate row")
        node = int(parts[0])
        xy = [float(parts[1]), float(parts[2])]
        if node in coordinates:
            raise ValueError("duplicate TSPLIB node")
        if not all(math.isfinite(value) for value in xy):
            raise ValueError("nonfinite TSPLIB coordinate")
        coordinates[node] = xy
    if set(coordinates) != set(range(1, expected_n + 1)):
        raise ValueError("TSPLIB nodes must be exactly 1..dimension")
    return {"nodes": list(range(expected_n)),
            "coordinates": [coordinates[i] for i in range(1, expected_n + 1)],
            "edge_weight_type": "EUC_2D"}


def parse_maxcut(data: bytes, expected_n: int, expected_edges: int) -> dict:
    """Parse signed integer weights; reject loops and duplicate undirected edges."""
    if (type(expected_n) is not int or not 2 <= expected_n <= 254
            or type(expected_edges) is not int
            or not 1 <= expected_edges <= expected_n * (expected_n - 1) // 2):
        raise ValueError("invalid bounded MaxCut dimensions")
    lines = _lines(data)
    if not lines or len(lines[0].split()) != 2:
        raise ValueError("invalid MaxCut header")
    n, m = map(int, lines[0].split())
    if (n, m) != (expected_n, expected_edges) or len(lines) - 1 != m:
        raise ValueError("MaxCut counts disagree with catalog or header")
    seen, edges = set(), []
    for line in lines[1:]:
        parts = line.split()
        if len(parts) != 3:
            raise ValueError("invalid MaxCut edge row")
        u, v, weight = map(int, parts)
        if not 1 <= u <= n or not 1 <= v <= n or u == v:
            raise ValueError("MaxCut edge has invalid nodes or a self-loop")
        u, v = sorted((u - 1, v - 1))
        if (u, v) in seen:
            raise ValueError("duplicate undirected MaxCut edge")
        seen.add((u, v))
        edges.append([u, v, weight])
    return {"nodes": list(range(n)), "edges": sorted(edges)}


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _check_output(path: Path) -> None:
    if ".." in path.parts:
        raise ValueError("output path may not contain '..'")
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError("output path may not traverse symlinks")
    if path.exists():
        raise FileExistsError("public-data output must be a new directory")
    if not path.parent.is_dir():
        raise ValueError("output parent must be an existing directory")


def _catalog(path: Path) -> tuple[dict, bytes]:
    raw = path.read_bytes()
    catalog = json.loads(raw)
    if catalog.get("schema_version") != 1:
        raise ValueError("unsupported public catalog version")
    rows = catalog.get("instances")
    if not isinstance(rows, list) or not 1 <= len(rows) <= MAX_CATALOG_ENTRIES:
        raise ValueError("catalog requires a bounded instance list")
    seen = set()
    for row in rows:
        name = row.get("id", "")
        if not isinstance(name, str) or re.fullmatch(r"[A-Za-z0-9_-]+", name) is None or name in seen:
            raise ValueError("invalid or duplicate catalog ID")
        seen.add(name)
        if row.get("task") not in ("tsp_public", "maxcut_public"):
            raise ValueError("unsupported catalog task")
        if row.get("role") not in ("main", "calibration"):
            raise ValueError("unsupported catalog role")
        if row.get("compression") not in ("none", "gzip"):
            raise ValueError("unsupported catalog compression")
        for field in ("wire_sha256", "raw_sha256"):
            if re.fullmatch(r"[0-9a-f]{64}", str(row.get(field, ""))) is None:
                raise ValueError("catalog requires SHA256 pins")
        for field in ("source_url", "download_url"):
            _https(row[field])
        reference = row["reference"]
        if (set(reference) != {"value", "status", "source"}
                or type(reference["value"]) not in (int, float)
                or not math.isfinite(reference["value"]) or reference["value"] <= 0
                or reference["status"] not in ("proven_optimum", "best_known")):
            raise ValueError("invalid catalog reference")
        _https(reference["source"])
    return catalog, raw


def _unpack(wire: bytes, row: dict) -> bytes:
    if not isinstance(wire, bytes) or len(wire) > MAX_FILE_BYTES:
        raise ValueError("download must return bounded bytes")
    if _digest(wire) != row["wire_sha256"]:
        raise ValueError(f"wire SHA256 mismatch for {row['id']}")
    if row["compression"] == "gzip":
        try:
            with gzip.GzipFile(fileobj=io.BytesIO(wire)) as stream:
                raw = stream.read(MAX_FILE_BYTES + 1)
        except (OSError, EOFError) as exc:
            raise ValueError("invalid gzip source") from exc
    else:
        raw = wire
    if len(raw) > MAX_FILE_BYTES or _digest(raw) != row["raw_sha256"]:
        raise ValueError(f"raw SHA256 mismatch or size limit for {row['id']}")
    return raw


def prepare(
    output_dir: str | Path,
    catalog_path: str | Path = DEFAULT_CATALOG,
    include_calibration: bool = False,
    *,
    dataset_ids: Sequence[str] | None = None,
    fetch: Callable[[str], bytes] | None = None,
) -> dict:
    """Download selected files explicitly and freeze lists plus ``provenance.json``.

    By default select all 17 main graphs. ``dataset_ids`` restricts main IDs;
    ``include_calibration=True`` additionally exports calibration in a separate
    list, never the main aggregate. Returns the persisted provenance manifest.
    All downloads/hashes/parsing validate before creating the new output directory.
    An interrupted write is never treated as a resumable cache.
    """
    output = Path(output_dir).absolute()
    _check_output(output)
    catalog, catalog_bytes = _catalog(Path(catalog_path))
    main_ids = {row["id"] for row in catalog["instances"] if row["role"] == "main"}
    selected_ids = main_ids if dataset_ids is None else set(dataset_ids)
    if (isinstance(dataset_ids, (str, bytes)) or not selected_ids
            or not selected_ids <= main_ids
            or (dataset_ids is not None and len(selected_ids) != len(dataset_ids))):
        raise ValueError("dataset_ids must be unique catalog main IDs")
    rows = [row for row in catalog["instances"]
            if row["id"] in selected_ids or (include_calibration and row["role"] == "calibration")]
    download = fetch if fetch is not None else fetch_public_bytes
    main, calibration, receipts, raw_files = [], [], [], {}
    for row in rows:
        wire = download(row["download_url"])
        raw = _unpack(wire, row)
        if row["task"] == "tsp_public":
            if (row.get("edge_weight_type") != "EUC_2D"
                    or row["edge_count"] != row["dimension"] * (row["dimension"] - 1) // 2):
                raise ValueError("invalid TSP catalog distances or edge count")
            state = parse_tsplib_euc_2d(raw, row["dimension"])
            suffix = ".tsp"
        else:
            state = parse_maxcut(raw, row["dimension"], row["edge_count"])
            suffix = ".mc"
        raw_name = f"raw/{row['id']}{suffix}"
        record = {
            "id": row["id"], "task": row["task"], "state": state,
            "private": {"reference": row["reference"]},
            "metadata": {
                "role": row["role"], "complete_original_graph": True,
                "source_url": row["source_url"], "download_url": row["download_url"],
                "raw_path": raw_name, "raw_sha256": row["raw_sha256"],
                "reference_as_of": row.get("reference_as_of"),
                "reference_note": row.get("reference_note"),
                "license_note": catalog.get("license_note"),
            },
        }
        validate_instance(record)
        (main if row["role"] == "main" else calibration).append(record)
        raw_files[raw_name] = raw
        if row["compression"] == "gzip":
            raw_files[raw_name + ".gz"] = wire
        receipts.append({"catalog_entry": row, "raw_path": raw_name,
                         "wire_bytes": len(wire), "raw_bytes": len(raw)})
    payloads = {"instances.json": main}
    if include_calibration:
        payloads["calibration-instances.json"] = calibration
    files = {name: (json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
             for name, data in payloads.items()}
    files.update(raw_files)
    files["catalog.json"] = catalog_bytes
    manifest = {
        "schema_version": 1, "catalog_sha256": _digest(catalog_bytes),
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "main_ids": [row["id"] for row in main],
        "calibration_ids": [row["id"] for row in calibration],
        "instances_path": "instances.json",
        "calibration_path": "calibration-instances.json" if include_calibration else None,
        "hash_convention": "wire_sha256 pins downloaded gzip/plain payload; raw_sha256 pins decompressed original file",
        "normalization": "Complete graphs, 1-based raw IDs to 0-based; no cropping, candidate filtering, loop removal or duplicate merging; signed weights preserved; double sqrt+nint EUC_2D in task core",
        "sources": receipts,
        "files": {name: {"sha256": _digest(data), "bytes": len(data)}
                  for name, data in files.items()},
    }
    _check_output(output)
    output.mkdir(mode=0o700)
    (output / "raw").mkdir()
    for name, data in files.items():
        with (output / name).open("xb") as stream:
            stream.write(data)
    with (output / "provenance.json").open("x") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return manifest
