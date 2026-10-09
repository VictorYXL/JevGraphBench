# Data: sources, downloads and public results

This directory contains source provenance, an optimization dataset catalog and
the canonical public result bundle. Downloading and loading source graphs do
not perform subgraph sampling, question generation, labeling or model calls.

## Directory guide and result scope

| Resource | Contents |
| --- | --- |
| [manifest.json](manifest.json) | Source URLs, citations, archive checksums and normalized statistics for the four real networks below |
| [optimization-catalog.json](optimization-catalog.json) | TSPLIB and signed-MaxCut source definitions, pinned hashes, distance conventions and reference provenance |
| [results/summary.json](results/summary.json) | Primary manuscript-transcribed aggregates for fourteen model-interface configurations |
| [results/README.md](results/README.md) | Aggregate schema, metrics, provenance and interpretation limits |

**Public results are aggregate-only for fourteen model-interface configurations
across RQ1, RQ2 and RQ3.** Per-query and per-trajectory outputs are not
distributed. Charts and tables render from the summary, not raw response replay.
The optimization catalog describes source datasets and their references, not
the membership of every primary result cohort or individual trajectory outcomes.

Historical and supplementary result assets are not distributed in the current
compact bundle. They describe independent cohorts with potentially different
sampling, candidate pools and references, and have not been merged into the
primary results. See [reproduction and scope](../docs/reproduction.md) for
offline reporting and the limits of fresh-clone reproduction.

## Available datasets

The following counts were measured after loading and normalizing the data using the rules below. They are not model input sizes.

| Dataset ID | Domain | Nodes | Undirected non-loop edges | Components | Removed self-loop records |
| --- | --- | ---: | ---: | ---: | ---: |
| `facebook` | Social friendships | 4,039 | 88,234 | 1 | 0 |
| `ca-grqc` | Scientific collaboration | 5,242 | 14,484 | 355 | 12 |
| `power` | Power-grid infrastructure | 4,941 | 6,594 | 1 | 0 |
| `human-ppi` | Human protein interactions | 21,557 | 338,636 | 26 | 3,717 |

See [manifest.json](manifest.json) for source/download URLs, citations, timestamps, archive sizes, SHA-256 checksums, and loading statistics. Pinned source definitions are in [sources.py](../src/datasets/sources.py).

### Original releases

1. [SNAP ego-Facebook](https://snap.stanford.edu/data/ego-Facebook.html): the combined edge list, not the full ego package containing personal attributes. It combines 10 ego networks and does not represent the entire platform.
2. [SNAP ca-GrQc](https://snap.stanford.edu/data/ca-GrQc.html): collaboration among arXiv GR-QC authors; nodes represent authors, not papers.
3. [Newman's network data page](https://websites.umich.edu/~mejn/netdata/): the original HTTPS [power.zip](https://websites.umich.edu/~mejn/netdata/power.zip), containing the Western US power grid compiled by Watts and Strogatz. The archive retains its description and citation. This GML release is used instead of the SuiteSparse Matrix Market link that redirects to HTTP. Do not mix node identifiers across releases.
4. [BioSNAP Human PPI](https://snap.stanford.edu/biodata/datasets/10000/10000-PP-Pathways.html): an experimental human protein-interaction edge list with no CSV header.

These are historical snapshots, not live networks. Published counts may differ from normalized counts: the ca-GrQc page lists 14,496 edges; its raw file has 28,980 edge records, including 14,484 duplicate/reverse records removed during normalization and 12 self-loops. The PPI release has 342,353 records; removing 3,717 self-loops leaves 338,636 edges.

## Storage and licensing

- Raw archives remain in `data/raw/`, unchanged and without filesystem extraction.
- The archives total approximately 2 MiB. No profiles, product text, or credentials are downloaded.
- Explicit redistribution licenses have not been verified on the source pages. Public availability does not imply unrestricted redistribution. Raw archives are therefore Git-ignored; provenance, checksums, statistics, and download code can be versioned.
- Datasets do not inherit the project's code license. Check source terms and citation requirements before publishing archives or derived subgraphs.
- Manifest hashes pin downloaded bytes; they are not publisher signatures. Future downloads must match instead of silently accepting upstream replacements.

## Download and verify

After installing the project in editable mode, run from the repository root:

```bash
python -m src.datasets.download
python -m src.datasets.download --verify-only
```

To select datasets or use a different directory:

```bash
python -m src.datasets.download --datasets facebook power
python -m src.datasets.download --data-dir /path/to/data
```

Existing files are verified before use; a matching cache requires no network request. Corrupt files are never overwritten automatically: explicitly move or remove them before downloading again. Downloads use temporary files and atomic installation after checksum verification. HTTPS is required and HTTP downgrade redirects are rejected. Failures are reported without automatic retries. No API key is needed.

The dataset downloader's `--verify-only` option checks local archives and prints statistics without network access or file changes. It is separate from the benchmark runner, which directly evaluates Jev. The default data directory is the checkout's `data/`, not the working directory; other installation layouts should set `--data-dir` / `data_dir` explicitly. Configurations made before the directory flattening may still name the old nested path; update new configurations, but never rewrite archived run configurations or their hashes.

## Unified interface

```python
from src.datasets import available_datasets, load_graph

print(available_datasets())
dataset = load_graph("facebook")
graph = dataset.graph

print(dataset.source.domain, dataset.source.source_url)
print(dataset.raw_sha256, dataset.stats)
print(graph.number_of_nodes(), graph.number_of_edges())
print(list(graph.nodes)[:5])
print(list(graph.edges)[:5])
print(list(graph.neighbors(0))[:5])

# Use NetworkX algorithms directly; this does not invoke a model.
import networkx as nx
print(nx.shortest_path_length(graph, source=0, target=1))
```

`load_graph()` reads local data only, verifies the archive hash on every load, and never downloads implicitly. It returns a `GraphDataset`:

- `graph`: a structurally frozen NetworkX graph. All four sources are simple, undirected, and unweighted, with integer node IDs. Use `graph.copy()` before changing structure.
- `source`: dataset ID, domain, source/download URLs, citation, published counts, and license note.
- `stats`: raw edge records, duplicate/self-loop removals, final node/edge counts, isolates, and components.
- `raw_sha256`: the verified raw archive hash.

The loader uses NetworkX rather than a separate adjacency format and does not create oversized model payloads. Provenance and statistics are separate from topology. Do not include source metadata or labels when constructing model inputs.

## Fixed normalization rules

1. Preserve original integer node IDs. Do not relabel, sample, or extract the largest component in the loader. The benchmark sampler later records an anonymous mapping.
2. Treat these sources as undirected according to their documented semantics. The ca-GrQc file header says Directed, but the source page describes undirected collaboration.
3. Merge `(u,v)` with `(v,u)` and remove duplicate edges, recording the number removed.
4. Remove self-loops while retaining all observed nodes. This leaves one isolate in ca-GrQc and 19 in PPI; they are not silently discarded.
5. Removing self-loops is a simple-topology benchmark convention, not a claim that PPI self-interactions are biologically meaningless. Future self-interaction tasks need a separate protocol using the retained raw release.
6. Sort nodes and edges deterministically by integer IDs. This ordering is not a perturbation protocol.
7. Edge lists cannot recover isolated nodes absent from the source file. Explicit GML nodes, including isolates, are retained.
8. Only the registered unweighted sources are supported. Physical weights are not invented, and unexpected GML weights are rejected rather than silently discarded. The directed normalization branch preserves edge direction, but no directed source is currently registered.

Unweighted shortest paths measure hops, not electrical flow, physical distance, or biological propagation strength. Subgraph task labels must be computed on exactly the graph shown to the model.

## Tests

```bash
python -m unittest discover -s tests -t . -v
```

Tests are grouped into unit, integration and publication packages. Dataset tests
use temporary small graphs and `httpx.MockTransport`, without real archives or
internet access. They cover duplicates, self-loops, isolates, direction, GML,
CSV, checksum failures, redirects, caching, interrupted downloads, manifests,
and offline verification. Optional historical-fixture checks are disabled by
default; see [tests/README.md](../tests/README.md).