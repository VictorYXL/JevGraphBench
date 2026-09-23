# JevGraphBench

A configuration-driven, **topology-only graph decision benchmark**: real networks → induced subgraphs → anonymization → questions and algorithmic ground truth → Jev predictions → accuracy, latency, and failure statistics.

The current tasks are adjacency and shortest-distance threshold decisions. There are no perturbation experiments, training, NLGraph inputs, or model-generated labels. **Running the benchmark directly calls Jev and may incur API charges.** There is no preparation-only CLI mode. Mocks are used only in tests and are never reported as Jev results.

## 1. Configuration

- [configs/adjacency.yaml](configs/adjacency.yaml): **adjacency only**, 960 planned/generated questions.
- [configs/distance_threshold.yaml](configs/distance_threshold.yaml): **distance threshold only**, 3,840 planned / 3,492 generated questions across thresholds 2/3/4/6.
- Both single-task configurations use four real networks, 32/64/128 nodes, and twenty subgraphs per source and size. Each has its own call ceiling and output directory.
- [configs/pilot.yaml](configs/pilot.yaml) and [configs/comprehensive.yaml](configs/comprehensive.yaml) remain as historical mixed-task configurations for reproducing earlier runs. Prefer the single-task configurations for new runs. See the [expanded evaluation protocol](docs/comprehensive.md) for coverage and limitations.
- [Dataset sources and download instructions](data/real/README.md). Benchmark runs never download data implicitly.

Experiment configurations use **YAML** (`.yaml` / `.yml`) with a strict PyYAML SafeLoader subclass. Duplicate or non-string keys, unsafe object tags, multiple documents, and merge keys (`<<`) are rejected. Unknown fields, duplicate sources/sizes, odd question counts, and invalid values also fail explicitly. Quote strings to avoid implicit YAML type conversion. Legacy TOML experiment configurations are not supported; Python packaging still uses [pyproject.toml](pyproject.toml).

| Parameter | Meaning |
|---|---|
| `data.datasets` | Source IDs: facebook, ca-grqc, power, human-ppi |
| `data.data_dir` | Dataset root containing the raw archive subdirectory |
| `sampling.node_counts` | List of subgraph node counts |
| `sampling.samples_per_size` | Subgraphs sampled **per source and per size** |
| `sampling.max_attempts` | Maximum attempts to find a distinct node set for one sample; not model retries |
| `run.seed` | Seed for sampling, anonymous IDs, and node-pair selection |
| `tasks.adjacency_questions` | Questions per subgraph; even count, or 0 to disable |
| `tasks.distance_thresholds` | Distance thresholds, each at least 2 to avoid duplicating adjacency |
| `tasks.distance_questions_per_threshold` | Questions per subgraph **and per threshold**; even count |
| `run.repetitions` | Calls per identical question; neither new graph sampling nor majority voting |
| `run.max_calls` | Hard ceiling checked against the planned call count before sampling or network access |
| `run.output_dir` | New output directory; existing directories are never overwritten or appended to |
| `model.provider` / `model.model` | Use `typesafe` / `jev-latest`, or an available pinned Jev version; resolved versions are logged |
| `model.timeout_seconds` | Total per-prediction deadline including network I/O; also passed to the HTTP client |

Paths in YAML are relative to the **configuration file's directory**. CLI `--output` paths are relative to the current working directory.

Planned call ceiling = sources × sizes × samples per size × (adjacency questions + thresholds × questions per threshold) × repetitions.

The pilot plans 60 graphs × 4 questions × 1 repetition = **at most 240 calls**. Unavailable question groups reduce the actual call count; the runner never silently truncates a run to fit the limit.

## 2. Install and run

From the repository root, install the package in your Python environment:

```bash
python -m pip install -e .
```

Make sure the selected raw datasets have been downloaded. Set `TYPESAFE_API_KEY` securely in your terminal environment; do not place credentials in YAML, source code, chat, or logs.

Run adjacency evaluation only:

```bash
python run_benchmark.py --config configs/adjacency.yaml
```

Run distance-threshold evaluation only:

```bash
python run_benchmark.py --config configs/distance_threshold.yaml
```

The default output directories are `results/adjacency-v1` and `results/distance-threshold-v1`. Use `--output` with a new directory when repeating a run. Each directory contains its own accuracy, latency, coverage, and attempt logs.

Splitting tasks does not change graph sampling, anonymous IDs, questions, labels, or the requested model. Both configurations reuse the comprehensive seed and sampling settings; their generated records were verified against the corresponding subsets of the completed mixed run. Their union contains the same 4,452 questions. Full request-file hashes differ because each file now contains only one task. Existing results remain valid; merely changing configuration organization does not require rerunning paid predictions. Running either command again does make new API calls.

To disable adjacency, set `tasks.adjacency_questions` to 0. To disable distance questions, set `tasks.distance_questions_per_threshold` to 0 and `tasks.distance_thresholds` to an empty list. Multiple thresholds are difficulty levels within one distance task, not separate task types. To evaluate just one threshold, retain only that value in the distance list and adjust the call ceiling if desired. The loader still accepts historical mixed configurations for compatibility.

These commands **prepare the data and immediately run Jev evaluation**. No additional execution flag is required. Configuration validation, source checksum verification, ground-truth validation, and the call ceiling remain enforced.

### Terminal progress

Progress is enabled by default. A preparation message appears while loading graphs, sampling, and verifying labels; this stage has no percentage or ETA. Evaluation then displays a `tqdm` progress bar with finalized attempts / actual total calls, percentage, elapsed time, estimated remaining time, calls per second, running accuracy, and failures. The total is **generated questions × repetitions**, not the planned call ceiling. Display refreshes are throttled to approximately once per second during calls, with a final update on completion or abort. ETA is an estimate based on observed throughput, not a guarantee.

Accuracy includes all recorded attempts, including failed or interrupted calls. A failed request advances progress because it was attempted; it does not count as a correct answer. Aborted runs retain partial progress and print an aborted status rather than claiming completion. Preparation and evaluation timings remain separate in the saved artifacts.

Progress goes to **stderr**; stdout remains the final JSON summary. Add `--no-progress` to suppress the display, especially when redirecting stderr to a plain log file. Library calls to `run_experiment()` stay quiet unless passed `progress=True`. No credentials, provider error bodies, or question payloads are printed by the progress display. This feature does not change sampling, requests, retries, or result schemas.

[run_benchmark.py](run_benchmark.py) is the recommended script entry point for a repository checkout. It delegates to the existing CLI without duplicating benchmark logic or modifying the import path. Module-based invocation remains supported, but is not required; neither launch style changes evaluation behavior or constitutes a package release.

`model.provider: "typesafe"` creates the Jev API adapter through the client registry and calls the Choice interface at `POST https://api.typesafe.ai/v1/systemone`. `model.model` selects the requested version; the response's resolved version is recorded. Missing credentials fail before HTTP client creation; there is no fallback to a mock model.

Use a new output directory for every run. Each run regenerates samples from its configuration instead of loading another run's prepared files. With the same seed, source data, and dependency versions, request/label artifacts are reproducible and can be compared by hash.

Execution is **sequential, concurrency 1, with no retries, warm-up, voting, or resume**. Repetitions run in rounds across all questions and are reported separately. A timeout or cancellation may still incur a provider charge. A forcibly killed process cannot guarantee a final summary, but previously flushed attempt records may remain available.

## 3. Topology-only inputs and label isolation

The model-visible `state` has exactly two allowed fields:

- `nodes`: anonymous integers from 0 to n−1;
- `edges`: undirected edges between anonymous endpoints.

The graph, nodes, and edges are rebuilt from scratch with **no original attributes copied**. Original IDs, names, domains, dataset names, descriptions, categories, and geographic information are not sent. Each sample receives one anonymous mapping; this is de-identification, not a perturbation experiment.

Questions contain only necessary task instructions: the graph is undirected/unweighted, the query endpoints and threshold, and the requirement to use only the supplied graph. Choice options are `yes` / `no`. Task instructions are not external semantic attributes.

Local request IDs may contain source names for bookkeeping; **the TypeSafe adapter does not send local request IDs to the provider**. Original ID mappings, true distances, answers, and provenance remain in separate evaluator-side artifacts. Full sample records are never used as model payloads.

## 4. Sampling and tasks

1. Select a starting node uniformly among nodes in components large enough for the requested size.
2. Repeatedly select uniformly from the external neighbor frontier until reaching the target node count.
3. Retain **all original edges between selected nodes**, not only a spanning tree, then anonymize.
4. Reject identical source node sets within each source/size group, with a bounded resampling budget.

This is biased connected-region sampling, not uniform sampling over all induced subgraphs. Subgraphs can overlap and are not independent samples or train/test splits. Current sources are normalized undirected, unweighted simple graphs. Isolated nodes do not enter these connected subgraphs.

Within each subgraph and task/threshold, select distinct node pairs without replacement using equal yes/no quotas:

- **Adjacency:** positive pairs have an edge; negative pairs do not. Degree matching is not implemented.
- **Distance threshold k:** positive pairs have distance exactly k; negative pairs have distance exactly k+1. This tests boundary decisions, not a uniform distribution over all distances or unreachable pairs.

If either class has too few candidates, **skip the entire question group and record candidate counts and missing questions**. Keep the sampled graph; do not alter edges or repeatedly select graphs based on question availability. Failed graph sampling is also recorded. Generation coverage and model accuracy are reported separately.

Labels are generated with BFS. Before export, the final anonymous payload is reconstructed and independently checked with Floyd–Warshall. Models never generate ground truth.

## 5. Artifacts and metrics

Each run writes to its own directory; local results are Git-ignored:

JSON artifacts use two-space indentation for readability. JSONL artifacts retain one compact record per line for streaming and incremental logging. This formatting applies to new runs; existing artifacts are not rewritten, preserving their recorded hashes.

| Artifact | Contents |
|---|---|
| config.json | Resolved configuration with absolute paths, without credentials |
| run.json | Configuration hash, protocol/dependency versions, timestamps, status, and artifact hashes |
| sources.json | Source provenance, raw SHA256, and normalization statistics |
| graphs.jsonl | Anonymous topology, original ID mappings, and sampling metadata; evaluator-only |
| requests.jsonl | Label-free local DecisionRequest records, not complete wire-level POST bodies |
| labels.jsonl | Answers, distances, source and difficulty metadata; evaluator-only |
| preparation.json | Planned/actual graph and question counts, per-source/size/task/threshold coverage (including zero-coverage cells), and missing groups |
| attempts.jsonl | Per-call predictions, probabilities, token usage, latency, error type, and repetition |
| summary.json | Overall, per-task/source/size/threshold/repetition metrics, plus joint strata with generation coverage and null accuracy for untested cells |

Primary metrics:

- `accuracy`: correct predictions / **all recorded attempts**, including invalid responses, timeouts, HTTP failures, and interrupted calls.
- `accuracy_on_success`: correct predictions / valid responses; supplementary, not a substitute for the primary metric.
- `failure_rate`: unsuccessful attempts / recorded attempts. Error types and HTTP statuses are retained, but error bodies are not.
- `latency_all` / `latency_success`: mean, p50, p95, and total seconds for all/successful calls. Measured with a monotonic clock, including network and response parsing, excluding data preparation and log writes. This is not model-internal compute time.
- `token_usage`: known token sums and reporting-call counts; unavailable values remain null, not zero.
- `brier_yes`: binary Brier score for responses with probabilities. Provider confidence is logged separately and is not treated as an answer probability.
- `execution_coverage`: recorded attempts / (generated questions × repetitions). `completed_calls` counts finalized attempt records, including failures and interruptions, not only successful responses.
- `cost`: currently null; pricing is not inferred or invented.

A completed run may still contain failures; inspect `run_status`, coverage, and failure rate together. Cancellation records the in-flight attempt and propagates. Programming errors abort rather than being hidden as recoverable model failures. Summaries from aborted runs are partial results, not complete evaluations. Failures before data preparation completes may leave only configuration and run-status artifacts.

Raw HTTP response bodies and authorization headers are not saved; only normalized evaluation fields are retained. Statistical confidence intervals are not implemented, and questions/repetitions from the same graph are not independent samples. Overall accuracy is attempt-weighted, so missing groups change its weighting. Compare models using identical request artifact hashes.

## 6. Tests and current scope

```bash
python -m unittest discover -s tests -v
```

Tests use temporary fixtures and mocked HTTP only. They cover attribute stripping, payload leakage, deterministic sampling, balanced quotas, label verification, safe YAML parsing, duplicate keys, call ceilings, failures/timeouts/cancellation, output protection, and the complete YAML → CLI → real registry/TypeSafe adapter → MockTransport → metrics pipeline. No credentials or live service are required to run the tests.

The default pilot's verified data-generation result is 60 graphs and 230/240 questions: five Facebook subgraphs lack distance-3 pairs, each omitting two threshold questions. These are dataset counts, not Jev performance results.

Perturbations, degree matching, cross-split deduplication, reasoning-budget comparisons, and a second model adapter are not included. The expanded configuration broadens the two existing decision tasks, not the entire space of graph algorithms. Inspect per-stratum coverage and costs before drawing capability conclusions.