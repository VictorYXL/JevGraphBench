# GraphDecide

### Benchmarking System One Models on Graph Tasks

GraphDecide separates **graph cognition**, **graph-text decision-making**, and
**sequential solution quality**. Task definitions, legal candidates and scoring
are independent of the model adapter. Native selectors, constrained generation
and candidate scoring are different interfaces, not interchangeable evidence
of reasoning ability or computational efficiency.

**Current manuscript: v33, October 5, 2026.** This README describes its primary
panels. Earlier public, synthetic and 200-target graph-text experiments remain
separate supporting studies; they must not be pooled with the primary results.
The repository and Python distribution retain the name `JevGraphBench` /
`jevgraphbench`.

| Primary panel | Evaluation data | Scheduled units per configuration |
| --- | --- | ---: |
| **RQ1 — Graph Cognition** | Six structural tasks; induced graphs from four real networks; 8–12 vertices | **800 queries** |
| **RQ2 — Graph-Text Decision-Making** | 100 ogbn-arxiv targets and 100 STaRK-Prime queries, each under five matched input conditions | **1,000 query–condition pairs** |
| **RQ3 — Sequential Decision-Making** | TSP, signed MaxCut, deterministic LT and binary modularity; ten graphs per task, two construction modes | **80 trajectories** |

Fourteen model-interface configurations are evaluated. Unsupported interfaces
remain visible; they are not zero scores. Repeated conditions are not
independent graphs. There is no overall score combining classification
accuracy, percentage optimization gaps and absolute modularity gaps.

[RQ1](#rq1--graph-cognition) · [RQ2](#rq2--graph-text-decision-making) ·
[RQ3](#rq3--sequential-decision-making) · [Model settings](#model-settings) ·
[Offline reproduction](#reproduce-the-distributed-v33-results-offline) ·
[Running models](#installation-and-new-model-evaluations) ·
[Release scope](#release-scope-and-historical-results)

## RQ1 — Graph Cognition

Queries cover adjacency, exact degree, cycle existence, pair connectivity,
distance thresholds and articulation points. The four sources are
**Facebook, ca-GrQc, Power Grid and Human PPI**, with 200 queries per source.
Explicit vertex sets preserve isolates. Ground truth is computed on exactly
the supplied simple undirected graph using independent algorithms.

The 800-query cohort contains 400 degree queries and 80 for each other task.
At size `n`, each source contributes `2n` degree queries and four queries for
each binary task. Consequently, **pooled task accuracy is not generally the
size-equal task score**.

The primary task score averages accuracy equally over sizes 8, 9, 10, 11 and 12:

| Configuration | Adjacency | Degree | Cycle | Connectivity | Distance | Articulation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Jev-1.13.0 | 100.00 | 67.80 | 87.50 | 88.75 | 90.00 | 61.25 |
| Decider-2B v11 | 72.50 | 9.92 | 78.75 | 50.00 | 61.25 | 50.00 |
| Kev-4B | 88.75 | 10.20 | 85.00 | 60.00 | 83.75 | 62.50 |
| Laya | 50.00 | 10.38 | 50.00 | 51.25 | 50.00 | 50.00 |
| Qwen3.5-0.8B | 50.00 | 12.09 | 52.50 | 50.00 | 50.00 | 50.00 |
| Qwen3.5-2B | 50.00 | 14.75 | 50.00 | 50.00 | 50.00 | 50.00 |
| Qwen3.5-4B generation | 78.75 | 40.40 | 80.00 | 50.00 | 50.00 | 51.25 |
| Qwen3.5-9B | 57.50 | 34.74 | 81.25 | 73.75 | 75.00 | 50.00 |
| Qwen3.5-27B | 100.00 | 63.20 | 87.50 | 87.50 | 80.00 | 70.00 |
| Qwen3.8-27B | 96.25 | 55.66 | 85.00 | 86.25 | 72.50 | 51.25 |
| Qwen2.5-72B-Instruct | 98.75 | 67.08 | 78.75 | 88.75 | 91.25 | 72.50 |
| Qwen3.5-4B scoring | 90.00 | 17.53 | 78.75 | 50.00 | 61.25 | 55.00 |
| GPT-5.4, no reasoning | 100.00 | 91.42 | 92.50 | 96.25 | 97.50 | 75.00 |
| GPT-6-Astra, reasoning reference | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |

Values are percentages from v33 Figure 2 / Appendix C. Jev's perfect adjacency
score does not extend to exact degree or articulation. Larger-model comparisons
are task-specific, rather than a universal ordering.

For GPT-5.4, the distinct aggregate definitions give:

| Aggregation | Accuracy (%) |
| --- | ---: |
| Query weighted: 733/800 | 91.6250 |
| Task macro: pool within each task, then average six tasks | 92.0417 |
| Task-size macro: equal weight for every task and its five sizes | 92.1109 |

Its degree score is **91.4154% size-equal**, versus **91.0000% pooled**.
Invalid outputs count as incorrect. All 800 final GPT-5.4 outputs are legal.
[Per-query data](assets/benchmark/v33/gpt54/rq1.csv) includes source and size.

## RQ2 — Graph-Text Decision-Making

Each dataset has 100 fixed test targets/queries and five paired conditions:

- **T:** text; **G:** relational input; **TG:** joint text and relations.
- **BAG:** the same nodes and text as TG, without explicit edges.
- **A / A\*:** the no-additional-entity-text/edge comparator. On ogbn-arxiv,
  A retains anonymous training-label anchors; on Prime, A* retains the query
  and candidate identity grounding. CSVs encode both as `A`.

ogbn-arxiv uses all forty classes and shared training-label anchors. Context
identities are graph-selected even in T, so T is not graph-independent.
TG−T adds both context text and edges, not just edges.

STaRK-Prime is **selection within forty gold-containing candidates**, not
end-to-end retrieval. Starting from BM25 top-40, a missing native answer is
deterministically inserted by replacing the lowest-ranked candidate; the set
is shuffled and its evidence rebuilt without answer markers. Any native
correct answer is accepted. Candidate Hit@40 is 100% by design and candidate
Recall@40 is 77.7241%. The older label-blind Prime pool is not this primary cohort.

| Configuration | arXiv T / G / TG / BAG / A | Prime T / G / TG / BAG / A* |
| --- | --- | --- |
| Jev-1.13.0 | 55 / 61 / 58 / 57 / 51 | 48 / 47 / 49 / 46 / 42 |
| Decider-2B v11 | 35 / 45 / 42 / 38 / 24 | 20 / 17 / 18 / 21 / 20 |
| Kev-4B | 29 / 61 / 27 / 27 / 41 | 39 / 19 / 26 / 35 / 34 |
| Laya | U / U / U / U / U | U / U / U / U / U |
| Qwen3.5-0.8B | 32 / 34 / 26 / 24 / 33 | 9 / 9 / 9 / 7 / 8 |
| Qwen3.5-2B | 25 / 12 / 19 / 23 / 15 | 16 / 10 / 16 / 12 / 13 |
| Qwen3.5-4B generation | 21 / 59 / 30 / 19 / 42 | 36 / 17 / 24 / 28 / 26 |
| Qwen3.5-9B | 32 / 55 / 41 / 36 / 39 | 43 / 25 / 31 / 38 / 33 |
| Qwen3.5-27B | 46 / 61 / 58 / 56 / 49 | 50 / 44 / 46 / 43 / 41 |
| Qwen3.8-27B | 62 / 60 / 71 / 70 / 51 | 53 / 50 / 51 / 51 / 48 |
| Qwen2.5-72B-Instruct | 51 / 58 / 59 / 52 / 45 | 39 / 35 / 33 / 35 / 36 |
| Qwen3.5-4B scoring | U / U / U / U / U | U / U / U / U / U |
| GPT-5.4, no reasoning | 73 / 61 / 73 / 72 / 48 | 60 / 51 / 53 / 56 / 56 |
| GPT-6-Astra, reasoning reference | 73 / 60 / 76 / 73 / 55 | 69 / 69 / 67 / 73 / 71 |

Values are arXiv accuracy / Prime Hit@1 (%) from v33 Table 1. `U` means
unsupported, not zero. arXiv macro-F1 uses a fixed forty-class denominator,
including zero F1 for classes absent from both truth and predictions.

The principal paired contrasts are **TG−BAG** and **G−A/A\***. For Jev,
arXiv G−A is +10 pp [3, 18], but TG−BAG is +1 pp [−4, 7].
Prime TG−BAG is +3 pp [−6, 12]. Thus high joint-input performance and
additional benefit from explicit edges are different observations.

[GPT-5.4 paired rows](assets/benchmark/v33/gpt54/rq2.csv) retain the same
`(dataset, query_id)` across conditions. Its 603 correct outcomes include
two remaining illegal Prime/TG outputs scored as incorrect, not dropped.
The offline reporter reproduces its pointwise 95% paired intervals using
1,000 resamples, seed 20261001, numeric query-ID order, common index draws
across contrasts, and sorted endpoints 24 and 974. Marginal correct totals
alone cannot reconstruct these intervals.

## RQ3 — Sequential Decision-Making

Each configuration has four tasks × ten graphs × two modes:

- **A — Direct construction:** all currently legal next actions.
- **C — Heuristic-proposal selection:** distinct next actions nominated by
  four named, fixed rules. Duplicate actions are merged. A singleton is forced
  and incurs **no model call**. Proposals are actions, not completed solutions.

| Task | Instances | Reference and gap |
| --- | --- | --- |
| TSP | Ten TSPLIB EUC_2D graphs, 76–225 cities | Published proven optimum; `100 × (objective − reference) / reference` |
| Signed MaxCut | Ten OPTSICOM Set2 graphs, 125 vertices, 375 edges, weights ±1 | Historical 2016 BKS, not necessarily an optimum; `100 × (reference − objective) / reference` |
| Deterministic LT | Ten directed email-Eu-core induced graphs, 50–150 vertices; two seeds activated together | Exact exhaustive seed-pair optimum; the same maximizing percentage gap |
| Binary modularity | Ten ca-GrQc induced graphs, 50–150 vertices; at most two communities, resolution 1 | Numerical MILP-solver-certified optimum; **absolute** `reference − objective` |

Lower gaps are better. Negative signed-cut objectives can give gaps above
100%; do not clamp them. A MILP numerical certificate is not a formal proof
log. Coverage always retains every scheduled trajectory. Mean quality excludes
incomplete/unsupported trajectories without imputing an objective; an empty
feasible subset has undefined quality. Paired A/C comparisons use only graphs
completed in both modes, and controls must use that same subset.

### Jev and matched controls

| Task | A mean gap | C mean gap | Within-pool random | Classical | Feasible, each mode |
| --- | ---: | ---: | ---: | ---: | --- |
| TSP (%) | 59.6641 | 25.8731 | 43.8850 | 5.7532 | 10/10 |
| Signed MaxCut (%) | 99.9614 | 9.0517 | 82.7206 | 15.4096 | 10/10 |
| Deterministic LT (%) | 52.8359 | 0.5000 | 21.9123 | 6.8438 | **8/10** |
| Binary modularity (absolute Q) | 0.4784 | 0.0555 | 0.4175 | 0.4784 | 10/10 |

Values follow v33 Appendix E. LT controls here use Jev's **eight completed
graphs**, not all ten. Jev completes 76/80 trajectories; the four failures
are `max_tokens_exceeded` HTTP errors on the two 150-node LT instances in
both modes. Across fourteen configurations, 951/1,120 complete, 165 are
unsupported and four fail. Forced-only completion does not establish support
for a model interface that was never called.

Proposals help relative to direct construction, but do not establish
superiority over fixed rules: TSP `return_reserve` reaches 15.5805%, MaxCut
`greedy_completion` 8.8727%, and modularity `greedy_completion` 0.0361 Q.
Random-candidate controls average five seeds within each graph before
averaging graphs. Non-GPT modularity tables use four-decimal per-graph
objectives, limiting the meaningful precision of their rebased means.

### GPT-5.4 no-reasoning results

| Task | A mean gap | C mean gap | Feasible A / C | Selected model calls A / C |
| --- | ---: | ---: | --- | --- |
| TSP (%) | 29.485870 | 32.244044 | 10/10 / 10/10 | 1,257 / 668 |
| Signed MaxCut (%) | 99.667296 | 9.411631 | 10/10 / 10/10 | 1,240 / 987 |
| Deterministic LT (%) | 33.431162 | 5.475000 | 10/10 / 10/10 | 20 / 12 |
| Binary modularity (absolute Q) | 0.404240 | 0.042286 | 10/10 / 10/10 | 990 / 704 |

[All 80 final trajectories](assets/benchmark/v33/gpt54/rq3.csv) include graph
IDs, objectives, reference values, gap units, final status and model calls.
The ten-graph TSP/A mean must not be replaced by the older nine-completion
mean. Modularity values must not be compared against historical heuristic
reference gaps without rebasing.

**Selection and retry accounting.** These are final selected results, not a
single-attempt success-rate claim. Invalid-only retries retain a legal result
even if wrong; unresolved invalids stay invalid. RQ1 had no retries; RQ2 had
nine initial invalids, 84 additional calls and two final invalids; RQ3 had one
initial invalid trajectory, two restarts costing 211 additional calls, and no
final invalids. Result CSVs contain no attempt-count columns.
[Separate retry statistics](assets/benchmark/v33/gpt54/retry_summary.csv)
retain those costs; `model_calls` in the trajectory CSV counts only the selected
trajectory. No incorrect-but-legal answer is replaced for better quality.

## Model settings

- **Native selection:** Jev-1.13.0, Decider-2B v11, Kev-4B and Laya retain
  their native support constraints. An unsupported candidate count or context
  is not repaired by shrinking the task.
- **Qwen generation:** thinking disabled, candidate-constrained generation;
  configured output tokens are **64 for RQ1/RQ3** and **16 for RQ2**.
  Qwen3.5-4B candidate-token scoring is a separate interface.
- **GPT-5.4:** `github_copilot`, model `gpt-5.4`, `think: false`,
  `reasoning_effort: "none"`, `output_format: function_call`,
  `max_tokens: 4096`, request timeout 600 s. Temperature, seed and sampling
  overrides are unset. A fresh decision session exposes only `submit_decision`.
  Accepted calls require reported effort `none` and zero reasoning tokens.
- **GPT-6-Astra:** a reasoning-enabled reference. Historical `think` and
  `reasoning_effort` are unset; **medium was observed**, not explicitly
  requested. It is not a matched no-thinking configuration. A tested explicit
  `none` request was rejected; no low-effort replacement experiment was run.

The string `"none"` is not YAML/Python null. `low` still reasons. A configured
4,096 output-token value is not a verified total reasoning-token limit.
The Copilot tool enum is not verified strict server-side constrained decoding,
unlike the local Qwen choice constraint; report legality separately.
Cloud and local hardware paths are not matched speed comparisons.

## Reproduce the distributed v33 results offline

The included [v33 GPT-5.4 CSV bundle](assets/benchmark/v33/gpt54/) has
**800 / 1,000 / 80** final rows. It contains result metadata, not prompts,
credentials or raw provider responses. Provenance and checksums are documented
in its [README](assets/benchmark/v33/gpt54/README.md).

From the repository root:

```bash
python -m src.utils.public_results_v33 \
  --data-dir assets/benchmark/v33/gpt54 > gpt54-v33-summary.json
```

This performs **no inference or downloads**. It validates IDs, structural
source/size quotas, paired conditions, legality, gap units and A/C coverage.
It produces source/task/size scores, all three RQ1 aggregate definitions,
RQ2 paired intervals and RQ3 quality/call summaries. It cannot recover
macro-F1 from binary correctness alone; that requires labels and predictions.
It does not claim raw-response replay or all-fourteen-model reproduction.

## Installation and new model evaluations

Python 3.11 or newer:

```bash
python -m pip install -e .
# Optional, only for the GitHub Copilot provider:
python -m pip install -e '.[copilot]'
```

### Core smoke evaluation

[configs/pilot.yaml](configs/pilot.yaml) is a small adjacency/distance
demonstration, **not the v33 six-task 800-query cohort**:

```bash
python run_benchmark.py --config configs/pilot.yaml
```

**This makes live model calls and may download datasets.** Jev reads
`TYPESAFE_API_KEY`. Copilot accepts `COPILOT_GITHUB_TOKEN` or an existing CLI
login; no token belongs in YAML or source code. Existing-login mode uses
`COPILOT_HOME` (default `~/.copilot`) for authentication while keeping decision
workspaces and sessions isolated. Explicit-token mode retains an isolated
temporary runtime home. No automatic device-login flow is started.

For a Copilot smoke run, replace only the `model` section in a copy of the
configuration:

```yaml
model:
  provider: github_copilot
  model: gpt-5.4
  think: false
  reasoning_effort: "none"
  output_format: function_call
  max_tokens: 4096
  timeout_seconds: 600
```

Use a new output directory and verify provider/model availability before
inference. This example demonstrates the adapter, not the exact paper cohort.
`python -m src.benchmark --config ...` delegates to the same YAML entrypoint.

### Offline task development

```bash
python -m src.utils.extended_graph_suite plan \
  --root output/smoke-plan --design structural_dev \
  --node-counts 8 --samples-per-size 1 --models jev
python -m src.utils.extended_graph_suite report --root output/smoke-plan
```

Planning/reporting do not call models. This synthetic development plan is
not a primary v33 result. Its `run` command performs inference; `analyze`
replays sealed trajectories. Frozen runs require their matching source and
artifacts; never rewrite archived manifests to bypass an integrity failure.
See [data/README.md](data/README.md) for real-network downloads and normalization.

### Tests

Focused, offline manuscript and adapter regression checks:

```bash
python -m unittest tests.test_public_results_v33 tests.test_llm_clients \
  tests.test_benchmark -q
```

The full offline suite is `python -m unittest discover -s tests -v`.
Tests use mocked providers; they do not authenticate or start benchmark inference.

## Release scope and historical results

| Path | Role |
| --- | --- |
| [src/benchmark/](src/benchmark/) | Task contracts, candidate rules, evaluators and core runners |
| [src/clients/](src/clients/) | Jev, vLLM and optional Copilot adapters |
| [src/utils/public_results_v33.py](src/utils/public_results_v33.py) | Portable offline reporting for the final v33 CSV schema |
| [src/utils/public_structural_v32.py](src/utils/public_structural_v32.py) | Historical public structural sampling, execution and replay |
| [src/utils/public_graphtext_v32.py](src/utils/public_graphtext_v32.py) | Historical arXiv/label-blind Prime preparation and scoring |
| [src/utils/public_optimization_v32.py](src/utils/public_optimization_v32.py) | Public optimization construction and audited trajectory utilities |
| [assets/benchmark/v33/](assets/benchmark/v33/) | Current distributed final-result subset |
| [assets/benchmark/](assets/benchmark/) | Separately labeled historical result assets |

The `*_v32` filenames identify the implementation's origin; renaming them
would break frozen provenance without changing a protocol. Their historical
freeze/run commands depend on **separately supplied local artifacts and
deployment approvals**. In particular, the old RQ2 freeze's label-blind Prime
pool and default-reasoning GPT settings are not the revised v33 primary
configuration. Do not run those defaults and call the result a v33 reproduction.
The original optimization freeze also retains honest heuristic modularity
references; eight of ten later certified optima improve on those references.
v33 gaps require the certified overlay rather than relabeling the old bounds
as optima.
The gold-containing Prime revision and no-thinking rerun orchestration are
preserved in the local experiment bundle, not presented as fresh-clone commands.

Historical [public optimization](assets/benchmark/public-summary.csv),
[graph-text](assets/benchmark/graph-text.csv),
[action abstraction](assets/benchmark/abstraction-summary.csv) and
[four-task proposals](assets/benchmark/proposal-summary.csv) remain unchanged.
They use different cohorts, conditions and sometimes reference values.
Their figure/data assets are supporting evidence, not updated v33 primary
figures. The independent 200-target arXiv experiment is not pooled with the
100-target primary panel.

The ignored `output/` directory contains local manuscripts, raw ledgers and
machine-specific orchestration; it is not supplied by a fresh clone.
Model weights, credentials and provider access are not distributed.
Recomputing a table from included CSVs, independently replaying raw decisions,
running new inference and compiling the paper are four different operations.
This repository does not claim a one-command replay of every historical
model or a public manuscript-source bundle.
