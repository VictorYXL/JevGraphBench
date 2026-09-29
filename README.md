# JevGraphBench

### Benchmarking Decision Models on Graphs

**Do correct local decisions compose into good graph solutions?**
JevGraphBench evaluates Jev, open trained decision models, bounded Qwen
configurations and separate GPT reasoning references using exactly verifiable
graph tasks.

| Evaluated configurations | Graph tasks | Shared instances | Graph sizes |
|:---:|:---:|:---:|:---:|
| **10 direct-decision + 2 reasoning references** | **6 single-step + 4 sequential** | **960 per configuration** | **8–12 vertices** |

**Results snapshot: September 27, 2026** · 11,520 scheduled episode evaluations ·
720 model × task × size metric cells

[Evaluation pipeline](#what-the-benchmark-measures) · [Key findings](#key-findings) · [Exact decisions](#exact-decisions) ·
[Sequential construction](#sequential-construction) ·
[Trajectory diagnosis](#when-does-optimality-become-unreachable) ·
[Reasoning references](#reasoning-references) ·
[Evaluation protocol](#evaluation-protocol) ·
[Download results (CSV)](assets/benchmark/benchmark.csv) ·
[Run the benchmark](#run-the-benchmark)

> **A taskwise comparison, not a universal leaderboard.** Higher accuracy and
> lower objective gaps are better, but gaps have different units across tasks.
> We do not combine them into one overall score. GPT references use different
> reasoning budgets and are not compute-matched to the direct-decision models.

## What the benchmark measures

A bounded output can be valid but wrong. A complete sequence of legal actions
can also produce a poor solution. Final accuracy or a single completion score
cannot distinguish these outcomes.

![Evaluation pipeline: public graph state, bounded choice, actual-history transitions and privileged offline verification. No oracle values enter model requests.](assets/benchmark/evaluation-pipeline.svg)

JevGraphBench addresses this measurement problem at three levels:

1. **Independent truth:** graph queries have mathematically checked answers,
   rather than labels defined by a model's own predictions.
2. **End-to-end quality:** sequential constructions have exact optimal objectives;
   gap and completion coverage are reported separately.
3. **Where quality is lost:** exact continuation analysis measures the best
   solution still reachable after each recorded action. A later locally optimal
   decision cannot undo an earlier irreversible loss.

## Key findings

| Observation | Evidence |
| --- | --- |
| **Local recognition is not structural reasoning.** | Jev reaches **100% adjacency** accuracy, but **61.43% degree** and **58.75% distance-threshold** accuracy. |
| **Optimization quality depends on the task.** | Jev has a smaller TSP gap than Qwen3.5-27B grammar (**45.85 vs 100.90**); Qwen has a smaller deterministic LT gap (**0.425 vs 0.600**). |
| **Legal completion is not optimality.** | Jev completes **160/160** constructions, yet its MaxCut gap (**8.150**) exceeds the exact uniform-policy reference (**5.325**). |
| **Trajectories explain losses hidden by final scores.** | The first TSP choice excludes every optimal completion in **3/40 Jev** episodes versus **22/40 Qwen3.5-4B grammar** episodes in the original four-configuration analysis. |

## Exact decisions

**Accuracy (%), higher is better.** Each cell averages accuracy across graph
sizes 8–12. This is **size-macro accuracy**, not pooled accuracy; the distinction
matters for degree, whose number of answer classes grows with graph size.
All twelve configurations return valid answers on all 800 exact queries, but
a valid answer can still be wrong.

![Exact-task accuracy heatmap for ten direct-decision configurations and two separately grouped GPT reasoning references. Full values are in the table below.](assets/benchmark/exact-accuracy.svg)

<details>
<summary><strong>View exact-task values and model interfaces</strong></summary>

Best direct-decision point estimates are **bold** within each column; this
does not imply statistically significant differences. Row order is fixed,
not an overall ranking.

| Model | Interface | Adjacency | Degree | Cycle | Connectivity | Distance | Articulation |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Jev-1.13.0 | Native | **100.00** | **61.43** | **66.25** | 85.00 | 58.75 | **65.00** |
| Qwen3.5-4B | Grammar | 60.00 | 18.32 | 55.00 | 50.00 | 50.00 | 50.00 |
| Qwen3.5-9B | Grammar | 58.75 | 26.99 | 51.25 | 53.75 | 42.50 | 50.00 |
| Qwen3.5-27B | Grammar | 98.75 | 47.70 | 63.75 | 80.00 | 52.50 | 47.50 |
| Qwen3.8-27B | Grammar | 90.00 | 45.50 | 61.25 | 85.00 | 48.75 | 50.00 |
| Qwen2.5-72B-Instruct | Grammar | 92.50 | 41.24 | 50.00 | **86.25** | **60.00** | 46.25 |
| Qwen3.5-4B | Candidate scoring | 77.50 | 23.89 | 51.25 | 51.25 | 52.50 | 50.00 |
| Decider-2B | Native | 58.75 | 12.09 | 50.00 | 50.00 | 46.25 | 50.00 |
| Kev-4B | Native | 78.75 | 19.88 | 57.50 | 58.75 | 51.25 | 45.00 |
| Laya | Native | 50.00 | 10.87 | 50.00 | 50.00 | 50.00 | 50.00 |

The five binary tasks have 80 queries each; degree has 400. A 50% binary
chance reference does not apply to multiclass degree.

</details>

## Sequential construction

**Mean additive objective gap, lower is better.** Each model follows its own
previous choices. Deterministic code applies legal actions without supplying
oracle choices, repairing incorrect responses or revising earlier decisions.
Zero gap means the completed solution is optimal.

![Construction objective-gap heatmap. Each cell includes feasible/scheduled coverage, and each task has its own color scale and units. Full values are in the table below.](assets/benchmark/construction-gap.svg)

<details>
<summary><strong>View construction values, units and completion coverage</strong></summary>

Gaps are conditional on **feasible completed episodes**, not all scheduled
episodes. Best direct-decision point estimates are **bold**. Coverage is part
of the result, not a reason to remove failed cases.

| Model | Interface | MaxCut ↓ | Manhattan TSP ↓ | Deterministic LT ↓ | Modularity ↓ | Completed |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Jev-1.13.0 | Native | 8.150 | **45.85** | 0.600 | **0.1694** | 160/160 |
| Qwen3.5-4B | Grammar | 11.125 | 338.15 | 2.625 | 0.2900 | 160/160 |
| Qwen3.5-9B | Grammar | 11.050 | 306.45 | 3.725 | 0.3203 | 160/160 |
| Qwen3.5-27B | Grammar | **8.025** | 100.90 | **0.425** | 0.2810 | 160/160 |
| Qwen3.8-27B | Grammar | 11.150 | 183.70 | 2.675 | 0.3012 | 160/160 |
| Qwen2.5-72B-Instruct | Grammar | 9.775 | 152.40 | 1.950 | 0.2645 | 160/160 |
| Qwen3.5-4B | Candidate scoring | 9.200 | 351.00 | 4.125 | 0.2810 | 160/160 |
| Decider-2B | Native | 11.125 | 355.70 | 4.100 | 0.2810 | 160/160 |
| Kev-4B | Native | 11.075 | 349.65 | 0.600 | 0.2502 | 160/160 |
| Laya | Native | 15.250 | 344.75 | 2.871 | 0.3039 | **151/160** |

| Task | Objective | Gap units | Scheduled per model |
| --- | --- | --- | ---: |
| MaxCut | Maximize crossing edges | Edges | 40 |
| Manhattan TSP | Minimize closed-tour distance | Manhattan distance | 40 |
| Deterministic normalized LT | Maximize activation from two seeds | Activated vertices | 40 |
| At-most-two-community modularity | Maximize modularity | Modularity | 40 |

**Laya's LT gap uses 31/40 feasible episodes.** Nine inputs exceed its lossless
native context/head limits; they remain in the scheduled denominator.
Its other construction tasks each have 40/40 coverage. Different coverage
does not constitute a matched-subset comparison.

</details>

## When does optimality become unreachable?

For each recorded prefix, compute the **best full solution still reachable**
without revising earlier decisions. The loss at a step is the deterioration
in that value after the chosen action. For a complete trajectory, these losses
sum to its final additive objective gap. This is retrospective evaluation,
**not** oracle-assisted inference or a new optimization algorithm.

![A real Jev TSP trajectory: optimum 192, first irreversible loss at step 2, final tour length 216 and total gap 24.](assets/benchmark/trajectory.svg)

This is **one actual 12-city Jev episode**, selected as the first Jev TSP
entry in the preserved analysis ledger, not as a representative performance
estimate. Step 2 raises the best reachable tour length from **192 to 194**.
At step 4, the chosen action is locally optimal, but the best reachable value
remains **200**, already worse than the global optimum. The final tour length
is **216**, with an additive gap of **24**.

The original four-configuration analysis covers **640 complete construction
episodes and 4,480 decisions**. All step losses were replayed with the core
offline analyzer and match the historical diagnostic values. This analysis
does not imply that all twelve configurations have the same trajectory audit.

[Replayable example (JSON)](assets/benchmark/trajectory-example.json) ·
[First-step loss counts (CSV)](assets/benchmark/first-loss.csv) ·
[Offline analyzer](src/benchmark/trajectory.py)

<details>
<summary><strong>Replay this recorded example locally, without model calls</strong></summary>

After installing the core dependencies, run from the benchmark root:

```bash
python - <<'PY'
import json
from pathlib import Path
from src.benchmark.trajectory import analyze_trajectory

example = json.loads(Path("assets/benchmark/trajectory-example.json").read_text())
result = analyze_trajectory(example["instance"], example["decisions"])
assert result == example["analysis"]
print("First loss:", result["first_loss_step"])
print("Final gap:", result["absolute_gap"])
PY
```

Expected output: first loss **2**, final gap **24**. The example includes
privileged references for offline verification; **do not send the whole JSON
to a model**. The standard request builder allowlists public state only.

`analyze_trajectory(instance, decisions)` also accepts incomplete legal
histories. Their final objective and final gap remain `None`; an unavoidable
prefix gap is not a completed solution. Invalid actions or corrupt reference
records raise an error rather than being repaired.

</details>

## Reasoning references

GPT models are **separate reference points**, not members of a matched-budget
direct-decision ranking. Both use tool-free default reasoning with a
4,096-output-token ceiling and a 180-second request deadline.

| Model | Each of the six exact-task accuracies | MaxCut gap / coverage | TSP gap / coverage | LT gap / coverage | Modularity gap / coverage | Construction completion |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GPT-5.4 | 100.00% | 0.000 / 36/40 | 0.00 / 39/40 | 0.000 / 40/40 | 0.0000 / 36/40 | **151/160** |
| GPT-6-Astra | 100.00% | 0.000 / 40/40 | 0.00 / 40/40 | 0.000 / 40/40 | 0.0000 / 40/40 | **160/160** |

GPT-5.4's nine construction timeouts are retained. Its zero gaps describe
completed solutions, **not** success on every scheduled episode.

## Evaluation protocol

- **Shared bank:** 960 instances per configuration: 800 exact queries and
  160 sequential constructions. The 11,520 evaluations reuse these instances;
  they are not 11,520 independent graphs.
- **Exact verification:** small graphs permit deterministic answers and exact
  optimal objectives. For maximization, gap = optimum − solution; for
  minimization, gap = solution − optimum.
- **Interfaces:** Grammar means no-thinking, greedy choice-constrained
  generation with a 64-token ceiling. Candidate scoring is a separate
  Qwen3.5-4B readout, not a different model or a trained Jev-like selector.
  Native denotes each model's own selection interface, not identical internals.
  Decider-2B uses the v11 checkpoint.
- **Failure accounting:** unsupported inputs, timeouts and invalid outputs
  remain in scheduled denominators. No silent retries or answer repair.
- **Scope:** these are point estimates on an inspected bank, not proof of
  large-graph generalization, architecture superiority or matched-hardware
  speed. Only the original four configurations have the separate 952-instance
  new-graph confirmation; this is not a twelve-configuration replication.
- **Candidate-scoring caveat:** its CPU/GPU SemIf probability reproduction
  check failed the registered tolerance. Treat it as an independent interface
  reference, not a certified numerical reproduction.

### Relation to other decision benchmarks

[JevAdvBench](https://arxiv.org/abs/2609.31142) studies how input interventions
change typed decisions relative to a clean answer and an identical-request
noise baseline. Our complementary question is whether decisions are
mathematically correct and compose into good solutions, using exact graph
oracles and actual-history continuation values.
Decision stability, correctness and optimization quality are different
properties; neither benchmark substitutes for the other. Our presentation
controls are not a comprehensive adversarial evaluation.

### Data behind the figures

The [aggregate CSV](assets/benchmark/benchmark.csv) contains all **720**
model/task/size rows, including per-size results, feasible and scheduled counts,
failures, mean gaps and optimum-hit rates. `nodeCount=all` denotes the aggregate;
raw experimental identifiers are preserved even where display names are expanded.
No new model calls were made to produce this page.

<details>
<summary>Snapshot identity and data integrity</summary>

- Evidence date: **2026-09-27**.
- Immutable aggregate version: `canonical-20260927-v14`.
- Original numeric results, coverage and failures are unchanged. Display names
  distinguish model identity from inference settings; “Qwen4 G” is not a model.
- CSV task and model IDs map to the configurations shown above. For example,
  `qwen25_72b_bf16` is Qwen2.5-72B-Instruct with grammar generation, and
  `gpt6astra_default_reasoning` is GPT-6-Astra.

SHA-256 checksums (CSV included here; original JSON retained in the research archive):

```text
benchmark.csv
0b5b91f5dcdba51e21319e1aa72e188247522735a0a3dc7ba8245c05e93dfbf7

benchmark.json
e00fac08430afb1d11ec21a58ede8bd0537a0e2bf874022ad5d7f5f2409f8616
```

</details>

## Run the benchmark

The results above are a fixed experimental snapshot, not a live service.
This checkout retains the runnable core; historical all-model orchestration
and paper-production tools remain outside it. The Python distribution name
is still `jevgraphbench`.

<details>
<summary><strong>Installation, configuration, repository layout and offline tests</strong></summary>

### Repository layout

| Path | Purpose |
| --- | --- |
| [src/benchmark/](src/benchmark/) | Task generation, exact scoring, offline trajectory analysis and experiment runner |
| [src/clients/](src/clients/) | Jev, vLLM and optional GitHub Copilot clients |
| [src/datasets/](src/datasets/) | Real-network download, parsing and verification |
| [configs/](configs/) | YAML configuration templates |
| [data/real/](data/real/) | Dataset metadata; downloaded archives are ignored |
| [scripts/](scripts/) | Three required runtime/audit utilities, described below |
| [tests/](tests/) | Core regression tests only |
| `output/`, `results/` | Ignored local artifacts; never required merely to import the core |

Paper sources, PDFs, the website, historical experiments, publication scripts
and their tests remain in ignored local `output/`. Only the compact result
figures, aggregate CSV and replayable example used by this README are included
under [assets/benchmark/](assets/benchmark/).
No weights, credentials or raw provider ledgers are distributed here.

Run the following commands from the benchmark root containing
`run_benchmark.py` and `configs/`, so Python imports this project's `src` package.

### Install

Python 3.11 or newer:

```bash
python -m pip install -e .
```

For the optional GitHub Copilot provider:

```bash
python -m pip install -e '.[copilot]'
```

### Real-network evaluation

Inspect [configs/pilot.yaml](configs/pilot.yaml) and
[data/real/README.md](data/real/README.md) before running:

```bash
python run_benchmark.py --config configs/pilot.yaml
```

**This starts live evaluation and may download the configured datasets.**
The Jev client reads `TYPESAFE_API_KEY` from the environment; do not put
credentials in configuration files. The alternative module entry point is
`python -m src.benchmark --config configs/pilot.yaml`.
Provider, endpoint, model and inference settings are explicit in the YAML
configuration. Do not assume that a locally running endpoint serves the
requested model.

### Synthetic ten-task suite

Exact queries: adjacency, degree, cycle detection, connectivity, distance
thresholds and articulation points. Sequential constructions: MaxCut,
Manhattan TSP, deterministic normalized linear-threshold influence, and
modularity with at most two communities.

Create a small, independently checked plan **without model calls**:

```bash
python scripts/extended_graph_suite.py plan \
  --root output/smoke-plan --design structural_dev \
  --node-counts 8 --samples-per-size 1 --models jev
python scripts/extended_graph_suite.py report --root output/smoke-plan
```

Use a new output directory for each plan; existing plans are not overwritten.
The `structural_holdout` design additionally requires an explicit historical
graph pool via `--prior-pool`. Supported model lanes and options are listed by
`python scripts/extended_graph_suite.py plan --help`.
The lane `gpt6astra` denotes GPT-6-Astra, not an additional model.

After checking the generated configuration and provider availability, the
following command makes **live model calls**:

```bash
python scripts/extended_graph_suite.py run --root output/smoke-plan --models jev
```

Analyze a completed or interrupted **sealed** core-suite run without model
calls or modifications to its artifacts:

```bash
python scripts/extended_graph_suite.py analyze \
  --root output/smoke-plan --models jev
```

This performs the same integrity checks and replay as `report`, then adds
per-construction action values, incremental losses and the first loss step.
Original scheduled denominators, failures and incomplete histories are retained.
Unstarted lanes are explicitly marked `not_started`; they are not successful
trajectories. Analysis does not resume inference or provide oracle advice.
Like `report`, it refuses changed frozen source or corrupt artifacts: use the
matching source snapshot for old runs rather than editing their manifests.
The bundled JSON example can be replayed independently of historical runners.

The retained scripts are:

- [extended_graph_suite.py](scripts/extended_graph_suite.py): plan, run, report and offline analyze.
- [paired_graph_ablation.py](scripts/paired_graph_ablation.py): shared model
  configuration, preflight, locking and integrity utilities required by the suite.
- [audit_task_shortcuts.py](scripts/audit_task_shortcuts.py): offline label and
  simple-feature audits used by the structural task tests.

Historical twelve-configuration orchestration and publication tools are
archived, not part of this minimal CLI. The core suite does not claim to
reproduce every historical model lane from a single command.

### Scoring and failures

Report exact-answer accuracy separately from legal completion.
For constructions, report mean additive objective gap on feasible completed
episodes **alongside feasible/scheduled coverage**. Maximization uses
`optimum - solution`; minimization uses `solution - optimum`.
Gap units differ across tasks and must not be pooled into one score.
Timeouts, unsupported requests and invalid outputs are not silently repaired
or removed from scheduled denominators.

### Tests

The retained tests use offline fixtures and mocked providers:

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

No credentials, model servers, paper builders, browser tooling or archived
experiment outputs are required for these core tests.

</details>
