# GraphDecide benchmark core

GraphDecide evaluates graph decisions and sequential constructions with
deterministic scoring. This checkout contains the runnable benchmark, not the
paper-production workspace. The Python distribution remains `jevgraphbench`.

## Repository layout

| Path | Purpose |
| --- | --- |
| [src/benchmark/](src/benchmark/) | Task generation, exact scoring, configuration and experiment runner |
| [src/clients/](src/clients/) | Jev, vLLM and optional GitHub Copilot clients |
| [src/datasets/](src/datasets/) | Real-network download, parsing and verification |
| [configs/](configs/) | YAML configuration templates |
| [data/real/](data/real/) | Dataset metadata; downloaded archives are ignored |
| [scripts/](scripts/) | Three required runtime/audit utilities, described below |
| [tests/](tests/) | Core regression tests only |
| `output/`, `results/` | Ignored local artifacts; never required merely to import the core |

Paper sources, PDFs, the website, historical experiments, publication scripts
and their tests have been moved into ignored local `output/`. They are not
included in a fresh clone. Existing released versions remain in Git history.
No weights, credentials or raw provider ledgers are distributed here.

In the HeurAgenix repository, run the following commands **from
`benchmarks/graphdecide/`**, not the HeurAgenix root: both projects use a
Python package named `src`.

## Install

Python 3.11 or newer:

```bash
python -m pip install -e .
```

For the optional GitHub Copilot provider:

```bash
python -m pip install -e '.[copilot]'
```

## Real-network evaluation

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

## Synthetic ten-task suite

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

The retained scripts are:

- [extended_graph_suite.py](scripts/extended_graph_suite.py): plan, run and report.
- [paired_graph_ablation.py](scripts/paired_graph_ablation.py): shared model
  configuration, preflight, locking and integrity utilities required by the suite.
- [audit_task_shortcuts.py](scripts/audit_task_shortcuts.py): offline label and
  simple-feature audits used by the structural task tests.

Historical twelve-configuration orchestration and publication tools are
archived, not part of this minimal CLI. The core suite does not claim to
reproduce every historical model lane from a single command.

## Scoring and failures

Report exact-answer accuracy separately from legal completion.
For constructions, report mean additive objective gap on feasible completed
episodes **alongside feasible/scheduled coverage**. Maximization uses
`optimum - solution`; minimization uses `solution - optimum`.
Gap units differ across tasks and must not be pooled into one score.
Timeouts, unsupported requests and invalid outputs are not silently repaired
or removed from scheduled denominators.

## Tests

The retained tests use offline fixtures and mocked providers:

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

No credentials, model servers, paper builders, browser tooling or archived
experiment outputs are required for these core tests.
