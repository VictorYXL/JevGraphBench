# Reproduction and model settings

[Results overview](../README.md) · [Complete tables](results.md) ·
[Aggregate result scope and provenance](../data/results/README.md) ·
[Workflow configurations](../configs/README.md)

The README is a results-first overview of **GraphDecisionBench**. This page
documents what can be recomputed or run from this checkout, without confusing
the primary paper panels with historical experiments.

## What is distributed?

- Task definitions, model adapters, evaluators and offline tests.
- [Summary data](../data/results/summary.json) transcribed from
  the primary paper tables for all fourteen model-interface configurations
  across RQ1, RQ2 and RQ3, with unsupported interfaces explicitly marked.
  These rounded table values are not raw experiment records.
- Static charts and complete tables rendered from that aggregate summary.
- [Dataset provenance](../data/README.md), including the
  [optimization catalog](../data/optimization-catalog.json).

Per-query and per-trajectory outputs are not distributed. The public result
bundle is aggregate-only; it does not support raw response replay or independent
recomputation of the primary statistics from individual outcomes.

Historical and supplementary assets are not distributed in the current compact
bundle. They describe independent cohorts, not additional rows collapsed into
the primary aggregates; their sampling and reference distinctions remain relevant.

The ignored `output/` directory, model weights, raw provider ledgers, private
deployment environments and manuscript sources are **not** included in a fresh
clone. Historical originals are preserved in an ignored archive, outside the
distributed bundle.
No public DOI, arXiv identifier or manuscript-download URL is claimed.

## Offline aggregate rendering

From the repository root, using Python 3.11 or newer:

```bash
python run_benchmark.py --config configs/results.yaml
```

No inference or download is performed. The renderer checks the aggregate array
shapes, score ranges and completion counts, then compares the generated charts
and tables with the committed outputs. The default action is `check`; select
`--action render` to regenerate them. Adding CLI `--check` instead only validates
the YAML and previews dispatch, without invoking the renderer or reading artifacts.
This verifies presentation consistency, not the underlying model evaluations.

RQ1 gives each graph size equal weight within each task. RQ2 paired pointwise
95% bootstrap intervals cannot be recovered from aggregate correct counts;
macro-F1 requires class labels and predictions. RQ3 means use completed
trajectories, and paired-mode comparisons require jointly completed graphs,
not subtraction of means with unequal coverage. See the
[aggregate schema and metric notes](../data/results/README.md).

## Configuration and action selection

All public task/results commands use [run_benchmark.py](../run_benchmark.py)
and a YAML config. The [configuration guide](../configs/README.md) covers every
workflow, its valid action parameters and artifact prerequisites. Backend
module CLIs are internal compatibility APIs; dataset acquisition and unittest
remain separate utilities.

All configs use `schema_version: 1`. Workflow configs declare `workflow`,
a YAML-relative `root` (forbidden for `results`),
`default_action`, and an `actions` mapping with one parameter dictionary per
action. Supported workflows are `development`, `structural`, `graph_text`,
`optimization`, `proposals` and `results`. `--action` selects exactly one
configured action; no preparation, validation, inference or reporting steps
are chained automatically. The original three topology configs omit `workflow`, remain
backward compatible and support only `run`.

```bash
python run_benchmark.py --config configs/structural.yaml --action run --check
python run_benchmark.py --config configs/optimization.yaml --action freeze --check
```

CLI `--check` validates **all sections** and previews the selected dispatch,
without runner execution, artifact I/O, downloads or inference. It neither
guarantees required files exist nor establishes primary reproducibility.
Without CLI `--check`, actions such as `validate`, `verify` and the results
`check` actually invoke their backends and inspect artifacts.

`--output ROOT` overrides only the experiment root, relative to the working
directory, not any action path parameter such as `approval`, `tests_log`,
`model_config`, `cohort` or `output`. Configured relative paths remain
YAML-relative. Results forbids `--output`; `--no-progress` is supported only
for schema-1 topology evaluation. Every `run` needs `--allow-inference`.
Download-capable graph-text `freeze` and optimization `prepare` /
`source-evidence` need `--allow-downloads` even when cached. These flags never
bypass existing authorization, deployment, integrity or budget gates.

## Model configuration

The benchmark compares multiple configurations across all three research
questions, not only GPT models. Reasoning/no-reasoning settings are part of
the experimental configuration, alongside interface and capacity differences.

| Configuration | Interface | Thinking / output |
| --- | --- | --- |
| Jev-1.13.0 | Native selection | Provider-native choice |
| Decider-2B v11, Kev-4B, Laya | Native plugins | Native capacity constraints retained |
| Qwen3.5-0.8B / 2B / 4B / 9B / 27B, Qwen3.8-27B, Qwen2.5-72B | Candidate-constrained generation | No thinking; 64 configured output tokens for RQ1/RQ3, 16 for RQ2 |
| Qwen3.5-4B scoring | Candidate-token scoring | Separate readout, not generation |
| GPT-5.4 | Copilot function calling | Explicit `none`, 4,096 configured output tokens |
| GPT-6-Astra | Copilot historical reference | Effort unset; medium observed in response metadata |

GPT-5.4 uses `think=False`, `reasoning_effort="none"` and
`output_format="function_call"`, with a 600-second per-request timeout.
Temperature, seed, `top_p`, `top_k`, presence penalty and `constrain_choices`
are unset. Fresh isolated sessions expose only the `submit_decision` tool.
Accepted responses must report explicit `none` and integer zero reasoning
tokens. The declared option enum is **not** verified strict server-side
decoding; do not confuse it with local Qwen's candidate constraint.

The string `"none"` is not Python/YAML null. `low` still reasons. Astra's
tested `none` request was rejected; no matched no-thinking Astra or substitute
low-effort experiment was run. The 4,096 output setting is not a verified cap
on total reasoning tokens. Cloud/local hardware paths are not speed-matched.
Larger Qwen context extensions are described in manuscript Appendix B.1.

## Evaluation semantics

Invalid outputs count as wrong in RQ1/RQ2. Incomplete or unsupported RQ3
trajectories remain in coverage without an imputed objective. Forced actions,
including singleton proposals, require no model call; forced-only completion
does not establish interface support. Offline aggregate rendering adds no
model calls and does not replay individual decisions.

## Install and run a new evaluation

```bash
python -m pip install -e .
# Optional Copilot support:
python -m pip install -e '.[copilot]'
```

[configs/pilot.yaml](../configs/pilot.yaml) is an adjacency/distance smoke
evaluation, **not** the primary six-task 800-query cohort:

```bash
python run_benchmark.py --config configs/pilot.yaml --check
python run_benchmark.py --config configs/pilot.yaml --allow-inference
```

**The second command makes live model calls and may incur charges.** The
legacy loader is offline: acquire and verify local source archives separately
with the [dataset utility](../data/README.md#download-and-verify) before running.
Jev reads `TYPESAFE_API_KEY`. Copilot reads `COPILOT_GITHUB_TOKEN` or uses an
existing CLI login. Never put tokens in configuration files. Existing-login
mode uses `COPILOT_HOME` (default `~/.copilot`) for authentication while keeping
decision workspaces isolated; explicit-token mode uses a temporary runtime
home. No automatic device login is started.

To smoke-test the no-thinking Copilot adapter, replace the `model` section in
a copy of the configuration and choose a new output directory:

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

This tests the adapter, not the exact paper cohort. Verify the served model and
provider availability before inference. In workflow configs, `jev`,
`jev_action` and `qwen2` are workflow-specific frozen/backend aliases, not
provider model IDs. Do not change frozen interfaces, reasoning or capacity
settings to make a lane pass. New-run configurations do not redefine the
primary scientific settings above.

### Offline task development

```bash
python run_benchmark.py --config configs/development.yaml --action plan
python run_benchmark.py --config configs/development.yaml --action report
```

Planning and subsequent reporting are offline and executable in a fresh
installed checkout. Despite the design name `structural_dev`, this plans
**all ten synthetic tasks**, six structural plus four optimization, not just
the six primary structural tasks. It is not the primary real-network cohort.
Select `run` with `--allow-inference` only when authorized; `analyze` replays
sealed trajectories and needs existing artifacts. See
[data documentation](../data/README.md) for download and normalization rules.

## Figures and tests

The README uses static SVGs, native Markdown and links: no JavaScript, remote
chart service or GitHub Pages deployment is required.

```bash
python run_benchmark.py --config configs/results.yaml --action render
python run_benchmark.py --config configs/results.yaml
python -m unittest tests.publication.test_public_repository tests.publication.test_readme_assets -q
```

The SVGs and [complete tables](results.md) are generated from the same
[summary data](../data/results/summary.json). The four figures are
[overview](../assets/benchmark/overview.svg),
[structural queries](../assets/benchmark/structural-queries.svg),
[graph-text decisions](../assets/benchmark/graph-text-decisions.svg) and
[optimization trajectories](../assets/benchmark/optimization-trajectories.svg).
The results `check` action verifies generated content without rewriting it;
CLI `--check` only previews that action.

Full offline tests: `python -m unittest discover -s tests -t . -v`.
Tests are grouped into unit, integration and publication packages; provider
tests use mocks. Optional historical fixtures are disabled by default and
require explicit external configuration. See [tests/README.md](../tests/README.md).

### Optional local scoring adapter

The preserved SemIf scoring adapter requires an explicitly configured local
checkpoint directory via `GRAPHDECISIONBENCH_SCORING_MODEL`. It never assumes a
developer-specific home or model path. `GRAPHDECISIONBENCH_SCORING_BRIDGE` and
`GRAPHDECISIONBENCH_SCORING_REFERENCE` can select locally supplied bridge source and
reference artifacts; the defaults live under `output/runtime/scoring/`.
The bridge source hash is still checked before loading. These optional artifacts
are not distributed, and configuring them does not itself authorize inference.

## Supplementary preparers and primary aggregates

The [structural](../configs/structural.yaml),
[arXiv](../configs/graph_text_arxiv.yaml),
[Prime](../configs/graph_text_prime.yaml),
[optimization](../configs/optimization.yaml) and
[proposals](../configs/proposals.yaml) configs expose preparation, freeze,
validation, execution and reporting through the same root CLI. They are not
one-command reproduction of the primary experiments. Safe previews include:

```bash
python run_benchmark.py --config configs/structural.yaml --action freeze --check
python run_benchmark.py --config configs/graph_text_arxiv.yaml --action freeze --check
python run_benchmark.py --config configs/graph_text_prime.yaml --action run --check
python run_benchmark.py --config configs/optimization.yaml --action prepare --check
python run_benchmark.py --config configs/proposals.yaml --action freeze --check
```

The `output/config-inputs/` paths in these examples are placeholders, not
distributed artifacts. Supply authentic inputs, approvals and logs; do not
fabricate them or use config validation as an artifact-readiness check.

- Structural freeze is local-only but needs existing verified data,
  authorizations, pinned adapters and panel artifacts. Its six real-network
  tasks include queried-pair connectivity, not whole-graph connectivity.
- Graph-text freezing uses explicit portable input locations but still needs
  externally supplied historical dependencies; see the
  [input contract](#portable-graph-text-inputs). Both configs share one root
  and freeze both datasets together: do not freeze twice.
  Execution approvals are model/dataset-specific; replace the placeholder
  with the correct authentic approval for each lane. Old arXiv lanes are
  reuse-only, including failures; `qwen2` is supported for new runs, not a
  promise that all fourteen lanes can freshly run arXiv.
- Optimization preparation/source-evidence can download and be CPU-heavy.
  Freeze requires real successful unittest evidence, externally supplied
  cohort artifacts and applicable authorizations. Its pilot run/report scope
  is not primary evaluation.
- Proposal freeze selects portable user-supplied instances and a backend JSON
  model map, never the archived-input fallback. That model map is not the
  unified schema-1 model section. A genuine successful unittest log must
  already be stored as `tests.log` under the selected root before freezing.

After satisfying prerequisites, remove `--check` to execute the selected
action, adding `--allow-downloads` for download-capable actions or
`--allow-inference` for `run`. Keep the same root across dependent steps and
select each step explicitly. To render the distributed primary aggregates
instead, use `python run_benchmark.py --config configs/results.yaml --action render`.

The neutral protocols identify newly frozen cohorts. Protocol identifiers can
participate in deterministic sampling, so a newly frozen cohort may not
reproduce the original sampling even with otherwise identical settings.

- The supplementary graph-text freeze uses a label-blind Prime pool. The
  primary panels instead use gold-containing candidates.
- Original GPT configurations omit reasoning controls. The later no-thinking
  execution layer supplies the GPT-5.4 configuration above.
- The supplementary optimization freeze correctly labels heuristic modularity references.
  Eight of ten later certified optima improve them. The primary panels require those certified
  values, not relabeling heuristic bounds as optima.
- Earlier public-optimization, 200-target graph-text, action-abstraction and
  four-task-proposal experiments have different cohorts and sometimes reference
  values. Their historical assets are not distributed in this compact bundle.
  They remain independent experiments and are not pooled with the primary panels.

Never rewrite frozen artifacts, manifests, protocol identifiers or recorded
hashes to bypass integrity checks. Preserve original outputs externally rather
than modifying them to satisfy a newly frozen protocol. Recomputing a table,
replaying raw decisions, running new inference and compiling a manuscript are
four distinct operations.

### Historical reference-input defaults

Compatibility backends use semantic locations under `output/reference-inputs/`
for externally supplied historical inputs. These are **placeholders**, not
distributed data or aliases that discover ignored experiment directories:

| Input | Default location under `output/reference-inputs/` |
| --- | --- |
| Proposal synthetic instances | `proposals/synthetic/instances.jsonl` |
| Original main-panel scores | `proposals/main-ledger/scores.jsonl` |
| Original portable public records | `proposals/portable/public/structural_challenge.jsonl` |
| Original private scoring references | `proposals/portable/references/structural_challenge.jsonl` |
| Public TSP/MaxCut conditions | `proposals/public/instances.json` |
| Original model-panel description | `proposals/panel/benchmark.json` |
| Structural/optimization sealed model configurations | `model-panel/` |
| Paired-development prior adjacency bank | `paired-graph/adjacency/graphs.jsonl` |

Supply authentic records without changing their values, IDs or seals; no data
is moved or manufactured by these defaults. The model-panel root needs the
original run configurations and applicable protocol/configuration seals.
Explicit workflow inputs such as `cohort` and `historical_root` still override
their backend defaults; changing the output root alone does not relocate inputs.
The paired-development default output is `output/experiments/development/paired-graph/`.

New proposal freezes use the semantic cohort identifier
`graph-proposals-raw-capture` (with `-user-inputs` for explicit inputs). This is
an experiment identifier, not a changed wire protocol or schema version.
The raw-response schema remains unchanged. Existing sealed runs must use their
original frozen source; do not relabel them or change hashes to pass validation.
Confirmatory-bank pilots and pre-raw-capture episodes remain quarantined, with
no episode reuse or retrospective raw-response reconstruction. Portable proposal
inputs remain the recommended route for new evaluations, not a claim of primary
paper replication.

### Portable graph-text inputs

The graph-text backend's repository-relative defaults are
`output/config-inputs/graph-text/arxiv/` (the `OLD` arXiv bundle),
`output/config-inputs/graph-text/remote/` (the `REMOTE` reuse root), and
`output/config-inputs/graph-text/native_endpoint_client.py` (the unchanged,
hash-pinned native adapter). Changing the workflow output root does not relocate
these inputs. No CLI or workflow parameters are added for them.

Supply `reuse-index.json` directly under the remote reuse root. This is a
**separate, unsealed location index**, not a replacement scientific manifest or
authorization. Its exact schema is illustrated below; replace the hash
placeholders with the authentic SHA-256 digests of the unchanged source bytes:

```json
{
  "models": {
    "qwen4": {
      "directory": "runs/qwen4",
      "requests_sha256": "<64 lowercase hexadecimal characters>",
      "attempts_sha256": "<64 lowercase hexadecimal characters>"
    },
    "decider": {
      "directory": "runs/decider",
      "requests_sha256": "<64 lowercase hexadecimal characters>",
      "attempts_sha256": "<64 lowercase hexadecimal characters>"
    }
  }
}
```

Each entry has exactly these three fields. Only existing remote reuse aliases
(`qwen4`, `qwen9`, `qwen27`, `qwen72`, `decider`, `kev`, `laya`) are accepted;
duplicate JSON keys and unknown fields are rejected. Include every remote lane
whose genuine results are to be reused; `decider` is required explicitly.
Local `jev`, `gpt54` and `gpt6astra` lanes remain under `OLD/runs/` and cannot be
overridden by the index. Missing inputs/results are not manufactured.

Directories must be nonempty portable paths relative to the remote reuse root:
no absolute paths, traversal, backslashes, drive prefixes or symlinks. The index
and each lane's `requests.jsonl`, `attempts.jsonl` and `status.json` must be regular
files, not symlinks. Request/attempt byte hashes are checked before the existing
prompt, request-hash, manifest and model-identity reuse audit. A matching locator
hash does not bypass those checks or authorize inference.

Users must consciously supply this locator and unchanged source blobs; there is
no discovery or automatic conversion of old machine-specific indexes or paths.
Keep sealed originals, frozen-source blobs, manifests and recorded hashes
unchanged, with the new locator outside those seals. The `OLD` bundle still needs
its original manifests, records, labels, classes, request hashes, frozen sources
and local result lanes. Parent authorization, upstream data, pinned adapter and
deployment approvals remain required by their existing gates. These assets are
not distributed. Portable locations do **not** make the primary panels
reproducible from a fresh checkout: this preparer retains the supplementary
label-blind Prime protocol, not the primary gold-containing candidate protocol.
