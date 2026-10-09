# Configuration-driven workflows

Use the repository-root [entrypoint](../run_benchmark.py) for all public task
and result workflows. Backend runner CLIs are internal compatibility APIs, not
separate public entrypoints. Dataset acquisition remains a separate
[dataset utility](../data/README.md#download-and-verify); tests remain unittest
commands. Examples below run from the repository root after installation.

## Coverage: tasks are not datasets

| Configuration | Tasks / data | Default action and prerequisites |
| --- | --- | --- |
| [development.yaml](development.yaml) | All **ten synthetic tasks**: six structural plus TSP, MaxCut, LT and modularity; `structural_dev` does not mean only six tasks | `plan`; executable offline in a fresh installed checkout, then explicitly select `report` |
| [structural.yaml](structural.yaml) | Six real-network tasks: adjacency, degree, cycle, **pair connectivity**, distance threshold and articulation; Facebook, ca-GrQc, Power Grid and Human PPI | `validate`; needs an existing frozen root and data, not just this YAML |
| [graph_text_arxiv.yaml](graph_text_arxiv.yaml) | arXiv subject classification under matched graph/text conditions | `verify`; frozen artifacts and historical dependencies required |
| [graph_text_prime.yaml](graph_text_prime.yaml) | Supplementary **label-blind** STaRK-Prime candidate selection, not primary gold-containing selection | `verify`; shares the arXiv root and dependencies |
| [optimization.yaml](optimization.yaml) | Supplementary TSP, signed MaxCut, deterministic LT and binary modularity on task-specific source instances | `validate`; prepared local artifacts required; heuristic modularity references are not primary certified optima |
| [proposals.yaml](proposals.yaml) | Action/proposal ablations on explicitly supplied, backend-valid instances; task coverage depends on those inputs | `report`; needs a frozen supplied-input experiment |
| [results.yaml](results.yaml) | Distributed primary aggregates for all fourteen model-interface configurations across RQ1/RQ2/RQ3 | `check`; offline presentation consistency, not new task execution |
| [pilot.yaml](pilot.yaml) | Legacy adjacency + distance two-task smoke evaluation on four real networks | Implicit `run`; local verified source archives and inference permission required |
| [adjacency.yaml](adjacency.yaml), [distance_threshold.yaml](distance_threshold.yaml) | Single-task smoke variants of that legacy two-task topology interface | Implicit `run`; neither is the primary six-task cohort |

Development uses synthetic graphs rather than the four real networks. Pair
connectivity asks whether the queried vertices are reachable from one another,
not whether the whole graph is connected. Task coverage alone does not reproduce
the primary sampling, prompts, interfaces, references or results.

## Schema 1: topology and explicit workflows

All configs use `schema_version: 1`. The three topology configs retain their existing
`run`, `data`, `sampling`, `tasks` and `model` sections, defaults and path
semantics. Only the implicit or explicit `run` action is supported. They still
need `--allow-inference`; their loader never downloads implicitly.

Configs with a `workflow` field use this exact envelope instead:

```yaml
schema_version: 1
workflow: development
root: ../output/config-examples/development
default_action: plan
actions:
  plan:
    design: structural_dev
    seed: 20260926
    node_counts: [8]
    samples_per_size: 1
    no_think_max_tokens: 64
    models: [jev]
  report: {}
```

- `workflow` is allowlisted: `development`, `structural`, `graph_text`,
  `optimization`, `proposals` or `results`.
- `root` is required except for `results`, where it is forbidden. Relative
  roots and action path parameters resolve against the **YAML directory**.
- `actions` is a nonempty mapping from supported action names to their own
  parameter mappings, including `{}` for an action with no supplied parameters.
  `default_action` must name one of these configured actions.
- All sections are validated, not only the selected action. Unknown envelope
  fields, actions and parameters are rejected. Omitted optional parameters keep
  backend defaults; an action does not inherit another action's parameters.
- `--action` selects exactly one configured action. There are **no automatic
  chains**: `run` does not plan, freeze, validate or report on your behalf.

The authoritative action parameters, types, choices and required-field checks
are in the [workflow registry](../src/benchmark/workflows.py). This summary
also lists supported actions not included in the minimal examples:

| Workflow | Actions and accepted parameter names (`*` = required) |
| --- | --- |
| `development` | `plan`: `design`, `seed`, `samples_per_size`, `node_counts`, `prior_pool`, `no_think_max_tokens`, `models`; `run`: `models*`; `report`, `analyze`: `models` |
| `structural` | `freeze`: `data_dir`, `historical_root`, `authorization`, `pilot_model_config`, `test_model_config`; `validate`: `data_dir`; `run`: `model*`, `split`, `concurrency`, `base_url`; `report`, `combined-report`: `output`; `check-lane`: `model*`, `require_complete`, `output`; `combine`: `primary_root*`, `deployment_root*`, `approval*`, `outer_plan`, `activation_proof`; `scoring-recovery-check`: `primary_root*`, `approval*`, `output` |
| `graph_text` | `freeze`, `verify`: no parameters; `report`: `output`; `run`: `model*`, `dataset*`, `approval*` |
| `optimization` | `prepare`, `source-evidence`, `validate`: no parameters; `freeze`: `tests_log*`, `cohort`; `run`: `models*`, `model_config`, `concurrency`, `scope`; `report`: `scope` |
| `proposals` | `freeze`: `instances*`, `model_config*`, `models*`; `run`: `model_config*`, `models*`, `concurrency`; `report`: no parameters; `monitor`: `interval` |
| `results` | `check`, `render`: no parameters |

For development, `prior_pool` is required only for `structural_holdout` and
forbidden for other designs; `pilot` does not accept `node_counts`. Structural
pilot/test model configs must be supplied together, as must `outer_plan` and
`activation_proof`. Concurrency is 1–16; optimization scope is `pilot` or
`main`; structural split is `test` or `public-pilot`. A schema-valid model
identifier still needs backend support and authorization for its specific lane.

## Preview, execute and authorize

```bash
# Validate all config sections and preview just the selected dispatch.
python run_benchmark.py --config configs/development.yaml --check
python run_benchmark.py --config configs/structural.yaml --action run --check
python run_benchmark.py --config configs/graph_text_prime.yaml --action freeze --check

# Fresh-checkout offline development: two explicit actions, no inference.
python run_benchmark.py --config configs/development.yaml --action plan
python run_benchmark.py --config configs/development.yaml --action report

# Actual offline aggregate consistency check, then optional regeneration.
python run_benchmark.py --config configs/results.yaml
python run_benchmark.py --config configs/results.yaml --action render

# Preview a root override; this changes no configured input/output path parameter.
python run_benchmark.py --config configs/structural.yaml --output output/my-structural --check
```

**CLI `--check` is configuration validation and dispatch preview only.** It
invokes no runner, reads/writes no experiment artifacts, and performs no
download or inference. It neither guarantees that artifacts exist nor verifies
them, and is not evidence of primary reproduction. For example,
`--config configs/results.yaml --check` only previews, whereas
`--config configs/results.yaml --action check` actually invokes the offline
renderer and compares the committed figures/tables. Likewise `validate` and
`verify` are real backend actions only when CLI `--check` is absent.

`--output ROOT` overrides **only the workflow root**, relative to the current
working directory. It does not relocate `data_dir`, `cohort`, `approval`,
`model_config`, `tests_log`, an action's `output`, or any other path parameter.
It is forbidden for `results`. Use the same root for dependent actions.
`--no-progress` is supported only for legacy schema-1 topology evaluation.

Every `run`, including the three schema-1 configs, requires
`--allow-inference`. Download-capable `graph_text.freeze`,
`optimization.prepare` and `optimization.source-evidence` require
`--allow-downloads` **even with a populated cache**. These permissions are
independent and do not replace existing authorization, deployment, budget,
hash, hold or readiness gates. Planning/freeze is not permission for inference.
Previews do not need either permission flag.

Only after satisfying the relevant prerequisites, invocation takes this form:

```bash
python run_benchmark.py --config configs/pilot.yaml --allow-inference
python run_benchmark.py --config configs/structural.yaml --action run --allow-inference
python run_benchmark.py --config configs/graph_text_arxiv.yaml --action freeze --allow-downloads
python run_benchmark.py --config configs/graph_text_arxiv.yaml --action run --allow-inference
python run_benchmark.py --config configs/graph_text_prime.yaml --action run --allow-inference
python run_benchmark.py --config configs/optimization.yaml --action prepare --allow-downloads
python run_benchmark.py --config configs/optimization.yaml --action source-evidence --allow-downloads
python run_benchmark.py --config configs/optimization.yaml --action run --allow-inference
python run_benchmark.py --config configs/proposals.yaml --action run --allow-inference
```

These are independent action examples, **not** a ready-to-run reproduction
script. Model calls may incur charges; preparation may be CPU-heavy.

## Inputs, frozen aliases and scientific limits

The sample `output/config-inputs/` paths are **placeholders** for externally
supplied panel artifacts, genuine authorization/approval documents, instances,
backend model maps and successful test logs. None are supplied or synthesized
by these configs. Do not fabricate test logs, approvals or manifests. Output
roots under `output/config-examples/` keep new experiments separate from
published aggregates; creating a directory does not satisfy backend gates.

- **Structural:** freeze is local-only. It still requires preexisting verified
  data, authorizations, pinned adapters and panel artifacts. The example does
  not make the primary cohort reproducible from a fresh clone.
- **Graph-text:** both configs intentionally share one root. A freeze prepares
  both datasets; **do not freeze twice**. Explicit local input defaults are
  `output/config-inputs/graph-text/arxiv/` for the unchanged old arXiv bundle,
  `output/config-inputs/graph-text/remote/` for imported reuse blobs and their
  user-supplied `reuse-index.json`, and
  `output/config-inputs/graph-text/native_endpoint_client.py` for the pinned
  adapter. These repository-relative inputs are not supplied through the YAML
  and are not relocated by `--output`. The portable locator schema and remaining
  historical prerequisites are described in the
  [graph-text input guide](../docs/reproduction.md#portable-graph-text-inputs).
  Old arXiv lanes are reuse-only,
  including failures; `qwen2` is a supported new-run lane, not evidence that
  all fourteen configurations can be freshly run on arXiv. Approval is bound
  to a model/dataset lane: the common placeholder must be replaced with the
  appropriate authentic document for each run, not reused across lanes.
  Supplementary label-blind Prime is not the primary gold-containing pool.
- **Optimization:** preparation/source evidence can download and perform
  substantial CPU work. Freeze needs the real successful unittest log named
  by `tests_log`, cohort artifacts and applicable external authorizations.
  The supplementary heuristic modularity references must not be presented as
  the primary certified optima. No config grants external authorization.
- **Proposals:** freeze uses supplied `instances`, `model_config` and `models`,
  never the archived-input fallback. The backend additionally requires a
  genuine successful unittest log named `tests.log` **under the selected root**
  before freezing. `model_config` is the existing backend JSON model map,
  not the topology `model` section or another workflow config; see
  [supplied-input validation](../src/utils/graph_abstraction_suite.py).
  Instance validity, model support, integrity and execution gates still apply.

Workflow names such as `jev`, `jev_action` and `qwen2` are backend/frozen lane
aliases, not interchangeable provider model IDs. Their allowlists differ by
workflow. Topology configs instead contain a provider configuration such as
`provider: typesafe` with its provider `model` ID. Preserve frozen alias,
provider ID, interface, reasoning, token/context budgets, precision and pinned
adapter settings; never override scientific settings or hashes to make a lane
pass. Changing a new config does not retroactively redefine a frozen cohort.
Never put credentials in YAML or model-map JSON.

See [reproduction and model settings](../docs/reproduction.md) and
[aggregate scope](../data/results/README.md) for the distinction between new
experiments, supplementary cohorts and rendering the primary aggregates.