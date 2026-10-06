# Reproduction and model settings

[Results overview](../README.md) · [Complete tables](results.md) ·
[Final GPT-5.4 data](../assets/benchmark/v33/gpt54/)

The README is a results-first overview of **GraphDecide v33**. This page
documents what can be recomputed or run from this checkout, without confusing
the primary paper panels with historical experiments.

## What is distributed?

- Task definitions, model adapters, evaluators and offline tests.
- [Display data](../assets/benchmark/v33/display-data.json) transcribed from
  v33 Appendix C, Table 1 and Appendix E for all fourteen configurations.
  These rounded table values are not raw experiment records.
- [GPT-5.4 final-result CSVs](../assets/benchmark/v33/gpt54/): 800 RQ1 queries,
  1,000 RQ2 query–condition pairs and 80 RQ3 trajectories, plus retry statistics.
  Their provenance and SHA-256 checksums are documented in that directory.
- Historical [result assets](../assets/benchmark/), kept separate from v33.

The ignored `output/` directory, model weights, raw provider ledgers, private
deployment environments and manuscript sources are **not** included in a fresh
clone. The local manuscript checked for these results was `GraphDecide_v33.pdf`
(SHA-256 `891ac7eb2ec3c575fdc6d13994e06288277cd7fcc9effbe9652623cf0115c0ed`).
No public DOI, arXiv identifier or manuscript-download URL is claimed.

## Offline result summaries

From the repository root, using Python 3.11 or newer:

```bash
python -m src.utils.public_results_v33 \
  --data-dir assets/benchmark/v33/gpt54 > gpt54-v33-summary.json
```

No inference or download is performed. The reporter checks unique IDs,
source/size quotas, complete RQ2 condition pairs, RQ3 references and gap units.
It computes:

- RQ1 source/task/size results, query-weighted accuracy, task macro and
  task-size macro. The primary task metric gives each size equal weight.
- RQ2 paired differences: TG−BAG, G−A/A*, TG−T and TG−G. For GPT-5.4,
  numeric query IDs are sorted; 1,000 bootstrap resamples use seed 20261001,
  common sampled indices across contrasts and sorted endpoints 24 and 974.
  These are pointwise 95% intervals, not multiplicity-adjusted claims.
- RQ3 feasible-subset mean gaps, paired-mode differences and selected-trajectory
  model calls. A singleton proposal is forced and requires no model call.

Binary correctness alone cannot reconstruct macro-F1; labels and predictions
are needed. This command does not replay private raw responses or reproduce
the other thirteen configurations' inference.

## Model configuration

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

## Final selection and retry costs

Only invalid outputs were retried. A legal answer is retained even if wrong.
The final-result CSVs omit retry columns, but are not single-attempt claims:

| Panel | Initially invalid | Added calls | Finally invalid |
| --- | ---: | ---: | ---: |
| RQ1 | 0 | 0 | 0 |
| RQ2 | 9 | 84 | 2 |
| RQ3 | 1 trajectory | 211 across two restarts | 0 |

See [per-condition statistics](../assets/benchmark/v33/gpt54/retry_summary.csv).
The two unresolved Prime/TG outputs stay illegal and incorrect. RQ3 calls in
the main result CSV refer to the selected trajectory only, not all failed
attempts. Zero-inference startup recovery and offline replay add no model calls.

## Install and run a new evaluation

```bash
python -m pip install -e .
# Optional Copilot support:
python -m pip install -e '.[copilot]'
```

[configs/pilot.yaml](../configs/pilot.yaml) is an adjacency/distance smoke
evaluation, **not** the v33 six-task 800-query cohort:

```bash
python run_benchmark.py --config configs/pilot.yaml
```

**This makes live model calls, may download data and may incur charges.**
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
provider availability before inference. `python -m src.benchmark --config ...`
delegates to the same YAML entrypoint.

### Offline task development

```bash
python -m src.utils.extended_graph_suite plan \
  --root output/smoke-plan --design structural_dev \
  --node-counts 8 --samples-per-size 1 --models jev
python -m src.utils.extended_graph_suite report --root output/smoke-plan
```

Planning/reporting do not call models. The synthetic development plan is not
the v33 primary cohort. Its `run` command performs inference; `analyze` replays
sealed trajectories. See [data documentation](../data/README.md) for download
and normalization rules.

## Figures and tests

The README uses static SVGs, native Markdown and links: no JavaScript, remote
chart service or GitHub Pages deployment is required.

```bash
python assets/benchmark/v33/render.py
python assets/benchmark/v33/render.py --check
python -m unittest tests.test_public_results_v33 tests.test_readme_assets -q
```

The SVGs and [complete tables](results.md) are generated from the same
[display data](../assets/benchmark/v33/display-data.json). Full offline tests:
`python -m unittest discover -s tests -v`. Provider tests use mocks.

## Historical code and evidence

The `*_v32` utility filenames preserve implementation/provenance identity.
Their original freeze/run commands depend on separately supplied local
artifacts and deployment approvals; they are not one-command v33 reproduction.

- The old graph-text freeze uses a label-blind Prime pool. v33 instead uses
  the later gold-containing candidate revision.
- Original GPT configurations omit reasoning controls. The later no-thinking
  execution layer supplies the GPT-5.4 configuration above.
- The old optimization freeze correctly labels heuristic modularity references.
  Eight of ten later certified optima improve them. v33 requires those certified
  values, not relabeling heuristic bounds as optima.
- Earlier [public optimization](../assets/benchmark/public-summary.csv),
  [200-target graph-text](../assets/benchmark/graph-text.csv),
  [action abstraction](../assets/benchmark/abstraction-summary.csv) and
  [four-task proposals](../assets/benchmark/proposal-summary.csv) have different
  cohorts and sometimes reference values. They are not pooled with v33.

Never rewrite frozen manifests to bypass integrity checks. Recomputing a table,
replaying raw decisions, running new inference and compiling a manuscript are
four distinct operations.
