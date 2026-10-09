# Public results: aggregate scope and provenance

This compact bundle contains **primary aggregate results for fourteen
model-interface configurations across RQ1, RQ2 and RQ3**, with unsupported
interfaces explicitly marked. [summary.json](summary.json) contains
manuscript-transcribed display values, not raw experimental output. Per-query
and per-trajectory outputs are not distributed and must not be reconstructed
from aggregates. Charts and tables render from this summary, not raw response
replay or new inference.

Historical and supplementary experiments are independent cohorts with
different sampling, candidate pools or references; their assets are not
distributed in this bundle and are not pooled into the primary results.

Reasoning settings describe experimental configurations, not the scope of the
benchmark: GPT-5.4 uses explicit no reasoning; GPT-6-Astra uses observed medium
reasoning and is not a matched no-reasoning baseline. Native selection, Qwen
constrained generation and candidate-token scoring retain their distinct
interfaces and support limits. See [model settings](../../docs/reproduction.md#model-configuration).

## Aggregate summary

In [summary.json](summary.json), `benchmark` and `cohort` identify the benchmark
and primary cohort, while `sources` records the manuscript table provenance.
`models` contains fourteen configurations with `id`, `label` and `family`.
The ordered arrays are interpreted as follows:

| Field | Meaning and units |
| --- | --- |
| `rq1_tasks` | Order of the six structural tasks |
| Each model's `rq1` | Size-equal task accuracy, percent, in `rq1_tasks` order |
| `rq2_conditions` | Condition order: `T`, `G`, `TG`, `BAG`, `A` |
| Each model's `arxiv`, `prime` | Accuracy and candidate Hit@1, respectively, in percent and condition order |
| `rq3_tasks` | Order: TSP, signed MaxCut, deterministic LT, binary modularity |
| Each model's `rq3` | One `[mean_gap_A, mean_gap_C, completed_A, completed_C]` array per task; ten scheduled graphs per mode |

RQ3 gaps use percent for TSP, MaxCut and LT, but absolute modularity Q for
binary modularity. Means include completed trajectories only. `null` is not
zero: it denotes an unavailable score or a mode without completed trajectories.
Completion counts must remain visible, and forced-only completion does not
establish interface support. Printed precision is display precision, not a
claim of additional measurement accuracy. No cross-task overall rank is defined.

## Metrics and comparison limits

### RQ1: structural queries

The primary cohort schedules 800 queries per configuration across Facebook,
ca-GrQc, Power Grid and Human PPI: 200 per source, 400 degree queries overall
and 80 for each other task. Accuracy gives sizes 8–12 equal weight within each
task, rather than pooling queries. Invalid outputs count as wrong.
Connectivity means reachability of the queried vertex pair, not connectivity
of the whole graph. Distances are unweighted shortest-path hops; task labels
are defined on exactly the supplied graph, including isolates.

### RQ2: graph-text decisions

Each dataset has 100 queries under five matched conditions. T supplies text;
G supplies relational input; TG combines them; BAG removes explicit edges from
TG; A/A* is the comparator without additional entity text or edges. In arXiv,
context identities are graph-selected even in T. In Prime, names and types
remain available across conditions, and G versus A* adds context entities as
well as relations. These are not interchangeable notions of graph-free input.

Scores are arXiv accuracy and Prime candidate Hit@1, in percent. Prime uses
forty gold-containing candidates, not end-to-end retrieval. Invalid outputs
count as wrong. Paired contrasts TG−BAG, G−A/A*, TG−T and TG−G are measured in
**percentage points** on the same queries within each dataset. Reported
pointwise 95% bootstrap intervals are not simultaneous intervals and cannot
be recovered from aggregate correct counts. Macro-F1 likewise requires class
labels and predictions, not just aggregate accuracy. These outputs are not
distributed, so rendering the summary does not recompute these statistics.

### RQ3: optimization trajectories

Each configuration schedules ten graphs per task in each of two modes:
A chooses among all legal next actions; C chooses among distinct next actions
proposed by four fixed rules. Forced actions, including singleton proposals,
make no model call. Completion or improvement over A alone does not establish
improvement over the proposal rules themselves.

Let $f_i$ be the attained TSP, MaxCut or LT objective and $R_i>0$ its reference.
Their relative gap is $\Delta_i=\frac{|f_i-R_i|}{R_i}\times100$, reported in percent.
The factor 100 converts the ratio to the numeric value shown under `Gap (%)`.
Let $Q_i$ and $Q_i^*$ denote attained and reference modularity.

| Task | Objective and reference | Gap |
| --- | --- | --- |
| TSP | Minimize tour length; published optimum | Relative gap (%) |
| Signed MaxCut | Maximize signed cut weight; historical 2016 source-reported BKS, not a proven optimum or asserted current global BKS | Relative gap (%) |
| Deterministic LT | Maximize spread with two seeds; exhaustive optimum | Relative gap (%) |
| Binary modularity | Maximize modularity Q; numerical MILP-certified optimum | $Q_i^*-Q_i$ in absolute Q units |

No completed primary TSP, MaxCut or LT solution improves on its reference, so
this absolute-deviation form agrees with the directional shortfall in the recorded
primary evaluations. Existing execution scorers retain directional diagnostics
for future or supplementary runs; an improvement over a historical BKS must be
identified separately rather than interpreted as worse solution quality.
Modularity residuals retain their sign and are not clamped at zero.
The [optimization catalog](../optimization-catalog.json)
documents source conventions and historical reference provenance, not every
primary cohort's membership. Non-GPT modularity objectives were rounded to
four decimals before rebasing; extra printed gap digits are not extra precision.

Means use completed trajectories; incomplete or unsupported outcomes remain
in coverage without an imputed objective. Paired C−A differences require the
same graphs completed in both modes, with matching references and units. They
cannot generally be reconstructed by subtracting means with unequal coverage.
Partial-cohort means must not be ranked against full-cohort means without a
matched comparison.

## Offline validation and rendering

From the repository root:

```bash
python run_benchmark.py --config configs/results.yaml
python -m unittest tests.publication.test_public_repository tests.publication.test_readme_assets -q
```

The renderer uses the fourteen-configuration summary to produce the
[overview](../../assets/benchmark/overview.svg),
[structural-query](../../assets/benchmark/structural-queries.svg),
[graph-text](../../assets/benchmark/graph-text-decisions.svg) and
[optimization](../../assets/benchmark/optimization-trajectories.svg) figures,
plus the [complete tables](../../docs/results.md). The default results action
is `check`; use `--action render` to regenerate those outputs. CLI `--check`
instead validates the config and previews dispatch only: it invokes no renderer,
reads/writes no result artifacts and does not check the committed outputs.
The results workflow has no root and forbids `--output`.

The actual renderer check verifies presentation consistency, not raw-result
replay or independent reproduction of inference. Public result and task
workflows use this unified YAML entrypoint; see the
[configuration guide](../../configs/README.md), [reproduction](../../docs/reproduction.md) and
[source data and citations](../README.md) for further scope and provenance.