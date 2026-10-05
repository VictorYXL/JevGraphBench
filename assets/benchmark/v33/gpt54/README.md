# GPT-5.4 final no-thinking results used in GraphDecide v33

| File | Rows | Unit |
| --- | ---: | --- |
| [rq1.csv](rq1.csv) | 800 | Structural query |
| [rq2.csv](rq2.csv) | 1,000 | Dataset/query/condition |
| [rq3.csv](rq3.csv) | 80 | Task/graph/construction mode |
| [retry_summary.csv](retry_summary.csv) | 24 | RQ/task/condition retry accounting |

These files are byte-identical copies of the final exports prepared on
October 4, 2026. They contain no raw prompts, answers from providers, tokens
or credential files. The source exports were independently checked against
the frozen experiment outcomes before publication.

**Final selection is not a single-attempt claim.** Only illegal outputs were
retried. A legal answer was retained even when incorrect; two unresolved
Prime/TG answers remain `output_legal=0, correct=0`. Retry costs remain separate.
The main CSVs do not contain attempt-count columns.

- Boolean fields are `0`/`1`. RQ1 source and size come from instance provenance.
  Compute accuracy per size before equal size averaging within each task.
- Pair RQ2 by `(dataset, query_id)`, not the condition-specific `record_id`.
  `A` denotes Prime's published `A*`. Each query has all five conditions.
  Binary correctness cannot reconstruct macro-F1.
- Pair RQ3 by `(task, graph_id)`. `A` means direct construction; `C` means
  fixed-heuristic next-action selection. `model_calls` excludes deterministic
  forced steps and counts only the selected trajectory.
- TSP/LT/MaxCut gaps are percentages; binary modularity uses absolute Q gaps
  against numerical MILP-certified optima. Signed MaxCut uses historical BKS.
- In the retry summary, `additional_model_calls` counts calls in all retry
  attempts, not total calls minus selected calls. For bier127/A the selected
  trajectory uses 125 calls, two restarts add 211 calls, and all three
  trajectories together use 278 calls.

## Offline reporting

From the repository root:

```bash
python -m src.utils.public_results_v33 --data-dir assets/benchmark/v33/gpt54
```

This validates the final-result schema and recomputes aggregates and paired
intervals; it does not replay private raw responses or make new model calls.
The report matches v33 Appendix C's GPT-5.4 size-equal and aggregate scores,
Appendix D's paired intervals and the final 80-trajectory optimization panel.
Other configurations' row-level records are not part of this bundle.

## Provenance

Local source: `output/papers/v32/GPT5.4-nothink-data/` (ignored, not required).
Local audit: `output/experiments/public/v32-20261003/orchestration/no-think-export-v1/verification.json`.
The historical directory version is a provenance label, not a different
selection. Local manuscript: `output/papers/v33/GraphDecide_v33.pdf`.
SHA-256 of the manuscript checked for this update:
`891ac7eb2ec3c575fdc6d13994e06288277cd7fcc9effbe9652623cf0115c0ed`.

| File | SHA-256 |
| --- | --- |
| rq1.csv | `3e25a5c154e8d13568b875d1bb6d9f5ebc15f07c5e845cb674291cc1d0be4078` |
| rq2.csv | `65effb5ff1020debcadfd814ea7208aaed1c818eb432199b66a67c3cff4711e5` |
| rq3.csv | `9ac0e5d5aad9d369ed26f2720f7ef9db13c27704ecdc62976e55ffa37070885a` |
| retry_summary.csv | `3254c7ccf7fea65e114676d658f0fb1030f29f6ee8f9dd51991b273e752ed84f` |
