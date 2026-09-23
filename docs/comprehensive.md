# Expanded real-network evaluation, version 1

## Purpose

The earlier smoke run already used a real power-grid source, but evaluated only eight questions on two 16-node subgraphs. This protocol broadens the **two existing topology-only decision tasks** across domains, graph sizes, and distance thresholds. It is not a complete test of all graph algorithms or real-world domain reasoning.

Run [configs/comprehensive.yaml](../configs/comprehensive.yaml) with [run_benchmark.py](../run_benchmark.py) after setting `TYPESAFE_API_KEY` securely in your terminal:

```bash
python run_benchmark.py --config configs/comprehensive.yaml --output results/comprehensive-live-001
```

Run from the repository root and use a new output directory. This command starts live Jev calls immediately and may incur charges. Credentials must not be placed in the YAML file. No additional execution flag is required.

## Fixed configuration

| Setting | Value |
|---|---|
| Sources | Facebook, ca-GrQc collaboration, Western US power grid, Human PPI |
| Nodes per subgraph | 32, 64, 128 |
| Samples per source and size | 20 |
| Planned graphs | 240 |
| Adjacency questions per graph | 4 (2 yes / 2 no) |
| Distance thresholds | 2, 3, 4, 6 |
| Questions per threshold per graph | 4 (2 yes / 2 no) |
| Repetitions | 1 |
| Maximum planned calls | 4,800 |
| Seed | 20260924 |
| Requested model | jev-1.13.0 |
| Per-call deadline | 30 seconds |
| Concurrency / retries | 1 / 0 |

The requested model is the version returned by the successful smoke evaluation. Acceptance of that pinned identifier on a subsequent live call has not been verified by this audit. The runner records the actual returned version; there is no automatic fallback. If the provider no longer accepts it, select an available version explicitly and treat that as a configuration change.

Distance questions use positive pairs at distance k and negative pairs at distance k+1. The hardest configured boundary is therefore 6 versus 7 hops. Larger graphs add distractor structure, but node count alone is not a controlled causal measure of reasoning difficulty: density, distance, and input length may vary together.

All selected nodes and induced edges are preserved. Inputs remain anonymous integer nodes and edges, with task instructions only. There are no attributes, semantic names, perturbations, solver hints, or answer paths in model inputs.

## Offline data audit

The configuration was audited on the locally pinned raw archives with network and model-client access blocked. This audit invokes data-generation functions only; it does not add a preparation-only mode to the benchmark CLI. **The following numbers are dataset statistics, not model results.**

| Source | Generated graphs | Generated / planned questions | Yes / no | Maximum edges in a sampled graph |
|---|---:|---:|---:|---:|
| Facebook | 60 | 956 / 1,200 | 478 / 478 | 1,098 |
| ca-GrQc | 60 | 1,164 / 1,200 | 582 / 582 | 795 |
| Power grid | 60 | 1,192 / 1,200 | 596 / 596 | 180 |
| Human PPI | 60 | 1,140 / 1,200 | 570 / 570 | 314 |
| **Total** | **240** | **4,452 / 4,800** | **2,226 / 2,226** | — |

| Task / threshold | Generated / planned questions |
|---|---:|
| Adjacency | 960 / 960 |
| Distance k=2 | 948 / 960 |
| Distance k=3 | 940 / 960 |
| Distance k=4 | 880 / 960 |
| Distance k=6 | 724 / 960 |

Overall question coverage is 92.75%. The 348 missing questions correspond to 87 unavailable balanced groups. Sampled graphs are not altered or replaced to force these groups into existence. Missing groups are more frequent in small or short-diameter graphs; for example, Facebook at 32 nodes and k=6 supplies only 16/80 planned questions. Do not interpret a missing cell as success, failure, or evidence that the model can solve it.

The largest serialized `state` in this audit was 11,903 UTF-8 bytes using Python's default JSON formatting. This is **not a token count** or proof that all inputs fit the provider's context limit. No token limit has been verified; context/transport/response failures must remain visible in the results. Do not silently truncate graphs.

## Reading results

Use the run's summary artifact to examine:

- `by_task`: adjacency versus multi-hop decisions;
- `by_dataset`: variation across source networks;
- `by_node_count`: 32/64/128-node performance;
- `by_threshold`: 2/3/4/6-hop decision boundaries;
- `strata`: joint source × size × task × threshold results and generation coverage;
- `overall`: attempt-weighted accuracy, failures, latency, token usage, and Brier score.

Every planned stratum remains in `strata`, including unavailable cells with zero attempts and null accuracy. Marginal aggregates contain observed calls only. The `generation` field counts questions before repetitions; accuracy and latency count actual attempts. Keep generation coverage distinct from execution coverage.

Do not compare only aggregate accuracy: source and threshold weights differ because not all question groups exist. Compare models using the same request artifact hashes. Check actual model versions, failure rates, and completed-call counts before interpreting accuracy.

## Budget and interpretation limits

With the audited inputs and one repetition, a complete run makes 4,452 prediction attempts. There are no retries, but failed or timed-out requests may still be billable. Currency cost is unknown and remains null; a call ceiling is not a monetary budget. Latency measured on eight small smoke requests should not be extrapolated as a guaranteed runtime for this larger run.

This configuration deliberately favors new graphs over repeat predictions. The changed seed does not guarantee independence from pilot/smoke samples or prevent subgraph overlap. There is no train/test split or statistical power guarantee, and each domain has only one mother graph. Do not treat 4,452 questions as 4,452 independent graph samples.

This is a broader diagnostic of structure reading and distance-boundary decisions. It does not yet evaluate disconnected reachability, weighted/directed graphs, cycle detection, optimization, full path construction, representation robustness, or another model baseline. Keep this configuration fixed after observing its model results; use a separately documented protocol for subsequent changes.