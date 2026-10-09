# Complete GraphDecisionBench result tables

Manuscript-transcribed display values; not fresh experiments. Sources: Appendix C (RQ1), Table 2 (RQ2), Appendix E (RQ3). [Source data](../data/results/summary.json) · [Overview](../README.md).

Primary aggregate results cover fourteen model-interface configurations across RQ1/RQ2/RQ3, with unsupported interfaces explicitly marked. Per-query and per-trajectory outputs are not distributed. Charts and tables render from the summary, not raw response replay.

GPT-5.4 uses explicit no reasoning. GPT-6-Astra uses observed medium reasoning and is not a compute-matched baseline. Fixed model order is not an overall ranking.

## RQ1: size-equal task accuracy (%)

| Model | Adjacency | Degree | Cycle | Connectivity | Distance | Articulation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Jev-1.13.0 | 100.0000 | 67.7999 | 87.5000 | 88.7500 | 90.0000 | 61.2500 |
| Decider-2B | 72.5000 | 9.9205 | 78.7500 | 50.0000 | 61.2500 | 50.0000 |
| Kev-4B | 88.7500 | 10.1982 | 85.0000 | 60.0000 | 83.7500 | 62.5000 |
| Laya | 50.0000 | 10.3838 | 50.0000 | 51.2500 | 50.0000 | 50.0000 |
| Qwen3.5-0.8B | 50.0000 | 12.0903 | 52.5000 | 50.0000 | 50.0000 | 50.0000 |
| Qwen3.5-2B | 50.0000 | 14.7494 | 50.0000 | 50.0000 | 50.0000 | 50.0000 |
| Qwen3.5-4B gen. | 78.7500 | 40.3990 | 80.0000 | 50.0000 | 50.0000 | 51.2500 |
| Qwen3.5-9B | 57.5000 | 34.7405 | 81.2500 | 73.7500 | 75.0000 | 50.0000 |
| Qwen3.5-27B | 100.0000 | 63.2020 | 87.5000 | 87.5000 | 80.0000 | 70.0000 |
| Qwen3.8-27B | 96.2500 | 55.6616 | 85.0000 | 86.2500 | 72.5000 | 51.2500 |
| Qwen2.5-72B | 98.7500 | 67.0833 | 78.7500 | 88.7500 | 91.2500 | 72.5000 |
| Qwen3.5-4B score | 90.0000 | 17.5271 | 78.7500 | 50.0000 | 61.2500 | 55.0000 |
| GPT-5.4 (none) | 100.0000 | 91.4154 | 92.5000 | 96.2500 | 97.5000 | 75.0000 |
| GPT-6-Astra * | 100.0000 | 100.0000 | 100.0000 | 100.0000 | 100.0000 | 100.0000 |

Higher is better. Accuracy is averaged equally over sizes 8–12 within each task, not pooled across queries. Degree has 400 queries; each other task has 80.

## RQ2: ogbn-arxiv accuracy (%)

| Model | T | G | TG | BAG | A |
| --- | ---: | ---: | ---: | ---: | ---: |
| Jev-1.13.0 | 55 | 61 | 58 | 57 | 51 |
| Decider-2B | 35 | 45 | 42 | 38 | 24 |
| Kev-4B | 29 | 61 | 27 | 27 | 41 |
| Laya | U | U | U | U | U |
| Qwen3.5-0.8B | 32 | 34 | 26 | 24 | 33 |
| Qwen3.5-2B | 25 | 12 | 19 | 23 | 15 |
| Qwen3.5-4B gen. | 21 | 59 | 30 | 19 | 42 |
| Qwen3.5-9B | 32 | 55 | 41 | 36 | 39 |
| Qwen3.5-27B | 46 | 61 | 58 | 56 | 49 |
| Qwen3.8-27B | 62 | 60 | 71 | 70 | 51 |
| Qwen2.5-72B | 51 | 58 | 59 | 52 | 45 |
| Qwen3.5-4B score | U | U | U | U | U |
| GPT-5.4 (none) | 73 | 61 | 73 | 72 | 48 |
| GPT-6-Astra * | 73 | 60 | 76 | 73 | 55 |

100 queries per condition; higher is better. U = unsupported, not zero. Prime uses gold-containing candidate sets, not end-to-end retrieval.

## RQ2: STaRK-Prime Hit@1 (%)

| Model | T | G | TG | BAG | A* |
| --- | ---: | ---: | ---: | ---: | ---: |
| Jev-1.13.0 | 48 | 47 | 49 | 46 | 42 |
| Decider-2B | 20 | 17 | 18 | 21 | 20 |
| Kev-4B | 39 | 19 | 26 | 35 | 34 |
| Laya | U | U | U | U | U |
| Qwen3.5-0.8B | 9 | 9 | 9 | 7 | 8 |
| Qwen3.5-2B | 16 | 10 | 16 | 12 | 13 |
| Qwen3.5-4B gen. | 36 | 17 | 24 | 28 | 26 |
| Qwen3.5-9B | 43 | 25 | 31 | 38 | 33 |
| Qwen3.5-27B | 50 | 44 | 46 | 43 | 41 |
| Qwen3.8-27B | 53 | 50 | 51 | 51 | 48 |
| Qwen2.5-72B | 39 | 35 | 33 | 35 | 36 |
| Qwen3.5-4B score | U | U | U | U | U |
| GPT-5.4 (none) | 60 | 51 | 53 | 56 | 56 |
| GPT-6-Astra * | 69 | 69 | 67 | 73 | 71 |

100 queries per condition; higher is better. U = unsupported, not zero. Prime uses gold-containing candidate sets, not end-to-end retrieval.

## RQ3: TSP (% gap)

| Model | A: direct | Complete A | C: proposals | Complete C |
| --- | ---: | ---: | ---: | ---: |
| Jev-1.13.0 | 59.6641 | 10/10 | 25.8731 | 10/10 |
| Decider-2B | 583.3950 | 10/10 | 40.4541 | 10/10 |
| Kev-4B | 420.8638 | 8/10 | 37.4815 | 10/10 |
| Laya | — | 0/10 | — | 0/10 |
| Qwen3.5-0.8B | 576.7172 | 10/10 | 33.8624 | 10/10 |
| Qwen3.5-2B | 539.5417 | 10/10 | 34.7646 | 10/10 |
| Qwen3.5-4B gen. | 242.7966 | 10/10 | 38.4328 | 10/10 |
| Qwen3.5-9B | 252.0041 | 10/10 | 24.0653 | 10/10 |
| Qwen3.5-27B | 50.4023 | 10/10 | 27.2948 | 10/10 |
| Qwen3.8-27B | 51.4061 | 10/10 | 28.8632 | 10/10 |
| Qwen2.5-72B | 116.9987 | 10/10 | 29.2820 | 10/10 |
| Qwen3.5-4B score | — | 0/10 | — | 0/10 |
| GPT-5.4 (none) | 29.4859 | 10/10 | 32.2440 | 10/10 |
| GPT-6-Astra * | 2.2117 | 10/10 | 6.2354 | 10/10 |

Lower is better. Means use completed trajectories; partial subsets are not ranked against full coverage. A dash is undefined quality, not zero. A zero-gap forced-only completion does not establish model support.

## RQ3: Signed MaxCut (% gap)

| Model | A: direct | Complete A | C: proposals | Complete C |
| --- | ---: | ---: | ---: | ---: |
| Jev-1.13.0 | 99.9614 | 10/10 | 9.0517 | 10/10 |
| Decider-2B | 99.8284 | 10/10 | 19.0108 | 10/10 |
| Kev-4B | 99.8284 | 10/10 | 21.9226 | 10/10 |
| Laya | — | 0/10 | — | 0/10 |
| Qwen3.5-0.8B | 99.8054 | 10/10 | 70.6073 | 10/10 |
| Qwen3.5-2B | 99.8284 | 10/10 | 40.1330 | 10/10 |
| Qwen3.5-4B gen. | 99.8284 | 10/10 | 12.3026 | 10/10 |
| Qwen3.5-9B | 99.3003 | 10/10 | 19.9961 | 10/10 |
| Qwen3.5-27B | 104.0532 | 10/10 | 10.4834 | 10/10 |
| Qwen3.8-27B | 99.8284 | 10/10 | 23.1526 | 10/10 |
| Qwen2.5-72B | 105.9990 | 10/10 | 11.5578 | 10/10 |
| Qwen3.5-4B score | — | 0/10 | — | 0/10 |
| GPT-5.4 (none) | 99.6673 | 10/10 | 9.4116 | 10/10 |
| GPT-6-Astra * | 6.3073 | 10/10 | 7.7818 | 10/10 |

Lower is better. Means use completed trajectories; partial subsets are not ranked against full coverage. A dash is undefined quality, not zero. A zero-gap forced-only completion does not establish model support.

## RQ3: Deterministic LT (% gap)

| Model | A: direct | Complete A | C: proposals | Complete C |
| --- | ---: | ---: | ---: | ---: |
| Jev-1.13.0 | 52.8359 | 8/10 | 0.5000 | 8/10 |
| Decider-2B | 75.4013 | 10/10 | 9.8681 | 10/10 |
| Kev-4B | 87.2393 | 3/10 | 4.4118 | 4/10 |
| Laya | — | 0/10 | 0.0000 | 1/10 |
| Qwen3.5-0.8B | 84.1737 | 10/10 | 16.3237 | 10/10 |
| Qwen3.5-2B | 77.6601 | 10/10 | 22.4095 | 10/10 |
| Qwen3.5-4B gen. | 79.6852 | 10/10 | 17.4011 | 10/10 |
| Qwen3.5-9B | 47.1873 | 10/10 | 15.2482 | 10/10 |
| Qwen3.5-27B | 46.5092 | 10/10 | 5.6717 | 10/10 |
| Qwen3.8-27B | 43.1821 | 10/10 | 15.9568 | 10/10 |
| Qwen2.5-72B | 39.9037 | 10/10 | 18.0717 | 10/10 |
| Qwen3.5-4B score | — | 0/10 | 0.0000 | 1/10 |
| GPT-5.4 (none) | 33.4312 | 10/10 | 5.4750 | 10/10 |
| GPT-6-Astra * | 0.0000 | 10/10 | 0.4000 | 10/10 |

Lower is better. Means use completed trajectories; partial subsets are not ranked against full coverage. A dash is undefined quality, not zero. A zero-gap forced-only completion does not establish model support.

## RQ3: Binary modularity (absolute Q gap)

| Model | A: direct | Complete A | C: proposals | Complete C |
| --- | ---: | ---: | ---: | ---: |
| Jev-1.13.0 | 0.4783832 | 10/10 | 0.0555132 | 10/10 |
| Decider-2B | 0.4783832 | 10/10 | 0.1465732 | 10/10 |
| Kev-4B | 0.4783832 | 10/10 | 0.0964632 | 10/10 |
| Laya | 0.5105148 | 3/10 | — | 0/10 |
| Qwen3.5-0.8B | 0.4783832 | 10/10 | 0.1936732 | 10/10 |
| Qwen3.5-2B | 0.4783832 | 10/10 | 0.3656432 | 10/10 |
| Qwen3.5-4B gen. | 0.4654832 | 10/10 | 0.0885532 | 10/10 |
| Qwen3.5-9B | 0.4783832 | 10/10 | 0.1158132 | 10/10 |
| Qwen3.5-27B | 0.4783832 | 10/10 | 0.0892932 | 10/10 |
| Qwen3.8-27B | 0.4752832 | 10/10 | 0.1086932 | 10/10 |
| Qwen2.5-72B | 0.3758732 | 10/10 | 0.0849032 | 10/10 |
| Qwen3.5-4B score | 0.4804942 | 4/10 | 0.1562265 | 1/10 |
| GPT-5.4 (none) | 0.4042398 | 10/10 | 0.0422862 | 10/10 |
| GPT-6-Astra * | 0.0002611 | 10/10 | 0.0129828 | 10/10 |

Lower is better. Means use completed trajectories; partial subsets are not ranked against full coverage. A dash is undefined quality, not zero. A zero-gap forced-only completion does not establish model support.

## Reading the results

- RQ1/2 invalid outputs count as wrong. RQ3 incomplete/unsupported trajectories remain in coverage, without an imputed objective.
- RQ2 paired pointwise intervals cannot be recovered from aggregate correct counts. Aggregate accuracy also does not reconstruct macro-F1.
- Forced singleton actions require no model call; completion alone does not establish improvement over the proposal rules.
- RQ3: TSP/LT use exact optima; signed MaxCut uses historical BKS; modularity uses numerical MILP certificates. Non-GPT modularity objectives were rounded to four decimals before rebasing; printed extra gap digits do not imply extra measurement precision.
- Primary TSP, MaxCut and LT relative gaps are $|f_i-R_i|/R_i\times100$ (%) with positive references. No completed primary solution improves on its reference; the absolute form therefore agrees with the recorded directional shortfall. Modularity uses $Q_i^*-Q_i$ in absolute Q units and retains signed residuals.
- Compare paired modes only on jointly completed graphs. Their paired difference cannot generally be reconstructed by subtracting means with unequal coverage.
- [Reproduction and model settings](reproduction.md).
