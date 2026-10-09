# Tests

- `unit/`: isolated clients, datasets, tasks, scoring, and trajectory checks.
- `integration/`: entrypoints and runner/pipeline behavior using tiny fixtures and fake clients.
- `publication/`: frozen aggregate preservation, export utilities, figures, links, and public repository contracts.
- `fixtures/`: small committed fixtures, including the trajectory example.

Primary publication checks cover all fourteen configuration entries and their
rendered figures/tables. They validate aggregate consistency and preservation,
not independent recomputation from per-query or per-trajectory model outputs.

From the repository root, discover all groups with
`python -B -m unittest discover -s tests -t . -v`.
To select one group, use `-s tests/unit` (or `tests/integration`, `tests/publication`)
with the same `-t .`. A single module can be run with
`python -B -m unittest tests.unit.test_clients -v`.

Tests are offline: fake clients and synthetic inputs do not perform model inference
or require service credentials. Ordinary temporary files use system temporary
directories; runners requiring repository-descendant paths use ignored `.test-work/`
directories with registered cleanup. Existing experiment outputs are never fixtures
by default.

## Workflow configuration checks

All public task/results workflows use the repository-root YAML entrypoint;
backend runner CLIs are internal compatibility APIs. Unittest remains the test
interface. See the [configuration guide](../configs/README.md) for workflow
coverage, valid parameters and execution prerequisites.

```bash
python run_benchmark.py --config configs/development.yaml --check
python run_benchmark.py --config configs/proposals.yaml --action freeze --check
python run_benchmark.py --config configs/results.yaml --check
```

These commands validate every config section and preview only the selected
dispatch. They invoke no runner, perform no artifact I/O, download or inference,
and do not prove inputs exist or primary experiments are reproducible.
To actually check committed aggregate figures/tables offline, use
`python run_benchmark.py --config configs/results.yaml` (default action `check`),
without CLI `--check`. Regeneration uses `--action render`.

Every `run`, including legacy schema-1 topology configs, requires
`--allow-inference`; download-capable actions require `--allow-downloads`
even if cached. Do not enable either permission for config validation.
Existing approvals and integrity gates remain in effect.

If preparing a proposal freeze, capture a **genuine successful unittest run**
as `tests.log` under the selected experiment root before freezing. Optimization
freeze likewise requires a genuine successful log at its configured `tests_log`
path. A preview is not a test run or substitute for this evidence; never create
dummy success logs. Config examples do not supply logs, approvals or instances.

## Optional external fixtures

Historical checks skip before probing files unless `GRAPHDECISIONBENCH_TEST_FIXTURE_DIR`
is explicitly set to an absolute directory outside this checkout. It must contain:

- `graphtext/`: `records.json`, `classes.json`, `request-hashes.json`, `manifest.json`,
  and the SHA256-pinned `native_endpoint_client.py` for historical prompt hashes,
  configuration checks, and the native bridge test (using mocked HTTP).
- `public-data/`: `task-instances/tsp225.json` and `raw/tsp225.opt.tour`
  for the official-tour rounding regression.

Opt-in checks only read these files; they neither acquire data nor run inference.
Missing explicitly configured fixtures fail rather than silently skipping.