# InspectRT

InspectRT currently provides a reproducible reduced feature-memory baseline for
one MVTec AD category at a time. `RT` means **Runtime**; the project measures
runtime behavior and has no hard-real-time guarantees.

## Status

The implemented profile, `inspectrt_feature_memory_v1`, uses a frozen ResNet-50
feature extractor, a complete nominal patch bank, exact nearest-neighbor
retrieval, raw anomaly maps, and threshold-free metrics. The CLI supports
complete `evaluate` and `benchmark` runs. It also exports and validates
binary exact-retrieval fixture files; see
[docs/retrieval-fixtures.md](docs/retrieval-fixtures.md) for the format,
commands, and validation limits.

The baseline freeze covers `bottle` and `leather` on a ThinkPad P53 running the
current locked Linux stack with a Quadro T1000. Other systems will get benchmarked in the future. See the measured results and method contract in [docs/baseline.md](docs/baseline.md).

## Installation

InspectRT requires Python 3.11 or later and uses [uv](https://docs.astral.sh/uv/)
for its locked environment:

```bash
uv sync --locked
```

The repository checks run with:

```bash
uv run pytest
uv run ruff check .
```

## MVTec AD

Obtain MVTec AD from the [official dataset page](https://www.mvtec.com/research-teaching/datasets/mvtec-ad) (you will need to enter your email, name and occupation, also commercial use is not allowed)
and place categories manually below one dataset root:

```text
datasets/mvtec_ad/
├── bottle/
└── leather/
```

The dataset remains untracked through gitignore. Pass `datasets/mvtec_ad` (not an individual
category directory) as `--dataset-root`.

## Evaluation

```bash
uv run inspectrt evaluate \
  --config configs/baseline.toml \
  --dataset-root datasets/mvtec_ad \
  --category bottle \
  --device cuda:0 \
  --output-root outputs
```

## Benchmark

```bash
uv run inspectrt benchmark \
  --config configs/baseline.toml \
  --dataset-root datasets/mvtec_ad \
  --category bottle \
  --device cuda:0 \
  --output-root outputs \
  --warmup-count 5 \
  --repeat-count 30
```

Both commands create `outputs/runs/<generated-run-id>/`. Evaluation runs contain
seven files with the resolved run metadata, ordered inventory, memory bank,
predictions, retrieval results, anomaly maps and masks, and metrics. Benchmark
runs add `benchmark.json`. Generated runs are gitignored.
