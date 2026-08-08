# InspectRT

InspectRT is currently a reproducible, reduced feature-memory baseline for one
MVTec AD category at a time. `RT` means **Runtime**: it measures runtime
behavior, but it does not make hard real-time guarantees.

## Status

`inspectrt_feature_memory_v1` is the implemented profile. It uses a frozen
ResNet-50 feature extractor, a complete nominal patch bank, exact
nearest-neighbor retrieval, raw anomaly maps, and threshold-free metrics. The
CLI can run complete `evaluate` and `benchmark` jobs, and it can export and
validate binary exact-retrieval fixtures. See
[docs/retrieval-fixtures.md](docs/retrieval-fixtures.md) for the format,
commands, and validation limits.

The baseline freeze covers `bottle` and `leather` on a ThinkPad P53 running the
current locked Linux stack with a Quadro T1000. It does not cover other systems
yet; they will be benchmarked separately. See the measured results and method
contract in [docs/baseline.md](docs/baseline.md).

## Installation

InspectRT requires Python 3.11 or later and uses [uv](https://docs.astral.sh/uv/)
for its locked environment:

```bash
uv sync --locked
```

Run the repository checks with:

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

Both commands write to `outputs/runs/<generated-run-id>/`. An evaluation run has
seven files containing the resolved run metadata, ordered inventory, memory
bank, predictions, retrieval results, anomaly maps and masks, and metrics. A
benchmark run adds `benchmark.json`. Generated runs are gitignored.
