# InspectRT

InspectRT is a reproducible, reduced feature-memory baseline for one MVTec AD
category at a time. `RT` means **Runtime**. InspectRT measures runtime behavior,
but it does not make hard real-time guarantees.

## Status

`inspectrt_feature_memory_v1` is the implemented profile. It uses a frozen
ResNet-50 feature extractor, a complete nominal patch bank, exact
nearest-neighbor retrieval, raw anomaly maps, and threshold-free metrics. The
CLI can run complete `evaluate` and `benchmark` jobs, and it can export and
validate binary exact-retrieval fixtures. See
[docs/retrieval-fixtures.md](docs/retrieval-fixtures.md) for the format,
commands, and validation limits.

The baseline freeze covers `bottle` and `leather` on a ThinkPad P53 with a
Quadro T1000 and the current locked Linux stack. The frozen `bottle` workload
also has reviewed results for Intel Core i7-9850H CPU, RTX 4080 Super CUDA under
WSL 2, M1 Pro CPU, and M1 Pro MPS. Floating outputs and metrics stayed within
the reviewed envelope, although exact nearest-neighbour indices varied across
devices. The synchronized wall-clock record contains six descriptive timing
rows. See
[docs/portability.md](docs/portability.md) and the baseline method contract in
[docs/baseline.md](docs/baseline.md).

The frozen `bottle` workload also has a reviewed ONNX feature boundary for
`layer2` and row-major patch embeddings. ONNX Runtime CPU results pass the
policy-v2 calibration and independent Ryzen/WSL2 holdout. See
[docs/onnx-portability.md](docs/onnx-portability.md) and the compact
[scientific evidence](docs/evidence/inspectrt_onnx_feature_portability_v1/scientific.json).

## Installation

InspectRT requires Python 3.11 or later and uses [uv](https://docs.astral.sh/uv/)
for its locked environment:

```bash
uv sync --locked
```

The optional ONNX artifact tools and CPU consumer use a separate extra:

```bash
uv sync --locked --extra onnx
```

Run the repository checks with:

```bash
uv run pytest
uv run ruff check .
```

## ONNX feature artifact

Export the static feature graph from a clean source tree with the accepted
pretrained weight already cached:

```bash
uv run --extra onnx inspectrt onnx export \
  --output-root outputs
```

The generated artifact contains only `manifest.json` and `model.onnx`. Its model
bytes remain ignored. Validate the artifact and graph identities with:

```bash
uv run --extra onnx inspectrt onnx validate \
  --artifact \
  outputs/artifacts/inspectrt_onnx_feature_portability_v1/<artifact-id>
```

Validation covers artifact structure and graph identities. The scientific
comparison is a separate procedure. The graph performs feature extraction only;
preprocessing, retrieval, scoring, and metrics remain outside it. See the
[ONNX portability guide](docs/onnx-portability.md) for the direct CPU consumer
and reviewed limits.

## MVTec AD

Obtain MVTec AD from the [official dataset page](https://www.mvtec.com/research-teaching/datasets/mvtec-ad).
The download form asks for your email, name, and occupation, and commercial use
is not allowed. Place the categories below one dataset root:

```text
datasets/mvtec_ad/
├── bottle/
└── leather/
```

The dataset remains untracked through `.gitignore`. Pass `datasets/mvtec_ad`,
not an individual category directory, as `--dataset-root`.

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

Both commands write to `outputs/runs/<generated-run-id>/`. An evaluation run
has seven files. They contain the resolved run metadata, ordered inventory,
memory bank, predictions, retrieval results, anomaly maps and masks, and
metrics. A benchmark run adds `benchmark.json`. Generated runs are gitignored.
