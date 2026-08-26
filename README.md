# InspectRT

InspectRT is a reproducible, reduced feature-memory baseline for one MVTec AD
category at a time. `RT` means **Runtime**. InspectRT records scientific outputs
and runtime measurements for its frozen workloads.

## Status

`inspectrt_feature_memory_v1` is the implemented profile. It uses a frozen
ResNet-50 feature extractor, a complete nominal patch bank, exact
nearest-neighbor retrieval, raw anomaly maps, and threshold-free metrics. The
supported installed/user commands are `evaluate`, `benchmark`,
`fixture validate` and `onnx validate`. Fixture and ONNX export are
source-checkout reproducibility workflows; portability commands are scoped
maintainer/evidence tooling. The complete command, schema and experimental
Python API boundary is in
[docs/public-interface.md](docs/public-interface.md). See
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

## Bundled fixture quickstart

InspectRT supports CPython 3.11 and 3.12. Install a reviewed wheel by following
[docs/installation.md](docs/installation.md), including the CPU-first or
CUDA-first PyTorch sequence for Linux.

An installed distribution can validate the bundled canonical synthetic
fixture offline on CPU:

```bash
inspectrt fixture validate --device cpu
```

The fixture identity and validation limits are documented in
[docs/retrieval-fixtures.md](docs/retrieval-fixtures.md).

## ONNX feature artifact

The optional ONNX tools use the `onnx` extra in a locked source checkout:

```bash
uv sync --locked --extra onnx
```

Export is a source-checkout reproducibility workflow. It requires a clean
source tree and the accepted pretrained weight already cached:

```bash
uv run --extra onnx inspectrt onnx export \
  --output-root outputs
```

The generated artifact contains only `manifest.json` and `model.onnx`. Its
model bytes remain ignored. Validation is a supported installed/user workflow
when `inspectrt[onnx]` is installed:

```bash
uv run --extra onnx inspectrt onnx validate \
  --artifact \
  outputs/artifacts/inspectrt_onnx_feature_portability_v1/<artifact-id>
```

Validation covers artifact structure and graph identities. The graph emits the
two feature tensors; InspectRT's PyTorch pipeline handles preprocessing,
retrieval, scoring, maps, and metrics. See the
[ONNX portability guide](docs/onnx-portability.md) for the experimental direct
CPU consumer API and reviewed limits.

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

## Installed MVTec quickstart

```bash
inspectrt evaluate \
  --dataset-root /path/to/mvtec_ad \
  --category bottle \
  --device cpu \
  --output-root /path/to/output
```

The wheel includes the frozen baseline profile. Use a separately obtained
MVTec root; torchvision supplies the official pretrained weight. An explicit
`--config` remains supported. See the [baseline guide](docs/baseline.md) for
the method, complete run bundle and benchmark command.

## Benchmark

`benchmark` is supported for the frozen `bottle` workload with exactly five
warm-ups and 30 measured repetitions. See
[docs/baseline.md](docs/baseline.md) for the command, device rules, eight-file
bundle and reviewed measurements.

## Method scope

`inspectrt_feature_memory_v1` is InspectRT's reduced feature-memory reference.
Interpret its results under the method in the [baseline guide](docs/baseline.md).
Supported installation paths are listed in the
[installation and support guide](docs/installation.md), while reviewed
scientific portability evidence is documented in
[docs/portability.md](docs/portability.md).

## License and distribution

InspectRT-authored code is licensed under the
[Apache License 2.0](LICENSE), and [NOTICE](NOTICE) preserves first-party
attribution. Dependencies, datasets, and pretrained weights retain separate
terms. MVTec AD is obtained separately by the user.

Release archives contain InspectRT code, the baseline profile, the synthetic
fixture, and legal metadata. Users obtain MVTec and torchvision weights
separately and generate run bundles or ONNX models locally.
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) records the dependency,
dataset, weight, and generated-artifact boundaries.
