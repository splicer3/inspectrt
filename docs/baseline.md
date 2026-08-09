# Feature-memory baseline

## Scope

`inspectrt_feature_memory_v1` is a reduced feature-memory baseline for the
original MVTec AD dataset. It evaluates one category per run and fits only that
category's nominal training images.

## Method

Pillow decodes each image and resizes it directly to `256 x 256` with bilinear
interpolation and antialiasing. The pipeline applies ImageNet normalization
without a center crop. A frozen ResNet-50 using
`ResNet50_Weights.IMAGENET1K_V2` produces `layer2` features with shape
`[B, 512, 32, 32]`. A `3 x 3` average pool has stride 1 and padding 1, with the
padding included in the average, so the shape stays the same. The pipeline then
flattens the spatial positions in row-major `(y, x)` order into FP32 patches
with shape `[B, 1024, 512]`.

Every patch from the ordered `train/good` split enters the nominal memory bank.
For each test patch, the implementation performs exact chunked top-1 squared-L2
retrieval. The bank chunk size is `16384`, and exact ties retain the lower bank
index. The largest patch distance is the image score. Patch distances are
reshaped to `32 x 32` and bilinearly interpolated with `align_corners=False` to
a raw `256 x 256` anomaly map.

The baseline reports image AUROC, image Average Precision, and pixel AUROC.
There is no crop, coreset, approximate search, score reweighting, Gaussian
smoothing, score normalization, decision threshold, or threshold fitting.

## Dataset layout

Download MVTec AD from its
[official dataset page](https://www.mvtec.com/research-teaching/datasets/mvtec-ad)
and keep it outside Git:

```text
datasets/mvtec_ad/
├── bottle/
└── leather/
```

Commands receive `datasets/mvtec_ad` as the dataset root. Each selected category
must retain the original `train`, `test`, and `ground_truth` structure.

## Running an evaluation

Install the locked environment, then evaluate one category:

```bash
uv sync --locked
uv run inspectrt evaluate \
  --config configs/baseline.toml \
  --dataset-root datasets/mvtec_ad \
  --category bottle \
  --device cuda:0 \
  --output-root outputs
```

`--config` selects the committed profile. `--dataset-root` is the parent of the
category directories. Use `--category` to choose a category and `--device` to
choose the PyTorch device. `--output-root` sets the run directory location.
`--run-id` is optional; InspectRT generates a safe ID when it is omitted.

## Running a benchmark

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

The benchmark runs the same evaluation and adds batch-1 stage timing.
`--warmup-count` controls excluded warm-ups, and `--repeat-count` controls the
measured repetitions. Both values must be positive integers.

## Run bundle

Evaluation writes seven files under `outputs/runs/<run-id>/`; benchmark writes
the same files plus `benchmark.json`.

| File | Contents |
|---|---|
| `run.json` | Records the resolved profile, environment, source state, digests, weights, tensor contracts, and benchmark reference. |
| `samples.jsonl` | Stores the exact ordered category inventory used by the run. |
| `memory_bank.pt` | Stores the contiguous CPU FP32 nominal patch bank and its shape metadata. |
| `predictions.jsonl` | Stores each ordered test sample's label, raw image score, and tensor index. |
| `retrieval.pt` | Stores raw patch distances, nearest-bank indices, and ordered test sample IDs. |
| `anomaly_maps.pt` | Stores raw anomaly maps, evaluation masks, and ordered test sample IDs. |
| `metrics.json` | Stores the three threshold-free metrics and sample and pixel counts. |
| `benchmark.json` | Stores benchmark identity, workload, methodology, stage timing, and device-memory measurements. |

Raw maps and masks permit metric recomputation. `benchmark.json` is absent from
evaluation-only runs. Outputs are ignored by Git, and the complete memory bank
makes accepted runs large.

## Accepted results

Both accepted bundles recorded `source.dirty=false` at commit
`bc330b9070c5ca8db9cb7cfbb27617256388536b`. The results below are from a
ThinkPad P53 with a Quadro T1000 and the current locked Linux stack.

| Category | Train good | Test good | Anomalous test | Bank shape | Bank size | Image AUROC | Image AP | Pixel AUROC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `bottle` | 209 | 20 | 63 | `[214016, 512]` | 438,304,768 B (418 MiB) | 1.0000000000 | 1.0000000000 | 0.9880022372 |
| `leather` | 245 | 32 | 92 | `[250880, 512]` | 513,802,240 B (490 MiB) | 1.0000000000 | 1.0000000000 | 0.9948517303 |

The accepted `bottle` benchmark used five warm-ups and 30 measured repetitions.
CUDA events measured the repeated device stages. Synchronized end-to-end wall
time runs from image decode through the raw map. It excludes model load, bank
build, bank transfer, masks, metrics, persistence, and console output.

| Bottle benchmark measurement | Value |
|---|---:|
| Device | Quadro T1000 |
| `Q / M / D / k` | `1024 / 214016 / 512 / 1` |
| Bank chunk size | `16384` |
| Model and weight load | 350.789 ms |
| Full nominal bank build | 6020.114 ms |
| Bank transfer and device setup | 42.703 ms |
| Feature extraction p50 / p95 | 6.046 / 6.102 ms |
| Exact retrieval p50 / p95 | 245.958 / 246.059 ms |
| Synchronized end-to-end p50 / p95 | 273.224 / 274.390 ms |
| Peak allocated device memory | 614,825,984 B (586.344 MiB) |

On this T1000 run, exact retrieval accounts for most of the measured batch-1
latency.

## Reproducibility

The profile fixes the sample order and seed `0`. PyTorch uses deterministic
algorithms with cuDNN benchmarking disabled. FP32 precision is set to `ieee`,
with TF32 disabled and `CUBLAS_WORKSPACE_CONFIG=:4096:8`. The profile pins
`ResNet50_Weights.IMAGENET1K_V2`; the accepted cached weight SHA-256 is
`11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca`.
Each run records its Git commit and dirty state. It also records dependency
versions, platform, lockfile digest, and the ordered sample-inventory digest.

The accepted `uv.lock` SHA-256 is
`ddaddc99b318a1c3a04d5d7cc433cf736d321b56f98a8ae8b532e71e19e6d76b`.
Inventory SHA-256 values are
`022df1a49e0f1ab33d57696db2ed667a9603b493d838f4e2f2a850fd95a581c3`
for `bottle` and
`ea6db1eaf7a544cfb6d618c4ace19a3caf304eb15f42224b07dab4610e211569`
for `leather`.

A second `bottle` evaluation used the same commit, locked stack, and T1000. Its
inventory bytes, ordered test IDs, FP32 bank, and nearest-bank indices were
identical. Maximum absolute differences were `0.0` for image scores, patch
distances, and anomaly maps. All three metric differences were also `0.0`.
Both runs used the same host, so this only establishes same-stack behavior.

## Limitations

The evidence covers the original MVTec AD dataset and two accepted categories.
The only implemented profile uses one ResNet-50 layer and a complete FP32 bank,
which costs 418 MiB for `bottle` and 490 MiB for `leather`. Exact full-bank
search has substantial measured latency, and the baseline fits no decision
threshold.

The method is not comparable to something like PatchCore results.
