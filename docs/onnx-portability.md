# ONNX feature portability

## Scope

The ONNX record covers the feature-extraction boundary of
`inspectrt_feature_memory_v1` for the frozen `bottle` workload. It compares the
existing PyTorch path with ONNX Runtime 1.28.0 on CPU in three reviewed
environments. The generated graph stops at `layer2` and row-major patch
embeddings; retrieval, scoring, anomaly-map construction, and metrics remain in
the existing PyTorch pipeline.

The result establishes numerical portability under policy-v2 floating-point
bounds.

## Graph boundary

The graph uses ResNet-50 `IMAGENET1K_V2` through `layer2`. Image decoding,
resizing, and normalization happen before the graph. Its static input is FP32
NCHW `images` with shape `[1, 3, 256, 256]`. The ordered outputs are:

1. FP32 NCHW `layer2`, shape `[1, 512, 32, 32]`;
2. FP32 NLC `patch_embeddings`, shape `[1, 1024, 512]`.

The patch output applies a padded `3 x 3` average pool with stride 1,
`count_include_pad=true`, and ceiling mode disabled. A `[0, 2, 3, 1]` transpose moves
NCHW to NHWC before the spatial dimensions are flattened in row-major `(y, x)`
order. Batch and spatial dimensions are static. The ONNX opset is 20. The model
keeps all tensor data inside the ONNX model.

## Installation

Wheel installation and platform support are covered by
[Installation and support](installation.md). The commands below describe the
locked source-checkout environment used for artifact reproduction.

Set up the base environment with:

```bash
uv sync --locked
```

Install the optional ONNX dependencies with:

```bash
uv sync --locked --extra onnx
```

The official pretrained weight stays in the local cache. Export reads the exact
accepted bytes from that cache, requires a clean source tree, and keeps those
accepted weight bytes unchanged. The generated model remains ignored, and
repository distribution excludes both files.

## Artifact export and validation

Export is a supported source-checkout reproducibility command and requires
`inspectrt[onnx]`. Run it from the verified repository:

```bash
uv run --extra onnx inspectrt onnx export \
  --output-root outputs
```

Export creates one artifact directory under
`outputs/artifacts/inspectrt_onnx_feature_portability_v1/<artifact-id>` with
exactly two files:

```text
manifest.json
model.onnx
```

Validation is a supported installed/user command when `inspectrt[onnx]` is
installed. Validate those artifact bytes and their ONNX structure with:

```bash
uv run --extra onnx inspectrt onnx validate \
  --artifact \
  outputs/artifacts/inspectrt_onnx_feature_portability_v1/<artifact-id>
```

Validation stops after checking the artifact inventory, identities, digest,
model structure, and graph contract. ONNX Runtime execution and scientific
comparison are separate steps.

## Direct ORT CPU consumer usage

This direct consumer is an experimental Python API. It requests only
`CPUExecutionProvider` and requires it to be the only active provider.
Provider fallback is disabled. Supply your own image and artifact paths; see
the [Python API policy](public-interface.md#python-api):

```python
from pathlib import Path

from inspectrt.onnx_runtime import OnnxRuntimeCpuFeatureConsumer
from inspectrt.preprocessing import preprocess_image


image_path = Path("path/to/image.png")
artifact_path = Path(
    "outputs/artifacts/inspectrt_onnx_feature_portability_v1/<artifact-id>"
)

prepared = preprocess_image(image_path)
consumer = OnnxRuntimeCpuFeatureConsumer.from_artifact(artifact_path)
outputs = consumer.extract(prepared.image.unsqueeze(0))

layer2 = outputs.layer2
patch_embeddings = outputs.patch_embeddings
```

The caller preprocesses each image before the graph. The consumer returns the
two feature tensors, which the PyTorch pipeline uses for retrieval, scoring,
map construction, and metric evaluation.

## Scientific method

All comparisons used the same accepted source commit, lockfile, profile,
ResNet-50 weight bytes, and artifact bytes. They also used the same ordered
292-image `bottle` inventory. The complete nominal bank contains 214,016 FP32
patches. Each environment compared the PyTorch reference with the ORT CPU
candidate at the raw feature boundary and after the unchanged exact top-1
squared-L2 retrieval and evaluation path.

The recorded measurements cover maximum absolute differences for `layer2`,
patch embeddings, the memory bank, patch distances, image scores, and anomaly
maps. They also cover absolute metric deltas and nearest-index mismatch counts.
Each comparison required exact agreement for the category, ordered samples and
observations, original-image metadata, labels, evaluation masks, tensor shapes,
dtypes, CPU devices, and contiguity. Each environment's second local ORT
execution was exact.

Policy v2 uses `rtol=0` for every floating component. Its absolute limits are
`0.0004` for `layer2`, `0.0002` for patch embeddings, `0.00011` for the memory
bank, `0.003` for patch distances, `0.0015` for image scores, and `0.003` for
anomaly maps. The absolute metric limits are `0` for image AUROC, `0` for image
Average Precision, and `3e-9` for pixel AUROC. Nearest-index mismatches may be
at most `1/25` globally and `1/20` within each test sample. The policy applies
those rational limits with exact integer cross-multiplication.

Policy v2 permits nearest-index differences within the stated bounds. The
differing indices reflect numerical variation at the inputs to the same
retrieval contract. Retrieval remains exhaustive top-1 squared L2 with `k=1`.
An exact computed tie retains the lower global bank index. Tie handling uses
exact equality rather than an epsilon.

## Policy lineage

Policy v2, `inspectrt-onnx-bottle-p53-m1-cpu-v2`, used the Intel Core i7-9850H
and M1 Pro CPUs as its two calibration environments. Ryzen 7 9700X under WSL2
remained untouched until the independent CPU holdout. It passed all 14 policy
checks.
The policy retained its frozen bytes throughout that holdout, which remained
separate from calibration. The recorded history ends with policy v2 and the
third Ryzen/WSL2 holdout.

## Reviewed results

All values below are maximum absolute differences between the PyTorch reference
and the ORT CPU candidate.

| Environment | Policy-v2 role | `layer2` | Patch embeddings | Memory bank | Patch distances | Image scores | Anomaly maps |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Intel Core i7-9850H Linux CPU | calibration | `0.00031107664108276367` | `0.00011277198791503906` | `0.00010371208190917969` | `0.0018463134765625` | `0.0010986328125` | `0.00164794921875` |
| Apple M1 Pro macOS CPU | calibration | `0.0003688335418701172` | `0.00010776519775390625` | `0.00010776519775390625` | `0.00225830078125` | `0.000946044921875` | `0.0020294189453125` |
| Ryzen 7 9700X WSL2 CPU | holdout | `0.00033849477767944336` | `0.00010854005813598633` | `0.00010466575622558594` | `0.00164794921875` | `0.00146484375` | `0.001556396484375` |

| Environment | Image AUROC delta | Image AP delta | Pixel AUROC delta | Index differences | Worst test sample | Local ORT repeat | Policy v2 |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Intel Core i7-9850H Linux CPU | `0` | `0` | `5.415049519896797e-10` | `2504 / 84992` | `47 / 1024` | exact | `PASS` |
| Apple M1 Pro macOS CPU | `0` | `0` | `1.3017835698292402e-9` | `2956 / 84992` | `49 / 1024` | exact | `PASS` |
| Ryzen 7 9700X WSL2 CPU | `0` | `0` | `1.644684388679707e-10` | `2466 / 84992` | `41 / 1024` | exact | `PASS` |

Across all three policy-v2 records, exact structures, labels, and masks agree,
while floating values and nearest indices differ at the bit level.

## Evidence identities

The compact public record is
[scientific.json](evidence/inspectrt_onnx_feature_portability_v1/scientific.json),
8,705 bytes with SHA-256
`b07bbd05d7d6535e0d1088ce23b54e83b0a7754e0ffd921ced778e81d7c5430f`.
The compact record carries the reviewed results and source identities.

| Item | Identity |
| --- | --- |
| Source implementation commit | `d99225474e4760becb1c46ce811a71c016c292e0` |
| Root `uv.lock` SHA-256 | `d92724be7ede2442141cf898a67d12752e44d3bd5df6077dbd5ae97df325df42` |
| Artifact ID | `resnet50-layer2-opset20-143b305b37a9` |
| Artifact digest | `3dcb94af00219dd504a42f013f32b88acc866541e307cc4593d1ceaa6ec0c154` |
| Manifest | 2,035 bytes; `53ebba985e1c82209da89124403270a527122f10207166ae4c5f8d879641b0ae` |
| Model | 5,857,483 bytes; `143b305b37a92e3f2c7dc4268c25baccdf3cfb01c5304f29068f422ff9d8146a` |
| Pretrained weight | 102,540,417 bytes; `11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca` |
| Profile configuration | 303 bytes; `8df093df5eb8e35f77e0e8c088746b34fe69023f115f89fb822a5682d66cdfb6` |
| Inventory | 60,349 bytes; `022df1a49e0f1ab33d57696db2ed667a9603b493d838f4e2f2a850fd95a581c3` |
| Ordered complete sample IDs | `84553df5e4730e2ab6aac4b3298ce0d5b54f3f94b24748756479e2eab41ebcaa` |
| Ordered test sample IDs | `13823f20ef1eccdf8cf0b2baead55fe5587e0e99546082fa6c5c8b764e2a955c` |
| Policy v2 | `inspectrt-onnx-bottle-p53-m1-cpu-v2`; `1d367dc55747b23d5941231a5f5d7c7434f32b0f71a55d0b09f6144321dbf6f3` |
| Intel Core i7-9850H evidence | `cc5e7c5715da514d2b538d447b176ce339b89c4aad351a026a84f842ea3ea560` |
| M1 evidence | `7f2fc282149bb4e9128a6edb92e7118218585bb62ba23ebbe1bbfd581c767b11` |
| Ryzen/WSL2 evidence | `f1d784ca314ca7731ec589073df66025d14e6098cd13aa22bec5d66ba480077e` |

Git tracks the compact scientific record. Export generates `model.onnx`
locally from the accepted pretrained weight in the user's cache.

## Evidence scope

The evidence covers one `bottle` profile, one ResNet-50 weight identity, one
static batch-1 FP32 `256 x 256` graph, ONNX Runtime 1.28.0 on
`CPUExecutionProvider`, the three named environments, the complete frozen
292-image inventory, and one local repeatability scope per environment.
