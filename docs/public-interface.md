# Public interface

InspectRT 0.1.0 supports the documented command-line and serialized interfaces
below. The `fixture`, `onnx` and `portability` names are required-subcommand
namespaces; their leaf commands have separate scopes.

## v0.1.0 command surface

| Command | Category | Availability | Prerequisites | Primary output | Support status | Guide |
| --- | --- | --- | --- | --- | --- | --- |
| `inspectrt evaluate` | Installed/user | Installed distribution or source checkout | MVTec AD category, official pretrained weight and a valid PyTorch device | Seven-file evaluation run bundle | Supported in 0.1.0 | [Baseline](baseline.md) |
| `inspectrt benchmark` | Installed/user | Installed distribution or source checkout | Frozen `bottle` inventory, accepted pretrained weight already cached, supported device, exactly 5 warm-ups and 30 measured repetitions | Evaluation bundle plus benchmark schema 2 | Supported in 0.1.0 for the frozen workload | [Baseline](baseline.md) |
| `inspectrt fixture validate` | Installed/user | Installed distribution or source checkout | Base installation; CPU for the bundled synthetic fixture | Validation status and fixture identity | Supported in 0.1.0 | [Retrieval fixtures](retrieval-fixtures.md) |
| `inspectrt fixture export` | Source-checkout reproducibility | Verified clean source checkout only | Explicit config, frozen accepted schema-1 run, dataset, sample ID, accepted cached weight, recorded device and output root | Schema-1 `manifest.json` and `tensors.bin` | Supported for the documented reproduction workflow | [Retrieval fixtures](retrieval-fixtures.md) |
| `inspectrt onnx validate` | Installed/user | Installed distribution or source checkout | `inspectrt[onnx]` and a two-file schema-1 artifact | Validated artifact and graph identities | Supported in 0.1.0 | [ONNX portability](onnx-portability.md) |
| `inspectrt onnx export` | Source-checkout reproducibility | Verified clean source checkout only | `inspectrt[onnx]`, accepted cached weight and output root | Schema-1 `manifest.json` and `model.onnx` | Supported for the documented reproduction workflow | [ONNX portability](onnx-portability.md) |
| `inspectrt portability compare` | Maintainer/evidence | Source checkout only | Schema-1 reference and candidate runs, environment map, optional policy and new output directory | Scientific schema 1 and descriptive performance schema 1 | Documented, scoped evidence tooling | [Cross-platform evidence](portability.md) |
| `inspectrt portability performance` | Maintainer/evidence | Source checkout only | Reviewed scientific record, policy, six-row environment map, exactly six ordered timing runs and output file | `inspectrt_portability_performance_v2` JSON | Documented, scoped evidence tooling | [Cross-platform evidence](portability.md) |

`evaluate` and `benchmark` accept `--dataset-root`, `--category` and
`--device`. They also accept optional `--config`, `--output-root` and
`--run-id`. The benchmark accepts only `--warmup-count 5` and
`--repeat-count 30`. A CUDA device must include its index, such as `cuda:0`.

`fixture export` requires one occurrence each of `--config`, `--run-dir`,
`--dataset-root`, `--sample-id`, `--device` and `--output-root`. It consumes
the frozen historical benchmark schema 1. Current benchmarks emit schema 2.

`portability compare` requires `--reference-run`, `--environment-map` and
`--output`. `--candidate-run` is repeatable, must occur at least once and must
occur once for every candidate in the environment map. `--policy` is optional.
`portability performance` requires `--scientific`, `--policy`,
`--environment-map` and `--output`, while `--timing-run` must occur exactly
six times in environment-map order. Both commands require source provenance
and write to an output path separate from the source bundles.

## Serialized contracts

Stability applies to each complete declared format.

| Contract | Classification | Compatibility boundary |
| --- | --- | --- |
| `inspectrt_feature_memory_v1` baseline TOML, schema 1 | Supported public contract | Strict keys and frozen profile values |
| Repository evaluation run, schema 1 | Supported public contract | Complete seven-file bundle with Git and lock provenance |
| Installed evaluation run, schema 2 | Supported public contract | Complete seven-file bundle with distribution and profile provenance |
| Current `benchmark.json`, schema 2 | Supported public contract | Linked to its evaluation run; frozen `bottle` methodology |
| Retrieval fixture, schema 1 | Supported public contract | Complete two-file fixture and exact inventory |
| `inspectrt_onnx_feature_portability_v1` artifact, schema 1 | Supported public contract | Complete two-file artifact and graph metadata |
| `inspectrt_portability_comparison_v1` | Published evidence format | Tracked scientific evidence identity is frozen |
| `inspectrt_portability_performance_v2` | Published evidence format | Tracked descriptive timing identity is frozen |
| `inspectrt_onnx_feature_portability_scientific_v1` | Published evidence format | Tracked ONNX scientific identity is frozen |
| `inspectrt_portability_policy_v1` and retrieval workload matrix schema 1 | Published reviewed reference formats | Inputs to the frozen evidence |
| Historical benchmark schema 1 | Published evidence input | Read by the frozen reproduction workflows; `benchmark` emits schema 2 |
| `inspectrt_portability_environment_map_v1`, `inspectrt_portability_performance_v1` and the exact timing-run input | Maintainer-generation formats | Scoped to reviewed evidence generation |

CLI text output and individual run or fixture members follow their enclosing
command and bundle contracts.

## Compatibility during 0.x

- Fixes preserve documented behavior within the same schema version.
- An incompatible serialized change requires a new schema version.
- A breaking documented CLI change during 0.x requires an explicit minor
  release and changelog entry.
- Frozen evidence files retain their recorded identities.

## Python API

Importable Python modules, classes, functions, constants and names listed in a
module's `__all__` are experimental unless a public document explicitly names
them as supported. Experimental Python symbols may change between 0.x minor
releases.

The direct `OnnxRuntimeCpuFeatureConsumer`,
`OnnxRuntimeFeatureOutputs` and `OnnxRuntimeSessionMetadata` path is an
experimental Python API. The supported ONNX interface is the documented
artifact contract together with the export and validation commands in their
declared scopes. The [ONNX guide](onnx-portability.md) documents the fixed ONNX
Runtime 1.28.0, `CPUExecutionProvider`, fallback, graph-validation and
output-tensor behavior.
