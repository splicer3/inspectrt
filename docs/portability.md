# Cross-platform evidence

This page records one frozen `inspectrt_feature_memory_v1` workload on a small,
named set of systems. The scope is the original MVTec AD `bottle` category and
source commit `bc330b9070c5ca8db9cb7cfbb27617256388536b`. These measurements do
not imply that InspectRT behaves identically on every device.

## What was tested

Every run used the accepted configuration, lock, ResNet-50
`IMAGENET1K_V2` weights, and ordered `bottle` inventory. Images were resized to
`256 x 256`; frozen `layer2` features became row-major `[1, 1024, 512]` FP32
patches. Retrieval remained exact chunked top-1 squared L2 against the complete
`[214016, 512]` nominal bank. The profile, sample order, image-score semantics,
map reconstruction, and threshold-free metrics were unchanged.

The scientific source was the clean commit above. The comparison was generated
from clean commit `25654276682de466efa4df3d86c3d3d3d165113e`.

| Policy role | Environment | Execution | Bundle | Timing |
| --- | --- | --- | --- | --- |
| Reference | Quadro T1000 CUDA, Ubuntu 24.04.4 | Native Linux | Benchmark | Included |
| Same-stack control | Quadro T1000 CUDA, Ubuntu 24.04.4 | Native Linux | Benchmark | Included |
| Calibration | Intel Core i7-9850H CPU, Ubuntu 24.04.4 | Native Linux | Benchmark | Included |
| Calibration | RTX 4080 Super CUDA, Ubuntu 24.04.4 | WSL 2 | Benchmark | Included |
| Holdout | Apple M1 Pro CPU, macOS 26.5.2 | Native macOS | Benchmark | Included |
| Post-policy evaluation | Apple M1 Pro MPS, macOS 26.5.2 | Native macOS | Evaluation | Excluded: `evaluation_bundle` |

The reference, same-stack control, P53 CPU, and RTX WSL 2 runs established the
calibration record. The policy was reviewed before the M1 Pro CPU holdout and
the later MPS evaluation. Neither Mac result changed its limits.

## Frozen workload and evidence matrix

The policy requires exact provenance and exact sample IDs, labels, masks, and
nearest-neighbour indices. Floating components use the elementwise rule:

```text
abs(candidate - reference) <= atol + rtol * abs(reference)
```

The reviewed absolute limits are `0.00011` for `memory_bank`, `0.003` for
`patch_distances`, `0.0015` for `image_scores`, and `0.003` for
`anomaly_maps`; every `rtol` is zero. Metric absolute-delta limits are zero for
image AUROC and image Average Precision, and `0.000000003` for pixel AUROC.

All 29 structural gates were comparable for every completed candidate. The
ordered 83 sample IDs and labels were exact, as were all 5,439,488 mask
elements. Every floating component had zero policy violations.

## Scientific results

| Environment | Policy role | Status | Floating violations | Nearest-index mismatch | Metric limits | Timing eligibility |
| --- | --- | --- | ---: | ---: | --- | --- |
| Quadro T1000 control | Same-stack control | `within_policy` | 0 | 0 / 84,992 (0.00%) | Satisfied | Included |
| P53 CPU | Calibration | `drift_detected` | 0 | 3,569 / 84,992 (4.20%) | Satisfied | Included |
| RTX 4080 Super CUDA under WSL 2 | Calibration | `drift_detected` | 0 | 1,826 / 84,992 (2.15%) | Satisfied | Included |
| M1 Pro CPU | Holdout | `drift_detected` | 0 | 2,701 / 84,992 (3.18%) | Satisfied | Included |
| M1 Pro MPS | Post-policy evaluation | `drift_detected` | 0 | 3,019 / 84,992 (3.55%) | Satisfied | Excluded: `evaluation_bundle` |

Image AUROC and image Average Precision were unchanged. Every pixel AUROC
delta remained inside the reviewed limit. The cross-device records are
`drift_detected` because nearest-neighbour identity is an exact gate, not
because floating limits or metric limits were exceeded. The result is neither
a scientific failure nor a statement of universal equivalence.

## Descriptive latency observations

![Panel A shows feature extraction, exact retrieval, and end-to-end p50 to p95 latency for five timing-valid environments. Panel B shows the share of 84,992 patch queries whose nearest-neighbour index differs from the T1000 reference for all five completed candidates, including MPS.](evidence/inspectrt_cross_platform_evidence_v1/latency.svg)

The table contains the persisted millisecond values in evidence order. These
are absolute observations, not ratios or rankings.

| Environment | Frozen feature extraction p50 / p95 | Exact retrieval p50 / p95 | Synchronized end to end p50 / p95 |
| --- | ---: | ---: | ---: |
| Quadro T1000 reference | `6.046144008636475 / 6.101880002021789` | `245.9578399658203 / 246.05908203125` | `273.224114 / 274.39006565` |
| Quadro T1000 control | `6.015615940093994 / 6.034054493904113` | `245.87073516845703 / 245.91729965209962` | `273.4881115 / 273.74749525` |
| P53 CPU | `26.251528 / 26.845392999999998` | `1029.2755245 / 1045.63147215` | `1082.7934194999998 / 1099.7182692` |
| RTX 4080 Super CUDA under WSL 2 | `3.5883361101150513 / 10.51667079925537` | `18.99009609222412 / 19.977069091796874` | `38.485780500000004 / 45.33286065` |
| M1 Pro CPU | `18.197083499999998 / 19.20166235` | `202.98639550000001 / 205.35461859999998` | `236.024375 / 238.37803385` |

Each benchmark used five warm-ups and 30 measured repeats on one batch-1 test
sample. Feature timing includes frozen `layer2` inference, local averaging,
row-major layout, and output allocation. Retrieval includes the complete
chunked search, stable merge, reshape, image maximum, and allocations. The
synchronized end-to-end boundary starts before image decode and ends after the
raw anomaly map is ready. It excludes model and bank setup, masks, metrics,
serialization, persistence, and console output.

The summaries do not retain raw repetitions. Host load, power state, and
thermal state were not controlled across machines, so `performance.json` is
`descriptive_only`. Host peak memory was not measured; no approximation was
added.

## MPS scientific-only result

The MPS run produced a valid seven-file evaluation bundle. It was
post-policy, non-calibrating, non-gating, and scientific-only. Its floating
outputs had zero policy violations, its metrics remained inside their limits,
and 3,019 nearest indices differed. No MPS latency was measured or inferred.

## How the policy was established

The same-stack control, P53 CPU, and RTX 4080 Super CUDA under WSL 2 defined the
observed calibration envelope. Each absolute tolerance is the next simple
decimal above the largest calibration error for that component. Relative
limits remain zero because near-zero reference values made relative maxima a
poor policy boundary. Exact discrete requirements stayed exact, including
nearest-neighbour identity.

The M1 Pro CPU was held out until after review. Its floating and metric results
fit the existing envelope, while exact-index differences remained visible.
The subsequent MPS evaluation followed the same policy and did not widen it.

## Reproducing and inspecting the evidence

The two JSON files are the canonical machine-readable evidence:

- [scientific.json](evidence/inspectrt_cross_platform_evidence_v1/scientific.json)
- [performance.json](evidence/inspectrt_cross_platform_evidence_v1/performance.json)
- [latency.svg](evidence/inspectrt_cross_platform_evidence_v1/latency.svg)
- [portability policy](../configs/portability_policy.json)

Check the tracked graph against an in-memory regeneration with:

```bash
uv run python scripts/render_portability_latency.py \
  --scientific docs/evidence/inspectrt_cross_platform_evidence_v1/scientific.json \
  --performance docs/evidence/inspectrt_cross_platform_evidence_v1/performance.json \
  --check docs/evidence/inspectrt_cross_platform_evidence_v1/latency.svg
```

| Artifact | Identity |
| --- | --- |
| Comparison | `1dec773f2d237598305a315145bec7bc40b9f94fbd326ed44f6330d3c9a11fe5` |
| Policy | `inspectrt-bottle-bc330b9-v1` |
| Policy SHA-256 | `576717b70e53714eed8370619cc08c81517405728f767942298b0c8c415836a2` |
| `scientific.json` SHA-256 | `81318cd81c0e5f23be953719c2bb03604c22c75bb8c1dd17c389786623d32b8a` |
| `performance.json` SHA-256 | `b8e57ac6098c89d11d002e797a6d7a79774c871e548f6fbf02d14ded786cb893` |
| `latency.svg` SHA-256 | `05e14dd3d671a70439e5100b1ee40feb32eceefffc9da2cf014731b87ca66e18` |

## Limits of the result

The scope stops at the named profile, original `bottle` category, versions, and
machines. The comparison directly covers the nominal feature bank and persisted
downstream artifacts; raw test feature tensors are not stored. One complete run
per candidate cannot establish behavior across future library versions,
drivers, operating-system updates, or hardware.

The file hashes above are current snapshots. They cannot prove the historical
identity of files that lacked earlier per-file anchors. Without raw repetitions,
the latency summaries cannot support confidence intervals. Uncontrolled host
conditions also prevent broader cross-machine performance conclusions.

`RT` means Runtime. InspectRT measures latency, but this evidence makes no
hard-real-time claim.
