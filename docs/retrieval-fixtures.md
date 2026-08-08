# Retrieval fixtures

## Scope

A retrieval fixture contains one exact top-1 squared-L2 retrieval problem. This
keeps retrieval separate from image decoding, preprocessing, feature
extraction, anomaly-map reconstruction, and PyTorch serialization.

The fixture contains the retrieval inputs and the expected outputs needed to
check another consumer. InspectRT currently validates those outputs with its
existing exact PyTorch retrieval reference. The repository does not contain an
alternative retrieval backend.

## Retrieval contract

The two input matrices share one feature dimension:

```text
queries:     [Q, D]
memory bank: [M, D]
```

For each query, exhaustive search returns one bank row:

```text
expected squared-L2 distances: [Q]
expected global bank indices:  [Q]
```

The distance is the sum of squared component differences. Queries and bank rows
are FP32, finite, C-contiguous, and row-major. Distances are raw FP32 squared L2
values. No square-rooted distance is stored or compared, and inputs are not
normalized. Indices are signed 64-bit global bank-row indices.

The contract fixes `k=1` and exact exhaustive search (no approximation).
If computed distances tie exactly, the lower global index is retained,
including when equal minimum occurs in different reference chunks.

## Two-file format

A fixture directory contains exactly:

```text
manifest.json
tensors.bin
```

`manifest.json` is canonical, schema-versioned UTF-8 JSON. It records the
workload, retrieval semantics, source identity, payload identity, and storage
metadata for these ordered segments:

| Segment | Shape | Storage |
|---|---:|---|
| `queries` | `[Q, D]` | little-endian IEEE-754 binary32 |
| `memory_bank` | `[M, D]` | little-endian IEEE-754 binary32 |
| `expected_squared_l2_distances` | `[Q]` | little-endian IEEE-754 binary32 |
| `expected_indices` | `[Q]` | little-endian signed 64-bit integer |

Each segment in `tensors.bin` begins at a 64-byte-aligned file offset.
Intervening padding is zero. The manifest gives every tensor's shape, offset,
byte count, byte order, layout, and SHA-256 hash. It also gives the complete
payload length and SHA-256 hash, including padding.

The 64-byte offsets are a file-layout property. They do not guarantee that a
segment will have the same in-memory alignment after a consumer loads or maps
the file.

You only need JSON and binary parsing to read the fixtures.

## Committed synthetic fixture

The committed fixture is:

```text
tests/fixtures/retrieval_v1/
├── manifest.json
└── tensors.bin
```

Its frozen identity is:

```text
fixture_id = synthetic-correctness-v1
Q/M/D/k = 4/7/5/1
reference chunk size = 3
```

Seven bank rows span three chunks, so the final chunk is not full. The values
exercise a tie within one chunk, a tie across chunks, and lower-index retention
in both cases. All inputs and expected outputs are synthetic and exactly
representable. The committed manifest and tests are the source for the complete
tensor values.

Validate it on the canonical acceptance device with:

```bash
uv run inspectrt fixture validate \
  --fixture tests/fixtures/retrieval_v1 \
  --device cpu
```

Validation performs these stages:

1. Parse and validate the manifest.
2. Verify offsets, lengths, zero padding, segment hashes, and the payload hash.
3. Load the four raw arrays.
4. Run the current exact PyTorch reference with the recorded chunk size.
5. Require exact indices and exact distances.
6. Report `status=accepted`.

The command reports `fixture_id`, `fixture_class`, `Q`, `M`, `D`, `k`,
`chunk_size`, `payload_sha256`, `fixture_digest`, `indices`, `distances`, and
`status`. The accepted comparison fields are:

```text
indices=exact
distances=exact
status=accepted
```

CPU is the canonical acceptance device for this synthetic fixture, so it can
run on pretty much any device.

## Real application fixture

The accepted real fixture is generated locally at:

```text
outputs/fixtures/inspectrt_retrieval_fixture_v1/
└── bottle-broken-large-000-bc330b9070c5/
    ├── manifest.json
    └── tensors.bin
```

It records:

```text
category = bottle
sample_id = mvtec_ad/bottle/test/broken_large/000.png
Q/M/D/k = 1024/214016/512/1
```

The payload contains:

```text
queries                           [1024, 512] float32
memory_bank                       [214016, 512] float32
expected_squared_l2_distances     [1024] float32
expected_indices                  [1024] int64
```

The payload is approximately 420 MiB and includes the complete accepted nominal
`bottle` bank. It is derived from a local MVTec AD installation and an accepted
local benchmark run.

The real fixture is generated below the gitignored `outputs/` tree.

## Export

When the deterministic destination is absent, export the local bottle fixture
from the accepted run with:

```bash
uv run inspectrt fixture export \
  --config configs/baseline.toml \
  --run-dir outputs/runs/20260715T202846302048Z-bottle-bc330b9 \
  --dataset-root datasets/mvtec_ad \
  --sample-id mvtec_ad/bottle/test/broken_large/000.png \
  --device cuda:0 \
  --output-root outputs
```

Export starts from a verified benchmark run, then checks its source identities
and five recorded source-artifact hashes. Since the query tensor is not in the
run bundle, export extracts it again and recomputes retrieval with the frozen
profile. The accepted export ran on the recorded environment. Every export must
match the accepted run's stored distances and indices exactly before it writes
the fixture.

The repository working tree must be clean, and the write is atomic. The
deterministic fixture directory must not already exist because export refuses
to overwrite it. The dataset and pretrained weight must be obtained manually.

## Real-fixture validation

On an environment matching the recorded accepted run, validate with:

```bash
uv run inspectrt fixture validate \
  --fixture outputs/fixtures/inspectrt_retrieval_fixture_v1/bottle-broken-large-000-bc330b9070c5 \
  --device cuda:0
```

Successful reference acceptance ends with:

```text
environment=exact
indices=exact
distances=exact
status=accepted
```

If structural and hash validation succeeds but the device, platform, or
dependency identity does not match, the outcome fields include:

```text
status=structurally_valid
reference_status=unavailable
```

The command also reports `environment_mismatches` with the differing recorded
identity fields.

## Workload matrix

`configs/retrieval_workloads.json` defines frozen workload shapes and synthetic
input-generation rules. It contains no benchmark results.

| Workload                       | Class                |     Q |       M |   D |  k |
| ------------------------------ | -------------------- | ----: | ------: | --: | -: |
| `synthetic-correctness`        | correctness          |     4 |       7 |   5 |  1 |
| `synthetic-development-small`  | development          |    32 |   4,096 | 512 |  1 |
| `synthetic-development-medium` | development          |   256 |  65,536 | 512 |  1 |
| `mvtec-bottle-image`           | application          | 1,024 | 214,016 | 512 |  1 |
| `mvtec-leather-image`          | application metadata | 1,024 | 250,880 | 512 |  1 |

Both application rows use `class: application` in the JSON. The
`application metadata` label above distinguishes the leather row's
`accepted_baseline_shape` source from a generated fixture.

The matrix records two independent scaling axes:

```text
query scaling:
Q = [1, 32, 256, 1024, 4096]
M = 214016

bank scaling:
M = [4096, 16384, 65536, 214016, 250880]
Q = 1024
```

Both axes otherwise fix:

```text
D = 512
k = 1
float32
C-contiguous row-major
```

## Deterministic synthetic values

Development and scaling inputs use `counter_fp32_v1`. For zero-based row `i`
and dimension `j`, compute with unsigned 64-bit integer arithmetic:

```text
n = (a * (i + 1) + b * (j + 1) + salt) mod 16777213
s = int64(n) - 8388606
value = float32(s) / 262144
```

The coefficient sets are:

```text
queries: a=131071, b=524287, salt=17
banks:   a=104729, b=130363, salt=31
```

## Current limits

- Only fixture schema version 1 is supported.
- Retrieval fixes `k=1`, FP32 inputs, and C-contiguous row-major layout.
- The real fixture remains local and gitignored.
- Real numerical acceptance is limited to the recorded environment.
- No cross-implementation floating-point tolerance is fixed.
- No separate C++, OpenMP, native CUDA, FAISS, cuVS, or other alternative
  retrieval backend is included: CUDA validation uses the PyTorch reference.
- No performance conclusion currently follows from the fixture or workload matrix.
