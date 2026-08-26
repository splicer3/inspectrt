# Retrieval fixtures

## Scope

A retrieval fixture contains one exact top-1 squared-L2 retrieval problem. This
keeps retrieval separate from image decoding, preprocessing, feature
extraction, anomaly-map reconstruction, and PyTorch serialization.

The fixture contains the retrieval inputs and expected outputs needed to check
another consumer. InspectRT validates those outputs with its exact PyTorch
retrieval reference.

`fixture validate` is a supported installed/user command. `fixture export` is
a verified-source reproducibility command. See
[Public interface](public-interface.md) for their exact availability and the
schema compatibility boundary.

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
are FP32, finite, C-contiguous, and row-major. The fixture stores and compares
raw FP32 squared-L2 values directly, with the stored inputs passed unchanged to
retrieval. Indices are signed 64-bit global bank-row indices.

The contract fixes `k=1` and exact exhaustive search.
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

The 64-byte offsets define the file layout. A consumer that requires aligned
runtime allocation copies the decoded tensor into aligned storage.

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
manifest bytes = 1577
manifest SHA-256 = 9e72d4238ee0cae7f8236a82e50acf8f811c0e3f7b5e2815a11c56a9e1193c12
payload bytes = 416
payload SHA-256 = 18c2c4333a060ff25b7304dd396cf4b292617c4593d7cbfc2576b406ed5a14bb
fixture digest = ec30a68439f52051028a56cbd5a1c560edc2bccc4e77e603fa2d3355a26a4e9e
```

Seven bank rows span three chunks, so the final chunk is not full. The values
exercise a tie within one chunk, a tie across chunks, and lower-index retention
in both cases. All inputs and expected outputs are synthetic and exactly
representable. The committed manifest and tests are the source for the complete
tensor values.

Validate it on the canonical acceptance device with:

```bash
inspectrt fixture validate --device cpu
```

Omitting `--fixture` selects this exact bundled fixture. Explicit-path
validation remains supported:

```bash
inspectrt fixture validate \
  --fixture /path/to/retrieval_v1 \
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

CPU is the canonical acceptance device for this synthetic fixture. The check
runs offline from the installed fixture bytes.

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

The real fixture is generated below the gitignored `outputs/` tree. It remains
local and follows the source dataset's distribution terms.

## Export

This source-checkout workflow reproduces the accepted real fixture from the
frozen historical schema-1 benchmark bundle. Export it only when its
deterministic destination does not exist:

```bash
uv run inspectrt fixture export \
  --config configs/baseline.toml \
  --run-dir outputs/runs/20260715T202846302048Z-bottle-bc330b9 \
  --dataset-root datasets/mvtec_ad \
  --sample-id mvtec_ad/bottle/test/broken_large/000.png \
  --device cuda:0 \
  --output-root outputs
```

Before writing, the command verifies the benchmark run, its source identities,
and five recorded source-artifact hashes. It regenerates the query tensor from
the dataset and recomputes retrieval with the frozen profile. It requires an
exact match with the accepted run's stored distances and indices. The accepted
export ran on the recorded environment.

Export requires a clean working tree and writes the fixture atomically. It
refuses to overwrite the deterministic fixture directory. The dataset and
pretrained weight must be obtained manually. The command also requires the
repository `uv.lock`, explicit baseline config, accepted cached weight and the
recorded device. It reads the frozen historical benchmark schema 1.

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
identity fields. An installed distribution can establish structural validity,
but exact reference acceptance requires the matching source checkout, lock,
platform, dependencies and device.

## Workload matrix

`configs/retrieval_workloads.json` defines frozen workload shapes and synthetic
input-generation rules. Timing results live in the portability evidence.

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

## Contract scope

Installation and platform support are documented in
[Installation and support](installation.md). The supported fixture contract is
classified in [Public interface](public-interface.md).

- Fixture schema 1 uses `k=1`, FP32 inputs, and C-contiguous row-major layout.
- The bundled synthetic fixture provides exact CPU correctness acceptance.
- The local real fixture provides exact acceptance on its recorded environment.
- CPU and CUDA validation use the same PyTorch retrieval reference.
- Performance measurements are published in the portability evidence.
