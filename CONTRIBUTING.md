# Contributing to InspectRT

InspectRT is a reduced, reproducible feature-memory baseline for industrial
anomaly-detection research. Contributions should stay within the documented
[public interface](docs/public-interface.md) and the current scientific
boundary. The [installation guide](docs/installation.md) describes supported
user environments.

## Development setup

Clone the repository and use CPython 3.11 or 3.12. The public CI path is
CPU-only and pins uv 0.12.6 while uv's PyTorch backend selection remains a
preview feature. From the checkout, create a base development environment:

```bash
uv venv --python 3.11 .venv --no-project --no-config
uv --quiet export --locked --python 3.11 --no-emit-project \
  --prune torch --prune torchvision \
  --output-file .venv/inspectrt-locked.txt
uv pip install --python .venv/bin/python --no-config --strict \
  --torch-backend cpu --requirement .venv/inspectrt-locked.txt \
  "torch==2.13.0+cpu" "torchvision==0.28.0+cpu"
uv pip install --python .venv/bin/python --no-config --strict \
  --no-deps --editable .
```

Use `--python 3.12` for the other supported interpreter. To include the
optional ONNX tools, use the same CPU-first path with the extra selected:

```bash
uv --quiet export --locked --python 3.11 --extra onnx --no-emit-project \
  --prune torch --prune torchvision \
  --output-file .venv/inspectrt-locked.txt
uv pip install --python .venv/bin/python --no-config --strict \
  --torch-backend cpu --requirement .venv/inspectrt-locked.txt \
  "torch==2.13.0+cpu" "torchvision==0.28.0+cpu" \
  "onnxruntime==1.28.0"
uv pip install --python .venv/bin/python --no-config --strict \
  --no-deps --editable ".[onnx]"
```

Do not substitute `uv sync --locked` for these Linux CPU commands: the
reviewed lock records the ordinary PyPI PyTorch graph, which includes CUDA
packages on Linux. The lock remains an input and integrity boundary, while CI
audits the resolved CPU graph separately.

## Checks

Run the checks relevant to the change, and run the complete suite before
requesting review:

```bash
.venv/bin/python -m pytest -q --runxfail -p no:cacheprovider
.venv/bin/python -m ruff check .
.venv/bin/python -m ruff format --check .
.venv/bin/python -m compileall -q inspectrt scripts tests
uv lock --check
uv build --no-sources
git diff --check
```

The current collection is 96 nodes; the normal ceiling is 100. A clean base
environment passes 93 and skips exactly:

- `tests/test_fixtures.py::test_ignored_real_fixture_manifest_matches_bottle_metadata_when_present`;
- `tests/test_onnx_artifacts.py::test_graph_contract_metadata_and_external_data_fail_closed`;
- `tests/test_onnx_features.py::test_exports_static_dual_outputs_with_pool_and_row_major_layout`.

With the ONNX extra, 95 pass and only the real-fixture node above skips.
Unexpected skips, xfails, xpasses, deselection, failures or errors are not an
accepted result.

## Change boundaries

Open an issue before implementing a change to the scientific method, a
serialized schema, the documented CLI or frozen evidence. These boundaries
need agreement before code changes because apparently small edits can make
existing results incomparable. Frozen evidence and canonical fixtures are not
silently regenerated; an intentional change must preserve its review trail
and update every affected contract together.

Commit source, focused tests and concise documentation when they belong to the
public project. Do not commit `_extra`, datasets or MVTec bytes, pretrained
weights, outputs, generated models, real fixtures, memory banks, accepted run
bundles or private evidence. Keep generated build and test artifacts out of
the repository. Sanitize logs, paths, metadata and screenshots before sharing
them, and never include credentials or other secrets.

Report suspected vulnerabilities through [the security policy](SECURITY.md),
not a public issue. Participation is governed by the
[Code of Conduct](CODE_OF_CONDUCT.md).

## Pull requests and licensing

Pull-request titles follow this exact Conventional Commit-style rule:

```regex
^(?:build|chore|ci|docs|feat|fix|perf|refactor|revert|style|test)(?:\([A-Za-z0-9][A-Za-z0-9._/-]*\))?!?: \S.*$
```

For example, `fix(cli): reject an invalid argument` and
`docs!: revise a documented contract` are valid. Intermediate commits do not
need Conventional Commit messages. Squash merge is the normal merge strategy,
so the reviewed pull-request title supplies the final subject.

Signed contributor commits are not required. InspectRT requires neither a
Contributor License Agreement nor a Developer Certificate of Origin. Unless
explicitly stated otherwise, an intentionally submitted contribution is
licensed under Apache-2.0, consistently with section 5 of that license.
