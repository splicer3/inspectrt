# Installation and support

InspectRT 0.1.0 supports CPython 3.11 and 3.12. Set `WHEEL` to a reviewed
release artifact on your filesystem:

```bash
WHEEL=/absolute/path/to/inspectrt-0.1.0-py3-none-any.whl
```

A standards-compliant installer builds a source distribution into a wheel
before installation.

## Default installation

On platforms where the ordinary PyTorch resolution is appropriate:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python "$WHEEL"
uv pip check --python .venv/bin/python
```

Linux CPU users should use the CPU-first sequence below. On Linux x86-64,
PyTorch's ordinary package-index resolution can include CUDA support and large
NVIDIA dependencies even for CPU use. Published dependency metadata leaves the
PyTorch wheel-index choice to the installer.

In a 2026-08-25 CPython 3.12 resolver snapshot, the InspectRT wheel selected 38
distributions, including PyTorch 2.13.0, torchvision 0.28.0, Triton, CUDA
bindings and 15 NVIDIA packages. The selected compatible wheel sizes totalled
2,809,649,979 bytes (2.617 GiB) compressed. These figures describe that Linux
resolver snapshot.

## Linux x86-64 CPU first

Install the exact supported PyTorch pair from the
[official PyTorch CPU index](https://download.pytorch.org/whl/cpu) before
installing InspectRT:

```bash
uv venv --python 3.12 .venv --no-project --no-config
uv pip install --python .venv/bin/python --no-config \
  torch==2.13.0 torchvision==0.28.0 \
  --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python --no-config "$WHEEL"
uv pip check --python .venv/bin/python --no-config
```

Verify the installed distribution, CPU builds and offline fixture:

```bash
.venv/bin/python -c 'import inspectrt, torch, torchvision; print(torch.__version__, torchvision.__version__); assert not torch.cuda.is_available()'
.venv/bin/python -c 'from importlib.metadata import version; print(version("inspectrt"))'
.venv/bin/inspectrt fixture validate --device cpu
```

This sequence was tested on Linux x86-64 with CPython 3.12.13 and resolved 19
distributions: `inspectrt 0.1.0`, `torch 2.13.0+cpu` and
`torchvision 0.28.0+cpu` among them. `torch.version.cuda` was `None`,
`torch.cuda.is_available()` was false and the environment occupied
1,014,686,965 bytes after validation. The resolver displayed approximately
261 MiB of compressed downloads. These measurements record the tested
environment.

## Linux x86-64 CUDA first

Use the current official [PyTorch installation
selector](https://pytorch.org/get-started/locally/) to choose Linux, Python,
the package installer and the compute platform appropriate to the installed
driver. For the supported pair, install `torch==2.13.0` and
`torchvision==0.28.0` from the official index returned by that selector before
installing InspectRT. On 2026-08-25 the selector listed CUDA 12.6, 13.0 and
13.2 choices; use its current result for the target system.

Then install and verify the InspectRT wheel:

```bash
uv pip install --python .venv/bin/python "$WHEEL"
uv pip check --python .venv/bin/python
.venv/bin/python -c 'import inspectrt, torch, torchvision; print(torch.__version__, torchvision.__version__, torch.version.cuda); assert torch.cuda.is_available()'
```

InspectRT addresses CUDA devices through PyTorch as `cuda:<index>`, for example
`cuda:0`. The optional ONNX tools use ONNX Runtime's
`CPUExecutionProvider`.

## WSL 2

Treat WSL 2 Ubuntu x86-64 as Linux. Use the CPU-first sequence for CPU
execution. For CUDA, first satisfy Microsoft's and NVIDIA's WSL driver
requirements, then use the current PyTorch selector and the same CUDA-first
policy above. CUDA inside WSL uses the Windows-host NVIDIA driver.

## macOS 14+ ARM64

The current PyTorch 2.13.0 and ONNX Runtime 1.28.0 ARM64 wheels require macOS
14 or later. The ordinary macOS ARM64 PyTorch wheels provide CPU execution and,
where available, PyTorch MPS. Install the InspectRT wheel in a fresh CPython
3.11 or 3.12 environment, run `uv pip check` and validate the bundled fixture
on CPU:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python \
  torch==2.13.0 torchvision==0.28.0
uv pip install --python .venv/bin/python "$WHEEL"
uv pip check --python .venv/bin/python
.venv/bin/inspectrt fixture validate --device cpu
```

Use `--device mps` for evaluation or the frozen benchmark only after
`torch.backends.mps.is_built()` and `torch.backends.mps.is_available()` both
report true. The synthetic fixture's canonical acceptance device remains CPU.

## Optional ONNX tools

Install the extra from the same local wheel:

```bash
uv pip install --python .venv/bin/python "$WHEEL[onnx]"
uv pip check --python .venv/bin/python
.venv/bin/python -c 'from importlib.metadata import version; print(version("onnxruntime"))'
```

The extra installs ONNX Runtime 1.28.0 for CPU artifact validation. Artifact
export also requires the accepted pretrained weight in the torchvision cache
and a verified source checkout.

## Network and storage behavior

The wheel contains the baseline profile and canonical synthetic fixture. The
`fixture validate --device cpu` quickstart runs offline from those bundled
bytes and leaves the Torch weight cache unchanged.

Obtain MVTec AD separately under its own terms. `evaluate` uses the official
torchvision `ResNet50_Weights.IMAGENET1K_V2` weight and may download it when it
is absent from the user's cache. `benchmark`, fixture export and ONNX export
require the accepted weight bytes to be cached already. Evaluations,
benchmarks, fixtures, and ONNX exports write their artifacts to user-selected
output directories.

Source-checkout reproduction is documented in the relevant
[public-interface](public-interface.md), [baseline](baseline.md),
[retrieval-fixture](retrieval-fixtures.md) and
[ONNX](onnx-portability.md) guides.

## v0.1.0 platform support

InspectRT verifies its supported platforms through these release gates:

| Platform | Execution path | Release verification |
| --- | --- | --- |
| Linux x86-64 CPU, CPython 3.11 and 3.12 | PyTorch CPU wheels | Automated checks on both Python versions before release; the P3 wheel path is validated. |
| Linux x86-64 CUDA | PyTorch `cuda:<index>` | Manual check of the release wheel and selected PyTorch CUDA build. |
| WSL 2 Ubuntu x86-64 | PyTorch CPU or `cuda:<index>` | Manual CPU and CUDA checks of the release wheel. |
| macOS 14+ ARM64 | PyTorch CPU or MPS | Manual CPU and MPS checks of the release wheel. |

The [portability record](portability.md) gives the reviewed software and
hardware identities for earlier Linux, WSL 2, and macOS runs.
