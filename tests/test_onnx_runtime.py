from dataclasses import FrozenInstanceError
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

import inspectrt.onnx_runtime as runtime
from inspectrt.onnx_artifacts import LoadedOnnxFeatureArtifact

_ARTIFACT = LoadedOnnxFeatureArtifact(
    "resnet50-layer2-opset20-143b305b37a9",
    5_857_483,
    "143b305b37a92e3f2c7dc4268c25baccdf3cfb01c5304f29068f422ff9d8146a",
    "9b17d7dda2aea8979b4e89e00ba540485bec5863127294a9b2dc4db6fcc5e0b0",
    20,
)


class _EnumValue:
    def __init__(self, name: str) -> None:
        self.name = name


class _SessionOptions:
    def __init__(self) -> None:
        self.execution_mode = _EnumValue("ORT_SEQUENTIAL")
        self.graph_optimization_level = _EnumValue("ORT_ENABLE_ALL")
        self.enable_cpu_mem_arena = True
        self.enable_mem_pattern = True
        self.enable_mem_reuse = True
        self.intra_op_num_threads = 0
        self.inter_op_num_threads = 0
        self.enable_profiling = False
        self.use_deterministic_compute = False
        self.log_severity_level = -1
        self.log_verbosity_level = 0
        self.optimized_model_filepath = ""
        self.profile_file_prefix = "onnxruntime_profile_"


def _node(name: str, tensor_type: str, shape: list[object]) -> object:
    return SimpleNamespace(name=name, type=tensor_type, shape=shape)


class _Session:
    def __init__(self) -> None:
        self.events: list[str] = []
        self.active_providers = ["CPUExecutionProvider"]
        self.provider_options = {"CPUExecutionProvider": {}}
        self.inputs = [_node("images", "tensor(float)", [1, 3, 256, 256])]
        self.outputs = [
            _node("layer2", "tensor(float)", [1, 512, 32, 32]),
            _node("patch_embeddings", "tensor(float)", [1, 1024, 512]),
        ]
        self.options = _SessionOptions()
        self.run_results = [
            np.zeros((1, 512, 32, 32), dtype=np.float32),
            np.zeros((1, 1024, 512), dtype=np.float32),
        ]
        self.run_calls: list[tuple[list[str], dict[str, np.ndarray]]] = []
        self.input_snapshot: np.ndarray | None = None

    def disable_fallback(self) -> None:
        self.events.append("disable_fallback")

    def get_providers(self) -> list[str]:
        self.events.append("get_providers")
        return self.active_providers

    def get_provider_options(self) -> dict[str, dict[str, str]]:
        self.events.append("get_provider_options")
        return self.provider_options

    def get_session_options(self) -> _SessionOptions:
        self.events.append("get_session_options")
        return self.options

    def get_inputs(self) -> list[object]:
        self.events.append("get_inputs")
        return self.inputs

    def get_outputs(self) -> list[object]:
        self.events.append("get_outputs")
        return self.outputs

    def run(
        self, output_names: list[str], input_feed: dict[str, np.ndarray]
    ) -> list[object]:
        self.run_calls.append((output_names, input_feed))
        self.input_snapshot = input_feed["images"].copy()
        input_feed["images"].flat[0] = 123.0
        return self.run_results


class _OnnxRuntime:
    __version__ = "1.28.0"

    def __init__(self, session: _Session | None = None) -> None:
        self.available_providers = ["AzureExecutionProvider", "CPUExecutionProvider"]
        self.session = session or _Session()
        self.created_options: _SessionOptions | None = None
        self.inference_call: tuple[Path, _SessionOptions, list[str]] | None = None

    def get_available_providers(self) -> list[str]:
        return self.available_providers

    def get_version_string(self) -> str:
        return "1.28.0"

    def get_build_info(self) -> str:
        return "ORT Build Info: test"

    def SessionOptions(self) -> _SessionOptions:
        self.created_options = _SessionOptions()
        return self.created_options

    def InferenceSession(
        self,
        path: Path,
        *,
        sess_options: _SessionOptions,
        providers: list[str],
    ) -> _Session:
        self.inference_call = (path, sess_options, providers)
        return self.session


def _write_manifest(directory: Path, version: str = "1.28.0") -> None:
    (directory / "manifest.json").write_text(
        json.dumps({"environment": {"dependency_versions": {"onnxruntime": version}}})
    )


def _versions(cpu: str | None = "1.28.0", gpu: str | None = None) -> Any:
    def version(name: str) -> str:
        selected = cpu if name == "onnxruntime" else gpu
        if selected is None:
            raise runtime.importlib_metadata.PackageNotFoundError(name)
        return selected

    return version


def _install_fakes(
    monkeypatch: pytest.MonkeyPatch,
    directory: Path,
    onnxruntime: _OnnxRuntime,
    *,
    cpu_version: str | None = "1.28.0",
    gpu_version: str | None = None,
    manifest_version: str = "1.28.0",
) -> None:
    _write_manifest(directory, manifest_version)
    monkeypatch.setattr(
        runtime,
        "load_onnx_feature_artifact",
        lambda path: _ARTIFACT,
    )
    monkeypatch.setattr(
        runtime.importlib_metadata,
        "version",
        _versions(cpu_version, gpu_version),
    )
    monkeypatch.setattr(
        runtime.importlib,
        "import_module",
        lambda name: onnxruntime,
    )


def test_delays_runtime_import_and_loads_artifact_before_runtime_actions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        (
            sys.executable,
            "-c",
            "import sys; import inspectrt.onnx_runtime; "
            "assert not any(n == 'onnxruntime' or n.startswith('onnxruntime.') "
            "for n in sys.modules)",
        ),
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    _write_manifest(tmp_path)
    monkeypatch.setattr(runtime, "load_onnx_feature_artifact", lambda path: _ARTIFACT)
    monkeypatch.setattr(runtime.importlib_metadata, "version", _versions())
    monkeypatch.setattr(
        runtime.importlib,
        "import_module",
        lambda name: (_ for _ in ()).throw(ModuleNotFoundError(name)),
    )
    with pytest.raises(RuntimeError, match=r"inspectrt\[onnx\]"):
        runtime.OnnxRuntimeCpuFeatureConsumer.from_artifact(tmp_path)
    calls = []

    def reject(path: Path) -> object:
        calls.append(("artifact", path))
        raise ValueError("invalid artifact")

    monkeypatch.setattr(runtime, "load_onnx_feature_artifact", reject)
    monkeypatch.setattr(
        runtime.importlib,
        "import_module",
        lambda name: (_ for _ in ()).throw(AssertionError("runtime imported")),
    )
    with pytest.raises(ValueError, match="invalid artifact"):
        runtime.OnnxRuntimeCpuFeatureConsumer.from_artifact(tmp_path)
    assert calls == [("artifact", tmp_path)]


def test_rejects_wrong_runtime_distribution_or_unavailable_cpu_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cases = (
        (
            "1.28.0",
            "1.28.0",
            "1.28.0",
            ["CPUExecutionProvider"],
            "onnxruntime-gpu",
        ),
        (None, None, "1.28.0", ["CPUExecutionProvider"], r"inspectrt\[onnx\]"),
        ("1.27.0", None, "1.28.0", ["CPUExecutionProvider"], "artifact manifest"),
        ("1.29.0", None, "1.29.0", ["CPUExecutionProvider"], "unsupported"),
        ("1.28.0", None, "1.28.0", ["AzureExecutionProvider"], "unavailable"),
    )
    for cpu, gpu, manifest, providers, message in cases:
        onnxruntime = _OnnxRuntime()
        onnxruntime.available_providers = providers
        _install_fakes(
            monkeypatch,
            tmp_path,
            onnxruntime,
            cpu_version=cpu,
            gpu_version=gpu,
            manifest_version=manifest,
        )
        with pytest.raises(RuntimeError, match=message):
            runtime.OnnxRuntimeCpuFeatureConsumer.from_artifact(tmp_path)
        assert onnxruntime.inference_call is None


def test_constructs_exact_cpu_session_and_captures_immutable_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _Session()
    onnxruntime = _OnnxRuntime(session)
    _install_fakes(monkeypatch, tmp_path, onnxruntime)
    consumer = runtime.OnnxRuntimeCpuFeatureConsumer.from_artifact(tmp_path)
    metadata = consumer.metadata

    assert onnxruntime.inference_call == (
        tmp_path / "model.onnx",
        onnxruntime.created_options,
        ["CPUExecutionProvider"],
    )
    assert session.events[0] == "disable_fallback"
    assert metadata == runtime.OnnxRuntimeSessionMetadata(
        artifact_id=_ARTIFACT.artifact_id,
        model_sha256=_ARTIFACT.model_sha256,
        onnxruntime_distribution_version="1.28.0",
        onnxruntime_version="1.28.0",
        onnxruntime_version_string="1.28.0",
        onnxruntime_build_info="ORT Build Info: test",
        available_providers=("AzureExecutionProvider", "CPUExecutionProvider"),
        requested_providers=("CPUExecutionProvider",),
        active_providers=("CPUExecutionProvider",),
        provider_options=(("CPUExecutionProvider", ()),),
        fallback_disabled=True,
        session_options=(
            ("execution_mode", "ORT_SEQUENTIAL"),
            ("graph_optimization_level", "ORT_ENABLE_ALL"),
            ("enable_cpu_mem_arena", True),
            ("enable_mem_pattern", True),
            ("enable_mem_reuse", True),
            ("intra_op_num_threads", 0),
            ("inter_op_num_threads", 0),
            ("enable_profiling", False),
            ("use_deterministic_compute", False),
            ("log_severity_level", -1),
            ("log_verbosity_level", 0),
            ("optimized_model_filepath", ""),
            ("profile_file_prefix", "onnxruntime_profile_"),
        ),
        inputs=(("images", "tensor(float)", (1, 3, 256, 256)),),
        outputs=(
            ("layer2", "tensor(float)", (1, 512, 32, 32)),
            ("patch_embeddings", "tensor(float)", (1, 1024, 512)),
        ),
    )

    onnxruntime.available_providers.clear()
    session.active_providers.clear()
    session.provider_options["CPUExecutionProvider"]["changed"] = "yes"
    session.inputs[0].shape[0] = 9
    assert metadata.available_providers == (
        "AzureExecutionProvider",
        "CPUExecutionProvider",
    )
    assert metadata.active_providers == ("CPUExecutionProvider",)
    assert metadata.provider_options == (("CPUExecutionProvider", ()),)
    assert metadata.inputs[0][2] == (1, 3, 256, 256)
    with pytest.raises(FrozenInstanceError):
        metadata.artifact_id = "changed"


def test_rejects_wrong_active_provider_or_session_io_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def wrong_active(session: _Session) -> None:
        session.active_providers = ["AzureExecutionProvider"]

    def extra_active(session: _Session) -> None:
        session.active_providers.append("AzureExecutionProvider")

    def missing_input(session: _Session) -> None:
        session.inputs = []

    def noninteger_shape(session: _Session) -> None:
        session.inputs[0].shape[0] = 1.0

    def wrong_input_type(session: _Session) -> None:
        session.inputs[0].type = "tensor(double)"

    def reversed_outputs(session: _Session) -> None:
        session.outputs.reverse()

    def wrong_output_name(session: _Session) -> None:
        session.outputs[1].name = "patches"

    def wrong_output_shape(session: _Session) -> None:
        session.outputs[1].shape[-1] = 256

    for mutate in (
        wrong_active,
        extra_active,
        missing_input,
        noninteger_shape,
        wrong_input_type,
        reversed_outputs,
        wrong_output_name,
        wrong_output_shape,
    ):
        session = _Session()
        mutate(session)
        _install_fakes(monkeypatch, tmp_path, _OnnxRuntime(session))
        with pytest.raises(RuntimeError):
            runtime.OnnxRuntimeCpuFeatureConsumer.from_artifact(tmp_path)


def test_rejects_invalid_inputs_before_runtime_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _Session()
    _install_fakes(monkeypatch, tmp_path, _OnnxRuntime(session))
    consumer = runtime.OnnxRuntimeCpuFeatureConsumer.from_artifact(tmp_path)
    nonfinite = torch.zeros((1, 3, 256, 256), dtype=torch.float32)
    nonfinite[0, 0, 0, 0] = float("nan")
    infinite = torch.zeros((1, 3, 256, 256), dtype=torch.float32)
    infinite[0, 0, 0, 0] = float("inf")

    for images in (
        object(),
        torch.zeros((3, 256, 256), dtype=torch.float32),
        torch.zeros((2, 3, 256, 256), dtype=torch.float32),
        torch.zeros((1, 3, 256, 256), dtype=torch.float64),
        torch.empty((1, 3, 256, 256), dtype=torch.float32, device="meta"),
        nonfinite,
        infinite,
    ):
        with pytest.raises((TypeError, ValueError)):
            consumer.extract(images)  # type: ignore[arg-type]
    assert session.run_calls == []


def test_converts_inputs_and_returns_owned_contiguous_cpu_tensors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _Session()
    raw_layer2 = (
        np.arange(512 * 32 * 32, dtype=np.float32)
        .reshape(1, 512, 32, 32)
        .swapaxes(2, 3)
    )
    raw_patches = (
        np.arange(512 * 1024, dtype=np.float32).reshape(1, 512, 1024).transpose(0, 2, 1)
    )
    assert not raw_layer2.flags.c_contiguous
    assert not raw_patches.flags.c_contiguous
    expected_layer2 = raw_layer2.copy()
    expected_patches = raw_patches.copy()
    session.run_results = [raw_layer2, raw_patches]
    _install_fakes(monkeypatch, tmp_path, _OnnxRuntime(session))
    consumer = runtime.OnnxRuntimeCpuFeatureConsumer.from_artifact(tmp_path)

    images = (
        torch.linspace(
            -1.0,
            1.0,
            3 * 256 * 256,
            dtype=torch.float32,
        )
        .reshape(1, 3, 256, 256)
        .transpose(2, 3)
    )
    images.requires_grad_(True)
    original = images.detach().clone()
    outputs = consumer.extract(images)

    assert len(session.run_calls) == 1
    output_names, feed = session.run_calls[0]
    assert output_names == ["layer2", "patch_embeddings"]
    assert list(feed) == ["images"]
    assert session.input_snapshot is not None
    assert session.input_snapshot.dtype == np.float32
    assert session.input_snapshot.flags.c_contiguous
    assert np.array_equal(session.input_snapshot, original.contiguous().numpy())
    assert torch.equal(images, original)
    assert torch.equal(outputs.layer2, torch.from_numpy(expected_layer2))
    assert torch.equal(outputs.patch_embeddings, torch.from_numpy(expected_patches))
    for tensor, shape in (
        (outputs.layer2, (1, 512, 32, 32)),
        (outputs.patch_embeddings, (1, 1024, 512)),
    ):
        assert tensor.dtype == torch.float32
        assert tensor.device.type == "cpu"
        assert tensor.shape == shape
        assert tensor.is_contiguous()
        assert not tensor.requires_grad

    raw_layer2.fill(-1)
    raw_patches.fill(-1)
    assert torch.equal(outputs.layer2, torch.from_numpy(expected_layer2))
    assert torch.equal(outputs.patch_embeddings, torch.from_numpy(expected_patches))


def test_rejects_invalid_runtime_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = _Session()
    _install_fakes(monkeypatch, tmp_path, _OnnxRuntime(session))
    consumer = runtime.OnnxRuntimeCpuFeatureConsumer.from_artifact(tmp_path)
    images = torch.zeros((1, 3, 256, 256), dtype=torch.float32)
    layer2 = np.zeros((1, 512, 32, 32), dtype=np.float32)
    patches = np.zeros((1, 1024, 512), dtype=np.float32)
    nan_layer2 = layer2.copy()
    nan_layer2.flat[0] = np.nan
    infinite_patches = patches.copy()
    infinite_patches.flat[0] = np.inf

    for values in (
        [layer2],
        [object(), patches],
        [layer2.astype(np.float64), patches],
        [np.zeros((1, 512, 32, 31), dtype=np.float32), patches],
        [nan_layer2, patches],
        [layer2, infinite_patches],
    ):
        session.run_results = values
        with pytest.raises(RuntimeError):
            consumer.extract(images)
