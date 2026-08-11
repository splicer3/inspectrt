"""Direct ONNX Runtime CPU consumption of the fixed feature artifact."""

from dataclasses import dataclass
import importlib
from importlib import metadata as importlib_metadata
import json
from pathlib import Path
from typing import Any, Self

import numpy as np
import torch
from torch import Tensor

from inspectrt.onnx_artifacts import load_onnx_feature_artifact

_REQUESTED_PROVIDERS = ("CPUExecutionProvider",)
_EXPECTED_ONNXRUNTIME_VERSION = "1.28.0"
_INPUTS = (("images", "tensor(float)", (1, 3, 256, 256)),)
_OUTPUTS = (
    ("layer2", "tensor(float)", (1, 512, 32, 32)),
    ("patch_embeddings", "tensor(float)", (1, 1024, 512)),
)
_SESSION_OPTION_NAMES = (
    "execution_mode",
    "graph_optimization_level",
    "enable_cpu_mem_arena",
    "enable_mem_pattern",
    "enable_mem_reuse",
    "intra_op_num_threads",
    "inter_op_num_threads",
    "enable_profiling",
    "use_deterministic_compute",
    "log_severity_level",
    "log_verbosity_level",
    "optimized_model_filepath",
    "profile_file_prefix",
)


@dataclass(frozen=True, slots=True)
class OnnxRuntimeFeatureOutputs:
    """Validated independently owned CPU feature tensors."""

    layer2: Tensor
    patch_embeddings: Tensor


@dataclass(frozen=True, slots=True)
class OnnxRuntimeSessionMetadata:
    """Immutable identity and observations for one accepted CPU session."""

    artifact_id: str
    model_sha256: str
    onnxruntime_distribution_version: str
    onnxruntime_version: str
    onnxruntime_version_string: str
    onnxruntime_build_info: str
    available_providers: tuple[str, ...]
    requested_providers: tuple[str, ...]
    active_providers: tuple[str, ...]
    provider_options: tuple[tuple[str, tuple[tuple[str, str], ...]], ...]
    fallback_disabled: bool
    session_options: tuple[tuple[str, str | int | bool], ...]
    inputs: tuple[tuple[str, str, tuple[int, ...]], ...]
    outputs: tuple[tuple[str, str, tuple[int, ...]], ...]


class OnnxRuntimeCpuFeatureConsumer:
    """Run the fixed batch-1 feature artifact through ORT CPU."""

    __slots__ = ("_metadata", "_session")

    def __init__(self, artifact_directory: Path) -> None:
        artifact = load_onnx_feature_artifact(artifact_directory)
        expected_runtime_version = _artifact_runtime_version(artifact_directory)
        distribution_version = _runtime_distribution_version(expected_runtime_version)
        onnxruntime = _import_onnxruntime()

        available_providers = tuple(onnxruntime.get_available_providers())
        if "CPUExecutionProvider" not in available_providers:
            raise RuntimeError("ONNX Runtime CPUExecutionProvider is unavailable")

        options = onnxruntime.SessionOptions()
        session = onnxruntime.InferenceSession(
            artifact_directory / "model.onnx",
            sess_options=options,
            providers=list(_REQUESTED_PROVIDERS),
        )
        session.disable_fallback()

        active_providers = tuple(session.get_providers())
        if active_providers != _REQUESTED_PROVIDERS:
            raise RuntimeError(
                "ONNX Runtime active providers must be exactly CPUExecutionProvider"
            )

        inputs = _node_metadata(session.get_inputs())
        outputs = _node_metadata(session.get_outputs())
        if inputs != _INPUTS:
            raise RuntimeError("ONNX Runtime session input metadata is invalid")
        if outputs != _OUTPUTS:
            raise RuntimeError("ONNX Runtime session output metadata is invalid")

        self._session = session
        self._metadata = OnnxRuntimeSessionMetadata(
            artifact_id=artifact.artifact_id,
            model_sha256=artifact.model_sha256,
            onnxruntime_distribution_version=distribution_version,
            onnxruntime_version=onnxruntime.__version__,
            onnxruntime_version_string=onnxruntime.get_version_string(),
            onnxruntime_build_info=onnxruntime.get_build_info(),
            available_providers=available_providers,
            requested_providers=_REQUESTED_PROVIDERS,
            active_providers=active_providers,
            provider_options=_provider_options(session.get_provider_options()),
            fallback_disabled=True,
            session_options=_session_options(session.get_session_options()),
            inputs=inputs,
            outputs=outputs,
        )

    @classmethod
    def from_artifact(cls, artifact_directory: Path) -> Self:
        """Construct a CPU consumer from one strict schema-1 artifact directory."""
        return cls(artifact_directory)

    @property
    def metadata(self) -> OnnxRuntimeSessionMetadata:
        return self._metadata

    def extract(self, images: Tensor) -> OnnxRuntimeFeatureOutputs:
        """Run and validate the fixed batch-1 FP32 feature operation."""
        _validate_images(images)
        input_numpy = np.array(
            images.detach().contiguous().numpy(),
            dtype=np.float32,
            order="C",
            copy=True,
        )
        raw_outputs = self._session.run(
            ["layer2", "patch_embeddings"],
            {"images": input_numpy},
        )
        if not isinstance(raw_outputs, list) or len(raw_outputs) != 2:
            raise RuntimeError("ONNX Runtime must return exactly two outputs")
        layer2 = _output_tensor(raw_outputs[0], "layer2", (1, 512, 32, 32))
        patches = _output_tensor(raw_outputs[1], "patch_embeddings", (1, 1024, 512))
        return OnnxRuntimeFeatureOutputs(layer2, patches)


def _artifact_runtime_version(artifact_directory: Path) -> str:
    manifest = json.loads((artifact_directory / "manifest.json").read_bytes())
    version = manifest["environment"]["dependency_versions"]["onnxruntime"]
    if type(version) is not str:
        raise RuntimeError("validated artifact has invalid ONNX Runtime metadata")
    return version


def _runtime_distribution_version(expected: str) -> str:
    if expected != _EXPECTED_ONNXRUNTIME_VERSION:
        raise RuntimeError(
            "artifact requires unsupported onnxruntime distribution version: "
            f"{expected}; expected={_EXPECTED_ONNXRUNTIME_VERSION}"
        )
    try:
        gpu_version = importlib_metadata.version("onnxruntime-gpu")
    except importlib_metadata.PackageNotFoundError:
        pass
    else:
        raise RuntimeError(
            f"onnxruntime-gpu {gpu_version} must not be installed for ORT CPU"
        )
    try:
        installed = importlib_metadata.version("onnxruntime")
    except importlib_metadata.PackageNotFoundError as error:
        raise RuntimeError(
            "install inspectrt[onnx] to use the ONNX Runtime CPU feature consumer"
        ) from error
    if installed != expected:
        raise RuntimeError(
            "onnxruntime distribution version does not match the artifact manifest: "
            f"installed={installed}, artifact={expected}"
        )
    return installed


def _import_onnxruntime() -> Any:
    try:
        return importlib.import_module("onnxruntime")
    except (ImportError, ModuleNotFoundError) as error:
        raise RuntimeError(
            "install inspectrt[onnx] to use the ONNX Runtime CPU feature consumer"
        ) from error


def _node_metadata(
    nodes: list[Any],
) -> tuple[tuple[str, str, tuple[int, ...]], ...]:
    records = []
    for node in nodes:
        if (
            type(node.name) is not str
            or type(node.type) is not str
            or type(node.shape) is not list
            or any(type(dimension) is not int for dimension in node.shape)
        ):
            raise RuntimeError("ONNX Runtime NodeArg metadata is invalid")
        records.append((node.name, node.type, tuple(node.shape)))
    return tuple(records)


def _provider_options(
    options: object,
) -> tuple[tuple[str, tuple[tuple[str, str], ...]], ...]:
    if type(options) is not dict:
        raise RuntimeError("ONNX Runtime provider options metadata is invalid")
    records = []
    for provider, values in sorted(options.items()):
        if type(provider) is not str or type(values) is not dict:
            raise RuntimeError("ONNX Runtime provider options metadata is invalid")
        if any(
            type(key) is not str or type(value) is not str
            for key, value in values.items()
        ):
            raise RuntimeError("ONNX Runtime provider options metadata is invalid")
        records.append((provider, tuple(sorted(values.items()))))
    return tuple(records)


def _session_options(
    options: object,
) -> tuple[tuple[str, str | int | bool], ...]:
    return tuple(
        (name, _session_option_value(getattr(options, name)))
        for name in _SESSION_OPTION_NAMES
    )


def _session_option_value(value: object) -> str | int | bool:
    enum_name = getattr(value, "name", None)
    if type(enum_name) is str:
        return enum_name
    if type(value) in (str, int, bool):
        return value
    raise RuntimeError("ONNX Runtime session option metadata is invalid")


def _validate_images(images: Tensor) -> None:
    if not isinstance(images, Tensor):
        raise TypeError("images must be a torch.Tensor")
    if tuple(images.shape) != (1, 3, 256, 256):
        raise ValueError("images must have shape [1, 3, 256, 256]")
    if images.dtype != torch.float32:
        raise TypeError("images must have dtype torch.float32")
    if images.device.type != "cpu":
        raise ValueError("images must be on the CPU")
    if not bool(torch.isfinite(images).all().item()):
        raise ValueError("images must contain only finite values")


def _output_tensor(value: object, name: str, shape: tuple[int, ...]) -> Tensor:
    if not isinstance(value, np.ndarray):
        raise RuntimeError(f"ONNX Runtime output {name} must be a numpy.ndarray")
    if value.dtype != np.float32:
        raise RuntimeError(f"ONNX Runtime output {name} must have dtype float32")
    if value.shape != shape:
        raise RuntimeError(f"ONNX Runtime output {name} has an invalid shape")
    if not bool(np.isfinite(value).all()):
        raise RuntimeError(f"ONNX Runtime output {name} contains non-finite values")
    return torch.from_numpy(np.array(value, dtype=np.float32, order="C", copy=True))
