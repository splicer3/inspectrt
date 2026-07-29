"""Strict loading of frozen schema-1 baseline run bundles."""

from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
from types import MappingProxyType
from typing import BinaryIO, Literal

import torch
from torch import Tensor

from inspectrt.artifacts import _canonical_json
from inspectrt.benchmark import _methodology
from inspectrt.data import MvtecSample
from inspectrt.metrics import (
    ThresholdFreeMetrics,
    compute_threshold_free_metrics,
)

__all__ = (
    "BundleMetrics",
    "BundleValidationError",
    "ComparableBundle",
    "MemoryBankMetadata",
    "PredictionRecord",
    "SourceFileSnapshot",
    "load_comparable_bundle",
)

_EVALUATION_FILES = (
    "run.json",
    "samples.jsonl",
    "memory_bank.pt",
    "predictions.jsonl",
    "retrieval.pt",
    "anomaly_maps.pt",
    "metrics.json",
)
_BENCHMARK_FILES = (*_EVALUATION_FILES, "benchmark.json")
_RUN_FIELDS = {
    "bank_chunk_size",
    "batch_size",
    "benchmark",
    "category",
    "dataset_root",
    "determinism",
    "device",
    "environment",
    "feature_extractor",
    "feature_layer",
    "inventory",
    "map_interpolation",
    "preprocessing_profile",
    "profile_id",
    "retrieval_semantics",
    "run_id",
    "schema_version",
    "source",
    "tensors",
    "weights",
}
_SAMPLE_FIELDS = {
    "category",
    "defect_type",
    "image_relpath",
    "is_anomalous",
    "mask_relpath",
    "sample_id",
    "split",
}
_PREDICTION_FIELDS = {
    "defect_type",
    "image_label",
    "image_score",
    "sample_id",
    "tensor_index",
}
_METRIC_FIELDS = {
    "anomalous_pixel_count",
    "anomalous_test_sample_count",
    "evaluated_pixel_count",
    "image_auroc",
    "image_average_precision",
    "pixel_auroc",
    "test_good_sample_count",
    "test_sample_count",
    "training_sample_count",
}
_BENCHMARK_FIELDS = {
    "benchmark_sample_id",
    "category",
    "created_at_utc",
    "device",
    "environment",
    "methodology",
    "profile_id",
    "results",
    "run_id",
    "schema_version",
    "workload",
}
_DEPENDENCIES = {
    "inspectrt",
    "numpy",
    "pillow",
    "scikit-learn",
    "torch",
    "torchvision",
}
_DETERMINISM = {
    "allow_tf32": False,
    "cublas_workspace_config": ":4096:8",
    "cudnn_benchmark": False,
    "deterministic_algorithms_warn_only": False,
    "fp32_precision": "ieee",
    "numpy_seed": 0,
    "python_random_seed": 0,
    "torch_cpu_seed": 0,
    "use_deterministic_algorithms": True,
}
_TENSOR_NAMES = {
    "anomaly_maps",
    "evaluation_masks",
    "image_scores",
    "memory_bank",
    "nearest_bank_indices",
    "patch_distances",
    "test_labels",
}
_REPEATED_STAGES = {
    "anomaly_map_reconstruction",
    "canonical_image_preprocessing",
    "exact_chunked_retrieval",
    "frozen_feature_extraction",
    "host_to_device_transfer",
    "image_decode",
}
_SUMMARY_FIELDS = {"count", "maximum", "mean", "minimum", "p50", "p95"}
_RUN_ID = re.compile(r"[A-Za-z0-9._-]+")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-fA-F]{64}")
_PATCH_COUNT = 1024
_EMBEDDING_DIMENSION = 512
_MAP_SIZE = (256, 256)
_WEIGHT_ENUM = "ResNet50_Weights.IMAGENET1K_V2"
_WEIGHT_URL = "https://download.pytorch.org/models/resnet50-11ad3fa6.pth"


class BundleValidationError(ValueError):
    """A comparable bundle violates the frozen schema-1 contract."""


@dataclass(frozen=True, slots=True)
class SourceFileSnapshot:
    """Current identity of one source artifact."""

    name: str
    byte_count: int
    sha256: str


@dataclass(frozen=True, slots=True)
class MemoryBankMetadata:
    """Persisted memory-bank storage metadata."""

    dtype: str
    shape: tuple[int, int]
    embedding_dimension: int
    patches_per_training_sample: int


@dataclass(frozen=True, slots=True)
class PredictionRecord:
    """One ordered persisted test prediction."""

    sample_id: str
    defect_type: str
    image_label: int
    image_score: float
    tensor_index: int


@dataclass(frozen=True, slots=True)
class BundleMetrics:
    """Stored and recomputed threshold-free metrics and counts."""

    image_auroc: float
    image_average_precision: float
    pixel_auroc: float
    training_sample_count: int
    test_sample_count: int
    test_good_sample_count: int
    anomalous_test_sample_count: int
    evaluated_pixel_count: int
    anomalous_pixel_count: int


@dataclass(frozen=True, slots=True)
class ComparableBundle:
    """One internally consistent evaluation or benchmark bundle.

    The record containers are immutable. Its tensors remain ordinary mutable
    PyTorch tensors and are not copied merely to simulate deep immutability.
    """

    path: Path
    kind: Literal["evaluation", "benchmark"]
    run_metadata: Mapping[str, object]
    benchmark_metadata: Mapping[str, object] | None
    source_files: tuple[SourceFileSnapshot, ...]
    samples: tuple[MvtecSample, ...]
    predictions: tuple[PredictionRecord, ...]
    test_sample_ids: tuple[str, ...]
    test_labels: Tensor
    image_scores: Tensor
    memory_bank_metadata: MemoryBankMetadata
    memory_bank: Tensor
    patch_distances: Tensor
    nearest_bank_indices: Tensor
    anomaly_maps: Tensor
    evaluation_masks: Tensor
    metrics: BundleMetrics


def load_comparable_bundle(bundle_path: Path) -> ComparableBundle:
    """Load and completely validate one frozen schema-1 run bundle."""
    if not isinstance(bundle_path, Path):
        raise TypeError("bundle_path must be a pathlib.Path")
    path = _validate_bundle_directory(bundle_path)
    kind, names = _classify_bundle(path)
    source_files = _snapshot_sources(path, names)
    snapshots = {snapshot.name: snapshot for snapshot in source_files}

    run = _parse_json(
        _read_regular(path / "run.json", "run.json", snapshots["run.json"]),
        "run.json",
    )
    sample_bytes = _read_regular(
        path / "samples.jsonl", "samples.jsonl", snapshots["samples.jsonl"]
    )
    sample_values = _parse_json_lines(sample_bytes, "samples.jsonl")
    prediction_values = _parse_json_lines(
        _read_regular(
            path / "predictions.jsonl",
            "predictions.jsonl",
            snapshots["predictions.jsonl"],
        ),
        "predictions.jsonl",
    )
    metric_values = _parse_json(
        _read_regular(path / "metrics.json", "metrics.json", snapshots["metrics.json"]),
        "metrics.json",
    )
    bank_payload = _load_tensor_payload(
        path / "memory_bank.pt", snapshots["memory_bank.pt"]
    )
    retrieval_payload = _load_tensor_payload(
        path / "retrieval.pt", snapshots["retrieval.pt"]
    )
    maps_payload = _load_tensor_payload(
        path / "anomaly_maps.pt", snapshots["anomaly_maps.pt"]
    )
    benchmark = (
        _parse_json(
            _read_regular(
                path / "benchmark.json",
                "benchmark.json",
                snapshots["benchmark.json"],
            ),
            "benchmark.json",
        )
        if kind == "benchmark"
        else None
    )

    device = _validate_run(run, path.name, kind)
    samples, test_samples = _validate_samples(sample_values, sample_bytes, run)
    test_ids = tuple(sample.sample_id for sample in test_samples)
    predictions, labels = _validate_predictions(prediction_values, test_samples)
    bank_metadata, memory_bank = _validate_memory_bank(bank_payload, run)
    patch_distances, nearest_indices = _validate_retrieval(
        retrieval_payload, test_ids, memory_bank.shape[0]
    )
    scores = _validate_prediction_scores(predictions, patch_distances)
    anomaly_maps, evaluation_masks = _validate_maps(
        maps_payload, test_samples, test_ids
    )
    metrics = _validate_metrics(
        metric_values,
        labels,
        scores,
        evaluation_masks,
        anomaly_maps,
        run,
    )
    if benchmark is not None:
        _validate_benchmark(
            benchmark,
            run,
            device,
            test_ids,
            bank_metadata,
        )

    final_kind, final_names = _classify_bundle(path)
    if final_kind != kind or final_names != names:
        raise BundleValidationError("bundle file inventory changed during loading")
    final_sources = _snapshot_sources(path, names)
    if source_files != final_sources:
        changed = next(
            (
                before.name
                for before, after in zip(source_files, final_sources, strict=True)
                if before != after
            ),
            "inventory",
        )
        raise BundleValidationError(f"{changed}: source file changed during loading")

    return ComparableBundle(
        path=path,
        kind=kind,
        run_metadata=_freeze_json(run),
        benchmark_metadata=(_freeze_json(benchmark) if benchmark is not None else None),
        source_files=source_files,
        samples=samples,
        predictions=predictions,
        test_sample_ids=test_ids,
        test_labels=labels,
        image_scores=scores,
        memory_bank_metadata=bank_metadata,
        memory_bank=memory_bank,
        patch_distances=patch_distances,
        nearest_bank_indices=nearest_indices,
        anomaly_maps=anomaly_maps,
        evaluation_masks=evaluation_masks,
        metrics=metrics,
    )


def _validate_bundle_directory(path: Path) -> Path:
    try:
        metadata = path.lstat()
    except FileNotFoundError as error:
        raise BundleValidationError(
            f"bundle directory does not exist: {path}"
        ) from error
    except OSError as error:
        raise BundleValidationError(
            f"bundle directory cannot be inspected: {path}: {error.strerror}"
        ) from error
    if stat.S_ISLNK(metadata.st_mode):
        raise BundleValidationError(f"bundle directory must not be a symlink: {path}")
    if not stat.S_ISDIR(metadata.st_mode):
        raise BundleValidationError(f"bundle path must be a directory: {path}")
    try:
        return path.resolve(strict=True)
    except OSError as error:
        raise BundleValidationError(
            f"bundle directory cannot be resolved safely: {path}"
        ) from error


def _classify_bundle(
    path: Path,
) -> tuple[Literal["evaluation", "benchmark"], tuple[str, ...]]:
    _validate_bundle_directory(path)
    try:
        with os.scandir(path) as entries:
            names = {entry.name for entry in entries}
    except OSError as error:
        raise BundleValidationError(
            f"bundle directory cannot be listed: {path}: {error.strerror}"
        ) from error
    evaluation = set(_EVALUATION_FILES)
    benchmark = set(_BENCHMARK_FILES)
    if names == evaluation:
        kind: Literal["evaluation", "benchmark"] = "evaluation"
        ordered = _EVALUATION_FILES
    elif names == benchmark:
        kind = "benchmark"
        ordered = _BENCHMARK_FILES
    else:
        expected = benchmark if "benchmark.json" in names else evaluation
        raise BundleValidationError(
            "bundle file set is invalid; "
            f"missing={sorted(expected - names)}, "
            f"unexpected={sorted(names - expected)}"
        )
    return kind, ordered


def _snapshot_sources(
    bundle_path: Path, names: tuple[str, ...]
) -> tuple[SourceFileSnapshot, ...]:
    snapshots = []
    for name in names:
        digest = hashlib.sha256()
        byte_count = 0
        with _open_regular(bundle_path / name, name) as (stream, before):
            try:
                while chunk := stream.read(1024 * 1024):
                    digest.update(chunk)
                    byte_count += len(chunk)
                after = os.fstat(stream.fileno())
            except OSError as error:
                raise BundleValidationError(
                    f"{name}: artifact cannot be hashed: {error.strerror}"
                ) from error
        if _stat_identity(before) != _stat_identity(after):
            raise BundleValidationError(f"{name}: artifact changed while hashing")
        if byte_count != after.st_size:
            raise BundleValidationError(f"{name}: artifact size changed while hashing")
        snapshots.append(SourceFileSnapshot(name, byte_count, digest.hexdigest()))
    return tuple(snapshots)


@contextmanager
def _open_regular(path: Path, name: str) -> tuple[BinaryIO, os.stat_result]:
    flags = os.O_RDONLY
    for option in ("O_CLOEXEC", "O_NOFOLLOW", "O_NONBLOCK"):
        flags |= getattr(os, option, 0)
    try:
        if path.is_symlink():
            raise BundleValidationError(f"{name}: artifact must not be a symlink")
        descriptor = os.open(path, flags)
    except BundleValidationError:
        raise
    except OSError as error:
        raise BundleValidationError(
            f"{name}: artifact cannot be opened safely: {error.strerror}"
        ) from error
    try:
        try:
            metadata = os.fstat(descriptor)
        except OSError as error:
            raise BundleValidationError(
                f"{name}: artifact cannot be inspected: {error.strerror}"
            ) from error
        if not stat.S_ISREG(metadata.st_mode):
            raise BundleValidationError(f"{name}: artifact must be a regular file")
        with os.fdopen(descriptor, "rb", closefd=True) as stream:
            descriptor = -1
            yield stream, metadata
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _stat_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _read_regular(path: Path, name: str, expected: SourceFileSnapshot) -> bytes:
    with _open_regular(path, name) as (stream, before):
        try:
            payload = stream.read()
            after = os.fstat(stream.fileno())
        except OSError as error:
            raise BundleValidationError(
                f"{name}: artifact cannot be read: {error.strerror}"
            ) from error
    if _stat_identity(before) != _stat_identity(after) or len(payload) != after.st_size:
        raise BundleValidationError(f"{name}: artifact changed while reading")
    _match_snapshot(expected, len(payload), hashlib.sha256(payload).hexdigest())
    return payload


def _load_tensor_payload(path: Path, expected: SourceFileSnapshot) -> object:
    with _open_regular(path, path.name) as (stream, before):
        digest = hashlib.sha256()
        byte_count = 0
        try:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
                byte_count += len(chunk)
            after_hash = os.fstat(stream.fileno())
        except OSError as error:
            raise BundleValidationError(
                f"{path.name}: artifact cannot be hashed: {error.strerror}"
            ) from error
        if (
            _stat_identity(before) != _stat_identity(after_hash)
            or byte_count != after_hash.st_size
        ):
            raise BundleValidationError(f"{path.name}: artifact changed while hashing")
        _match_snapshot(expected, byte_count, digest.hexdigest())
        try:
            stream.seek(0)
            payload = torch.load(stream, map_location="cpu", weights_only=True)
        except Exception as error:
            raise BundleValidationError(
                f"{path.name}: safe tensor loading failed ({type(error).__name__})"
            ) from error
        try:
            after = os.fstat(stream.fileno())
        except OSError as error:
            raise BundleValidationError(
                f"{path.name}: artifact cannot be inspected after loading"
            ) from error
    if _stat_identity(before) != _stat_identity(after):
        raise BundleValidationError(f"{path.name}: artifact changed while loading")
    return payload


def _match_snapshot(expected: SourceFileSnapshot, byte_count: int, sha256: str) -> None:
    if byte_count != expected.byte_count or sha256 != expected.sha256:
        raise BundleValidationError(
            f"{expected.name}: source file changed during loading"
        )


def _parse_json(payload: bytes, name: str) -> dict[str, object]:
    def pairs(values: list[tuple[str, object]]) -> dict[str, object]:
        result = {}
        for key, value in values:
            if key in result:
                raise BundleValidationError(
                    f"{name}: duplicate JSON object key {key!r}"
                )
            result[key] = value
        return result

    def constant(value: str) -> object:
        raise BundleValidationError(f"{name}: non-finite JSON number {value}")

    try:
        parsed = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=constant,
        )
    except BundleValidationError:
        raise
    except UnicodeDecodeError as error:
        raise BundleValidationError(f"{name}: invalid UTF-8 JSON") from error
    except (json.JSONDecodeError, RecursionError) as error:
        raise BundleValidationError(f"{name}: malformed JSON") from error
    if type(parsed) is not dict:
        raise BundleValidationError(f"{name}: JSON root must be an object")
    try:
        canonical = _canonical_json(parsed)
    except (TypeError, ValueError, OverflowError, RecursionError) as error:
        raise BundleValidationError(f"{name}: JSON values are invalid") from error
    if canonical != payload:
        raise BundleValidationError(f"{name}: JSON bytes are not canonical")
    return parsed


def _parse_json_lines(payload: bytes, name: str) -> list[dict[str, object]]:
    if not payload or not payload.endswith(b"\n"):
        raise BundleValidationError(f"{name}: expected nonempty LF-terminated JSONL")
    return [
        _parse_json(line, f"{name} record {index}")
        for index, line in enumerate(payload.splitlines(keepends=True), start=1)
    ]


def _validate_run(
    run: dict[str, object],
    directory_name: str,
    kind: Literal["evaluation", "benchmark"],
) -> torch.device:
    _keys(run, _RUN_FIELDS, "run.json")
    _equals(run["schema_version"], 1, "run.json.schema_version")
    _equals(run["profile_id"], "inspectrt_feature_memory_v1", "run.json.profile_id")
    _equals(
        run["preprocessing_profile"],
        "inspectrt_resize256_v1",
        "run.json.preprocessing_profile",
    )
    _equals(run["feature_extractor"], "ResNet-50", "run.json.feature_extractor")
    _equals(run["feature_layer"], "layer2", "run.json.feature_layer")
    _equals(
        run["retrieval_semantics"],
        "exact top-1 squared L2",
        "run.json.retrieval_semantics",
    )
    _equals(run["batch_size"], 1, "run.json.batch_size")
    _positive_integer(run["bank_chunk_size"], "run.json.bank_chunk_size")
    category = _component(run["category"], "run.json.category")
    _string(run["dataset_root"], "run.json.dataset_root")
    run_id = _component(run["run_id"], "run.json.run_id")
    if not _RUN_ID.fullmatch(run_id):
        raise BundleValidationError("run.json.run_id has invalid characters")
    if run_id != directory_name:
        raise BundleValidationError("run.json.run_id must match the bundle directory")

    device_value = _string(run["device"], "run.json.device")
    try:
        device = torch.device(device_value)
    except (RuntimeError, ValueError) as error:
        raise BundleValidationError("run.json.device is not a valid device") from error
    if str(device) != device_value:
        raise BundleValidationError("run.json.device is not canonical")
    if device.type == "cuda" and device.index is None:
        raise BundleValidationError("run.json.device must index a CUDA device")

    source = _object(run["source"], "run.json.source")
    _keys(source, {"dirty", "git_commit", "uv_lock_sha256"}, "run.json.source")
    _boolean(source["dirty"], "run.json.source.dirty")
    _commit(source["git_commit"], "run.json.source.git_commit")
    _sha256(source["uv_lock_sha256"], "run.json.source.uv_lock_sha256")

    environment = _object(run["environment"], "run.json.environment")
    _keys(
        environment,
        {
            "created_at_utc",
            "dependency_versions",
            "platform_description",
            "python_version",
        },
        "run.json.environment",
    )
    _utc_timestamp(environment["created_at_utc"], "run.json.environment.created_at_utc")
    _string(
        environment["platform_description"],
        "run.json.environment.platform_description",
    )
    _string(environment["python_version"], "run.json.environment.python_version")
    dependencies = _object(
        environment["dependency_versions"],
        "run.json.environment.dependency_versions",
    )
    _keys(
        dependencies,
        _DEPENDENCIES,
        "run.json.environment.dependency_versions",
    )
    for name, value in dependencies.items():
        _string(value, f"run.json.environment.dependency_versions.{name}")

    determinism = _object(run["determinism"], "run.json.determinism")
    _keys(
        determinism,
        {*_DETERMINISM, "torch_cuda_seed_all"},
        "run.json.determinism",
    )
    seed_names = ("numpy_seed", "python_random_seed", "torch_cpu_seed")
    for name, expected in _DETERMINISM.items():
        if name in seed_names:
            continue
        _equals(determinism[name], expected, f"run.json.determinism.{name}")
    seeds = {
        _integer(determinism[name], f"run.json.determinism.{name}")
        for name in seed_names
    }
    if len(seeds) != 1:
        raise BundleValidationError("run.json determinism seeds are inconsistent")
    seed = seeds.pop()
    cuda_seed = determinism["torch_cuda_seed_all"]
    if device.type == "cuda" and cuda_seed is None:
        raise BundleValidationError(
            "run.json.determinism.torch_cuda_seed_all is required on CUDA"
        )
    if cuda_seed is not None:
        _equals(
            cuda_seed,
            seed,
            "run.json.determinism.torch_cuda_seed_all",
        )

    weights = _object(run["weights"], "run.json.weights")
    _keys(
        weights,
        {"cached_file_sha256", "enum", "source_url"},
        "run.json.weights",
    )
    _sha256(weights["cached_file_sha256"], "run.json.weights.cached_file_sha256")
    _equals(weights["enum"], _WEIGHT_ENUM, "run.json.weights.enum")
    _equals(weights["source_url"], _WEIGHT_URL, "run.json.weights.source_url")

    interpolation = _object(run["map_interpolation"], "run.json.map_interpolation")
    expected_interpolation = {
        "align_corners": False,
        "input_size": [32, 32],
        "mode": "bilinear",
        "output_size": [256, 256],
        "values": "raw squared-L2 patch distances",
    }
    if _canonical_json(interpolation) != _canonical_json(expected_interpolation):
        raise BundleValidationError("run.json.map_interpolation is invalid")

    inventory = _object(run["inventory"], "run.json.inventory")
    inventory_fields = {
        "anomalous_test_sample_count",
        "sample_inventory_sha256",
        "test_good_sample_count",
        "test_sample_count",
        "total_sample_count",
        "training_sample_count",
    }
    _keys(inventory, inventory_fields, "run.json.inventory")
    _sha256(
        inventory["sample_inventory_sha256"],
        "run.json.inventory.sample_inventory_sha256",
    )
    counts = {
        name: _positive_integer(inventory[name], f"run.json.inventory.{name}")
        for name in sorted(inventory_fields - {"sample_inventory_sha256"})
    }
    if counts["test_sample_count"] != (
        counts["test_good_sample_count"] + counts["anomalous_test_sample_count"]
    ):
        raise BundleValidationError("run.json inventory test counts are inconsistent")
    if counts["total_sample_count"] != (
        counts["training_sample_count"] + counts["test_sample_count"]
    ):
        raise BundleValidationError("run.json inventory total count is inconsistent")

    _validate_run_tensors(run, counts)
    declaration = run["benchmark"]
    if kind == "evaluation":
        if declaration is not None:
            raise BundleValidationError(
                "run.json must declare no benchmark for an evaluation bundle"
            )
    else:
        benchmark = _object(declaration, "run.json.benchmark")
        _keys(
            benchmark,
            {"artifact", "schema_version", "timing_device"},
            "run.json.benchmark",
        )
        _equals(benchmark["artifact"], "benchmark.json", "run.json.benchmark.artifact")
        _equals(benchmark["schema_version"], 1, "run.json.benchmark.schema_version")
        _equals(
            benchmark["timing_device"],
            device_value,
            "run.json.benchmark.timing_device",
        )
    _component(category, "run.json.category")
    return device


def _validate_run_tensors(run: dict[str, object], counts: Mapping[str, int]) -> None:
    tensors = _object(run["tensors"], "run.json.tensors")
    _keys(tensors, _TENSOR_NAMES, "run.json.tensors")
    training_count = counts["training_sample_count"]
    test_count = counts["test_sample_count"]
    bank_rows = training_count * _PATCH_COUNT
    contracts = {
        "anomaly_maps": ("float32", [test_count, *_MAP_SIZE]),
        "evaluation_masks": ("uint8", [test_count, *_MAP_SIZE]),
        "image_scores": ("float32", [test_count]),
        "nearest_bank_indices": ("int64", [test_count, _PATCH_COUNT]),
        "patch_distances": ("float32", [test_count, _PATCH_COUNT]),
        "test_labels": ("uint8", [test_count]),
    }
    for name, (dtype, shape) in contracts.items():
        _tensor_declaration(tensors[name], name, dtype, shape)
    bank = _object(tensors["memory_bank"], "run.json.tensors.memory_bank")
    _keys(
        bank,
        {"byte_count", "dtype", "shape"},
        "run.json.tensors.memory_bank",
    )
    _equals(bank["dtype"], "float32", "run.json.tensors.memory_bank.dtype")
    _shape(
        bank["shape"],
        [bank_rows, _EMBEDDING_DIMENSION],
        "run.json.tensors.memory_bank.shape",
    )
    _equals(
        bank["byte_count"],
        bank_rows * _EMBEDDING_DIMENSION * 4,
        "run.json.tensors.memory_bank.byte_count",
    )


def _tensor_declaration(value: object, name: str, dtype: str, shape: list[int]) -> None:
    record = _object(value, f"run.json.tensors.{name}")
    _keys(record, {"dtype", "shape"}, f"run.json.tensors.{name}")
    _equals(record["dtype"], dtype, f"run.json.tensors.{name}.dtype")
    _shape(record["shape"], shape, f"run.json.tensors.{name}.shape")


def _validate_samples(
    values: list[dict[str, object]], payload: bytes, run: dict[str, object]
) -> tuple[tuple[MvtecSample, ...], tuple[MvtecSample, ...]]:
    category = str(run["category"])
    samples = []
    ids = []
    image_paths = []
    mask_paths = []
    for index, value in enumerate(values):
        name = f"samples.jsonl record {index + 1}"
        _keys(value, _SAMPLE_FIELDS, name)
        sample_id = _relative_posix(value["sample_id"], f"{name}.sample_id")
        sample_category = _component(value["category"], f"{name}.category")
        defect = _component(value["defect_type"], f"{name}.defect_type")
        split = _string(value["split"], f"{name}.split")
        image_path = _relative_posix(value["image_relpath"], f"{name}.image_relpath")
        anomalous = _boolean(value["is_anomalous"], f"{name}.is_anomalous")
        mask_value = value["mask_relpath"]
        mask_path = (
            None
            if mask_value is None
            else _relative_posix(mask_value, f"{name}.mask_relpath")
        )
        if sample_category != category:
            raise BundleValidationError(f"{name}: category differs from run.json")
        if split not in {"train", "test"}:
            raise BundleValidationError(f"{name}: split must be 'train' or 'test'")
        parts = image_path.split("/")
        if (
            len(parts) != 4
            or parts[:3] != [category, split, defect]
            or not parts[3].endswith(".png")
            or parts[3] == ".png"
        ):
            raise BundleValidationError(f"{name}: image_relpath is inconsistent")
        if sample_id != f"mvtec_ad/{image_path}":
            raise BundleValidationError(f"{name}: sample_id is inconsistent")
        expected_anomaly = split == "test" and defect != "good"
        if anomalous is not expected_anomaly:
            raise BundleValidationError(f"{name}: anomaly fields are inconsistent")
        if split == "train" and defect != "good":
            raise BundleValidationError(f"{name}: training samples must be good")
        expected_mask = (
            f"{category}/ground_truth/{defect}/{parts[3][:-4]}_mask.png"
            if expected_anomaly
            else None
        )
        if mask_path != expected_mask:
            raise BundleValidationError(f"{name}: mask_relpath is inconsistent")
        samples.append(
            MvtecSample(
                sample_id,
                sample_category,
                split,
                defect,
                anomalous,
                image_path,
                mask_path,
            )
        )
        ids.append(sample_id)
        image_paths.append(image_path)
        if mask_path is not None:
            mask_paths.append(mask_path)

    if ids != sorted(ids):
        raise BundleValidationError("samples.jsonl records are not canonically ordered")
    if len(ids) != len(set(ids)):
        raise BundleValidationError("samples.jsonl sample IDs must be unique")
    if len(image_paths) != len(set(image_paths)):
        raise BundleValidationError("samples.jsonl image paths must be unique")
    if len(mask_paths) != len(set(mask_paths)):
        raise BundleValidationError("samples.jsonl mask paths must be unique")

    inventory = _object(run["inventory"], "run.json.inventory")
    test_samples = tuple(sample for sample in samples if sample.split == "test")
    training = tuple(sample for sample in samples if sample.split == "train")
    anomalous_count = sum(sample.is_anomalous for sample in test_samples)
    actual = {
        "anomalous_test_sample_count": anomalous_count,
        "test_good_sample_count": len(test_samples) - anomalous_count,
        "test_sample_count": len(test_samples),
        "total_sample_count": len(samples),
        "training_sample_count": len(training),
    }
    for name, count in actual.items():
        if inventory[name] != count:
            raise BundleValidationError(
                f"run.json.inventory.{name} differs from samples.jsonl"
            )
    digest = hashlib.sha256(payload).hexdigest()
    if inventory["sample_inventory_sha256"] != digest:
        raise BundleValidationError(
            "run.json inventory SHA-256 differs from samples.jsonl"
        )
    return tuple(samples), test_samples


def _validate_predictions(
    values: list[dict[str, object]],
    test_samples: tuple[MvtecSample, ...],
) -> tuple[tuple[PredictionRecord, ...], Tensor]:
    if len(values) != len(test_samples):
        raise BundleValidationError(
            "predictions.jsonl record count differs from test samples"
        )
    predictions = []
    for index, (value, sample) in enumerate(zip(values, test_samples, strict=True)):
        name = f"predictions.jsonl record {index + 1}"
        _keys(value, _PREDICTION_FIELDS, name)
        sample_id = _string(value["sample_id"], f"{name}.sample_id")
        defect = _string(value["defect_type"], f"{name}.defect_type")
        label = _integer(value["image_label"], f"{name}.image_label")
        score = _float(value["image_score"], f"{name}.image_score")
        tensor_index = _integer(value["tensor_index"], f"{name}.tensor_index")
        if sample_id != sample.sample_id:
            raise BundleValidationError(f"{name}: sample ID is out of order")
        if defect != sample.defect_type:
            raise BundleValidationError(f"{name}: defect type differs from sample")
        if label != int(sample.is_anomalous):
            raise BundleValidationError(f"{name}: image label differs from sample")
        if tensor_index != index:
            raise BundleValidationError(f"{name}: tensor index is out of order")
        predictions.append(
            PredictionRecord(sample_id, defect, label, score, tensor_index)
        )
    labels = torch.tensor(
        [prediction.image_label for prediction in predictions],
        dtype=torch.uint8,
    ).contiguous()
    return tuple(predictions), labels


def _validate_memory_bank(
    payload: object, run: dict[str, object]
) -> tuple[MemoryBankMetadata, Tensor]:
    value = _object(payload, "memory_bank.pt")
    _keys(
        value,
        {
            "dtype",
            "embedding_dimension",
            "memory_bank",
            "patches_per_training_sample",
            "shape",
        },
        "memory_bank.pt",
    )
    inventory = _object(run["inventory"], "run.json.inventory")
    rows = int(inventory["training_sample_count"]) * _PATCH_COUNT
    shape = (rows, _EMBEDDING_DIMENSION)
    _equals(value["dtype"], "float32", "memory_bank.pt.dtype")
    _equals(
        value["embedding_dimension"],
        _EMBEDDING_DIMENSION,
        "memory_bank.pt.embedding_dimension",
    )
    _equals(
        value["patches_per_training_sample"],
        _PATCH_COUNT,
        "memory_bank.pt.patches_per_training_sample",
    )
    _shape(value["shape"], list(shape), "memory_bank.pt.shape")
    tensor = _tensor(
        value["memory_bank"],
        "memory_bank.pt.memory_bank",
        shape,
        torch.float32,
        finite=True,
    )
    return (
        MemoryBankMetadata("float32", shape, _EMBEDDING_DIMENSION, _PATCH_COUNT),
        tensor,
    )


def _validate_retrieval(
    payload: object, test_ids: tuple[str, ...], bank_rows: int
) -> tuple[Tensor, Tensor]:
    value = _object(payload, "retrieval.pt")
    _keys(
        value,
        {"nearest_bank_indices", "patch_distances", "test_sample_ids"},
        "retrieval.pt",
    )
    _test_ids(value["test_sample_ids"], test_ids, "retrieval.pt.test_sample_ids")
    shape = (len(test_ids), _PATCH_COUNT)
    distances = _tensor(
        value["patch_distances"],
        "retrieval.pt.patch_distances",
        shape,
        torch.float32,
        finite=True,
    )
    if torch.lt(distances, 0).any().item():
        raise BundleValidationError("retrieval.pt.patch_distances must be nonnegative")
    indices = _tensor(
        value["nearest_bank_indices"],
        "retrieval.pt.nearest_bank_indices",
        shape,
        torch.int64,
    )
    if torch.lt(indices, 0).any().item() or torch.ge(indices, bank_rows).any().item():
        raise BundleValidationError(
            "retrieval.pt nearest indices lie outside the memory bank"
        )
    return distances, indices


def _validate_prediction_scores(
    predictions: tuple[PredictionRecord, ...],
    patch_distances: Tensor,
) -> Tensor:
    maxima = patch_distances.max(dim=1).values
    for index, (prediction, maximum) in enumerate(
        zip(predictions, maxima, strict=True)
    ):
        if prediction.image_score != float(maximum.item()):
            raise BundleValidationError(
                "predictions.jsonl image score differs from "
                f"retrieval.pt row maximum at tensor index {index}"
            )
    return maxima


def _validate_maps(
    payload: object,
    test_samples: tuple[MvtecSample, ...],
    test_ids: tuple[str, ...],
) -> tuple[Tensor, Tensor]:
    value = _object(payload, "anomaly_maps.pt")
    _keys(
        value,
        {"anomaly_maps", "evaluation_masks", "test_sample_ids"},
        "anomaly_maps.pt",
    )
    _test_ids(value["test_sample_ids"], test_ids, "anomaly_maps.pt.test_sample_ids")
    shape = (len(test_ids), *_MAP_SIZE)
    maps = _tensor(
        value["anomaly_maps"],
        "anomaly_maps.pt.anomaly_maps",
        shape,
        torch.float32,
        finite=True,
    )
    masks = _tensor(
        value["evaluation_masks"],
        "anomaly_maps.pt.evaluation_masks",
        shape,
        torch.uint8,
    )
    if not torch.logical_or(masks == 0, masks == 1).all().item():
        raise BundleValidationError("anomaly_maps.pt evaluation masks must be binary")
    for index, sample in enumerate(test_samples):
        foreground = masks[index].count_nonzero().item()
        if sample.is_anomalous and foreground == 0:
            raise BundleValidationError(
                f"anomaly_maps.pt mask {index} is empty for an anomalous sample"
            )
        if not sample.is_anomalous and foreground != 0:
            raise BundleValidationError(
                f"anomaly_maps.pt mask {index} is nonempty for a good sample"
            )
    return maps, masks


def _validate_metrics(
    value: dict[str, object],
    labels: Tensor,
    scores: Tensor,
    masks: Tensor,
    maps: Tensor,
    run: dict[str, object],
) -> BundleMetrics:
    _keys(value, _METRIC_FIELDS, "metrics.json")
    metric_values = {
        name: _float(value[name], f"metrics.json.{name}")
        for name in ("image_auroc", "image_average_precision", "pixel_auroc")
    }
    counts = {
        name: _integer(value[name], f"metrics.json.{name}")
        for name in sorted(_METRIC_FIELDS - set(metric_values))
    }
    inventory = _object(run["inventory"], "run.json.inventory")
    expected_counts = {
        "training_sample_count": inventory["training_sample_count"],
        "test_sample_count": labels.shape[0],
        "test_good_sample_count": int(labels.eq(0).sum().item()),
        "anomalous_test_sample_count": int(labels.eq(1).sum().item()),
        "evaluated_pixel_count": masks.numel(),
        "anomalous_pixel_count": int(masks.count_nonzero().item()),
    }
    for name, expected in expected_counts.items():
        if counts[name] != expected:
            raise BundleValidationError(
                f"metrics.json.{name} differs from loaded artifacts"
            )
    try:
        recomputed = compute_threshold_free_metrics(labels, scores, masks, maps)
    except (TypeError, ValueError) as error:
        raise BundleValidationError(
            f"metrics.json inputs violate the metric contract: {error}"
        ) from error
    stored = ThresholdFreeMetrics(
        metric_values["image_auroc"],
        metric_values["image_average_precision"],
        metric_values["pixel_auroc"],
    )
    if stored != recomputed:
        raise BundleValidationError(
            "metrics.json values do not match recomputed frozen metrics"
        )
    return BundleMetrics(
        stored.image_auroc,
        stored.image_average_precision,
        stored.pixel_auroc,
        counts["training_sample_count"],
        counts["test_sample_count"],
        counts["test_good_sample_count"],
        counts["anomalous_test_sample_count"],
        counts["evaluated_pixel_count"],
        counts["anomalous_pixel_count"],
    )


def _validate_benchmark(
    benchmark: dict[str, object],
    run: dict[str, object],
    device: torch.device,
    test_ids: tuple[str, ...],
    bank: MemoryBankMetadata,
) -> None:
    _keys(benchmark, _BENCHMARK_FIELDS, "benchmark.json")
    _equals(benchmark["schema_version"], 1, "benchmark.json.schema_version")
    environment = _object(run["environment"], "run.json.environment")
    links = {
        "category": run["category"],
        "created_at_utc": environment["created_at_utc"],
        "device": run["device"],
        "profile_id": run["profile_id"],
        "run_id": run["run_id"],
    }
    for name, expected in links.items():
        _equals(benchmark[name], expected, f"benchmark.json.{name}")
    benchmark_sample = _relative_posix(
        benchmark["benchmark_sample_id"], "benchmark.json.benchmark_sample_id"
    )
    if benchmark_sample != test_ids[0]:
        raise BundleValidationError(
            "benchmark.json benchmark sample must be the first test sample"
        )

    workload = _object(benchmark["workload"], "benchmark.json.workload")
    workload_fields = {
        "D",
        "M",
        "Q",
        "bank_bytes",
        "bank_chunk_size",
        "bank_shape",
        "batch_size",
        "dtype",
        "k",
        "tensor_layout",
        "test_sample_count",
        "training_sample_count",
    }
    _keys(workload, workload_fields, "benchmark.json.workload")
    inventory = _object(run["inventory"], "run.json.inventory")
    bank_bytes = bank.shape[0] * bank.shape[1] * 4
    expected_workload = {
        "D": _EMBEDDING_DIMENSION,
        "M": bank.shape[0],
        "Q": _PATCH_COUNT,
        "bank_bytes": bank_bytes,
        "bank_chunk_size": run["bank_chunk_size"],
        "bank_shape": list(bank.shape),
        "batch_size": 1,
        "dtype": "float32",
        "k": 1,
        "test_sample_count": inventory["test_sample_count"],
        "training_sample_count": inventory["training_sample_count"],
    }
    for name, expected in expected_workload.items():
        _equals(workload[name], expected, f"benchmark.json.workload.{name}")
    layouts = _object(
        workload["tensor_layout"], "benchmark.json.workload.tensor_layout"
    )
    expected_layouts = {
        "anomaly_map": "BHW contiguous row-major",
        "image": "NCHW contiguous",
        "memory_bank": "MD contiguous row-major",
        "patch_embeddings": "BQD contiguous row-major",
    }
    if _canonical_json(layouts) != _canonical_json(expected_layouts):
        raise BundleValidationError("benchmark.json tensor layout is invalid")

    methodology = _object(benchmark["methodology"], "benchmark.json.methodology")
    warmups = _positive_integer(
        methodology.get("warmup_count"),
        "benchmark.json.methodology.warmup_count",
    )
    repeats = _positive_integer(
        methodology.get("repeat_count"),
        "benchmark.json.methodology.repeat_count",
    )
    expected_methodology = _methodology(device, warmups, repeats)
    if _canonical_json(methodology) != _canonical_json(expected_methodology):
        raise BundleValidationError("benchmark.json methodology is invalid")

    benchmark_environment = _object(
        benchmark["environment"], "benchmark.json.environment"
    )
    _keys(
        benchmark_environment,
        {
            "cuda_compute_capability",
            "cuda_device_name",
            "pytorch_cuda_runtime_version",
        },
        "benchmark.json.environment",
    )
    if device.type == "cuda":
        capability = benchmark_environment["cuda_compute_capability"]
        if (
            type(capability) is not list
            or len(capability) != 2
            or any(type(item) is not int or item < 0 for item in capability)
        ):
            raise BundleValidationError(
                "benchmark.json CUDA compute capability is invalid"
            )
        _string(
            benchmark_environment["cuda_device_name"],
            "benchmark.json.environment.cuda_device_name",
        )
        _string(
            benchmark_environment["pytorch_cuda_runtime_version"],
            "benchmark.json.environment.pytorch_cuda_runtime_version",
        )
    elif any(value is not None for value in benchmark_environment.values()):
        raise BundleValidationError(
            "benchmark.json CUDA environment fields must be null off CUDA"
        )
    _validate_benchmark_results(
        benchmark["results"], repeats, bank_bytes, device.type == "cuda"
    )


def _validate_benchmark_results(
    value: object, repeats: int, bank_bytes: int, cuda: bool
) -> None:
    results = _object(value, "benchmark.json.results")
    _keys(
        results,
        {
            "device_memory",
            "one_off_ms",
            "repeated_stages",
            "synchronized_end_to_end",
        },
        "benchmark.json.results",
    )
    one_off = _object(results["one_off_ms"], "benchmark.json.results.one_off_ms")
    _keys(
        one_off,
        {
            "bank_transfer_and_device_setup",
            "full_nominal_bank_build",
            "model_and_weight_load",
        },
        "benchmark.json.results.one_off_ms",
    )
    for name, duration in one_off.items():
        _nonnegative_float(duration, f"benchmark.json.results.one_off_ms.{name}")

    stages = _object(
        results["repeated_stages"],
        "benchmark.json.results.repeated_stages",
    )
    _keys(
        stages,
        _REPEATED_STAGES,
        "benchmark.json.results.repeated_stages",
    )
    for name, summary in stages.items():
        _validate_summary(summary, repeats, f"benchmark.json.results.{name}")
    _validate_summary(
        results["synchronized_end_to_end"],
        repeats,
        "benchmark.json.results.synchronized_end_to_end",
    )

    memory = _object(results["device_memory"], "benchmark.json.results.device_memory")
    _keys(
        memory,
        {
            "peak_allocated_boundary",
            "peak_allocated_bytes",
            "peak_reserved_boundary",
            "peak_reserved_bytes",
            "persistent_bank_bytes",
        },
        "benchmark.json.results.device_memory",
    )
    _equals(
        memory["persistent_bank_bytes"],
        bank_bytes,
        "benchmark.json.results.device_memory.persistent_bank_bytes",
    )
    expected_boundaries = (
        (
            "Reset after setup and warm-ups; peak covers persistent model and "
            "full-bank allocations plus PyTorch allocator activity during "
            "measured repeats, excluding setup and warm-up activity, full-category "
            "scoring, driver memory, and non-PyTorch allocations.",
            "Reset after setup and warm-ups; the reset retains the CUDA caching "
            "pool, so the peak includes reservations retained from setup and "
            "warm-ups plus any growth during measured repeats.",
        )
        if cuda
        else (
            "Not measured on CPU; no host peak approximation is made.",
            "Not measured on CPU; no host reservation approximation is made.",
        )
    )
    _equals(
        memory["peak_allocated_boundary"],
        expected_boundaries[0],
        "benchmark.json.results.device_memory.peak_allocated_boundary",
    )
    _equals(
        memory["peak_reserved_boundary"],
        expected_boundaries[1],
        "benchmark.json.results.device_memory.peak_reserved_boundary",
    )
    if cuda:
        allocated = _integer(
            memory["peak_allocated_bytes"],
            "benchmark.json.results.device_memory.peak_allocated_bytes",
        )
        reserved = _integer(
            memory["peak_reserved_bytes"],
            "benchmark.json.results.device_memory.peak_reserved_bytes",
        )
        if allocated < bank_bytes or reserved < allocated:
            raise BundleValidationError(
                "benchmark.json CUDA allocator byte counts are inconsistent"
            )
    elif (
        memory["peak_allocated_bytes"] is not None
        or memory["peak_reserved_bytes"] is not None
    ):
        raise BundleValidationError(
            "benchmark.json allocator byte counts must be null off CUDA"
        )


def _validate_summary(value: object, repeats: int, name: str) -> None:
    summary = _object(value, name)
    _keys(summary, _SUMMARY_FIELDS, name)
    _equals(summary["count"], repeats, f"{name}.count")
    numbers = {
        field: _nonnegative_float(summary[field], f"{name}.{field}")
        for field in sorted(_SUMMARY_FIELDS - {"count"})
    }
    if not (
        numbers["minimum"] <= numbers["p50"] <= numbers["p95"] <= numbers["maximum"]
        and numbers["minimum"] <= numbers["mean"] <= numbers["maximum"]
    ):
        raise BundleValidationError(f"{name}: summary statistics are inconsistent")


def _tensor(
    value: object,
    name: str,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    *,
    finite: bool = False,
) -> Tensor:
    if type(value) is not Tensor:
        raise BundleValidationError(f"{name} must be a base torch.Tensor")
    if value.device.type != "cpu":
        raise BundleValidationError(f"{name} must be on the CPU")
    if value.is_nested or value.layout is not torch.strided:
        raise BundleValidationError(f"{name} must use dense strided storage")
    if value.dtype != dtype:
        raise BundleValidationError(f"{name} must use {dtype}")
    if tuple(value.shape) != shape:
        raise BundleValidationError(f"{name} must have shape {shape}")
    if not value.is_contiguous():
        raise BundleValidationError(f"{name} must be contiguous")
    if finite and not torch.isfinite(value).all().item():
        raise BundleValidationError(f"{name} must contain only finite values")
    return value


def _test_ids(value: object, expected: tuple[str, ...], name: str) -> None:
    if type(value) is not list or any(type(item) is not str for item in value):
        raise BundleValidationError(f"{name} must be a list of strings")
    if tuple(value) != expected:
        raise BundleValidationError(f"{name} differs from ordered test samples")


def _keys(value: dict[str, object], expected: set[str], name: str) -> None:
    actual = set(value)
    if actual != expected:
        raise BundleValidationError(
            f"{name} fields are invalid; "
            f"missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}"
        )


def _object(value: object, name: str) -> dict[str, object]:
    if type(value) is not dict:
        raise BundleValidationError(f"{name} must be an object")
    return value


def _string(value: object, name: str) -> str:
    if (
        type(value) is not str
        or not value
        or any(ord(character) < 32 for character in value)
    ):
        raise BundleValidationError(
            f"{name} must be a nonempty string without control characters"
        )
    return value


def _boolean(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise BundleValidationError(f"{name} must be a boolean")
    return value


def _integer(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise BundleValidationError(f"{name} must be a nonnegative integer")
    return value


def _positive_integer(value: object, name: str) -> int:
    result = _integer(value, name)
    if result == 0:
        raise BundleValidationError(f"{name} must be a positive integer")
    return result


def _float(value: object, name: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise BundleValidationError(f"{name} must be a finite float")
    return value


def _nonnegative_float(value: object, name: str) -> float:
    result = _float(value, name)
    if result < 0:
        raise BundleValidationError(f"{name} must be nonnegative")
    return result


def _equals(value: object, expected: object, name: str) -> None:
    if type(value) is not type(expected) or value != expected:
        raise BundleValidationError(f"{name} must be {expected!r}")


def _shape(value: object, expected: list[int], name: str) -> None:
    if (
        type(value) is not list
        or any(type(item) is not int for item in value)
        or value != expected
    ):
        raise BundleValidationError(f"{name} must be {expected}")


def _sha256(value: object, name: str) -> str:
    result = _string(value, name)
    if not _SHA256.fullmatch(result):
        raise BundleValidationError(f"{name} must be a SHA-256 hex digest")
    return result


def _commit(value: object, name: str) -> str:
    result = _string(value, name)
    if not _COMMIT.fullmatch(result):
        raise BundleValidationError(f"{name} must be a full lowercase commit hash")
    return result


def _relative_posix(value: object, name: str) -> str:
    result = _string(value, name)
    if (
        "\\" in result
        or PurePosixPath(result).is_absolute()
        or any(part in {"", ".", ".."} for part in result.split("/"))
    ):
        raise BundleValidationError(f"{name} must be a safe relative POSIX path")
    return result


def _component(value: object, name: str) -> str:
    result = _relative_posix(value, name)
    if "/" in result:
        raise BundleValidationError(f"{name} must be one path component")
    return result


def _utc_timestamp(value: object, name: str) -> str:
    result = _string(value, name)
    timestamp = f"{result[:-1]}+00:00" if result.endswith("Z") else result
    try:
        parsed = datetime.fromisoformat(timestamp)
    except ValueError as error:
        raise BundleValidationError(f"{name} must be an ISO 8601 timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise BundleValidationError(f"{name} must be an ISO 8601 UTC timestamp")
    return result


def _freeze_json(value: object) -> object:
    if type(value) is dict:
        return MappingProxyType(
            {key: _freeze_json(item) for key, item in value.items()}
        )
    if type(value) is list:
        return tuple(_freeze_json(item) for item in value)
    return value
