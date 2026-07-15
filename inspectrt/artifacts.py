"""Persistence for reproducible feature-memory baseline runs."""

from collections.abc import Mapping
import ctypes
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
import errno
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tempfile
from types import MappingProxyType

import torch
from torch import Tensor

from inspectrt.benchmark import BaselineBenchmark
from inspectrt.data import MvtecSample
from inspectrt.evaluation import CategoryEvaluation, MvtecSampleObservation
from inspectrt.metrics import ThresholdFreeMetrics, compute_threshold_free_metrics
from inspectrt.preprocessing import PREPROCESSING_PROFILE

_JsonPrimitive = str | int | float | bool | None
_RUN_ID = re.compile(r"[A-Za-z0-9._-]+")
_SHA256 = re.compile(r"[0-9a-fA-F]{64}")
_PATCH_COUNT = 1024
_EMBEDDING_DIMENSION = 512
_MAP_SIZE = (256, 256)


@dataclass(frozen=True, slots=True)
class BaselineRunMetadata:
    """Caller-resolved identity needed to reproduce one baseline run."""

    run_id: str
    created_at_utc: str
    dataset_root: str
    requested_device: str
    bank_chunk_size: int
    git_commit: str
    git_dirty: bool
    uv_lock_sha256: str
    python_version: str
    platform_description: str
    dependency_versions: Mapping[str, _JsonPrimitive]
    determinism_flags: Mapping[str, _JsonPrimitive]
    weight_enum: str
    weight_source_url: str
    weight_file_sha256: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.run_id, str)
            or self.run_id in {"", ".", ".."}
            or (not _RUN_ID.fullmatch(self.run_id))
        ):
            raise ValueError(
                "run_id must be one path component using only letters, digits, '.', "
                "'_', and '-'"
            )
        for name in (
            "created_at_utc",
            "dataset_root",
            "requested_device",
            "git_commit",
            "python_version",
            "platform_description",
            "weight_enum",
            "weight_source_url",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a nonempty string")
        timestamp = (
            f"{self.created_at_utc[:-1]}+00:00"
            if self.created_at_utc.endswith("Z")
            else self.created_at_utc
        )
        try:
            parsed_timestamp = datetime.fromisoformat(timestamp)
        except ValueError as error:
            raise ValueError(
                "created_at_utc must be an ISO 8601 UTC timestamp"
            ) from error
        if parsed_timestamp.tzinfo is None or parsed_timestamp.utcoffset() != timedelta(
            0
        ):
            raise ValueError("created_at_utc must be an ISO 8601 UTC timestamp")
        if (
            not isinstance(self.bank_chunk_size, int)
            or isinstance(self.bank_chunk_size, bool)
            or self.bank_chunk_size <= 0
        ):
            raise ValueError("bank_chunk_size must be a positive integer")
        if not isinstance(self.git_dirty, bool):
            raise TypeError("git_dirty must be a boolean")
        for name in ("uv_lock_sha256", "weight_file_sha256"):
            value = getattr(self, name)
            if not isinstance(value, str) or not _SHA256.fullmatch(value):
                raise ValueError(f"{name} must be a full SHA-256 hex digest")
        for name in ("dependency_versions", "determinism_flags"):
            object.__setattr__(
                self, name, _frozen_primitives(getattr(self, name), name)
            )


def persist_baseline_run(
    evaluation: CategoryEvaluation,
    output_root: Path,
    metadata: BaselineRunMetadata,
    *,
    benchmark: BaselineBenchmark | None = None,
) -> Path:
    """Persist one validated evaluation using an atomic run-directory rename."""
    if not isinstance(output_root, Path):
        raise TypeError("output_root must be a pathlib.Path")
    if not isinstance(metadata, BaselineRunMetadata):
        raise TypeError("metadata must be BaselineRunMetadata")

    samples, test_ids, counts = _validate_evaluation(evaluation)
    _validate_benchmark(evaluation, metadata, benchmark)
    sample_bytes = b"".join(_canonical_json(record) for record in samples)
    inventory_sha256 = hashlib.sha256(sample_bytes).hexdigest()
    predictions = b"".join(
        _canonical_json(
            {
                "defect_type": observation.sample.defect_type,
                "image_label": int(evaluation.test_labels[index].item()),
                "image_score": float(evaluation.image_scores[index].item()),
                "sample_id": observation.sample.sample_id,
                "tensor_index": index,
            }
        )
        for index, observation in enumerate(evaluation.test_samples)
    )
    metrics = {
        "anomalous_pixel_count": int(evaluation.pixel_masks.count_nonzero().item()),
        "anomalous_test_sample_count": counts["anomalous_test_sample_count"],
        "evaluated_pixel_count": evaluation.pixel_masks.numel(),
        "image_auroc": evaluation.metrics.image_auroc,
        "image_average_precision": evaluation.metrics.image_average_precision,
        "pixel_auroc": evaluation.metrics.pixel_auroc,
        "test_good_sample_count": counts["test_good_sample_count"],
        "test_sample_count": counts["test_sample_count"],
        "training_sample_count": counts["training_sample_count"],
    }
    payloads = (
        ("samples.jsonl", sample_bytes),
        (
            "memory_bank.pt",
            {
                "dtype": _dtype_name(evaluation.memory_bank),
                "embedding_dimension": _EMBEDDING_DIMENSION,
                "memory_bank": evaluation.memory_bank,
                "patches_per_training_sample": _PATCH_COUNT,
                "shape": list(evaluation.memory_bank.shape),
            },
        ),
        ("predictions.jsonl", predictions),
        (
            "retrieval.pt",
            {
                "nearest_bank_indices": evaluation.nearest_bank_indices,
                "patch_distances": evaluation.patch_distances,
                "test_sample_ids": test_ids,
            },
        ),
        (
            "anomaly_maps.pt",
            {
                "anomaly_maps": evaluation.anomaly_maps,
                "evaluation_masks": evaluation.pixel_masks,
                "test_sample_ids": test_ids,
            },
        ),
        ("metrics.json", _canonical_json(metrics)),
        *((("benchmark.json", benchmark.canonical_json()),) if benchmark else ()),
        (
            "run.json",
            _canonical_json(
                _run_record(evaluation, metadata, inventory_sha256, counts, benchmark)
            ),
        ),
    )

    runs_root = output_root / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    destination = runs_root / metadata.run_id
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Run directory already exists: {destination}")
    temporary = Path(tempfile.mkdtemp(prefix=f".{metadata.run_id}.tmp-", dir=runs_root))
    try:
        for name, payload in payloads:
            path = temporary / name
            if isinstance(payload, bytes):
                path.write_bytes(payload)
            else:
                torch.save(payload, path)
        _rename_without_overwrite(temporary, destination)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return destination


def _validate_benchmark(
    evaluation: CategoryEvaluation,
    metadata: BaselineRunMetadata,
    benchmark: BaselineBenchmark | None,
) -> None:
    if benchmark is None:
        return
    if not isinstance(benchmark, BaselineBenchmark):
        raise TypeError("benchmark must be BaselineBenchmark or None")
    expected = {
        "category": evaluation.category,
        "created_at_utc": metadata.created_at_utc,
        "device": metadata.requested_device,
        "profile_id": "inspectrt_feature_memory_v1",
        "run_id": metadata.run_id,
    }
    for name, value in expected.items():
        if getattr(benchmark, name) != value:
            raise ValueError(f"Benchmark {name} must match the run {name}")


def _validate_evaluation(
    evaluation: CategoryEvaluation,
) -> tuple[list[dict[str, _JsonPrimitive]], list[str], dict[str, int]]:
    if not isinstance(evaluation, CategoryEvaluation):
        raise TypeError("evaluation must be CategoryEvaluation")
    if not isinstance(evaluation.category, str) or not evaluation.category.strip():
        raise ValueError("Evaluation category must be nonempty")
    if not evaluation.samples:
        raise ValueError("Sample inventory must be nonempty")

    records: list[dict[str, _JsonPrimitive]] = []
    inventory_ids = []
    for observation in evaluation.samples:
        if not isinstance(observation, MvtecSampleObservation) or not isinstance(
            observation.sample, MvtecSample
        ):
            raise TypeError("Inventory entries must be MvtecSampleObservation values")
        sample = observation.sample
        if not isinstance(sample.sample_id, str) or not sample.sample_id:
            raise ValueError("sample_id must be nonempty")
        if sample.category != evaluation.category:
            raise ValueError(
                "Every inventory sample must match the evaluation category"
            )
        if sample.split not in {"train", "test"}:
            raise ValueError(f"Invalid sample split {sample.split!r}")
        if not isinstance(sample.defect_type, str) or not sample.defect_type:
            raise ValueError("defect_type must be nonempty")
        if not isinstance(sample.is_anomalous, bool):
            raise TypeError("is_anomalous must be a boolean")
        if sample.split == "train" and (
            sample.defect_type != "good" or sample.is_anomalous
        ):
            raise ValueError("Training inventory may contain only nominal good samples")
        _relative_posix(sample.image_relpath, "image_relpath")
        if sample.mask_relpath is not None:
            _relative_posix(sample.mask_relpath, "mask_relpath")
        inventory_ids.append(sample.sample_id)
        records.append(asdict(sample))
    if len(inventory_ids) != len(set(inventory_ids)):
        raise ValueError("Sample inventory IDs must be unique")

    inventory_tests = tuple(
        item for item in evaluation.samples if item.sample.split == "test"
    )
    if evaluation.test_samples != inventory_tests:
        raise ValueError("Test sample order must match inventory test order")
    test_ids = [item.sample.sample_id for item in evaluation.test_samples]
    if not test_ids:
        raise ValueError("Evaluation must contain test samples")
    if len(test_ids) != len(set(test_ids)):
        raise ValueError("Test sample IDs must be unique")
    training_count = sum(item.sample.split == "train" for item in evaluation.samples)
    if not training_count:
        raise ValueError("Inventory must contain nominal training samples")

    test_count = len(test_ids)
    contracts = {
        "memory_bank": (
            (training_count * _PATCH_COUNT, _EMBEDDING_DIMENSION),
            torch.float32,
        ),
        "patch_distances": ((test_count, _PATCH_COUNT), torch.float32),
        "nearest_bank_indices": ((test_count, _PATCH_COUNT), torch.int64),
        "image_scores": ((test_count,), torch.float32),
        "anomaly_maps": ((test_count, *_MAP_SIZE), torch.float32),
        "pixel_masks": ((test_count, *_MAP_SIZE), torch.uint8),
        "test_labels": ((test_count,), None),
    }
    for name, (shape, dtype) in contracts.items():
        _tensor_contract(getattr(evaluation, name), name, shape, dtype)
    for tensor, name in (
        (evaluation.memory_bank, "Memory bank"),
        (evaluation.patch_distances, "Patch distances"),
    ):
        if not torch.isfinite(tensor).all().item():
            raise ValueError(f"{name} must contain only finite values")
    if evaluation.nearest_bank_indices.min().item() < 0 or (
        evaluation.nearest_bank_indices.max().item() >= evaluation.memory_bank.shape[0]
    ):
        raise ValueError("Nearest-bank indices must lie within the memory bank")
    expected_labels = [
        int(item.sample.is_anomalous) for item in evaluation.test_samples
    ]
    if evaluation.test_labels.tolist() != expected_labels:
        raise ValueError("Test labels must match ordered test samples")

    recomputed = compute_threshold_free_metrics(
        evaluation.test_labels,
        evaluation.image_scores,
        evaluation.pixel_masks,
        evaluation.anomaly_maps,
    )
    metric_values = (
        (
            evaluation.metrics.image_auroc,
            evaluation.metrics.image_average_precision,
            evaluation.metrics.pixel_auroc,
        )
        if isinstance(evaluation.metrics, ThresholdFreeMetrics)
        else ()
    )
    if len(metric_values) != 3 or any(
        type(value) is not float for value in metric_values
    ):
        raise TypeError("Stored metrics must contain Python floats")
    if not all(math.isfinite(value) for value in metric_values):
        raise ValueError("Stored metrics must contain only finite values")
    if evaluation.metrics != recomputed:
        raise ValueError("Stored metrics do not match evaluation tensors")

    anomalous_count = sum(expected_labels)
    return (
        records,
        test_ids,
        {
            "anomalous_test_sample_count": anomalous_count,
            "test_good_sample_count": test_count - anomalous_count,
            "test_sample_count": test_count,
            "total_sample_count": len(records),
            "training_sample_count": training_count,
        },
    )


def _run_record(
    evaluation: CategoryEvaluation,
    metadata: BaselineRunMetadata,
    inventory_sha256: str,
    counts: dict[str, int],
    benchmark: BaselineBenchmark | None,
) -> dict[str, object]:
    tensors = {
        name: {"dtype": _dtype_name(tensor), "shape": list(tensor.shape)}
        for name, tensor in (
            ("anomaly_maps", evaluation.anomaly_maps),
            ("evaluation_masks", evaluation.pixel_masks),
            ("image_scores", evaluation.image_scores),
            ("memory_bank", evaluation.memory_bank),
            ("nearest_bank_indices", evaluation.nearest_bank_indices),
            ("patch_distances", evaluation.patch_distances),
            ("test_labels", evaluation.test_labels),
        )
    }
    tensors["memory_bank"]["byte_count"] = (
        evaluation.memory_bank.numel() * evaluation.memory_bank.element_size()
    )
    return {
        "bank_chunk_size": metadata.bank_chunk_size,
        "batch_size": 1,
        "benchmark": (
            None
            if benchmark is None
            else {
                "artifact": "benchmark.json",
                "schema_version": benchmark.schema_version,
                "timing_device": benchmark.device,
            }
        ),
        "category": evaluation.category,
        "dataset_root": metadata.dataset_root,
        "determinism": dict(metadata.determinism_flags),
        "device": metadata.requested_device,
        "environment": {
            "created_at_utc": metadata.created_at_utc,
            "dependency_versions": dict(metadata.dependency_versions),
            "platform_description": metadata.platform_description,
            "python_version": metadata.python_version,
        },
        "feature_extractor": "ResNet-50",
        "feature_layer": "layer2",
        "inventory": {"sample_inventory_sha256": inventory_sha256, **counts},
        "map_interpolation": {
            "align_corners": False,
            "input_size": [32, 32],
            "mode": "bilinear",
            "output_size": [256, 256],
            "values": "raw squared-L2 patch distances",
        },
        "preprocessing_profile": PREPROCESSING_PROFILE,
        "profile_id": "inspectrt_feature_memory_v1",
        "retrieval_semantics": "exact top-1 squared L2",
        "run_id": metadata.run_id,
        "schema_version": 1,
        "source": {
            "dirty": metadata.git_dirty,
            "git_commit": metadata.git_commit,
            "uv_lock_sha256": metadata.uv_lock_sha256,
        },
        "tensors": tensors,
        "weights": {
            "cached_file_sha256": metadata.weight_file_sha256,
            "enum": metadata.weight_enum,
            "source_url": metadata.weight_source_url,
        },
    }


def _tensor_contract(
    tensor: object,
    name: str,
    shape: tuple[int, ...],
    dtype: torch.dtype | None,
) -> None:
    if not isinstance(tensor, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}; got {tuple(tensor.shape)}")
    if dtype is not None and tensor.dtype != dtype:
        raise TypeError(f"{name} must use {dtype}; got {tensor.dtype}")
    if tensor.device.type != "cpu":
        raise ValueError(f"{name} must be on the CPU")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _relative_posix(value: object, name: str) -> None:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"{name} must be a relative POSIX path")
    if PurePosixPath(value).is_absolute() or any(
        part in {"", ".", ".."} for part in value.split("/")
    ):
        raise ValueError(f"{name} must be a relative POSIX path")


def _frozen_primitives(value: object, name: str) -> Mapping[str, _JsonPrimitive]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    result = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise TypeError(f"{name} keys must be strings")
        if item is not None and type(item) not in {str, int, float, bool}:
            raise TypeError(f"{name} values must be JSON primitive values")
        if isinstance(item, float) and not math.isfinite(item):
            raise ValueError(f"{name} values must be finite")
        result[key] = item
    return MappingProxyType(result)


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _rename_without_overwrite(source: Path, destination: Path) -> None:
    if sys.platform.startswith("linux"):
        function = ctypes.CDLL(None, use_errno=True).renameat2
        arguments = (
            -100,
            os.fsencode(source),
            -100,
            os.fsencode(destination),
            1,
        )
    elif sys.platform == "darwin":
        function = ctypes.CDLL(None, use_errno=True).renamex_np
        arguments = (os.fsencode(source), os.fsencode(destination), 4)
    elif os.name == "nt":
        source.rename(destination)
        return
    else:
        raise OSError(errno.ENOTSUP, "Atomic no-overwrite rename is unsupported")
    result = function(*arguments)
    if result != 0:
        error = ctypes.get_errno()
        if error in {errno.EEXIST, errno.ENOTEMPTY}:
            raise FileExistsError(error, os.strerror(error), destination)
        raise OSError(error, os.strerror(error), destination)


def _dtype_name(tensor: Tensor) -> str:
    return str(tensor.dtype).removeprefix("torch.")
