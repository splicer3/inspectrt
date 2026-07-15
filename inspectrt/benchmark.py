"""Truthful batch-1 stage measurement for the feature-memory baseline."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import math
from pathlib import Path
from time import perf_counter_ns
from types import MappingProxyType

import numpy as np
import torch
from torch import nn

from inspectrt.evaluation import (
    CategoryEvaluation,
    MvtecSampleObservation,
    _build_nominal_memory_bank,
    _discover_category_samples,
    _resolve_evaluation_device,
    _score_and_finalize_category,
    _transfer_memory_bank,
)
from inspectrt.features import extract_patch_embeddings
from inspectrt.preprocessing import decode_image, preprocess_decoded_image
from inspectrt.retrieval import exact_top1_squared_l2, reconstruct_anomaly_maps

_PROFILE_ID = "inspectrt_feature_memory_v1"
_REPEATED_STAGES = (
    "image_decode",
    "canonical_image_preprocessing",
    "host_to_device_transfer",
    "frozen_feature_extraction",
    "exact_chunked_retrieval",
    "anomaly_map_reconstruction",
)


@dataclass(frozen=True, slots=True)
class BaselineBenchmark:
    """Immutable, finite benchmark record with a canonical JSON form."""

    schema_version: int
    profile_id: str
    category: str
    device: str
    benchmark_sample_id: str
    run_id: str
    created_at_utc: str
    workload: Mapping[str, object]
    methodology: Mapping[str, object]
    environment: Mapping[str, object]
    results: Mapping[str, object]

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("Benchmark schema_version must be 1")
        for name in (
            "profile_id",
            "category",
            "device",
            "benchmark_sample_id",
            "run_id",
            "created_at_utc",
        ):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be a nonempty string")
        for name in ("workload", "methodology", "environment", "results"):
            object.__setattr__(self, name, _freeze_mapping(getattr(self, name), name))

    def to_json_value(self) -> dict[str, object]:
        """Return JSON-compatible built-in Python values."""
        return {
            "benchmark_sample_id": self.benchmark_sample_id,
            "category": self.category,
            "created_at_utc": self.created_at_utc,
            "device": self.device,
            "environment": _thaw_json(self.environment),
            "methodology": _thaw_json(self.methodology),
            "profile_id": self.profile_id,
            "results": _thaw_json(self.results),
            "run_id": self.run_id,
            "schema_version": self.schema_version,
            "workload": _thaw_json(self.workload),
        }

    def canonical_json(self) -> bytes:
        """Serialize with the run bundle's canonical finite JSON rules."""
        return (
            json.dumps(
                self.to_json_value(),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")


def benchmark_mvtec_category(
    dataset_root: Path,
    category: str,
    feature_extractor: nn.Module,
    *,
    device: torch.device | str,
    bank_chunk_size: int,
    warmup_count: int,
    repeat_count: int,
    model_and_weight_load_ms: float,
    run_id: str,
    created_at_utc: str,
) -> tuple[CategoryEvaluation, BaselineBenchmark]:
    """Evaluate a category and measure its real batch-1 baseline stages."""
    _positive_count(warmup_count, "warmup_count")
    _positive_count(repeat_count, "repeat_count")
    if (
        type(model_and_weight_load_ms) is not float
        or not math.isfinite(model_and_weight_load_ms)
        or model_and_weight_load_ms < 0
    ):
        raise ValueError("model_and_weight_load_ms must be a finite nonnegative float")

    samples, nominal_samples, test_samples = _discover_category_samples(
        dataset_root, category
    )
    requested_device = torch.device(device)
    if requested_device.type == "cuda" and requested_device.index is None:
        raise ValueError("CUDA benchmark device must include an explicit index")
    resolved_device = _resolve_evaluation_device(feature_extractor, requested_device)
    observations: dict[str, MvtecSampleObservation] = {}

    _synchronize_if_cuda(resolved_device)
    bank_start = perf_counter_ns()
    memory_bank = _build_nominal_memory_bank(
        dataset_root,
        nominal_samples,
        feature_extractor,
        resolved_device,
        observations,
    )
    _synchronize_if_cuda(resolved_device)
    bank_build_ms = _wall_milliseconds(bank_start, perf_counter_ns())

    retrieval_bank, bank_transfer_ms = _time_bank_transfer(memory_bank, resolved_device)
    canonical_sample = test_samples[0]
    image_path = dataset_root / canonical_sample.image_relpath
    cuda_events = _new_cuda_stage_events() if resolved_device.type == "cuda" else None

    for _ in range(warmup_count):
        _measure_batch1(
            image_path,
            feature_extractor,
            retrieval_bank,
            resolved_device,
            bank_chunk_size,
            cuda_events=cuda_events,
        )

    peak_allocated_bytes: int | None = None
    peak_reserved_bytes: int | None = None
    if resolved_device.type == "cuda":
        torch.cuda.synchronize(resolved_device)
        torch.cuda.reset_peak_memory_stats(resolved_device)

    measurements = {name: [] for name in (*_REPEATED_STAGES, "end_to_end")}
    for _ in range(repeat_count):
        sample_measurements = _measure_batch1(
            image_path,
            feature_extractor,
            retrieval_bank,
            resolved_device,
            bank_chunk_size,
            cuda_events=cuda_events,
        )
        for name, duration in sample_measurements.items():
            measurements[name].append(duration)

    if resolved_device.type == "cuda":
        torch.cuda.synchronize(resolved_device)
        peak_allocated_bytes = int(torch.cuda.max_memory_allocated(resolved_device))
        peak_reserved_bytes = int(torch.cuda.max_memory_reserved(resolved_device))

    evaluation = _score_and_finalize_category(
        dataset_root,
        category,
        samples,
        test_samples,
        feature_extractor,
        resolved_device,
        bank_chunk_size,
        memory_bank,
        retrieval_bank,
        observations,
    )
    bank_bytes = memory_bank.numel() * memory_bank.element_size()
    repeated = {name: _statistics(measurements[name]) for name in _REPEATED_STAGES}
    benchmark = BaselineBenchmark(
        schema_version=1,
        profile_id=_PROFILE_ID,
        category=category,
        device=str(resolved_device),
        benchmark_sample_id=canonical_sample.sample_id,
        run_id=run_id,
        created_at_utc=created_at_utc,
        workload={
            "D": 512,
            "M": int(memory_bank.shape[0]),
            "Q": 1024,
            "bank_bytes": int(bank_bytes),
            "bank_chunk_size": bank_chunk_size,
            "bank_shape": list(memory_bank.shape),
            "batch_size": 1,
            "dtype": "float32",
            "k": 1,
            "tensor_layout": {
                "anomaly_map": "BHW contiguous row-major",
                "image": "NCHW contiguous",
                "memory_bank": "MD contiguous row-major",
                "patch_embeddings": "BQD contiguous row-major",
            },
            "test_sample_count": len(test_samples),
            "training_sample_count": len(nominal_samples),
        },
        methodology=_methodology(resolved_device, warmup_count, repeat_count),
        environment=_cuda_environment(resolved_device),
        results={
            "device_memory": {
                "peak_allocated_bytes": peak_allocated_bytes,
                "peak_reserved_bytes": peak_reserved_bytes,
                "persistent_bank_bytes": int(bank_bytes),
                "peak_allocated_boundary": (
                    (
                        "Reset after setup and warm-ups; peak covers persistent model and "
                        "full-bank allocations plus PyTorch allocator activity during "
                        "measured repeats, excluding setup and warm-up activity, full-category "
                        "scoring, driver memory, and non-PyTorch allocations."
                    )
                    if resolved_device.type == "cuda"
                    else "Not measured on CPU; no host peak approximation is made."
                ),
                "peak_reserved_boundary": (
                    (
                        "Reset after setup and warm-ups; the reset retains the CUDA caching "
                        "pool, so the peak includes reservations retained from setup and "
                        "warm-ups plus any growth during measured repeats."
                    )
                    if resolved_device.type == "cuda"
                    else "Not measured on CPU; no host reservation approximation is made."
                ),
            },
            "one_off_ms": {
                "bank_transfer_and_device_setup": bank_transfer_ms,
                "full_nominal_bank_build": bank_build_ms,
                "model_and_weight_load": model_and_weight_load_ms,
            },
            "repeated_stages": repeated,
            "synchronized_end_to_end": _statistics(measurements["end_to_end"]),
        },
    )
    return evaluation, benchmark


def _measure_batch1(
    image_path: Path,
    feature_extractor: nn.Module,
    retrieval_bank: torch.Tensor,
    device: torch.device,
    bank_chunk_size: int,
    *,
    cuda_events: tuple[tuple[torch.cuda.Event, torch.cuda.Event], ...] | None = None,
) -> dict[str, float]:
    if device.type == "cuda":
        return _measure_cuda_batch1(
            image_path,
            feature_extractor,
            retrieval_bank,
            device,
            bank_chunk_size,
            cuda_events,
        )
    return _measure_cpu_batch1(
        image_path, feature_extractor, retrieval_bank, device, bank_chunk_size
    )


def _measure_cpu_batch1(
    image_path: Path,
    feature_extractor: nn.Module,
    retrieval_bank: torch.Tensor,
    device: torch.device,
    bank_chunk_size: int,
) -> dict[str, float]:
    total_start = perf_counter_ns()
    decode_start = perf_counter_ns()
    decoded = decode_image(image_path)
    decode_ms = _wall_milliseconds(decode_start, perf_counter_ns())
    try:
        preprocessing_start = perf_counter_ns()
        image = preprocess_decoded_image(decoded)
        preprocessing_ms = _wall_milliseconds(preprocessing_start, perf_counter_ns())

        transfer_start = perf_counter_ns()
        images = image.unsqueeze(0).to(device)
        transfer_ms = _wall_milliseconds(transfer_start, perf_counter_ns())

        feature_start = perf_counter_ns()
        patches = extract_patch_embeddings(feature_extractor, images)
        feature_ms = _wall_milliseconds(feature_start, perf_counter_ns())

        retrieval_start = perf_counter_ns()
        distances, indices = exact_top1_squared_l2(
            patches[0], retrieval_bank, bank_chunk_size=bank_chunk_size
        )
        patch_distances = distances.reshape(1, 1024).contiguous()
        nearest_bank_indices = indices.reshape(1, 1024).contiguous()
        image_scores = patch_distances.max(dim=1).values.contiguous()
        retrieval_ms = _wall_milliseconds(retrieval_start, perf_counter_ns())

        map_start = perf_counter_ns()
        anomaly_maps = reconstruct_anomaly_maps(patch_distances)
        map_ms = _wall_milliseconds(map_start, perf_counter_ns())
        end_to_end_ms = _wall_milliseconds(total_start, perf_counter_ns())
        del nearest_bank_indices, image_scores, anomaly_maps
    finally:
        decoded.image.close()
    return {
        "image_decode": decode_ms,
        "canonical_image_preprocessing": preprocessing_ms,
        "host_to_device_transfer": transfer_ms,
        "frozen_feature_extraction": feature_ms,
        "exact_chunked_retrieval": retrieval_ms,
        "anomaly_map_reconstruction": map_ms,
        "end_to_end": end_to_end_ms,
    }


def _measure_cuda_batch1(
    image_path: Path,
    feature_extractor: nn.Module,
    retrieval_bank: torch.Tensor,
    device: torch.device,
    bank_chunk_size: int,
    cuda_events: tuple[tuple[torch.cuda.Event, torch.cuda.Event], ...] | None,
) -> dict[str, float]:
    torch.cuda.synchronize(device)
    stream = torch.cuda.current_stream(device)
    events = cuda_events or _new_cuda_stage_events()
    total_start = perf_counter_ns()
    decode_start = perf_counter_ns()
    decoded = decode_image(image_path)
    decode_ms = _wall_milliseconds(decode_start, perf_counter_ns())
    try:
        preprocessing_start = perf_counter_ns()
        image = preprocess_decoded_image(decoded)
        preprocessing_ms = _wall_milliseconds(preprocessing_start, perf_counter_ns())

        batched_image = image.unsqueeze(0)
        transfer_events = events[0]
        transfer_events[0].record(stream)
        images = batched_image.to(device)
        transfer_events[1].record(stream)

        feature_events = events[1]
        feature_events[0].record(stream)
        patches = extract_patch_embeddings(feature_extractor, images)
        feature_events[1].record(stream)

        retrieval_events = events[2]
        retrieval_events[0].record(stream)
        distances, indices = exact_top1_squared_l2(
            patches[0], retrieval_bank, bank_chunk_size=bank_chunk_size
        )
        patch_distances = distances.reshape(1, 1024).contiguous()
        nearest_bank_indices = indices.reshape(1, 1024).contiguous()
        image_scores = patch_distances.max(dim=1).values.contiguous()
        retrieval_events[1].record(stream)

        map_events = events[3]
        map_events[0].record(stream)
        anomaly_maps = reconstruct_anomaly_maps(patch_distances)
        map_events[1].record(stream)
        torch.cuda.synchronize(device)
        end_to_end_ms = _wall_milliseconds(total_start, perf_counter_ns())
        cuda_stage_ms = [
            float(start.elapsed_time(end))
            for start, end in (
                transfer_events,
                feature_events,
                retrieval_events,
                map_events,
            )
        ]
        del nearest_bank_indices, image_scores, anomaly_maps
    finally:
        decoded.image.close()
    return {
        "image_decode": decode_ms,
        "canonical_image_preprocessing": preprocessing_ms,
        "host_to_device_transfer": cuda_stage_ms[0],
        "frozen_feature_extraction": cuda_stage_ms[1],
        "exact_chunked_retrieval": cuda_stage_ms[2],
        "anomaly_map_reconstruction": cuda_stage_ms[3],
        "end_to_end": end_to_end_ms,
    }


def _time_bank_transfer(
    memory_bank: torch.Tensor, device: torch.device
) -> tuple[torch.Tensor, float]:
    if device.type != "cuda":
        start = perf_counter_ns()
        bank = _transfer_memory_bank(memory_bank, device)
        return bank, _wall_milliseconds(start, perf_counter_ns())
    torch.cuda.synchronize(device)
    stream = torch.cuda.current_stream(device)
    start_event, end_event = _start_cuda_stage(stream)
    bank = _transfer_memory_bank(memory_bank, device)
    end_event.record(stream)
    torch.cuda.synchronize(device)
    return bank, float(start_event.elapsed_time(end_event))


def _start_cuda_stage(stream: object) -> tuple[torch.cuda.Event, torch.cuda.Event]:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record(stream)
    return start, end


def _new_cuda_stage_events() -> tuple[tuple[torch.cuda.Event, torch.cuda.Event], ...]:
    return tuple(
        (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        for _ in range(4)
    )


def _statistics(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("Cannot summarize an empty timing sequence")
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all() or (array < 0).any():
        raise ValueError("Timing values must be finite and nonnegative")
    p50, p95 = np.percentile(array, (50, 95), method="linear")
    return {
        "count": len(values),
        "maximum": float(array.max()),
        "mean": float(array.mean()),
        "minimum": float(array.min()),
        "p50": float(p50),
        "p95": float(p95),
    }


def _methodology(
    device: torch.device, warmup_count: int, repeat_count: int
) -> dict[str, object]:
    cuda = device.type == "cuda"
    return {
        "cpu_timing_method": "time.perf_counter_ns wall clock",
        "cuda_timing_method": (
            "torch.cuda.Event(enable_timing=True) on the requested device stream"
            if cuda
            else None
        ),
        "repeat_count": repeat_count,
        "stage_inclusion_boundaries": {
            "anomaly_map_reconstruction": (
                "Includes raw 32x32-to-256x256 bilinear reconstruction and output "
                "allocation; excludes retrieval, transfer, mask work, and persistence."
            ),
            "canonical_image_preprocessing": (
                "Includes canonical resize, tensor conversion, FP32 normalization, and "
                "allocations; excludes decode and all mask work."
            ),
            "exact_chunked_retrieval": (
                "Includes exact full-bank chunked squared-L2 top-1, stable chunk merge, "
                "result reshape, image maximum, and allocations; excludes bank transfer "
                "and map reconstruction."
            ),
            "frozen_feature_extraction": (
                "Includes frozen ResNet layer2 inference, local average, row-major patch "
                "layout, and output allocation; excludes input transfer and retrieval."
            ),
            "full_nominal_bank_build": (
                "Wall clock includes ordered train/good decode and preprocessing, each "
                "image transfer, feature and patch extraction, CPU copy, contiguous FP32 "
                "concatenation and allocations, plus completion synchronization on CUDA; "
                "excludes model load and full-bank transfer."
            ),
            "host_to_device_transfer": (
                "CPU wall time includes the batch view and .to(cpu) no-copy path. CUDA is "
                "the device timestamp interval around .to(device); destination creation is "
                "invoked between events, while host API and allocation overhead are not "
                "measured independently."
            ),
            "image_decode": (
                "Includes file validation, Pillow open/decompression, RGB conversion, and "
                "decoded-image allocation; excludes resize, tensor conversion, and masks."
            ),
            "model_and_weight_load": (
                "Wall clock includes pinned cached-weight resolution/read, model and frozen "
                "extractor construction, allocation, requested-device placement, and CUDA "
                "completion synchronization when requested; excludes imports, CLI/config "
                "parsing, Git and dependency metadata, bank work, and persistence."
            ),
            "bank_transfer_and_device_setup": (
                "Includes the one-time full-bank .to(requested device) operation. CUDA is "
                "the device timestamp interval around that call; destination allocation is "
                "requested between events but host overhead is not independently measured, "
                "and synchronization waits are excluded. CPU uses wall time. Excludes model "
                "load, bank build, and repeats."
            ),
            "synchronized_end_to_end": (
                "Wall clock begins before decode and ends after the raw anomaly map is ready; "
                "final completion synchronization is included on CUDA. Model load, bank "
                "build and transfer, masks, metrics, JSON, persistence, and console output "
                "are excluded."
            ),
        },
        "synchronization_policy": (
            "The explicitly indexed CUDA device is synchronized before each measured "
            "sequence and before event durations are consumed; pre-synchronization is "
            "excluded and the final end-to-end synchronization is included. CUDA stage and "
            "bank-transfer values exclude synchronization waits; event-record overhead "
            "remains in synchronized end-to-end wall time."
            if cuda
            else "CPU operations are measured synchronously with no CUDA synchronization."
        ),
        "timing_unit": "milliseconds",
        "warmup_count": warmup_count,
        "warmup_samples_in_statistics": False,
    }


def _cuda_environment(device: torch.device) -> dict[str, object]:
    if device.type != "cuda":
        return {
            "cuda_compute_capability": None,
            "cuda_device_name": None,
            "pytorch_cuda_runtime_version": None,
        }
    return {
        "cuda_compute_capability": list(torch.cuda.get_device_capability(device)),
        "cuda_device_name": torch.cuda.get_device_name(device),
        "pytorch_cuda_runtime_version": torch.version.cuda,
    }


def _positive_count(value: object, name: str) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _wall_milliseconds(start_ns: int, end_ns: int) -> float:
    return float(end_ns - start_ns) / 1_000_000.0


def _synchronize_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _freeze_mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    frozen = _freeze_json(value)
    assert isinstance(frozen, Mapping)
    return frozen


def _freeze_json(value: object) -> object:
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("Benchmark JSON object keys must be strings")
            result[key] = _freeze_json(item)
        return MappingProxyType(result)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    if value is None or type(value) in {str, int, bool}:
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise TypeError("Benchmark values must be finite JSON-compatible Python primitives")


def _thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value
