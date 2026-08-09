"""Synchronized batch-1 timing for the frozen feature-memory baseline."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import math
import os
from pathlib import Path
from time import perf_counter_ns
from types import MappingProxyType
from typing import TypeVar

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
_METHODOLOGY_ID = "inspectrt_synchronized_wall_clock_v2"
_WARMUP_COUNT = 5
_REPEAT_COUNT = 30
_MAX_NANOSECONDS = 9_223_372_036_854_775_807
_BENCHMARK_SAMPLE_ID = "mvtec_ad/bottle/test/broken_large/000.png"
_ONE_OFF_STAGES = (
    "model_and_weight_load",
    "full_nominal_bank_build",
    "bank_transfer_and_device_setup",
)
_REPEATED_STAGES = (
    "image_decode",
    "canonical_image_preprocessing",
    "host_to_device_transfer",
    "frozen_feature_extraction",
    "exact_chunked_retrieval",
    "anomaly_map_reconstruction",
)
_SUMMARY_FIELDS = {"count", "minimum", "maximum", "mean", "p50", "p95"}
_T = TypeVar("_T")


@dataclass(frozen=True, slots=True)
class BaselineBenchmark:
    """Immutable schema-2 timing record for the frozen bottle workload."""

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
        if type(self.schema_version) is not int or self.schema_version != 2:
            raise ValueError("Benchmark schema_version must be 2")
        if self.profile_id != _PROFILE_ID:
            raise ValueError(f"profile_id must be {_PROFILE_ID}")
        if self.category != "bottle":
            raise ValueError("category must be bottle")
        if self.benchmark_sample_id != _BENCHMARK_SAMPLE_ID:
            raise ValueError(f"benchmark_sample_id must be {_BENCHMARK_SAMPLE_ID}")
        for name in ("device", "run_id", "created_at_utc"):
            if type(getattr(self, name)) is not str or not getattr(self, name):
                raise ValueError(f"{name} must be a nonempty string")
        _validate_workload(self.workload)
        _validate_methodology(self.methodology)
        backend = _validate_environment(self.device, self.environment)
        _validate_results(self.results, backend)
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


def benchmark_mvtec_category(
    dataset_root: Path,
    category: str,
    feature_extractor: nn.Module,
    *,
    device: torch.device | str,
    bank_chunk_size: int,
    warmup_count: int,
    repeat_count: int,
    model_and_weight_load_ns: int,
    run_id: str,
    created_at_utc: str,
) -> tuple[CategoryEvaluation, BaselineBenchmark]:
    """Evaluate bottle and retain synchronized raw timing observations."""
    if warmup_count != _WARMUP_COUNT:
        raise ValueError(f"warmup_count must be {_WARMUP_COUNT}")
    if repeat_count != _REPEAT_COUNT:
        raise ValueError(f"repeat_count must be {_REPEAT_COUNT}")
    _raw_nanoseconds(model_and_weight_load_ns, "model_and_weight_load_ns")
    if category != "bottle":
        raise ValueError("Schema-2 benchmark category must be bottle")
    if bank_chunk_size != 16_384:
        raise ValueError("Schema-2 benchmark bank_chunk_size must be 16384")

    requested_device = torch.device(device)
    _synchronize_backend(requested_device)
    resolved_device = _resolve_evaluation_device(feature_extractor, requested_device)
    if resolved_device != requested_device:
        raise RuntimeError("Requested benchmark device changed during resolution")
    samples, nominal_samples, test_samples = _discover_category_samples(
        dataset_root, category
    )
    if len(nominal_samples) != 209 or len(test_samples) != 83:
        raise ValueError("Schema-2 benchmark requires 209 train and 83 test samples")
    canonical_sample = test_samples[0]
    if canonical_sample.sample_id != _BENCHMARK_SAMPLE_ID:
        raise ValueError("Schema-2 benchmark sample must be the frozen bottle sample")
    observations: dict[str, MvtecSampleObservation] = {}

    memory_bank, bank_build_ns = _time_backend_operation(
        resolved_device,
        lambda: _build_nominal_memory_bank(
            dataset_root,
            nominal_samples,
            feature_extractor,
            resolved_device,
            observations,
        ),
    )
    if tuple(memory_bank.shape) != (214_016, 512):
        raise ValueError("Schema-2 benchmark memory bank must have shape [214016, 512]")
    retrieval_bank, bank_transfer_ns = _time_backend_operation(
        resolved_device,
        lambda: _transfer_memory_bank(memory_bank, resolved_device),
    )
    image_path = dataset_root / canonical_sample.image_relpath

    mps_observations = []
    if resolved_device.type == "mps":
        mps_observations.append(_mps_memory_observation("after_setup"))

    for _ in range(_WARMUP_COUNT):
        _measure_segmented_batch1(
            image_path,
            feature_extractor,
            retrieval_bank,
            resolved_device,
            bank_chunk_size,
        )
        _measure_end_to_end_batch1(
            image_path,
            feature_extractor,
            retrieval_bank,
            resolved_device,
            bank_chunk_size,
        )

    if resolved_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(resolved_device)
    elif resolved_device.type == "mps":
        mps_observations.append(_mps_memory_observation("after_warmups"))

    measurements = {name: [] for name in _REPEATED_STAGES}
    end_to_end = []
    for _ in range(_REPEAT_COUNT):
        stage_values = _measure_segmented_batch1(
            image_path,
            feature_extractor,
            retrieval_bank,
            resolved_device,
            bank_chunk_size,
        )
        for name, duration in zip(_REPEATED_STAGES, stage_values, strict=True):
            measurements[name].append(duration)
        end_to_end.append(
            _measure_end_to_end_batch1(
                image_path,
                feature_extractor,
                retrieval_bank,
                resolved_device,
                bank_chunk_size,
            )
        )

    if resolved_device.type == "mps":
        mps_observations.append(_mps_memory_observation("after_measured_passes"))
    memory_observations = _memory_observations(resolved_device, mps_observations)

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
    benchmark = BaselineBenchmark(
        schema_version=2,
        profile_id=_PROFILE_ID,
        category=category,
        device=str(resolved_device),
        benchmark_sample_id=canonical_sample.sample_id,
        run_id=run_id,
        created_at_utc=created_at_utc,
        workload=_workload(),
        methodology=_methodology(),
        environment=_environment(resolved_device),
        results={
            "one_off": {
                "model_and_weight_load": _timing_component([model_and_weight_load_ns]),
                "full_nominal_bank_build": _timing_component([bank_build_ns]),
                "bank_transfer_and_device_setup": _timing_component([bank_transfer_ns]),
            },
            "repeated_stages": {
                name: _timing_component(values) for name, values in measurements.items()
            },
            "synchronized_end_to_end": _timing_component(end_to_end),
            "memory_observations": memory_observations,
        },
    )
    return evaluation, benchmark


def _measure_segmented_batch1(
    image_path: Path,
    feature_extractor: nn.Module,
    retrieval_bank: torch.Tensor,
    device: torch.device,
    bank_chunk_size: int,
) -> tuple[int, int, int, int, int, int]:
    decode_start = perf_counter_ns()
    decoded = decode_image(image_path)
    decode_ns = _elapsed_nanoseconds(decode_start, perf_counter_ns())
    try:
        preprocessing_start = perf_counter_ns()
        image = preprocess_decoded_image(decoded)
        preprocessing_ns = _elapsed_nanoseconds(preprocessing_start, perf_counter_ns())

        images, transfer_ns = _time_backend_operation(
            device, lambda: image.unsqueeze(0).to(device)
        )
        patches, feature_ns = _time_backend_operation(
            device, lambda: extract_patch_embeddings(feature_extractor, images)
        )

        def retrieve() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            distances, indices = exact_top1_squared_l2(
                patches[0], retrieval_bank, bank_chunk_size=bank_chunk_size
            )
            patch_distances = distances.reshape(1, 1024).contiguous()
            nearest_bank_indices = indices.reshape(1, 1024).contiguous()
            image_scores = patch_distances.max(dim=1).values.contiguous()
            return patch_distances, nearest_bank_indices, image_scores

        (patch_distances, nearest_bank_indices, image_scores), retrieval_ns = (
            _time_backend_operation(device, retrieve)
        )
        anomaly_maps, map_ns = _time_backend_operation(
            device, lambda: reconstruct_anomaly_maps(patch_distances)
        )
        del nearest_bank_indices, image_scores, anomaly_maps
    finally:
        decoded.image.close()
    return (
        decode_ns,
        preprocessing_ns,
        transfer_ns,
        feature_ns,
        retrieval_ns,
        map_ns,
    )


def _measure_end_to_end_batch1(
    image_path: Path,
    feature_extractor: nn.Module,
    retrieval_bank: torch.Tensor,
    device: torch.device,
    bank_chunk_size: int,
) -> int:
    _synchronize_backend(device, validate=False)
    start = perf_counter_ns()
    decoded = decode_image(image_path)
    try:
        image = preprocess_decoded_image(decoded)
        images = image.unsqueeze(0).to(device)
        patches = extract_patch_embeddings(feature_extractor, images)
        distances, indices = exact_top1_squared_l2(
            patches[0], retrieval_bank, bank_chunk_size=bank_chunk_size
        )
        patch_distances = distances.reshape(1, 1024).contiguous()
        nearest_bank_indices = indices.reshape(1, 1024).contiguous()
        image_scores = patch_distances.max(dim=1).values.contiguous()
        anomaly_maps = reconstruct_anomaly_maps(patch_distances)
        _synchronize_backend(device, validate=False)
        duration = _elapsed_nanoseconds(start, perf_counter_ns())
        del nearest_bank_indices, image_scores, anomaly_maps
        return duration
    finally:
        decoded.image.close()


def _time_backend_operation(
    device: torch.device, operation: Callable[[], _T]
) -> tuple[_T, int]:
    _synchronize_backend(device, validate=False)
    start = perf_counter_ns()
    result = operation()
    _synchronize_backend(device, validate=False)
    return result, _elapsed_nanoseconds(start, perf_counter_ns())


def _synchronize_backend(device: torch.device, *, validate: bool = True) -> None:
    if device.type == "cpu":
        if str(device) != "cpu":
            raise ValueError("CPU benchmark device must be exactly cpu")
        return
    if device.type == "cuda":
        if device.index is None:
            raise ValueError("CUDA benchmark device must include an explicit index")
        if validate and (
            not torch.cuda.is_available() or device.index >= torch.cuda.device_count()
        ):
            raise RuntimeError(f"CUDA device {device} requested but unavailable")
        torch.cuda.synchronize(device)
        return
    if device.type == "mps":
        if str(device) != "mps":
            raise ValueError("MPS benchmark device must be exactly mps")
        if validate and not torch.backends.mps.is_built():
            raise RuntimeError(
                "MPS backend requested but PyTorch was not built with MPS"
            )
        if validate and not torch.backends.mps.is_available():
            raise RuntimeError("MPS backend requested but unavailable")
        if validate and "PYTORCH_ENABLE_MPS_FALLBACK" in os.environ:
            raise RuntimeError("PYTORCH_ENABLE_MPS_FALLBACK must be absent")
        torch.mps.synchronize()
        return
    raise ValueError(f"Unsupported benchmark device: {device}")


def _summary_ns(values: Sequence[int]) -> dict[str, int | float]:
    if not values:
        raise ValueError("Cannot summarize an empty timing sequence")
    checked = [_raw_nanoseconds(value, "raw_ns") for value in values]
    ordered = sorted(checked)

    def percentile(q: float) -> float:
        rank = (len(ordered) - 1) * q
        lower_index = math.floor(rank)
        upper_index = math.ceil(rank)
        lower = ordered[lower_index]
        upper = ordered[upper_index]
        return lower + (upper - lower) * (rank - lower_index)

    return {
        "count": len(checked),
        "minimum": min(checked),
        "maximum": max(checked),
        "mean": sum(checked) / len(checked),
        "p50": percentile(0.50),
        "p95": percentile(0.95),
    }


def _timing_component(values: Sequence[int]) -> dict[str, object]:
    raw = list(values)
    return {"raw_ns": raw, "summary_ns": _summary_ns(raw)}


def _methodology() -> dict[str, object]:
    return {
        "clock": "time.perf_counter_ns",
        "end_to_end_measurement_pass": "separate_uninterrupted_complete_pipeline",
        "methodology_id": _METHODOLOGY_ID,
        "methodology_version": 2,
        "percentile_method": "linear interpolation at rank (n - 1) * q",
        "repeat_count": _REPEAT_COUNT,
        "stage_inclusion_boundaries": {
            "anomaly_map_reconstruction": (
                "From before reconstruct_anomaly_maps() through raw 256 x 256 map "
                "materialization, allocation, and included backend completion "
                "synchronization; excludes retrieval, masks, metrics, and persistence."
            ),
            "bank_transfer_and_device_setup": (
                "From before memory_bank.to(device) on an already selected and initialized "
                "backend through destination allocation, copy, and included backend "
                "completion synchronization; excludes device validation, context "
                "initialization, model load, bank construction, and repeats."
            ),
            "canonical_image_preprocessing": (
                "From before preprocess_decoded_image() through resize, tensor conversion, "
                "FP32 normalization, and output allocation; excludes decode, masks, and "
                "transfer."
            ),
            "exact_chunked_retrieval": (
                "From before exact_top1_squared_l2() through chunked search, stable merge, "
                "result reshape, image maximum, allocations, and included backend "
                "completion synchronization; excludes bank transfer and map reconstruction."
            ),
            "frozen_feature_extraction": (
                "From before extract_patch_embeddings() through frozen ResNet layer2 "
                "inference, local pooling, row-major patch layout, output allocation, and "
                "included backend completion synchronization; excludes input transfer and "
                "retrieval."
            ),
            "full_nominal_bank_build": (
                "From before the first ordered train/good decode through per-image "
                "preprocessing, transfer, feature extraction, CPU copy, contiguous FP32 "
                "concatenation, allocations, and included final backend completion "
                "synchronization; excludes model load, full-bank transfer, and test scoring."
            ),
            "host_to_device_transfer": (
                "From before image.unsqueeze(0).to(device) through the batch view, "
                "destination allocation, copy, and included backend completion "
                "synchronization; the CPU path may be a no-copy operation; excludes decode "
                "and preprocessing."
            ),
            "image_decode": (
                "From before decode_image() through file validation, Pillow decode, RGB "
                "conversion, and decoded-image allocation; excludes resize, tensor "
                "conversion, masks, and image close."
            ),
            "model_and_weight_load": (
                "From before cached weight resolution and model construction through cached "
                "read, frozen extractor construction, requested-device placement, eval "
                "mode, allocations, and included final backend completion synchronization; "
                "excludes imports, config and repository capture, downloads, and bank work."
            ),
            "synchronized_end_to_end": (
                "From before image decode through completed raw anomaly-map materialization "
                "and included final backend synchronization, with no internal stage "
                "synchronizations; excludes model load, bank construction and transfer, "
                "masks, metrics, image close, serialization, persistence, and console output."
            ),
        },
        "stage_measurement_pass": "segmented_complete_pipeline",
        "synchronization_policy": {
            "cpu": (
                "No accelerator synchronization; time.perf_counter_ns surrounds synchronous "
                "host work."
            ),
            "cuda": (
                "Call torch.cuda.synchronize(requested_device) immediately before each "
                "accelerator or complete-pass boundary outside the timer, then call it "
                "after submitted work inside the timer before the end timestamp."
            ),
            "mps": (
                "Call torch.mps.synchronize() immediately before each accelerator or "
                "complete-pass boundary outside the timer, then call it after submitted "
                "work inside the timer before the end timestamp."
            ),
        },
        "timing_unit": "nanoseconds",
        "warmup_count": _WARMUP_COUNT,
        "warmup_samples_in_statistics": False,
    }


def _workload() -> dict[str, object]:
    return {
        "D": 512,
        "M": 214_016,
        "Q": 1024,
        "bank_bytes": 438_304_768,
        "bank_chunk_size": 16_384,
        "bank_shape": [214_016, 512],
        "batch_size": 1,
        "dtype": "float32",
        "k": 1,
        "tensor_layout": {
            "anomaly_map": "BHW contiguous row-major",
            "image": "NCHW contiguous",
            "memory_bank": "MD contiguous row-major",
            "patch_embeddings": "BQD contiguous row-major",
        },
        "test_sample_count": 83,
        "training_sample_count": 209,
    }


def _environment(device: torch.device) -> dict[str, object]:
    if device.type == "cpu":
        return {"kind": "cpu", "properties": {}}
    if device.type == "cuda":
        return {
            "kind": "cuda",
            "properties": {
                "available": True,
                "device_index": device.index,
                "compute_capability": list(torch.cuda.get_device_capability(device)),
                "device_name": torch.cuda.get_device_name(device),
                "pytorch_cuda_runtime_version": torch.version.cuda,
            },
        }
    return {
        "kind": "mps",
        "properties": {
            "available": True,
            "built": True,
            "pytorch_enable_mps_fallback": "unset",
        },
    }


def _memory_observations(
    device: torch.device, mps_observations: Sequence[Mapping[str, object]]
) -> dict[str, object]:
    if device.type == "cpu":
        return {"kind": "cpu", "host_peak_memory": "not_measured"}
    if device.type == "cuda":
        return {
            "kind": "cuda",
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            "peak_window": "after_warmups_through_all_measured_passes",
        }
    return {
        "kind": "mps",
        "observations": list(mps_observations),
        "peak_memory": "not_available_in_selected_pytorch_api",
        "recommended_max_memory_bytes": int(torch.mps.recommended_max_memory()),
    }


def _mps_memory_observation(boundary: str) -> dict[str, object]:
    return {
        "boundary": boundary,
        "current_allocated_bytes": int(torch.mps.current_allocated_memory()),
        "driver_allocated_bytes": int(torch.mps.driver_allocated_memory()),
    }


def _validate_workload(value: object) -> None:
    if _thaw_json(value) != _workload():
        raise ValueError("workload must match the frozen bottle timing workload")


def _validate_methodology(value: object) -> None:
    if _thaw_json(value) != _methodology():
        raise ValueError("methodology must match synchronized wall-clock v2")


def _validate_environment(device_name: str, value: object) -> str:
    environment = _mapping(value, "environment")
    _keys(environment, {"kind", "properties"}, "environment")
    kind = environment["kind"]
    properties = _mapping(environment["properties"], "environment.properties")
    if device_name == "cpu" and kind == "cpu":
        _keys(properties, set(), "environment.properties")
        return "cpu"
    if device_name == "mps" and kind == "mps":
        expected = {
            "available": True,
            "built": True,
            "pytorch_enable_mps_fallback": "unset",
        }
        if _thaw_json(properties) != expected:
            raise ValueError("MPS environment properties are invalid")
        return "mps"
    if type(device_name) is str and device_name.startswith("cuda:") and kind == "cuda":
        suffix = device_name.removeprefix("cuda:")
        if not suffix.isdecimal():
            raise ValueError("CUDA benchmark device must include an explicit index")
        _keys(
            properties,
            {
                "available",
                "device_index",
                "compute_capability",
                "device_name",
                "pytorch_cuda_runtime_version",
            },
            "environment.properties",
        )
        index = properties["device_index"]
        if properties["available"] is not True or type(index) is not int:
            raise ValueError("CUDA environment availability and index are invalid")
        if index < 0 or index != int(suffix):
            raise ValueError("CUDA environment device index does not match device")
        capability = properties["compute_capability"]
        if (
            type(capability) not in {list, tuple}
            or len(capability) != 2
            or any(type(item) is not int or item < 0 for item in capability)
        ):
            raise ValueError("CUDA compute capability is invalid")
        for name in ("device_name", "pytorch_cuda_runtime_version"):
            if type(properties[name]) is not str or not properties[name]:
                raise ValueError(f"CUDA {name} must be a nonempty string")
        return "cuda"
    raise ValueError("Benchmark device and environment backend must agree")


def _validate_results(value: object, backend: str) -> None:
    results = _mapping(value, "results")
    _keys(
        results,
        {
            "one_off",
            "repeated_stages",
            "synchronized_end_to_end",
            "memory_observations",
        },
        "results",
    )
    one_off = _mapping(results["one_off"], "results.one_off")
    _keys(one_off, set(_ONE_OFF_STAGES), "results.one_off")
    for name in _ONE_OFF_STAGES:
        _validate_timing_component(one_off[name], 1, f"results.one_off.{name}")
    repeated = _mapping(results["repeated_stages"], "results.repeated_stages")
    _keys(repeated, set(_REPEATED_STAGES), "results.repeated_stages")
    for name in _REPEATED_STAGES:
        _validate_timing_component(
            repeated[name], _REPEAT_COUNT, f"results.repeated_stages.{name}"
        )
    _validate_timing_component(
        results["synchronized_end_to_end"],
        _REPEAT_COUNT,
        "results.synchronized_end_to_end",
    )
    _validate_memory_observations(results["memory_observations"], backend)


def _validate_timing_component(value: object, count: int, name: str) -> None:
    component = _mapping(value, name)
    _keys(component, {"raw_ns", "summary_ns"}, name)
    raw = component["raw_ns"]
    if type(raw) not in {list, tuple} or len(raw) != count:
        raise ValueError(f"{name}.raw_ns must contain exactly {count} values")
    checked = [_raw_nanoseconds(item, f"{name}.raw_ns") for item in raw]
    summary = _mapping(component["summary_ns"], f"{name}.summary_ns")
    _keys(summary, _SUMMARY_FIELDS, f"{name}.summary_ns")
    expected = _summary_ns(checked)
    for field, expected_value in expected.items():
        actual = summary[field]
        if type(actual) is not type(expected_value) or actual != expected_value:
            raise ValueError(f"{name}.summary_ns.{field} does not match raw_ns")


def _validate_memory_observations(value: object, backend: str) -> None:
    memory = _mapping(value, "results.memory_observations")
    if memory.get("kind") != backend:
        raise ValueError("Memory observations must match the timing backend")
    if backend == "cpu":
        if _thaw_json(memory) != {
            "kind": "cpu",
            "host_peak_memory": "not_measured",
        }:
            raise ValueError("CPU memory observations are invalid")
        return
    if backend == "cuda":
        _keys(
            memory,
            {
                "kind",
                "peak_allocated_bytes",
                "peak_reserved_bytes",
                "peak_window",
            },
            "results.memory_observations",
        )
        for name in ("peak_allocated_bytes", "peak_reserved_bytes"):
            if type(memory[name]) is not int or memory[name] <= 0:
                raise ValueError(f"CUDA {name} must be a positive integer")
        if memory["peak_window"] != "after_warmups_through_all_measured_passes":
            raise ValueError("CUDA peak window is invalid")
        return
    _keys(
        memory,
        {"kind", "observations", "peak_memory", "recommended_max_memory_bytes"},
        "results.memory_observations",
    )
    observations = memory["observations"]
    if type(observations) not in {list, tuple} or len(observations) != 3:
        raise ValueError("MPS memory observations must contain three points")
    expected_boundaries = ("after_setup", "after_warmups", "after_measured_passes")
    for observation, boundary in zip(observations, expected_boundaries, strict=True):
        point = _mapping(observation, "MPS memory observation")
        _keys(
            point,
            {"boundary", "current_allocated_bytes", "driver_allocated_bytes"},
            "MPS memory observation",
        )
        if point["boundary"] != boundary:
            raise ValueError("MPS memory observation boundaries are out of order")
        for name in ("current_allocated_bytes", "driver_allocated_bytes"):
            if type(point[name]) is not int or point[name] < 0:
                raise ValueError(f"MPS {name} must be a nonnegative integer")
    if memory["peak_memory"] != "not_available_in_selected_pytorch_api":
        raise ValueError("MPS memory must not be labeled as peak memory")
    recommended = memory["recommended_max_memory_bytes"]
    if type(recommended) is not int or recommended <= 0:
        raise ValueError("MPS recommended memory must be a positive integer")


def _raw_nanoseconds(value: object, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} values must be ordinary Python integers")
    if value < 0 or value > _MAX_NANOSECONDS:
        raise ValueError(f"{name} values must be nonnegative signed-64 integers")
    return value


def _elapsed_nanoseconds(start: int, end: int) -> int:
    return _raw_nanoseconds(end - start, "elapsed nanoseconds")


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _keys(value: Mapping[str, object], expected: set[str], name: str) -> None:
    if set(value) != expected or any(type(key) is not str for key in value):
        raise ValueError(f"{name} fields are invalid")


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
            if type(key) is not str:
                raise TypeError("Benchmark JSON object keys must be strings")
            result[key] = _freeze_json(item)
        return MappingProxyType(result)
    if type(value) in {list, tuple}:
        return tuple(_freeze_json(item) for item in value)
    if value is None or type(value) in {str, int, bool}:
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise TypeError("Benchmark values must be finite JSON-compatible Python primitives")


def _thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if type(value) in {list, tuple}:
        return [_thaw_json(item) for item in value]
    return value
