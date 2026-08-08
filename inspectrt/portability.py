"""Strict loading of frozen schema-1 baseline run bundles."""

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
import hashlib
import ipaddress
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tempfile
from types import MappingProxyType
from typing import BinaryIO, Literal

import torch
from torch import Tensor

from inspectrt.artifacts import _canonical_json, _rename_without_overwrite
from inspectrt.benchmark import _methodology
from inspectrt.data import MvtecSample
from inspectrt.metrics import (
    ThresholdFreeMetrics,
    compute_threshold_free_metrics,
)

__all__ = (
    "BundleMetrics",
    "BundleValidationError",
    "CandidateComparability",
    "CandidateScientificResult",
    "CanonicalInputIdentity",
    "ComparableBundle",
    "ComparisonValidationError",
    "DiscreteComponentComparison",
    "FloatingComponentComparison",
    "FloatingStatistics",
    "IndexMismatch",
    "MemoryBankMetadata",
    "MetricDelta",
    "PolicyDerivation",
    "PolicyTolerance",
    "PortabilityEnvironmentDescriptor",
    "PortabilityEnvironmentMap",
    "PortabilityPerformance",
    "PortabilityPerformanceExclusion",
    "PortabilityPerformanceRun",
    "PortabilityPolicy",
    "PortabilityPolicyIdentity",
    "PredictionRecord",
    "ScientificBundleDescriptor",
    "ScientificComparison",
    "ScientificExecutionAttempt",
    "ScientificGenerator",
    "ScientificRunIdentity",
    "SourceFileSnapshot",
    "build_portability_performance",
    "compare_scientific_bundles",
    "encode_portability_performance",
    "encode_scientific_comparison",
    "load_comparable_bundle",
    "load_portability_environment_map",
    "load_portability_policy",
    "publish_portability_comparison",
    "publish_portability_records",
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
_SCIENTIFIC_SOURCE_COMMIT = "bc330b9070c5ca8db9cb7cfbb27617256388536b"
_ACCEPTED_LOCK_SHA256 = (
    "ddaddc99b318a1c3a04d5d7cc433cf736d321b56f98a8ae8b532e71e19e6d76b"
)
_ACCEPTED_WEIGHT_SHA256 = (
    "11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca"
)
_SCHEMA_ID = "inspectrt_portability_comparison_v1"
_MILESTONE_ID = "inspectrt_cross_platform_evidence_v1"
_ENVIRONMENT_MAP_SCHEMA_ID = "inspectrt_portability_environment_map_v1"
_POLICY_SCHEMA_ID = "inspectrt_portability_policy_v1"
_PERFORMANCE_SCHEMA_ID = "inspectrt_portability_performance_v1"
_FLOAT_CHUNK_SIZE = 65_536
_INDEX_MISMATCH_LIMIT = 16
_FLOATING_COMPONENTS = (
    "memory_bank",
    "patch_distances",
    "image_scores",
    "anomaly_maps",
)
_SCIENTIFIC_METRICS = (
    "image_auroc",
    "image_average_precision",
    "pixel_auroc",
)
_IMAGE_SCORE_SEMANTICS = "maximum patch distance"
_POLICY_ROLES = {
    "reference",
    "same_stack_control",
    "calibration",
    "holdout",
    "post_policy_attempt",
}
_CANDIDATE_ROLES = _POLICY_ROLES - {"reference"}
_EXECUTION_LAYERS = {"native", "wsl2"}
_ATTEMPT_STATUSES = {"unsupported", "execution_failed"}
_CANDIDATE_STATUSES = {
    "structurally_incomparable",
    "observed_unclassified",
    "within_policy",
    "drift_detected",
}
_ENVIRONMENT_ID = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?")
_MACHINE_CODE = re.compile(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)*")
_MACHINE_ID = re.compile(r"[a-z0-9](?:[a-z0-9_-]{0,62}[a-z0-9])?")
_REQUESTED_DEVICE = re.compile(r"(?:cpu|mps|cuda:[0-9]+)")
_PROVENANCE_REQUIREMENTS = (
    "clean_source",
    "scientific_source_commit",
    "lock_identity",
    "weight_identity",
    "inventory_identity",
)
_DISCRETE_REQUIREMENTS = (
    "test_sample_ids",
    "test_labels",
    "evaluation_masks",
    "nearest_bank_indices",
)
_SCIENTIFIC_GATE_NAMES = (
    "run_schema",
    "profile",
    "category",
    "preprocessing",
    "feature_contract",
    "weight_identity",
    "configuration",
    "lock_identity",
    "clean_source",
    "scientific_source_commit",
    "inventory_identity",
    "samples_source",
    "ordered_sample_ids",
    "ordered_sample_metadata",
    "ordered_training_ids",
    "ordered_test_sample_ids",
    "ordered_labels",
    "sample_counts",
    "memory_bank_contract",
    "patch_distance_contract",
    "image_score_contract",
    "nearest_index_contract",
    "test_label_contract",
    "anomaly_map_contract",
    "mask_contract",
    "retrieval_semantics",
    "image_score_semantics",
    "anomaly_map_semantics",
    "metric_fields",
)
_MAP_INTERPOLATION = {
    "align_corners": False,
    "input_size": [32, 32],
    "mode": "bilinear",
    "output_size": [256, 256],
    "values": "raw squared-L2 patch distances",
}


class BundleValidationError(ValueError):
    """A comparable bundle violates the frozen schema-1 contract."""


class ComparisonValidationError(BundleValidationError):
    """A scientific comparison input or record is invalid."""


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


@dataclass(frozen=True, slots=True)
class CanonicalInputIdentity:
    """Identity of one exact canonical JSON input."""

    byte_count: int
    sha256: str

    def __post_init__(self) -> None:
        if type(self.byte_count) is not int or self.byte_count <= 0:
            raise ComparisonValidationError(
                "canonical input byte_count must be positive"
            )
        _comparison_sha256(self.sha256, "canonical input SHA-256")


@dataclass(frozen=True, slots=True)
class PortabilityEnvironmentDescriptor:
    """Sanitized public identity bound positionally to one run bundle."""

    environment_id: str
    policy_role: Literal[
        "reference",
        "same_stack_control",
        "calibration",
        "holdout",
        "post_policy_attempt",
    ]
    os_label: str
    execution_layer: Literal["native", "wsl2"]
    hardware_label: str
    requested_device: str

    def __post_init__(self) -> None:
        _validate_descriptor_identity(self)


@dataclass(frozen=True, slots=True)
class PortabilityEnvironmentMap:
    """Strict ordered schema-1 mapping from CLI runs to public identities."""

    schema_version: int
    schema_id: str
    reference: PortabilityEnvironmentDescriptor
    candidates: tuple[PortabilityEnvironmentDescriptor, ...]
    attempts: tuple["ScientificExecutionAttempt", ...]

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != 1
            or type(self.schema_id) is not str
            or self.schema_id != _ENVIRONMENT_MAP_SCHEMA_ID
        ):
            raise ComparisonValidationError("environment map identity is invalid")
        if (
            type(self.reference) is not PortabilityEnvironmentDescriptor
            or self.reference.policy_role != "reference"
            or type(self.candidates) is not tuple
            or not self.candidates
            or any(
                type(candidate) is not PortabilityEnvironmentDescriptor
                or candidate.policy_role not in _CANDIDATE_ROLES
                for candidate in self.candidates
            )
            or type(self.attempts) is not tuple
            or any(
                type(attempt) is not ScientificExecutionAttempt
                for attempt in self.attempts
            )
        ):
            raise ComparisonValidationError("environment map run roles are invalid")
        environment_ids = (
            self.reference.environment_id,
            *(candidate.environment_id for candidate in self.candidates),
            *(attempt.environment_id for attempt in self.attempts),
        )
        if len(environment_ids) != len(set(environment_ids)):
            raise ComparisonValidationError(
                "environment map IDs must be unique across bundles and attempts"
            )


@dataclass(frozen=True, slots=True)
class ScientificBundleDescriptor:
    """One loaded bundle plus caller-supplied sanitized public identity."""

    bundle: ComparableBundle
    environment_id: str
    policy_role: Literal[
        "reference",
        "same_stack_control",
        "calibration",
        "holdout",
        "post_policy_attempt",
    ]
    os_label: str
    execution_layer: Literal["native", "wsl2"]
    hardware_label: str
    requested_device: str

    def __post_init__(self) -> None:
        if type(self.bundle) is not ComparableBundle:
            raise ComparisonValidationError(
                "bundle descriptor must contain a ComparableBundle"
            )
        _validate_descriptor_identity(self)


@dataclass(frozen=True, slots=True)
class ScientificExecutionAttempt:
    """Sanitized non-gating execution outcome that produced no bundle."""

    environment_id: str
    status: Literal["unsupported", "execution_failed"]
    reason_code: str
    stage_code: str

    def __post_init__(self) -> None:
        _environment_id(self.environment_id, "attempt.environment_id")
        if type(self.status) is not str or self.status not in _ATTEMPT_STATUSES:
            raise ComparisonValidationError("attempt status is invalid")
        _machine_code(self.reason_code, "attempt.reason_code")
        _machine_code(self.stage_code, "attempt.stage_code")


@dataclass(frozen=True, slots=True)
class ScientificGenerator:
    """Explicit sanitized identity of the comparison implementation."""

    source_commit: str
    dirty: bool

    def __post_init__(self) -> None:
        if type(self.source_commit) is not str or not _COMMIT.fullmatch(
            self.source_commit
        ):
            raise ComparisonValidationError(
                "generator source_commit must be a full lowercase commit hash"
            )
        if type(self.dirty) is not bool:
            raise ComparisonValidationError("generator dirty must be a boolean")


@dataclass(frozen=True, slots=True)
class PolicyTolerance:
    """Explicit elementwise absolute and relative policy limits."""

    atol: int | float
    rtol: int | float

    def __post_init__(self) -> None:
        _nonnegative_policy_number(self.atol, "policy tolerance atol")
        _nonnegative_policy_number(self.rtol, "policy tolerance rtol")


@dataclass(frozen=True, slots=True)
class PolicyDerivation:
    """Machine-readable record identifying reviewed derivation evidence."""

    method_id: str
    comparison_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _bounded_machine_id(self.method_id, "policy derivation method_id")
        if (
            type(self.comparison_ids) is not tuple
            or not self.comparison_ids
            or len(self.comparison_ids) != len(set(self.comparison_ids))
        ):
            raise ComparisonValidationError(
                "policy derivation comparison_ids must be a nonempty unique array"
            )
        for value in self.comparison_ids:
            _comparison_sha256(value, "policy derivation comparison ID")


@dataclass(frozen=True, slots=True)
class PortabilityPolicy:
    """Strict reviewed schema-1 policy plus its exact source-byte identity."""

    schema_version: int
    schema_id: str
    policy_id: str
    profile_id: str
    category: str
    reference_environment_id: str
    calibration_environment_ids: tuple[str, ...]
    holdout_environment_ids: tuple[str, ...]
    provenance_requirements: tuple[str, ...]
    discrete_output_requirements: tuple[str, ...]
    floating_component_limits: Mapping[str, PolicyTolerance]
    metric_absolute_delta_limits: Mapping[str, int | float]
    derivation: PolicyDerivation
    reviewed_evidence_hashes: tuple[str, ...]
    limitation: str
    source: CanonicalInputIdentity

    def __post_init__(self) -> None:
        if isinstance(self.floating_component_limits, Mapping):
            object.__setattr__(
                self,
                "floating_component_limits",
                MappingProxyType(dict(self.floating_component_limits)),
            )
        if isinstance(self.metric_absolute_delta_limits, Mapping):
            object.__setattr__(
                self,
                "metric_absolute_delta_limits",
                MappingProxyType(dict(self.metric_absolute_delta_limits)),
            )
        _validate_portability_policy(self)


@dataclass(frozen=True, slots=True)
class PortabilityPolicyIdentity:
    """Reviewed policy identity exposed by policy-mode scientific output."""

    policy_id: str
    sha256: str

    def __post_init__(self) -> None:
        _bounded_machine_id(self.policy_id, "policy identity policy_id")
        _comparison_sha256(self.sha256, "policy identity SHA-256")


@dataclass(frozen=True, slots=True)
class ScientificRunIdentity:
    """Sanitized portable identity projected from one loaded bundle."""

    environment_id: str
    policy_role: str
    bundle_kind: Literal["evaluation", "benchmark"]
    os_label: str
    execution_layer: Literal["native", "wsl2"]
    hardware_label: str
    requested_device: str
    run: Mapping[str, object]
    source_files: tuple[SourceFileSnapshot, ...]
    benchmark_workload: Mapping[str, object] | None
    benchmark_methodology: Mapping[str, object] | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "run", _freeze_comparison_json(self.run))
        if self.benchmark_workload is not None:
            object.__setattr__(
                self,
                "benchmark_workload",
                _freeze_comparison_json(self.benchmark_workload),
            )
        if self.benchmark_methodology is not None:
            object.__setattr__(
                self,
                "benchmark_methodology",
                _freeze_comparison_json(self.benchmark_methodology),
            )
        _validate_scientific_run_identity(self)


@dataclass(frozen=True, slots=True)
class CandidateComparability:
    """Every exact scientific compatibility gate for one candidate."""

    environment_id: str
    comparable: bool
    gates: tuple[tuple[str, bool], ...]
    structural_components: tuple["DiscreteComponentComparison", ...]


@dataclass(frozen=True, slots=True)
class FloatingStatistics:
    """Chunked drift.

    ``maximum_relative_error`` is ``abs(candidate-reference) / abs(reference)``
    over nonzero reference values. Zero references are counted and excluded;
    the field is ``None`` when every reference value is zero.
    """

    element_count: int
    exact_count: int
    differing_count: int
    maximum_absolute_error: float
    mean_absolute_error: float
    root_mean_square_error: float
    maximum_relative_error: float | None
    zero_reference_count: int
    policy_violation_count: int | None = None


@dataclass(frozen=True, slots=True)
class FloatingComponentComparison:
    """Named floating artifact comparison."""

    name: str
    statistics: FloatingStatistics


@dataclass(frozen=True, slots=True)
class IndexMismatch:
    """One bounded row-major nearest-index mismatch."""

    coordinate: tuple[int, int]
    reference_value: int
    candidate_value: int


@dataclass(frozen=True, slots=True)
class DiscreteComponentComparison:
    """Named exact discrete comparison and optional bounded index diagnostics."""

    name: str
    exact: bool
    element_count: int
    exact_count: int
    mismatch_count: int
    mismatch_rate: float
    first_mismatches: tuple[IndexMismatch, ...] = ()


@dataclass(frozen=True, slots=True)
class MetricDelta:
    """One frozen metric observation in contract order."""

    metric_name: str
    reference_value: float
    candidate_value: float
    absolute_delta: float


@dataclass(frozen=True, slots=True)
class CandidateScientificResult:
    """Scientific result for one completed candidate bundle."""

    environment_id: str
    status: Literal[
        "structurally_incomparable",
        "observed_unclassified",
        "within_policy",
        "drift_detected",
    ]
    floating_components: tuple[FloatingComponentComparison, ...] | None
    discrete_components: tuple[DiscreteComponentComparison, ...] | None
    metrics: tuple[MetricDelta, ...] | None


@dataclass(frozen=True, slots=True)
class ScientificComparison:
    """Immutable schema-1 scientific comparison record."""

    schema_version: int
    schema_id: str
    milestone_id: str
    comparison_id: str
    generator: ScientificGenerator
    reference: ScientificRunIdentity
    candidates: tuple[ScientificRunIdentity, ...]
    attempts: tuple[ScientificExecutionAttempt, ...]
    comparability: tuple[CandidateComparability, ...]
    scientific_results: tuple[CandidateScientificResult, ...]
    limitations: tuple[str, ...]
    policy: PortabilityPolicyIdentity | None = None


@dataclass(frozen=True, slots=True)
class PortabilityPerformanceRun:
    """One timing-eligible run with benchmark observations copied verbatim."""

    environment_id: str
    os_label: str
    execution_layer: Literal["native", "wsl2"]
    hardware_label: str
    requested_device: str
    run_id: str
    benchmark_sample_id: str
    timing_methodology: Mapping[str, object]
    measurements: Mapping[str, object]

    def __post_init__(self) -> None:
        _environment_id(self.environment_id, "performance run environment_id")
        _public_label(self.os_label, "performance run os_label")
        if self.execution_layer not in _EXECUTION_LAYERS:
            raise ComparisonValidationError(
                "performance run execution_layer is invalid"
            )
        _public_label(self.hardware_label, "performance run hardware_label")
        if not _timing_device(self.requested_device):
            raise ComparisonValidationError(
                "performance run requested_device is not timing-valid"
            )
        run_id = _comparison_string(self.run_id, "performance run run_id")
        if not _RUN_ID.fullmatch(run_id):
            raise ComparisonValidationError("performance run run_id is invalid")
        _portable_identity_string(
            self.benchmark_sample_id, "performance run benchmark_sample_id"
        )
        object.__setattr__(
            self,
            "timing_methodology",
            _freeze_comparison_json(self.timing_methodology),
        )
        object.__setattr__(
            self, "measurements", _freeze_comparison_json(self.measurements)
        )


@dataclass(frozen=True, slots=True)
class PortabilityPerformanceExclusion:
    """One completed scientific candidate excluded from the timing matrix."""

    environment_id: str
    reason_code: Literal[
        "evaluation_bundle",
        "unsupported_timing_device",
        "workload_mismatch",
        "methodology_mismatch",
    ]

    def __post_init__(self) -> None:
        _environment_id(self.environment_id, "performance exclusion environment_id")
        if self.reason_code not in {
            "evaluation_bundle",
            "unsupported_timing_device",
            "workload_mismatch",
            "methodology_mismatch",
        }:
            raise ComparisonValidationError(
                "performance exclusion reason_code is invalid"
            )


@dataclass(frozen=True, slots=True)
class PortabilityPerformance:
    """Immutable descriptive-only schema-1 performance record."""

    schema_version: int
    schema_id: str
    milestone_id: str
    status: Literal["descriptive_only"]
    comparison_id: str
    scientific_sha256: str
    generator: ScientificGenerator
    workload: Mapping[str, object]
    timing_methodology: Mapping[str, object]
    included_runs: tuple[PortabilityPerformanceRun, ...]
    excluded_candidates: tuple[PortabilityPerformanceExclusion, ...]
    attempts: tuple[ScientificExecutionAttempt, ...]
    limitations: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "workload", _freeze_comparison_json(self.workload))
        object.__setattr__(
            self,
            "timing_methodology",
            _freeze_comparison_json(self.timing_methodology),
        )
        _validate_portability_performance(self)


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


def load_portability_environment_map(path: Path) -> PortabilityEnvironmentMap:
    """Load one exact canonical sanitized environment map."""
    _, value = _load_canonical_input(path, "environment map")
    _comparison_keys(
        value,
        {"schema_version", "schema_id", "reference", "candidates", "attempts"},
        "environment map",
    )
    if (
        type(value["schema_version"]) is not int
        or value["schema_version"] != 1
        or type(value["schema_id"]) is not str
        or value["schema_id"] != _ENVIRONMENT_MAP_SCHEMA_ID
    ):
        raise ComparisonValidationError("environment map identity is invalid")
    reference = _environment_descriptor_record(value["reference"], "reference")
    candidate_values = _comparison_list(value["candidates"], "candidates")
    if not candidate_values:
        raise ComparisonValidationError(
            "environment map must contain at least one candidate"
        )
    candidates = tuple(
        _environment_descriptor_record(candidate, f"candidates[{index}]")
        for index, candidate in enumerate(candidate_values)
    )
    attempt_values = _comparison_list(value["attempts"], "attempts")
    attempts = tuple(
        _execution_attempt_record(attempt, f"attempts[{index}]")
        for index, attempt in enumerate(attempt_values)
    )
    return PortabilityEnvironmentMap(
        schema_version=1,
        schema_id=_ENVIRONMENT_MAP_SCHEMA_ID,
        reference=reference,
        candidates=candidates,
        attempts=attempts,
    )


def load_portability_policy(path: Path) -> PortabilityPolicy:
    """Load one exact canonical reviewed portability policy without tuning it."""
    payload, value = _load_canonical_input(path, "portability policy")
    _comparison_keys(
        value,
        {
            "schema_version",
            "schema_id",
            "policy_id",
            "profile_id",
            "category",
            "reference_environment_id",
            "calibration_environment_ids",
            "holdout_environment_ids",
            "provenance_requirements",
            "discrete_output_requirements",
            "floating_component_limits",
            "metric_absolute_delta_limits",
            "derivation",
            "reviewed_evidence_hashes",
            "limitation",
        },
        "portability policy",
    )
    if (
        type(value["schema_version"]) is not int
        or value["schema_version"] != 1
        or type(value["schema_id"]) is not str
        or value["schema_id"] != _POLICY_SCHEMA_ID
    ):
        raise ComparisonValidationError("portability policy identity is invalid")

    floating_values = _comparison_dict(
        value["floating_component_limits"], "floating_component_limits"
    )
    _comparison_keys(
        floating_values, set(_FLOATING_COMPONENTS), "floating_component_limits"
    )
    floating_limits = {}
    for name in _FLOATING_COMPONENTS:
        tolerance = _comparison_dict(
            floating_values[name], f"floating_component_limits.{name}"
        )
        _comparison_keys(
            tolerance, {"atol", "rtol"}, f"floating_component_limits.{name}"
        )
        floating_limits[name] = PolicyTolerance(tolerance["atol"], tolerance["rtol"])

    metric_values = _comparison_dict(
        value["metric_absolute_delta_limits"], "metric_absolute_delta_limits"
    )
    _comparison_keys(
        metric_values, set(_SCIENTIFIC_METRICS), "metric_absolute_delta_limits"
    )
    metric_limits = {
        name: _nonnegative_policy_number(
            metric_values[name], f"metric_absolute_delta_limits.{name}"
        )
        for name in _SCIENTIFIC_METRICS
    }

    derivation_value = _comparison_dict(value["derivation"], "derivation")
    _comparison_keys(derivation_value, {"method_id", "comparison_ids"}, "derivation")
    derivation = PolicyDerivation(
        _comparison_string(derivation_value["method_id"], "derivation.method_id"),
        tuple(
            _comparison_string(item, "derivation.comparison_ids item")
            for item in _comparison_list(
                derivation_value["comparison_ids"], "derivation.comparison_ids"
            )
        ),
    )
    return PortabilityPolicy(
        schema_version=1,
        schema_id=_POLICY_SCHEMA_ID,
        policy_id=_comparison_string(value["policy_id"], "policy_id"),
        profile_id=_comparison_string(value["profile_id"], "profile_id"),
        category=_comparison_string(value["category"], "category"),
        reference_environment_id=_comparison_string(
            value["reference_environment_id"], "reference_environment_id"
        ),
        calibration_environment_ids=_environment_id_array(
            value["calibration_environment_ids"], "calibration_environment_ids"
        ),
        holdout_environment_ids=_environment_id_array(
            value["holdout_environment_ids"], "holdout_environment_ids"
        ),
        provenance_requirements=_exact_requirement_array(
            value["provenance_requirements"],
            _PROVENANCE_REQUIREMENTS,
            "provenance_requirements",
        ),
        discrete_output_requirements=_exact_requirement_array(
            value["discrete_output_requirements"],
            _DISCRETE_REQUIREMENTS,
            "discrete_output_requirements",
        ),
        floating_component_limits=MappingProxyType(floating_limits),
        metric_absolute_delta_limits=MappingProxyType(metric_limits),
        derivation=derivation,
        reviewed_evidence_hashes=tuple(
            _comparison_string(item, "reviewed_evidence_hashes item")
            for item in _comparison_list(
                value["reviewed_evidence_hashes"], "reviewed_evidence_hashes"
            )
        ),
        limitation=_comparison_string(value["limitation"], "limitation"),
        source=CanonicalInputIdentity(
            len(payload), hashlib.sha256(payload).hexdigest()
        ),
    )


def compare_scientific_bundles(
    reference: ScientificBundleDescriptor,
    candidates: Sequence[ScientificBundleDescriptor],
    *,
    generator: ScientificGenerator,
    attempts: Sequence[ScientificExecutionAttempt] = (),
    policy: PortabilityPolicy | None = None,
) -> ScientificComparison:
    """Compare frozen bundles, applying only an explicitly supplied policy."""
    if type(reference) is not ScientificBundleDescriptor:
        raise ComparisonValidationError(
            "reference must be a ScientificBundleDescriptor"
        )
    if type(generator) is not ScientificGenerator:
        raise ComparisonValidationError("generator must be a ScientificGenerator")
    if policy is not None and type(policy) is not PortabilityPolicy:
        raise ComparisonValidationError("policy must be a PortabilityPolicy or None")
    if policy is not None:
        _validate_portability_policy(policy)
    candidate_records = _comparison_sequence(
        candidates, ScientificBundleDescriptor, "candidates"
    )
    if not candidate_records:
        raise ComparisonValidationError("at least one candidate is required")
    attempt_records = _comparison_sequence(
        attempts, ScientificExecutionAttempt, "attempts"
    )
    if reference.policy_role != "reference":
        raise ComparisonValidationError("reference policy_role must be reference")
    if any(
        candidate.policy_role not in _CANDIDATE_ROLES for candidate in candidate_records
    ):
        raise ComparisonValidationError("candidate policy_role must not be reference")

    environment_ids = (
        reference.environment_id,
        *(candidate.environment_id for candidate in candidate_records),
        *(attempt.environment_id for attempt in attempt_records),
    )
    if len(environment_ids) != len(set(environment_ids)):
        raise ComparisonValidationError(
            "environment IDs must be unique across bundles and attempts"
        )

    descriptors = (reference, *candidate_records)
    for descriptor in descriptors:
        _revalidate_comparable_bundle(descriptor.bundle)
        recorded_device = _comparison_string(
            _comparison_mapping(
                descriptor.bundle.run_metadata, "bundle.run_metadata"
            ).get("device"),
            "bundle.run_metadata.device",
        )
        if descriptor.requested_device != recorded_device:
            raise ComparisonValidationError(
                f"{descriptor.environment_id}: requested_device differs from the bundle"
            )
    if policy is not None:
        _validate_policy_scope(policy, reference, candidate_records)

    reference_identity = _scientific_run_identity(reference)
    candidate_identities = tuple(
        _scientific_run_identity(candidate) for candidate in candidate_records
    )
    identity_payload = {
        "attempts": [_attempt_value(attempt) for attempt in attempt_records],
        "candidates": [
            _run_identity_value(identity) for identity in candidate_identities
        ],
        "generator": _generator_value(generator),
        "milestone_id": _MILESTONE_ID,
        "reference": _run_identity_value(reference_identity),
        "schema_id": _SCHEMA_ID,
        "schema_version": 1,
    }
    if policy is not None:
        identity_payload["policy_sha256"] = policy.source.sha256
    # The ID is the full SHA-256 of this canonical identity payload; it excludes
    # itself and all measured results.
    comparison_id = hashlib.sha256(_canonical_json(identity_payload)).hexdigest()

    comparability = []
    results = []
    for candidate in candidate_records:
        gates = _scientific_gates(reference.bundle, candidate.bundle)
        comparable = all(result for _, result in gates)
        structural_components = (
            _structural_sequence_discrete(
                "test_sample_ids",
                reference.bundle.test_sample_ids,
                candidate.bundle.test_sample_ids,
            ),
            _structural_tensor_discrete(
                "test_labels",
                reference.bundle.test_labels,
                candidate.bundle.test_labels,
            ),
        )
        comparability.append(
            CandidateComparability(
                candidate.environment_id,
                comparable,
                gates,
                structural_components,
            )
        )
        if not comparable:
            results.append(
                CandidateScientificResult(
                    candidate.environment_id,
                    "structurally_incomparable",
                    None,
                    None,
                    None,
                )
            )
            continue

        floating = tuple(
            FloatingComponentComparison(
                name,
                _floating_statistics(
                    getattr(reference.bundle, name),
                    getattr(candidate.bundle, name),
                    name,
                    tolerance=(
                        policy.floating_component_limits[name]
                        if policy is not None
                        else None
                    ),
                ),
            )
            for name in _FLOATING_COMPONENTS
        )
        discrete = (
            _sequence_discrete(
                "test_sample_ids",
                reference.bundle.test_sample_ids,
                candidate.bundle.test_sample_ids,
            ),
            _tensor_discrete(
                "test_labels",
                reference.bundle.test_labels,
                candidate.bundle.test_labels,
            ),
            _tensor_discrete(
                "evaluation_masks",
                reference.bundle.evaluation_masks,
                candidate.bundle.evaluation_masks,
            ),
            _nearest_index_comparison(
                reference.bundle.nearest_bank_indices,
                candidate.bundle.nearest_bank_indices,
            ),
        )
        metrics = tuple(
            MetricDelta(
                metric_name=name,
                reference_value=getattr(reference.bundle.metrics, name),
                candidate_value=getattr(candidate.bundle.metrics, name),
                absolute_delta=abs(
                    getattr(candidate.bundle.metrics, name)
                    - getattr(reference.bundle.metrics, name)
                ),
            )
            for name in _SCIENTIFIC_METRICS
        )
        status = "observed_unclassified"
        if policy is not None:
            floating_drift = any(
                component.statistics.policy_violation_count != 0
                for component in floating
            )
            discrete_drift = any(not component.exact for component in discrete)
            metric_drift = any(
                metric.absolute_delta
                > policy.metric_absolute_delta_limits[metric.metric_name]
                for metric in metrics
            )
            status = (
                "drift_detected"
                if floating_drift or discrete_drift or metric_drift
                else "within_policy"
            )
        results.append(
            CandidateScientificResult(
                candidate.environment_id,
                status,
                floating,
                discrete,
                metrics,
            )
        )

    return ScientificComparison(
        schema_version=1,
        schema_id=_SCHEMA_ID,
        milestone_id=_MILESTONE_ID,
        comparison_id=comparison_id,
        generator=generator,
        reference=reference_identity,
        candidates=candidate_identities,
        attempts=attempt_records,
        comparability=tuple(comparability),
        scientific_results=tuple(results),
        limitations=(
            *(("observation_only_no_policy",) if policy is None else ()),
            "nominal_memory_bank_and_downstream_artifacts_only",
            "source_hashes_are_current_bundle_snapshots",
            *(("reviewed_observed_envelope_not_universal",) if policy else ()),
        ),
        policy=(
            PortabilityPolicyIdentity(policy.policy_id, policy.source.sha256)
            if policy is not None
            else None
        ),
    )


def encode_scientific_comparison(comparison: ScientificComparison) -> bytes:
    """Return canonical in-memory ``scientific.json`` UTF-8 bytes."""
    if type(comparison) is not ScientificComparison:
        raise ComparisonValidationError(
            "comparison must be a ScientificComparison record"
        )
    _validate_scientific_comparison(comparison)
    try:
        return _canonical_json(_scientific_comparison_value(comparison))
    except (TypeError, ValueError, OverflowError, RecursionError) as error:
        raise ComparisonValidationError(
            "scientific comparison cannot be canonically encoded"
        ) from error


def build_portability_performance(
    comparison: ScientificComparison,
    scientific_bytes: bytes,
    reference: ScientificBundleDescriptor,
    candidates: Sequence[ScientificBundleDescriptor],
) -> PortabilityPerformance:
    """Project validated benchmark observations without changing science."""
    if type(comparison) is not ScientificComparison:
        raise ComparisonValidationError("comparison must be a ScientificComparison")
    if type(scientific_bytes) is not bytes or scientific_bytes != (
        encode_scientific_comparison(comparison)
    ):
        raise ComparisonValidationError(
            "scientific_bytes must be the exact canonical comparison bytes"
        )
    if type(reference) is not ScientificBundleDescriptor:
        raise ComparisonValidationError(
            "reference must be a ScientificBundleDescriptor"
        )
    candidate_records = _comparison_sequence(
        candidates, ScientificBundleDescriptor, "performance candidates"
    )
    if (
        _scientific_run_identity(reference) != comparison.reference
        or tuple(_scientific_run_identity(candidate) for candidate in candidate_records)
        != comparison.candidates
    ):
        raise ComparisonValidationError(
            "performance descriptors differ from the scientific comparison"
        )
    if reference.bundle.kind != "benchmark":
        raise ComparisonValidationError(
            "reference run must be an eight-file benchmark bundle"
        )
    if not _timing_device(reference.requested_device):
        raise ComparisonValidationError(
            "reference requested device must be cpu or explicitly indexed CUDA"
        )

    reference_benchmark = _performance_benchmark(reference.bundle)
    workload = _comparison_mapping(
        reference_benchmark["workload"], "reference workload"
    )
    methodology = _comparison_mapping(
        reference_benchmark["methodology"], "reference methodology"
    )
    _validate_timing_methodology(methodology, reference.requested_device)
    included = [_performance_run(reference)]
    excluded = []
    for descriptor, candidate_identity in zip(
        candidate_records, comparison.candidates, strict=True
    ):
        reason = _performance_exclusion_reason(
            descriptor,
            comparison.reference,
            candidate_identity,
            workload,
            methodology,
        )
        if reason is None:
            included.append(_performance_run(descriptor))
        else:
            excluded.append(
                PortabilityPerformanceExclusion(descriptor.environment_id, reason)
            )

    return PortabilityPerformance(
        schema_version=1,
        schema_id=_PERFORMANCE_SCHEMA_ID,
        milestone_id=_MILESTONE_ID,
        status="descriptive_only",
        comparison_id=comparison.comparison_id,
        scientific_sha256=hashlib.sha256(scientific_bytes).hexdigest(),
        generator=comparison.generator,
        workload=workload,
        timing_methodology=_methodology_compatibility_identity(methodology),
        included_runs=tuple(included),
        excluded_candidates=tuple(excluded),
        attempts=comparison.attempts,
        limitations=(
            "benchmark_summaries_without_raw_repetitions",
            "host_conditions_are_uncontrolled",
            "absolute_observations_only_no_cross_machine_inference",
        ),
    )


def encode_portability_performance(performance: PortabilityPerformance) -> bytes:
    """Return canonical in-memory ``performance.json`` UTF-8 bytes."""
    if type(performance) is not PortabilityPerformance:
        raise ComparisonValidationError(
            "performance must be a PortabilityPerformance record"
        )
    _validate_portability_performance(performance)
    try:
        return _canonical_json(_portability_performance_value(performance))
    except (TypeError, ValueError, OverflowError, RecursionError) as error:
        raise ComparisonValidationError(
            "performance record cannot be canonically encoded"
        ) from error


def publish_portability_records(
    scientific_bytes: bytes, performance_bytes: bytes, output: Path
) -> Path:
    """Atomically publish exactly two canonical records without overwrite."""
    if type(scientific_bytes) is not bytes or type(performance_bytes) is not bytes:
        raise TypeError("publication payloads must be bytes")
    if not isinstance(output, Path):
        raise TypeError("output must be a pathlib.Path")
    scientific = _parse_json(scientific_bytes, "scientific.json")
    performance = _parse_json(performance_bytes, "performance.json")
    _reject_absolute_identity_values(scientific, "scientific.json")
    _reject_absolute_identity_values(performance, "performance.json")
    if (
        performance.get("scientific_sha256")
        != hashlib.sha256(scientific_bytes).hexdigest()
    ):
        raise ComparisonValidationError(
            "performance.json does not identify the exact scientific.json bytes"
        )
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"output directory already exists: {output}")
    parent = output.parent
    if not parent.is_dir() or parent.is_symlink():
        raise ComparisonValidationError(
            "output parent must be an existing real directory"
        )
    with tempfile.TemporaryDirectory(
        prefix=f".{output.name}.tmp-", dir=parent
    ) as temporary_name:
        temporary = Path(temporary_name)
        _write_portability_file(temporary / "scientific.json", scientific_bytes)
        _write_portability_file(temporary / "performance.json", performance_bytes)
        if tuple(sorted(path.name for path in temporary.iterdir())) != (
            "performance.json",
            "scientific.json",
        ):
            raise ComparisonValidationError("temporary output inventory is invalid")
        _rename_without_overwrite(temporary, output)
    return output


def publish_portability_comparison(
    reference_run: Path,
    candidate_runs: Sequence[Path],
    environment_map_path: Path,
    output: Path,
    *,
    generator: ScientificGenerator,
    policy_path: Path | None = None,
) -> tuple[ScientificComparison, PortabilityPerformance]:
    """Load, compare, encode, and atomically publish one CLI comparison."""
    environment_map = load_portability_environment_map(environment_map_path)
    run_paths = _path_sequence(candidate_runs, "candidate_runs")
    if len(run_paths) != len(environment_map.candidates):
        raise ComparisonValidationError(
            "candidate run count must match the environment map candidate count"
        )
    reference_bundle = load_comparable_bundle(reference_run)
    candidate_bundles = tuple(load_comparable_bundle(path) for path in run_paths)
    if not isinstance(output, Path):
        raise TypeError("output must be a pathlib.Path")
    resolved_output = output.resolve(strict=False)
    if any(
        resolved_output.is_relative_to(bundle.path)
        for bundle in (reference_bundle, *candidate_bundles)
    ):
        raise ComparisonValidationError("output must be outside source run bundles")
    reference = _bind_environment_descriptor(
        reference_bundle, environment_map.reference
    )
    candidates = tuple(
        _bind_environment_descriptor(bundle, descriptor)
        for bundle, descriptor in zip(
            candidate_bundles, environment_map.candidates, strict=True
        )
    )
    policy = load_portability_policy(policy_path) if policy_path is not None else None
    comparison = compare_scientific_bundles(
        reference,
        candidates,
        generator=generator,
        attempts=environment_map.attempts,
        policy=policy,
    )
    scientific_bytes = encode_scientific_comparison(comparison)
    performance = build_portability_performance(
        comparison, scientific_bytes, reference, candidates
    )
    performance_bytes = encode_portability_performance(performance)
    publish_portability_records(scientific_bytes, performance_bytes, output)
    return comparison, performance


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
    if _canonical_json(interpolation) != _canonical_json(_MAP_INTERPOLATION):
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


def _load_canonical_input(path: Path, name: str) -> tuple[bytes, dict[str, object]]:
    if not isinstance(path, Path):
        raise TypeError(f"{name} path must be a pathlib.Path")
    with _open_regular(path, name) as (stream, before):
        try:
            payload = stream.read()
            after = os.fstat(stream.fileno())
        except OSError as error:
            raise ComparisonValidationError(f"{name} cannot be read") from error
    if _stat_identity(before) != _stat_identity(after) or len(payload) != after.st_size:
        raise ComparisonValidationError(f"{name} changed while reading")
    return payload, _parse_json(payload, name)


def _comparison_keys(
    value: Mapping[str, object], expected: set[str], name: str
) -> None:
    actual = set(value)
    missing = sorted(expected - actual)
    if missing:
        raise ComparisonValidationError(
            f"{name} is missing required keys: {', '.join(missing)}"
        )
    unknown = sorted(actual - expected)
    if unknown:
        raise ComparisonValidationError(
            f"{name} contains unknown keys: {', '.join(unknown)}"
        )


def _comparison_dict(value: object, name: str) -> dict[str, object]:
    if type(value) is not dict or any(type(key) is not str for key in value):
        raise ComparisonValidationError(f"{name} must be an object")
    return value


def _comparison_list(value: object, name: str) -> list[object]:
    if type(value) is not list:
        raise ComparisonValidationError(f"{name} must be an array")
    return value


def _validate_descriptor_identity(value: object) -> None:
    for name in (
        "environment_id",
        "policy_role",
        "os_label",
        "execution_layer",
        "hardware_label",
        "requested_device",
    ):
        if not hasattr(value, name):
            raise ComparisonValidationError("environment descriptor is invalid")
    _environment_id(value.environment_id, "environment_id")
    if type(value.policy_role) is not str or value.policy_role not in _POLICY_ROLES:
        raise ComparisonValidationError("policy_role is invalid")
    _public_label(value.os_label, "os_label")
    if (
        type(value.execution_layer) is not str
        or value.execution_layer not in _EXECUTION_LAYERS
    ):
        raise ComparisonValidationError("execution_layer is invalid")
    _public_label(value.hardware_label, "hardware_label")
    if type(value.requested_device) is not str or not _REQUESTED_DEVICE.fullmatch(
        value.requested_device
    ):
        raise ComparisonValidationError("requested_device is invalid")
    if value.requested_device == "mps" and value.policy_role != "post_policy_attempt":
        raise ComparisonValidationError(
            "MPS bundles must use policy_role post_policy_attempt"
        )


def _environment_descriptor_record(
    value: object, name: str
) -> PortabilityEnvironmentDescriptor:
    record = _comparison_dict(value, name)
    fields = {
        "environment_id",
        "policy_role",
        "os_label",
        "execution_layer",
        "hardware_label",
        "requested_device",
    }
    _comparison_keys(record, fields, name)
    return PortabilityEnvironmentDescriptor(
        environment_id=record["environment_id"],  # type: ignore[arg-type]
        policy_role=record["policy_role"],  # type: ignore[arg-type]
        os_label=record["os_label"],  # type: ignore[arg-type]
        execution_layer=record["execution_layer"],  # type: ignore[arg-type]
        hardware_label=record["hardware_label"],  # type: ignore[arg-type]
        requested_device=record["requested_device"],  # type: ignore[arg-type]
    )


def _execution_attempt_record(value: object, name: str) -> ScientificExecutionAttempt:
    record = _comparison_dict(value, name)
    _comparison_keys(
        record,
        {
            "environment_id",
            "gating",
            "policy_role",
            "reason_code",
            "stage_code",
            "status",
        },
        name,
    )
    if record["gating"] is not False or record["policy_role"] != "post_policy_attempt":
        raise ComparisonValidationError(
            f"{name} must be a canonical non-gating post-policy attempt"
        )
    return ScientificExecutionAttempt(
        record["environment_id"],  # type: ignore[arg-type]
        record["status"],  # type: ignore[arg-type]
        record["reason_code"],  # type: ignore[arg-type]
        record["stage_code"],  # type: ignore[arg-type]
    )


def _environment_id_array(value: object, name: str) -> tuple[str, ...]:
    result = tuple(
        _environment_id(item, f"{name} item") for item in _comparison_list(value, name)
    )
    if not result or len(result) != len(set(result)):
        raise ComparisonValidationError(f"{name} must be a nonempty unique array")
    return result


def _exact_requirement_array(
    value: object, expected: tuple[str, ...], name: str
) -> tuple[str, ...]:
    result = tuple(
        _comparison_string(item, f"{name} item")
        for item in _comparison_list(value, name)
    )
    if result != expected:
        raise ComparisonValidationError(
            f"{name} must contain every exact requirement in contract order"
        )
    return result


def _bounded_machine_id(value: object, name: str) -> str:
    if type(value) is not str or not _MACHINE_ID.fullmatch(value):
        raise ComparisonValidationError(f"{name} must be a bounded machine ID")
    return value


def _nonnegative_policy_number(value: object, name: str) -> int | float:
    if type(value) not in {int, float} or value < 0:
        raise ComparisonValidationError(f"{name} must be finite and nonnegative")
    try:
        finite = math.isfinite(float(value))
    except (OverflowError, ValueError):
        finite = False
    if not finite:
        raise ComparisonValidationError(f"{name} must be finite and nonnegative")
    return value


def _portability_policy_value(policy: PortabilityPolicy) -> dict[str, object]:
    return {
        "calibration_environment_ids": list(policy.calibration_environment_ids),
        "category": policy.category,
        "derivation": {
            "comparison_ids": list(policy.derivation.comparison_ids),
            "method_id": policy.derivation.method_id,
        },
        "discrete_output_requirements": list(policy.discrete_output_requirements),
        "floating_component_limits": {
            name: {
                "atol": policy.floating_component_limits[name].atol,
                "rtol": policy.floating_component_limits[name].rtol,
            }
            for name in _FLOATING_COMPONENTS
        },
        "holdout_environment_ids": list(policy.holdout_environment_ids),
        "limitation": policy.limitation,
        "metric_absolute_delta_limits": {
            name: policy.metric_absolute_delta_limits[name]
            for name in _SCIENTIFIC_METRICS
        },
        "policy_id": policy.policy_id,
        "profile_id": policy.profile_id,
        "provenance_requirements": list(policy.provenance_requirements),
        "reference_environment_id": policy.reference_environment_id,
        "reviewed_evidence_hashes": list(policy.reviewed_evidence_hashes),
        "schema_id": policy.schema_id,
        "schema_version": policy.schema_version,
    }


def _validate_portability_policy(policy: PortabilityPolicy) -> None:
    if (
        type(policy.schema_version) is not int
        or policy.schema_version != 1
        or type(policy.schema_id) is not str
        or policy.schema_id != _POLICY_SCHEMA_ID
    ):
        raise ComparisonValidationError("portability policy identity is invalid")
    _bounded_machine_id(policy.policy_id, "policy_id")
    _bounded_machine_id(policy.profile_id, "profile_id")
    _bounded_machine_id(policy.category, "category")
    _environment_id(policy.reference_environment_id, "reference_environment_id")
    calibration = _environment_id_array(
        list(policy.calibration_environment_ids), "calibration_environment_ids"
    )
    holdout = _environment_id_array(
        list(policy.holdout_environment_ids), "holdout_environment_ids"
    )
    if policy.reference_environment_id in {*calibration, *holdout}:
        raise ComparisonValidationError(
            "policy reference environment must be separate from candidates"
        )
    if set(calibration) & set(holdout):
        raise ComparisonValidationError(
            "policy calibration and holdout environments must not overlap"
        )
    if policy.provenance_requirements != _PROVENANCE_REQUIREMENTS:
        raise ComparisonValidationError("policy provenance requirements are invalid")
    if policy.discrete_output_requirements != _DISCRETE_REQUIREMENTS:
        raise ComparisonValidationError("policy discrete requirements are invalid")
    if set(policy.floating_component_limits) != set(_FLOATING_COMPONENTS) or any(
        type(value) is not PolicyTolerance
        for value in policy.floating_component_limits.values()
    ):
        raise ComparisonValidationError("policy floating limits are invalid")
    if set(policy.metric_absolute_delta_limits) != set(_SCIENTIFIC_METRICS):
        raise ComparisonValidationError("policy metric limits are invalid")
    for name, value in policy.metric_absolute_delta_limits.items():
        _nonnegative_policy_number(value, f"policy metric limit {name}")
    if type(policy.derivation) is not PolicyDerivation:
        raise ComparisonValidationError("policy derivation is invalid")
    if (
        type(policy.reviewed_evidence_hashes) is not tuple
        or not policy.reviewed_evidence_hashes
        or len(policy.reviewed_evidence_hashes)
        != len(set(policy.reviewed_evidence_hashes))
    ):
        raise ComparisonValidationError(
            "reviewed_evidence_hashes must be a nonempty unique array"
        )
    for value in policy.reviewed_evidence_hashes:
        _comparison_sha256(value, "reviewed evidence hash")
    if (
        type(policy.limitation) is not str
        or len(policy.limitation) > 500
        or "observed envelope" not in policy.limitation.casefold()
        or "not a universal guarantee" not in policy.limitation.casefold()
        or any(ord(character) < 32 for character in policy.limitation)
    ):
        raise ComparisonValidationError(
            "policy limitation must bound the observed envelope"
        )
    _portable_identity_string(policy.limitation, "policy limitation")
    if type(policy.source) is not CanonicalInputIdentity:
        raise ComparisonValidationError("policy source identity is invalid")
    canonical = _canonical_json(_portability_policy_value(policy))
    if (
        policy.source.byte_count != len(canonical)
        or policy.source.sha256 != hashlib.sha256(canonical).hexdigest()
    ):
        raise ComparisonValidationError(
            "policy values differ from their canonical source identity"
        )


def _validate_policy_scope(
    policy: PortabilityPolicy,
    reference: ScientificBundleDescriptor,
    candidates: tuple[object, ...],
) -> None:
    run = _comparison_mapping(reference.bundle.run_metadata, "reference run")
    if policy.reference_environment_id != reference.environment_id:
        raise ComparisonValidationError(
            "policy reference environment differs from the reference descriptor"
        )
    if policy.profile_id != run.get("profile_id"):
        raise ComparisonValidationError("policy profile differs from the reference")
    if policy.category != run.get("category"):
        raise ComparisonValidationError("policy category differs from the reference")
    calibration = set(policy.calibration_environment_ids)
    holdout = set(policy.holdout_environment_ids)
    for value in candidates:
        assert type(value) is ScientificBundleDescriptor
        if value.policy_role == "post_policy_attempt":
            continue
        allowed = (
            calibration
            if value.policy_role
            in {
                "same_stack_control",
                "calibration",
            }
            else holdout
        )
        if value.environment_id not in allowed:
            raise ComparisonValidationError(
                f"{value.environment_id}: candidate is outside policy scope"
            )


def _path_sequence(value: object, name: str) -> tuple[Path, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{name} must be an ordered collection of paths")
    result = tuple(value)
    if not result or any(not isinstance(path, Path) for path in result):
        raise TypeError(f"{name} must contain at least one pathlib.Path")
    return result


def _bind_environment_descriptor(
    bundle: ComparableBundle, descriptor: PortabilityEnvironmentDescriptor
) -> ScientificBundleDescriptor:
    return ScientificBundleDescriptor(
        bundle=bundle,
        environment_id=descriptor.environment_id,
        policy_role=descriptor.policy_role,
        os_label=descriptor.os_label,
        execution_layer=descriptor.execution_layer,
        hardware_label=descriptor.hardware_label,
        requested_device=descriptor.requested_device,
    )


def _comparison_sequence(
    value: object, expected_type: type, name: str
) -> tuple[object, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ComparisonValidationError(f"{name} must be an ordered collection")
    result = tuple(value)
    if any(type(item) is not expected_type for item in result):
        raise ComparisonValidationError(
            f"{name} contains an invalid {expected_type.__name__} record"
        )
    return result


def _environment_id(value: object, name: str) -> str:
    if type(value) is not str or not _ENVIRONMENT_ID.fullmatch(value) or "--" in value:
        raise ComparisonValidationError(
            f"{name} must be a bounded lowercase kebab-case token"
        )
    return value


def _machine_code(value: object, name: str) -> str:
    if type(value) is not str or len(value) > 64 or not _MACHINE_CODE.fullmatch(value):
        raise ComparisonValidationError(
            f"{name} must be a bounded lowercase snake-case token"
        )
    return value


def _public_label(value: object, name: str) -> str:
    if (
        type(value) is not str
        or not value
        or value != value.strip()
        or len(value) > 120
        or any(ord(character) < 32 for character in value)
        or any(fragment in value for fragment in ("/", "\\", "@", "://", "~"))
    ):
        raise ComparisonValidationError(f"{name} is not a sanitized public label")
    ip_tokens = re.findall(
        r"(?<![0-9])(?:[0-9]{1,3}\.){3}[0-9]{1,3}(?![0-9])"
        r"|(?<![A-Za-z0-9])\[?[0-9A-Fa-f:]{3,}\]?(?![A-Za-z0-9])",
        value,
    )
    for token in ip_tokens:
        try:
            ipaddress.ip_address(token.strip("[]"))
        except ValueError:
            continue
        raise ComparisonValidationError(f"{name} must not contain a private host IP")
    if re.search(
        r"(?<![A-Za-z0-9-])(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,62})\.)+"
        r"[A-Za-z]{2,63}(?![A-Za-z0-9-])",
        value,
    ):
        raise ComparisonValidationError(f"{name} must not contain a hostname")
    return value


def _comparison_mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise ComparisonValidationError(f"{name} must be a string-keyed mapping")
    return value


def _comparison_string(value: object, name: str) -> str:
    if (
        type(value) is not str
        or not value
        or any(ord(character) < 32 for character in value)
    ):
        raise ComparisonValidationError(f"{name} must be a nonempty string")
    return value


def _comparison_integer(value: object, name: str, *, positive: bool = False) -> int:
    if type(value) is not int or value < int(positive):
        qualifier = "positive" if positive else "nonnegative"
        raise ComparisonValidationError(f"{name} must be a {qualifier} integer")
    return value


def _comparison_boolean(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise ComparisonValidationError(f"{name} must be a boolean")
    return value


def _comparison_sha256(value: object, name: str) -> str:
    result = _comparison_string(value, name)
    if not _SHA256.fullmatch(result):
        raise ComparisonValidationError(f"{name} must be a SHA-256 hex digest")
    return result.lower()


def _comparison_commit(value: object, name: str) -> str:
    result = _comparison_string(value, name)
    if not _COMMIT.fullmatch(result):
        raise ComparisonValidationError(f"{name} must be a full lowercase commit hash")
    return result


def _comparison_tensor(
    value: object,
    name: str,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    *,
    finite: bool = False,
) -> Tensor:
    if type(value) is not Tensor:
        raise ComparisonValidationError(f"{name} must be a base torch.Tensor")
    if value.device.type != "cpu":
        raise ComparisonValidationError(f"{name} must be on the CPU")
    if value.is_nested or value.layout is not torch.strided:
        raise ComparisonValidationError(f"{name} must use dense strided storage")
    if value.dtype != dtype:
        raise ComparisonValidationError(f"{name} must use {dtype}")
    if tuple(value.shape) != shape:
        raise ComparisonValidationError(f"{name} must have shape {shape}")
    if not value.is_contiguous():
        raise ComparisonValidationError(f"{name} must be contiguous")
    if finite:
        flattened = value.reshape(-1)
        for start in range(0, flattened.numel(), _FLOAT_CHUNK_SIZE):
            if (
                not torch.isfinite(flattened[start : start + _FLOAT_CHUNK_SIZE])
                .all()
                .item()
            ):
                raise ComparisonValidationError(
                    f"{name} must contain only finite values"
                )
    return value


def _revalidate_comparable_bundle(bundle: ComparableBundle) -> None:
    """Recheck mutable payloads of an already loaded bundle."""
    if type(bundle) is not ComparableBundle:
        raise ComparisonValidationError("comparison bundle must be ComparableBundle")
    test_count = len(bundle.test_sample_ids)
    bank_shape = bundle.memory_bank_metadata.shape
    rows = bank_shape[0]
    _comparison_tensor(
        bundle.memory_bank,
        "bundle.memory_bank",
        bank_shape,
        torch.float32,
        finite=True,
    )
    retrieval_shape = (test_count, _PATCH_COUNT)
    distances = _comparison_tensor(
        bundle.patch_distances,
        "bundle.patch_distances",
        retrieval_shape,
        torch.float32,
        finite=True,
    )
    if torch.lt(distances, 0).any().item():
        raise ComparisonValidationError("bundle.patch_distances must be nonnegative")
    indices = _comparison_tensor(
        bundle.nearest_bank_indices,
        "bundle.nearest_bank_indices",
        retrieval_shape,
        torch.int64,
    )
    if torch.lt(indices, 0).any().item() or torch.ge(indices, rows).any().item():
        raise ComparisonValidationError(
            "bundle nearest indices lie outside the memory bank"
        )
    labels = _comparison_tensor(
        bundle.test_labels,
        "bundle.test_labels",
        (test_count,),
        torch.uint8,
    )
    if not torch.logical_or(labels == 0, labels == 1).all().item():
        raise ComparisonValidationError("bundle.test_labels must be binary")
    _comparison_tensor(
        bundle.image_scores,
        "bundle.image_scores",
        (test_count,),
        torch.float32,
        finite=True,
    )
    map_shape = (test_count, *_MAP_SIZE)
    _comparison_tensor(
        bundle.anomaly_maps,
        "bundle.anomaly_maps",
        map_shape,
        torch.float32,
        finite=True,
    )
    masks = _comparison_tensor(
        bundle.evaluation_masks,
        "bundle.evaluation_masks",
        map_shape,
        torch.uint8,
    )
    if not torch.logical_or(masks == 0, masks == 1).all().item():
        raise ComparisonValidationError("bundle.evaluation_masks must be binary")

    if type(bundle.metrics) is not BundleMetrics:
        raise ComparisonValidationError("bundle metrics are invalid")
    for name in _SCIENTIFIC_METRICS:
        value = getattr(bundle.metrics, name)
        if type(value) is not float or not math.isfinite(value):
            raise ComparisonValidationError(
                f"bundle.metrics.{name} must be a finite float"
            )


def _scientific_run_identity(
    descriptor: ScientificBundleDescriptor,
) -> ScientificRunIdentity:
    bundle = descriptor.bundle
    run = _comparison_mapping(bundle.run_metadata, "bundle.run_metadata")
    source = _comparison_mapping(run["source"], "bundle.run_metadata.source")
    environment = _comparison_mapping(
        run["environment"], "bundle.run_metadata.environment"
    )
    weights = _comparison_mapping(run["weights"], "bundle.run_metadata.weights")
    inventory = _normalized_inventory(bundle)
    scientific_workload = {
        "bank_chunk_size": run["bank_chunk_size"],
        "batch_size": run["batch_size"],
        "determinism": _thaw_comparison_json(run["determinism"]),
        "feature_extractor": run["feature_extractor"],
        "feature_layer": run["feature_layer"],
        "image_score_semantics": _IMAGE_SCORE_SEMANTICS,
        "map_interpolation": _thaw_comparison_json(run["map_interpolation"]),
        "metric_fields": list(_SCIENTIFIC_METRICS),
        "preprocessing_profile": run["preprocessing_profile"],
        "retrieval_semantics": run["retrieval_semantics"],
        "tensors": _thaw_comparison_json(run["tensors"]),
    }
    run_identity = {
        "category": run["category"],
        "dependency_versions": _thaw_comparison_json(
            environment["dependency_versions"]
        ),
        "inventory": inventory,
        "profile_id": run["profile_id"],
        "python_version": environment["python_version"],
        "run_id": run["run_id"],
        "schema_version": run["schema_version"],
        "scientific_workload": scientific_workload,
        "source": {
            "dirty": source["dirty"],
            "git_commit": source["git_commit"],
            "uv_lock_sha256": _comparison_sha256(
                source["uv_lock_sha256"],
                "bundle.run_metadata.source.uv_lock_sha256",
            ),
        },
        "weights": {
            "cached_file_sha256": _comparison_sha256(
                weights["cached_file_sha256"],
                "bundle.run_metadata.weights.cached_file_sha256",
            ),
            "enum": weights["enum"],
        },
    }
    workload = None
    methodology = None
    if bundle.kind == "benchmark":
        benchmark = _comparison_mapping(
            bundle.benchmark_metadata, "bundle.benchmark_metadata"
        )
        run_identity["benchmark_identity"] = {
            "benchmark_sample_id": benchmark["benchmark_sample_id"],
            "schema_version": benchmark["schema_version"],
        }
        workload = _thaw_comparison_json(benchmark["workload"])
        methodology = _thaw_comparison_json(benchmark["methodology"])
    return ScientificRunIdentity(
        environment_id=descriptor.environment_id,
        policy_role=descriptor.policy_role,
        bundle_kind=bundle.kind,
        os_label=descriptor.os_label,
        execution_layer=descriptor.execution_layer,
        hardware_label=descriptor.hardware_label,
        requested_device=descriptor.requested_device,
        run=run_identity,
        source_files=bundle.source_files,
        benchmark_workload=workload,  # type: ignore[arg-type]
        benchmark_methodology=methodology,  # type: ignore[arg-type]
    )


def _normalized_inventory(bundle: ComparableBundle) -> dict[str, object]:
    run = _comparison_mapping(bundle.run_metadata, "bundle.run_metadata")
    inventory = _comparison_mapping(run["inventory"], "bundle.run_metadata.inventory")
    result = _thaw_comparison_json(inventory)
    if type(result) is not dict:
        raise ComparisonValidationError("bundle inventory projection is invalid")
    result["sample_inventory_sha256"] = _comparison_sha256(
        inventory["sample_inventory_sha256"],
        "bundle.run_metadata.inventory.sample_inventory_sha256",
    )
    return result


def _scientific_gates(
    reference: ComparableBundle, candidate: ComparableBundle
) -> tuple[tuple[str, bool], ...]:
    reference_run = _comparison_mapping(
        reference.run_metadata, "reference.run_metadata"
    )
    candidate_run = _comparison_mapping(
        candidate.run_metadata, "candidate.run_metadata"
    )
    reference_source = _comparison_mapping(
        reference_run["source"], "reference.run_metadata.source"
    )
    candidate_source = _comparison_mapping(
        candidate_run["source"], "candidate.run_metadata.source"
    )
    reference_weights = _comparison_mapping(
        reference_run["weights"], "reference.run_metadata.weights"
    )
    candidate_weights = _comparison_mapping(
        candidate_run["weights"], "candidate.run_metadata.weights"
    )
    reference_commit = reference_source["git_commit"]
    candidate_commit = candidate_source["git_commit"]
    reference_lock = _comparison_sha256(
        reference_source["uv_lock_sha256"],
        "reference.run_metadata.source.uv_lock_sha256",
    )
    candidate_lock = _comparison_sha256(
        candidate_source["uv_lock_sha256"],
        "candidate.run_metadata.source.uv_lock_sha256",
    )
    reference_weight_hash = _comparison_sha256(
        reference_weights["cached_file_sha256"],
        "reference.run_metadata.weights.cached_file_sha256",
    )
    candidate_weight_hash = _comparison_sha256(
        candidate_weights["cached_file_sha256"],
        "candidate.run_metadata.weights.cached_file_sha256",
    )
    reference_determinism = _comparison_mapping(
        reference_run["determinism"], "reference.run_metadata.determinism"
    )
    candidate_determinism = _comparison_mapping(
        candidate_run["determinism"], "candidate.run_metadata.determinism"
    )
    reference_configuration = {
        "bank_chunk_size": reference_run["bank_chunk_size"],
        "batch_size": reference_run["batch_size"],
        "determinism": {
            name: value
            for name, value in reference_determinism.items()
            if name != "torch_cuda_seed_all"
        },
    }
    candidate_configuration = {
        "bank_chunk_size": candidate_run["bank_chunk_size"],
        "batch_size": candidate_run["batch_size"],
        "determinism": {
            name: value
            for name, value in candidate_determinism.items()
            if name != "torch_cuda_seed_all"
        },
    }
    reference_sample_snapshot = _source_snapshot(reference, "samples.jsonl")
    candidate_sample_snapshot = _source_snapshot(candidate, "samples.jsonl")
    reference_tensors = _comparison_mapping(
        reference_run["tensors"], "reference.run_metadata.tensors"
    )
    candidate_tensors = _comparison_mapping(
        candidate_run["tensors"], "candidate.run_metadata.tensors"
    )
    reference_sample_ids = tuple(sample.sample_id for sample in reference.samples)
    candidate_sample_ids = tuple(sample.sample_id for sample in candidate.samples)
    reference_training_ids = tuple(
        sample.sample_id for sample in reference.samples if sample.split == "train"
    )
    candidate_training_ids = tuple(
        sample.sample_id for sample in candidate.samples if sample.split == "train"
    )
    return (
        (
            "run_schema",
            reference_run["schema_version"] == candidate_run["schema_version"] == 1,
        ),
        (
            "profile",
            reference_run["profile_id"]
            == candidate_run["profile_id"]
            == "inspectrt_feature_memory_v1",
        ),
        ("category", reference_run["category"] == candidate_run["category"]),
        (
            "preprocessing",
            reference_run["preprocessing_profile"]
            == candidate_run["preprocessing_profile"]
            == "inspectrt_resize256_v1",
        ),
        (
            "feature_contract",
            (
                reference_run["feature_extractor"],
                reference_run["feature_layer"],
                reference.memory_bank_metadata.embedding_dimension,
                reference.memory_bank_metadata.patches_per_training_sample,
            )
            == (
                candidate_run["feature_extractor"],
                candidate_run["feature_layer"],
                candidate.memory_bank_metadata.embedding_dimension,
                candidate.memory_bank_metadata.patches_per_training_sample,
            )
            == ("ResNet-50", "layer2", _EMBEDDING_DIMENSION, _PATCH_COUNT),
        ),
        (
            "weight_identity",
            (
                reference_weights["enum"],
                reference_weights["source_url"],
                reference_weight_hash,
            )
            == (
                candidate_weights["enum"],
                candidate_weights["source_url"],
                candidate_weight_hash,
            )
            == (_WEIGHT_ENUM, _WEIGHT_URL, _ACCEPTED_WEIGHT_SHA256),
        ),
        (
            "configuration",
            reference_configuration
            == candidate_configuration
            == {
                "bank_chunk_size": 16_384,
                "batch_size": 1,
                "determinism": _DETERMINISM,
            },
        ),
        (
            "lock_identity",
            reference_lock == candidate_lock == _ACCEPTED_LOCK_SHA256,
        ),
        (
            "clean_source",
            reference_source["dirty"] is False and candidate_source["dirty"] is False,
        ),
        (
            "scientific_source_commit",
            reference_commit == candidate_commit == _SCIENTIFIC_SOURCE_COMMIT,
        ),
        (
            "inventory_identity",
            _normalized_inventory(reference) == _normalized_inventory(candidate),
        ),
        (
            "samples_source",
            reference_sample_snapshot.byte_count == candidate_sample_snapshot.byte_count
            and reference_sample_snapshot.sha256.lower()
            == candidate_sample_snapshot.sha256.lower(),
        ),
        ("ordered_sample_ids", reference_sample_ids == candidate_sample_ids),
        ("ordered_sample_metadata", reference.samples == candidate.samples),
        ("ordered_training_ids", reference_training_ids == candidate_training_ids),
        (
            "ordered_test_sample_ids",
            reference.test_sample_ids == candidate.test_sample_ids,
        ),
        (
            "ordered_labels",
            torch.equal(reference.test_labels, candidate.test_labels),
        ),
        (
            "sample_counts",
            _sample_counts(reference) == _sample_counts(candidate),
        ),
        (
            "memory_bank_contract",
            reference.memory_bank_metadata == candidate.memory_bank_metadata
            and reference_tensors["memory_bank"] == candidate_tensors["memory_bank"],
        ),
        (
            "patch_distance_contract",
            reference_tensors["patch_distances"]
            == candidate_tensors["patch_distances"],
        ),
        (
            "image_score_contract",
            reference_tensors["image_scores"] == candidate_tensors["image_scores"],
        ),
        (
            "nearest_index_contract",
            reference_tensors["nearest_bank_indices"]
            == candidate_tensors["nearest_bank_indices"],
        ),
        (
            "test_label_contract",
            reference_tensors["test_labels"] == candidate_tensors["test_labels"],
        ),
        (
            "anomaly_map_contract",
            reference_tensors["anomaly_maps"] == candidate_tensors["anomaly_maps"],
        ),
        (
            "mask_contract",
            reference_tensors["evaluation_masks"]
            == candidate_tensors["evaluation_masks"],
        ),
        (
            "retrieval_semantics",
            reference_run["retrieval_semantics"]
            == candidate_run["retrieval_semantics"]
            == "exact top-1 squared L2",
        ),
        (
            "image_score_semantics",
            True,
        ),
        (
            "anomaly_map_semantics",
            reference_run["map_interpolation"]
            == candidate_run["map_interpolation"]
            == _freeze_json(_MAP_INTERPOLATION),
        ),
        (
            "metric_fields",
            True,
        ),
    )


def _source_snapshot(bundle: ComparableBundle, name: str) -> SourceFileSnapshot:
    try:
        return next(
            snapshot for snapshot in bundle.source_files if snapshot.name == name
        )
    except StopIteration as error:
        raise ComparisonValidationError(
            f"bundle source snapshot {name!r} is missing"
        ) from error


def _sample_counts(bundle: ComparableBundle) -> tuple[int, ...]:
    inventory = _normalized_inventory(bundle)
    return tuple(
        int(inventory[name])
        for name in (
            "training_sample_count",
            "test_sample_count",
            "test_good_sample_count",
            "anomalous_test_sample_count",
            "total_sample_count",
        )
    )


def _floating_statistics(
    reference: Tensor,
    candidate: Tensor,
    name: str,
    *,
    tolerance: PolicyTolerance | None = None,
) -> FloatingStatistics:
    reference_values = reference.reshape(-1)
    candidate_values = candidate.reshape(-1)
    element_count = reference_values.numel()
    exact_count = 0
    zero_reference_count = 0
    absolute_sum = 0.0
    squared_sum = 0.0
    maximum_absolute_error = 0.0
    maximum_relative_error: float | None = None
    policy_violation_count = 0 if tolerance is not None else None
    for start in range(0, element_count, _FLOAT_CHUNK_SIZE):
        stop = min(start + _FLOAT_CHUNK_SIZE, element_count)
        reference_chunk = reference_values[start:stop]
        candidate_chunk = candidate_values[start:stop]
        exact_count += int(reference_chunk.eq(candidate_chunk).sum().item())
        reference64 = reference_chunk.to(dtype=torch.float64)
        candidate64 = candidate_chunk.to(dtype=torch.float64)
        absolute_error = candidate64.sub(reference64).abs()
        if tolerance is not None:
            assert policy_violation_count is not None
            allowed = (
                reference64.abs().mul(float(tolerance.rtol)).add(float(tolerance.atol))
            )
            policy_violation_count += int(absolute_error.gt(allowed).sum().item())
        absolute_sum += float(absolute_error.sum(dtype=torch.float64).item())
        squared_sum += float(absolute_error.square().sum(dtype=torch.float64).item())
        maximum_absolute_error = max(
            maximum_absolute_error, float(absolute_error.max().item())
        )
        nonzero = reference64.ne(0)
        nonzero_count = int(nonzero.sum().item())
        zero_reference_count += stop - start - nonzero_count
        if nonzero_count:
            relative_maximum = float(
                absolute_error[nonzero].div(reference64[nonzero].abs()).max().item()
            )
            maximum_relative_error = (
                relative_maximum
                if maximum_relative_error is None
                else max(maximum_relative_error, relative_maximum)
            )
    return FloatingStatistics(
        element_count=element_count,
        exact_count=exact_count,
        differing_count=element_count - exact_count,
        maximum_absolute_error=maximum_absolute_error,
        mean_absolute_error=absolute_sum / element_count,
        root_mean_square_error=math.sqrt(squared_sum / element_count),
        maximum_relative_error=maximum_relative_error,
        zero_reference_count=zero_reference_count,
        policy_violation_count=policy_violation_count,
    )


def _sequence_discrete(
    name: str, reference: Sequence[object], candidate: Sequence[object]
) -> DiscreteComponentComparison:
    if len(reference) != len(candidate):
        raise ComparisonValidationError(f"{name} lengths must match")
    return _structural_sequence_discrete(name, reference, candidate)


def _structural_sequence_discrete(
    name: str, reference: Sequence[object], candidate: Sequence[object]
) -> DiscreteComponentComparison:
    element_count = max(len(reference), len(candidate))
    exact_count = sum(
        reference_value == candidate_value
        for reference_value, candidate_value in zip(reference, candidate)
    )
    mismatch_count = element_count - exact_count
    return DiscreteComponentComparison(
        name=name,
        exact=mismatch_count == 0,
        element_count=element_count,
        exact_count=exact_count,
        mismatch_count=mismatch_count,
        mismatch_rate=mismatch_count / element_count,
    )


def _structural_tensor_discrete(
    name: str, reference: Tensor, candidate: Tensor
) -> DiscreteComponentComparison:
    reference_values = reference.reshape(-1)
    candidate_values = candidate.reshape(-1)
    common_count = min(reference_values.numel(), candidate_values.numel())
    exact_count = 0
    for start in range(0, common_count, _FLOAT_CHUNK_SIZE):
        stop = min(start + _FLOAT_CHUNK_SIZE, common_count)
        exact_count += int(
            reference_values[start:stop].eq(candidate_values[start:stop]).sum().item()
        )
    element_count = max(reference_values.numel(), candidate_values.numel())
    mismatch_count = element_count - exact_count
    return DiscreteComponentComparison(
        name=name,
        exact=mismatch_count == 0,
        element_count=element_count,
        exact_count=exact_count,
        mismatch_count=mismatch_count,
        mismatch_rate=mismatch_count / element_count,
    )


def _tensor_discrete(
    name: str, reference: Tensor, candidate: Tensor
) -> DiscreteComponentComparison:
    if reference.shape != candidate.shape or reference.dtype != candidate.dtype:
        raise ComparisonValidationError(f"{name} tensor contracts must match")
    return _structural_tensor_discrete(name, reference, candidate)


def _nearest_index_comparison(
    reference: Tensor, candidate: Tensor
) -> DiscreteComponentComparison:
    if (
        reference.shape != candidate.shape
        or reference.dtype != torch.int64
        or candidate.dtype != torch.int64
    ):
        raise ComparisonValidationError(
            "nearest_bank_indices tensor contracts must match"
        )
    reference_values = reference.reshape(-1)
    candidate_values = candidate.reshape(-1)
    element_count = reference_values.numel()
    mismatch_count = 0
    first_mismatches = []
    width = reference.shape[1]
    for start in range(0, element_count, _FLOAT_CHUNK_SIZE):
        stop = min(start + _FLOAT_CHUNK_SIZE, element_count)
        reference_chunk = reference_values[start:stop]
        candidate_chunk = candidate_values[start:stop]
        mismatch_positions = torch.nonzero(
            reference_chunk.ne(candidate_chunk), as_tuple=False
        ).reshape(-1)
        mismatch_count += mismatch_positions.numel()
        remaining = _INDEX_MISMATCH_LIMIT - len(first_mismatches)
        for offset in mismatch_positions[:remaining].tolist():
            flat_index = start + int(offset)
            first_mismatches.append(
                IndexMismatch(
                    coordinate=divmod(flat_index, width),
                    reference_value=int(reference_values[flat_index].item()),
                    candidate_value=int(candidate_values[flat_index].item()),
                )
            )
    return DiscreteComponentComparison(
        name="nearest_bank_indices",
        exact=mismatch_count == 0,
        element_count=element_count,
        exact_count=element_count - mismatch_count,
        mismatch_count=mismatch_count,
        mismatch_rate=mismatch_count / element_count,
        first_mismatches=tuple(first_mismatches),
    )


def _freeze_comparison_json(value: object) -> Mapping[str, object]:
    thawed = _thaw_comparison_json(value)
    if type(thawed) is not dict:
        raise ComparisonValidationError("comparison JSON projection must be an object")
    frozen = _freeze_json(thawed)
    if not isinstance(frozen, Mapping):
        raise ComparisonValidationError("comparison JSON projection is invalid")
    return frozen


def _thaw_comparison_json(value: object) -> object:
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            if type(key) is not str:
                raise ComparisonValidationError("comparison JSON keys must be strings")
            result[key] = _thaw_comparison_json(item)
        return result
    if isinstance(value, (tuple, list)):
        return [_thaw_comparison_json(item) for item in value]
    if value is None or type(value) in {str, int, bool}:
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise ComparisonValidationError("comparison JSON value is invalid")


def _generator_value(generator: ScientificGenerator) -> dict[str, object]:
    return {"dirty": generator.dirty, "source_commit": generator.source_commit}


def _attempt_value(attempt: ScientificExecutionAttempt) -> dict[str, object]:
    return {
        "environment_id": attempt.environment_id,
        "gating": False,
        "policy_role": "post_policy_attempt",
        "reason_code": attempt.reason_code,
        "stage_code": attempt.stage_code,
        "status": attempt.status,
    }


def _run_identity_value(identity: ScientificRunIdentity) -> dict[str, object]:
    value = {
        "bundle_kind": identity.bundle_kind,
        "environment_id": identity.environment_id,
        "execution_layer": identity.execution_layer,
        "hardware_label": identity.hardware_label,
        "os_label": identity.os_label,
        "policy_role": identity.policy_role,
        "requested_device": identity.requested_device,
        "run": _thaw_comparison_json(identity.run),
        "source_files": [
            {
                "byte_count": snapshot.byte_count,
                "name": snapshot.name,
                "sha256": snapshot.sha256.lower(),
            }
            for snapshot in identity.source_files
        ],
    }
    if identity.bundle_kind == "benchmark":
        value["benchmark"] = {
            "methodology": _thaw_comparison_json(identity.benchmark_methodology),
            "workload": _thaw_comparison_json(identity.benchmark_workload),
        }
    return value


def _floating_statistics_value(value: FloatingStatistics) -> dict[str, object]:
    result = {
        "differing_count": value.differing_count,
        "element_count": value.element_count,
        "exact_count": value.exact_count,
        "maximum_absolute_error": value.maximum_absolute_error,
        "maximum_relative_error": value.maximum_relative_error,
        "mean_absolute_error": value.mean_absolute_error,
        "root_mean_square_error": value.root_mean_square_error,
        "zero_reference_count": value.zero_reference_count,
    }
    if value.policy_violation_count is not None:
        result["policy_violation_count"] = value.policy_violation_count
    return result


def _discrete_component_value(
    value: DiscreteComponentComparison,
) -> dict[str, object]:
    result = {
        "element_count": value.element_count,
        "exact": value.exact,
        "exact_count": value.exact_count,
        "mismatch_count": value.mismatch_count,
        "mismatch_rate": value.mismatch_rate,
    }
    if value.name == "nearest_bank_indices":
        result["first_mismatches"] = [
            {
                "candidate_value": mismatch.candidate_value,
                "coordinate": list(mismatch.coordinate),
                "reference_value": mismatch.reference_value,
            }
            for mismatch in value.first_mismatches
        ]
    return result


def _scientific_result_value(
    result: CandidateScientificResult,
) -> dict[str, object]:
    value: dict[str, object] = {"status": result.status}
    if result.status != "structurally_incomparable":
        assert result.floating_components is not None
        assert result.discrete_components is not None
        assert result.metrics is not None
        value["floating_components"] = {
            component.name: _floating_statistics_value(component.statistics)
            for component in result.floating_components
        }
        value["discrete_components"] = {
            component.name: _discrete_component_value(component)
            for component in result.discrete_components
        }
        value["metrics"] = [
            {
                "absolute_delta": metric.absolute_delta,
                "candidate_value": metric.candidate_value,
                "metric_name": metric.metric_name,
                "reference_value": metric.reference_value,
            }
            for metric in result.metrics
        ]
    return value


def _scientific_comparison_value(
    comparison: ScientificComparison,
) -> dict[str, object]:
    value = {
        "attempts": [_attempt_value(attempt) for attempt in comparison.attempts],
        "candidates": [
            _run_identity_value(candidate) for candidate in comparison.candidates
        ],
        "comparability": {
            item.environment_id: {
                "comparable": item.comparable,
                "gates": dict(item.gates),
                "structural_components": {
                    component.name: _discrete_component_value(component)
                    for component in item.structural_components
                },
            }
            for item in comparison.comparability
        },
        "comparison_id": comparison.comparison_id,
        "generator": _generator_value(comparison.generator),
        "limitations": list(comparison.limitations),
        "milestone_id": comparison.milestone_id,
        "reference": _run_identity_value(comparison.reference),
        "schema_id": comparison.schema_id,
        "schema_version": comparison.schema_version,
        "scientific_results": {
            item.environment_id: _scientific_result_value(item)
            for item in comparison.scientific_results
        },
    }
    if comparison.policy is not None:
        value["policy"] = {
            "policy_id": comparison.policy.policy_id,
            "sha256": comparison.policy.sha256,
        }
    return value


def _timing_device(value: object) -> bool:
    return type(value) is str and (
        value == "cpu" or re.fullmatch(r"cuda:[0-9]+", value) is not None
    )


def _performance_benchmark(bundle: ComparableBundle) -> Mapping[str, object]:
    if bundle.kind != "benchmark" or not isinstance(bundle.benchmark_metadata, Mapping):
        raise ComparisonValidationError("performance run has no benchmark metadata")
    benchmark = _comparison_mapping(bundle.benchmark_metadata, "benchmark metadata")
    payload = _canonical_json(_thaw_comparison_json(benchmark))
    snapshot = _source_snapshot(bundle, "benchmark.json")
    if (
        len(payload) != snapshot.byte_count
        or hashlib.sha256(payload).hexdigest() != snapshot.sha256.lower()
    ):
        raise ComparisonValidationError(
            "benchmark metadata differs from its validated source snapshot"
        )
    return benchmark


def _methodology_compatibility_identity(
    methodology: Mapping[str, object],
) -> dict[str, object]:
    fields = (
        "cpu_timing_method",
        "repeat_count",
        "stage_inclusion_boundaries",
        "timing_unit",
        "warmup_count",
        "warmup_samples_in_statistics",
    )
    if any(name not in methodology for name in fields):
        raise ComparisonValidationError("benchmark methodology identity is incomplete")
    return {name: _thaw_comparison_json(methodology[name]) for name in fields}


def _performance_profile_identity(
    identity: ScientificRunIdentity,
) -> dict[str, object]:
    run = _comparison_mapping(identity.run, "performance profile run identity")
    workload = _comparison_dict(
        _thaw_comparison_json(run["scientific_workload"]),
        "performance scientific workload",
    )
    determinism = _comparison_dict(workload["determinism"], "performance determinism")
    determinism.pop("torch_cuda_seed_all", None)
    return {
        "benchmark_identity": _thaw_comparison_json(run["benchmark_identity"]),
        "category": run["category"],
        "inventory": _thaw_comparison_json(run["inventory"]),
        "profile_id": run["profile_id"],
        "scientific_workload": workload,
    }


def _validate_timing_methodology(
    methodology: Mapping[str, object], requested_device: str
) -> None:
    warmups = _comparison_integer(
        methodology.get("warmup_count"), "methodology warmup_count", positive=True
    )
    repeats = _comparison_integer(
        methodology.get("repeat_count"), "methodology repeat_count", positive=True
    )
    expected = _methodology(torch.device(requested_device), warmups, repeats)
    if _canonical_json(_thaw_comparison_json(methodology)) != _canonical_json(expected):
        raise ComparisonValidationError("benchmark methodology is invalid")


def _performance_exclusion_reason(
    descriptor: ScientificBundleDescriptor,
    reference_identity: ScientificRunIdentity,
    candidate_identity: ScientificRunIdentity,
    reference_workload: Mapping[str, object],
    reference_methodology: Mapping[str, object],
) -> (
    Literal[
        "evaluation_bundle",
        "unsupported_timing_device",
        "workload_mismatch",
        "methodology_mismatch",
    ]
    | None
):
    if descriptor.bundle.kind == "evaluation":
        return "evaluation_bundle"
    if not _timing_device(descriptor.requested_device):
        return "unsupported_timing_device"
    benchmark = _performance_benchmark(descriptor.bundle)
    workload = _comparison_mapping(benchmark.get("workload"), "candidate workload")
    if _performance_profile_identity(
        candidate_identity
    ) != _performance_profile_identity(reference_identity) or _canonical_json(
        _thaw_comparison_json(workload)
    ) != _canonical_json(_thaw_comparison_json(reference_workload)):
        return "workload_mismatch"
    methodology = _comparison_mapping(
        benchmark.get("methodology"), "candidate methodology"
    )
    try:
        _validate_timing_methodology(methodology, descriptor.requested_device)
    except ComparisonValidationError:
        return "methodology_mismatch"
    if _canonical_json(_methodology_compatibility_identity(methodology)) != (
        _canonical_json(_methodology_compatibility_identity(reference_methodology))
    ):
        return "methodology_mismatch"
    return None


def _performance_run(
    descriptor: ScientificBundleDescriptor,
) -> PortabilityPerformanceRun:
    benchmark = _performance_benchmark(descriptor.bundle)
    run = _comparison_mapping(descriptor.bundle.run_metadata, "performance run")
    return PortabilityPerformanceRun(
        environment_id=descriptor.environment_id,
        os_label=descriptor.os_label,
        execution_layer=descriptor.execution_layer,
        hardware_label=descriptor.hardware_label,
        requested_device=descriptor.requested_device,
        run_id=_comparison_string(run.get("run_id"), "performance run_id"),
        benchmark_sample_id=_comparison_string(
            benchmark.get("benchmark_sample_id"), "benchmark sample_id"
        ),
        timing_methodology=_comparison_mapping(
            benchmark.get("methodology"), "benchmark methodology"
        ),
        measurements=_comparison_mapping(benchmark.get("results"), "benchmark results"),
    )


def _performance_run_value(value: PortabilityPerformanceRun) -> dict[str, object]:
    return {
        "benchmark_sample_id": value.benchmark_sample_id,
        "environment_id": value.environment_id,
        "execution_layer": value.execution_layer,
        "hardware_label": value.hardware_label,
        "measurements": _thaw_comparison_json(value.measurements),
        "os_label": value.os_label,
        "requested_device": value.requested_device,
        "run_id": value.run_id,
        "timing_methodology": _thaw_comparison_json(value.timing_methodology),
    }


def _validate_performance_measurements(
    run: PortabilityPerformanceRun, workload: Mapping[str, object]
) -> None:
    repeats = _comparison_integer(
        run.timing_methodology.get("repeat_count"),
        "performance repeat_count",
        positive=True,
    )
    bank_bytes = _comparison_integer(
        workload.get("bank_bytes"), "performance bank_bytes", positive=True
    )
    results = _thaw_comparison_json(run.measurements)
    if type(results) is not dict:
        raise ComparisonValidationError("performance measurements are invalid")
    _validate_benchmark_results(
        results, repeats, bank_bytes, run.requested_device.startswith("cuda:")
    )


def _validate_performance_workload(workload: Mapping[str, object]) -> None:
    fields = {
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
    if set(workload) != fields:
        raise ComparisonValidationError("performance workload fields are invalid")
    rows = _comparison_integer(workload["M"], "performance workload M", positive=True)
    training = _comparison_integer(
        workload["training_sample_count"],
        "performance workload training_sample_count",
        positive=True,
    )
    _comparison_integer(
        workload["test_sample_count"],
        "performance workload test_sample_count",
        positive=True,
    )
    expected = {
        "D": _EMBEDDING_DIMENSION,
        "Q": _PATCH_COUNT,
        "bank_bytes": rows * _EMBEDDING_DIMENSION * 4,
        "bank_shape": [rows, _EMBEDDING_DIMENSION],
        "batch_size": 1,
        "dtype": "float32",
        "k": 1,
        "tensor_layout": {
            "anomaly_map": "BHW contiguous row-major",
            "image": "NCHW contiguous",
            "memory_bank": "MD contiguous row-major",
            "patch_embeddings": "BQD contiguous row-major",
        },
    }
    if rows != training * _PATCH_COUNT or any(
        _thaw_comparison_json(workload[name]) != value
        for name, value in expected.items()
    ):
        raise ComparisonValidationError("performance workload is invalid")
    _comparison_integer(
        workload["bank_chunk_size"],
        "performance workload bank_chunk_size",
        positive=True,
    )


def _portability_performance_value(
    performance: PortabilityPerformance,
) -> dict[str, object]:
    return {
        "attempts": [_attempt_value(attempt) for attempt in performance.attempts],
        "comparison_id": performance.comparison_id,
        "excluded_candidates": [
            {
                "environment_id": item.environment_id,
                "reason_code": item.reason_code,
            }
            for item in performance.excluded_candidates
        ],
        "generator": _generator_value(performance.generator),
        "included_runs": [
            _performance_run_value(item) for item in performance.included_runs
        ],
        "limitations": list(performance.limitations),
        "milestone_id": performance.milestone_id,
        "schema_id": performance.schema_id,
        "schema_version": performance.schema_version,
        "scientific_sha256": performance.scientific_sha256,
        "status": performance.status,
        "timing_methodology": _thaw_comparison_json(performance.timing_methodology),
        "workload": _thaw_comparison_json(performance.workload),
    }


def _validate_portability_performance(performance: PortabilityPerformance) -> None:
    if (
        type(performance.schema_version) is not int
        or performance.schema_version != 1
        or type(performance.schema_id) is not str
        or performance.schema_id != _PERFORMANCE_SCHEMA_ID
        or type(performance.milestone_id) is not str
        or performance.milestone_id != _MILESTONE_ID
        or type(performance.status) is not str
        or performance.status != "descriptive_only"
        or not re.fullmatch(r"[0-9a-f]{64}", performance.comparison_id)
    ):
        raise ComparisonValidationError("performance identity is invalid")
    _comparison_sha256(performance.scientific_sha256, "scientific_sha256")
    if type(performance.generator) is not ScientificGenerator:
        raise ComparisonValidationError("performance generator is invalid")
    if not isinstance(performance.workload, Mapping) or not isinstance(
        performance.timing_methodology, Mapping
    ):
        raise ComparisonValidationError("performance benchmark identity is invalid")
    _validate_performance_workload(performance.workload)
    methodology_fields = {
        "cpu_timing_method",
        "repeat_count",
        "stage_inclusion_boundaries",
        "timing_unit",
        "warmup_count",
        "warmup_samples_in_statistics",
    }
    if set(performance.timing_methodology) != methodology_fields:
        raise ComparisonValidationError(
            "performance timing methodology fields are invalid"
        )
    if (
        type(performance.included_runs) is not tuple
        or not performance.included_runs
        or any(
            type(item) is not PortabilityPerformanceRun
            for item in performance.included_runs
        )
        or type(performance.excluded_candidates) is not tuple
        or any(
            type(item) is not PortabilityPerformanceExclusion
            for item in performance.excluded_candidates
        )
        or type(performance.attempts) is not tuple
        or any(
            type(item) is not ScientificExecutionAttempt
            for item in performance.attempts
        )
    ):
        raise ComparisonValidationError("performance run records are invalid")
    included_ids = tuple(item.environment_id for item in performance.included_runs)
    excluded_ids = tuple(
        item.environment_id for item in performance.excluded_candidates
    )
    attempt_ids = tuple(item.environment_id for item in performance.attempts)
    all_ids = (*included_ids, *excluded_ids, *attempt_ids)
    if len(all_ids) != len(set(all_ids)):
        raise ComparisonValidationError("performance environment IDs are not unique")
    for item in performance.included_runs:
        _validate_timing_methodology(item.timing_methodology, item.requested_device)
        if _canonical_json(
            _methodology_compatibility_identity(item.timing_methodology)
        ) != _canonical_json(_thaw_comparison_json(performance.timing_methodology)):
            raise ComparisonValidationError(
                "included run methodology differs from the common methodology"
            )
        _validate_performance_measurements(item, performance.workload)
    if type(performance.limitations) is not tuple or any(
        type(value) is not str or not value for value in performance.limitations
    ):
        raise ComparisonValidationError("performance limitations are invalid")
    _reject_absolute_identity_values(
        _portability_performance_value(performance), "performance record"
    )


def _write_portability_file(path: Path, payload: bytes) -> None:
    path.write_bytes(payload)


def _validate_scientific_run_identity(identity: ScientificRunIdentity) -> None:
    _environment_id(identity.environment_id, "run identity environment_id")
    if (
        type(identity.policy_role) is not str
        or identity.policy_role not in _POLICY_ROLES
    ):
        raise ComparisonValidationError("run identity policy_role is invalid")
    if type(identity.bundle_kind) is not str or identity.bundle_kind not in {
        "evaluation",
        "benchmark",
    }:
        raise ComparisonValidationError("run identity bundle_kind is invalid")
    _public_label(identity.os_label, "run identity os_label")
    if (
        type(identity.execution_layer) is not str
        or identity.execution_layer not in _EXECUTION_LAYERS
    ):
        raise ComparisonValidationError("run identity execution_layer is invalid")
    _public_label(identity.hardware_label, "run identity hardware_label")
    if type(identity.requested_device) is not str or not _REQUESTED_DEVICE.fullmatch(
        identity.requested_device
    ):
        raise ComparisonValidationError("run identity requested_device is invalid")

    run = _comparison_mapping(identity.run, "run identity")
    expected_run_fields = {
        "category",
        "dependency_versions",
        "inventory",
        "profile_id",
        "python_version",
        "run_id",
        "schema_version",
        "scientific_workload",
        "source",
        "weights",
    }
    if identity.bundle_kind == "benchmark":
        expected_run_fields.add("benchmark_identity")
    if set(run) != expected_run_fields:
        raise ComparisonValidationError("run identity fields are invalid")
    category = _comparison_string(run["category"], "run identity category")
    if "/" in category or "\\" in category:
        raise ComparisonValidationError("run identity category is invalid")
    run_id = _comparison_string(run["run_id"], "run identity run_id")
    if not _RUN_ID.fullmatch(run_id):
        raise ComparisonValidationError("run identity run_id is invalid")
    _comparison_string(run["profile_id"], "run identity profile_id")
    _portable_identity_string(run["python_version"], "run identity python_version")
    dependencies = _comparison_mapping(
        run["dependency_versions"], "run identity dependency_versions"
    )
    for name, version in dependencies.items():
        _comparison_string(name, "run identity dependency name")
        _portable_identity_string(version, f"run identity dependency {name}")

    source = _comparison_mapping(run["source"], "run identity source")
    if set(source) != {"dirty", "git_commit", "uv_lock_sha256"}:
        raise ComparisonValidationError("run identity source fields are invalid")
    _comparison_boolean(source["dirty"], "run identity source dirty")
    _comparison_commit(source["git_commit"], "run identity source commit")
    _comparison_sha256(source["uv_lock_sha256"], "run identity lock")
    weights = _comparison_mapping(run["weights"], "run identity weights")
    if set(weights) != {"cached_file_sha256", "enum"}:
        raise ComparisonValidationError("run identity weight fields are invalid")
    _comparison_sha256(weights["cached_file_sha256"], "run identity weight hash")
    _comparison_string(weights["enum"], "run identity weight enum")
    inventory = _comparison_mapping(run["inventory"], "run identity inventory")
    _comparison_sha256(
        inventory.get("sample_inventory_sha256"), "run identity inventory hash"
    )
    _comparison_mapping(run["scientific_workload"], "run identity scientific_workload")
    if identity.bundle_kind == "benchmark":
        benchmark_identity = _comparison_mapping(
            run["benchmark_identity"], "run benchmark identity"
        )
        if set(benchmark_identity) != {"benchmark_sample_id", "schema_version"}:
            raise ComparisonValidationError("run benchmark identity fields are invalid")
        _portable_identity_string(
            benchmark_identity["benchmark_sample_id"],
            "run benchmark sample ID",
        )
        _comparison_integer(
            benchmark_identity["schema_version"],
            "run benchmark schema",
            positive=True,
        )
    _comparison_integer(run["schema_version"], "run identity schema", positive=True)
    _reject_absolute_identity_values(run, "run identity")

    expected_files = (
        _EVALUATION_FILES if identity.bundle_kind == "evaluation" else _BENCHMARK_FILES
    )
    if (
        type(identity.source_files) is not tuple
        or any(
            type(snapshot) is not SourceFileSnapshot
            for snapshot in identity.source_files
        )
        or tuple(snapshot.name for snapshot in identity.source_files) != expected_files
    ):
        raise ComparisonValidationError("run identity source files are invalid")
    for snapshot in identity.source_files:
        if (
            type(snapshot.byte_count) is not int
            or snapshot.byte_count < 0
            or type(snapshot.sha256) is not str
            or not _SHA256.fullmatch(snapshot.sha256)
        ):
            raise ComparisonValidationError("run identity source snapshot is invalid")
    if identity.bundle_kind == "evaluation":
        if (
            identity.benchmark_workload is not None
            or identity.benchmark_methodology is not None
        ):
            raise ComparisonValidationError(
                "evaluation identity must omit benchmark metadata"
            )
    elif not isinstance(identity.benchmark_workload, Mapping) or not isinstance(
        identity.benchmark_methodology, Mapping
    ):
        raise ComparisonValidationError(
            "benchmark identity must include workload and methodology"
        )
    else:
        _reject_absolute_identity_values(
            identity.benchmark_workload, "benchmark workload"
        )
        _reject_absolute_identity_values(
            identity.benchmark_methodology, "benchmark methodology"
        )


def _portable_identity_string(value: object, name: str) -> str:
    result = _comparison_string(value, name)
    if (
        re.search(r"(?<![A-Za-z0-9_.-])/(?!/)", result)
        or re.search(r"(?<![A-Za-z0-9_.-])~(?:[A-Za-z0-9_.-]+)?(?:[\\/]|$)", result)
        or re.search(r"(?<![A-Za-z0-9_.-])[A-Za-z]:[\\/]", result)
        or re.search(r"(?<![A-Za-z0-9_.-])\\", result)
        or "\n" in result
        or "\r" in result
    ):
        raise ComparisonValidationError(f"{name} contains private path-like data")
    return result


def _reject_absolute_identity_values(value: object, name: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _portable_identity_string(key, f"{name} key")
            _reject_absolute_identity_values(item, f"{name}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _reject_absolute_identity_values(item, f"{name}[{index}]")
    elif type(value) is str:
        _portable_identity_string(value, name)


def _validate_scientific_comparison(comparison: ScientificComparison) -> None:
    if (
        type(comparison.schema_version) is not int
        or comparison.schema_version != 1
        or type(comparison.schema_id) is not str
        or comparison.schema_id != _SCHEMA_ID
        or type(comparison.milestone_id) is not str
        or comparison.milestone_id != _MILESTONE_ID
        or not re.fullmatch(r"[0-9a-f]{64}", comparison.comparison_id)
    ):
        raise ComparisonValidationError("scientific comparison identity is invalid")
    if type(comparison.generator) is not ScientificGenerator:
        raise ComparisonValidationError("scientific comparison generator is invalid")
    if comparison.policy is not None and type(comparison.policy) is not (
        PortabilityPolicyIdentity
    ):
        raise ComparisonValidationError("scientific comparison policy is invalid")
    if (
        type(comparison.reference) is not ScientificRunIdentity
        or comparison.reference.policy_role != "reference"
        or not comparison.candidates
    ):
        raise ComparisonValidationError("scientific comparison run roles are invalid")
    if any(
        type(candidate) is not ScientificRunIdentity
        or candidate.policy_role not in _CANDIDATE_ROLES
        for candidate in comparison.candidates
    ):
        raise ComparisonValidationError("scientific comparison candidates are invalid")
    _validate_scientific_run_identity(comparison.reference)
    for candidate in comparison.candidates:
        _validate_scientific_run_identity(candidate)
    candidate_ids = tuple(
        candidate.environment_id for candidate in comparison.candidates
    )
    if (
        tuple(item.environment_id for item in comparison.comparability) != candidate_ids
        or tuple(item.environment_id for item in comparison.scientific_results)
        != candidate_ids
        or len(candidate_ids) != len(set(candidate_ids))
    ):
        raise ComparisonValidationError(
            "scientific comparison candidate ordering is inconsistent"
        )
    all_ids = (
        comparison.reference.environment_id,
        *candidate_ids,
        *(attempt.environment_id for attempt in comparison.attempts),
    )
    if len(all_ids) != len(set(all_ids)):
        raise ComparisonValidationError(
            "scientific comparison environment IDs are not unique"
        )
    for attempt in comparison.attempts:
        if type(attempt) is not ScientificExecutionAttempt:
            raise ComparisonValidationError("scientific comparison attempt is invalid")
    identity_payload = {
        "attempts": [_attempt_value(attempt) for attempt in comparison.attempts],
        "candidates": [
            _run_identity_value(candidate) for candidate in comparison.candidates
        ],
        "generator": _generator_value(comparison.generator),
        "milestone_id": comparison.milestone_id,
        "reference": _run_identity_value(comparison.reference),
        "schema_id": comparison.schema_id,
        "schema_version": comparison.schema_version,
    }
    if comparison.policy is not None:
        identity_payload["policy_sha256"] = comparison.policy.sha256
    expected_id = hashlib.sha256(_canonical_json(identity_payload)).hexdigest()
    if comparison.comparison_id != expected_id:
        raise ComparisonValidationError(
            "scientific comparison ID differs from its canonical inputs"
        )
    for comparable, result in zip(
        comparison.comparability, comparison.scientific_results, strict=True
    ):
        if (
            type(comparable) is not CandidateComparability
            or type(result) is not CandidateScientificResult
            or type(comparable.gates) is not tuple
            or any(
                type(gate) is not tuple
                or len(gate) != 2
                or type(gate[0]) is not str
                or type(gate[1]) is not bool
                for gate in comparable.gates
            )
            or comparable.comparable != all(value for _, value in comparable.gates)
            or tuple(name for name, _ in comparable.gates) != _SCIENTIFIC_GATE_NAMES
            or any(
                type(component) is not DiscreteComponentComparison
                for component in comparable.structural_components
            )
            or tuple(component.name for component in comparable.structural_components)
            != ("test_sample_ids", "test_labels")
            or type(result.status) is not str
            or result.status not in _CANDIDATE_STATUSES
        ):
            raise ComparisonValidationError(
                "scientific comparison candidate result is invalid"
            )
        if comparable.comparable != (result.status != "structurally_incomparable"):
            raise ComparisonValidationError(
                "scientific comparison status differs from compatibility gates"
            )
        if comparison.policy is None and result.status not in {
            "structurally_incomparable",
            "observed_unclassified",
        }:
            raise ComparisonValidationError(
                "observation comparison contains a policy classification"
            )
        if comparison.policy is not None and result.status == "observed_unclassified":
            raise ComparisonValidationError(
                "policy comparison contains an unclassified candidate"
            )
        for component in comparable.structural_components:
            if (
                type(component) is not DiscreteComponentComparison
                or component.element_count <= 0
                or component.exact_count + component.mismatch_count
                != component.element_count
                or component.exact != (component.mismatch_count == 0)
                or not math.isfinite(component.mismatch_rate)
                or not 0 <= component.mismatch_rate <= 1
            ):
                raise ComparisonValidationError(
                    "structural comparison statistics are invalid"
                )
        measured = (
            result.floating_components,
            result.discrete_components,
            result.metrics,
        )
        if result.status == "structurally_incomparable":
            if any(value is not None for value in measured):
                raise ComparisonValidationError(
                    "incomparable candidates must not contain measurements"
                )
            continue
        if any(value is None for value in measured):
            raise ComparisonValidationError(
                "comparable candidates must contain measurements"
            )
        assert result.floating_components is not None
        assert result.discrete_components is not None
        assert result.metrics is not None
        if (
            any(
                type(component) is not FloatingComponentComparison
                for component in result.floating_components
            )
            or tuple(component.name for component in result.floating_components)
            != _FLOATING_COMPONENTS
        ):
            raise ComparisonValidationError(
                "floating comparison component order is invalid"
            )
        if (
            any(
                type(component) is not DiscreteComponentComparison
                for component in result.discrete_components
            )
            or tuple(component.name for component in result.discrete_components)
            != _DISCRETE_REQUIREMENTS
        ):
            raise ComparisonValidationError(
                "discrete comparison component order is invalid"
            )
        for component in result.floating_components:
            statistics = component.statistics
            numbers = (
                statistics.maximum_absolute_error,
                statistics.mean_absolute_error,
                statistics.root_mean_square_error,
            )
            if (
                statistics.element_count <= 0
                or statistics.exact_count + statistics.differing_count
                != statistics.element_count
                or statistics.zero_reference_count > statistics.element_count
                or (
                    statistics.policy_violation_count is not None
                    and (
                        type(statistics.policy_violation_count) is not int
                        or not 0
                        <= statistics.policy_violation_count
                        <= statistics.element_count
                    )
                )
                or any(not math.isfinite(value) or value < 0 for value in numbers)
                or (
                    statistics.maximum_relative_error is not None
                    and (
                        not math.isfinite(statistics.maximum_relative_error)
                        or statistics.maximum_relative_error < 0
                    )
                )
            ):
                raise ComparisonValidationError(
                    "floating comparison statistics are invalid"
                )
            if (statistics.policy_violation_count is None) != (
                comparison.policy is None
            ):
                raise ComparisonValidationError(
                    "floating policy counts differ from comparison mode"
                )
        for component in result.discrete_components:
            if (
                component.element_count <= 0
                or component.exact_count + component.mismatch_count
                != component.element_count
                or component.exact != (component.mismatch_count == 0)
                or not math.isfinite(component.mismatch_rate)
                or not 0 <= component.mismatch_rate <= 1
                or len(component.first_mismatches) > _INDEX_MISMATCH_LIMIT
            ):
                raise ComparisonValidationError(
                    "discrete comparison statistics are invalid"
                )
        if (
            tuple(metric.metric_name for metric in result.metrics)
            != _SCIENTIFIC_METRICS
        ):
            raise ComparisonValidationError("metric comparison order is invalid")
        if any(
            not all(
                math.isfinite(value)
                for value in (
                    metric.reference_value,
                    metric.candidate_value,
                    metric.absolute_delta,
                )
            )
            for metric in result.metrics
        ):
            raise ComparisonValidationError("metric comparison values are invalid")
        floating_violation = any(
            component.statistics.policy_violation_count != 0
            for component in result.floating_components
        )
        discrete_violation = any(
            not component.exact for component in result.discrete_components
        )
        metric_difference = any(metric.absolute_delta != 0 for metric in result.metrics)
        if result.status == "within_policy" and (
            floating_violation or discrete_violation
        ):
            raise ComparisonValidationError(
                "within_policy result contains a policy violation"
            )
        if result.status == "drift_detected" and not (
            floating_violation or discrete_violation or metric_difference
        ):
            raise ComparisonValidationError(
                "drift_detected result contains no possible policy violation"
            )
    if type(comparison.limitations) is not tuple or any(
        type(value) is not str or not value for value in comparison.limitations
    ):
        raise ComparisonValidationError("scientific comparison limitations are invalid")
