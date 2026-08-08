from dataclasses import FrozenInstanceError, fields, replace
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
from typing import Any

import pytest
import torch

import inspectrt.benchmark as benchmark_module
import inspectrt.portability as portability
from inspectrt.artifacts import BaselineRunMetadata, persist_baseline_run
from inspectrt.benchmark import BaselineBenchmark
from inspectrt.data import MvtecSample
from inspectrt.evaluation import CategoryEvaluation, MvtecSampleObservation
from inspectrt.metrics import compute_threshold_free_metrics
from inspectrt.portability import (
    BundleMetrics,
    BundleValidationError,
    CandidateComparability,
    CandidateScientificResult,
    CanonicalInputIdentity,
    ComparableBundle,
    ComparisonValidationError,
    DiscreteComponentComparison,
    FloatingComponentComparison,
    FloatingStatistics,
    IndexMismatch,
    MemoryBankMetadata,
    MetricDelta,
    PolicyDerivation,
    PolicyTolerance,
    PortabilityEnvironmentDescriptor,
    PortabilityEnvironmentMap,
    PortabilityPerformance,
    PortabilityPerformanceExclusion,
    PortabilityPerformanceRun,
    PortabilityPolicy,
    PortabilityPolicyIdentity,
    PredictionRecord,
    ScientificBundleDescriptor,
    ScientificComparison,
    ScientificExecutionAttempt,
    ScientificGenerator,
    ScientificRunIdentity,
    SourceFileSnapshot,
    build_portability_performance,
    compare_scientific_bundles,
    encode_portability_performance,
    encode_scientific_comparison,
    load_comparable_bundle,
    load_portability_environment_map,
    load_portability_policy,
    publish_portability_comparison,
    publish_portability_records,
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
_REPEATED_STAGES = (
    "image_decode",
    "canonical_image_preprocessing",
    "host_to_device_transfer",
    "frozen_feature_extraction",
    "exact_chunked_retrieval",
    "anomaly_map_reconstruction",
)
_DELETE = object()
_EXECUTED_UNSAFE_PAYLOAD = False
_SCIENTIFIC_SOURCE_COMMIT = "bc330b9070c5ca8db9cb7cfbb27617256388536b"
_ACCEPTED_LOCK_SHA256 = (
    "ddaddc99b318a1c3a04d5d7cc433cf736d321b56f98a8ae8b532e71e19e6d76b"
)
_ACCEPTED_WEIGHT_SHA256 = (
    "11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca"
)


def _observation(
    sample_id: str,
    split: str,
    defect_type: str,
    is_anomalous: bool,
    image_relpath: str,
    mask_relpath: str | None = None,
) -> MvtecSampleObservation:
    return MvtecSampleObservation(
        MvtecSample(
            sample_id,
            "bottle",
            split,
            defect_type,
            is_anomalous,
            image_relpath,
            mask_relpath,
        ),
        32,
        32,
        "RGB",
    )


def _evaluation() -> CategoryEvaluation:
    samples = tuple(
        sorted(
            (
                _observation(
                    "mvtec_ad/bottle/train/good/000.png",
                    "train",
                    "good",
                    False,
                    "bottle/train/good/000.png",
                ),
                _observation(
                    "mvtec_ad/bottle/test/good/001.png",
                    "test",
                    "good",
                    False,
                    "bottle/test/good/001.png",
                ),
                _observation(
                    "mvtec_ad/bottle/test/crack/002.png",
                    "test",
                    "crack",
                    True,
                    "bottle/test/crack/002.png",
                    "bottle/ground_truth/crack/002_mask.png",
                ),
            ),
            key=lambda item: item.sample.sample_id,
        )
    )
    test_samples = tuple(item for item in samples if item.sample.split == "test")
    labels = torch.tensor((1, 0), dtype=torch.uint8)
    masks = torch.zeros((2, 256, 256), dtype=torch.uint8)
    masks[0, 0, 0] = 1
    maps = torch.zeros((2, 256, 256), dtype=torch.float32)
    maps[0, 0, 0] = 1.0
    distances = torch.full((2, 1024), 0.1, dtype=torch.float32)
    distances[0].fill_(0.2)
    distances[0, -1] = 0.9
    scores = distances.max(dim=1).values.contiguous()
    return CategoryEvaluation(
        category="bottle",
        samples=samples,
        test_samples=test_samples,
        memory_bank=torch.zeros((1024, 512), dtype=torch.float32),
        test_labels=labels,
        pixel_masks=masks,
        patch_distances=distances,
        nearest_bank_indices=torch.arange(1024, dtype=torch.int64)
        .expand(2, -1)
        .contiguous(),
        image_scores=scores,
        anomaly_maps=maps,
        metrics=compute_threshold_free_metrics(labels, scores, masks, maps),
    )


def _metadata(
    run_id: str, device: str = "cpu", *, bank_chunk_size: int = 16_384
) -> BaselineRunMetadata:
    return BaselineRunMetadata(
        run_id=run_id,
        created_at_utc="2026-07-15T12:00:00Z",
        dataset_root="/private/mvtec-ad",
        requested_device=device,
        bank_chunk_size=bank_chunk_size,
        git_commit=_SCIENTIFIC_SOURCE_COMMIT,
        git_dirty=False,
        uv_lock_sha256=_ACCEPTED_LOCK_SHA256,
        python_version="3.11.15",
        platform_description="Linux-test",
        dependency_versions={
            "inspectrt": "0.1.0",
            "numpy": "2.4.6",
            "pillow": "12.3.0",
            "scikit-learn": "1.9.0",
            "torch": "2.13.0",
            "torchvision": "0.28.0",
        },
        determinism_flags={
            "allow_tf32": False,
            "cublas_workspace_config": ":4096:8",
            "cudnn_benchmark": False,
            "deterministic_algorithms_warn_only": False,
            "fp32_precision": "ieee",
            "numpy_seed": 0,
            "python_random_seed": 0,
            "torch_cpu_seed": 0,
            "torch_cuda_seed_all": (0 if torch.device(device).type == "cuda" else None),
            "use_deterministic_algorithms": True,
        },
        weight_enum="ResNet50_Weights.IMAGENET1K_V2",
        weight_source_url=("https://download.pytorch.org/models/resnet50-11ad3fa6.pth"),
        weight_file_sha256=_ACCEPTED_WEIGHT_SHA256,
    )


def _benchmark(
    evaluation: CategoryEvaluation,
    metadata: BaselineRunMetadata,
    *,
    repeats: int = 2,
) -> BaselineBenchmark:
    statistics = benchmark_module._statistics(
        tuple(float(value) for value in range(1, repeats + 1))
    )
    bank_bytes = evaluation.memory_bank.numel() * evaluation.memory_bank.element_size()
    device = torch.device(metadata.requested_device)
    cuda = device.type == "cuda"
    return BaselineBenchmark(
        schema_version=1,
        profile_id="inspectrt_feature_memory_v1",
        category=evaluation.category,
        device=metadata.requested_device,
        benchmark_sample_id=evaluation.test_samples[0].sample.sample_id,
        run_id=metadata.run_id,
        created_at_utc=metadata.created_at_utc,
        workload={
            "D": 512,
            "M": 1024,
            "Q": 1024,
            "bank_bytes": bank_bytes,
            "bank_chunk_size": metadata.bank_chunk_size,
            "bank_shape": [1024, 512],
            "batch_size": 1,
            "dtype": "float32",
            "k": 1,
            "tensor_layout": {
                "anomaly_map": "BHW contiguous row-major",
                "image": "NCHW contiguous",
                "memory_bank": "MD contiguous row-major",
                "patch_embeddings": "BQD contiguous row-major",
            },
            "test_sample_count": 2,
            "training_sample_count": 1,
        },
        methodology=benchmark_module._methodology(device, 1, repeats),
        environment=(
            {
                "cuda_compute_capability": [7, 5],
                "cuda_device_name": "Synthetic GPU",
                "pytorch_cuda_runtime_version": "13.0",
            }
            if cuda
            else benchmark_module._cuda_environment(device)
        ),
        results={
            "device_memory": {
                "peak_allocated_bytes": bank_bytes if cuda else None,
                "peak_reserved_bytes": bank_bytes if cuda else None,
                "persistent_bank_bytes": bank_bytes,
                "peak_allocated_boundary": (
                    "Reset after setup and warm-ups; peak covers persistent model and "
                    "full-bank allocations plus PyTorch allocator activity during "
                    "measured repeats, excluding setup and warm-up activity, full-category "
                    "scoring, driver memory, and non-PyTorch allocations."
                    if cuda
                    else "Not measured on CPU; no host peak approximation is made."
                ),
                "peak_reserved_boundary": (
                    "Reset after setup and warm-ups; the reset retains the CUDA caching "
                    "pool, so the peak includes reservations retained from setup and "
                    "warm-ups plus any growth during measured repeats."
                    if cuda
                    else "Not measured on CPU; no host reservation approximation is made."
                ),
            },
            "one_off_ms": {
                "bank_transfer_and_device_setup": 1.0,
                "full_nominal_bank_build": 2.0,
                "model_and_weight_load": 3.0,
            },
            "repeated_stages": {name: dict(statistics) for name in _REPEATED_STAGES},
            "synchronized_end_to_end": dict(statistics),
        },
    )


def _bundle(
    tmp_path: Path,
    *,
    benchmark: bool = False,
    device: str = "cpu",
    bank_chunk_size: int = 16_384,
    repeats: int = 2,
) -> Path:
    evaluation = _evaluation()
    run_id = (
        "tiny-evaluation"
        if not benchmark
        else (
            "tiny-benchmark"
            if device == "cpu"
            else f"tiny-benchmark-{device.replace(':', '-')}"
        )
    )
    metadata = _metadata(run_id, device, bank_chunk_size=bank_chunk_size)
    return persist_baseline_run(
        evaluation,
        tmp_path,
        metadata,
        benchmark=(
            _benchmark(evaluation, metadata, repeats=repeats) if benchmark else None
        ),
    )


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    assert isinstance(value, dict)
    return value


def _write_json(path: Path, value: object) -> None:
    path.write_bytes(_canonical(value))


def _records(path: Path) -> list[dict[str, Any]]:
    values = [json.loads(line) for line in path.read_bytes().splitlines()]
    assert all(isinstance(value, dict) for value in values)
    return values


def _write_records(path: Path, values: list[dict[str, Any]]) -> None:
    path.write_bytes(b"".join(_canonical(value) for value in values))


def _payload(path: Path) -> dict[str, Any]:
    value = torch.load(path, map_location="cpu", weights_only=True)
    assert isinstance(value, dict)
    return value


def _set_path(root: dict[str, Any], path: tuple[str, ...], value: object) -> None:
    parent = root
    for name in path[:-1]:
        child = parent[name]
        assert isinstance(child, dict)
        parent = child
    if value is _DELETE:
        del parent[path[-1]]
    else:
        parent[path[-1]] = value


def _rewrite_samples(bundle: Path, records: list[dict[str, Any]]) -> None:
    sample_bytes = b"".join(_canonical(record) for record in records)
    (bundle / "samples.jsonl").write_bytes(sample_bytes)
    run = _json(bundle / "run.json")
    run["inventory"]["sample_inventory_sha256"] = hashlib.sha256(
        sample_bytes
    ).hexdigest()
    _write_json(bundle / "run.json", run)


def _source_state(directory: Path) -> tuple[tuple[object, ...], ...]:
    state: list[tuple[object, ...]] = []
    directory_stat = directory.lstat()
    state.append(
        (
            ".",
            directory_stat.st_dev,
            directory_stat.st_ino,
            stat.S_IFMT(directory_stat.st_mode),
            stat.S_IMODE(directory_stat.st_mode),
            directory_stat.st_nlink,
            directory_stat.st_uid,
            directory_stat.st_gid,
            directory_stat.st_size,
            directory_stat.st_mtime_ns,
            directory_stat.st_ctime_ns,
        )
    )
    for path in sorted(directory.iterdir(), key=lambda item: item.name):
        metadata = path.lstat()
        digest = (
            hashlib.sha256(path.read_bytes()).hexdigest()
            if stat.S_ISREG(metadata.st_mode)
            else None
        )
        state.append(
            (
                path.name,
                metadata.st_dev,
                metadata.st_ino,
                stat.S_IFMT(metadata.st_mode),
                stat.S_IMODE(metadata.st_mode),
                metadata.st_nlink,
                metadata.st_uid,
                metadata.st_gid,
                metadata.st_size,
                metadata.st_mtime_ns,
                metadata.st_ctime_ns,
                digest,
            )
        )
    return tuple(state)


def _unsafe_payload_was_loaded() -> dict[str, object]:
    global _EXECUTED_UNSAFE_PAYLOAD
    _EXECUTED_UNSAFE_PAYLOAD = True
    return {}


class _UnsafePayload:
    def __reduce__(self) -> tuple[object, tuple[()]]:
        return _unsafe_payload_was_loaded, ()


def test_loads_valid_evaluation_bundle_as_immutable_typed_records(
    tmp_path: Path,
) -> None:
    source = _bundle(tmp_path)
    loaded = load_comparable_bundle(source)

    assert isinstance(loaded, ComparableBundle)
    assert loaded.path == source
    assert loaded.kind == "evaluation"
    assert loaded.run_metadata["run_id"] == "tiny-evaluation"
    assert loaded.benchmark_metadata is None
    assert tuple(item.sample_id for item in loaded.samples) == (
        "mvtec_ad/bottle/test/crack/002.png",
        "mvtec_ad/bottle/test/good/001.png",
        "mvtec_ad/bottle/train/good/000.png",
    )
    assert loaded.test_sample_ids == tuple(
        item.sample_id for item in loaded.predictions
    )
    assert loaded.test_labels.tolist() == [1, 0]
    assert loaded.image_scores.tolist() == pytest.approx([0.9, 0.1])
    assert loaded.memory_bank_metadata == MemoryBankMetadata(
        dtype="float32",
        shape=(1024, 512),
        embedding_dimension=512,
        patches_per_training_sample=1024,
    )
    assert loaded.memory_bank.shape == (1024, 512)
    assert loaded.patch_distances.shape == (2, 1024)
    assert loaded.nearest_bank_indices.shape == (2, 1024)
    assert loaded.anomaly_maps.shape == (2, 256, 256)
    assert loaded.evaluation_masks.shape == (2, 256, 256)
    assert loaded.metrics == BundleMetrics(
        anomalous_pixel_count=1,
        anomalous_test_sample_count=1,
        evaluated_pixel_count=2 * 256 * 256,
        image_auroc=1.0,
        image_average_precision=1.0,
        pixel_auroc=1.0,
        test_good_sample_count=1,
        test_sample_count=2,
        training_sample_count=1,
    )
    assert isinstance(loaded.predictions[0], PredictionRecord)
    with pytest.raises(FrozenInstanceError):
        loaded.kind = "benchmark"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        loaded.predictions[0].image_score = 0.0  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        loaded.source_files[0].byte_count = 0  # type: ignore[misc]
    with pytest.raises(TypeError):
        loaded.run_metadata["run_id"] = "changed"  # type: ignore[index]
    assert loaded.run_metadata["tensors"]["memory_bank"]["shape"] == (1024, 512)


def test_loads_valid_benchmark_and_classifies_exactly(tmp_path: Path) -> None:
    evaluation = load_comparable_bundle(_bundle(tmp_path))
    benchmark = load_comparable_bundle(_bundle(tmp_path, benchmark=True))

    assert evaluation.kind == "evaluation"
    assert evaluation.benchmark_metadata is None
    assert tuple(item.name for item in evaluation.source_files) == _EVALUATION_FILES
    assert benchmark.kind == "benchmark"
    assert benchmark.benchmark_metadata is not None
    assert benchmark.benchmark_metadata["run_id"] == "tiny-benchmark"
    assert tuple(item.name for item in benchmark.source_files) == _BENCHMARK_FILES


def test_accepts_writer_compatible_seeds_and_uppercase_sha256(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    run = _json(bundle / "run.json")
    for name in (
        "numpy_seed",
        "python_random_seed",
        "torch_cpu_seed",
    ):
        run["determinism"][name] = 7
    run["source"]["uv_lock_sha256"] = "A" * 64
    run["weights"]["cached_file_sha256"] = "B" * 64
    _write_json(bundle / "run.json", run)

    loaded = load_comparable_bundle(bundle)

    assert loaded.run_metadata["determinism"]["numpy_seed"] == 7


def test_loads_cuda_benchmark_without_accessing_hardware(tmp_path: Path) -> None:
    loaded = load_comparable_bundle(_bundle(tmp_path, benchmark=True, device="cuda:0"))

    assert loaded.kind == "benchmark"
    assert loaded.run_metadata["device"] == "cuda:0"
    assert loaded.benchmark_metadata is not None
    assert loaded.benchmark_metadata["environment"] == {
        "cuda_compute_capability": (7, 5),
        "cuda_device_name": "Synthetic GPU",
        "pytorch_cuda_runtime_version": "13.0",
    }


def test_cuda_run_requires_recorded_cuda_seed(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path, device="cuda:0")
    run = _json(bundle / "run.json")
    run["determinism"]["torch_cuda_seed_all"] = None
    _write_json(bundle / "run.json", run)

    with pytest.raises(BundleValidationError):
        load_comparable_bundle(bundle)


def test_loader_does_not_apply_future_timing_eligibility(tmp_path: Path) -> None:
    loaded = load_comparable_bundle(_bundle(tmp_path, benchmark=True, device="mps"))

    assert loaded.kind == "benchmark"
    assert loaded.run_metadata["device"] == "mps"
    assert loaded.benchmark_metadata is not None
    assert all(
        value is None for value in loaded.benchmark_metadata["environment"].values()
    )


def test_public_records_have_only_the_loader_contract_fields() -> None:
    assert portability.__all__ == (
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
    assert tuple(field.name for field in fields(SourceFileSnapshot)) == (
        "name",
        "byte_count",
        "sha256",
    )
    assert tuple(field.name for field in fields(MemoryBankMetadata)) == (
        "dtype",
        "shape",
        "embedding_dimension",
        "patches_per_training_sample",
    )
    assert tuple(field.name for field in fields(PredictionRecord)) == (
        "sample_id",
        "defect_type",
        "image_label",
        "image_score",
        "tensor_index",
    )
    assert tuple(field.name for field in fields(BundleMetrics)) == (
        "image_auroc",
        "image_average_precision",
        "pixel_auroc",
        "training_sample_count",
        "test_sample_count",
        "test_good_sample_count",
        "anomalous_test_sample_count",
        "evaluated_pixel_count",
        "anomalous_pixel_count",
    )
    assert tuple(field.name for field in fields(ComparableBundle)) == (
        "path",
        "kind",
        "run_metadata",
        "benchmark_metadata",
        "source_files",
        "samples",
        "predictions",
        "test_sample_ids",
        "test_labels",
        "image_scores",
        "memory_bank_metadata",
        "memory_bank",
        "patch_distances",
        "nearest_bank_indices",
        "anomaly_maps",
        "evaluation_masks",
        "metrics",
    )
    assert tuple(field.name for field in fields(ScientificBundleDescriptor)) == (
        "bundle",
        "environment_id",
        "policy_role",
        "os_label",
        "execution_layer",
        "hardware_label",
        "requested_device",
    )
    assert tuple(field.name for field in fields(ScientificExecutionAttempt)) == (
        "environment_id",
        "status",
        "reason_code",
        "stage_code",
    )
    assert tuple(field.name for field in fields(ScientificGenerator)) == (
        "source_commit",
        "dirty",
    )
    assert tuple(field.name for field in fields(ScientificRunIdentity)) == (
        "environment_id",
        "policy_role",
        "bundle_kind",
        "os_label",
        "execution_layer",
        "hardware_label",
        "requested_device",
        "run",
        "source_files",
        "benchmark_workload",
        "benchmark_methodology",
    )
    assert tuple(field.name for field in fields(CandidateComparability)) == (
        "environment_id",
        "comparable",
        "gates",
        "structural_components",
    )
    assert tuple(field.name for field in fields(FloatingStatistics)) == (
        "element_count",
        "exact_count",
        "differing_count",
        "maximum_absolute_error",
        "mean_absolute_error",
        "root_mean_square_error",
        "maximum_relative_error",
        "zero_reference_count",
        "policy_violation_count",
    )
    assert tuple(field.name for field in fields(FloatingComponentComparison)) == (
        "name",
        "statistics",
    )
    assert tuple(field.name for field in fields(IndexMismatch)) == (
        "coordinate",
        "reference_value",
        "candidate_value",
    )
    assert tuple(field.name for field in fields(DiscreteComponentComparison)) == (
        "name",
        "exact",
        "element_count",
        "exact_count",
        "mismatch_count",
        "mismatch_rate",
        "first_mismatches",
    )
    assert tuple(field.name for field in fields(MetricDelta)) == (
        "metric_name",
        "reference_value",
        "candidate_value",
        "absolute_delta",
    )
    assert tuple(field.name for field in fields(CandidateScientificResult)) == (
        "environment_id",
        "status",
        "floating_components",
        "discrete_components",
        "metrics",
    )
    assert tuple(field.name for field in fields(ScientificComparison)) == (
        "schema_version",
        "schema_id",
        "milestone_id",
        "comparison_id",
        "generator",
        "reference",
        "candidates",
        "attempts",
        "comparability",
        "scientific_results",
        "limitations",
        "policy",
    )
    assert tuple(field.name for field in fields(PortabilityEnvironmentDescriptor)) == (
        "environment_id",
        "policy_role",
        "os_label",
        "execution_layer",
        "hardware_label",
        "requested_device",
    )
    assert tuple(field.name for field in fields(PortabilityEnvironmentMap)) == (
        "schema_version",
        "schema_id",
        "reference",
        "candidates",
        "attempts",
    )
    assert tuple(field.name for field in fields(PolicyTolerance)) == ("atol", "rtol")
    assert tuple(field.name for field in fields(PolicyDerivation)) == (
        "method_id",
        "comparison_ids",
    )
    assert tuple(field.name for field in fields(CanonicalInputIdentity)) == (
        "byte_count",
        "sha256",
    )
    assert tuple(field.name for field in fields(PortabilityPolicyIdentity)) == (
        "policy_id",
        "sha256",
    )
    assert tuple(field.name for field in fields(PortabilityPerformanceRun)) == (
        "environment_id",
        "os_label",
        "execution_layer",
        "hardware_label",
        "requested_device",
        "run_id",
        "benchmark_sample_id",
        "timing_methodology",
        "measurements",
    )
    assert tuple(field.name for field in fields(PortabilityPerformanceExclusion)) == (
        "environment_id",
        "reason_code",
    )


def test_rejects_missing_bundle_and_non_directory(tmp_path: Path) -> None:
    with pytest.raises(BundleValidationError):
        load_comparable_bundle(tmp_path / "missing")
    regular_file = tmp_path / "file"
    regular_file.write_text("not a bundle")
    with pytest.raises(BundleValidationError):
        load_comparable_bundle(regular_file)


@pytest.mark.parametrize("case", ["missing", "extra", "nested", "wrong-case"])
def test_rejects_any_nonexact_file_inventory(tmp_path: Path, case: str) -> None:
    bundle = _bundle(tmp_path)
    if case == "missing":
        (bundle / "metrics.json").unlink()
    elif case == "extra":
        (bundle / "extra.json").write_bytes(b"{}\n")
    elif case == "nested":
        (bundle / "nested").mkdir()
    else:
        (bundle / "metrics.json").rename(bundle / "Metrics.json")

    with pytest.raises(BundleValidationError):
        load_comparable_bundle(bundle)


def test_rejects_bundle_directory_symlink(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path / "real")
    link = tmp_path / "bundle-link"
    link.symlink_to(bundle, target_is_directory=True)

    with pytest.raises(BundleValidationError):
        load_comparable_bundle(link)


def test_rejects_artifact_symlink_even_when_target_is_regular(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    target = tmp_path / "outside-metrics.json"
    target.write_bytes((bundle / "metrics.json").read_bytes())
    (bundle / "metrics.json").unlink()
    (bundle / "metrics.json").symlink_to(target)

    with pytest.raises(BundleValidationError):
        load_comparable_bundle(bundle)


def test_rejects_nonregular_artifact(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    (bundle / "metrics.json").unlink()
    os.mkfifo(bundle / "metrics.json")

    with pytest.raises(BundleValidationError):
        load_comparable_bundle(bundle)


@pytest.mark.parametrize(
    "case",
    (
        "malformed",
        "duplicate-key",
        "nan",
        "positive-infinity",
        "negative-infinity",
        "noncanonical-spacing",
        "bom",
        "invalid-utf8",
        "missing-lf",
        "extra-lf",
        "deeply-nested",
    ),
)
def test_rejects_noncanonical_or_malformed_json(tmp_path: Path, case: str) -> None:
    bundle = _bundle(tmp_path)
    path = bundle / "metrics.json"
    canonical = path.read_bytes()
    value = _json(path)
    if case == "malformed":
        damaged = b"{\n"
    elif case == "duplicate-key":
        damaged = (
            b'{"anomalous_pixel_count":1,"anomalous_pixel_count":1,'
            + canonical.removeprefix(b"{")
        )
    elif case == "nan":
        damaged = canonical.replace(b'"image_auroc":1.0', b'"image_auroc":NaN')
    elif case == "positive-infinity":
        damaged = canonical.replace(b'"image_auroc":1.0', b'"image_auroc":Infinity')
    elif case == "negative-infinity":
        damaged = canonical.replace(b'"image_auroc":1.0', b'"image_auroc":-Infinity')
    elif case == "noncanonical-spacing":
        damaged = (json.dumps(value, sort_keys=True) + "\n").encode()
    elif case == "bom":
        damaged = b"\xef\xbb\xbf" + canonical
    elif case == "invalid-utf8":
        damaged = b"\xff" + canonical
    elif case == "missing-lf":
        damaged = canonical[:-1]
    elif case == "deeply-nested":
        damaged = b'{"value":' + (b"[" * 2000) + (b"]" * 2000) + b"}\n"
    else:
        damaged = canonical + b"\n"
    path.write_bytes(damaged)

    with pytest.raises(BundleValidationError, match="metrics.json"):
        load_comparable_bundle(bundle)


@pytest.mark.parametrize(
    "case",
    ("malformed", "blank", "noncanonical", "duplicate-key", "missing-lf"),
)
def test_rejects_malformed_or_noncanonical_jsonl(tmp_path: Path, case: str) -> None:
    bundle = _bundle(tmp_path)
    path = bundle / "samples.jsonl"
    lines = path.read_bytes().splitlines(keepends=True)
    if case == "malformed":
        lines[0] = b"{\n"
    elif case == "blank":
        lines.insert(1, b"\n")
    elif case == "noncanonical":
        lines[0] = (json.dumps(json.loads(lines[0])) + "\n").encode()
    elif case == "duplicate-key":
        lines[0] = b'{"category":"bottle","category":"bottle"}\n'
    else:
        lines[-1] = lines[-1][:-1]
    path.write_bytes(b"".join(lines))

    with pytest.raises(BundleValidationError, match="samples.jsonl"):
        load_comparable_bundle(bundle)


@pytest.mark.parametrize(
    "case",
    ("duplicate-id", "duplicate-path", "reordered", "unsafe-path"),
)
def test_rejects_invalid_sample_identity_or_order(tmp_path: Path, case: str) -> None:
    bundle = _bundle(tmp_path)
    records = _records(bundle / "samples.jsonl")
    if case == "duplicate-id":
        records[1]["sample_id"] = records[0]["sample_id"]
    elif case == "duplicate-path":
        records[1]["image_relpath"] = records[0]["image_relpath"]
    elif case == "reordered":
        records[0], records[1] = records[1], records[0]
    else:
        records[0]["image_relpath"] = "../outside.png"
    _rewrite_samples(bundle, records)

    with pytest.raises(BundleValidationError):
        load_comparable_bundle(bundle)


@pytest.mark.parametrize("case", ("reordered", "duplicate-id", "wrong-index"))
def test_rejects_invalid_prediction_identity_or_order(
    tmp_path: Path, case: str
) -> None:
    bundle = _bundle(tmp_path)
    records = _records(bundle / "predictions.jsonl")
    if case == "reordered":
        records.reverse()
    elif case == "duplicate-id":
        records[1]["sample_id"] = records[0]["sample_id"]
    else:
        records[1]["tensor_index"] = 0
    _write_records(bundle / "predictions.jsonl", records)

    with pytest.raises(BundleValidationError):
        load_comparable_bundle(bundle)


@pytest.mark.parametrize("benchmark", (False, True))
def test_rejects_run_benchmark_declaration_disagreeing_with_files(
    tmp_path: Path, benchmark: bool
) -> None:
    bundle = _bundle(tmp_path, benchmark=benchmark)
    run = _json(bundle / "run.json")
    run["benchmark"] = (
        None
        if benchmark
        else {
            "artifact": "benchmark.json",
            "schema_version": 1,
            "timing_device": "cpu",
        }
    )
    _write_json(bundle / "run.json", run)

    with pytest.raises(BundleValidationError):
        load_comparable_bundle(bundle)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("artifact", "other.json"),
        ("schema_version", 2),
        ("timing_device", "cuda:0"),
    ),
)
def test_rejects_invalid_run_benchmark_linkage(
    tmp_path: Path, field: str, value: object
) -> None:
    bundle = _bundle(tmp_path, benchmark=True)
    run = _json(bundle / "run.json")
    run["benchmark"][field] = value
    _write_json(bundle / "run.json", run)

    with pytest.raises(BundleValidationError):
        load_comparable_bundle(bundle)


@pytest.mark.parametrize(
    ("path", "value"),
    (
        (("run_id",), "other"),
        (("category",), "leather"),
        (("profile_id",), "other-profile"),
        (("device",), "cuda:0"),
        (("created_at_utc",), "2026-07-15T13:00:00Z"),
        (("benchmark_sample_id",), "mvtec_ad/bottle/test/good/001.png"),
    ),
)
def test_rejects_benchmark_run_identity_mismatch(
    tmp_path: Path, path: tuple[str, ...], value: object
) -> None:
    bundle = _bundle(tmp_path, benchmark=True)
    record = _json(bundle / "benchmark.json")
    _set_path(record, path, value)
    _write_json(bundle / "benchmark.json", record)

    with pytest.raises(BundleValidationError):
        load_comparable_bundle(bundle)


@pytest.mark.parametrize(
    ("filename", "case"),
    (
        ("memory_bank.pt", "not-object"),
        ("memory_bank.pt", "missing"),
        ("memory_bank.pt", "extra"),
        ("retrieval.pt", "not-object"),
        ("retrieval.pt", "missing"),
        ("retrieval.pt", "extra"),
        ("anomaly_maps.pt", "not-object"),
        ("anomaly_maps.pt", "missing"),
        ("anomaly_maps.pt", "extra"),
    ),
)
def test_rejects_malformed_or_nonclosed_tensor_container(
    tmp_path: Path, filename: str, case: str
) -> None:
    bundle = _bundle(tmp_path)
    path = bundle / filename
    if case == "not-object":
        torch.save([], path)
    else:
        payload = _payload(path)
        if case == "missing":
            payload.pop(next(iter(payload)))
        else:
            payload["unexpected"] = torch.tensor(0)
        torch.save(payload, path)

    with pytest.raises(BundleValidationError, match=filename):
        load_comparable_bundle(bundle)


def test_tensor_loading_is_cpu_only_and_weights_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _bundle(tmp_path)
    original = torch.load
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def guarded_load(
        source: object, *args: object, **kwargs: object
    ) -> dict[str, object]:
        calls.append((args, kwargs))
        return original(source, *args, **kwargs)

    monkeypatch.setattr(portability.torch, "load", guarded_load)
    loaded = load_comparable_bundle(bundle)

    assert loaded.memory_bank.device.type == "cpu"
    assert len(calls) == 3
    assert all(not args for args, _ in calls)
    assert all(
        kwargs["map_location"] == "cpu" and kwargs["weights_only"] is True
        for _, kwargs in calls
    )


def test_weights_only_loading_does_not_execute_pickle_payload(tmp_path: Path) -> None:
    global _EXECUTED_UNSAFE_PAYLOAD
    _EXECUTED_UNSAFE_PAYLOAD = False
    bundle = _bundle(tmp_path)
    torch.save(_UnsafePayload(), bundle / "memory_bank.pt")

    with pytest.raises(BundleValidationError):
        load_comparable_bundle(bundle)
    assert not _EXECUTED_UNSAFE_PAYLOAD


@pytest.mark.parametrize(
    ("filename", "field", "case"),
    (
        ("memory_bank.pt", "memory_bank", "bank"),
        ("retrieval.pt", "patch_distances", "distance"),
        ("retrieval.pt", "nearest_bank_indices", "index"),
        ("anomaly_maps.pt", "anomaly_maps", "map"),
        ("anomaly_maps.pt", "evaluation_masks", "mask"),
    ),
)
def test_rejects_incorrect_tensor_dtype(
    tmp_path: Path, filename: str, field: str, case: str
) -> None:
    bundle = _bundle(tmp_path)
    path = bundle / filename
    payload = _payload(path)
    dtypes = {
        "bank": torch.float64,
        "distance": torch.float64,
        "index": torch.int32,
        "map": torch.float64,
        "mask": torch.int64,
    }
    payload[field] = payload[field].to(dtypes[case])
    torch.save(payload, path)

    with pytest.raises(BundleValidationError):
        load_comparable_bundle(bundle)


@pytest.mark.parametrize(
    ("filename", "field"),
    (
        ("memory_bank.pt", "memory_bank"),
        ("retrieval.pt", "patch_distances"),
        ("retrieval.pt", "nearest_bank_indices"),
        ("anomaly_maps.pt", "anomaly_maps"),
        ("anomaly_maps.pt", "evaluation_masks"),
    ),
)
def test_rejects_incorrect_tensor_rank_or_shape(
    tmp_path: Path, filename: str, field: str
) -> None:
    bundle = _bundle(tmp_path)
    path = bundle / filename
    payload = _payload(path)
    payload[field] = payload[field].reshape(-1)
    torch.save(payload, path)

    with pytest.raises(BundleValidationError):
        load_comparable_bundle(bundle)


@pytest.mark.parametrize(
    ("filename", "field"),
    (
        ("memory_bank.pt", "memory_bank"),
        ("retrieval.pt", "patch_distances"),
        ("retrieval.pt", "nearest_bank_indices"),
        ("anomaly_maps.pt", "anomaly_maps"),
        ("anomaly_maps.pt", "evaluation_masks"),
    ),
)
def test_rejects_noncontiguous_tensor(
    tmp_path: Path, filename: str, field: str
) -> None:
    bundle = _bundle(tmp_path)
    path = bundle / filename
    payload = _payload(path)
    tensor = payload[field]
    if tensor.ndim == 2:
        payload[field] = tensor.transpose(0, 1).contiguous().transpose(0, 1)
    else:
        payload[field] = tensor.transpose(1, 2)
    assert not payload[field].is_contiguous()
    torch.save(payload, path)

    with pytest.raises(BundleValidationError):
        load_comparable_bundle(bundle)


@pytest.mark.parametrize(
    ("filename", "field"),
    (
        ("memory_bank.pt", "memory_bank"),
        ("retrieval.pt", "patch_distances"),
        ("anomaly_maps.pt", "anomaly_maps"),
    ),
)
def test_rejects_nonfinite_floating_tensor(
    tmp_path: Path, filename: str, field: str
) -> None:
    bundle = _bundle(tmp_path)
    path = bundle / filename
    payload = _payload(path)
    payload[field][0].reshape(-1)[0] = float("nan")
    torch.save(payload, path)

    with pytest.raises(BundleValidationError):
        load_comparable_bundle(bundle)


def test_rejects_tensor_subclasses(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    path = bundle / "memory_bank.pt"
    payload = _payload(path)
    payload["memory_bank"] = torch.nn.Parameter(
        payload["memory_bank"], requires_grad=False
    )
    torch.save(payload, path)

    with pytest.raises(BundleValidationError):
        load_comparable_bundle(bundle)


def test_rejects_sparse_non_strided_tensor(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    path = bundle / "memory_bank.pt"
    payload = _payload(path)
    payload["memory_bank"] = torch.zeros((1024, 512), dtype=torch.float32).to_sparse()
    torch.save(payload, path)

    with pytest.raises(BundleValidationError):
        load_comparable_bundle(bundle)


def test_rejects_memory_bank_row_count_mismatch(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    path = bundle / "memory_bank.pt"
    payload = _payload(path)
    payload["memory_bank"] = torch.zeros((2048, 512), dtype=torch.float32)
    payload["shape"] = [2048, 512]
    torch.save(payload, path)
    run = _json(bundle / "run.json")
    run["tensors"]["memory_bank"]["shape"] = [2048, 512]
    run["tensors"]["memory_bank"]["byte_count"] = 2048 * 512 * 4
    _write_json(bundle / "run.json", run)

    with pytest.raises(BundleValidationError):
        load_comparable_bundle(bundle)


def test_rejects_retrieval_test_id_mismatch(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    path = bundle / "retrieval.pt"
    payload = _payload(path)
    payload["test_sample_ids"].reverse()
    torch.save(payload, path)

    with pytest.raises(BundleValidationError):
        load_comparable_bundle(bundle)


@pytest.mark.parametrize("index", (-1, 1024))
def test_rejects_nearest_index_outside_bank(tmp_path: Path, index: int) -> None:
    bundle = _bundle(tmp_path)
    path = bundle / "retrieval.pt"
    payload = _payload(path)
    payload["nearest_bank_indices"][0, 0] = index
    torch.save(payload, path)

    with pytest.raises(BundleValidationError):
        load_comparable_bundle(bundle)


def test_rejects_prediction_score_different_from_patch_maximum(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    records = _records(bundle / "predictions.jsonl")
    records[0]["image_score"] = 0.8
    _write_records(bundle / "predictions.jsonl", records)

    with pytest.raises(BundleValidationError):
        load_comparable_bundle(bundle)


@pytest.mark.parametrize(
    "case", ("test-ids", "good-mask", "empty-anomalous-mask", "map-count")
)
def test_rejects_anomaly_map_or_mask_mismatch(tmp_path: Path, case: str) -> None:
    bundle = _bundle(tmp_path)
    path = bundle / "anomaly_maps.pt"
    payload = _payload(path)
    if case == "test-ids":
        payload["test_sample_ids"].reverse()
    elif case == "good-mask":
        payload["evaluation_masks"][1, 0, 0] = 1
    elif case == "empty-anomalous-mask":
        payload["evaluation_masks"][0].zero_()
    else:
        payload["anomaly_maps"] = payload["anomaly_maps"][:1]
    torch.save(payload, path)

    with pytest.raises(BundleValidationError):
        load_comparable_bundle(bundle)


def test_rejects_nonbinary_mask(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    path = bundle / "anomaly_maps.pt"
    payload = _payload(path)
    payload["evaluation_masks"][0, 0, 0] = 2
    torch.save(payload, path)

    with pytest.raises(BundleValidationError):
        load_comparable_bundle(bundle)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("image_auroc", 0.5),
        ("training_sample_count", 2),
        ("test_sample_count", 3),
        ("anomalous_pixel_count", 2),
    ),
)
def test_rejects_metric_recomputation_or_count_mismatch(
    tmp_path: Path, field: str, value: object
) -> None:
    bundle = _bundle(tmp_path)
    metrics = _json(bundle / "metrics.json")
    metrics[field] = value
    _write_json(bundle / "metrics.json", metrics)

    with pytest.raises(BundleValidationError):
        load_comparable_bundle(bundle)


def test_rejects_corrupt_tensor_artifact_bytes(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    (bundle / "retrieval.pt").write_bytes(b"not a torch weights-only archive")

    with pytest.raises(BundleValidationError, match="retrieval.pt"):
        load_comparable_bundle(bundle)


def test_rejects_source_snapshot_change_during_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _bundle(tmp_path)
    original = portability._snapshot_sources
    calls = 0

    def changed(path: Path, names: tuple[str, ...]) -> tuple[SourceFileSnapshot, ...]:
        nonlocal calls
        calls += 1
        snapshots = original(path, names)
        if calls == 2:
            snapshots = (
                replace(snapshots[0], byte_count=snapshots[0].byte_count + 1),
                *snapshots[1:],
            )
        return snapshots

    monkeypatch.setattr(portability, "_snapshot_sources", changed)
    with pytest.raises(BundleValidationError):
        load_comparable_bundle(bundle)
    assert calls == 2


def test_rejects_transient_swap_restored_before_final_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _bundle(tmp_path)
    run_path = bundle / "run.json"
    original_bytes = run_path.read_bytes()
    swapped = _json(run_path)
    swapped["dataset_root"] = "/transient/mvtec-ad"
    swapped_bytes = _canonical(swapped)
    original_snapshot = portability._snapshot_sources
    snapshot_calls = 0

    def swap_after_initial_snapshot(
        path: Path, names: tuple[str, ...]
    ) -> tuple[SourceFileSnapshot, ...]:
        nonlocal snapshot_calls
        snapshot_calls += 1
        snapshots = original_snapshot(path, names)
        if snapshot_calls == 1:
            run_path.write_bytes(swapped_bytes)
        return snapshots

    original_classify = portability._classify_bundle
    classify_calls = 0

    def restore_before_final_snapshot(
        path: Path,
    ) -> tuple[str, tuple[str, ...]]:
        nonlocal classify_calls
        classify_calls += 1
        if classify_calls == 2:
            run_path.write_bytes(original_bytes)
        return original_classify(path)

    monkeypatch.setattr(portability, "_snapshot_sources", swap_after_initial_snapshot)
    monkeypatch.setattr(portability, "_classify_bundle", restore_before_final_snapshot)
    try:
        with pytest.raises(BundleValidationError):
            load_comparable_bundle(bundle)
    finally:
        run_path.write_bytes(original_bytes)


def test_snapshot_has_exact_order_current_sizes_and_sha256(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path, benchmark=True)
    loaded = load_comparable_bundle(bundle)

    assert tuple(snapshot.name for snapshot in loaded.source_files) == _BENCHMARK_FILES
    assert all(type(snapshot.byte_count) is int for snapshot in loaded.source_files)
    assert all(
        snapshot.byte_count == (bundle / snapshot.name).stat().st_size
        and snapshot.sha256
        == hashlib.sha256((bundle / snapshot.name).read_bytes()).hexdigest()
        for snapshot in loaded.source_files
    )


@pytest.mark.parametrize("benchmark", (False, True))
def test_successful_loading_does_not_change_source_bundle(
    tmp_path: Path, benchmark: bool
) -> None:
    bundle = _bundle(tmp_path, benchmark=benchmark)
    before = _source_state(bundle)

    load_comparable_bundle(bundle)

    assert _source_state(bundle) == before


def test_rejected_loading_does_not_change_source_bundle(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    (bundle / "metrics.json").write_bytes(b"{\n")
    before = _source_state(bundle)

    with pytest.raises(BundleValidationError):
        load_comparable_bundle(bundle)

    assert _source_state(bundle) == before


@pytest.mark.parametrize(
    ("filename", "path", "value", "benchmark"),
    (
        ("run.json", ("unexpected",), 1, False),
        ("run.json", ("batch_size",), _DELETE, False),
        ("run.json", ("source", "unexpected"), 1, False),
        ("run.json", ("source", "dirty"), _DELETE, False),
        ("metrics.json", ("unexpected",), 1, False),
        ("metrics.json", ("pixel_auroc",), _DELETE, False),
        ("benchmark.json", ("unexpected",), 1, True),
        ("benchmark.json", ("schema_version",), _DELETE, True),
    ),
)
def test_rejects_unknown_or_missing_json_fields(
    tmp_path: Path,
    filename: str,
    path: tuple[str, ...],
    value: object,
    benchmark: bool,
) -> None:
    bundle = _bundle(tmp_path, benchmark=benchmark)
    record = _json(bundle / filename)
    _set_path(record, path, value)
    _write_json(bundle / filename, record)

    with pytest.raises(BundleValidationError):
        load_comparable_bundle(bundle)


@pytest.mark.parametrize(
    ("filename", "field", "missing"),
    (
        ("samples.jsonl", "unexpected", False),
        ("samples.jsonl", "category", True),
        ("predictions.jsonl", "unexpected", False),
        ("predictions.jsonl", "defect_type", True),
    ),
)
def test_rejects_unknown_or_missing_jsonl_fields(
    tmp_path: Path, filename: str, field: str, missing: bool
) -> None:
    bundle = _bundle(tmp_path)
    records = _records(bundle / filename)
    if missing:
        del records[0][field]
    else:
        records[0][field] = 1
    if filename == "samples.jsonl":
        _rewrite_samples(bundle, records)
    else:
        _write_records(bundle / filename, records)

    with pytest.raises(BundleValidationError):
        load_comparable_bundle(bundle)


@pytest.mark.parametrize(
    ("path", "value"),
    (
        (("schema_version",), 2),
        (("profile_id",), "other"),
        (("category",), "leather"),
        (("batch_size",), 2),
        (("bank_chunk_size",), 0),
        (("device",), ""),
        (("source", "dirty"), 0),
        (("source", "git_commit"), "short"),
        (("source", "uv_lock_sha256"), "short"),
        (("environment", "created_at_utc"), "not-a-timestamp"),
        (("weights", "cached_file_sha256"), "short"),
        (("weights", "enum"), "Other"),
        (("feature_extractor",), "Other"),
        (("feature_layer",), "layer3"),
        (("preprocessing_profile",), "other"),
        (("retrieval_semantics",), "other"),
        (("map_interpolation", "align_corners"), True),
        (("inventory", "training_sample_count"), 2),
        (("inventory", "sample_inventory_sha256"), "0" * 64),
        (("tensors", "patch_distances", "dtype"), "float64"),
        (("tensors", "anomaly_maps", "shape"), [2, 255, 256]),
    ),
)
def test_rejects_invalid_run_identity_or_contract(
    tmp_path: Path, path: tuple[str, ...], value: object
) -> None:
    bundle = _bundle(tmp_path)
    run = _json(bundle / "run.json")
    _set_path(run, path, value)
    _write_json(bundle / "run.json", run)

    with pytest.raises(BundleValidationError):
        load_comparable_bundle(bundle)


_BENCHMARK_MUTATIONS = (
    ("workload-q", ("workload", "Q"), 1),
    ("workload-d", ("workload", "D"), 256),
    ("workload-k", ("workload", "k"), 2),
    ("workload-m", ("workload", "M"), 1023),
    ("workload-bank-shape", ("workload", "bank_shape"), [1023, 512]),
    ("workload-bank-bytes", ("workload", "bank_bytes"), 1),
    ("workload-chunk", ("workload", "bank_chunk_size"), 1),
    ("workload-batch", ("workload", "batch_size"), 2),
    ("workload-dtype", ("workload", "dtype"), "float64"),
    ("workload-training-count", ("workload", "training_sample_count"), 2),
    ("workload-test-count", ("workload", "test_sample_count"), 3),
    (
        "workload-layout",
        ("workload", "tensor_layout", "memory_bank"),
        "other",
    ),
    ("warmup-zero", ("methodology", "warmup_count"), 0),
    ("repeat-bool", ("methodology", "repeat_count"), True),
    (
        "warmup-in-statistics",
        ("methodology", "warmup_samples_in_statistics"),
        True,
    ),
    ("timing-unit", ("methodology", "timing_unit"), "seconds"),
    (
        "cpu-cuda-method",
        ("methodology", "cuda_timing_method"),
        "cuda event",
    ),
    (
        "cpu-sync-policy",
        ("methodology", "synchronization_policy"),
        "other",
    ),
    (
        "missing-stage-boundary",
        ("methodology", "stage_inclusion_boundaries", "image_decode"),
        _DELETE,
    ),
    (
        "extra-stage-boundary",
        ("methodology", "stage_inclusion_boundaries", "extra"),
        "extra",
    ),
    (
        "missing-repeated-stage",
        ("results", "repeated_stages", "image_decode"),
        _DELETE,
    ),
    (
        "extra-repeated-stage",
        ("results", "repeated_stages", "extra"),
        {
            "count": 2,
            "maximum": 2.0,
            "mean": 1.5,
            "minimum": 1.0,
            "p50": 1.5,
            "p95": 1.95,
        },
    ),
    (
        "summary-count",
        ("results", "repeated_stages", "image_decode", "count"),
        1,
    ),
    (
        "summary-negative",
        ("results", "repeated_stages", "image_decode", "minimum"),
        -1.0,
    ),
    (
        "summary-order",
        ("results", "repeated_stages", "image_decode", "p50"),
        3.0,
    ),
    (
        "summary-mean",
        ("results", "repeated_stages", "image_decode", "mean"),
        3.0,
    ),
    (
        "end-to-end-count",
        ("results", "synchronized_end_to_end", "count"),
        1,
    ),
    (
        "negative-one-off",
        ("results", "one_off_ms", "model_and_weight_load"),
        -1.0,
    ),
    (
        "cpu-environment",
        ("environment", "cuda_device_name"),
        "GPU",
    ),
    (
        "cpu-peak-allocated",
        ("results", "device_memory", "peak_allocated_bytes"),
        1,
    ),
    (
        "cpu-peak-reserved",
        ("results", "device_memory", "peak_reserved_bytes"),
        1,
    ),
    (
        "persistent-bank",
        ("results", "device_memory", "persistent_bank_bytes"),
        1,
    ),
    (
        "allocated-boundary",
        ("results", "device_memory", "peak_allocated_boundary"),
        "other",
    ),
    (
        "reserved-boundary",
        ("results", "device_memory", "peak_reserved_boundary"),
        "other",
    ),
)


@pytest.mark.parametrize(
    ("case", "path", "value"),
    _BENCHMARK_MUTATIONS,
    ids=[case[0] for case in _BENCHMARK_MUTATIONS],
)
def test_rejects_invalid_benchmark_workload_methodology_or_summary(
    tmp_path: Path, case: str, path: tuple[str, ...], value: object
) -> None:
    del case
    bundle = _bundle(tmp_path, benchmark=True)
    record = _json(bundle / "benchmark.json")
    _set_path(record, path, value)
    _write_json(bundle / "benchmark.json", record)

    with pytest.raises(BundleValidationError):
        load_comparable_bundle(bundle)


@pytest.mark.parametrize(
    ("path", "value"),
    (
        (("environment", "cuda_compute_capability"), [7]),
        (("environment", "cuda_compute_capability"), [7, True]),
        (("environment", "cuda_device_name"), ""),
        (("environment", "pytorch_cuda_runtime_version"), None),
        (
            ("results", "device_memory", "peak_allocated_bytes"),
            1024 * 512 * 4 - 1,
        ),
        (
            ("results", "device_memory", "peak_reserved_bytes"),
            1024 * 512 * 4 - 1,
        ),
    ),
)
def test_rejects_invalid_cuda_benchmark_environment_or_allocator(
    tmp_path: Path, path: tuple[str, ...], value: object
) -> None:
    bundle = _bundle(tmp_path, benchmark=True, device="cuda:0")
    record = _json(bundle / "benchmark.json")
    _set_path(record, path, value)
    _write_json(bundle / "benchmark.json", record)

    with pytest.raises(BundleValidationError):
        load_comparable_bundle(bundle)


_A2_GENERATOR = ScientificGenerator("d" * 40, False)
_A2_GATE_NAMES = (
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


@pytest.fixture(scope="module")
def _a2_bundles(
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[str, ComparableBundle]:
    root = tmp_path_factory.mktemp("portability-science")
    return {
        "evaluation": load_comparable_bundle(_bundle(root / "evaluation")),
        "benchmark": load_comparable_bundle(
            _bundle(root / "benchmark", benchmark=True)
        ),
        "mps": load_comparable_bundle(_bundle(root / "mps", device="mps")),
    }


def _descriptor(
    bundle: ComparableBundle,
    environment_id: str,
    policy_role: str,
    *,
    execution_layer: str = "native",
    os_label: str = "Test OS",
    hardware_label: str = "Synthetic device",
) -> ScientificBundleDescriptor:
    return ScientificBundleDescriptor(
        bundle=bundle,
        environment_id=environment_id,
        policy_role=policy_role,  # type: ignore[arg-type]
        os_label=os_label,
        execution_layer=execution_layer,  # type: ignore[arg-type]
        hardware_label=hardware_label,
        requested_device=str(bundle.run_metadata["device"]),
    )


def _comparison(
    reference: ComparableBundle,
    *candidates: ComparableBundle,
    attempts: tuple[ScientificExecutionAttempt, ...] = (),
    candidate_roles: tuple[str, ...] = (),
) -> ScientificComparison:
    roles = candidate_roles or ("holdout",) * len(candidates)
    return compare_scientific_bundles(
        _descriptor(reference, "reference-env", "reference"),
        tuple(
            _descriptor(candidate, f"candidate-{index}", role)
            for index, (candidate, role) in enumerate(
                zip(candidates, roles, strict=True), start=1
            )
        ),
        generator=_A2_GENERATOR,
        attempts=attempts,
    )


def _run_variant(
    bundle: ComparableBundle, path: tuple[str, ...], value: object
) -> ComparableBundle:
    run = portability._thaw_comparison_json(bundle.run_metadata)
    assert isinstance(run, dict)
    _set_path(run, path, value)
    frozen = portability._freeze_json(run)
    assert isinstance(frozen, dict | portability.MappingProxyType)
    return replace(bundle, run_metadata=frozen)


def _tensor_variant(
    bundle: ComparableBundle,
    name: str,
    index: object,
    value: float | int,
) -> ComparableBundle:
    tensor = getattr(bundle, name).clone()
    tensor[index] = value
    return replace(bundle, **{name: tensor})


def _floating(
    comparison: ScientificComparison, name: str, candidate_index: int = 0
) -> FloatingStatistics:
    components = comparison.scientific_results[candidate_index].floating_components
    assert components is not None
    return next(item.statistics for item in components if item.name == name)


def _discrete(
    comparison: ScientificComparison, name: str, candidate_index: int = 0
) -> DiscreteComponentComparison:
    components = comparison.scientific_results[candidate_index].discrete_components
    assert components is not None
    return next(item for item in components if item.name == name)


def _larger_bank_variant(bundle: ComparableBundle) -> ComparableBundle:
    added_sample = MvtecSample(
        "mvtec_ad/bottle/train/good/003.png",
        "bottle",
        "train",
        "good",
        False,
        "bottle/train/good/003.png",
        None,
    )
    run = portability._thaw_comparison_json(bundle.run_metadata)
    assert isinstance(run, dict)
    run["inventory"]["training_sample_count"] = 2
    run["inventory"]["total_sample_count"] = 4
    run["inventory"]["sample_inventory_sha256"] = "0" * 64
    run["tensors"]["memory_bank"]["shape"] = [2048, 512]
    run["tensors"]["memory_bank"]["byte_count"] = 2048 * 512 * 4
    frozen = portability._freeze_json(run)
    assert isinstance(frozen, dict | portability.MappingProxyType)
    return replace(
        bundle,
        run_metadata=frozen,
        samples=tuple(
            sorted((*bundle.samples, added_sample), key=lambda item: item.sample_id)
        ),
        memory_bank_metadata=MemoryBankMetadata("float32", (2048, 512), 512, 1024),
        memory_bank=torch.zeros((2048, 512), dtype=torch.float32),
        metrics=replace(bundle.metrics, training_sample_count=2),
    )


@pytest.fixture(scope="module")
def _exact_scientific(
    _a2_bundles: dict[str, ComparableBundle],
) -> ScientificComparison:
    bundle = _a2_bundles["evaluation"]
    return _comparison(bundle, bundle)


@pytest.mark.parametrize(
    ("reference_kind", "candidate_kind"),
    (
        ("evaluation", "evaluation"),
        ("benchmark", "benchmark"),
        ("benchmark", "evaluation"),
        ("evaluation", "benchmark"),
    ),
)
def test_scientific_comparison_supports_all_bundle_kind_pairs(
    _a2_bundles: dict[str, ComparableBundle],
    reference_kind: str,
    candidate_kind: str,
) -> None:
    comparison = _comparison(_a2_bundles[reference_kind], _a2_bundles[candidate_kind])

    assert comparison.comparability[0].comparable
    assert comparison.scientific_results[0].status == "observed_unclassified"
    assert comparison.reference.bundle_kind == reference_kind
    assert comparison.candidates[0].bundle_kind == candidate_kind
    assert tuple(name for name, _ in comparison.comparability[0].gates) == (
        _A2_GATE_NAMES
    )
    assert all(value for _, value in comparison.comparability[0].gates)
    assert all(
        component.exact
        for component in comparison.comparability[0].structural_components
    )


def test_scientific_comparison_preserves_explicit_candidate_order(
    _a2_bundles: dict[str, ComparableBundle],
) -> None:
    bundle = _a2_bundles["evaluation"]
    reference = _descriptor(bundle, "reference-env", "reference")
    candidates = (
        _descriptor(bundle, "z-candidate", "holdout"),
        _descriptor(bundle, "a-candidate", "same_stack_control"),
    )

    comparison = compare_scientific_bundles(
        reference, candidates, generator=_A2_GENERATOR
    )
    document = json.loads(encode_scientific_comparison(comparison))

    assert tuple(item.environment_id for item in comparison.candidates) == (
        "z-candidate",
        "a-candidate",
    )
    assert tuple(item.environment_id for item in comparison.comparability) == (
        "z-candidate",
        "a-candidate",
    )
    assert tuple(item.environment_id for item in comparison.scientific_results) == (
        "z-candidate",
        "a-candidate",
    )
    assert [item["environment_id"] for item in document["candidates"]] == [
        "z-candidate",
        "a-candidate",
    ]


@pytest.mark.parametrize("case", ("reference-candidate", "candidate-candidate"))
def test_scientific_comparison_rejects_duplicate_environment_ids(
    _a2_bundles: dict[str, ComparableBundle], case: str
) -> None:
    bundle = _a2_bundles["evaluation"]
    reference = _descriptor(bundle, "same-env", "reference")
    candidates = (
        (
            _descriptor(bundle, "same-env", "holdout"),
            _descriptor(bundle, "other-env", "calibration"),
        )
        if case == "reference-candidate"
        else (
            _descriptor(bundle, "candidate-env", "holdout"),
            _descriptor(bundle, "candidate-env", "calibration"),
        )
    )

    with pytest.raises(ComparisonValidationError):
        compare_scientific_bundles(reference, candidates, generator=_A2_GENERATOR)


@pytest.mark.parametrize("case", ("reference-role", "candidate-role"))
def test_scientific_comparison_rejects_invalid_bundle_roles(
    _a2_bundles: dict[str, ComparableBundle], case: str
) -> None:
    bundle = _a2_bundles["evaluation"]
    reference = _descriptor(
        bundle, "reference-env", "holdout" if case == "reference-role" else "reference"
    )
    candidate = _descriptor(
        bundle,
        "candidate-env",
        "reference" if case == "candidate-role" else "holdout",
    )

    with pytest.raises(ComparisonValidationError):
        compare_scientific_bundles(reference, (candidate,), generator=_A2_GENERATOR)


def test_scientific_descriptor_rejects_invalid_execution_layer(
    _a2_bundles: dict[str, ComparableBundle],
) -> None:
    with pytest.raises(ComparisonValidationError):
        _descriptor(
            _a2_bundles["evaluation"],
            "candidate-env",
            "holdout",
            execution_layer="container",
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("environment_id", "../private"),
        ("environment_id", "a" * 65),
        ("os_label", "/home/alice"),
        ("os_label", "line\nbreak"),
        ("hardware_label", "alice@example"),
        ("hardware_label", "x" * 121),
        ("requested_device", "cuda"),
    ),
)
def test_scientific_descriptor_rejects_private_or_malformed_identity(
    _a2_bundles: dict[str, ComparableBundle], field: str, value: str
) -> None:
    bundle = _a2_bundles["evaluation"]
    values = {
        "bundle": bundle,
        "environment_id": "candidate-env",
        "policy_role": "holdout",
        "os_label": "Test OS",
        "execution_layer": "native",
        "hardware_label": "Synthetic device",
        "requested_device": "cpu",
    }
    values[field] = value

    with pytest.raises(ComparisonValidationError):
        ScientificBundleDescriptor(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("name", "index", "value"),
    (
        ("memory_bank", (0, 0), 1.0),
        ("patch_distances", (0, 0), 1.0),
        ("image_scores", 0, 1.0),
        ("anomaly_maps", (0, 0, 0), 2.0),
    ),
)
def test_scientific_comparison_observes_one_element_floating_drift(
    _a2_bundles: dict[str, ComparableBundle],
    name: str,
    index: object,
    value: float,
) -> None:
    reference = _a2_bundles["evaluation"]
    candidate = _tensor_variant(reference, name, index, value)

    comparison = _comparison(reference, candidate)
    changed = _floating(comparison, name)

    assert comparison.scientific_results[0].status == "observed_unclassified"
    assert changed.differing_count == 1
    assert changed.exact_count == changed.element_count - 1
    assert changed.maximum_absolute_error > 0.0
    assert changed.mean_absolute_error > 0.0
    assert changed.root_mean_square_error > 0.0
    assert all(
        _floating(comparison, other).differing_count == 0
        for other in (
            "memory_bank",
            "patch_distances",
            "image_scores",
            "anomaly_maps",
        )
        if other != name
    )


def test_floating_drift_exactly_at_chunk_boundary(
    _a2_bundles: dict[str, ComparableBundle],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference = _a2_bundles["evaluation"]
    chunk_size = 257_000
    candidate_bank = reference.memory_bank.clone()
    candidate_bank.reshape(-1)[chunk_size] = 1.0
    candidate = replace(reference, memory_bank=candidate_bank)
    monkeypatch.setattr(portability, "_FLOAT_CHUNK_SIZE", chunk_size)

    statistics = _floating(_comparison(reference, candidate), "memory_bank")

    assert statistics.differing_count == 1
    assert statistics.exact_count == reference.memory_bank.numel() - 1


def test_floating_difference_allocation_is_bounded_to_multiple_chunks(
    _a2_bundles: dict[str, ComparableBundle],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _a2_bundles["evaluation"]
    chunk_size = 100_000
    subtraction_sizes: list[int] = []
    original = torch.Tensor.sub

    def guarded_sub(
        tensor: torch.Tensor, other: object, *args: object, **kwargs: object
    ) -> torch.Tensor:
        if isinstance(other, torch.Tensor):
            subtraction_sizes.append(tensor.numel())
            assert tensor.numel() <= chunk_size
            assert other.numel() <= chunk_size
        return original(tensor, other, *args, **kwargs)

    monkeypatch.setattr(portability, "_FLOAT_CHUNK_SIZE", chunk_size)
    monkeypatch.setattr(torch.Tensor, "sub", guarded_sub)

    comparison = _comparison(bundle, bundle)

    assert comparison.scientific_results[0].status == "observed_unclassified"
    assert len(subtraction_sizes) > 4
    assert max(subtraction_sizes) <= chunk_size


def test_floating_statistics_count_exact_zero_references(
    _a2_bundles: dict[str, ComparableBundle],
) -> None:
    bundle = _a2_bundles["evaluation"]
    distances = bundle.patch_distances.clone()
    distances[0, 0] = 0.0
    reference = replace(bundle, patch_distances=distances)

    statistics = _floating(_comparison(reference, reference), "patch_distances")

    assert statistics.zero_reference_count == 1
    assert statistics.differing_count == 0
    assert statistics.maximum_relative_error == 0.0


def test_floating_drift_at_zero_reference_is_excluded_from_relative_maximum(
    _a2_bundles: dict[str, ComparableBundle],
) -> None:
    bundle = _a2_bundles["evaluation"]
    reference_distances = bundle.patch_distances.clone()
    candidate_distances = reference_distances.clone()
    reference_distances[0, 0] = 0.0
    candidate_distances[0, 0] = 1.0
    reference = replace(bundle, patch_distances=reference_distances)
    candidate = replace(bundle, patch_distances=candidate_distances)

    statistics = _floating(_comparison(reference, candidate), "patch_distances")

    assert statistics.zero_reference_count == 1
    assert statistics.differing_count == 1
    assert statistics.maximum_absolute_error == 1.0
    assert statistics.maximum_relative_error == 0.0


def test_all_zero_reference_has_null_relative_error(
    _exact_scientific: ScientificComparison,
) -> None:
    statistics = _floating(_exact_scientific, "memory_bank")

    assert statistics.zero_reference_count == statistics.element_count
    assert statistics.maximum_relative_error is None
    document = json.loads(encode_scientific_comparison(_exact_scientific))
    assert (
        document["scientific_results"]["candidate-1"]["floating_components"][
            "memory_bank"
        ]["maximum_relative_error"]
        is None
    )


def test_binary64_accumulation_handles_large_finite_float32_values(
    _a2_bundles: dict[str, ComparableBundle],
) -> None:
    bundle = _a2_bundles["evaluation"]
    maximum = torch.finfo(torch.float32).max
    reference_bank = bundle.memory_bank.clone()
    candidate_bank = bundle.memory_bank.clone()
    reference_bank[0, 0] = maximum
    candidate_bank[0, 0] = -maximum
    reference = replace(bundle, memory_bank=reference_bank)
    candidate = replace(bundle, memory_bank=candidate_bank)

    statistics = _floating(_comparison(reference, candidate), "memory_bank")

    assert statistics.differing_count == 1
    assert statistics.maximum_absolute_error == pytest.approx(2.0 * maximum)
    assert math.isfinite(statistics.root_mean_square_error)
    assert statistics.maximum_relative_error == pytest.approx(2.0)


@pytest.mark.parametrize(
    ("side", "name"),
    (
        ("reference", "memory_bank"),
        ("candidate", "anomaly_maps"),
    ),
)
def test_scientific_comparison_rejects_mutated_nonfinite_tensors(
    _a2_bundles: dict[str, ComparableBundle], side: str, name: str
) -> None:
    bundle = _a2_bundles["evaluation"]
    invalid = _tensor_variant(bundle, name, (0,) * getattr(bundle, name).ndim, math.nan)
    reference, candidate = (
        (invalid, bundle) if side == "reference" else (bundle, invalid)
    )

    with pytest.raises(ComparisonValidationError):
        _comparison(reference, candidate)


def test_scientific_comparison_rejects_mutated_nonfinite_metric(
    _a2_bundles: dict[str, ComparableBundle],
) -> None:
    bundle = _a2_bundles["evaluation"]
    candidate = replace(
        bundle, metrics=replace(bundle.metrics, image_auroc=float("nan"))
    )

    with pytest.raises(ComparisonValidationError):
        _comparison(bundle, candidate)


@pytest.mark.parametrize("case", ("shape", "dtype"))
def test_scientific_comparison_rejects_tensor_contract_mutation_after_loading(
    _a2_bundles: dict[str, ComparableBundle], case: str
) -> None:
    bundle = _a2_bundles["evaluation"]
    bank = (
        bundle.memory_bank.clone().reshape(-1)
        if case == "shape"
        else bundle.memory_bank.clone().to(torch.float64)
    )
    candidate = replace(bundle, memory_bank=bank)

    with pytest.raises(ComparisonValidationError):
        _comparison(bundle, candidate)


@pytest.mark.parametrize("case", ("index-range", "binary-mask"))
def test_scientific_comparison_revalidates_discrete_tensor_values(
    _a2_bundles: dict[str, ComparableBundle], case: str
) -> None:
    bundle = _a2_bundles["evaluation"]
    candidate = (
        _tensor_variant(bundle, "nearest_bank_indices", (0, 0), -1)
        if case == "index-range"
        else _tensor_variant(bundle, "evaluation_masks", (0, 0, 0), 2)
    )

    with pytest.raises(ComparisonValidationError):
        _comparison(bundle, candidate)


def test_nearest_indices_exact_pair(
    _exact_scientific: ScientificComparison,
) -> None:
    component = _discrete(_exact_scientific, "nearest_bank_indices")

    assert component.exact
    assert component.exact_count == component.element_count
    assert component.mismatch_count == 0
    assert component.mismatch_rate == 0.0
    assert component.first_mismatches == ()


def test_nearest_index_mismatch_records_values_and_coordinate(
    _a2_bundles: dict[str, ComparableBundle],
) -> None:
    bundle = _a2_bundles["evaluation"]
    candidate = _tensor_variant(bundle, "nearest_bank_indices", (0, 0), 1)

    component = _discrete(_comparison(bundle, candidate), "nearest_bank_indices")

    assert not component.exact
    assert component.mismatch_count == 1
    assert component.exact_count == component.element_count - 1
    assert component.mismatch_rate == pytest.approx(1 / component.element_count)
    assert component.first_mismatches == (IndexMismatch((0, 0), 0, 1),)


def test_nearest_index_mismatch_cap_preserves_row_major_order_and_total(
    _a2_bundles: dict[str, ComparableBundle],
) -> None:
    bundle = _a2_bundles["evaluation"]
    indices = bundle.nearest_bank_indices.clone()
    mismatch_count = portability._INDEX_MISMATCH_LIMIT + 3
    indices.reshape(-1)[:mismatch_count].add_(1)
    candidate = replace(bundle, nearest_bank_indices=indices)

    component = _discrete(_comparison(bundle, candidate), "nearest_bank_indices")

    assert component.mismatch_count == mismatch_count
    assert len(component.first_mismatches) == portability._INDEX_MISMATCH_LIMIT
    assert tuple(item.coordinate for item in component.first_mismatches) == tuple(
        (0, column) for column in range(portability._INDEX_MISMATCH_LIMIT)
    )
    assert all(
        item.reference_value == index and item.candidate_value == index + 1
        for index, item in enumerate(component.first_mismatches)
    )


def test_mask_mismatch_count_is_complete_without_coordinate_payload(
    _a2_bundles: dict[str, ComparableBundle],
) -> None:
    bundle = _a2_bundles["evaluation"]
    masks = bundle.evaluation_masks.clone()
    masks[0, 0, 1] = 1
    candidate = replace(
        bundle,
        evaluation_masks=masks,
        metrics=replace(bundle.metrics, anomalous_pixel_count=2),
    )

    component = _discrete(_comparison(bundle, candidate), "evaluation_masks")

    assert not component.exact
    assert component.mismatch_count == 1
    assert component.exact_count == component.element_count - 1
    assert component.first_mismatches == ()


@pytest.mark.parametrize(
    ("case", "expected_false"),
    (
        ("category", ("category",)),
        ("profile", ("profile",)),
        (
            "source",
            ("clean_source", "scientific_source_commit"),
        ),
        ("lock", ("lock_identity",)),
        ("weight", ("weight_identity",)),
        ("inventory", ("inventory_identity",)),
        ("ordered-ids", ("ordered_test_sample_ids",)),
        ("ordered-labels", ("ordered_labels",)),
        (
            "tensor-contract",
            ("memory_bank_contract",),
        ),
    ),
)
def test_structurally_incomparable_candidates_expose_all_failed_gates_and_no_drift(
    _a2_bundles: dict[str, ComparableBundle],
    case: str,
    expected_false: tuple[str, ...],
) -> None:
    reference = _a2_bundles["evaluation"]
    if case == "category":
        candidate = _run_variant(reference, ("category",), "capsule")
    elif case == "profile":
        candidate = _run_variant(reference, ("profile_id",), "other-profile")
    elif case == "source":
        candidate = _run_variant(reference, ("source", "dirty"), True)
        candidate = _run_variant(candidate, ("source", "git_commit"), "e" * 40)
    elif case == "lock":
        candidate = _run_variant(reference, ("source", "uv_lock_sha256"), "e" * 64)
    elif case == "weight":
        candidate = _run_variant(reference, ("weights", "cached_file_sha256"), "e" * 64)
    elif case == "inventory":
        candidate = _run_variant(
            reference, ("inventory", "sample_inventory_sha256"), "e" * 64
        )
    elif case == "ordered-ids":
        candidate = replace(
            reference, test_sample_ids=tuple(reversed(reference.test_sample_ids))
        )
    elif case == "ordered-labels":
        candidate = replace(reference, test_labels=reference.test_labels.flip(0))
    else:
        candidate = _larger_bank_variant(reference)

    comparison = _comparison(reference, candidate)
    comparability = comparison.comparability[0]
    result = comparison.scientific_results[0]
    failed = {name for name, exact in comparability.gates if not exact}

    assert not comparability.comparable
    assert set(expected_false) <= failed
    assert result.status == "structurally_incomparable"
    assert result.floating_components is None
    assert result.discrete_components is None
    assert result.metrics is None
    document = json.loads(encode_scientific_comparison(comparison))
    assert document["scientific_results"]["candidate-1"] == {
        "status": "structurally_incomparable"
    }
    structural = {item.name: item for item in comparability.structural_components}
    if case == "ordered-ids":
        assert structural["test_sample_ids"].mismatch_count == 2
        assert not structural["test_sample_ids"].exact
    if case == "ordered-labels":
        assert structural["test_labels"].mismatch_count == 2
        assert not structural["test_labels"].exact


def test_scientific_gates_report_every_independent_incompatibility(
    _a2_bundles: dict[str, ComparableBundle],
) -> None:
    reference = _a2_bundles["evaluation"]
    candidate = _run_variant(reference, ("category",), "capsule")
    candidate = _run_variant(candidate, ("profile_id",), "other-profile")
    candidate = _run_variant(candidate, ("source", "uv_lock_sha256"), "e" * 64)

    failed = {
        name
        for name, exact in _comparison(reference, candidate).comparability[0].gates
        if not exact
    }

    assert {"category", "profile", "lock_identity"} <= failed


def test_sha256_identity_is_case_insensitive(
    _a2_bundles: dict[str, ComparableBundle],
) -> None:
    reference = _a2_bundles["evaluation"]
    candidate = _run_variant(
        reference, ("source", "uv_lock_sha256"), _ACCEPTED_LOCK_SHA256.upper()
    )
    candidate = _run_variant(
        candidate,
        ("weights", "cached_file_sha256"),
        _ACCEPTED_WEIGHT_SHA256.upper(),
    )

    comparison = _comparison(reference, candidate)

    assert comparison.comparability[0].comparable
    assert comparison.candidates[0].run["source"]["uv_lock_sha256"] == (
        _ACCEPTED_LOCK_SHA256
    )
    assert comparison.candidates[0].run["weights"]["cached_file_sha256"] == (
        _ACCEPTED_WEIGHT_SHA256
    )


def test_expected_platform_varying_identities_are_not_scientific_gates(
    _a2_bundles: dict[str, ComparableBundle],
) -> None:
    reference = _a2_bundles["evaluation"]
    candidate = _run_variant(reference, ("run_id",), "different-run")
    candidate = _run_variant(
        candidate, ("environment", "python_version"), "3.11.15-platform-build"
    )
    candidate = _run_variant(
        candidate,
        ("environment", "dependency_versions", "torch"),
        "2.13.0+platform",
    )
    candidate = _run_variant(
        candidate,
        ("environment", "platform_description"),
        "Private host diagnostic",
    )

    comparison = _comparison(reference, candidate)
    encoded = encode_scientific_comparison(comparison)

    assert comparison.comparability[0].comparable
    assert b"Private host diagnostic" not in encoded
    assert b"platform_description" not in encoded


def test_metric_deltas_use_frozen_order_and_exact_values(
    _exact_scientific: ScientificComparison,
) -> None:
    metrics = _exact_scientific.scientific_results[0].metrics
    assert metrics is not None

    assert tuple(item.metric_name for item in metrics) == (
        "image_auroc",
        "image_average_precision",
        "pixel_auroc",
    )
    assert all(
        item.reference_value == 1.0
        and item.candidate_value == 1.0
        and item.absolute_delta == 0.0
        for item in metrics
    )


def test_metric_delta_observes_controlled_nonzero_change(
    _a2_bundles: dict[str, ComparableBundle],
) -> None:
    bundle = _a2_bundles["evaluation"]
    candidate = replace(bundle, metrics=replace(bundle.metrics, image_auroc=0.75))

    comparison = _comparison(bundle, candidate)
    metrics = comparison.scientific_results[0].metrics
    assert metrics is not None

    assert metrics[0] == MetricDelta("image_auroc", 1.0, 0.75, 0.25)
    assert comparison.scientific_results[0].status == "observed_unclassified"


def test_exact_pair_remains_unclassified_and_has_no_policy_fields(
    _exact_scientific: ScientificComparison,
) -> None:
    document = json.loads(encode_scientific_comparison(_exact_scientific))

    assert _exact_scientific.scientific_results[0].status == ("observed_unclassified")
    assert "policy" not in document
    assert document["scientific_results"]["candidate-1"]["status"] not in {
        "accepted",
        "within_policy",
        "drift_detected",
        "passed",
        "failed",
    }
    assert b"policy_violation" not in encode_scientific_comparison(_exact_scientific)


def test_completed_mps_execution_is_a_normal_post_policy_candidate(
    _a2_bundles: dict[str, ComparableBundle],
) -> None:
    reference = _a2_bundles["evaluation"]
    candidate = _a2_bundles["mps"]

    comparison = _comparison(
        reference,
        candidate,
        candidate_roles=("post_policy_attempt",),
    )

    assert comparison.candidates[0].policy_role == "post_policy_attempt"
    assert comparison.candidates[0].requested_device == "mps"
    assert comparison.attempts == ()
    assert comparison.scientific_results[0].status == "observed_unclassified"


@pytest.mark.parametrize(
    ("status", "reason_code", "stage_code"),
    (
        ("unsupported", "mps_backend_unavailable", "device_resolution"),
        ("execution_failed", "operator_unsupported", "evaluation"),
    ),
)
def test_non_gating_execution_attempt_outcomes_are_separate_from_candidates(
    _a2_bundles: dict[str, ComparableBundle],
    status: str,
    reason_code: str,
    stage_code: str,
) -> None:
    attempt = ScientificExecutionAttempt(
        "mps-attempt",
        status,  # type: ignore[arg-type]
        reason_code,
        stage_code,
    )
    bundle = _a2_bundles["evaluation"]

    comparison = _comparison(bundle, bundle, attempts=(attempt,))

    assert comparison.attempts == (attempt,)
    assert comparison.scientific_results[0].status == "observed_unclassified"
    document = json.loads(encode_scientific_comparison(comparison))
    assert document["attempts"] == [
        {
            "environment_id": "mps-attempt",
            "gating": False,
            "policy_role": "post_policy_attempt",
            "reason_code": reason_code,
            "stage_code": stage_code,
            "status": status,
        }
    ]


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("reason_code", "RuntimeError: /Users/alice"),
        ("reason_code", "driver\nfailed"),
        ("reason_code", "a" * 65),
        ("stage_code", "model load"),
        ("stage_code", "../evaluation"),
        ("status", "completed"),
    ),
)
def test_execution_attempt_rejects_raw_or_impossible_values(
    field: str, value: object
) -> None:
    values = {
        "environment_id": "mps-attempt",
        "status": "unsupported",
        "reason_code": "mps_backend_unavailable",
        "stage_code": "device_resolution",
    }
    values[field] = value

    with pytest.raises(ComparisonValidationError):
        ScientificExecutionAttempt(**values)  # type: ignore[arg-type]


def test_candidate_and_attempt_environment_collision_is_rejected(
    _a2_bundles: dict[str, ComparableBundle],
) -> None:
    bundle = _a2_bundles["evaluation"]
    attempt = ScientificExecutionAttempt(
        "candidate-env",
        "unsupported",
        "mps_backend_unavailable",
        "device_resolution",
    )

    with pytest.raises(ComparisonValidationError):
        compare_scientific_bundles(
            _descriptor(bundle, "reference-env", "reference"),
            (_descriptor(bundle, "candidate-env", "holdout"),),
            generator=_A2_GENERATOR,
            attempts=(attempt,),
        )


def test_comparison_id_is_deterministic_and_input_sensitive(
    _a2_bundles: dict[str, ComparableBundle],
    _exact_scientific: ScientificComparison,
) -> None:
    bundle = _a2_bundles["evaluation"]
    repeated = _comparison(bundle, bundle)
    changed_generator = compare_scientific_bundles(
        _descriptor(bundle, "reference-env", "reference"),
        (_descriptor(bundle, "candidate-1", "holdout"),),
        generator=ScientificGenerator("e" * 40, False),
    )

    assert repeated.comparison_id == _exact_scientific.comparison_id
    assert changed_generator.comparison_id != repeated.comparison_id
    assert len(repeated.comparison_id) == 64
    assert all(character in "0123456789abcdef" for character in repeated.comparison_id)


def test_repeated_canonical_encoding_is_byte_identical_and_reloadable(
    _exact_scientific: ScientificComparison,
) -> None:
    first = encode_scientific_comparison(_exact_scientific)
    second = encode_scientific_comparison(_exact_scientific)

    assert first == second
    assert json.loads(first) == json.loads(second)
    assert first == _canonical(json.loads(first))
    assert first.count(b"\n") == 1
    assert first.endswith(b"\n")


def test_equivalent_reconstructed_records_encode_identically(
    tmp_path: Path,
) -> None:
    left = load_comparable_bundle(_bundle(tmp_path / "left"))
    right = load_comparable_bundle(_bundle(tmp_path / "right"))

    left_comparison = _comparison(left, left)
    right_comparison = _comparison(right, right)

    assert left is not right
    assert left.source_files == right.source_files
    assert left_comparison.comparison_id == right_comparison.comparison_id
    assert encode_scientific_comparison(left_comparison) == (
        encode_scientific_comparison(right_comparison)
    )


def test_canonical_output_contains_no_absolute_or_private_bundle_path(
    _a2_bundles: dict[str, ComparableBundle],
    _exact_scientific: ScientificComparison,
) -> None:
    bundle = _a2_bundles["evaluation"]
    encoded = encode_scientific_comparison(_exact_scientific)

    assert str(bundle.path).encode() not in encoded
    assert str(bundle.path.parent).encode() not in encoded
    assert b"/private/mvtec-ad" not in encoded
    assert b"dataset_root" not in encoded
    assert b"benchmark" not in json.loads(encoded)["reference"]


@pytest.mark.parametrize(
    ("path", "value"),
    (
        (
            ("environment", "python_version"),
            "Python (/usr/local/bin/python3)",
        ),
        (
            ("environment", "dependency_versions", "inspectrt"),
            "editable install /home/alice/private-project",
        ),
    ),
)
def test_scientific_identity_rejects_embedded_private_paths(
    _a2_bundles: dict[str, ComparableBundle],
    path: tuple[str, ...],
    value: str,
) -> None:
    bundle = _a2_bundles["evaluation"]
    candidate = _run_variant(bundle, path, value)

    with pytest.raises(ComparisonValidationError):
        _comparison(bundle, candidate)


def test_comparison_and_encoding_leave_source_files_tensors_and_records_unchanged(
    _a2_bundles: dict[str, ComparableBundle],
) -> None:
    bundle = _a2_bundles["evaluation"]
    source_state = _source_state(bundle.path)
    tensor_state = {
        name: getattr(bundle, name).clone()
        for name in (
            "test_labels",
            "image_scores",
            "memory_bank",
            "patch_distances",
            "nearest_bank_indices",
            "anomaly_maps",
            "evaluation_masks",
        )
    }
    tensor_ids = {name: id(getattr(bundle, name)) for name in tensor_state}
    record_state = (
        bundle.source_files,
        bundle.samples,
        bundle.predictions,
        bundle.test_sample_ids,
        bundle.memory_bank_metadata,
        bundle.metrics,
    )

    comparison = _comparison(bundle, bundle)
    encode_scientific_comparison(comparison)

    assert _source_state(bundle.path) == source_state
    assert all(
        id(getattr(bundle, name)) == tensor_ids[name]
        and torch.equal(getattr(bundle, name), before)
        for name, before in tensor_state.items()
    )
    assert (
        bundle.source_files,
        bundle.samples,
        bundle.predictions,
        bundle.test_sample_ids,
        bundle.memory_bank_metadata,
        bundle.metrics,
    ) == record_state
    with pytest.raises(FrozenInstanceError):
        comparison.schema_version = 2  # type: ignore[misc]
    with pytest.raises(TypeError):
        comparison.reference.run["category"] = "changed"  # type: ignore[index]


def _environment_value(
    environment_id: str,
    policy_role: str,
    requested_device: str = "cpu",
    *,
    execution_layer: str = "native",
) -> dict[str, object]:
    return {
        "environment_id": environment_id,
        "execution_layer": execution_layer,
        "hardware_label": f"Hardware {environment_id}",
        "os_label": "Test OS",
        "policy_role": policy_role,
        "requested_device": requested_device,
    }


def _environment_map_value(
    *candidates: dict[str, object],
    reference_device: str = "cpu",
    attempts: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return {
        "attempts": attempts or [],
        "candidates": list(candidates)
        or [_environment_value("candidate-1", "holdout")],
        "reference": _environment_value("reference-env", "reference", reference_device),
        "schema_id": "inspectrt_portability_environment_map_v1",
        "schema_version": 1,
    }


def _policy_value(**changes: object) -> dict[str, object]:
    value: dict[str, object] = {
        "calibration_environment_ids": ["calibration-env"],
        "category": "bottle",
        "derivation": {
            "comparison_ids": ["a" * 64],
            "method_id": "observed_envelope",
        },
        "discrete_output_requirements": [
            "test_sample_ids",
            "test_labels",
            "evaluation_masks",
            "nearest_bank_indices",
        ],
        "floating_component_limits": {
            name: {"atol": 0.0, "rtol": 0.0}
            for name in (
                "memory_bank",
                "patch_distances",
                "image_scores",
                "anomaly_maps",
            )
        },
        "holdout_environment_ids": ["candidate-1"],
        "limitation": (
            "This reviewed observed envelope is bounded evidence and not a universal "
            "guarantee."
        ),
        "metric_absolute_delta_limits": {
            "image_auroc": 0.0,
            "image_average_precision": 0.0,
            "pixel_auroc": 0.0,
        },
        "policy_id": "synthetic-policy-v1",
        "profile_id": "inspectrt_feature_memory_v1",
        "provenance_requirements": [
            "clean_source",
            "scientific_source_commit",
            "lock_identity",
            "weight_identity",
            "inventory_identity",
        ],
        "reference_environment_id": "reference-env",
        "reviewed_evidence_hashes": ["b" * 64],
        "schema_id": "inspectrt_portability_policy_v1",
        "schema_version": 1,
    }
    value.update(changes)
    return value


def _loaded_policy(tmp_path: Path, **changes: object) -> PortabilityPolicy:
    path = tmp_path / f"policy-{len(list(tmp_path.iterdir()))}.json"
    _write_json(path, _policy_value(**changes))
    return load_portability_policy(path)


def _policy_comparison(
    reference: ComparableBundle,
    candidate: ComparableBundle,
    policy: PortabilityPolicy,
    *,
    role: str = "holdout",
    environment_id: str = "candidate-1",
    attempts: tuple[ScientificExecutionAttempt, ...] = (),
) -> ScientificComparison:
    return compare_scientific_bundles(
        _descriptor(reference, "reference-env", "reference"),
        (_descriptor(candidate, environment_id, role),),
        generator=_A2_GENERATOR,
        attempts=attempts,
        policy=policy,
    )


def test_loads_valid_canonical_environment_map_with_ordered_attempts(
    tmp_path: Path,
) -> None:
    value = _environment_map_value(
        _environment_value("candidate-a", "calibration"),
        _environment_value("candidate-b", "holdout"),
        attempts=[
            {
                "environment_id": "mps-attempt",
                "gating": False,
                "policy_role": "post_policy_attempt",
                "reason_code": "operator_unsupported",
                "stage_code": "evaluation",
                "status": "execution_failed",
            }
        ],
    )
    path = tmp_path / "environment-map.json"
    _write_json(path, value)

    loaded = load_portability_environment_map(path)

    assert tuple(item.environment_id for item in loaded.candidates) == (
        "candidate-a",
        "candidate-b",
    )
    assert loaded.attempts[0].environment_id == "mps-attempt"
    with pytest.raises(FrozenInstanceError):
        loaded.schema_version = 2  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field", "value"),
    (("gating", True), ("policy_role", "holdout"), ("gating", _DELETE)),
)
def test_environment_map_requires_canonical_a2_attempt_records(
    tmp_path: Path, field: str, value: object
) -> None:
    attempt: dict[str, object] = {
        "environment_id": "mps-attempt",
        "gating": False,
        "policy_role": "post_policy_attempt",
        "reason_code": "operator_unsupported",
        "stage_code": "evaluation",
        "status": "execution_failed",
    }
    if value is _DELETE:
        del attempt[field]
    else:
        attempt[field] = value
    path = tmp_path / "environment-map.json"
    _write_json(path, _environment_map_value(attempts=[attempt]))
    with pytest.raises(ComparisonValidationError):
        load_portability_environment_map(path)


@pytest.mark.parametrize(
    "payload",
    (
        b'{"attempts":[],"candidates":[],"reference":{},"schema_id":"x","schema_version":1}\n',
        b'{"schema_version":1,"schema_version":1}\n',
        b'{ "schema_version":1 }\n',
        b'{"schema_version":1}',
    ),
)
def test_environment_map_rejects_malformed_or_noncanonical_bytes(
    tmp_path: Path, payload: bytes
) -> None:
    path = tmp_path / "environment-map.json"
    path.write_bytes(payload)
    with pytest.raises(BundleValidationError):
        load_portability_environment_map(path)


@pytest.mark.parametrize("unknown", ("hostname", "run_path", "timing_eligible"))
def test_environment_map_rejects_private_or_derived_unknown_fields(
    tmp_path: Path, unknown: str
) -> None:
    value = _environment_map_value()
    value["reference"][unknown] = "private-host"  # type: ignore[index]
    path = tmp_path / "environment-map.json"
    _write_json(path, value)
    with pytest.raises(ComparisonValidationError):
        load_portability_environment_map(path)


def test_environment_map_rejects_environment_id_collision(tmp_path: Path) -> None:
    path = tmp_path / "environment-map.json"
    _write_json(
        path,
        _environment_map_value(_environment_value("reference-env", "holdout")),
    )
    with pytest.raises(ComparisonValidationError, match="unique"):
        load_portability_environment_map(path)


@pytest.mark.parametrize("schema_version", (True, 1.0))
def test_environment_map_schema_version_is_not_coerced(
    tmp_path: Path, schema_version: object
) -> None:
    value = _environment_map_value()
    value["schema_version"] = schema_version
    path = tmp_path / "environment-map.json"
    _write_json(path, value)
    with pytest.raises(ComparisonValidationError, match="identity"):
        load_portability_environment_map(path)


@pytest.mark.parametrize(
    "label",
    (
        "192.168.1.20",
        "ThinkPad 192.168.1.20",
        "192.168.1.20:22",
        "p53.internal",
        "host p53.internal",
        "GPU [fe80::1]",
    ),
)
def test_environment_map_rejects_embedded_host_identity(
    tmp_path: Path, label: str
) -> None:
    value = _environment_map_value()
    value["reference"]["hardware_label"] = label  # type: ignore[index]
    path = tmp_path / "environment-map.json"
    _write_json(path, value)
    with pytest.raises(ComparisonValidationError):
        load_portability_environment_map(path)


def test_loads_valid_policy_with_exact_source_identity(tmp_path: Path) -> None:
    path = tmp_path / "policy.json"
    payload = _canonical(_policy_value())
    path.write_bytes(payload)

    policy = load_portability_policy(path)

    assert policy.source == CanonicalInputIdentity(
        len(payload), hashlib.sha256(payload).hexdigest()
    )
    assert tuple(policy.floating_component_limits) == (
        "memory_bank",
        "patch_distances",
        "image_scores",
        "anomaly_maps",
    )
    assert tuple(policy.metric_absolute_delta_limits) == (
        "image_auroc",
        "image_average_precision",
        "pixel_auroc",
    )


def test_tracked_portability_policy_is_the_reviewed_canonical_artifact() -> None:
    path = Path(__file__).resolve().parents[1] / "configs/portability_policy.json"
    payload = path.read_bytes()
    value = json.loads(payload)
    policy = load_portability_policy(path)

    assert payload == _canonical(value)
    assert policy.schema_version == 1
    assert policy.schema_id == "inspectrt_portability_policy_v1"
    assert policy.profile_id == "inspectrt_feature_memory_v1"
    assert policy.category == "bottle"
    assert policy.reference_environment_id == "p53-linux-t1000-cuda-reference"
    assert policy.calibration_environment_ids == (
        "p53-linux-t1000-cuda-control",
        "p53-linux-cpu",
        "rtx4080-wsl2-cuda",
    )
    assert policy.holdout_environment_ids == ("m1pro-macos-cpu",)
    assert policy.discrete_output_requirements == (
        "test_sample_ids",
        "test_labels",
        "evaluation_masks",
        "nearest_bank_indices",
    )
    assert {
        name: (limit.atol, limit.rtol)
        for name, limit in policy.floating_component_limits.items()
    } == {
        "memory_bank": (0.00011, 0),
        "patch_distances": (0.003, 0),
        "image_scores": (0.0015, 0),
        "anomaly_maps": (0.003, 0),
    }
    assert dict(policy.metric_absolute_delta_limits) == {
        "image_auroc": 0,
        "image_average_precision": 0,
        "pixel_auroc": 0.000000003,
    }
    assert policy.derivation == PolicyDerivation(
        "observed_envelope",
        ("88954c93e76c58d00b0ac6df8701b100df196996998da4c6f3c7ccd1bdc4c732",),
    )
    assert policy.reviewed_evidence_hashes == (
        "a14135a8c34484480503d95b77ee15607a9c2e5cf3700463834bb6bc4b672415",
        "e82246a2fa7a3efafc8c98abcfc85535730acd1ba62350420817e5b72b194426",
    )
    assert b"/" not in payload and b"\\" not in payload and b"_extra" not in payload


@pytest.mark.parametrize("case", ("spacing", "duplicate", "nan", "missing-lf"))
def test_policy_rejects_malformed_or_noncanonical_bytes(
    tmp_path: Path, case: str
) -> None:
    canonical = _canonical(_policy_value())
    if case == "spacing":
        payload = canonical.replace(b'"schema_version":1', b'"schema_version": 1')
    elif case == "duplicate":
        payload = canonical.replace(
            b'{"calibration_environment_ids"',
            b'{"schema_version":1,"calibration_environment_ids"',
        )
    elif case == "nan":
        payload = canonical.replace(b'"atol":0.0', b'"atol":NaN', 1)
    else:
        payload = canonical.removesuffix(b"\n")
    path = tmp_path / "policy.json"
    path.write_bytes(payload)
    with pytest.raises(BundleValidationError):
        load_portability_policy(path)


@pytest.mark.parametrize(
    ("case", "mutation"),
    (
        ("negative", ("memory_bank", "atol", -1.0)),
        ("missing-component", ("anomaly_maps", None, None)),
        ("unknown-component", ("unknown", "atol", 0.0)),
        ("missing-metric", ("pixel_auroc", None, None)),
        ("unknown-metric", ("unknown_metric", None, 0.0)),
    ),
)
def test_policy_rejects_incomplete_unknown_or_negative_limits(
    tmp_path: Path, case: str, mutation: tuple[str, str | None, float | None]
) -> None:
    value = _policy_value()
    name, field, replacement = mutation
    table_name = (
        "floating_component_limits"
        if "component" in case or case == "negative"
        else "metric_absolute_delta_limits"
    )
    table = value[table_name]
    assert isinstance(table, dict)
    if field is None and replacement is None:
        del table[name]
    elif field is None:
        table[name] = replacement
    elif name == "unknown":
        table[name] = {"atol": replacement, "rtol": 0.0}
    else:
        table[name][field] = replacement
    path = tmp_path / "policy.json"
    _write_json(path, value)
    with pytest.raises(ComparisonValidationError):
        load_portability_policy(path)


def test_policy_numbers_are_not_coerced(tmp_path: Path) -> None:
    value = _policy_value()
    value["floating_component_limits"]["memory_bank"]["atol"] = "0"  # type: ignore[index]
    path = tmp_path / "policy.json"
    _write_json(path, value)
    with pytest.raises(ComparisonValidationError, match="finite and nonnegative"):
        load_portability_policy(path)


def test_policy_rejects_numbers_outside_binary64_application_range(
    tmp_path: Path,
) -> None:
    value = _policy_value()
    value["floating_component_limits"]["memory_bank"]["rtol"] = 10**309  # type: ignore[index]
    path = tmp_path / "policy.json"
    _write_json(path, value)
    with pytest.raises(ComparisonValidationError, match="finite and nonnegative"):
        load_portability_policy(path)


def test_large_finite_integer_policy_limit_applies_without_overflow(
    tmp_path: Path, _a2_bundles: dict[str, ComparableBundle]
) -> None:
    value = _policy_value()
    value["floating_component_limits"]["memory_bank"]["rtol"] = 10**100  # type: ignore[index]
    path = tmp_path / "policy.json"
    _write_json(path, value)
    policy = load_portability_policy(path)
    bundle = _a2_bundles["evaluation"]
    assert (
        _policy_comparison(bundle, bundle, policy).scientific_results[0].status
        == "within_policy"
    )


@pytest.mark.parametrize(
    "changes",
    (
        {"calibration_environment_ids": ["reference-env"]},
        {"calibration_environment_ids": ["candidate-1"]},
        {"holdout_environment_ids": ["candidate-1", "candidate-1"]},
        {"holdout_environment_ids": []},
        {"reviewed_evidence_hashes": ["not-a-hash"]},
        {"policy_id": "arbitrary prose"},
        {"limitation": "Universal."},
    ),
)
def test_policy_rejects_invalid_scope_identity_or_review_record(
    tmp_path: Path, changes: dict[str, object]
) -> None:
    path = tmp_path / "policy.json"
    _write_json(path, _policy_value(**changes))
    with pytest.raises(ComparisonValidationError):
        load_portability_policy(path)


@pytest.mark.parametrize(
    "field",
    ("provenance_requirements", "discrete_output_requirements"),
)
def test_policy_requires_complete_exact_requirement_arrays(
    tmp_path: Path, field: str
) -> None:
    value = _policy_value()
    requirements = value[field]
    assert isinstance(requirements, list)
    requirements.pop()
    path = tmp_path / "policy.json"
    _write_json(path, value)
    with pytest.raises(ComparisonValidationError):
        load_portability_policy(path)


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        ({"reference_environment_id": "other-reference"}, "reference"),
        ({"profile_id": "other_profile"}, "profile"),
        ({"category": "capsule"}, "category"),
        ({"holdout_environment_ids": ["other-candidate"]}, "outside"),
    ),
)
def test_policy_scope_must_match_reference_and_candidate(
    tmp_path: Path,
    _a2_bundles: dict[str, ComparableBundle],
    changes: dict[str, object],
    message: str,
) -> None:
    policy = _loaded_policy(tmp_path, **changes)
    with pytest.raises(ComparisonValidationError, match=message):
        _policy_comparison(_a2_bundles["evaluation"], _a2_bundles["evaluation"], policy)


def test_observation_mode_remains_byte_identical_with_explicit_none(
    _a2_bundles: dict[str, ComparableBundle],
) -> None:
    bundle = _a2_bundles["evaluation"]
    implicit = _comparison(bundle, bundle)
    explicit = compare_scientific_bundles(
        _descriptor(bundle, "reference-env", "reference"),
        (_descriptor(bundle, "candidate-1", "holdout"),),
        generator=_A2_GENERATOR,
        policy=None,
    )
    assert implicit.comparison_id == explicit.comparison_id
    encoded = encode_scientific_comparison(implicit)
    assert encoded == encode_scientific_comparison(explicit)
    assert len(encoded) == 9349
    assert hashlib.sha256(encoded).hexdigest() == (
        "15e4b477cce38c4f84163c28c688ab8f33a1235616a353d28925d138b1eb13f4"
    )


def test_exact_pair_is_within_zero_policy(
    tmp_path: Path, _a2_bundles: dict[str, ComparableBundle]
) -> None:
    policy = _loaded_policy(tmp_path)
    bundle = _a2_bundles["evaluation"]
    comparison = _policy_comparison(bundle, bundle, policy)

    assert comparison.scientific_results[0].status == "within_policy"
    assert all(
        component.statistics.policy_violation_count == 0
        for component in comparison.scientific_results[0].floating_components or ()
    )
    document = json.loads(encode_scientific_comparison(comparison))
    assert document["policy"] == {
        "policy_id": policy.policy_id,
        "sha256": policy.source.sha256,
    }


@pytest.mark.parametrize(
    ("atol", "expected"), ((1.0, "within_policy"), (0.5, "drift_detected"))
)
def test_elementwise_float_policy_is_inclusive_and_counts_exactly(
    tmp_path: Path,
    _a2_bundles: dict[str, ComparableBundle],
    atol: float,
    expected: str,
) -> None:
    limits = _policy_value()["floating_component_limits"]
    assert isinstance(limits, dict)
    limits["memory_bank"] = {"atol": atol, "rtol": 0.0}
    policy = _loaded_policy(tmp_path, floating_component_limits=limits)
    reference = _a2_bundles["evaluation"]
    bank = reference.memory_bank.clone()
    bank[0, 0] = 1.0
    bank[0, 1] = 0.75
    comparison = _policy_comparison(
        reference, replace(reference, memory_bank=bank), policy
    )
    statistics = _floating(comparison, "memory_bank")

    assert comparison.scientific_results[0].status == expected
    assert statistics.policy_violation_count == (0 if atol == 1.0 else 2)


def test_policy_violation_at_chunk_boundary_uses_bounded_traversal(
    tmp_path: Path,
    _a2_bundles: dict[str, ComparableBundle],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = _loaded_policy(tmp_path)
    reference = _a2_bundles["evaluation"]
    chunk_size = 100_000
    bank = reference.memory_bank.clone()
    bank.reshape(-1)[chunk_size] = 1.0
    monkeypatch.setattr(portability, "_FLOAT_CHUNK_SIZE", chunk_size)

    statistics = _floating(
        _policy_comparison(reference, replace(reference, memory_bank=bank), policy),
        "memory_bank",
    )

    assert statistics.policy_violation_count == 1


def test_relative_policy_limit_uses_absolute_reference_value(
    tmp_path: Path, _a2_bundles: dict[str, ComparableBundle]
) -> None:
    limits = _policy_value()["floating_component_limits"]
    assert isinstance(limits, dict)
    limits["patch_distances"] = {"atol": 0, "rtol": 0.6}
    policy = _loaded_policy(tmp_path, floating_component_limits=limits)
    reference = _a2_bundles["evaluation"]
    distances = reference.patch_distances.clone()
    distances[0, 0] = 0.3
    comparison = _policy_comparison(
        reference, replace(reference, patch_distances=distances), policy
    )
    assert comparison.scientific_results[0].status == "within_policy"
    assert _floating(comparison, "patch_distances").policy_violation_count == 0


@pytest.mark.parametrize(
    ("limit", "expected"), ((0.25, "within_policy"), (0.24, "drift_detected"))
)
def test_metric_absolute_delta_policy_is_inclusive(
    tmp_path: Path,
    _a2_bundles: dict[str, ComparableBundle],
    limit: float,
    expected: str,
) -> None:
    limits = dict(_policy_value()["metric_absolute_delta_limits"])
    limits["image_auroc"] = limit
    policy = _loaded_policy(tmp_path, metric_absolute_delta_limits=limits)
    reference = _a2_bundles["evaluation"]
    candidate = replace(reference, metrics=replace(reference.metrics, image_auroc=0.75))
    assert (
        _policy_comparison(reference, candidate, policy).scientific_results[0].status
        == expected
    )


@pytest.mark.parametrize("component", ("nearest_bank_indices", "evaluation_masks"))
def test_required_discrete_mismatch_detects_drift(
    tmp_path: Path,
    _a2_bundles: dict[str, ComparableBundle],
    component: str,
) -> None:
    reference = _a2_bundles["evaluation"]
    index = (0, 0) if component == "nearest_bank_indices" else (0, 0, 1)
    candidate = _tensor_variant(reference, component, index, 1)
    comparison = _policy_comparison(reference, candidate, _loaded_policy(tmp_path))
    assert comparison.scientific_results[0].status == "drift_detected"


def test_completed_post_policy_candidate_is_classified_but_attempt_is_not(
    tmp_path: Path, _a2_bundles: dict[str, ComparableBundle]
) -> None:
    attempt = ScientificExecutionAttempt(
        "failed-attempt", "unsupported", "backend_unavailable", "evaluation"
    )
    bundle = _a2_bundles["evaluation"]
    comparison = _policy_comparison(
        bundle,
        bundle,
        _loaded_policy(tmp_path),
        role="post_policy_attempt",
        environment_id="new-post-policy-env",
        attempts=(attempt,),
    )
    assert comparison.scientific_results[0].status == "within_policy"
    assert comparison.attempts == (attempt,)
    assert "failed-attempt" not in {
        result.environment_id for result in comparison.scientific_results
    }


def test_structurally_incomparable_remains_unclassified_in_policy_mode(
    tmp_path: Path, _a2_bundles: dict[str, ComparableBundle]
) -> None:
    reference = _a2_bundles["evaluation"]
    candidate = _run_variant(reference, ("profile_id",), "other-profile")
    result = _policy_comparison(
        reference, candidate, _loaded_policy(tmp_path)
    ).scientific_results[0]
    assert result.status == "structurally_incomparable"
    assert result.floating_components is None


def test_policy_hash_changes_comparison_id_and_equivalent_bytes_are_deterministic(
    tmp_path: Path, _a2_bundles: dict[str, ComparableBundle]
) -> None:
    bundle = _a2_bundles["evaluation"]
    first = _loaded_policy(tmp_path)
    second = _loaded_policy(tmp_path)
    changed = _loaded_policy(tmp_path, reviewed_evidence_hashes=["c" * 64])
    first_comparison = _policy_comparison(bundle, bundle, first)
    second_comparison = _policy_comparison(bundle, bundle, second)
    changed_comparison = _policy_comparison(bundle, bundle, changed)
    assert first.source.sha256 == second.source.sha256
    assert first_comparison == second_comparison
    assert first_comparison.comparison_id != changed_comparison.comparison_id


def test_loaded_policy_values_cannot_change_without_a_new_source_hash(
    tmp_path: Path,
) -> None:
    policy = _loaded_policy(tmp_path)
    changed_limits = dict(policy.floating_component_limits)
    changed_limits["memory_bank"] = PolicyTolerance(1.0, 0.0)
    with pytest.raises(ComparisonValidationError, match="source identity"):
        replace(policy, floating_component_limits=changed_limits)


def test_encoder_rejects_within_policy_result_with_a_violation(
    tmp_path: Path, _a2_bundles: dict[str, ComparableBundle]
) -> None:
    bundle = _a2_bundles["evaluation"]
    comparison = _policy_comparison(bundle, bundle, _loaded_policy(tmp_path))
    result = comparison.scientific_results[0]
    assert result.floating_components is not None
    first = result.floating_components[0]
    forged = replace(
        comparison,
        scientific_results=(
            replace(
                result,
                floating_components=(
                    replace(
                        first,
                        statistics=replace(first.statistics, policy_violation_count=1),
                    ),
                    *result.floating_components[1:],
                ),
            ),
        ),
    )
    with pytest.raises(ComparisonValidationError, match="contains a policy violation"):
        encode_scientific_comparison(forged)


@pytest.mark.parametrize("missing", ("gates", "floating", "discrete"))
def test_policy_encoder_requires_complete_acceptance_evidence(
    tmp_path: Path,
    _a2_bundles: dict[str, ComparableBundle],
    missing: str,
) -> None:
    bundle = _a2_bundles["evaluation"]
    comparison = _policy_comparison(bundle, bundle, _loaded_policy(tmp_path))
    if missing == "gates":
        forged = replace(
            comparison,
            comparability=(replace(comparison.comparability[0], gates=()),),
        )
    else:
        result = comparison.scientific_results[0]
        forged = replace(
            comparison,
            scientific_results=(replace(result, **{f"{missing}_components": ()}),),
        )
    with pytest.raises(ComparisonValidationError, match="order|candidate result"):
        encode_scientific_comparison(forged)


def _performance(
    reference: ComparableBundle,
    *candidates: ComparableBundle,
    candidate_layers: tuple[str, ...] = (),
) -> tuple[ScientificComparison, PortabilityPerformance, bytes]:
    reference_descriptor = _descriptor(reference, "reference-env", "reference")
    descriptors = tuple(
        _descriptor(
            candidate,
            f"candidate-{index}",
            "holdout",
            execution_layer=(
                candidate_layers[index - 1] if candidate_layers else "native"
            ),
        )
        for index, candidate in enumerate(candidates, start=1)
    )
    comparison = compare_scientific_bundles(
        reference_descriptor, descriptors, generator=_A2_GENERATOR
    )
    scientific = encode_scientific_comparison(comparison)
    return (
        comparison,
        build_portability_performance(
            comparison, scientific, reference_descriptor, descriptors
        ),
        scientific,
    )


def test_cpu_performance_copies_benchmark_observations_exactly(
    _a2_bundles: dict[str, ComparableBundle],
) -> None:
    benchmark = _a2_bundles["benchmark"]
    _, performance, scientific = _performance(benchmark, benchmark)
    encoded = encode_portability_performance(performance)
    document = json.loads(encoded)
    assert performance.status == "descriptive_only"
    assert len(performance.included_runs) == 2
    assert document["included_runs"][0][
        "measurements"
    ] == portability._thaw_comparison_json(
        benchmark.benchmark_metadata["results"]  # type: ignore[index]
    )
    assert document["scientific_sha256"] == hashlib.sha256(scientific).hexdigest()
    assert encoded == encode_portability_performance(performance)


def test_explicit_cuda_benchmark_is_timing_eligible(tmp_path: Path) -> None:
    benchmark = load_comparable_bundle(
        _bundle(tmp_path, benchmark=True, device="cuda:0")
    )
    _, performance, _ = _performance(benchmark, benchmark)
    assert [item.requested_device for item in performance.included_runs] == [
        "cuda:0",
        "cuda:0",
    ]


def test_cpu_and_cuda_benchmarks_share_the_performance_matrix(tmp_path: Path) -> None:
    cpu = load_comparable_bundle(_bundle(tmp_path / "cpu", benchmark=True))
    cuda = load_comparable_bundle(
        _bundle(tmp_path / "cuda", benchmark=True, device="cuda:0")
    )
    _, performance, _ = _performance(cpu, cuda)
    assert [item.requested_device for item in performance.included_runs] == [
        "cpu",
        "cuda:0",
    ]
    assert performance.excluded_candidates == ()


def test_scientific_provenance_failure_does_not_hide_valid_timing(
    _a2_bundles: dict[str, ComparableBundle],
) -> None:
    reference = _a2_bundles["benchmark"]
    candidate = _run_variant(reference, ("source", "dirty"), True)
    comparison, performance, _ = _performance(reference, candidate)
    assert comparison.scientific_results[0].status == "structurally_incomparable"
    assert [item.environment_id for item in performance.included_runs] == [
        "reference-env",
        "candidate-1",
    ]


def test_wsl2_execution_layer_is_preserved_in_performance(
    _a2_bundles: dict[str, ComparableBundle],
) -> None:
    benchmark = _a2_bundles["benchmark"]
    _, performance, _ = _performance(benchmark, benchmark, candidate_layers=("wsl2",))
    assert performance.included_runs[1].execution_layer == "wsl2"
    assert (
        json.loads(encode_portability_performance(performance))["included_runs"][1][
            "execution_layer"
        ]
        == "wsl2"
    )


def test_performance_descriptors_must_match_the_hashed_scientific_identity(
    _a2_bundles: dict[str, ComparableBundle],
) -> None:
    benchmark = _a2_bundles["benchmark"]
    reference = _descriptor(benchmark, "reference-env", "reference")
    candidate = _descriptor(benchmark, "candidate-1", "holdout")
    comparison = compare_scientific_bundles(
        reference, (candidate,), generator=_A2_GENERATOR
    )
    scientific = encode_scientific_comparison(comparison)
    relabeled = _descriptor(benchmark, "candidate-1", "holdout", execution_layer="wsl2")
    with pytest.raises(ComparisonValidationError, match="differ"):
        build_portability_performance(comparison, scientific, reference, (relabeled,))


def test_evaluation_candidate_is_scientific_and_performance_excluded(
    _a2_bundles: dict[str, ComparableBundle],
) -> None:
    comparison, performance, _ = _performance(
        _a2_bundles["benchmark"], _a2_bundles["evaluation"]
    )
    assert comparison.scientific_results[0].status == "observed_unclassified"
    assert performance.excluded_candidates == (
        PortabilityPerformanceExclusion("candidate-1", "evaluation_bundle"),
    )


def test_mps_benchmark_is_never_timing_eligible(tmp_path: Path) -> None:
    cpu = load_comparable_bundle(_bundle(tmp_path / "cpu", benchmark=True))
    mps = load_comparable_bundle(
        _bundle(tmp_path / "mps", benchmark=True, device="mps")
    )
    reference = _descriptor(cpu, "reference-env", "reference")
    candidate = _descriptor(mps, "candidate-1", "post_policy_attempt")
    comparison = compare_scientific_bundles(
        reference, (candidate,), generator=_A2_GENERATOR
    )
    scientific = encode_scientific_comparison(comparison)
    performance = build_portability_performance(
        comparison, scientific, reference, (candidate,)
    )
    assert performance.excluded_candidates[0].reason_code == (
        "unsupported_timing_device"
    )


@pytest.mark.parametrize("case", ("workload", "methodology", "profile", "inventory"))
def test_incompatible_benchmark_candidate_has_bounded_exclusion_reason(
    tmp_path: Path, _a2_bundles: dict[str, ComparableBundle], case: str
) -> None:
    reference = _a2_bundles["benchmark"]
    if case == "profile":
        candidate = _run_variant(reference, ("profile_id",), "other-profile")
    elif case == "inventory":
        candidate = _run_variant(
            reference, ("inventory", "sample_inventory_sha256"), "f" * 64
        )
    elif case == "workload":
        candidate = load_comparable_bundle(
            _bundle(tmp_path, benchmark=True, bank_chunk_size=8_192)
        )
    else:
        candidate = load_comparable_bundle(_bundle(tmp_path, benchmark=True, repeats=3))
    _, performance, _ = _performance(reference, candidate)
    assert performance.excluded_candidates[0].reason_code == (
        "methodology_mismatch" if case == "methodology" else "workload_mismatch"
    )


def test_performance_builder_binds_measurements_to_benchmark_source_snapshot(
    _a2_bundles: dict[str, ComparableBundle],
) -> None:
    reference = _a2_bundles["benchmark"]
    metadata = portability._thaw_comparison_json(reference.benchmark_metadata)
    assert isinstance(metadata, dict)
    metadata["results"]["one_off_ms"]["model_and_weight_load"] = 126.0
    frozen = portability._freeze_json(metadata)
    assert isinstance(frozen, dict | portability.MappingProxyType)
    candidate = replace(reference, benchmark_metadata=frozen)
    with pytest.raises(ComparisonValidationError, match="source snapshot"):
        _performance(reference, candidate)


def test_performance_record_revalidates_benchmark_measurements(
    _a2_bundles: dict[str, ComparableBundle],
) -> None:
    reference = _a2_bundles["benchmark"]
    _, performance, _ = _performance(reference, reference)
    run = performance.included_runs[1]
    measurements = portability._thaw_comparison_json(run.measurements)
    assert isinstance(measurements, dict)
    measurements["one_off_ms"]["model_and_weight_load"] = -1.0
    invalid = replace(run, measurements=measurements)
    with pytest.raises(BundleValidationError, match="nonnegative"):
        replace(
            performance,
            included_runs=(performance.included_runs[0], invalid),
        )


def test_performance_record_rejects_unknown_or_private_identity_values(
    _a2_bundles: dict[str, ComparableBundle],
) -> None:
    reference = _a2_bundles["benchmark"]
    _, performance, _ = _performance(reference, reference)
    with pytest.raises(ComparisonValidationError, match="workload fields"):
        replace(performance, workload={**dict(performance.workload), "unknown": 1})
    with pytest.raises(ComparisonValidationError, match="private path"):
        replace(performance, limitations=("source /home/alice/private-run",))
    with pytest.raises(ComparisonValidationError, match="run_id"):
        replace(performance.included_runs[0], run_id="/home/alice/private-run")


def test_performance_record_revalidates_each_full_methodology(
    _a2_bundles: dict[str, ComparableBundle],
) -> None:
    reference = _a2_bundles["benchmark"]
    _, performance, _ = _performance(reference, reference)
    run = performance.included_runs[1]
    methodology = portability._thaw_comparison_json(run.timing_methodology)
    assert isinstance(methodology, dict)
    methodology["repeat_count"] = 3
    invalid = replace(run, timing_methodology=methodology)
    with pytest.raises(ComparisonValidationError, match="methodology"):
        replace(
            performance,
            included_runs=(performance.included_runs[0], invalid),
        )


def test_performance_output_has_no_comparative_or_policy_fields(
    _a2_bundles: dict[str, ComparableBundle],
) -> None:
    _, performance, _ = _performance(_a2_bundles["benchmark"], _a2_bundles["benchmark"])
    document = json.loads(encode_portability_performance(performance))
    forbidden = {
        "ratio",
        "speedup",
        "percentage",
        "ranking",
        "winner",
        "regression",
        "confidence_interval",
        "policy",
        "scientific_status",
        "performance_gate",
    }

    rendered = json.dumps(document).casefold()
    assert not any(f'"{key}"' in rendered for key in forbidden)
    for pattern in (
        r"\bratios?\b",
        r"\bspeedups?\b",
        r"\bpercentages?\b",
        r"\brankings?\b",
        r"\bwinners?\b",
        r"\bregressions?\b",
        r"\bconfidence intervals?\b",
    ):
        assert re.search(pattern, rendered) is None


def _publication_bytes(scientific: bytes = b'{"science":1}\n') -> tuple[bytes, bytes]:
    performance = _canonical(
        {"scientific_sha256": hashlib.sha256(scientific).hexdigest()}
    )
    return scientific, performance


def test_atomic_publication_has_exact_inventory_and_rejects_existing(
    tmp_path: Path,
) -> None:
    scientific, performance = _publication_bytes()
    output = tmp_path / "comparison"
    assert publish_portability_records(scientific, performance, output) == output
    assert tuple(sorted(path.name for path in output.iterdir())) == (
        "performance.json",
        "scientific.json",
    )
    assert (output / "scientific.json").read_bytes() == scientific
    with pytest.raises(FileExistsError):
        publish_portability_records(scientific, performance, output)


def test_late_publication_failure_leaves_no_destination_or_temporary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scientific, performance = _publication_bytes()
    output = tmp_path / "comparison"
    original = portability._write_portability_file
    calls = 0

    def fail_second(path: Path, payload: bytes) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected late failure")
        original(path, payload)

    monkeypatch.setattr(portability, "_write_portability_file", fail_second)
    with pytest.raises(OSError, match="injected"):
        publish_portability_records(scientific, performance, output)
    assert not output.exists()
    assert list(tmp_path.iterdir()) == []


def test_publication_rejects_incorrect_scientific_hash(tmp_path: Path) -> None:
    with pytest.raises(ComparisonValidationError, match="exact scientific"):
        publish_portability_records(
            b'{"science":1}\n',
            _canonical({"scientific_sha256": "0" * 64}),
            tmp_path / "comparison",
        )


def test_publication_rejects_absolute_paths_in_canonical_payloads(
    tmp_path: Path,
) -> None:
    scientific = _canonical({"source_path": "/home/alice/private-run"})
    _, performance = _publication_bytes(scientific)
    with pytest.raises(ComparisonValidationError, match="private path"):
        publish_portability_records(scientific, performance, tmp_path / "comparison")


def test_high_level_publication_preserves_inputs_order_and_private_paths(
    tmp_path: Path,
) -> None:
    reference = _bundle(tmp_path / "reference", benchmark=True)
    candidate_a = _bundle(tmp_path / "candidate-a", benchmark=True)
    candidate_b = _bundle(tmp_path / "candidate-b", benchmark=True)
    environment_path = tmp_path / "environment-map.json"
    _write_json(
        environment_path,
        _environment_map_value(
            _environment_value("candidate-b", "holdout"),
            _environment_value("candidate-a", "holdout"),
        ),
    )
    before = {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for directory in (reference, candidate_a, candidate_b)
        for path in directory.iterdir()
    }
    inventories = {
        directory: tuple(sorted(path.name for path in directory.iterdir()))
        for directory in (reference, candidate_a, candidate_b)
    }
    environment_before = environment_path.read_bytes()
    output = tmp_path / "published"

    comparison, performance = publish_portability_comparison(
        reference,
        (candidate_b, candidate_a),
        environment_path,
        output,
        generator=_A2_GENERATOR,
    )

    assert tuple(item.environment_id for item in comparison.candidates) == (
        "candidate-b",
        "candidate-a",
    )
    assert tuple(item.environment_id for item in performance.included_runs[1:]) == (
        "candidate-b",
        "candidate-a",
    )
    assert environment_path.read_bytes() == environment_before
    assert all(
        hashlib.sha256(path.read_bytes()).hexdigest() == digest
        for path, digest in before.items()
    )
    assert all(
        tuple(sorted(path.name for path in directory.iterdir())) == inventory
        for directory, inventory in inventories.items()
    )
    combined = b"".join(path.read_bytes() for path in output.iterdir())
    assert str(tmp_path).encode() not in combined
    assert (
        hashlib.sha256((output / "scientific.json").read_bytes()).hexdigest()
        == json.loads((output / "performance.json").read_bytes())["scientific_sha256"]
    )


def test_high_level_candidate_count_mismatch_fails_before_publication(
    tmp_path: Path,
) -> None:
    environment_path = tmp_path / "environment-map.json"
    _write_json(
        environment_path,
        _environment_map_value(
            _environment_value("candidate-a", "holdout"),
            _environment_value("candidate-b", "holdout"),
        ),
    )
    with pytest.raises(ComparisonValidationError, match="count"):
        publish_portability_comparison(
            tmp_path / "never-loaded",
            (tmp_path / "one",),
            environment_path,
            tmp_path / "output",
            generator=_A2_GENERATOR,
        )
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("source", ("reference", "candidate"))
def test_high_level_rejects_output_inside_a_source_bundle(
    tmp_path: Path, source: str
) -> None:
    reference = _bundle(tmp_path / "reference", benchmark=True)
    candidate = _bundle(tmp_path / "candidate", benchmark=True)
    environment_path = tmp_path / "environment-map.json"
    _write_json(environment_path, _environment_map_value())
    inventories = {
        path: tuple(sorted(item.name for item in path.iterdir()))
        for path in (reference, candidate)
    }
    output = (reference if source == "reference" else candidate) / "comparison"

    with pytest.raises(ComparisonValidationError, match="outside source"):
        publish_portability_comparison(
            reference,
            (candidate,),
            environment_path,
            output,
            generator=_A2_GENERATOR,
        )

    assert not output.exists()
    assert all(
        tuple(sorted(item.name for item in path.iterdir())) == inventory
        for path, inventory in inventories.items()
    )


def test_high_level_policy_mode_preserves_map_and_policy_sources(
    tmp_path: Path,
) -> None:
    reference = _bundle(tmp_path / "reference", benchmark=True)
    candidate = _bundle(tmp_path / "candidate", benchmark=True)
    environment_path = tmp_path / "environment-map.json"
    policy_path = tmp_path / "policy.json"
    _write_json(environment_path, _environment_map_value())
    _write_json(policy_path, _policy_value())
    before = (environment_path.read_bytes(), policy_path.read_bytes())

    comparison, performance = publish_portability_comparison(
        reference,
        (candidate,),
        environment_path,
        tmp_path / "output",
        generator=_A2_GENERATOR,
        policy_path=policy_path,
    )

    assert comparison.scientific_results[0].status == "within_policy"
    assert comparison.policy is not None
    assert (environment_path.read_bytes(), policy_path.read_bytes()) == before
    assert "policy" not in json.loads(encode_portability_performance(performance))


def test_evaluation_reference_is_rejected_before_publication(tmp_path: Path) -> None:
    reference = _bundle(tmp_path / "reference")
    candidate = _bundle(tmp_path / "candidate")
    environment_path = tmp_path / "environment-map.json"
    _write_json(environment_path, _environment_map_value())
    with pytest.raises(ComparisonValidationError, match="eight-file benchmark"):
        publish_portability_comparison(
            reference,
            (candidate,),
            environment_path,
            tmp_path / "output",
            generator=_A2_GENERATOR,
        )
    assert not (tmp_path / "output").exists()
