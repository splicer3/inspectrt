from dataclasses import FrozenInstanceError, fields, replace
import hashlib
import json
import os
from pathlib import Path
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
    ComparableBundle,
    MemoryBankMetadata,
    PredictionRecord,
    SourceFileSnapshot,
    load_comparable_bundle,
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


def _metadata(run_id: str, device: str = "cpu") -> BaselineRunMetadata:
    return BaselineRunMetadata(
        run_id=run_id,
        created_at_utc="2026-07-15T12:00:00Z",
        dataset_root="/private/mvtec-ad",
        requested_device=device,
        bank_chunk_size=4096,
        git_commit="c" * 40,
        git_dirty=False,
        uv_lock_sha256="a" * 64,
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
        weight_file_sha256="b" * 64,
    )


def _benchmark(
    evaluation: CategoryEvaluation, metadata: BaselineRunMetadata
) -> BaselineBenchmark:
    statistics = benchmark_module._statistics((1.0, 2.0))
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
        methodology=benchmark_module._methodology(device, 1, 2),
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


def _bundle(tmp_path: Path, *, benchmark: bool = False, device: str = "cpu") -> Path:
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
    metadata = _metadata(run_id, device)
    return persist_baseline_run(
        evaluation,
        tmp_path,
        metadata,
        benchmark=_benchmark(evaluation, metadata) if benchmark else None,
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
        "ComparableBundle",
        "MemoryBankMetadata",
        "PredictionRecord",
        "SourceFileSnapshot",
        "load_comparable_bundle",
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
