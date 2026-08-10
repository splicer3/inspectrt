from dataclasses import FrozenInstanceError, replace
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import stat
from typing import Any

import pytest
import torch

import inspectrt.portability as portability
from inspectrt.artifacts import BaselineRunMetadata, persist_baseline_run
from inspectrt.benchmark import (
    BaselineBenchmark,
    _methodology as _timing_methodology,
    _timing_component,
    _workload as _timing_workload,
)
from inspectrt.data import MvtecSample
from inspectrt.evaluation import CategoryEvaluation, MvtecSampleObservation
from inspectrt.metrics import compute_threshold_free_metrics
from inspectrt.portability import (
    BundleMetrics,
    BundleValidationError,
    CanonicalInputIdentity,
    ComparableBundle,
    ComparisonValidationError,
    DiscreteComponentComparison,
    FloatingStatistics,
    IndexMismatch,
    MemoryBankMetadata,
    MetricDelta,
    PolicyDerivation,
    PolicyTolerance,
    PortabilityPolicy,
    PredictionRecord,
    ScientificBundleDescriptor,
    ScientificComparison,
    ScientificExecutionAttempt,
    ScientificGenerator,
    SourceFileSnapshot,
    TimingBundle,
    build_portability_performance_v2,
    compare_scientific_bundles,
    encode_portability_performance_v2,
    encode_scientific_comparison,
    load_comparable_bundle,
    load_portability_environment_map,
    load_portability_policy,
    load_portability_scientific_identity,
    load_timing_bundle,
    publish_portability_performance_v2,
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
_DELETE = object()
_EXECUTED_UNSAFE_PAYLOAD = False
_SCIENTIFIC_SOURCE_COMMIT = "bc330b9070c5ca8db9cb7cfbb27617256388536b"
_ACCEPTED_LOCK_SHA256 = (
    "ddaddc99b318a1c3a04d5d7cc433cf736d321b56f98a8ae8b532e71e19e6d76b"
)
_ACCEPTED_WEIGHT_SHA256 = (
    "11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca"
)
_INVENTORY_SHA256 = "022df1a49e0f1ab33d57696db2ed667a9603b493d838f4e2f2a850fd95a581c3"


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


def _bundle(
    tmp_path: Path,
    *,
    device: str = "cpu",
    bank_chunk_size: int = 16_384,
) -> Path:
    evaluation = _evaluation()
    run_id = "tiny-evaluation"
    metadata = _metadata(run_id, device, bank_chunk_size=bank_chunk_size)
    return persist_baseline_run(evaluation, tmp_path, metadata)


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


def test_cuda_run_requires_recorded_cuda_seed(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path, device="cuda:0")
    run = _json(bundle / "run.json")
    run["determinism"]["torch_cuda_seed_all"] = None
    _write_json(bundle / "run.json", run)

    with pytest.raises(BundleValidationError):
        load_comparable_bundle(bundle)


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


def test_rejects_noncanonical_or_malformed_json(tmp_path: Path) -> None:
    scenarios = (
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
    )
    for case in scenarios:
        bundle = _bundle(tmp_path / case)
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
            damaged = canonical.replace(
                b'"image_auroc":1.0', b'"image_auroc":-Infinity'
            )
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
    bundle = _bundle(tmp_path)
    loaded = load_comparable_bundle(bundle)

    assert tuple(snapshot.name for snapshot in loaded.source_files) == _EVALUATION_FILES
    assert all(type(snapshot.byte_count) is int for snapshot in loaded.source_files)
    assert all(
        snapshot.byte_count == (bundle / snapshot.name).stat().st_size
        and snapshot.sha256
        == hashlib.sha256((bundle / snapshot.name).read_bytes()).hexdigest()
        for snapshot in loaded.source_files
    )


def test_successful_loading_does_not_change_source_bundle(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
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
    ("filename", "path", "value"),
    (
        ("run.json", ("unexpected",), 1),
        ("run.json", ("batch_size",), _DELETE),
        ("run.json", ("source", "unexpected"), 1),
        ("run.json", ("source", "dirty"), _DELETE),
        ("metrics.json", ("unexpected",), 1),
        ("metrics.json", ("pixel_auroc",), _DELETE),
    ),
)
def test_rejects_unknown_or_missing_json_fields(
    tmp_path: Path,
    filename: str,
    path: tuple[str, ...],
    value: object,
) -> None:
    bundle = _bundle(tmp_path)
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


def test_rejects_invalid_run_identity_or_contract(tmp_path: Path) -> None:
    scenarios = (
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
    )
    for index, (path, value) in enumerate(scenarios):
        bundle = _bundle(tmp_path / str(index))
        run = _json(bundle / "run.json")
        _set_path(run, path, value)
        _write_json(bundle / "run.json", run)

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


def test_scientific_comparison_accepts_seven_file_evaluation_bundles(
    _a2_bundles: dict[str, ComparableBundle],
) -> None:
    comparison = _comparison(_a2_bundles["evaluation"], _a2_bundles["evaluation"])

    assert comparison.comparability[0].comparable
    assert comparison.scientific_results[0].status == "observed_unclassified"
    assert comparison.reference.bundle_kind == "evaluation"
    assert comparison.candidates[0].bundle_kind == "evaluation"
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
    assert "policy" not in json.loads(encoded)


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


_TIMING_MATRIX = (
    (
        "p53-linux-t1000-cuda-reference",
        "reference",
        "Ubuntu 24.04.4",
        "native",
        "NVIDIA Quadro T1000",
        "cuda:0",
    ),
    (
        "p53-linux-t1000-cuda-control",
        "same_stack_control",
        "Ubuntu 24.04.4",
        "native",
        "NVIDIA Quadro T1000",
        "cuda:0",
    ),
    (
        "p53-linux-cpu",
        "calibration",
        "Ubuntu 24.04.4",
        "native",
        "Intel Core i7-9850H CPU",
        "cpu",
    ),
    (
        "rtx4080-wsl2-cuda",
        "calibration",
        "Ubuntu 24.04.4 under WSL 2",
        "wsl2",
        "NVIDIA GeForce RTX 4080 SUPER",
        "cuda:0",
    ),
    (
        "m1pro-macos-cpu",
        "holdout",
        "macOS 26.5.2 build 25F84 arm64",
        "native",
        "Apple M1 Pro CPU",
        "cpu",
    ),
    (
        "m1pro-macos-mps",
        "post_policy_attempt",
        "macOS 26.5.2 build 25F84 arm64",
        "native",
        "Apple M1 Pro integrated GPU",
        "mps",
    ),
)
_TIMING_RUN_IDS = (
    "ptv2-bottle-p53-linux-t1000-cuda-reference-r01-4f23067",
    "ptv2-bottle-p53-linux-t1000-cuda-control-r01-4f23067",
    "ptv2-bottle-p53-linux-cpu-r01-4f23067",
    "ptv2-bottle-rtx4080-wsl2-cuda-r01-4f23067",
    "ptv2-bottle-m1pro-macos-cpu-r01-4f23067",
    "ptv2-bottle-m1pro-macos-mps-r01-4f23067",
)


def _timing_environment(device: str, hardware: str) -> dict[str, object]:
    if device == "cpu":
        return {"kind": "cpu", "properties": {}}
    if device == "mps":
        return {
            "kind": "mps",
            "properties": {
                "available": True,
                "built": True,
                "pytorch_enable_mps_fallback": "unset",
            },
        }
    return {
        "kind": "cuda",
        "properties": {
            "available": True,
            "compute_capability": [
                8 if "RTX 4080" in hardware else 7,
                9 if "RTX 4080" in hardware else 5,
            ],
            "device_index": 0,
            "device_name": (
                hardware if "RTX 4080" in hardware else hardware.removeprefix("NVIDIA ")
            ),
            "pytorch_cuda_runtime_version": "13.0",
        },
    }


def _timing_memory(device: str) -> dict[str, object]:
    if device == "cpu":
        return {"host_peak_memory": "not_measured", "kind": "cpu"}
    if device.startswith("cuda:"):
        return {
            "kind": "cuda",
            "peak_allocated_bytes": 438_304_768,
            "peak_reserved_bytes": 438_304_768,
            "peak_window": "after_warmups_through_all_measured_passes",
        }
    return {
        "kind": "mps",
        "observations": [
            {
                "boundary": boundary,
                "current_allocated_bytes": 1,
                "driver_allocated_bytes": 2,
            }
            for boundary in (
                "after_setup",
                "after_warmups",
                "after_measured_passes",
            )
        ],
        "peak_memory": "not_available_in_selected_pytorch_api",
        "recommended_max_memory_bytes": 3,
    }


def _timing_bundle_value(index: int) -> tuple[dict[str, object], dict[str, object]]:
    _, _, _, _, hardware, device = _TIMING_MATRIX[index]
    run_id = _TIMING_RUN_IDS[index]
    created = f"2026-08-09T21:{index:02d}:00.000000Z"
    repeated = list(range(30))
    result = {
        "memory_observations": _timing_memory(device),
        "one_off": {
            name: _timing_component([value])
            for name, value in (
                ("model_and_weight_load", 1),
                ("full_nominal_bank_build", 2),
                ("bank_transfer_and_device_setup", 3),
            )
        },
        "repeated_stages": {
            name: _timing_component(repeated)
            for name in (
                "image_decode",
                "canonical_image_preprocessing",
                "host_to_device_transfer",
                "frozen_feature_extraction",
                "exact_chunked_retrieval",
                "anomaly_map_reconstruction",
            )
        },
        "synchronized_end_to_end": _timing_component(repeated),
    }
    benchmark = BaselineBenchmark(
        schema_version=2,
        profile_id="inspectrt_feature_memory_v1",
        category="bottle",
        device=device,
        benchmark_sample_id="mvtec_ad/bottle/test/broken_large/000.png",
        run_id=run_id,
        created_at_utc=created,
        workload=_timing_workload(),
        methodology=_timing_methodology(),
        environment=_timing_environment(device, hardware),
        results=result,
    ).to_json_value()
    run = {
        "bank_chunk_size": 16384,
        "batch_size": 1,
        "benchmark": {
            "artifact": "benchmark.json",
            "present": True,
            "schema_version": 2,
            "timing_device": device,
        },
        "category": "bottle",
        "dataset_root": "/private/test-data",
        "determinism": {
            "allow_tf32": False,
            "cublas_workspace_config": ":4096:8",
            "cudnn_benchmark": False,
            "deterministic_algorithms_warn_only": False,
            "fp32_precision": "ieee",
            "numpy_seed": 0,
            "python_random_seed": 0,
            "torch_cpu_seed": 0,
            "torch_cuda_seed_all": 0 if device.startswith("cuda:") else None,
            "use_deterministic_algorithms": True,
        },
        "device": device,
        "environment": {
            "created_at_utc": created,
            "dependency_versions": {
                "inspectrt": "0.1.0",
                "numpy": "2.4.6",
                "pillow": "12.3.0",
                "scikit-learn": "1.9.0",
                "torch": "2.13.0",
                "torchvision": "0.28.0",
            },
            "platform_description": "Synthetic platform",
            "python_version": "3.11.15",
        },
        "feature_extractor": "ResNet-50",
        "feature_layer": "layer2",
        "inventory": {
            "anomalous_test_sample_count": 63,
            "sample_inventory_sha256": _INVENTORY_SHA256,
            "test_good_sample_count": 20,
            "test_sample_count": 83,
            "total_sample_count": 292,
            "training_sample_count": 209,
        },
        "map_interpolation": portability._MAP_INTERPOLATION,
        "preprocessing_profile": "inspectrt_resize256_v1",
        "profile_id": "inspectrt_feature_memory_v1",
        "retrieval_semantics": "exact top-1 squared L2",
        "run_id": run_id,
        "schema_version": 1,
        "source": {
            "dirty": False,
            "git_commit": "4f230679d52b5ed08e43230ebb1308cb85a33e57",
            "uv_lock_sha256": (
                "4464c375e3bf0f9c575504b427a0e82aedc954ef3491807306b72c382ce07d5c"
            ),
        },
        "tensors": {
            "anomaly_maps": {"dtype": "float32", "shape": [83, 256, 256]},
            "evaluation_masks": {"dtype": "uint8", "shape": [83, 256, 256]},
            "image_scores": {"dtype": "float32", "shape": [83]},
            "memory_bank": {
                "byte_count": 438304768,
                "dtype": "float32",
                "shape": [214016, 512],
            },
            "nearest_bank_indices": {"dtype": "int64", "shape": [83, 1024]},
            "patch_distances": {"dtype": "float32", "shape": [83, 1024]},
            "test_labels": {"dtype": "uint8", "shape": [83]},
        },
        "weights": {
            "cached_file_sha256": _ACCEPTED_WEIGHT_SHA256,
            "enum": "ResNet50_Weights.IMAGENET1K_V2",
            "source_url": ("https://download.pytorch.org/models/resnet50-11ad3fa6.pth"),
        },
    }
    return run, benchmark


def _write_timing_bundle(root: Path, index: int) -> Path:
    run, benchmark = _timing_bundle_value(index)
    path = root / _TIMING_RUN_IDS[index]
    path.mkdir(parents=True)
    for name in _EVALUATION_FILES:
        (path / name).write_bytes(f"synthetic-{name}\n".encode())
    _write_json(path / "run.json", run)
    _write_json(path / "benchmark.json", benchmark)
    return path


def _timing_inputs(
    tmp_path: Path,
) -> tuple[
    Mapping[str, object],
    PortabilityPolicy,
    portability.PortabilityEnvironmentMap,
    tuple[TimingBundle, ...],
]:
    root = Path(__file__).resolve().parents[1]
    environment_map_path = tmp_path / "environment-map.json"
    _write_json(
        environment_map_path,
        {
            "attempts": [],
            "candidates": [
                {
                    "environment_id": item[0],
                    "policy_role": item[1],
                    "os_label": item[2],
                    "execution_layer": item[3],
                    "hardware_label": item[4],
                    "requested_device": item[5],
                }
                for item in _TIMING_MATRIX[1:]
            ],
            "reference": {
                "environment_id": _TIMING_MATRIX[0][0],
                "policy_role": _TIMING_MATRIX[0][1],
                "os_label": _TIMING_MATRIX[0][2],
                "execution_layer": _TIMING_MATRIX[0][3],
                "hardware_label": _TIMING_MATRIX[0][4],
                "requested_device": _TIMING_MATRIX[0][5],
            },
            "schema_id": "inspectrt_portability_environment_map_v1",
            "schema_version": 1,
        },
    )
    bundle_root = tmp_path / "bundles"
    paths = tuple(_write_timing_bundle(bundle_root, index) for index in range(6))
    return (
        load_portability_scientific_identity(
            root / "docs/evidence/inspectrt_cross_platform_evidence_v2/scientific.json"
        ),
        load_portability_policy(root / "configs/portability_policy.json"),
        load_portability_environment_map(environment_map_path),
        tuple(load_timing_bundle(path) for path in paths),
    )


def _build_timing_performance(
    inputs: tuple[
        Mapping[str, object],
        PortabilityPolicy,
        portability.PortabilityEnvironmentMap,
        tuple[TimingBundle, ...],
    ],
    bundles: Sequence[TimingBundle] | None = None,
) -> Mapping[str, object]:
    scientific, policy, environment_map, loaded = inputs
    return build_portability_performance_v2(
        scientific,
        policy,
        environment_map,
        loaded if bundles is None else bundles,
        generator=ScientificGenerator("a" * 40, True),
    )


def test_timing_v2_loads_and_aggregates_without_tensor_deserialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        portability.torch,
        "load",
        lambda *args, **kwargs: pytest.fail("timing loader deserialized a tensor"),
    )
    inputs = _timing_inputs(tmp_path)
    performance = _build_timing_performance(inputs)
    encoded = encode_portability_performance_v2(performance)
    loaded = json.loads(encoded)

    assert set(loaded) == {
        "environment_order",
        "generator",
        "limitations",
        "milestone_id",
        "performance_id",
        "policy",
        "runs",
        "schema_id",
        "schema_version",
        "scientific",
        "status",
        "timing_harness",
        "timing_methodology",
        "workload",
    }
    assert loaded["environment_order"] == [item[0] for item in _TIMING_MATRIX]
    assert loaded["runs"][3]["environment"]["execution_layer"] == "wsl2"
    assert (
        loaded["runs"][5]["environment"]["backend"]["properties"][
            "pytorch_enable_mps_fallback"
        ]
        == "unset"
    )
    assert (
        len(
            loaded["runs"][0]["measurements"]["repeated_stages"]["image_decode"][
                "raw_ns"
            ]
        )
        == 30
    )
    assert "/private/test-data" not in encoded.decode()
    assert all(str(bundle.path) not in encoded.decode() for bundle in inputs[3])


def test_performance_v2_rejects_incomplete_duplicate_extra_and_reordered_matrix(
    tmp_path: Path,
) -> None:
    inputs = _timing_inputs(tmp_path)
    bundles = inputs[3]
    scenarios = (
        bundles[:-1],
        (bundles[0], bundles[0], *bundles[2:]),
        (*bundles, bundles[0]),
        (bundles[1], bundles[0], *bundles[2:]),
    )
    for scenario in scenarios:
        with pytest.raises(ComparisonValidationError, match="six-run matrix"):
            _build_timing_performance(inputs, scenario)


def test_performance_v2_rejects_common_provenance_runtime_and_contract_mismatch(
    tmp_path: Path,
) -> None:
    inputs = _timing_inputs(tmp_path)
    bundles = inputs[3]
    methodology = dict(bundles[1].methodology)
    methodology["repeat_count"] = 29
    workload = dict(bundles[1].workload)
    workload["Q"] = 1
    dependencies = dict(bundles[1].dependency_versions)
    dependencies["numpy"] = "0.0.0"
    replacements = (
        {"source_commit": "b" * 40},
        {"uv_lock_sha256": "c" * 64},
        {"dependency_versions": dependencies},
        {"methodology": methodology},
        {"workload": workload},
    )
    for changes in replacements:
        scenario = (bundles[0], replace(bundles[1], **changes), *bundles[2:])
        with pytest.raises(ComparisonValidationError, match="mismatch"):
            _build_timing_performance(inputs, scenario)


def test_timing_v2_rejects_schema_one(tmp_path: Path) -> None:
    path = _write_timing_bundle(tmp_path, 0)
    run = _json(path / "run.json")
    run["benchmark"]["schema_version"] = 1
    benchmark = _json(path / "benchmark.json")
    benchmark["schema_version"] = 1
    _write_json(path / "run.json", run)
    _write_json(path / "benchmark.json", benchmark)
    with pytest.raises(BundleValidationError):
        load_timing_bundle(path)


def test_timing_v2_rejects_raw_summary_mismatch_and_changed_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write_timing_bundle(tmp_path / "summary", 0)
    benchmark = _json(path / "benchmark.json")
    benchmark["results"]["repeated_stages"]["image_decode"]["summary_ns"]["p50"] = -1.0
    _write_json(path / "benchmark.json", benchmark)
    with pytest.raises(BundleValidationError, match="raw_ns"):
        load_timing_bundle(path)

    changed_path = _write_timing_bundle(tmp_path / "changed", 0)
    snapshot = portability._snapshot_sources
    calls = 0

    def change_after_snapshot(
        bundle_path: Path, names: tuple[str, ...]
    ) -> tuple[SourceFileSnapshot, ...]:
        nonlocal calls
        result = snapshot(bundle_path, names)
        calls += 1
        if calls == 1:
            (bundle_path / "metrics.json").write_bytes(b"changed\n")
        return result

    monkeypatch.setattr(portability, "_snapshot_sources", change_after_snapshot)
    with pytest.raises(BundleValidationError, match="changed"):
        load_timing_bundle(changed_path)


def test_performance_v2_encoding_and_publication_are_deterministic_and_atomic(
    tmp_path: Path,
) -> None:
    inputs = _timing_inputs(tmp_path)
    first = encode_portability_performance_v2(_build_timing_performance(inputs))
    second = encode_portability_performance_v2(_build_timing_performance(inputs))
    output = tmp_path / "performance_v2.json"

    assert first == second == _canonical(json.loads(first))
    assert publish_portability_performance_v2(first, output) == output
    assert output.read_bytes() == first
    with pytest.raises(FileExistsError):
        publish_portability_performance_v2(first, output)


def test_performance_v2_rejects_private_or_nonreviewed_environment(
    tmp_path: Path,
) -> None:
    inputs = _timing_inputs(tmp_path)
    with pytest.raises(ComparisonValidationError):
        replace(inputs[2].reference, hardware_label="host p53.internal")
    public_but_wrong = replace(inputs[2].reference, hardware_label="Different GPU")
    environment_map = replace(inputs[2], reference=public_but_wrong)
    with pytest.raises(ComparisonValidationError, match="reviewed six-row"):
        build_portability_performance_v2(
            inputs[0],
            inputs[1],
            environment_map,
            inputs[3],
            generator=ScientificGenerator("a" * 40, True),
        )
