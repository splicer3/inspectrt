from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

import pytest
import torch

from inspectrt.artifacts import BaselineRunMetadata, persist_baseline_run
from inspectrt.benchmark import BaselineBenchmark
import inspectrt.benchmark as benchmark_module
from inspectrt.data import MvtecSample
from inspectrt.evaluation import CategoryEvaluation, MvtecSampleObservation
from inspectrt.metrics import ThresholdFreeMetrics, compute_threshold_free_metrics

_FILES = {
    "anomaly_maps.pt",
    "memory_bank.pt",
    "metrics.json",
    "predictions.jsonl",
    "retrieval.pt",
    "run.json",
    "samples.jsonl",
}
_BENCHMARK_FILES = {*_FILES, "benchmark.json"}
_TIMED_STAGES = (
    "image_decode",
    "canonical_image_preprocessing",
    "host_to_device_transfer",
    "frozen_feature_extraction",
    "exact_chunked_retrieval",
    "anomaly_map_reconstruction",
)


def _observation(
    sample_id: str,
    split: str,
    defect: str,
    anomalous: bool,
    image: str,
    mask: str | None = None,
) -> MvtecSampleObservation:
    sample = MvtecSample(sample_id, "bottle", split, defect, anomalous, image, mask)
    return MvtecSampleObservation(sample, 900, 900, "RGB")


def _evaluation() -> CategoryEvaluation:
    good = _observation(
        "mvtec_ad/bottle/test/good/001.png",
        "test",
        "good",
        False,
        "bottle/test/good/001.png",
    )
    anomaly = _observation(
        "mvtec_ad/bottle/test/crack/002.png",
        "test",
        "crack",
        True,
        "bottle/test/crack/002.png",
        "bottle/ground_truth/crack/002_mask.png",
    )
    train = _observation(
        "mvtec_ad/bottle/train/good/000.png",
        "train",
        "good",
        False,
        "bottle/train/good/000.png",
    )
    labels = torch.tensor((0, 1), dtype=torch.uint8)
    masks = torch.zeros(2, 256, 256, dtype=torch.uint8)
    masks[1, 0, 0] = 1
    maps = torch.zeros(2, 256, 256)
    maps[1, 0, 0] = 1
    distances = torch.full((2, 1024), 0.1)
    distances[1].fill_(0.2)
    distances[1, -1] = 0.9
    scores = distances.max(dim=1).values.contiguous()
    return CategoryEvaluation(
        "bottle",
        (good, anomaly, train),
        (good, anomaly),
        torch.zeros(1024, 512),
        labels,
        masks,
        distances,
        torch.arange(1024).expand(2, -1).contiguous(),
        scores,
        maps,
        compute_threshold_free_metrics(labels, scores, masks, maps),
    )


def _metadata(run_id: str = "baseline-run", **changes: object) -> BaselineRunMetadata:
    values = {
        "run_id": run_id,
        "created_at_utc": "2026-07-15T12:00:00+00:00",
        "dataset_root": "/private/mvtec-ad",
        "requested_device": "cuda:0",
        "bank_chunk_size": 16384,
        "git_commit": "c" * 40,
        "git_dirty": True,
        "uv_lock_sha256": "a" * 64,
        "python_version": "3.11.15",
        "platform_description": "Linux-x86_64-qualità",
        "dependency_versions": {"torch": "2.13.0", "optional": None},
        "determinism_flags": {"deterministic": True, "seed": 0, "tf32": False},
        "weight_enum": "ResNet50_Weights.IMAGENET1K_V2",
        "weight_source_url": "https://example.invalid/resnet50.pth",
        "weight_file_sha256": "b" * 64,
    }
    values.update(changes)
    return BaselineRunMetadata(**values)  # type: ignore[arg-type]


def _benchmark(metadata: BaselineRunMetadata) -> BaselineBenchmark:
    repeated = list(range(30, 0, -1))

    def measurement(raw: list[int]) -> dict[str, object]:
        return {"raw_ns": raw, "summary_ns": benchmark_module._summary_ns(raw)}

    return BaselineBenchmark(
        schema_version=2,
        profile_id="inspectrt_feature_memory_v1",
        category="bottle",
        device="cpu",
        benchmark_sample_id="mvtec_ad/bottle/test/broken_large/000.png",
        run_id=metadata.run_id,
        created_at_utc=metadata.created_at_utc,
        workload=benchmark_module._workload(),
        methodology=benchmark_module._methodology(),
        environment={"kind": "cpu", "properties": {}},
        results={
            "one_off": {
                "model_and_weight_load": measurement([11]),
                "full_nominal_bank_build": measurement([12]),
                "bank_transfer_and_device_setup": measurement([13]),
            },
            "repeated_stages": {
                name: measurement(repeated.copy()) for name in _TIMED_STAGES
            },
            "synchronized_end_to_end": measurement(repeated.copy()),
            "memory_observations": {
                "kind": "cpu",
                "host_peak_memory": "not_measured",
            },
        },
    )


def _json_bytes(value: object) -> bytes:
    dumped = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return f"{dumped}\n".encode()


def _records(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_bytes().splitlines()]


def _assert_canonical(path: Path) -> None:
    payload = path.read_bytes()
    assert payload.endswith(b"\n") and not payload.endswith(b"\n\n")
    assert b"\r" not in payload
    assert all(
        line == _json_bytes(json.loads(line)) for line in payload.splitlines(True)
    )


def test_persists_complete_recomputable_bundle_without_mutation(tmp_path: Path) -> None:
    evaluation = _evaluation()
    metadata = _metadata()
    tensor_names = (
        "memory_bank",
        "test_labels",
        "pixel_masks",
        "patch_distances",
        "nearest_bank_indices",
        "image_scores",
        "anomaly_maps",
    )
    originals = {name: getattr(evaluation, name).clone() for name in tensor_names}

    run_dir = persist_baseline_run(evaluation, tmp_path, metadata)

    assert run_dir == tmp_path / "runs" / metadata.run_id
    assert {path.name for path in run_dir.iterdir()} == _FILES
    assert not (run_dir / "benchmark.json").exists()
    for name in ("run.json", "metrics.json", "samples.jsonl", "predictions.jsonl"):
        _assert_canonical(run_dir / name)

    sample_bytes = (run_dir / "samples.jsonl").read_bytes()
    samples = _records(run_dir / "samples.jsonl")
    assert samples == [asdict(item.sample) for item in evaluation.samples]
    assert metadata.dataset_root.encode() not in sample_bytes
    assert all(
        not str(record["image_relpath"]).startswith("/")
        and "\\" not in str(record["image_relpath"])
        for record in samples
    )

    run = json.loads((run_dir / "run.json").read_bytes())
    assert (
        run["schema_version"],
        run["profile_id"],
        run["preprocessing_profile"],
        run["feature_extractor"],
        run["feature_layer"],
    ) == (
        1,
        "inspectrt_feature_memory_v1",
        "inspectrt_resize256_v1",
        "ResNet-50",
        "layer2",
    )
    assert (
        run["category"],
        run["device"],
        run["bank_chunk_size"],
        run["batch_size"],
    ) == (
        "bottle",
        metadata.requested_device,
        metadata.bank_chunk_size,
        1,
    )
    assert run["benchmark"] is None
    assert run["map_interpolation"]["mode"] == "bilinear"
    assert set(run["source"]) == {"dirty", "git_commit", "uv_lock_sha256"}
    assert set(run["environment"]) == {
        "created_at_utc",
        "dependency_versions",
        "platform_description",
        "python_version",
    }
    assert set(run["weights"]) == {"cached_file_sha256", "enum", "source_url"}
    assert run["source"]["git_commit"] == metadata.git_commit
    assert run["environment"]["dependency_versions"] == dict(
        metadata.dependency_versions
    )
    assert run["weights"]["cached_file_sha256"] == metadata.weight_file_sha256
    assert run["determinism"] == dict(metadata.determinism_flags)
    digest = hashlib.sha256(sample_bytes).hexdigest()
    assert run["inventory"]["sample_inventory_sha256"] == digest
    assert run["inventory"]["training_sample_count"] == 1
    assert run["tensors"]["memory_bank"] == {
        "byte_count": 1024 * 512 * 4,
        "dtype": "float32",
        "shape": [1024, 512],
    }

    bank = torch.load(run_dir / "memory_bank.pt", weights_only=True)
    retrieval = torch.load(run_dir / "retrieval.pt", weights_only=True)
    maps = torch.load(run_dir / "anomaly_maps.pt", weights_only=True)
    assert (bank["shape"], bank["dtype"], bank["patches_per_training_sample"]) == (
        [1024, 512],
        "float32",
        1024,
    )
    persisted = {
        "memory_bank": bank["memory_bank"],
        "patch_distances": retrieval["patch_distances"],
        "nearest_bank_indices": retrieval["nearest_bank_indices"],
        "anomaly_maps": maps["anomaly_maps"],
        "pixel_masks": maps["evaluation_masks"],
    }
    for name, tensor in persisted.items():
        source = getattr(evaluation, name)
        assert torch.equal(tensor, source)
        assert tensor.dtype == source.dtype and tensor.shape == source.shape
        assert tensor.device.type == "cpu" and tensor.is_contiguous()

    predictions = _records(run_dir / "predictions.jsonl")
    test_ids = [item.sample.sample_id for item in evaluation.test_samples]
    assert retrieval["test_sample_ids"] == maps["test_sample_ids"] == test_ids
    assert [record["sample_id"] for record in predictions] == test_ids
    assert [record["tensor_index"] for record in predictions] == [0, 1]
    assert all(
        set(record)
        == {"sample_id", "image_label", "defect_type", "image_score", "tensor_index"}
        for record in predictions
    )

    metrics = json.loads((run_dir / "metrics.json").read_bytes())
    metric_names = {"image_auroc", "image_average_precision", "pixel_auroc"}
    count_names = {
        "training_sample_count",
        "test_sample_count",
        "test_good_sample_count",
        "anomalous_test_sample_count",
        "evaluated_pixel_count",
        "anomalous_pixel_count",
    }
    assert set(metrics) == metric_names | count_names
    assert {name: metrics[name] for name in count_names} == {
        "training_sample_count": 1,
        "test_sample_count": 2,
        "test_good_sample_count": 1,
        "anomalous_test_sample_count": 1,
        "evaluated_pixel_count": 2 * 256 * 256,
        "anomalous_pixel_count": 1,
    }
    recomputed = compute_threshold_free_metrics(
        torch.tensor(
            [record["image_label"] for record in predictions], dtype=torch.uint8
        ),
        torch.tensor([record["image_score"] for record in predictions]),
        maps["evaluation_masks"],
        maps["anomaly_maps"],
    )
    assert [getattr(recomputed, name) for name in metric_names] == [
        metrics[name] for name in metric_names
    ]
    assert evaluation.samples == tuple(evaluation.samples)
    assert all(
        torch.equal(getattr(evaluation, name), value)
        for name, value in originals.items()
    )


def test_persists_installed_distribution_provenance_as_schema_two(
    tmp_path: Path,
) -> None:
    from inspectrt.portability import BundleValidationError, load_comparable_bundle

    profile_digest = "8df093df5eb8e35f77e0e8c088746b34fe69023f115f89fb822a5682d66cdfb6"
    metadata = _metadata(
        "installed-run",
        git_commit=None,
        git_dirty=None,
        uv_lock_sha256=None,
        source_kind="installed_distribution",
        distribution_name="inspectrt",
        distribution_version="0.1.0",
        baseline_profile_sha256=profile_digest,
    )

    run_dir = persist_baseline_run(_evaluation(), tmp_path, metadata)

    assert {path.name for path in run_dir.iterdir()} == _FILES
    _assert_canonical(run_dir / "run.json")
    run = json.loads((run_dir / "run.json").read_bytes())
    assert run["schema_version"] == 2
    assert run["source"] == {
        "baseline_profile_sha256": profile_digest,
        "distribution_name": "inspectrt",
        "distribution_version": "0.1.0",
        "kind": "installed_distribution",
    }
    assert not ({"dirty", "git_commit", "uv_lock_sha256"} & run["source"].keys())
    with pytest.raises(BundleValidationError, match="schema_version"):
        load_comparable_bundle(run_dir)
    with pytest.raises(ValueError, match="must not contain repository fields"):
        _metadata(
            source_kind="installed_distribution",
            distribution_name="inspectrt",
            distribution_version="0.1.0",
            baseline_profile_sha256=profile_digest,
        )


def test_persists_canonical_schema_two_benchmark_and_run_link(tmp_path: Path) -> None:
    metadata = _metadata(requested_device="cpu")
    benchmark = _benchmark(metadata)

    run_dir = persist_baseline_run(
        _evaluation(), tmp_path, metadata, benchmark=benchmark
    )

    assert {path.name for path in run_dir.iterdir()} == _BENCHMARK_FILES
    benchmark_bytes = (run_dir / "benchmark.json").read_bytes()
    assert benchmark_bytes == _json_bytes(benchmark.to_json_value())
    _assert_canonical(run_dir / "benchmark.json")
    persisted = json.loads(benchmark_bytes)
    assert persisted["schema_version"] == 2
    assert persisted["results"]["repeated_stages"]["image_decode"]["raw_ns"] == (
        list(range(30, 0, -1))
    )
    run = json.loads((run_dir / "run.json").read_bytes())
    assert run["benchmark"] == {
        "artifact": "benchmark.json",
        "present": True,
        "schema_version": 2,
        "timing_device": "cpu",
    }


def _assert_no_overwrite(tmp_path: Path) -> None:
    destination = tmp_path / "runs" / "existing"
    destination.mkdir(parents=True)
    marker = destination / "keep.txt"
    marker.write_text("unchanged", encoding="utf-8")
    metadata = _metadata("existing", requested_device="cpu")
    with pytest.raises(FileExistsError, match="already exists"):
        persist_baseline_run(
            _evaluation(), tmp_path, metadata, benchmark=_benchmark(metadata)
        )
    assert marker.read_text(encoding="utf-8") == "unchanged"
    assert not list(destination.parent.glob(".existing.tmp-*"))


def test_failed_write_removes_final_and_temporary_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _assert_no_overwrite(tmp_path)
    real_write_bytes = Path.write_bytes

    def fail_late(path: Path, value: bytes) -> int:
        if path.name == "run.json":
            raise OSError("late write failed")
        return real_write_bytes(path, value)

    monkeypatch.setattr(Path, "write_bytes", fail_late)
    failed_root = tmp_path / "failed-root"
    metadata = _metadata("failed", requested_device="cpu")
    with pytest.raises(OSError, match="late write failed"):
        persist_baseline_run(
            _evaluation(), failed_root, metadata, benchmark=_benchmark(metadata)
        )
    assert list((failed_root / "runs").iterdir()) == []


def test_rejects_unsafe_identity_and_invalid_scientific_values(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="one path component"):
        _metadata("../unsafe")
    assert not (tmp_path / "runs").exists()

    evaluation = _evaluation()
    distances = evaluation.patch_distances.clone()
    distances[0, 0] = torch.nan
    cases = (
        (replace(evaluation, patch_distances=distances), ValueError, "finite"),
        (
            replace(evaluation, metrics=ThresholdFreeMetrics(True, True, True)),
            TypeError,
            "Python floats",
        ),
    )
    for index, (malformed, error, message) in enumerate(cases):
        output_root = tmp_path / str(index)
        with pytest.raises(error, match=message):
            persist_baseline_run(malformed, output_root, _metadata(f"bad-{index}"))
        assert not output_root.exists()
