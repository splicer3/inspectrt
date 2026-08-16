from dataclasses import FrozenInstanceError, replace
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
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
    ComparableBundle,
    ComparisonValidationError,
    DiscreteComponentComparison,
    FloatingStatistics,
    IndexMismatch,
    MemoryBankMetadata,
    MetricDelta,
    PolicyDerivation,
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
    inventory = run["inventory"]
    assert isinstance(inventory, dict)
    inventory["sample_inventory_sha256"] = hashlib.sha256(sample_bytes).hexdigest()
    _write_json(bundle / "run.json", run)


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


def test_rejects_noncanonical_or_malformed_json(tmp_path: Path) -> None:
    scenarios = (
        "malformed",
        "duplicate-key",
        "nan",
        "noncanonical-spacing",
        "invalid-utf8",
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
        elif case == "noncanonical-spacing":
            damaged = (json.dumps(value, sort_keys=True) + "\n").encode()
        elif case == "invalid-utf8":
            damaged = b"\xff" + canonical
        elif case == "deeply-nested":
            damaged = b'{"value":' + (b"[" * 2000) + (b"]" * 2000) + b"}\n"
        path.write_bytes(damaged)

        with pytest.raises(BundleValidationError, match="metrics.json") as raised:
            load_comparable_bundle(bundle)
        assert raised.value, case

    jsonl_bundle = _bundle(tmp_path / "jsonl-spacing")
    records = _records(jsonl_bundle / "samples.jsonl")
    (jsonl_bundle / "samples.jsonl").write_bytes(
        (json.dumps(records[0], sort_keys=True) + "\n").encode()
        + b"".join(_canonical(record) for record in records[1:])
    )
    with pytest.raises(BundleValidationError, match="samples.jsonl"):
        load_comparable_bundle(jsonl_bundle)


def test_rejects_nonexact_inventory_and_symlinked_entries(tmp_path: Path) -> None:
    for case in ("missing", "extra", "non-file", "bundle-link", "artifact-link"):
        bundle = _bundle(tmp_path / case / "real")
        if case == "missing":
            (bundle / "metrics.json").unlink()
        elif case == "extra":
            (bundle / "extra.json").write_bytes(b"{}\n")
        elif case == "non-file":
            (bundle / "metrics.json").unlink()
            (bundle / "metrics.json").mkdir()
        elif case == "bundle-link":
            link = tmp_path / case / "link"
            link.symlink_to(bundle, target_is_directory=True)
            bundle = link
        else:
            target = tmp_path / case / "metrics.json"
            target.write_bytes((bundle / "metrics.json").read_bytes())
            (bundle / "metrics.json").unlink()
            (bundle / "metrics.json").symlink_to(target)

        with pytest.raises(BundleValidationError) as raised:
            load_comparable_bundle(bundle)
        assert raised.value, case


def test_rejects_unknown_missing_and_identity_fields(tmp_path: Path) -> None:
    scenarios = (
        ("run-unknown", "run.json", ("unexpected",), 1),
        ("run-missing", "run.json", ("batch_size",), _DELETE),
        ("run-profile", "run.json", ("profile_id",), "other"),
        (
            "cuda-seed",
            "run.json",
            ("determinism", "torch_cuda_seed_all"),
            None,
        ),
        ("sample-unknown", "samples.jsonl", ("unexpected",), 1),
        ("prediction-missing", "predictions.jsonl", ("defect_type",), _DELETE),
    )
    for case, filename, path, value in scenarios:
        bundle = _bundle(
            tmp_path / case, device="cuda:0" if case == "cuda-seed" else "cpu"
        )
        if filename == "run.json":
            record = _json(bundle / filename)
            _set_path(record, path, value)
            _write_json(bundle / filename, record)
        else:
            records = _records(bundle / filename)
            field = path[0]
            if value is _DELETE:
                del records[0][field]
            else:
                records[0][field] = value
            if filename == "samples.jsonl":
                _rewrite_samples(bundle, records)
            else:
                _write_records(bundle / filename, records)

        with pytest.raises(BundleValidationError) as raised:
            load_comparable_bundle(bundle)
        assert raised.value, case


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
    _assert_pickle_payload_is_not_executed(tmp_path / "unsafe-pickle")


def _assert_pickle_payload_is_not_executed(tmp_path: Path) -> None:
    global _EXECUTED_UNSAFE_PAYLOAD
    _EXECUTED_UNSAFE_PAYLOAD = False
    bundle = _bundle(tmp_path)
    torch.save(_UnsafePayload(), bundle / "memory_bank.pt")

    with pytest.raises(BundleValidationError):
        load_comparable_bundle(bundle)
    assert not _EXECUTED_UNSAFE_PAYLOAD


def test_rejects_malformed_tensor_contracts(tmp_path: Path) -> None:
    scenarios = (
        ("container", "memory_bank.pt", "memory_bank", "container"),
        ("closed-keys", "retrieval.pt", "patch_distances", "extra"),
        ("dtype", "retrieval.pt", "patch_distances", "dtype"),
        ("shape", "retrieval.pt", "nearest_bank_indices", "shape"),
        ("layout", "anomaly_maps.pt", "anomaly_maps", "layout"),
        ("finite", "memory_bank.pt", "memory_bank", "finite"),
    )
    for case, filename, field, mutation in scenarios:
        bundle = _bundle(tmp_path / case)
        path = bundle / filename
        if mutation == "container":
            torch.save([], path)
        else:
            payload = _payload(path)
            if mutation == "extra":
                payload["unexpected"] = torch.tensor(0)
            elif mutation == "dtype":
                payload[field] = payload[field].to(torch.float64)
            elif mutation == "shape":
                payload[field] = payload[field].reshape(-1)
            elif mutation == "layout":
                payload[field] = payload[field].transpose(1, 2)
                assert not payload[field].is_contiguous()
            else:
                payload[field][0].reshape(-1)[0] = float("nan")
            torch.save(payload, path)

        with pytest.raises(BundleValidationError) as raised:
            load_comparable_bundle(bundle)
        assert raised.value, case


def test_rejects_cross_artifact_identity_score_and_metric_mismatches(
    tmp_path: Path,
) -> None:
    for case in ("sample-id", "score", "metric", "count", "nearest-index", "mask"):
        bundle = _bundle(tmp_path / case)
        if case in {"sample-id", "score"}:
            records = _records(bundle / "predictions.jsonl")
            records[0]["sample_id" if case == "sample-id" else "image_score"] = (
                "mvtec_ad/bottle/test/good/001.png" if case == "sample-id" else 0.8
            )
            _write_records(bundle / "predictions.jsonl", records)
        elif case in {"metric", "count"}:
            metrics = _json(bundle / "metrics.json")
            metrics["image_auroc" if case == "metric" else "test_sample_count"] = (
                0.5 if case == "metric" else 3
            )
            _write_json(bundle / "metrics.json", metrics)
        elif case == "nearest-index":
            path = bundle / "retrieval.pt"
            payload = _payload(path)
            payload["nearest_bank_indices"][0, 0] = 1024
            torch.save(payload, path)
        else:
            path = bundle / "anomaly_maps.pt"
            payload = _payload(path)
            payload["evaluation_masks"][0, 0, 0] = 2
            torch.save(payload, path)

        with pytest.raises(BundleValidationError) as raised:
            load_comparable_bundle(bundle)
        assert raised.value, case


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


@pytest.fixture(scope="module")
def _exact_scientific(
    _a2_bundles: dict[str, ComparableBundle],
) -> ScientificComparison:
    bundle = _a2_bundles["evaluation"]
    return _comparison(bundle, bundle)


def test_scientific_comparison_accepts_seven_file_evaluation_bundles(
    _a2_bundles: dict[str, ComparableBundle],
) -> None:
    reference = _a2_bundles["evaluation"]
    second = replace(reference, metrics=replace(reference.metrics, image_auroc=0.75))
    comparison = _comparison(reference, reference, second)

    assert comparison.comparability[0].comparable
    assert comparison.scientific_results[0].status == "observed_unclassified"
    assert comparison.reference.bundle_kind == "evaluation"
    assert comparison.candidates[0].bundle_kind == "evaluation"
    assert tuple(item.environment_id for item in comparison.candidates) == (
        "candidate-1",
        "candidate-2",
    )
    assert comparison.scientific_results[1].metrics[0] == MetricDelta(  # type: ignore[index]
        "image_auroc", 1.0, 0.75, 0.25
    )
    assert tuple(name for name, _ in comparison.comparability[0].gates) == (
        _A2_GATE_NAMES
    )
    assert all(value for _, value in comparison.comparability[0].gates)
    assert all(
        component.exact
        for component in comparison.comparability[0].structural_components
    )
    assert (
        comparison.comparison_id
        == _comparison(reference, reference, second).comparison_id
    )
    changed_identity = compare_scientific_bundles(
        _descriptor(reference, "reference-env", "reference"),
        (
            _descriptor(reference, "candidate-1", "holdout"),
            _descriptor(second, "candidate-2", "holdout"),
        ),
        generator=ScientificGenerator("e" * 40, False),
    )
    assert comparison.comparison_id != changed_identity.comparison_id
    attempt = ScientificExecutionAttempt(
        "mps-attempt", "unsupported", "operator_unsupported", "evaluation"
    )
    with_attempt = _comparison(reference, reference, attempts=(attempt,))
    assert with_attempt.attempts == (attempt,)
    assert len(with_attempt.scientific_results) == 1
    with pytest.raises(ComparisonValidationError, match="unique"):
        compare_scientific_bundles(
            _descriptor(reference, "reference-env", "reference"),
            (_descriptor(reference, "reference-env", "holdout"),),
            generator=_A2_GENERATOR,
        )
    with pytest.raises(ComparisonValidationError, match="policy_role"):
        compare_scientific_bundles(
            _descriptor(reference, "reference-env", "reference"),
            (_descriptor(reference, "candidate-env", "reference"),),
            generator=_A2_GENERATOR,
        )
    with pytest.raises(ComparisonValidationError, match="hostname"):
        _descriptor(
            reference,
            "private-environment",
            "holdout",
            hardware_label="host p53.internal",
        )
    _assert_independent_scientific_gates(_a2_bundles)


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
    assert statistics.element_count == reference.memory_bank.numel()
    assert statistics.maximum_absolute_error == 1.0
    assert statistics.mean_absolute_error == pytest.approx(
        1 / reference.memory_bank.numel()
    )
    assert statistics.root_mean_square_error == pytest.approx(
        1 / (reference.memory_bank.numel() ** 0.5)
    )
    assert statistics.maximum_relative_error is None
    assert statistics.zero_reference_count == reference.memory_bank.numel()

    distances = reference.patch_distances.clone()
    distances[0, 0] = 0.3
    relative = _floating(
        _comparison(reference, replace(reference, patch_distances=distances)),
        "patch_distances",
    )
    assert relative.maximum_relative_error == pytest.approx(0.5)
    _assert_nearest_index_mismatch(_a2_bundles)
    _assert_metric_delta(_a2_bundles)
    for malformed in (
        replace(reference, memory_bank=reference.memory_bank.flatten()),
        _tensor_variant(reference, "memory_bank", (0, 0), float("nan")),
        _tensor_variant(reference, "nearest_bank_indices", (0, 0), -1),
        _tensor_variant(reference, "evaluation_masks", (0, 0, 0), 2),
    ):
        with pytest.raises(ComparisonValidationError):
            _comparison(reference, malformed)


def _assert_nearest_index_mismatch(
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


def _assert_independent_scientific_gates(
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


def _assert_metric_delta(
    _a2_bundles: dict[str, ComparableBundle],
) -> None:
    bundle = _a2_bundles["evaluation"]
    candidate = replace(bundle, metrics=replace(bundle.metrics, image_auroc=0.75))

    comparison = _comparison(bundle, candidate)
    metrics = comparison.scientific_results[0].metrics
    assert metrics is not None

    assert metrics[0] == MetricDelta("image_auroc", 1.0, 0.75, 0.25)
    assert comparison.scientific_results[0].status == "observed_unclassified"


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

    for case in ("private-label", "unknown-field", "duplicate-id"):
        invalid = _environment_map_value()
        reference = invalid["reference"]
        assert isinstance(reference, dict)
        if case == "private-label":
            reference["hardware_label"] = "host p53.internal"
        elif case == "unknown-field":
            reference["hostname"] = "private-host"
        else:
            invalid["candidates"] = [_environment_value("reference-env", "holdout")]
        invalid_path = tmp_path / f"environment-map-{case}.json"
        _write_json(invalid_path, invalid)
        with pytest.raises(ComparisonValidationError) as raised:
            load_portability_environment_map(invalid_path)
        assert raised.value, case

    invalid_attempt = _environment_map_value(
        attempts=[
            {
                "environment_id": "mps-attempt",
                "gating": False,
                "policy_role": "post_policy_attempt",
                "reason_code": "RuntimeError: /Users/alice",
                "stage_code": "evaluation",
                "status": "execution_failed",
            }
        ]
    )
    invalid_attempt_path = tmp_path / "environment-map-private-attempt.json"
    _write_json(invalid_attempt_path, invalid_attempt)
    with pytest.raises(ComparisonValidationError, match="reason_code"):
        load_portability_environment_map(invalid_attempt_path)


def test_tracked_portability_policy_is_the_reviewed_canonical_artifact(
    tmp_path: Path,
) -> None:
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

    for case in ("unknown", "missing", "negative"):
        invalid = _policy_value()
        if case == "unknown":
            invalid["unexpected"] = True
        elif case == "missing":
            del invalid["category"]
        else:
            limits = invalid["floating_component_limits"]
            assert isinstance(limits, dict)
            limits["memory_bank"]["atol"] = -1.0  # type: ignore[index]
        invalid_path = tmp_path / f"policy-{case}.json"
        _write_json(invalid_path, invalid)
        with pytest.raises(ComparisonValidationError) as raised:
            load_portability_policy(invalid_path)
        assert raised.value, case
    assert b"/" not in payload and b"\\" not in payload and b"_extra" not in payload


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

    limits = _policy_value()["floating_component_limits"]
    assert isinstance(limits, dict)
    limits["memory_bank"] = {"atol": 1.0, "rtol": 0.0}
    metric_limits = dict(_policy_value()["metric_absolute_delta_limits"])
    metric_limits["image_auroc"] = 0.25
    inclusive = _loaded_policy(
        tmp_path,
        floating_component_limits=limits,
        metric_absolute_delta_limits=metric_limits,
    )
    bank = bundle.memory_bank.clone()
    bank[0, 0] = 1.0
    candidate = replace(
        bundle,
        memory_bank=bank,
        metrics=replace(bundle.metrics, image_auroc=0.75),
    )
    assert (
        _policy_comparison(bundle, candidate, inclusive).scientific_results[0].status
        == "within_policy"
    )
    float_limits = dict(limits)
    float_limits["memory_bank"] = {"atol": 0.5, "rtol": 0.0}
    float_strict = _loaded_policy(
        tmp_path,
        floating_component_limits=float_limits,
        metric_absolute_delta_limits=metric_limits,
    )
    assert (
        _policy_comparison(bundle, candidate, float_strict).scientific_results[0].status
        == "drift_detected"
    )
    strict_metrics = dict(metric_limits)
    strict_metrics["image_auroc"] = 0.24
    metric_strict = _loaded_policy(
        tmp_path,
        floating_component_limits=limits,
        metric_absolute_delta_limits=strict_metrics,
    )
    assert (
        _policy_comparison(bundle, candidate, metric_strict)
        .scientific_results[0]
        .status
        == "drift_detected"
    )
    wrong_scope = _loaded_policy(tmp_path, reference_environment_id="other-reference")
    with pytest.raises(ComparisonValidationError, match="reference"):
        _policy_comparison(bundle, bundle, wrong_scope)
    _assert_required_discrete_mismatch(tmp_path, _a2_bundles)


def _assert_required_discrete_mismatch(
    tmp_path: Path,
    _a2_bundles: dict[str, ComparableBundle],
) -> None:
    reference = _a2_bundles["evaluation"]
    for component, index in (
        ("nearest_bank_indices", (0, 0)),
        ("evaluation_masks", (0, 0, 1)),
    ):
        candidate = _tensor_variant(reference, component, index, 1)
        comparison = _policy_comparison(reference, candidate, _loaded_policy(tmp_path))
        assert comparison.scientific_results[0].status == "drift_detected", component


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


def _publication_bytes(scientific: bytes = b'{"science":1}\n') -> tuple[bytes, bytes]:
    performance = _canonical(
        {"scientific_sha256": hashlib.sha256(scientific).hexdigest()}
    )
    return scientific, performance


def test_atomic_publication_has_exact_inventory_and_rejects_existing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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

    with pytest.raises(ComparisonValidationError, match="exact scientific"):
        publish_portability_records(
            scientific,
            _canonical({"scientific_sha256": "0" * 64}),
            tmp_path / "wrong-hash",
        )
    private = _canonical({"source_path": "/home/alice/private-run"})
    _, private_performance = _publication_bytes(private)
    with pytest.raises(ComparisonValidationError, match="private path"):
        publish_portability_records(
            private, private_performance, tmp_path / "private-path"
        )
    failed_root = tmp_path / "failed"
    failed_root.mkdir()
    failed_output = failed_root / "comparison"
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
        publish_portability_records(scientific, performance, failed_output)
    assert not failed_output.exists()
    assert list(failed_root.iterdir()) == []


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
    invalid_root = tmp_path / "invalid-timing"
    invalid_root.mkdir()
    _assert_performance_v2_rejects_mismatches(invalid_root)

    invalid_path = _write_timing_bundle(tmp_path / "invalid-loader", 0)
    benchmark = _json(invalid_path / "benchmark.json")
    _set_path(
        benchmark,
        ("results", "repeated_stages", "image_decode", "summary_ns", "p50"),
        -1.0,
    )
    _write_json(invalid_path / "benchmark.json", benchmark)
    with pytest.raises(BundleValidationError, match="raw_ns"):
        load_timing_bundle(invalid_path)


def _assert_performance_v2_rejects_mismatches(
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
    with pytest.raises(ComparisonValidationError, match="reviewed six-run matrix"):
        _build_timing_performance(inputs, (bundles[1], bundles[0], *bundles[2:]))
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
