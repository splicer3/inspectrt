from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import inspectrt.fixtures as fixtures
from inspectrt.fixtures import (
    AcceptedRunFixtureSource,
    FixtureGenerator,
    RealApplicationFixtureSource,
    RetrievalFixture,
    RetrievalFixtureMetadata,
    SyntheticFixtureSource,
    encode_retrieval_fixture,
    load_retrieval_fixture,
    prepare_accepted_run_fixture,
    publish_accepted_run_fixture,
    real_fixture_id,
    write_retrieval_fixture,
)

_COMMIT = "bc330b9070c5ca8db9cb7cfbb27617256388536b"
_FIXTURE_GENERATOR_COMMIT = "de731ee404e2ab1cd381230ec042496282413662"
_COMMITTED_FIXTURE = Path(__file__).parent / "fixtures" / "retrieval_v1"
_SAMPLE_ID = "mvtec_ad/bottle/test/broken_large/000.png"
_TENSOR_NAMES = [
    "queries",
    "memory_bank",
    "expected_squared_l2_distances",
    "expected_indices",
]
_DEPENDENCIES = {
    "inspectrt": "0.1.0",
    "numpy": "2.4.6",
    "pillow": "12.3.0",
    "scikit-learn": "1.9.0",
    "torch": "2.13.0",
    "torchvision": "0.28.0",
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
    "torch_cuda_seed_all": 0,
    "use_deterministic_algorithms": True,
}
_ARTIFACT_HASHES = {
    name: f"{index:x}" * 64
    for index, name in enumerate(
        (
            "run.json",
            "samples.jsonl",
            "memory_bank.pt",
            "retrieval.pt",
            "benchmark.json",
        ),
        1,
    )
}
_CANONICAL_QUERIES = np.array(
    (
        (0, 1, 0, 0, 0),
        (0, 0, 0, 2, 0),
        (2, 0, 0, 0, 0),
        (0, 0, 0, 0, 4),
    ),
    dtype="<f4",
)
_CANONICAL_BANK = np.array(
    (
        (0, 0, 0, 0, 0),
        (1, 0, 0, 0, 0),
        (0, 2, 0, 0, 0),
        (0, 0, 3, 0, 0),
        (0, 0, 0, 4, 0),
        (-1, 0, 0, 0, 0),
        (0, 0, 0, 0, 5),
    ),
    dtype="<f4",
)
_CANONICAL_DISTANCES = np.array((1, 4, 1, 1), dtype="<f4")
_CANONICAL_INDICES = np.array((0, 0, 1, 6), dtype="<i8")


def _generator(**changes: object) -> FixtureGenerator:
    values = {
        "milestone_id": "inspectrt_retrieval_fixture_v1",
        "schema_version": 1,
        "git_commit": "f" * 40,
        "dirty": False,
    }
    values.update(changes)
    return FixtureGenerator(**values)  # type: ignore[arg-type]


def _synthetic_fixture(**changes: object) -> RetrievalFixture:
    fixture = RetrievalFixture(
        RetrievalFixtureMetadata(
            "synthetic-correctness-v1",
            "synthetic_correctness",
            2,
            SyntheticFixtureSource("inspectrt_synthetic_correctness_v1", 2),
            _generator(),
        ),
        np.array(((0, 1, 0), (2, 0, 0)), dtype="<f4"),
        np.array(((0, 0, 0), (1, 0, 0), (0, 2, 0)), dtype="<f4"),
        np.array((1, 1), dtype="<f4"),
        np.array((0, 1), dtype="<i8"),
    )
    return replace(fixture, **changes)


def _committed_fixture() -> RetrievalFixture:
    return RetrievalFixture(
        RetrievalFixtureMetadata(
            "synthetic-correctness-v1",
            "synthetic_correctness",
            3,
            SyntheticFixtureSource("inspectrt_synthetic_correctness_v1", 3),
            FixtureGenerator(
                "inspectrt_retrieval_fixture_v1",
                1,
                _FIXTURE_GENERATOR_COMMIT,
                False,
            ),
        ),
        _CANONICAL_QUERIES,
        _CANONICAL_BANK,
        _CANONICAL_DISTANCES,
        _CANONICAL_INDICES,
    )


def _real_source(**changes: object) -> RealApplicationFixtureSource:
    values = {
        "category": "bottle",
        "sample_id": _SAMPLE_ID,
        "test_tensor_index": 0,
        "accepted_run_id": "20260715T202846302048Z-bottle-bc330b9",
        "source_commit": _COMMIT,
        "source_dirty": False,
        "inventory_sha256": "a" * 64,
        "uv_lock_sha256": "b" * 64,
        "weight_enum": "ResNet50_Weights.IMAGENET1K_V2",
        "weight_file_sha256": "c" * 64,
        "baseline_profile": "inspectrt_feature_memory_v1",
        "configuration_sha256": "d" * 64,
        "preprocessing_identity": "inspectrt_resize256_v1",
        "feature_layer": "layer2",
        "source_image_sha256": "e" * 64,
        "python_version": "3.11.15",
        "dependency_versions": _DEPENDENCIES,
        "platform_description": "Linux-6.8.0-134-generic-x86_64-with-glibc2.39",
        "requested_device": "cuda:0",
        "determinism": _DETERMINISM,
        "cuda_device_name": "Quadro T1000",
        "cuda_compute_capability": (7, 5),
        "pytorch_cuda_runtime_version": "13.0",
        "source_artifact_sha256": _ARTIFACT_HASHES,
    }
    values.update(changes)
    return RealApplicationFixtureSource(**values)  # type: ignore[arg-type]


def _real_fixture(
    source: RealApplicationFixtureSource | None = None,
) -> RetrievalFixture:
    source = source or _real_source()
    metadata = RetrievalFixtureMetadata(
        real_fixture_id(source.category, source.sample_id, source.source_commit),
        "real_application",
        16384,
        source,
        _generator(),
    )
    base = _synthetic_fixture()
    return replace(base, metadata=metadata)


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


def _record(directory: Path) -> dict[str, object]:
    return json.loads((directory / "manifest.json").read_bytes())


def _write_record(directory: Path, record: dict[str, object]) -> None:
    (directory / "manifest.json").write_bytes(_canonical(record))


def _write_case(tmp_path: Path, name: str) -> Path:
    directory = tmp_path / name
    write_retrieval_fixture(_synthetic_fixture(), directory)
    return directory


def _tiny_accepted_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> SimpleNamespace:
    import torch

    monkeypatch.setattr(fixtures, "_REAL_Q", 2)
    monkeypatch.setattr(fixtures, "_REAL_M", 3)
    monkeypatch.setattr(fixtures, "_REAL_D", 2)
    monkeypatch.setattr(fixtures, "_REAL_CHUNK_SIZE", 2)
    repository = tmp_path / "repository"
    run = repository / "run"
    dataset = repository / "dataset"
    config = repository / "configs" / "baseline.toml"
    cache = repository / "cache" / "checkpoints"
    run.mkdir(parents=True)
    config.parent.mkdir(parents=True)
    cache.mkdir(parents=True)
    image = dataset / "bottle" / "test" / "broken_large" / "000.png"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"controlled image bytes")
    config_bytes = b"controlled committed baseline\n"
    config.write_bytes(config_bytes)
    weight = cache / "resnet50-11ad3fa6.pth"
    weight.write_bytes(b"controlled cached weight")
    monkeypatch.setattr(torch.hub, "get_dir", lambda: str(cache.parent))
    monkeypatch.setattr(fixtures, "_git_blob", lambda *args: config_bytes)

    records = [
        {
            "category": "bottle",
            "defect_type": "broken_large",
            "image_relpath": "bottle/test/broken_large/000.png",
            "is_anomalous": True,
            "mask_relpath": "bottle/ground_truth/broken_large/000_mask.png",
            "sample_id": _SAMPLE_ID,
            "split": "test",
        },
        {
            "category": "bottle",
            "defect_type": "good",
            "image_relpath": "bottle/test/good/000.png",
            "is_anomalous": False,
            "mask_relpath": None,
            "sample_id": "mvtec_ad/bottle/test/good/000.png",
            "split": "test",
        },
        {
            "category": "bottle",
            "defect_type": "good",
            "image_relpath": "bottle/train/good/000.png",
            "is_anomalous": False,
            "mask_relpath": None,
            "sample_id": "mvtec_ad/bottle/train/good/000.png",
            "split": "train",
        },
    ]
    samples = b"".join(_canonical(record) for record in records)
    (run / "samples.jsonl").write_bytes(samples)
    bank = torch.tensor(((0, 0), (1, 0), (0, 2)), dtype=torch.float32)
    distances = torch.tensor(((1, 1), (2, 2)), dtype=torch.float32)
    indices = torch.tensor(((0, 1), (0, 0)), dtype=torch.int64)
    torch.save(
        {
            "dtype": "float32",
            "embedding_dimension": 2,
            "memory_bank": bank,
            "patches_per_training_sample": 2,
            "shape": [3, 2],
        },
        run / "memory_bank.pt",
    )
    torch.save(
        {
            "nearest_bank_indices": indices,
            "patch_distances": distances,
            "test_sample_ids": [
                _SAMPLE_ID,
                "mvtec_ad/bottle/test/good/000.png",
            ],
        },
        run / "retrieval.pt",
    )
    for name in ("anomaly_maps.pt", "metrics.json", "predictions.jsonl"):
        (run / name).write_bytes(b"controlled")

    lock_digest = "a" * 64
    dependency_versions = dict(_DEPENDENCIES)
    run_record = {
        "bank_chunk_size": 2,
        "batch_size": 1,
        "benchmark": {
            "artifact": "benchmark.json",
            "schema_version": 1,
            "timing_device": "cuda:0",
        },
        "category": "bottle",
        "dataset_root": "dataset",
        "determinism": dict(_DETERMINISM),
        "device": "cuda:0",
        "environment": {
            "created_at_utc": "2026-07-15T20:28:46.302048Z",
            "dependency_versions": dependency_versions,
            "platform_description": "test platform",
            "python_version": "3.11.15",
        },
        "feature_extractor": "ResNet-50",
        "feature_layer": "layer2",
        "inventory": {
            "anomalous_test_sample_count": 1,
            "sample_inventory_sha256": hashlib.sha256(samples).hexdigest(),
            "test_good_sample_count": 1,
            "test_sample_count": 2,
            "total_sample_count": 3,
            "training_sample_count": 1,
        },
        "map_interpolation": {},
        "preprocessing_profile": "inspectrt_resize256_v1",
        "profile_id": "inspectrt_feature_memory_v1",
        "retrieval_semantics": "exact top-1 squared L2",
        "run_id": "run",
        "schema_version": 1,
        "source": {
            "dirty": False,
            "git_commit": _COMMIT,
            "uv_lock_sha256": lock_digest,
        },
        "tensors": {
            "anomaly_maps": {},
            "evaluation_masks": {},
            "image_scores": {},
            "memory_bank": {
                "byte_count": 24,
                "dtype": "float32",
                "shape": [3, 2],
            },
            "nearest_bank_indices": {"dtype": "int64", "shape": [2, 2]},
            "patch_distances": {"dtype": "float32", "shape": [2, 2]},
            "test_labels": {},
        },
        "weights": {
            "cached_file_sha256": hashlib.sha256(weight.read_bytes()).hexdigest(),
            "enum": "ResNet50_Weights.IMAGENET1K_V2",
            "source_url": "https://download.pytorch.org/models/resnet50-11ad3fa6.pth",
        },
    }
    benchmark_record = {
        "benchmark_sample_id": _SAMPLE_ID,
        "category": "bottle",
        "created_at_utc": "2026-07-15T20:28:46.302048Z",
        "device": "cuda:0",
        "environment": {
            "cuda_compute_capability": [7, 5],
            "cuda_device_name": "test GPU",
            "pytorch_cuda_runtime_version": "13.0",
        },
        "methodology": {},
        "profile_id": "inspectrt_feature_memory_v1",
        "results": {},
        "run_id": "run",
        "schema_version": 1,
        "workload": {
            "D": 2,
            "M": 3,
            "Q": 2,
            "bank_bytes": 24,
            "bank_chunk_size": 2,
            "bank_shape": [3, 2],
            "batch_size": 1,
            "dtype": "float32",
            "k": 1,
            "tensor_layout": {},
            "test_sample_count": 2,
            "training_sample_count": 1,
        },
    }
    (run / "run.json").write_bytes(_canonical(run_record))
    (run / "benchmark.json").write_bytes(_canonical(benchmark_record))
    return SimpleNamespace(
        repository=repository,
        run=run,
        dataset=dataset,
        config=config,
        cache=cache,
        image=image,
        bank=bank,
        distances=distances[0],
        indices=indices[0],
        lock_digest=lock_digest,
        run_record=run_record,
        benchmark_record=benchmark_record,
        torch=torch,
    )


def _prepare_tiny(bundle: SimpleNamespace) -> AcceptedRunFixtureSource:
    return prepare_accepted_run_fixture(
        run_directory=bundle.run,
        dataset_root=bundle.dataset,
        sample_id=_SAMPLE_ID,
        config_path=bundle.config,
        repository_root=bundle.repository,
        generator_commit="f" * 40,
        generator_dirty=False,
        current_lock_sha256=bundle.lock_digest,
        torch=bundle.torch,
    )


def test_committed_fixture_has_canonical_identity_workload_and_values() -> None:
    files = {path.name: path for path in _COMMITTED_FIXTURE.iterdir()}
    assert set(files) == {"manifest.json", "tensors.bin"}
    assert sum(path.stat().st_size for path in files.values()) < 4096

    loaded = load_retrieval_fixture(_COMMITTED_FIXTURE)
    assert loaded.metadata == _committed_fixture().metadata
    assert loaded.manifest["schema_version"] == 1
    assert loaded.manifest["workload"] == {
        "D": 5,
        "M": 7,
        "Q": 4,
        "dtype": "float32",
        "k": 1,
        "layout": "C",
        "reference_chunk_size": 3,
    }
    for actual, expected in zip(
        (
            loaded.queries,
            loaded.memory_bank,
            loaded.expected_squared_l2_distances,
            loaded.expected_indices,
        ),
        (
            _CANONICAL_QUERIES,
            _CANONICAL_BANK,
            _CANONICAL_DISTANCES,
            _CANONICAL_INDICES,
        ),
        strict=True,
    ):
        np.testing.assert_array_equal(actual, expected)
        assert actual.shape == expected.shape
        assert actual.dtype == expected.dtype
        assert actual.flags.c_contiguous

    fixture_bytes = (
        files["manifest.json"].read_bytes() + files["tensors.bin"].read_bytes()
    )
    for forbidden in (
        b"mvtec",
        b"accepted_run",
        b"run_id",
        b"weight",
        b"\x89PNG\r\n\x1a\n",
        b"\xff\xd8\xff",
    ):
        assert forbidden not in fixture_bytes.lower()


def test_committed_fixture_raw_bytes_reload_independently() -> None:
    manifest = json.loads((_COMMITTED_FIXTURE / "manifest.json").read_bytes())
    payload = (_COMMITTED_FIXTURE / "tensors.bin").read_bytes()
    expected_arrays = (
        _CANONICAL_QUERIES,
        _CANONICAL_BANK,
        _CANONICAL_DISTANCES,
        _CANONICAL_INDICES,
    )
    assert [entry["offset_bytes"] for entry in manifest["tensors"]] == [
        0,
        128,
        320,
        384,
    ]
    assert len(payload) == manifest["payload"]["nbytes"] == 416
    assert hashlib.sha256(payload).hexdigest() == manifest["payload"]["sha256"]

    cursor = 0
    for entry, expected in zip(manifest["tensors"], expected_arrays, strict=True):
        offset = entry["offset_bytes"]
        end = offset + entry["nbytes"]
        assert offset % manifest["payload"]["alignment_bytes"] == 0
        assert payload[cursor:offset] == bytes(offset - cursor)
        assert entry["byte_order"] == "little"
        assert entry["layout"] == "C"
        assert entry["shape"] == list(expected.shape)
        assert entry["dtype"] == (
            "int64" if expected.dtype == np.dtype("<i8") else "float32"
        )
        assert hashlib.sha256(payload[offset:end]).hexdigest() == entry["sha256"]
        dtype = "<i8" if entry["dtype"] == "int64" else "<f4"
        raw = np.frombuffer(payload[offset:end], dtype=dtype).reshape(entry["shape"])
        assert raw.dtype == expected.dtype
        assert raw.flags.c_contiguous
        np.testing.assert_array_equal(raw, expected)
        cursor = end
    assert cursor == len(payload)


def test_committed_fixture_regenerates_and_matches_reference(tmp_path: Path) -> None:
    regenerated = tmp_path / "retrieval_v1"
    digest = write_retrieval_fixture(_committed_fixture(), regenerated)
    for name in ("manifest.json", "tensors.bin"):
        assert (regenerated / name).read_bytes() == (
            _COMMITTED_FIXTURE / name
        ).read_bytes()
    committed_manifest = (_COMMITTED_FIXTURE / "manifest.json").read_bytes()
    committed_payload = (_COMMITTED_FIXTURE / "tensors.bin").read_bytes()
    assert digest == hashlib.sha256(committed_manifest + committed_payload).hexdigest()

    import torch

    from inspectrt.retrieval import exact_top1_squared_l2

    queries = torch.from_numpy(_CANONICAL_QUERIES)
    bank = torch.from_numpy(_CANONICAL_BANK)
    distances, indices = exact_top1_squared_l2(queries, bank, bank_chunk_size=3)
    assert torch.equal(distances, torch.from_numpy(_CANONICAL_DISTANCES))
    assert torch.equal(indices, torch.from_numpy(_CANONICAL_INDICES))
    all_distances = torch.cdist(queries, bank).square()
    assert all_distances[0, 0] == all_distances[0, 2] == 1
    assert all_distances[1, 0] == all_distances[1, 4] == 4
    assert indices[0] == indices[1] == 0
    assert len(_CANONICAL_BANK) % 3 == 1
    assert indices[3] == 6


def test_deterministic_write_and_load_recovers_exact_contract(tmp_path: Path) -> None:
    fixture = _synthetic_fixture()
    first = tmp_path / "first"
    second = tmp_path / "second"

    first_digest = write_retrieval_fixture(fixture, first)
    second_digest = write_retrieval_fixture(fixture, second)

    assert {path.name for path in first.iterdir()} == {
        "manifest.json",
        "tensors.bin",
    }
    manifest = (first / "manifest.json").read_bytes()
    payload = (first / "tensors.bin").read_bytes()
    assert manifest == (second / "manifest.json").read_bytes()
    assert payload == (second / "tensors.bin").read_bytes()
    assert (
        first_digest == second_digest == hashlib.sha256(manifest + payload).hexdigest()
    )
    assert manifest.endswith(b"\n") and not manifest.endswith(b"\n\n")
    assert manifest == _canonical(json.loads(manifest))

    record = json.loads(manifest)
    assert set(record) == {
        "schema_version",
        "fixture_id",
        "fixture_class",
        "workload",
        "retrieval",
        "payload",
        "tensors",
        "source",
        "generator",
    }
    assert "fixture_digest" not in record
    assert record["workload"] == {
        "D": 3,
        "M": 3,
        "Q": 2,
        "dtype": "float32",
        "k": 1,
        "layout": "C",
        "reference_chunk_size": 2,
    }
    assert record["retrieval"] == {
        "chunk_merge": "strictly_lower_distance",
        "distance_output": "raw_squared_l2",
        "index_scope": "global_memory_bank",
        "operation": "exact_top1_squared_l2",
        "tie_rule": "lower_global_index",
    }
    entries = record["tensors"]
    assert [entry["name"] for entry in entries] == _TENSOR_NAMES
    assert [entry["offset_bytes"] for entry in entries] == [0, 64, 128, 192]
    assert all(entry["offset_bytes"] % 64 == 0 for entry in entries)
    cursor = 0
    for entry in entries:
        offset = entry["offset_bytes"]
        end = offset + entry["nbytes"]
        assert payload[cursor:offset] == bytes(offset - cursor)
        assert entry["sha256"] == hashlib.sha256(payload[offset:end]).hexdigest()
        cursor = end
    assert cursor == len(payload) == record["payload"]["nbytes"]
    assert record["payload"]["sha256"] == hashlib.sha256(payload).hexdigest()

    loaded = load_retrieval_fixture(first)
    assert loaded.metadata == fixture.metadata
    assert loaded.fixture_digest == first_digest
    for expected, actual in zip(
        (
            fixture.queries,
            fixture.memory_bank,
            fixture.expected_squared_l2_distances,
            fixture.expected_indices,
        ),
        (
            loaded.queries,
            loaded.memory_bank,
            loaded.expected_squared_l2_distances,
            loaded.expected_indices,
        ),
        strict=True,
    ):
        np.testing.assert_array_equal(actual, expected)
        assert actual.dtype == expected.dtype
        assert actual.shape == expected.shape
        assert actual.flags.c_contiguous
        assert actual.base is None


def test_validates_real_application_metadata_round_trip(tmp_path: Path) -> None:
    fixture = _real_fixture()
    directory = tmp_path / fixture.metadata.fixture_id

    write_retrieval_fixture(fixture, directory)
    loaded = load_retrieval_fixture(directory)

    assert loaded.metadata == fixture.metadata
    source = loaded.metadata.source
    assert isinstance(source, RealApplicationFixtureSource)
    assert source.cuda_compute_capability == (7, 5)
    assert source.dependency_versions == _DEPENDENCIES
    assert source.determinism == _DETERMINISM
    assert source.source_artifact_sha256 == _ARTIFACT_HASHES


def test_real_fixture_id_uses_fixed_ascii_normalization() -> None:
    commit = "a" * 40
    assert (
        real_fixture_id(
            "Bøttle",
            "mvtec_ad/Bøttle/test/Cräck_Name/Fïle.Name.png",
            commit,
        )
        == "b-ttle-cr-ck-name-f-le-name-aaaaaaaaaaaa"
    )
    for category, sample, source_commit in (
        ("***", "mvtec_ad/***/test/crack/000.png", commit),
        ("bottle", "/bottle/test/crack/000.png", commit),
        ("bottle", "mvtec_ad/bottle/test/../000.png", commit),
        ("bottle", "mvtec_ad/bottle/train/crack/000.png", commit),
        ("bottle", "mvtec_ad/other/test/crack/000.png", commit),
        ("bottle", _SAMPLE_ID, "A" * 40),
        ("a" * 80, f"mvtec_ad/{'a' * 80}/test/{'b' * 20}/000.png", commit),
    ):
        with pytest.raises(ValueError):
            real_fixture_id(category, sample, source_commit)


def test_rejects_duplicate_noncanonical_unknown_and_missing_json(
    tmp_path: Path,
) -> None:
    directory = _write_case(tmp_path, "base")
    manifest = (directory / "manifest.json").read_bytes()
    duplicate = manifest.replace(b'"D":3', b'"D":3,"D":3')
    (directory / "manifest.json").write_bytes(duplicate)
    with pytest.raises(ValueError, match="duplicate"):
        load_retrieval_fixture(directory)

    for index, payload in enumerate(
        (
            b" " + manifest,
            manifest.removesuffix(b"\n"),
            manifest.replace(b'"Q":2', b'"Q":NaN'),
        )
    ):
        case = _write_case(tmp_path, f"noncanonical-{index}")
        (case / "manifest.json").write_bytes(payload)
        with pytest.raises(ValueError):
            load_retrieval_fixture(case)

    for index, change in enumerate(("unknown", "missing", "nested", "type")):
        case = _write_case(tmp_path, f"fields-{index}")
        record = _record(case)
        if change == "unknown":
            record["unknown"] = 1
        elif change == "missing":
            record.pop("source")
        elif change == "nested":
            record["workload"]["unknown"] = 1
        else:
            record["source"] = []
        _write_record(case, record)
        with pytest.raises((TypeError, ValueError)):
            load_retrieval_fixture(case)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    (
        ("schema_version", 2, ValueError),
        ("schema_version", True, TypeError),
        ("fixture_class", "other", ValueError),
    ),
)
def test_rejects_unsupported_schema_and_class(
    tmp_path: Path, field: str, value: object, error: type[Exception]
) -> None:
    directory = _write_case(tmp_path, field)
    record = _record(directory)
    record[field] = value
    _write_record(directory, record)
    with pytest.raises(error):
        load_retrieval_fixture(directory)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    (
        ("Q", True, TypeError),
        ("Q", 0, ValueError),
        ("M", -1, ValueError),
        ("D", 0, ValueError),
        ("k", 2, ValueError),
        ("reference_chunk_size", 0, ValueError),
    ),
)
def test_rejects_invalid_workload_integers(
    tmp_path: Path, field: str, value: object, error: type[Exception]
) -> None:
    directory = _write_case(tmp_path, f"workload-{field}")
    record = _record(directory)
    record["workload"][field] = value
    _write_record(directory, record)
    with pytest.raises(error):
        load_retrieval_fixture(directory)


@pytest.mark.parametrize("value", ("/tensors.bin", "dir/tensors.bin", "a\\b"))
def test_rejects_noncanonical_payload_file(tmp_path: Path, value: str) -> None:
    directory = _write_case(tmp_path, value.replace("/", "-").replace("\\", "-"))
    record = _record(directory)
    record["payload"]["file"] = value
    _write_record(directory, record)
    with pytest.raises(ValueError):
        load_retrieval_fixture(directory)


def test_rejects_invalid_synthetic_and_generator_metadata() -> None:
    fixture = _synthetic_fixture()
    metadata_cases = (
        replace(fixture.metadata, fixture_id="wrong"),
        replace(fixture.metadata, fixture_class="other"),
        replace(fixture.metadata, reference_chunk_size=0),
        replace(
            fixture.metadata,
            source=SyntheticFixtureSource("wrong", 2),
        ),
        replace(
            fixture.metadata,
            source=SyntheticFixtureSource("inspectrt_synthetic_correctness_v1", 3),
        ),
        replace(fixture.metadata, generator=_generator(milestone_id="wrong")),
        replace(fixture.metadata, generator=_generator(schema_version=2)),
        replace(fixture.metadata, generator=_generator(git_commit="a" * 39)),
        replace(fixture.metadata, generator=_generator(dirty=True)),
    )
    for metadata in metadata_cases:
        with pytest.raises((TypeError, ValueError)):
            encode_retrieval_fixture(replace(fixture, metadata=metadata))


def test_rejects_invalid_real_source_and_unsafe_paths() -> None:
    source_cases = (
        _real_source(source_dirty=True),
        _real_source(inventory_sha256="A" * 64),
        _real_source(baseline_profile="other"),
        _real_source(sample_id="/absolute/test/crack/000.png"),
        _real_source(sample_id="mvtec_ad/bottle/test/../000.png"),
        _real_source(accepted_run_id="../run"),
        _real_source(test_tensor_index=True),
        _real_source(dependency_versions={**_DEPENDENCIES, "numpy": "2.4\n6"}),
        _real_source(dependency_versions={"numpy": "2.4.6"}),
        _real_source(determinism={**_DETERMINISM, "numpy_seed": True}),
        _real_source(cuda_compute_capability=(7, True)),
        _real_source(source_artifact_sha256={"run.json": "a" * 64}),
        _real_source(source_artifact_sha256={**_ARTIFACT_HASHES, "run.json": "z" * 64}),
    )
    for source in source_cases:
        with pytest.raises((TypeError, ValueError)):
            encode_retrieval_fixture(_real_fixture(source))

    valid = _real_fixture()
    wrong_id = replace(
        valid,
        metadata=replace(valid.metadata, fixture_id="bottle-crack-000-aaaaaaaaaaaa"),
    )
    with pytest.raises(ValueError, match="fixture_id"):
        encode_retrieval_fixture(wrong_id)


def test_rejects_invalid_arrays_without_casting_or_reordering() -> None:
    fixture = _synthetic_fixture()
    noncontiguous = np.zeros((2, 6), dtype="<f4")[:, ::2]
    cases = (
        replace(fixture, queries=fixture.queries.astype("<f8")),
        replace(fixture, memory_bank=fixture.memory_bank.astype(">f4")),
        replace(fixture, expected_indices=fixture.expected_indices.astype("<i4")),
        replace(fixture, queries=np.zeros(6, dtype="<f4")),
        replace(fixture, memory_bank=np.zeros((3, 4), dtype="<f4")),
        replace(fixture, expected_squared_l2_distances=np.zeros(3, dtype="<f4")),
        replace(fixture, expected_indices=np.zeros(3, dtype="<i8")),
        replace(fixture, queries=noncontiguous),
        replace(fixture, queries=np.full((2, 3), np.nan, dtype="<f4")),
        replace(fixture, memory_bank=np.full((3, 3), np.inf, dtype="<f4")),
        replace(
            fixture,
            expected_squared_l2_distances=np.array((1, np.nan), dtype="<f4"),
        ),
        replace(
            fixture,
            expected_squared_l2_distances=np.array((1, -1), dtype="<f4"),
        ),
        replace(fixture, expected_indices=np.array((-1, 1), dtype="<i8")),
        replace(fixture, expected_indices=np.array((0, 3), dtype="<i8")),
        replace(fixture, queries=np.zeros((0, 3), dtype="<f4")),
        replace(fixture, memory_bank=np.zeros((0, 3), dtype="<f4")),
        replace(
            fixture,
            queries=np.zeros((2, 0), dtype="<f4"),
            memory_bank=np.zeros((3, 0), dtype="<f4"),
        ),
    )
    for malformed in cases:
        with pytest.raises((TypeError, ValueError)):
            encode_retrieval_fixture(malformed)


def test_rejects_malformed_tensor_segment_metadata(tmp_path: Path) -> None:
    def change(case: str, record: dict[str, object], payload: bytearray) -> None:
        entries = record["tensors"]
        if case == "unaligned":
            entries[1]["offset_bytes"] += 1
        elif case == "reordered":
            entries[0], entries[1] = entries[1], entries[0]
        elif case == "overlap":
            entries[1]["offset_bytes"] = entries[0]["offset_bytes"]
        elif case == "out-of-bounds":
            entries[-1]["offset_bytes"] = len(payload) + 64
        elif case == "nbytes":
            entries[0]["nbytes"] += 4
        elif case == "shape":
            entries[0]["shape"] = [2, 4]
        elif case == "boolean-shape":
            entries[0]["shape"] = [True, 3]
        elif case == "dtype":
            entries[0]["dtype"] = "float64"
        elif case == "byte-order":
            entries[0]["byte_order"] = "big"
        elif case == "layout":
            entries[0]["layout"] = "F"
        elif case == "duplicate-name":
            entries[1]["name"] = entries[0]["name"]
        elif case == "hash":
            entries[0]["sha256"] = "0" * 64

    cases = (
        "unaligned",
        "reordered",
        "overlap",
        "out-of-bounds",
        "nbytes",
        "shape",
        "boolean-shape",
        "dtype",
        "byte-order",
        "layout",
        "duplicate-name",
        "hash",
    )
    for case in cases:
        directory = _write_case(tmp_path, case)
        record = _record(directory)
        payload = bytearray((directory / "tensors.bin").read_bytes())
        change(case, record, payload)
        _write_record(directory, record)
        with pytest.raises((TypeError, ValueError)):
            load_retrieval_fixture(directory)


def test_rejects_payload_corruption_padding_lengths_and_hashes(
    tmp_path: Path,
) -> None:
    cases = (
        "nonzero-padding",
        "segment-corruption",
        "payload-corruption",
        "truncation",
        "trailing",
        "payload-length",
        "payload-hash",
    )
    for case in cases:
        directory = _write_case(tmp_path, case)
        record = _record(directory)
        payload = bytearray((directory / "tensors.bin").read_bytes())
        if case == "nonzero-padding":
            payload[24] = 1
            record["payload"]["sha256"] = hashlib.sha256(payload).hexdigest()
        elif case == "segment-corruption":
            payload[0] ^= 1
            record["payload"]["sha256"] = hashlib.sha256(payload).hexdigest()
        elif case == "payload-corruption":
            payload[0] ^= 1
        elif case == "truncation":
            payload.pop()
        elif case == "trailing":
            payload.extend(b"\0")
            record["payload"]["nbytes"] = len(payload)
            record["payload"]["sha256"] = hashlib.sha256(payload).hexdigest()
        elif case == "payload-length":
            record["payload"]["nbytes"] += 1
        elif case == "payload-hash":
            record["payload"]["sha256"] = "0" * 64
        _write_record(directory, record)
        (directory / "tensors.bin").write_bytes(payload)
        with pytest.raises(ValueError):
            load_retrieval_fixture(directory)


def test_requires_exactly_two_regular_files_and_never_overwrites(
    tmp_path: Path,
) -> None:
    fixture = _synthetic_fixture()
    existing = tmp_path / "existing"
    existing.mkdir()
    marker = existing / "keep"
    marker.write_text("unchanged", encoding="utf-8")
    with pytest.raises(FileExistsError):
        write_retrieval_fixture(fixture, existing)
    assert marker.read_text(encoding="utf-8") == "unchanged"

    extra = _write_case(tmp_path, "extra")
    (extra / "extra.bin").write_bytes(b"")
    with pytest.raises(ValueError, match="only"):
        load_retrieval_fixture(extra)

    missing = _write_case(tmp_path, "missing")
    (missing / "manifest.json").unlink()
    with pytest.raises(ValueError, match="only"):
        load_retrieval_fixture(missing)

    symlink = _write_case(tmp_path, "symlink")
    (symlink / "manifest.json").unlink()
    (symlink / "manifest.json").symlink_to(tmp_path / "missing-target")
    with pytest.raises(ValueError, match="regular"):
        load_retrieval_fixture(symlink)


def test_valid_tiny_benchmark_bundle_is_accepted_as_export_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _tiny_accepted_bundle(tmp_path, monkeypatch)
    source = _prepare_tiny(bundle)
    assert source.metadata.fixture_id == "bottle-broken-large-000-bc330b9070c5"
    assert source.image_path == bundle.image
    assert source.memory_bank is bundle.bank or bundle.torch.equal(
        source.memory_bank, bundle.bank
    )
    assert bundle.torch.equal(source.expected_squared_l2_distances, bundle.distances)
    assert bundle.torch.equal(source.expected_indices, bundle.indices)
    assert set(source.source_hashes) == {
        "run.json",
        "samples.jsonl",
        "memory_bank.pt",
        "retrieval.pt",
        "benchmark.json",
    }
    real_source = source.metadata.source
    assert isinstance(real_source, RealApplicationFixtureSource)
    assert (
        real_source.source_image_sha256
        == hashlib.sha256(bundle.image.read_bytes()).hexdigest()
    )
    assert (
        real_source.configuration_sha256
        == hashlib.sha256(bundle.config.read_bytes()).hexdigest()
    )


@pytest.mark.parametrize("case", ("evaluation-only", "missing", "unexpected"))
def test_rejects_incomplete_or_unexpected_source_file_sets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    bundle = _tiny_accepted_bundle(tmp_path, monkeypatch)
    if case == "evaluation-only":
        (bundle.run / "benchmark.json").unlink()
    elif case == "missing":
        (bundle.run / "retrieval.pt").unlink()
    else:
        (bundle.run / "extra").write_bytes(b"unexpected")
    with pytest.raises(ValueError, match="source run files"):
        _prepare_tiny(bundle)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("dirty", True, "clean"),
        ("lock", "b" * 64, "uv.lock"),
        ("profile", "other", "frozen"),
        ("preprocessing", "other", "frozen"),
        ("weight", "ResNet50_Weights.DEFAULT", "weight"),
    ),
)
def test_rejects_invalid_accepted_source_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
    message: str,
) -> None:
    bundle = _tiny_accepted_bundle(tmp_path, monkeypatch)
    run = bundle.run_record
    if field == "dirty":
        run["source"]["dirty"] = value
    elif field == "lock":
        run["source"]["uv_lock_sha256"] = value
    elif field == "profile":
        run["profile_id"] = value
    elif field == "preprocessing":
        run["preprocessing_profile"] = value
    else:
        run["weights"]["enum"] = value
    (bundle.run / "run.json").write_bytes(_canonical(run))
    with pytest.raises(ValueError, match=message):
        _prepare_tiny(bundle)


def test_rejects_inventory_weight_and_configuration_digest_mismatches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _tiny_accepted_bundle(tmp_path, monkeypatch)
    bundle.run_record["inventory"]["sample_inventory_sha256"] = "0" * 64
    (bundle.run / "run.json").write_bytes(_canonical(bundle.run_record))
    with pytest.raises(ValueError, match="inventory"):
        _prepare_tiny(bundle)

    bundle = _tiny_accepted_bundle(tmp_path / "weight", monkeypatch)
    (bundle.cache / "resnet50-11ad3fa6.pth").write_bytes(b"changed")
    with pytest.raises(ValueError, match="weight SHA-256"):
        _prepare_tiny(bundle)

    bundle = _tiny_accepted_bundle(tmp_path / "config", monkeypatch)
    monkeypatch.setattr(fixtures, "_git_blob", lambda *args: b"different")
    with pytest.raises(ValueError, match="configuration"):
        _prepare_tiny(bundle)


def test_requested_sample_must_match_benchmark_and_one_retrieval_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _tiny_accepted_bundle(tmp_path, monkeypatch)
    bundle.benchmark_record["benchmark_sample_id"] = "mvtec_ad/bottle/test/good/000.png"
    (bundle.run / "benchmark.json").write_bytes(_canonical(bundle.benchmark_record))
    with pytest.raises(ValueError, match="benchmark sample"):
        _prepare_tiny(bundle)

    bundle = _tiny_accepted_bundle(tmp_path / "duplicate", monkeypatch)
    payload = bundle.torch.load(
        bundle.run / "retrieval.pt", map_location="cpu", weights_only=True
    )
    payload["test_sample_ids"][1] = _SAMPLE_ID
    bundle.torch.save(payload, bundle.run / "retrieval.pt")
    with pytest.raises(ValueError, match="exactly once"):
        _prepare_tiny(bundle)


@pytest.mark.parametrize("case", ("bank-shape", "bank-dtype", "bank-layout"))
def test_rejects_invalid_bank_tensor_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
) -> None:
    bundle = _tiny_accepted_bundle(tmp_path, monkeypatch)
    payload = bundle.torch.load(
        bundle.run / "memory_bank.pt", map_location="cpu", weights_only=True
    )
    if case == "bank-shape":
        payload["memory_bank"] = bundle.torch.zeros((2, 2))
    elif case == "bank-dtype":
        payload["memory_bank"] = bundle.bank.to(bundle.torch.float64)
    else:
        payload["memory_bank"] = bundle.torch.zeros((2, 3)).T
    bundle.torch.save(payload, bundle.run / "memory_bank.pt")
    with pytest.raises((TypeError, ValueError)):
        _prepare_tiny(bundle)


@pytest.mark.parametrize(
    "case", ("distance-shape", "distance-dtype", "index-dtype", "index-range")
)
def test_rejects_invalid_retrieval_tensor_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
) -> None:
    bundle = _tiny_accepted_bundle(tmp_path, monkeypatch)
    payload = bundle.torch.load(
        bundle.run / "retrieval.pt", map_location="cpu", weights_only=True
    )
    if case == "distance-shape":
        payload["patch_distances"] = bundle.torch.zeros((2, 3))
    elif case == "distance-dtype":
        payload["patch_distances"] = payload["patch_distances"].double()
    elif case == "index-dtype":
        payload["nearest_bank_indices"] = payload["nearest_bank_indices"].int()
    else:
        payload["nearest_bank_indices"][0, 0] = 3
    bundle.torch.save(payload, bundle.run / "retrieval.pt")
    with pytest.raises((TypeError, ValueError)):
        _prepare_tiny(bundle)


def test_controlled_query_reconstruction_uses_existing_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import torch

    import inspectrt.features as feature_module
    import inspectrt.preprocessing as preprocessing

    closed: list[bool] = []
    decoded = SimpleNamespace(image=SimpleNamespace(close=lambda: closed.append(True)))
    image = torch.zeros((3, 256, 256), dtype=torch.float32)
    expected = torch.tensor(((1, 2), (3, 4)), dtype=torch.float32)
    monkeypatch.setattr(preprocessing, "decode_image", lambda path: decoded)
    monkeypatch.setattr(preprocessing, "preprocess_decoded_image", lambda value: image)
    monkeypatch.setattr(
        feature_module,
        "extract_patch_embeddings",
        lambda extractor, batch: expected.unsqueeze(0),
    )
    actual = fixtures.reconstruct_fixture_query(
        tmp_path / "image.png", object(), torch.device("cpu")
    )
    assert torch.equal(actual, expected)
    assert actual.shape == (2, 2)
    assert actual.is_contiguous()
    assert closed == [True]


def test_exact_parity_is_required_before_atomic_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _tiny_accepted_bundle(tmp_path, monkeypatch)
    source = _prepare_tiny(bundle)
    queries = bundle.torch.tensor(((0, 1), (2, 0)), dtype=bundle.torch.float32)
    with pytest.raises(ValueError, match="index mismatch"):
        publish_accepted_run_fixture(
            source,
            queries,
            bundle.distances,
            bundle.torch.tensor((1, 1), dtype=bundle.torch.int64),
            tmp_path / "bad-index",
        )
    with pytest.raises(ValueError, match="distance mismatch"):
        publish_accepted_run_fixture(
            source,
            queries,
            bundle.torch.tensor((2, 1), dtype=bundle.torch.float32),
            bundle.indices,
            tmp_path / "bad-distance",
        )
    assert not (tmp_path / "bad-index").exists()
    assert not (tmp_path / "bad-distance").exists()


def test_real_export_is_deterministic_two_file_and_preserves_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _tiny_accepted_bundle(tmp_path, monkeypatch)
    source = _prepare_tiny(bundle)
    before = {path.name: path.read_bytes() for path in bundle.run.iterdir()}
    queries = bundle.torch.tensor(((0, 1), (2, 0)), dtype=bundle.torch.float32)
    first, first_loaded = publish_accepted_run_fixture(
        source, queries, bundle.distances, bundle.indices, tmp_path / "first"
    )
    second, second_loaded = publish_accepted_run_fixture(
        source, queries, bundle.distances, bundle.indices, tmp_path / "second"
    )
    assert {path.name for path in first.iterdir()} == {"manifest.json", "tensors.bin"}
    for name in ("manifest.json", "tensors.bin"):
        assert (first / name).read_bytes() == (second / name).read_bytes()
    assert first_loaded.fixture_digest == second_loaded.fixture_digest
    assert {path.name: path.read_bytes() for path in bundle.run.iterdir()} == before
    assert first_loaded.memory_bank.shape == (3, 2)


def test_existing_destination_and_late_failure_leave_no_partial_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import inspectrt.artifacts as artifacts

    bundle = _tiny_accepted_bundle(tmp_path, monkeypatch)
    source = _prepare_tiny(bundle)
    queries = bundle.torch.tensor(((0, 1), (2, 0)), dtype=bundle.torch.float32)
    parent = tmp_path / "outputs" / "fixtures" / "inspectrt_retrieval_fixture_v1"
    destination = parent / source.metadata.fixture_id
    destination.mkdir(parents=True)
    marker = destination / "keep"
    marker.write_bytes(b"unchanged")
    with pytest.raises(FileExistsError):
        publish_accepted_run_fixture(
            source, queries, bundle.distances, bundle.indices, tmp_path / "outputs"
        )
    assert marker.read_bytes() == b"unchanged"

    marker.unlink()
    destination.rmdir()
    monkeypatch.setattr(
        artifacts,
        "_rename_without_overwrite",
        lambda *args: (_ for _ in ()).throw(OSError("late failure")),
    )
    with pytest.raises(OSError, match="late failure"):
        publish_accepted_run_fixture(
            source, queries, bundle.distances, bundle.indices, tmp_path / "outputs"
        )
    assert not destination.exists()
    assert not list(parent.glob(f".{source.metadata.fixture_id}.tmp-*"))


def test_source_change_during_export_prevents_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _tiny_accepted_bundle(tmp_path, monkeypatch)
    source = _prepare_tiny(bundle)
    queries = bundle.torch.tensor(((0, 1), (2, 0)), dtype=bundle.torch.float32)
    original = fixtures._source_hashes

    def changed(directory: Path) -> dict[str, str]:
        result = original(directory)
        result["run.json"] = "0" * 64
        return result

    monkeypatch.setattr(fixtures, "_source_hashes", changed)
    with pytest.raises(ValueError, match="changed during"):
        publish_accepted_run_fixture(
            source, queries, bundle.distances, bundle.indices, tmp_path / "outputs"
        )
    parent = tmp_path / "outputs" / "fixtures" / "inspectrt_retrieval_fixture_v1"
    assert not (parent / source.metadata.fixture_id).exists()
    assert not list(parent.glob(".*.tmp-*"))


def test_raw_reload_and_structural_environment_comparison_need_no_torch_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _tiny_accepted_bundle(tmp_path, monkeypatch)
    source = _prepare_tiny(bundle)
    queries = bundle.torch.tensor(((0, 1), (2, 0)), dtype=bundle.torch.float32)
    destination, _ = publish_accepted_run_fixture(
        source, queries, bundle.distances, bundle.indices, tmp_path / "outputs"
    )
    monkeypatch.setattr(
        bundle.torch,
        "load",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("torch.load")),
    )
    loaded = load_retrieval_fixture(destination)
    assert loaded.metadata.fixture_class == "real_application"
    real_source = loaded.metadata.source
    assert isinstance(real_source, RealApplicationFixtureSource)
    assert fixtures.basic_environment_mismatches(
        real_source,
        requested_device="cpu",
        current_lock_sha256=real_source.uv_lock_sha256,
        python_version=real_source.python_version,
        dependency_versions=real_source.dependency_versions,
        platform_description=real_source.platform_description,
    ) == ["requested_device"]
