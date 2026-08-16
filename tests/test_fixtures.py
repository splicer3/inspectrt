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
    canonical_retrieval_workload_matrix,
    counter_fp32_block,
    counter_fp32_value,
    encode_retrieval_workload_matrix,
    load_retrieval_workload_matrix,
    load_retrieval_fixture,
    prepare_accepted_run_fixture,
    publish_accepted_run_fixture,
    write_retrieval_fixture,
)

_COMMIT = "bc330b9070c5ca8db9cb7cfbb27617256388536b"
_FIXTURE_GENERATOR_COMMIT = "de731ee404e2ab1cd381230ec042496282413662"
_REPOSITORY_ROOT = Path(__file__).parents[1]
_COMMITTED_FIXTURE = Path(__file__).parent / "fixtures" / "retrieval_v1"
_WORKLOAD_MATRIX = _REPOSITORY_ROOT / "configs" / "retrieval_workloads.json"
_REAL_FIXTURE = (
    _REPOSITORY_ROOT
    / "outputs"
    / "fixtures"
    / "inspectrt_retrieval_fixture_v1"
    / "bottle-broken-large-000-bc330b9070c5"
)
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


def test_committed_workload_matrix_is_canonical_and_regenerates(
    tmp_path: Path,
) -> None:
    assert _WORKLOAD_MATRIX.is_file()
    committed = _WORKLOAD_MATRIX.read_bytes()
    matrix = load_retrieval_workload_matrix(_WORKLOAD_MATRIX)
    regenerated = encode_retrieval_workload_matrix(
        canonical_retrieval_workload_matrix()
    )
    temporary = tmp_path / "retrieval_workloads.json"
    temporary.write_bytes(regenerated)

    assert committed == regenerated == encode_retrieval_workload_matrix(matrix)
    assert temporary.read_bytes() == committed
    assert hashlib.sha256(committed).hexdigest() == (
        "7c27040bce5285172fa3febc49c7eb4185a3957681f18f86f11de1ea5d33340a"
    )
    assert committed.endswith(b"\n") and not committed.endswith(b"\n\n")
    assert b"\r" not in committed
    assert committed == _canonical(json.loads(committed))
    assert matrix.schema_version == 1
    assert matrix.matrix_id == "inspectrt-retrieval-workloads-v1"
    assert matrix.milestone == "inspectrt_retrieval_fixture_v1"
    assert set(json.loads(committed)) == {
        "schema_version",
        "matrix_id",
        "milestone",
        "retrieval_contract",
        "synthetic_generator",
        "workloads",
        "scaling_axes",
    }
    with pytest.raises(ValueError):
        encode_retrieval_workload_matrix(replace(matrix, schema_version=True))
    for case, mutation in (
        ("identity", ("matrix_id", "Unsafe_ID")),
        ("workload", ("workloads", 1, "M", 0)),
        ("scaling", ("scaling_axes", 0, "values", [])),
    ):
        record = json.loads(committed)
        if mutation[0] == "matrix_id":
            record[mutation[0]] = mutation[1]
        else:
            record[mutation[0]][mutation[1]][mutation[2]] = mutation[3]
        malformed = tmp_path / f"{case}.json"
        malformed.write_bytes(_canonical(record))
        with pytest.raises((TypeError, ValueError)) as raised:
            load_retrieval_workload_matrix(malformed)
        assert raised.value, case
    noncanonical = tmp_path / "noncanonical.json"
    noncanonical.write_bytes(b" " + committed)
    with pytest.raises(ValueError):
        load_retrieval_workload_matrix(noncanonical)


def test_ignored_real_fixture_manifest_matches_bottle_metadata_when_present() -> None:
    manifest_path = _REAL_FIXTURE / "manifest.json"
    if not manifest_path.exists():
        pytest.skip("accepted ignored real fixture is not present")
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    bottle = load_retrieval_workload_matrix(_WORKLOAD_MATRIX).workloads[3]
    assert hashlib.sha256(manifest_bytes).hexdigest() == (
        "631cebdd895dd15ff510c900f10d73e446c7f974f15d351f4417e39c64f0336c"
    )
    assert manifest["fixture_id"] == bottle["fixture_id"]
    assert tuple(manifest["workload"][name] for name in ("Q", "M", "D", "k")) == (
        bottle["Q"],
        bottle["M"],
        bottle["D"],
        bottle["k"],
    )
    bank = next(item for item in manifest["tensors"] if item["name"] == "memory_bank")
    assert bank["nbytes"] == bottle["bank_bytes"]
    with (_REAL_FIXTURE / "tensors.bin").open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == (
            "99cf70cd3a1cfd4555fb3d705b34916b6f6608c332ca0664921db3e0db8c70b1"
        )


def test_counter_fp32_small_block_is_bounded_c_order_and_rng_independent() -> None:
    np.random.seed(123)
    before = np.random.get_state()
    block = counter_fp32_block("query", 0, 2, 0, 3)
    after = np.random.get_state()
    expected_bits = np.array(
        (
            (0xC1EBFFDE, 0xC1DBFFE0, 0xC1CBFFE2),
            (0xC1E7FFE0, 0xC1D7FFE2, 0xC1C7FFE4),
        ),
        dtype="<u4",
    )
    np.testing.assert_array_equal(block.view("<u4"), expected_bits)
    assert block.shape == (2, 3)
    assert block.dtype == np.dtype("<f4")
    assert block.flags.c_contiguous
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]
    assert counter_fp32_value("query", 0, 0) != counter_fp32_value("bank", 0, 0)
    assert 131071 * 4096 + 524287 * 512 + 17 == 805301777 < 2**64
    assert 104729 * 250880 + 130363 * 512 + 31 == 26341157407 < 2**64
    for call in (
        lambda: counter_fp32_value("other", 0, 0),
        lambda: counter_fp32_value("query", 4096, 0),
        lambda: counter_fp32_block("query", 0, 65, 0, 64),
    ):
        with pytest.raises((TypeError, ValueError)):
            call()


def test_committed_fixture_regenerates_and_matches_reference(tmp_path: Path) -> None:
    regenerated = tmp_path / "retrieval_v1"
    digest = write_retrieval_fixture(_committed_fixture(), regenerated)
    for name in ("manifest.json", "tensors.bin"):
        assert (regenerated / name).read_bytes() == (
            _COMMITTED_FIXTURE / name
        ).read_bytes()
    committed_manifest = (_COMMITTED_FIXTURE / "manifest.json").read_bytes()
    committed_payload = (_COMMITTED_FIXTURE / "tensors.bin").read_bytes()
    assert hashlib.sha256(committed_manifest).hexdigest() == (
        "9e72d4238ee0cae7f8236a82e50acf8f811c0e3f7b5e2815a11c56a9e1193c12"
    )
    assert hashlib.sha256(committed_payload).hexdigest() == (
        "18c2c4333a060ff25b7304dd396cf4b292617c4593d7cbfc2576b406ed5a14bb"
    )
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

    existing = tmp_path / "existing"
    existing.mkdir()
    marker = existing / "keep"
    marker.write_bytes(b"unchanged")
    with pytest.raises(FileExistsError):
        write_retrieval_fixture(fixture, existing)
    assert marker.read_bytes() == b"unchanged"


def test_rejects_malformed_manifest_inventory_and_payload(
    tmp_path: Path,
) -> None:
    cases = (
        "manifest-unknown",
        "schema",
        "workload",
        "source",
        "tensor-descriptor",
        "extra-file",
        "symlink",
        "nonzero-padding",
        "segment-corruption",
        "truncation",
        "payload-hash",
    )
    for case in cases:
        directory = _write_case(tmp_path, case)
        record = _record(directory)
        payload = bytearray((directory / "tensors.bin").read_bytes())
        if case == "manifest-unknown":
            record["unexpected"] = True
        elif case == "schema":
            record["schema_version"] = 2
        elif case == "workload":
            record["workload"]["Q"] = 0
        elif case == "source":
            record["source"]["unexpected"] = True
        elif case == "tensor-descriptor":
            record["tensors"][0]["dtype"] = "float64"
        elif case == "extra-file":
            (directory / "extra.bin").write_bytes(b"")
        elif case == "symlink":
            (directory / "manifest.json").unlink()
            (directory / "manifest.json").symlink_to(tmp_path / "missing-target")
        elif case == "nonzero-padding":
            payload[24] = 1
            record["payload"]["sha256"] = hashlib.sha256(payload).hexdigest()
        elif case == "segment-corruption":
            payload[0] ^= 1
            record["payload"]["sha256"] = hashlib.sha256(payload).hexdigest()
        elif case == "truncation":
            payload.pop()
        elif case == "payload-hash":
            record["payload"]["sha256"] = "0" * 64
        if case != "symlink":
            _write_record(directory, record)
        (directory / "tensors.bin").write_bytes(payload)
        with pytest.raises(ValueError) as raised:
            load_retrieval_fixture(directory)
        assert raised.value, case

    malformed = replace(
        _synthetic_fixture(),
        queries=_synthetic_fixture().queries.astype("<f8"),
    )
    with pytest.raises((TypeError, ValueError)):
        write_retrieval_fixture(malformed, tmp_path / "bad-array")


def test_valid_tiny_benchmark_bundle_is_accepted_as_export_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _tiny_accepted_bundle(tmp_path, monkeypatch)
    source = _prepare_tiny(bundle)
    assert source.metadata.fixture_id == "bottle-broken-large-000-bc330b9070c5"
    with pytest.raises(ValueError):
        fixtures.real_fixture_id("bottle", "../private.png", "a" * 40)
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
    assert (
        fixtures.basic_environment_mismatches(
            real_source,
            requested_device=real_source.requested_device,
            current_lock_sha256=real_source.uv_lock_sha256,
            python_version=real_source.python_version,
            dependency_versions=real_source.dependency_versions,
            platform_description=real_source.platform_description,
        )
        == []
    )
    assert fixtures.basic_environment_mismatches(
        real_source,
        requested_device="cpu",
        current_lock_sha256=real_source.uv_lock_sha256,
        python_version=real_source.python_version,
        dependency_versions=real_source.dependency_versions,
        platform_description=real_source.platform_description,
    ) == ["requested_device"]
    cuda = SimpleNamespace(
        get_device_name=lambda index: real_source.cuda_device_name,
        get_device_capability=lambda index: real_source.cuda_compute_capability,
    )
    torch = SimpleNamespace(
        cuda=cuda,
        version=SimpleNamespace(cuda=real_source.pytorch_cuda_runtime_version),
    )
    assert (
        fixtures.cuda_environment_mismatches(
            real_source, SimpleNamespace(index=0), torch
        )
        == []
    )
    cuda.get_device_name = lambda index: "different GPU"
    assert fixtures.cuda_environment_mismatches(
        real_source, SimpleNamespace(index=0), torch
    ) == ["cuda_device_name"]

    dirty = _tiny_accepted_bundle(tmp_path / "dirty", monkeypatch)
    dirty.run_record["source"]["dirty"] = True
    (dirty.run / "run.json").write_bytes(_canonical(dirty.run_record))
    with pytest.raises(ValueError, match="clean"):
        _prepare_tiny(dirty)

    bad_digest = _tiny_accepted_bundle(tmp_path / "digest", monkeypatch)
    bad_digest.run_record["inventory"]["sample_inventory_sha256"] = "0" * 64
    (bad_digest.run / "run.json").write_bytes(_canonical(bad_digest.run_record))
    with pytest.raises(ValueError, match="inventory"):
        _prepare_tiny(bad_digest)

    missing_file = _tiny_accepted_bundle(tmp_path / "missing-file", monkeypatch)
    (missing_file.run / "benchmark.json").unlink()
    with pytest.raises(ValueError, match="source run files"):
        _prepare_tiny(missing_file)

    extra_file = _tiny_accepted_bundle(tmp_path / "extra-file", monkeypatch)
    (extra_file.run / "extra.json").write_bytes(b"{}\n")
    with pytest.raises(ValueError, match="source run files"):
        _prepare_tiny(extra_file)

    bad_bank = _tiny_accepted_bundle(tmp_path / "bank", monkeypatch)
    bank_payload = bad_bank.torch.load(
        bad_bank.run / "memory_bank.pt", map_location="cpu", weights_only=True
    )
    bank_payload["memory_bank"] = bad_bank.bank.to(bad_bank.torch.float64)
    bad_bank.torch.save(bank_payload, bad_bank.run / "memory_bank.pt")
    with pytest.raises((TypeError, ValueError)):
        _prepare_tiny(bad_bank)

    bad_weight = _tiny_accepted_bundle(tmp_path / "weight", monkeypatch)
    (bad_weight.cache / "resnet50-11ad3fa6.pth").write_bytes(b"changed weight")
    with pytest.raises(ValueError, match="weight SHA-256"):
        _prepare_tiny(bad_weight)

    bad_retrieval = _tiny_accepted_bundle(tmp_path / "retrieval", monkeypatch)
    retrieval_payload = bad_retrieval.torch.load(
        bad_retrieval.run / "retrieval.pt", map_location="cpu", weights_only=True
    )
    retrieval_payload["nearest_bank_indices"][0, 0] = 3
    bad_retrieval.torch.save(retrieval_payload, bad_retrieval.run / "retrieval.pt")
    with pytest.raises((TypeError, ValueError)):
        _prepare_tiny(bad_retrieval)

    wrong_sample = _tiny_accepted_bundle(tmp_path / "sample", monkeypatch)
    wrong_sample.benchmark_record["benchmark_sample_id"] = (
        "mvtec_ad/bottle/test/good/000.png"
    )
    (wrong_sample.run / "benchmark.json").write_bytes(
        _canonical(wrong_sample.benchmark_record)
    )
    with pytest.raises(ValueError, match="benchmark sample"):
        _prepare_tiny(wrong_sample)


def _assert_exact_parity_before_publication(
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
    import inspectrt.features as feature_module
    import inspectrt.preprocessing as preprocessing

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

    closed: list[bool] = []
    decoded = SimpleNamespace(image=SimpleNamespace(close=lambda: closed.append(True)))
    image = bundle.torch.zeros((3, 256, 256), dtype=bundle.torch.float32)
    expected = bundle.torch.tensor(((1, 2), (3, 4)), dtype=bundle.torch.float32)
    monkeypatch.setattr(preprocessing, "decode_image", lambda path: decoded)
    monkeypatch.setattr(preprocessing, "preprocess_decoded_image", lambda value: image)
    monkeypatch.setattr(
        feature_module,
        "extract_patch_embeddings",
        lambda extractor, batch: expected.unsqueeze(0),
    )
    reconstructed = fixtures.reconstruct_fixture_query(
        bundle.image, object(), bundle.torch.device("cpu")
    )
    assert bundle.torch.equal(reconstructed, expected)
    assert reconstructed.is_contiguous()
    assert closed == [True]

    parity_root = tmp_path / "parity"
    parity_root.mkdir()
    with monkeypatch.context() as scoped:
        _assert_exact_parity_before_publication(parity_root, scoped)


def test_existing_destination_and_late_failure_leave_no_partial_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import inspectrt.artifacts as artifacts

    changed_root = tmp_path / "changed"
    changed_root.mkdir()
    with monkeypatch.context() as scoped:
        _assert_source_change_prevents_publication(changed_root, scoped)

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


def _assert_source_change_prevents_publication(
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
