from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from inspectrt.fixtures import (
    FixtureGenerator,
    RealApplicationFixtureSource,
    RetrievalFixture,
    RetrievalFixtureMetadata,
    SyntheticFixtureSource,
    encode_retrieval_fixture,
    load_retrieval_fixture,
    real_fixture_id,
    write_retrieval_fixture,
)

_COMMIT = "bc330b9070c5ca8db9cb7cfbb27617256388536b"
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
