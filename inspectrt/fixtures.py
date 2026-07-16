"""Canonical cross-language retrieval fixture serialization."""

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re

import numpy as np
from numpy.typing import NDArray

_MILESTONE = "inspectrt_retrieval_fixture_v1"
_SYNTHETIC_RECIPE = "inspectrt_synthetic_correctness_v1"
_TENSOR_NAMES = (
    "queries memory_bank expected_squared_l2_distances expected_indices".split()
)
_ARTIFACT_NAMES = {
    "benchmark.json",
    "memory_bank.pt",
    "retrieval.pt",
    "run.json",
    "samples.jsonl",
}
_DEPENDENCIES = {"inspectrt", "numpy", "pillow", "scikit-learn", "torch", "torchvision"}
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
_SHA256 = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_FIXTURE_ID = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
_FLOAT32 = np.dtype("<f4")
_INT64 = np.dtype("<i8")


@dataclass(frozen=True, slots=True)
class SyntheticFixtureSource:
    recipe_id: str
    reference_chunk_size: int


@dataclass(frozen=True, slots=True)
class RealApplicationFixtureSource:
    category: str
    sample_id: str
    test_tensor_index: int
    accepted_run_id: str
    source_commit: str
    source_dirty: bool
    inventory_sha256: str
    uv_lock_sha256: str
    weight_enum: str
    weight_file_sha256: str
    baseline_profile: str
    configuration_sha256: str
    preprocessing_identity: str
    feature_layer: str
    source_image_sha256: str
    python_version: str
    dependency_versions: Mapping[str, str]
    platform_description: str
    requested_device: str
    determinism: Mapping[str, str | int | bool]
    cuda_device_name: str
    cuda_compute_capability: tuple[int, int]
    pytorch_cuda_runtime_version: str
    source_artifact_sha256: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class FixtureGenerator:
    milestone_id: str
    schema_version: int
    git_commit: str
    dirty: bool


@dataclass(frozen=True, slots=True)
class RetrievalFixtureMetadata:
    fixture_id: str
    fixture_class: str
    reference_chunk_size: int
    source: SyntheticFixtureSource | RealApplicationFixtureSource
    generator: FixtureGenerator


@dataclass(frozen=True, slots=True)
class RetrievalFixture:
    metadata: RetrievalFixtureMetadata
    queries: NDArray[np.float32]
    memory_bank: NDArray[np.float32]
    expected_squared_l2_distances: NDArray[np.float32]
    expected_indices: NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class LoadedRetrievalFixture:
    metadata: RetrievalFixtureMetadata
    queries: NDArray[np.float32]
    memory_bank: NDArray[np.float32]
    expected_squared_l2_distances: NDArray[np.float32]
    expected_indices: NDArray[np.int64]
    manifest: Mapping[str, object]
    fixture_digest: str


def real_fixture_id(category: str, sample_id: str, source_commit: str) -> str:
    _component(category, "category")
    _relative_posix(sample_id, "sample_id")
    _commit(source_commit, "source_commit")
    parts = sample_id.split("/")
    if len(parts) < 4 or parts[-3] != "test" or parts[-4] != category:
        raise ValueError("sample_id must end with <category>/test/<defect>/<file>")
    stem = PurePosixPath(parts[-1]).stem
    values = (_normalize(category), _normalize(parts[-2]), _normalize(stem))
    fixture_id = "-".join((*values, source_commit[:12]))
    if len(fixture_id.encode("ascii")) > 96:
        raise ValueError("fixture_id must be at most 96 bytes")
    return fixture_id


def encode_retrieval_fixture(fixture: RetrievalFixture) -> tuple[bytes, bytes]:
    if not isinstance(fixture, RetrievalFixture):
        raise TypeError("fixture must be RetrievalFixture")
    q, m, d = _validate_arrays(fixture)
    payload = bytearray()
    entries = []
    for name, array in (
        ("queries", fixture.queries),
        ("memory_bank", fixture.memory_bank),
        (
            "expected_squared_l2_distances",
            fixture.expected_squared_l2_distances,
        ),
        ("expected_indices", fixture.expected_indices),
    ):
        offset = _align(len(payload))
        payload.extend(b"\0" * (offset - len(payload)))
        data = array.tobytes(order="C")
        payload.extend(data)
        entries.append(
            {
                "byte_order": "little",
                "dtype": "float32" if array.dtype == _FLOAT32 else "int64",
                "layout": "C",
                "name": name,
                "nbytes": len(data),
                "offset_bytes": offset,
                "sha256": hashlib.sha256(data).hexdigest(),
                "shape": list(array.shape),
            }
        )
    payload_bytes = bytes(payload)
    metadata = fixture.metadata
    manifest = {
        "fixture_class": metadata.fixture_class,
        "fixture_id": metadata.fixture_id,
        "generator": _generator_record(metadata.generator),
        "payload": {
            "alignment_bytes": 64,
            "file": "tensors.bin",
            "nbytes": len(payload_bytes),
            "sha256": hashlib.sha256(payload_bytes).hexdigest(),
        },
        "retrieval": {
            "chunk_merge": "strictly_lower_distance",
            "distance_output": "raw_squared_l2",
            "index_scope": "global_memory_bank",
            "operation": "exact_top1_squared_l2",
            "tie_rule": "lower_global_index",
        },
        "schema_version": 1,
        "source": _source_record(metadata.source),
        "tensors": entries,
        "workload": {
            "D": d,
            "M": m,
            "Q": q,
            "dtype": "float32",
            "k": 1,
            "layout": "C",
            "reference_chunk_size": metadata.reference_chunk_size,
        },
    }
    manifest_bytes = _canonical_json(manifest)
    _validate_encoded(manifest_bytes, payload_bytes)
    return manifest_bytes, payload_bytes


def write_retrieval_fixture(fixture: RetrievalFixture, directory: Path) -> str:
    if not isinstance(directory, Path):
        raise TypeError("directory must be a pathlib.Path")
    manifest, payload = encode_retrieval_fixture(fixture)
    if directory.is_symlink() or (
        directory.exists() and (not directory.is_dir() or any(directory.iterdir()))
    ):
        raise FileExistsError(f"Fixture directory is not empty: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "tensors.bin").write_bytes(payload)
    (directory / "manifest.json").write_bytes(manifest)
    return hashlib.sha256(manifest + payload).hexdigest()


def load_retrieval_fixture(directory: Path) -> LoadedRetrievalFixture:
    """Load and fail-closed validate an existing two-file fixture."""
    if not isinstance(directory, Path):
        raise TypeError("directory must be a pathlib.Path")
    if not directory.is_dir() or directory.is_symlink():
        raise ValueError("fixture directory must be a real directory")
    paths = {path.name: path for path in directory.iterdir()}
    if set(paths) != {"manifest.json", "tensors.bin"}:
        raise ValueError(
            "fixture directory must contain only manifest.json and tensors.bin"
        )
    if any(path.is_symlink() or not path.is_file() for path in paths.values()):
        raise ValueError("fixture files must be regular files")
    manifest_bytes = paths["manifest.json"].read_bytes()
    payload = paths["tensors.bin"].read_bytes()
    manifest, metadata, entries = _validate_encoded(manifest_bytes, payload)
    arrays = []
    for entry in entries:
        dtype = _FLOAT32 if entry["dtype"] == "float32" else _INT64
        arrays.append(
            np.frombuffer(
                payload,
                dtype=dtype,
                count=math.prod(entry["shape"]),
                offset=entry["offset_bytes"],
            )
            .reshape(entry["shape"])
            .copy(order="C")
        )
    return LoadedRetrievalFixture(
        metadata,
        *arrays,
        manifest,
        hashlib.sha256(manifest_bytes + payload).hexdigest(),
    )


def _validate_encoded(
    manifest_bytes: bytes, payload: bytes
) -> tuple[dict[str, object], RetrievalFixtureMetadata, list[dict[str, object]]]:
    manifest = _parse_manifest(manifest_bytes)
    _keys(
        manifest,
        {
            "schema_version",
            "fixture_id",
            "fixture_class",
            "workload",
            "retrieval",
            "payload",
            "tensors",
            "source",
            "generator",
        },
        "manifest",
    )
    _integer(manifest["schema_version"], "schema_version", positive=True)
    if manifest["schema_version"] != 1:
        raise ValueError("unsupported fixture schema_version")
    fixture_class = _string(manifest["fixture_class"], "fixture_class")
    if fixture_class not in {"synthetic_correctness", "real_application"}:
        raise ValueError("unsupported fixture_class")
    workload = _object(manifest["workload"], "workload")
    _keys(
        workload,
        {"Q", "M", "D", "k", "reference_chunk_size", "dtype", "layout"},
        "workload",
    )
    q, m, d, k, chunk = (
        _integer(workload[name], f"workload.{name}", positive=True)
        for name in ("Q", "M", "D", "k", "reference_chunk_size")
    )
    if k != 1 or workload["dtype"] != "float32" or workload["layout"] != "C":
        raise ValueError("workload must use k=1, float32, and C layout")
    retrieval = _object(manifest["retrieval"], "retrieval")
    expected_retrieval = {
        "chunk_merge": "strictly_lower_distance",
        "distance_output": "raw_squared_l2",
        "index_scope": "global_memory_bank",
        "operation": "exact_top1_squared_l2",
        "tie_rule": "lower_global_index",
    }
    _keys(retrieval, set(expected_retrieval), "retrieval")
    if retrieval != expected_retrieval:
        raise ValueError("unsupported retrieval semantics")
    payload_record = _object(manifest["payload"], "payload")
    _keys(
        payload_record,
        {"file", "nbytes", "alignment_bytes", "sha256"},
        "payload",
    )
    alignment = _integer(
        payload_record["alignment_bytes"], "payload.alignment_bytes", positive=True
    )
    if payload_record["file"] != "tensors.bin" or alignment != 64:
        raise ValueError("payload must use tensors.bin and 64-byte alignment")
    payload_nbytes = _integer(payload_record["nbytes"], "payload.nbytes")
    _hash(payload_record["sha256"], "payload.sha256")
    if payload_nbytes != len(payload):
        raise ValueError("payload length does not match manifest")
    if hashlib.sha256(payload).hexdigest() != payload_record["sha256"]:
        raise ValueError("payload SHA-256 mismatch")
    entries_value = manifest["tensors"]
    if not isinstance(entries_value, list) or len(entries_value) != 4:
        raise ValueError("tensors must contain exactly four ordered entries")
    entries = [
        _object(value, f"tensors[{index}]") for index, value in enumerate(entries_value)
    ]
    shapes = ([q, d], [m, d], [q], [q])
    dtypes = ("float32", "float32", "float32", "int64")
    cursor = 0
    names = []
    for index, (entry, name, shape, dtype) in enumerate(
        zip(entries, _TENSOR_NAMES, shapes, dtypes, strict=True)
    ):
        _keys(
            entry,
            {
                "name",
                "offset_bytes",
                "nbytes",
                "shape",
                "dtype",
                "byte_order",
                "layout",
                "sha256",
            },
            f"tensors[{index}]",
        )
        names.append(entry["name"])
        tensor_shape = entry["shape"]
        if (
            not isinstance(tensor_shape, list)
            or any(type(value) is not int for value in tensor_shape)
            or tensor_shape != shape
            or entry["name"] != name
            or entry["dtype"] != dtype
        ):
            raise ValueError("tensor order, shape, or dtype does not match workload")
        if entry["byte_order"] != "little" or entry["layout"] != "C":
            raise ValueError("tensor byte order and layout must be little-endian C")
        offset = _integer(entry["offset_bytes"], f"{name}.offset_bytes")
        nbytes = _integer(entry["nbytes"], f"{name}.nbytes")
        _hash(entry["sha256"], f"{name}.sha256")
        expected_nbytes = math.prod(shape) * (4 if dtype == "float32" else 8)
        expected_offset = _align(cursor)
        if offset != expected_offset or offset % 64 or nbytes != expected_nbytes:
            raise ValueError("tensor offset, alignment, or byte count is inconsistent")
        if offset + nbytes > len(payload):
            raise ValueError("tensor segment is out of payload bounds")
        if any(payload[cursor:offset]):
            raise ValueError("tensor alignment padding must be zero")
        segment = payload[offset : offset + nbytes]
        if hashlib.sha256(segment).hexdigest() != entry["sha256"]:
            raise ValueError(f"{name} SHA-256 mismatch")
        cursor = offset + nbytes
    if len(set(names)) != 4 or cursor != len(payload):
        raise ValueError(
            "tensor segments overlap, are duplicated, or leave trailing bytes"
        )
    source = _object(manifest["source"], "source")
    generator = _parse_generator(_object(manifest["generator"], "generator"))
    fixture_id = _string(manifest["fixture_id"], "fixture_id")
    try:
        fixture_id_bytes = fixture_id.encode("ascii")
    except UnicodeEncodeError as error:
        raise ValueError("fixture_id must use lowercase ASCII") from error
    if not _FIXTURE_ID.fullmatch(fixture_id) or len(fixture_id_bytes) > 96:
        raise ValueError("fixture_id is not a valid lowercase ASCII identifier")
    if fixture_class == "synthetic_correctness":
        _keys(source, {"recipe_id", "reference_chunk_size"}, "source")
        source_chunk = _integer(
            source["reference_chunk_size"],
            "source.reference_chunk_size",
            positive=True,
        )
        if source["recipe_id"] != _SYNTHETIC_RECIPE or source_chunk != chunk:
            raise ValueError("synthetic source recipe or chunk size is invalid")
        if fixture_id != "synthetic-correctness-v1":
            raise ValueError("synthetic fixture_id is invalid")
        parsed_source: SyntheticFixtureSource | RealApplicationFixtureSource = (
            SyntheticFixtureSource(_SYNTHETIC_RECIPE, chunk)
        )
    else:
        parsed_source = _parse_real_source(source)
        expected_id = real_fixture_id(
            parsed_source.category,
            parsed_source.sample_id,
            parsed_source.source_commit,
        )
        if fixture_id != expected_id:
            raise ValueError("real fixture_id does not match its source")
    metadata = RetrievalFixtureMetadata(
        fixture_id, fixture_class, chunk, parsed_source, generator
    )
    return manifest, metadata, entries


def _parse_real_source(source: dict[str, object]) -> RealApplicationFixtureSource:
    fields = set(RealApplicationFixtureSource.__dataclass_fields__)
    _keys(source, fields, "source")
    category = _component(source["category"], "source.category")
    sample_id = _relative_posix(source["sample_id"], "source.sample_id")
    accepted_run_id = _component(source["accepted_run_id"], "source.accepted_run_id")
    source_commit = _commit(source["source_commit"], "source.source_commit")
    if source["source_dirty"] is not False:
        raise ValueError("real source must record a clean source state")
    for name in (
        "inventory_sha256",
        "uv_lock_sha256",
        "weight_file_sha256",
        "configuration_sha256",
        "source_image_sha256",
    ):
        _hash(source[name], f"source.{name}")
    fixed = {
        "baseline_profile": "inspectrt_feature_memory_v1",
        "preprocessing_identity": "inspectrt_resize256_v1",
        "feature_layer": "layer2",
    }
    if any(source[name] != value for name, value in fixed.items()):
        raise ValueError("real source baseline identity is unsupported")
    for name in (
        "weight_enum",
        "python_version",
        "platform_description",
        "requested_device",
        "cuda_device_name",
        "pytorch_cuda_runtime_version",
    ):
        _string(source[name], f"source.{name}")
    dependencies = _object(source["dependency_versions"], "source.dependency_versions")
    _keys(dependencies, _DEPENDENCIES, "source.dependency_versions")
    if any(not isinstance(value, str) or not value for value in dependencies.values()):
        raise ValueError("dependency versions must be nonempty strings")
    determinism = _object(source["determinism"], "source.determinism")
    _keys(determinism, set(_DETERMINISM), "source.determinism")
    if any(
        type(determinism[name]) is not type(value) or determinism[name] != value
        for name, value in _DETERMINISM.items()
    ):
        raise ValueError("determinism identity does not match the frozen baseline")
    capability = source["cuda_compute_capability"]
    if (
        not isinstance(capability, list)
        or len(capability) != 2
        or any(type(value) is not int or value < 0 for value in capability)
    ):
        raise ValueError(
            "cuda_compute_capability must contain two nonnegative integers"
        )
    artifacts = _object(
        source["source_artifact_sha256"], "source.source_artifact_sha256"
    )
    _keys(artifacts, _ARTIFACT_NAMES, "source.source_artifact_sha256")
    for name, digest in artifacts.items():
        _hash(digest, f"source.source_artifact_sha256.{name}")
    values = dict(source)
    values["test_tensor_index"] = _integer(
        source["test_tensor_index"], "source.test_tensor_index"
    )
    values["category"] = category
    values["sample_id"] = sample_id
    values["accepted_run_id"] = accepted_run_id
    values["source_commit"] = source_commit
    values["dependency_versions"] = dict(dependencies)
    values["determinism"] = dict(determinism)
    values["cuda_compute_capability"] = tuple(capability)
    values["source_artifact_sha256"] = dict(artifacts)
    return RealApplicationFixtureSource(**values)  # type: ignore[arg-type]


def _parse_generator(value: dict[str, object]) -> FixtureGenerator:
    _keys(
        value,
        {"milestone_id", "schema_version", "git_commit", "dirty"},
        "generator",
    )
    if value["milestone_id"] != _MILESTONE:
        raise ValueError("generator milestone_id is invalid")
    if (
        _integer(value["schema_version"], "generator.schema_version", positive=True)
        != 1
    ):
        raise ValueError("generator schema_version is unsupported")
    commit = _commit(value["git_commit"], "generator.git_commit")
    if value["dirty"] is not False:
        raise ValueError("generator must record a clean source state")
    return FixtureGenerator(_MILESTONE, 1, commit, False)


def _validate_arrays(fixture: RetrievalFixture) -> tuple[int, int, int]:
    if not isinstance(fixture.metadata, RetrievalFixtureMetadata):
        raise TypeError("metadata must be RetrievalFixtureMetadata")
    arrays = (
        (fixture.queries, "queries", _FLOAT32, 2),
        (fixture.memory_bank, "memory_bank", _FLOAT32, 2),
        (
            fixture.expected_squared_l2_distances,
            "expected_squared_l2_distances",
            _FLOAT32,
            1,
        ),
        (fixture.expected_indices, "expected_indices", _INT64, 1),
    )
    for array, name, dtype, rank in arrays:
        if not isinstance(array, np.ndarray):
            raise TypeError(f"{name} must be a numpy.ndarray")
        if array.dtype != dtype:
            raise TypeError(f"{name} must use little-endian {dtype.name}")
        if array.ndim != rank:
            raise ValueError(f"{name} must have rank {rank}")
        if not array.flags.c_contiguous:
            raise ValueError(f"{name} must be C-contiguous")
    q, d = fixture.queries.shape
    m, bank_d = fixture.memory_bank.shape
    if min(q, m, d) <= 0 or bank_d != d:
        raise ValueError(
            "Q, M, and D must be positive and feature dimensions must match"
        )
    if fixture.expected_squared_l2_distances.shape != (
        q,
    ) or fixture.expected_indices.shape != (q,):
        raise ValueError("expected outputs must have shape [Q]")
    if (
        not np.isfinite(fixture.queries).all()
        or not np.isfinite(fixture.memory_bank).all()
    ):
        raise ValueError("retrieval inputs must contain only finite values")
    if not np.isfinite(fixture.expected_squared_l2_distances).all():
        raise ValueError("expected distances must contain only finite values")
    if np.any(fixture.expected_indices < 0) or np.any(fixture.expected_indices >= m):
        raise ValueError("expected indices must lie within the memory bank")
    _integer(
        fixture.metadata.reference_chunk_size,
        "reference_chunk_size",
        positive=True,
    )
    return q, m, d


def _source_record(source: object) -> dict[str, object]:
    if isinstance(source, SyntheticFixtureSource):
        return {
            "recipe_id": source.recipe_id,
            "reference_chunk_size": source.reference_chunk_size,
        }
    if not isinstance(source, RealApplicationFixtureSource):
        raise TypeError("source must be a supported fixture source")
    return {
        name: (
            list(value)
            if name == "cuda_compute_capability"
            else dict(value)
            if isinstance(value, Mapping)
            else value
        )
        for name in RealApplicationFixtureSource.__dataclass_fields__
        for value in (getattr(source, name),)
    }


def _generator_record(generator: object) -> dict[str, object]:
    if not isinstance(generator, FixtureGenerator):
        raise TypeError("generator must be FixtureGenerator")
    return {
        "dirty": generator.dirty,
        "git_commit": generator.git_commit,
        "milestone_id": generator.milestone_id,
        "schema_version": generator.schema_version,
    }


def _parse_manifest(payload: bytes) -> dict[str, object]:
    def pairs(values: list[tuple[str, object]]) -> dict[str, object]:
        result = {}
        for key, value in values:
            if key in result:
                raise ValueError(f"duplicate JSON object key: {key}")
            result[key] = value
        return result

    def constant(value: str) -> object:
        raise ValueError(f"non-finite JSON number: {value}")

    try:
        parsed = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("manifest.json is not valid UTF-8 JSON") from error
    if not isinstance(parsed, dict):
        raise ValueError("manifest root must be an object")
    if _canonical_json(parsed) != payload:
        raise ValueError("manifest.json is not canonical")
    return parsed


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _keys(value: dict[str, object], expected: set[str], name: str) -> None:
    if set(value) != expected:
        missing = sorted(expected - set(value))
        unknown = sorted(set(value) - expected)
        raise ValueError(
            f"{name} fields are invalid; missing={missing}, unknown={unknown}"
        )


def _object(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be an object")
    return value


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or any(ord(char) < 32 for char in value):
        raise ValueError(f"{name} must be a nonempty string without control characters")
    return value


def _integer(value: object, name: str, *, positive: bool = False) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if value < (1 if positive else 0):
        raise ValueError(f"{name} is out of range")
    return value


def _hash(value: object, name: str) -> str:
    value = _string(value, name)
    if not _SHA256.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _commit(value: object, name: str) -> str:
    value = _string(value, name)
    if not _COMMIT.fullmatch(value):
        raise ValueError(f"{name} must be a full lowercase commit hash")
    return value


def _relative_posix(value: object, name: str) -> str:
    value = _string(value, name)
    if (
        "\\" in value
        or PurePosixPath(value).is_absolute()
        or any(part in {"", ".", ".."} for part in value.split("/"))
    ):
        raise ValueError(f"{name} must be a safe relative POSIX identifier")
    return value


def _component(value: object, name: str) -> str:
    value = _relative_posix(value, name)
    if "/" in value:
        raise ValueError(f"{name} must be one path component")
    return value


def _normalize(value: str) -> str:
    lowered = "".join(
        chr(ord(char) + 32) if "A" <= char <= "Z" else char for char in value
    )
    result = re.sub(r"[^a-z0-9]+", "-", lowered).strip("-")
    if not result:
        raise ValueError("fixture ID source components must not normalize to empty")
    return result


def _align(value: int) -> int:
    return (value + 63) // 64 * 64
