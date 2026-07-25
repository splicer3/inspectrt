"""Canonical cross-language retrieval fixture serialization."""

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tempfile
from typing import Any

import numpy as np
from numpy.typing import NDArray

_MILESTONE = "inspectrt_retrieval_fixture_v1"
_SYNTHETIC_RECIPE = "inspectrt_synthetic_correctness_v1"
_TENSOR_NAMES = (
    "queries memory_bank expected_squared_l2_distances expected_indices".split()
)
_ARTIFACT_NAMES = set(
    "benchmark.json memory_bank.pt retrieval.pt run.json samples.jsonl".split()
)
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
_RUN_FILES = {
    "anomaly_maps.pt",
    "benchmark.json",
    "memory_bank.pt",
    "metrics.json",
    "predictions.jsonl",
    "retrieval.pt",
    "run.json",
    "samples.jsonl",
}
_REAL_CATEGORY = "bottle"
_REAL_Q = 1024
_REAL_M = 214016
_REAL_D = 512
_REAL_CHUNK_SIZE = 16384
_WEIGHT_ENUM = "ResNet50_Weights.IMAGENET1K_V2"
_WEIGHT_URL = "https://download.pytorch.org/models/resnet50-11ad3fa6.pth"


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


@dataclass(frozen=True, slots=True)
class AcceptedRunFixtureSource:
    """Validated accepted-run data needed to publish one real fixture."""

    metadata: RetrievalFixtureMetadata
    memory_bank: Any
    expected_squared_l2_distances: Any
    expected_indices: Any
    image_path: Path
    run_directory: Path
    source_hashes: Mapping[str, str]


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


def prepare_accepted_run_fixture(
    *,
    run_directory: Path,
    dataset_root: Path,
    sample_id: str,
    config_path: Path,
    repository_root: Path,
    generator_commit: str,
    generator_dirty: bool,
    current_lock_sha256: str,
    torch: Any,
) -> AcceptedRunFixtureSource:
    """Validate one accepted benchmark bundle before model construction."""
    if not run_directory.is_dir() or run_directory.is_symlink():
        raise ValueError(f"source run must be a real directory: {run_directory}")
    paths = {path.name: path for path in run_directory.iterdir()}
    if set(paths) != _RUN_FILES:
        raise ValueError(
            "source run files are invalid; "
            f"missing={sorted(_RUN_FILES - set(paths))}, "
            f"unexpected={sorted(set(paths) - _RUN_FILES)}"
        )
    if any(path.is_symlink() or not path.is_file() for path in paths.values()):
        raise ValueError("source run artifacts must be regular files")
    source_hashes = _source_hashes(run_directory)
    run = _parse_manifest(paths["run.json"].read_bytes())
    benchmark = _parse_manifest(paths["benchmark.json"].read_bytes())
    _validate_run_metadata(run, benchmark, run_directory.name, current_lock_sha256)

    config_bytes = config_path.read_bytes()
    expected_config = repository_root / "configs" / "baseline.toml"
    if config_path.resolve() != expected_config.resolve():
        raise ValueError(f"config must be the committed baseline: {expected_config}")
    source_commit = run["source"]["git_commit"]
    if (
        _git_blob(repository_root, source_commit, "configs/baseline.toml")
        != config_bytes
    ):
        raise ValueError("baseline configuration differs from the accepted source")
    if (
        _git_blob(repository_root, generator_commit, "configs/baseline.toml")
        != config_bytes
    ):
        raise ValueError("baseline configuration differs from the generator commit")
    if generator_dirty:
        raise ValueError("fixture generator working tree must be clean")

    samples_bytes = paths["samples.jsonl"].read_bytes()
    inventory_digest = hashlib.sha256(samples_bytes).hexdigest()
    if inventory_digest != run["inventory"]["sample_inventory_sha256"]:
        raise ValueError("sample inventory SHA-256 mismatch")
    samples = _parse_json_lines(samples_bytes)
    inventory_matches = [
        record for record in samples if record.get("sample_id") == sample_id
    ]
    if len(inventory_matches) != 1:
        raise ValueError("sample ID must occur exactly once in the source inventory")
    sample_record = inventory_matches[0]
    if (
        sample_record.get("category") != _REAL_CATEGORY
        or sample_record.get("split") != "test"
        or not isinstance(sample_record.get("image_relpath"), str)
    ):
        raise ValueError("source sample inventory record is invalid")
    _relative_posix(sample_record["image_relpath"], "source image_relpath")
    image_path = dataset_root / sample_record["image_relpath"]
    if not image_path.is_file() or image_path.is_symlink():
        raise FileNotFoundError(f"source dataset image not found: {image_path}")

    cached_weight = (
        Path(torch.hub.get_dir())
        / "checkpoints"
        / PurePosixPath(run["weights"]["source_url"]).name
    )
    if not cached_weight.is_file() or cached_weight.is_symlink():
        raise FileNotFoundError(
            f"cached official weight file not found: {cached_weight}"
        )
    if _file_sha256(cached_weight) != run["weights"]["cached_file_sha256"]:
        raise ValueError("cached official weight SHA-256 mismatch")

    bank_payload = torch.load(
        paths["memory_bank.pt"], map_location="cpu", weights_only=True
    )
    retrieval_payload = torch.load(
        paths["retrieval.pt"], map_location="cpu", weights_only=True
    )
    memory_bank, distances, indices, test_index = _validate_source_tensors(
        bank_payload, retrieval_payload, run, benchmark, sample_id, torch
    )
    source = RealApplicationFixtureSource(
        category=_REAL_CATEGORY,
        sample_id=sample_id,
        test_tensor_index=test_index,
        accepted_run_id=run["run_id"],
        source_commit=source_commit,
        source_dirty=False,
        inventory_sha256=inventory_digest,
        uv_lock_sha256=current_lock_sha256,
        weight_enum=run["weights"]["enum"],
        weight_file_sha256=run["weights"]["cached_file_sha256"],
        baseline_profile=run["profile_id"],
        configuration_sha256=hashlib.sha256(config_bytes).hexdigest(),
        preprocessing_identity=run["preprocessing_profile"],
        feature_layer=run["feature_layer"],
        source_image_sha256=_file_sha256(image_path),
        python_version=run["environment"]["python_version"],
        dependency_versions=dict(run["environment"]["dependency_versions"]),
        platform_description=run["environment"]["platform_description"],
        requested_device=run["device"],
        determinism=dict(run["determinism"]),
        cuda_device_name=benchmark["environment"]["cuda_device_name"],
        cuda_compute_capability=tuple(
            benchmark["environment"]["cuda_compute_capability"]
        ),
        pytorch_cuda_runtime_version=benchmark["environment"][
            "pytorch_cuda_runtime_version"
        ],
        source_artifact_sha256=source_hashes,
    )
    metadata = RetrievalFixtureMetadata(
        real_fixture_id(_REAL_CATEGORY, sample_id, source_commit),
        "real_application",
        _REAL_CHUNK_SIZE,
        source,
        FixtureGenerator(_MILESTONE, 1, generator_commit, False),
    )
    return AcceptedRunFixtureSource(
        metadata,
        memory_bank,
        distances,
        indices,
        image_path,
        run_directory,
        source_hashes,
    )


def publish_accepted_run_fixture(
    source: AcceptedRunFixtureSource,
    queries: Any,
    recomputed_distances: Any,
    recomputed_indices: Any,
    output_root: Path,
) -> tuple[Path, LoadedRetrievalFixture]:
    """Require exact source parity and atomically publish one real fixture."""
    import torch

    _torch_tensor(
        queries, "queries", (_REAL_Q, _REAL_D), torch.float32, torch, finite=True
    )
    _torch_tensor(
        source.memory_bank,
        "memory bank",
        (_REAL_M, _REAL_D),
        torch.float32,
        torch,
        finite=True,
    )
    _torch_tensor(
        recomputed_distances,
        "recomputed distances",
        (_REAL_Q,),
        torch.float32,
        torch,
        finite=True,
    )
    _torch_tensor(
        recomputed_indices,
        "recomputed indices",
        (_REAL_Q,),
        torch.int64,
        torch,
    )
    if (
        recomputed_indices.min().item() < 0
        or recomputed_indices.max().item() >= _REAL_M
    ):
        raise ValueError("recomputed indices lie outside the memory bank")
    if not torch.equal(recomputed_indices, source.expected_indices):
        raise ValueError("reference index mismatch")
    if not torch.equal(recomputed_distances, source.expected_squared_l2_distances):
        raise ValueError("reference distance mismatch")

    fixture = RetrievalFixture(
        source.metadata,
        _numpy_tensor(queries, _FLOAT32),
        _numpy_tensor(source.memory_bank, _FLOAT32),
        _numpy_tensor(recomputed_distances, _FLOAT32),
        _numpy_tensor(recomputed_indices, _INT64),
    )
    parent = output_root / "fixtures" / _MILESTONE
    parent.mkdir(parents=True, exist_ok=True)
    destination = parent / source.metadata.fixture_id
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"fixture directory already exists: {destination}")
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{source.metadata.fixture_id}.tmp-", dir=parent)
    )
    try:
        write_retrieval_fixture(fixture, temporary)
        loaded = load_retrieval_fixture(temporary)
        for actual, expected in zip(
            (
                loaded.queries,
                loaded.memory_bank,
                loaded.expected_squared_l2_distances,
                loaded.expected_indices,
            ),
            (
                fixture.queries,
                fixture.memory_bank,
                fixture.expected_squared_l2_distances,
                fixture.expected_indices,
            ),
            strict=True,
        ):
            if not np.array_equal(actual, expected):
                raise ValueError("raw fixture reload differs from source tensors")
        if _source_hashes(source.run_directory) != dict(source.source_hashes):
            raise ValueError("source artifacts changed during fixture export")
        from inspectrt.artifacts import _rename_without_overwrite

        _rename_without_overwrite(temporary, destination)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return destination, loaded


def reconstruct_fixture_query(image_path: Path, extractor: Any, device: Any) -> Any:
    """Reconstruct the canonical query through the frozen baseline pipeline."""
    from inspectrt.features import extract_patch_embeddings
    from inspectrt.preprocessing import decode_image, preprocess_decoded_image

    decoded = decode_image(image_path)
    try:
        image = preprocess_decoded_image(decoded)
    finally:
        decoded.image.close()
    queries = extract_patch_embeddings(extractor, image.unsqueeze(0).to(device))[0]
    return queries.contiguous()


def basic_environment_mismatches(
    source: RealApplicationFixtureSource,
    *,
    requested_device: str,
    current_lock_sha256: str,
    python_version: str,
    dependency_versions: Mapping[str, str],
    platform_description: str,
) -> list[str]:
    """Compare real reference identity without importing or initializing CUDA."""
    mismatches = []
    values = (
        ("requested_device", requested_device, source.requested_device),
        ("uv_lock_sha256", current_lock_sha256, source.uv_lock_sha256),
        ("python_version", python_version, source.python_version),
        (
            "dependency_versions",
            dict(dependency_versions),
            dict(source.dependency_versions),
        ),
        ("platform_description", platform_description, source.platform_description),
    )
    for name, actual, expected in values:
        if actual != expected:
            mismatches.append(name)
    return mismatches


def cuda_environment_mismatches(
    source: RealApplicationFixtureSource, device: Any, torch: Any
) -> list[str]:
    """Compare the recorded CUDA identity after availability is established."""
    mismatches = []
    index = device.index
    values = (
        (
            "cuda_device_name",
            torch.cuda.get_device_name(index),
            source.cuda_device_name,
        ),
        (
            "cuda_compute_capability",
            tuple(torch.cuda.get_device_capability(index)),
            source.cuda_compute_capability,
        ),
        (
            "pytorch_cuda_runtime_version",
            torch.version.cuda,
            source.pytorch_cuda_runtime_version,
        ),
    )
    for name, actual, expected in values:
        if actual != expected:
            mismatches.append(name)
    return mismatches


def _validate_run_metadata(
    run: dict[str, object],
    benchmark: dict[str, object],
    directory_name: str,
    current_lock_sha256: str,
) -> None:
    _keys(
        run,
        {
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
        },
        "run.json",
    )
    fixed = {
        "schema_version": 1,
        "run_id": directory_name,
        "profile_id": "inspectrt_feature_memory_v1",
        "preprocessing_profile": "inspectrt_resize256_v1",
        "feature_extractor": "ResNet-50",
        "feature_layer": "layer2",
        "retrieval_semantics": "exact top-1 squared L2",
        "category": _REAL_CATEGORY,
        "device": "cuda:0",
        "bank_chunk_size": _REAL_CHUNK_SIZE,
        "batch_size": 1,
    }
    if any(run[name] != value for name, value in fixed.items()):
        raise ValueError("source run does not identify the frozen bottle baseline")
    _component(run["run_id"], "run.run_id")

    source = _object(run["source"], "run.source")
    _keys(source, {"dirty", "git_commit", "uv_lock_sha256"}, "run.source")
    if source["dirty"] is not False:
        raise ValueError("accepted source run must record a clean source state")
    _commit(source["git_commit"], "run.source.git_commit")
    _hash(source["uv_lock_sha256"], "run.source.uv_lock_sha256")
    _hash(current_lock_sha256, "current_lock_sha256")
    if source["uv_lock_sha256"] != current_lock_sha256:
        raise ValueError("current uv.lock SHA-256 differs from the accepted source")

    determinism = _object(run["determinism"], "run.determinism")
    _keys(determinism, set(_DETERMINISM), "run.determinism")
    if any(
        type(determinism[name]) is not type(value) or determinism[name] != value
        for name, value in _DETERMINISM.items()
    ):
        raise ValueError("source determinism does not match the frozen baseline")

    environment = _object(run["environment"], "run.environment")
    _keys(
        environment,
        {
            "created_at_utc",
            "dependency_versions",
            "platform_description",
            "python_version",
        },
        "run.environment",
    )
    for name in ("created_at_utc", "platform_description", "python_version"):
        _string(environment[name], f"run.environment.{name}")
    dependencies = _object(
        environment["dependency_versions"], "run.environment.dependency_versions"
    )
    _keys(dependencies, _DEPENDENCIES, "run.environment.dependency_versions")
    for name, value in dependencies.items():
        _string(value, f"run.environment.dependency_versions.{name}")

    weights = _object(run["weights"], "run.weights")
    _keys(weights, {"cached_file_sha256", "enum", "source_url"}, "run.weights")
    if weights["enum"] != _WEIGHT_ENUM or weights["source_url"] != _WEIGHT_URL:
        raise ValueError("source weight identity does not match IMAGENET1K_V2")
    _hash(weights["cached_file_sha256"], "run.weights.cached_file_sha256")

    inventory = _object(run["inventory"], "run.inventory")
    _keys(
        inventory,
        {
            "anomalous_test_sample_count",
            "sample_inventory_sha256",
            "test_good_sample_count",
            "test_sample_count",
            "total_sample_count",
            "training_sample_count",
        },
        "run.inventory",
    )
    _hash(inventory["sample_inventory_sha256"], "run.inventory digest")
    for name in set(inventory) - {"sample_inventory_sha256"}:
        _integer(inventory[name], f"run.inventory.{name}", positive=True)

    benchmark_record = _object(run["benchmark"], "run.benchmark")
    _keys(
        benchmark_record,
        {"artifact", "schema_version", "timing_device"},
        "run.benchmark",
    )
    if benchmark_record != {
        "artifact": "benchmark.json",
        "schema_version": 1,
        "timing_device": "cuda:0",
    }:
        raise ValueError("source run is not an accepted benchmark bundle")

    tensors = _object(run["tensors"], "run.tensors")
    expected_tensor_names = {
        "anomaly_maps",
        "evaluation_masks",
        "image_scores",
        "memory_bank",
        "nearest_bank_indices",
        "patch_distances",
        "test_labels",
    }
    _keys(tensors, expected_tensor_names, "run.tensors")
    bank = _object(tensors["memory_bank"], "run.tensors.memory_bank")
    _keys(bank, {"byte_count", "dtype", "shape"}, "run.tensors.memory_bank")
    if bank != {
        "byte_count": _REAL_M * _REAL_D * 4,
        "dtype": "float32",
        "shape": [_REAL_M, _REAL_D],
    }:
        raise ValueError("source bank tensor metadata is invalid")
    test_count = inventory["test_sample_count"]
    for name, dtype, shape in (
        ("patch_distances", "float32", [test_count, _REAL_Q]),
        ("nearest_bank_indices", "int64", [test_count, _REAL_Q]),
    ):
        record = _object(tensors[name], f"run.tensors.{name}")
        _keys(record, {"dtype", "shape"}, f"run.tensors.{name}")
        if record != {"dtype": dtype, "shape": shape}:
            raise ValueError(f"source {name} tensor metadata is invalid")

    _keys(
        benchmark,
        {
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
        },
        "benchmark.json",
    )
    benchmark_fixed = {
        "schema_version": 1,
        "run_id": run["run_id"],
        "category": _REAL_CATEGORY,
        "profile_id": "inspectrt_feature_memory_v1",
        "device": "cuda:0",
        "created_at_utc": environment["created_at_utc"],
    }
    if any(benchmark[name] != value for name, value in benchmark_fixed.items()):
        raise ValueError("benchmark identity differs from run.json")
    _relative_posix(benchmark["benchmark_sample_id"], "benchmark_sample_id")
    benchmark_environment = _object(benchmark["environment"], "benchmark.environment")
    _keys(
        benchmark_environment,
        {
            "cuda_compute_capability",
            "cuda_device_name",
            "pytorch_cuda_runtime_version",
        },
        "benchmark.environment",
    )
    _string(benchmark_environment["cuda_device_name"], "benchmark CUDA device")
    _string(
        benchmark_environment["pytorch_cuda_runtime_version"],
        "benchmark CUDA runtime",
    )
    capability = benchmark_environment["cuda_compute_capability"]
    if (
        not isinstance(capability, list)
        or len(capability) != 2
        or any(type(value) is not int or value < 0 for value in capability)
    ):
        raise ValueError("benchmark compute capability is invalid")
    workload = _object(benchmark["workload"], "benchmark.workload")
    _keys(
        workload,
        {
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
        },
        "benchmark.workload",
    )
    workload_fixed = {
        "Q": _REAL_Q,
        "M": _REAL_M,
        "D": _REAL_D,
        "k": 1,
        "bank_chunk_size": _REAL_CHUNK_SIZE,
        "bank_shape": [_REAL_M, _REAL_D],
        "bank_bytes": _REAL_M * _REAL_D * 4,
        "batch_size": 1,
        "dtype": "float32",
        "test_sample_count": inventory["test_sample_count"],
        "training_sample_count": inventory["training_sample_count"],
    }
    if any(workload[name] != value for name, value in workload_fixed.items()):
        raise ValueError("benchmark workload does not match the frozen baseline")
    _object(benchmark["methodology"], "benchmark.methodology")
    _object(benchmark["results"], "benchmark.results")


def _validate_source_tensors(
    bank_payload: object,
    retrieval_payload: object,
    run: Mapping[str, Any],
    benchmark: Mapping[str, Any],
    sample_id: str,
    torch: Any,
) -> tuple[Any, Any, Any, int]:
    if not isinstance(bank_payload, dict):
        raise TypeError("memory_bank.pt must contain an object")
    _keys(
        bank_payload,
        {
            "dtype",
            "embedding_dimension",
            "memory_bank",
            "patches_per_training_sample",
            "shape",
        },
        "memory_bank.pt",
    )
    fixed = {
        "dtype": "float32",
        "embedding_dimension": _REAL_D,
        "patches_per_training_sample": _REAL_Q,
        "shape": [_REAL_M, _REAL_D],
    }
    if any(bank_payload[name] != value for name, value in fixed.items()):
        raise ValueError("memory_bank.pt metadata is invalid")
    memory_bank = bank_payload["memory_bank"]
    _torch_tensor(
        memory_bank,
        "memory bank",
        (_REAL_M, _REAL_D),
        torch.float32,
        torch,
        finite=True,
    )

    if not isinstance(retrieval_payload, dict):
        raise TypeError("retrieval.pt must contain an object")
    _keys(
        retrieval_payload,
        {"nearest_bank_indices", "patch_distances", "test_sample_ids"},
        "retrieval.pt",
    )
    test_ids = retrieval_payload["test_sample_ids"]
    test_count = run["inventory"]["test_sample_count"]
    if (
        not isinstance(test_ids, list)
        or len(test_ids) != test_count
        or any(not isinstance(value, str) for value in test_ids)
    ):
        raise ValueError("retrieval test_sample_ids are invalid")
    matches = [index for index, value in enumerate(test_ids) if value == sample_id]
    if len(matches) != 1:
        raise ValueError("sample ID must occur exactly once in retrieval rows")
    if benchmark["benchmark_sample_id"] != sample_id:
        raise ValueError("requested sample does not match the benchmark sample")
    distances = retrieval_payload["patch_distances"]
    indices = retrieval_payload["nearest_bank_indices"]
    _torch_tensor(
        distances,
        "retrieval distances",
        (test_count, _REAL_Q),
        torch.float32,
        torch,
        finite=True,
    )
    _torch_tensor(
        indices,
        "retrieval indices",
        (test_count, _REAL_Q),
        torch.int64,
        torch,
    )
    if distances.min().item() < 0:
        raise ValueError("retrieval squared distances must be nonnegative")
    if indices.min().item() < 0 or indices.max().item() >= _REAL_M:
        raise ValueError("retrieval indices lie outside the memory bank")
    index = matches[0]
    return (
        memory_bank,
        distances[index].contiguous(),
        indices[index].contiguous(),
        index,
    )


def _torch_tensor(
    value: object,
    name: str,
    shape: tuple[int, ...],
    dtype: object,
    torch: Any,
    *,
    finite: bool = False,
) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}; got {tuple(value.shape)}")
    if value.dtype != dtype:
        raise TypeError(f"{name} must use {dtype}; got {value.dtype}")
    if value.device.type != "cpu":
        raise ValueError(f"{name} must be on the CPU")
    if not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if finite and not torch.isfinite(value).all().item():
        raise ValueError(f"{name} must contain only finite values")


def _numpy_tensor(tensor: Any, dtype: np.dtype[Any]) -> np.ndarray[Any, Any]:
    array = tensor.detach().numpy()
    if array.dtype != dtype or not array.flags.c_contiguous:
        raise ValueError("raw export tensor does not match its storage contract")
    return array


def _parse_json_lines(payload: bytes) -> list[dict[str, object]]:
    if not payload or not payload.endswith(b"\n"):
        raise ValueError("samples.jsonl must be nonempty canonical JSONL")
    return [_parse_manifest(line) for line in payload.splitlines(keepends=True)]


def _source_hashes(run_directory: Path) -> dict[str, str]:
    return {
        name: _file_sha256(run_directory / name) for name in sorted(_ARTIFACT_NAMES)
    }


def _file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _git_blob(repository_root: Path, commit: str, path: str) -> bytes:
    result = subprocess.run(
        ("git", "show", f"{commit}:{path}"),
        cwd=repository_root,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        detail = result.stderr.decode(errors="replace").strip() or "unknown error"
        raise ValueError(f"cannot read committed baseline configuration: {detail}")
    return result.stdout


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
    if source["weight_enum"] != _WEIGHT_ENUM or source["requested_device"] != "cuda:0":
        raise ValueError("real source must use IMAGENET1K_V2 on explicit cuda:0")
    dependencies = _object(source["dependency_versions"], "source.dependency_versions")
    _keys(dependencies, _DEPENDENCIES, "source.dependency_versions")
    for name, value in dependencies.items():
        _string(value, f"source.dependency_versions.{name}")
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
    if np.any(fixture.expected_squared_l2_distances < 0):
        raise ValueError("expected squared-L2 distances must be nonnegative")
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
