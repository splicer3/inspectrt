"""Strict publication and loading for the static ONNX feature artifact."""

from dataclasses import dataclass
import hashlib
import importlib
from importlib import metadata as importlib_metadata
import json
from pathlib import Path
import platform
import re
import shutil
import tempfile
from typing import Any
from urllib.parse import urlsplit

_MILESTONE_ID = "inspectrt_onnx_feature_portability_v1"
_MODEL_FILENAME = "model.onnx"
_EXPECTED_MODEL_BYTES = 5_857_483
_EXPECTED_MODEL_SHA256 = (
    "143b305b37a92e3f2c7dc4268c25baccdf3cfb01c5304f29068f422ff9d8146a"
)
_WEIGHT_URL = "https://download.pytorch.org/models/resnet50-11ad3fa6.pth"
_WEIGHT_BYTES = 102_540_417
_WEIGHT_SHA256 = "11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca"
_WEIGHT_ENUM = "ResNet50_Weights.IMAGENET1K_V2"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_DEPENDENCIES = (
    "inspectrt",
    "numpy",
    "torch",
    "torchvision",
    "onnx",
    "onnxscript",
    "onnx-ir",
    "onnxruntime",
    "ml-dtypes",
    "protobuf",
    "flatbuffers",
)
_TOP_LEVEL_KEYS = {
    "artifact_digest",
    "artifact_id",
    "environment",
    "export",
    "feature_contract",
    "milestone_id",
    "model",
    "onnx",
    "schema_version",
    "source",
    "weights",
}
_EXPORT = {
    "dynamic_shapes": False,
    "example_input": "zeros_fp32_nchw_1x3x256x256_v1",
    "exporter": "torch.onnx.export",
    "external_data": False,
    "mode": "dynamo",
    "opset_version": 20,
    "optimize": True,
    "report": False,
    "verify": False,
}
_FEATURE_CONTRACT = {
    "embedding_dimension": 512,
    "feature_layer": "layer2",
    "input": {
        "dtype": "float32",
        "layout": "NCHW",
        "name": "images",
        "shape": [1, 3, 256, 256],
    },
    "outputs": [
        {
            "dtype": "float32",
            "layout": "NCHW",
            "name": "layer2",
            "shape": [1, 512, 32, 32],
        },
        {
            "dtype": "float32",
            "layout": "NLC",
            "name": "patch_embeddings",
            "shape": [1, 1024, 512],
        },
    ],
    "patch_count": 1024,
    "patch_order": "row_major_y_then_x",
    "pool": {
        "ceil_mode": 0,
        "count_include_pad": 1,
        "kernel_shape": [3, 3],
        "pads": [1, 1, 1, 1],
        "strides": [1, 1],
    },
    "preprocessing": "outside_graph",
    "transpose_permutation": [0, 2, 3, 1],
}
_WEIGHTS = {
    "cached_file_sha256": _WEIGHT_SHA256,
    "enum": _WEIGHT_ENUM,
    "source_url": _WEIGHT_URL,
}


@dataclass(frozen=True, slots=True)
class LoadedOnnxFeatureArtifact:
    """Validated immutable identity for one schema-1 artifact."""

    artifact_id: str
    model_byte_count: int
    model_sha256: str
    artifact_digest: str
    opset_version: int


def publish_onnx_feature_artifact(
    output_root: Path,
    *,
    git_commit: str,
    git_dirty: bool,
    uv_lock_sha256: str,
) -> tuple[Path, LoadedOnnxFeatureArtifact]:
    """Export and atomically publish the fixed schema-1 feature artifact."""
    if not isinstance(output_root, Path):
        raise TypeError("output_root must be a pathlib.Path")
    source = _source_record(git_commit, git_dirty, uv_lock_sha256)
    onnx = _import_onnx(export=True)
    environment = {
        "dependency_versions": {
            name: importlib_metadata.version(name) for name in _DEPENDENCIES
        },
        "python_version": platform.python_version(),
    }

    artifact_root = output_root / "artifacts" / _MILESTONE_ID
    artifact_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".artifact.tmp-", dir=artifact_root))
    try:
        model_path = temporary / _MODEL_FILENAME
        _serialize_fixed_model(model_path)
        paths = list(temporary.iterdir())
        if paths != [model_path] or model_path.is_symlink() or not model_path.is_file():
            raise ValueError("ONNX export must create only one regular model.onnx file")

        model_bytes = model_path.read_bytes()
        model_sha256 = hashlib.sha256(model_bytes).hexdigest()
        onnx_record = _inspect_model(
            onnx,
            model_bytes,
            checker_path=model_path,
        )
        if (
            len(model_bytes) != _EXPECTED_MODEL_BYTES
            or model_sha256 != _EXPECTED_MODEL_SHA256
        ):
            raise RuntimeError(
                "exported model identity differs from the fixed candidate model"
            )
        artifact_id = _artifact_id(model_sha256)
        manifest = {
            "artifact_id": artifact_id,
            "environment": environment,
            "export": _EXPORT,
            "feature_contract": _FEATURE_CONTRACT,
            "milestone_id": _MILESTONE_ID,
            "model": {
                "byte_count": len(model_bytes),
                "filename": _MODEL_FILENAME,
                "sha256": model_sha256,
            },
            "onnx": onnx_record,
            "schema_version": 1,
            "source": source,
            "weights": _WEIGHTS,
        }
        artifact_digest = hashlib.sha256(
            _canonical_json(manifest) + model_bytes
        ).hexdigest()
        manifest["artifact_digest"] = artifact_digest
        (temporary / "manifest.json").write_bytes(_canonical_json(manifest))
        loaded = load_onnx_feature_artifact(temporary)

        destination = artifact_root / artifact_id
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"artifact directory already exists: {destination}")
        from inspectrt.artifacts import _rename_without_overwrite

        _rename_without_overwrite(temporary, destination)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return destination, loaded


def load_onnx_feature_artifact(directory: Path) -> LoadedOnnxFeatureArtifact:
    """Load and fail-closed validate one two-file schema-1 artifact."""
    if not isinstance(directory, Path):
        raise TypeError("directory must be a pathlib.Path")
    if not directory.is_dir() or directory.is_symlink():
        raise ValueError("artifact path must be a real directory")
    paths = {path.name: path for path in directory.iterdir()}
    if set(paths) != {"manifest.json", _MODEL_FILENAME}:
        raise ValueError(
            "artifact directory must contain only manifest.json and model.onnx"
        )
    if any(path.is_symlink() or not path.is_file() for path in paths.values()):
        raise ValueError("artifact files must be regular non-symlink files")

    manifest_bytes = paths["manifest.json"].read_bytes()
    model_bytes = paths[_MODEL_FILENAME].read_bytes()
    manifest = _parse_manifest(manifest_bytes)
    _validate_manifest(manifest)

    model_record = _object(manifest["model"], "model")
    model_sha256 = hashlib.sha256(model_bytes).hexdigest()
    if model_record["byte_count"] != len(model_bytes):
        raise ValueError("model byte count does not match model.onnx")
    if model_record["sha256"] != model_sha256:
        raise ValueError("model SHA-256 does not match model.onnx")
    artifact_id = _artifact_id(model_sha256)
    if manifest["artifact_id"] != artifact_id:
        raise ValueError("artifact_id does not match the model SHA-256")

    without_digest = dict(manifest)
    recorded_digest = without_digest.pop("artifact_digest")
    artifact_digest = hashlib.sha256(
        _canonical_json(without_digest) + model_bytes
    ).hexdigest()
    if recorded_digest != artifact_digest:
        raise ValueError("artifact digest does not match manifest and model bytes")

    onnx = _import_onnx(export=False)
    onnx_record = _inspect_model(onnx, model_bytes)
    _exact(manifest["onnx"], onnx_record, "onnx")
    if (
        len(model_bytes) != _EXPECTED_MODEL_BYTES
        or model_sha256 != _EXPECTED_MODEL_SHA256
    ):
        raise ValueError("model.onnx differs from the accepted artifact model")
    return LoadedOnnxFeatureArtifact(
        artifact_id,
        len(model_bytes),
        model_sha256,
        artifact_digest,
        20,
    )


def _serialize_fixed_model(model_path: Path) -> None:
    import torch
    from torchvision.models import ResNet50_Weights

    from inspectrt.onnx_features import build_onnx_feature_graph

    weights = ResNet50_Weights.IMAGENET1K_V2
    if str(weights) != _WEIGHT_ENUM or weights.url != _WEIGHT_URL:
        raise RuntimeError("resolved official weight identity is unexpected")
    cached = (
        Path(torch.hub.get_dir())
        / "checkpoints"
        / Path(urlsplit(weights.url).path).name
    )
    _check_cached_weight(cached)
    graph = build_onnx_feature_graph(weights=weights)
    _check_cached_weight(cached)
    images = torch.zeros((1, 3, 256, 256), dtype=torch.float32)
    program = torch.onnx.export(
        graph,
        (images,),
        f=None,
        dynamo=True,
        input_names=["images"],
        output_names=["layer2", "patch_embeddings"],
        opset_version=20,
        external_data=False,
        dynamic_shapes=None,
        optimize=True,
        report=False,
        verify=False,
    )
    if not isinstance(program, torch.onnx.ONNXProgram):
        raise RuntimeError("torch.onnx.export did not return an ONNXProgram")
    program.save(model_path, external_data=False)
    _check_cached_weight(cached)


def _check_cached_weight(path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(
            f"accepted cached weight must be a regular non-symlink file: {path}"
        )
    if path.stat().st_size != _WEIGHT_BYTES or _file_sha256(path) != _WEIGHT_SHA256:
        raise ValueError("accepted cached weight identity does not match")


def _inspect_model(
    onnx: Any,
    model_bytes: bytes,
    *,
    checker_path: Path | None = None,
) -> dict[str, object]:
    try:
        model = onnx.load_model_from_string(model_bytes)
    except Exception as error:
        raise ValueError("model.onnx is not a valid ONNX protobuf") from error
    messages = tuple(_protobuf_messages(model))
    for message in messages:
        if isinstance(message, onnx.TensorProto) and (
            message.data_location == onnx.TensorProto.EXTERNAL
            or bool(message.external_data)
        ):
            raise ValueError("model.onnx must not use external tensor data")
    try:
        onnx.checker.check_model(
            checker_path if checker_path is not None else model,
            full_check=True,
        )
    except Exception as error:
        raise ValueError("model.onnx failed ONNX full checking") from error

    inputs = [_value_info(value, onnx) for value in model.graph.input]
    outputs = [_value_info(value, onnx) for value in model.graph.output]
    if inputs != [("images", onnx.TensorProto.FLOAT, (1, 3, 256, 256))]:
        raise ValueError("model input contract is invalid")
    if outputs != [
        ("layer2", onnx.TensorProto.FLOAT, (1, 512, 32, 32)),
        ("patch_embeddings", onnx.TensorProto.FLOAT, (1, 1024, 512)),
    ]:
        raise ValueError("model output contract is invalid")

    opsets = [
        {"domain": opset.domain, "version": int(opset.version)}
        for opset in model.opset_import
    ]
    if opsets != [{"domain": "", "version": 20}]:
        raise ValueError("model must import only default-domain opset 20")
    all_nodes = [message for message in messages if isinstance(message, onnx.NodeProto)]
    if any(node.domain != "" for node in all_nodes):
        raise ValueError("model nodes must use only the default ONNX domain")
    if model.functions:
        raise ValueError("model must not contain local ONNX functions")

    nodes = {
        operation: [node for node in model.graph.node if node.op_type == operation]
        for operation in ("AveragePool", "Transpose", "Reshape")
    }
    if {name: len(values) for name, values in nodes.items()} != {
        "AveragePool": 1,
        "Transpose": 1,
        "Reshape": 1,
    }:
        raise ValueError("model pool and layout operator counts are invalid")
    average_pool = nodes["AveragePool"][0]
    transpose = nodes["Transpose"][0]
    reshape = nodes["Reshape"][0]
    if _attributes(average_pool, onnx) != {
        "auto_pad": b"NOTSET",
        "ceil_mode": 0,
        "count_include_pad": 1,
        "kernel_shape": [3, 3],
        "pads": [1, 1, 1, 1],
        "strides": [1, 1],
    }:
        raise ValueError("AveragePool attributes are invalid")
    if _attributes(transpose, onnx) != {"perm": [0, 2, 3, 1]}:
        raise ValueError("Transpose attributes are invalid")
    if (
        list(average_pool.input) != ["layer2"]
        or len(average_pool.output) != 1
        or list(transpose.input) != list(average_pool.output)
        or len(transpose.output) != 1
        or len(reshape.input) != 2
        or reshape.input[0] != transpose.output[0]
        or list(reshape.output) != ["patch_embeddings"]
    ):
        raise ValueError("model pool and row-major layout path is invalid")

    return {
        "checker_full_check": "passed",
        "external_data": False,
        "graph_name": model.graph.name,
        "ir_version": int(model.ir_version),
        "model_domain": model.domain,
        "model_version": int(model.model_version),
        "node_domains": sorted({node.domain for node in all_nodes}),
        "opset_imports": opsets,
        "producer_name": model.producer_name,
        "producer_version": model.producer_version,
    }


def _value_info(value: Any, onnx: Any) -> tuple[str, int, tuple[int, ...]]:
    if not value.type.HasField("tensor_type"):
        raise ValueError("model input and output types must be tensors")
    tensor_type = value.type.tensor_type
    if not tensor_type.HasField("shape"):
        raise ValueError("model input and output shapes must be present")
    dimensions = tensor_type.shape.dim
    if any(dimension.WhichOneof("value") != "dim_value" for dimension in dimensions):
        raise ValueError("model input and output dimensions must be static integers")
    return (
        value.name,
        int(tensor_type.elem_type),
        tuple(int(dimension.dim_value) for dimension in dimensions),
    )


def _attributes(node: Any, onnx: Any) -> dict[str, object]:
    result = {}
    for attribute in node.attribute:
        if attribute.name in result:
            raise ValueError(f"duplicate ONNX node attribute: {attribute.name}")
        result[attribute.name] = onnx.helper.get_attribute_value(attribute)
    return result


def _protobuf_messages(message: Any) -> Any:
    yield message
    for field, value in message.ListFields():
        if field.message_type is None:
            continue
        if field.is_repeated:
            for item in value:
                yield from _protobuf_messages(item)
        else:
            yield from _protobuf_messages(value)


def _source_record(
    git_commit: object,
    git_dirty: object,
    uv_lock_sha256: object,
) -> dict[str, object]:
    commit = _commit(git_commit, "git_commit")
    if type(git_dirty) is not bool:
        raise TypeError("git_dirty must be a boolean")
    if git_dirty:
        raise ValueError("ONNX artifact source working tree must be clean")
    return {
        "git_commit": commit,
        "git_dirty": False,
        "uv_lock_sha256": _hash(uv_lock_sha256, "uv_lock_sha256"),
    }


def _validate_manifest(manifest: dict[str, object]) -> None:
    _keys(manifest, _TOP_LEVEL_KEYS, "manifest")
    _exact(manifest["schema_version"], 1, "schema_version")
    _exact(manifest["milestone_id"], _MILESTONE_ID, "milestone_id")
    _hash(manifest["artifact_digest"], "artifact_digest")
    _string(manifest["artifact_id"], "artifact_id")
    _exact(manifest["export"], _EXPORT, "export")
    _exact(manifest["feature_contract"], _FEATURE_CONTRACT, "feature_contract")
    _exact(manifest["weights"], _WEIGHTS, "weights")

    source = _object(manifest["source"], "source")
    _keys(source, {"git_commit", "git_dirty", "uv_lock_sha256"}, "source")
    _commit(source["git_commit"], "source.git_commit")
    _exact(source["git_dirty"], False, "source.git_dirty")
    _hash(source["uv_lock_sha256"], "source.uv_lock_sha256")

    environment = _object(manifest["environment"], "environment")
    _keys(
        environment,
        {"python_version", "dependency_versions"},
        "environment",
    )
    _string(environment["python_version"], "environment.python_version")
    dependencies = _object(
        environment["dependency_versions"],
        "environment.dependency_versions",
    )
    _keys(dependencies, set(_DEPENDENCIES), "environment.dependency_versions")
    for name, value in dependencies.items():
        _string(value, f"environment.dependency_versions.{name}")

    model = _object(manifest["model"], "model")
    _keys(model, {"filename", "byte_count", "sha256"}, "model")
    _exact(model["filename"], _MODEL_FILENAME, "model.filename")
    _integer(model["byte_count"], "model.byte_count", positive=True)
    _hash(model["sha256"], "model.sha256")

    onnx = _object(manifest["onnx"], "onnx")
    _keys(
        onnx,
        {
            "checker_full_check",
            "external_data",
            "graph_name",
            "ir_version",
            "model_domain",
            "model_version",
            "node_domains",
            "opset_imports",
            "producer_name",
            "producer_version",
        },
        "onnx",
    )
    _exact(onnx["checker_full_check"], "passed", "onnx.checker_full_check")
    _exact(onnx["external_data"], False, "onnx.external_data")
    _string(onnx["graph_name"], "onnx.graph_name")
    _integer(onnx["ir_version"], "onnx.ir_version", positive=True)
    _string(onnx["model_domain"], "onnx.model_domain", allow_empty=True)
    _integer(onnx["model_version"], "onnx.model_version")
    _string(onnx["producer_name"], "onnx.producer_name")
    _string(onnx["producer_version"], "onnx.producer_version")
    node_domains = _list(onnx["node_domains"], "onnx.node_domains")
    for index, domain in enumerate(node_domains):
        _string(domain, f"onnx.node_domains[{index}]", allow_empty=True)
    opsets = _list(onnx["opset_imports"], "onnx.opset_imports")
    for index, value in enumerate(opsets):
        opset = _object(value, f"onnx.opset_imports[{index}]")
        _keys(opset, {"domain", "version"}, f"onnx.opset_imports[{index}]")
        _string(
            opset["domain"],
            f"onnx.opset_imports[{index}].domain",
            allow_empty=True,
        )
        _integer(
            opset["version"],
            f"onnx.opset_imports[{index}].version",
            positive=True,
        )


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
    from inspectrt.artifacts import _canonical_json as encode

    return encode(value)


def _import_onnx(*, export: bool) -> Any:
    try:
        onnx = importlib.import_module("onnx")
        if export:
            importlib.import_module("onnxscript")
    except (ImportError, ModuleNotFoundError) as error:
        raise RuntimeError(
            "install inspectrt[onnx] to use ONNX artifact commands"
        ) from error
    return onnx


def _artifact_id(model_sha256: str) -> str:
    return f"resnet50-layer2-opset20-{model_sha256[:12]}"


def _file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _keys(value: dict[str, object], expected: set[str], name: str) -> None:
    if set(value) != expected:
        raise ValueError(
            f"{name} fields are invalid; "
            f"missing={sorted(expected - set(value))}, "
            f"unknown={sorted(set(value) - expected)}"
        )


def _object(value: object, name: str) -> dict[str, object]:
    if type(value) is not dict:
        raise TypeError(f"{name} must be an object")
    return value


def _list(value: object, name: str) -> list[object]:
    if type(value) is not list:
        raise TypeError(f"{name} must be an array")
    return value


def _string(value: object, name: str, *, allow_empty: bool = False) -> str:
    if type(value) is not str or (not allow_empty and not value):
        raise TypeError(f"{name} must be {'a string' if allow_empty else 'nonempty'}")
    if any(ord(character) < 32 for character in value):
        raise ValueError(f"{name} must not contain control characters")
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


def _exact(value: object, expected: object, name: str) -> None:
    if type(value) is not type(expected):
        raise TypeError(f"{name} has the wrong primitive type")
    if isinstance(expected, dict):
        _keys(value, set(expected), name)
        for key, expected_item in expected.items():
            _exact(value[key], expected_item, f"{name}.{key}")
    elif isinstance(expected, list):
        if len(value) != len(expected):
            raise ValueError(f"{name} has the wrong array length")
        for index, (item, expected_item) in enumerate(
            zip(value, expected, strict=True)
        ):
            _exact(item, expected_item, f"{name}[{index}]")
    elif value != expected:
        raise ValueError(f"{name} must be {expected!r}")
