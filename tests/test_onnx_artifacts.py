from dataclasses import FrozenInstanceError
import hashlib
from importlib.util import find_spec
import json
from pathlib import Path
import shutil

import pytest

import inspectrt.onnx_artifacts as artifacts

_COMMIT = "3442fa51fa4134851147466d9cac662c7fd85953"
_LOCK = "d92724be7ede2442141cf898a67d12752e44d3bd5df6077dbd5ae97df325df42"
_REAL_IMPORT_ONNX = artifacts._import_onnx
_REAL_INSPECT_MODEL = artifacts._inspect_model
_MODEL_RECORD = {
    "checker_full_check": "passed",
    "external_data": False,
    "graph_name": "main_graph",
    "ir_version": 10,
    "model_domain": "",
    "model_version": 0,
    "node_domains": [""],
    "opset_imports": [{"domain": "", "version": 20}],
    "producer_name": "pytorch",
    "producer_version": "2.13.0+cu130",
}


@pytest.fixture(scope="module")
def contract_model(tmp_path_factory: pytest.TempPathFactory) -> Path:
    directory = tmp_path_factory.mktemp("onnx-artifact-contract")
    path = directory / "model.onnx"
    if find_spec("onnx") is None:
        path.write_bytes(b"base-install-model-double")
        return path

    import onnx
    from onnx import TensorProto, helper

    layer2_shape = helper.make_tensor(
        "layer2_shape", TensorProto.INT64, [4], [1, 512, 32, 32]
    )
    patch_shape = helper.make_tensor(
        "patch_shape", TensorProto.INT64, [3], [1, 1024, 512]
    )
    zero = helper.make_tensor("zero", TensorProto.FLOAT, [1], [0.0])
    nodes = [
        helper.make_node(
            "ConstantOfShape",
            ["layer2_shape"],
            ["layer2"],
            value=zero,
        ),
        helper.make_node(
            "AveragePool",
            ["layer2"],
            ["pooled"],
            auto_pad="NOTSET",
            ceil_mode=0,
            count_include_pad=1,
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1],
            strides=[1, 1],
        ),
        helper.make_node("Transpose", ["pooled"], ["nhwc"], perm=[0, 2, 3, 1]),
        helper.make_node(
            "Reshape",
            ["nhwc", "patch_shape"],
            ["patch_embeddings"],
            allowzero=1,
        ),
    ]
    graph = helper.make_graph(
        nodes,
        "main_graph",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 256, 256])],
        [
            helper.make_tensor_value_info(
                "layer2", TensorProto.FLOAT, [1, 512, 32, 32]
            ),
            helper.make_tensor_value_info(
                "patch_embeddings", TensorProto.FLOAT, [1, 1024, 512]
            ),
        ],
        [layer2_shape, patch_shape],
    )
    model = helper.make_model(
        graph,
        ir_version=10,
        opset_imports=[helper.make_opsetid("", 20)],
        producer_name="pytorch",
        producer_version="2.13.0+cu130",
    )
    path.write_bytes(model.SerializeToString())
    onnx.checker.check_model(path, full_check=True)
    return path


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


def _publish(
    root: Path,
    model_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, artifacts.LoadedOnnxFeatureArtifact]:
    model_bytes = model_path.read_bytes()
    monkeypatch.setattr(
        artifacts,
        "_serialize_fixed_model",
        lambda destination: shutil.copyfile(model_path, destination),
    )
    monkeypatch.setattr(artifacts, "_EXPECTED_MODEL_BYTES", len(model_bytes))
    monkeypatch.setattr(
        artifacts,
        "_EXPECTED_MODEL_SHA256",
        hashlib.sha256(model_bytes).hexdigest(),
    )
    monkeypatch.setattr(artifacts, "_import_onnx", lambda *, export: object())
    monkeypatch.setattr(
        artifacts,
        "_inspect_model",
        lambda *args, **kwargs: dict(_MODEL_RECORD),
    )
    monkeypatch.setattr(artifacts.importlib_metadata, "version", lambda name: "test")
    return artifacts.publish_onnx_feature_artifact(
        root,
        git_commit=_COMMIT,
        git_dirty=False,
        uv_lock_sha256=_LOCK,
    )


def _copy_artifact(source: Path, root: Path, name: str) -> Path:
    destination = root / name
    shutil.copytree(source, destination)
    return destination


def _manifest(directory: Path) -> dict[str, object]:
    return json.loads((directory / "manifest.json").read_bytes())


def _write_manifest(directory: Path, manifest: dict[str, object]) -> None:
    (directory / "manifest.json").write_bytes(_canonical(manifest))


def _rehash_artifact(directory: Path) -> tuple[int, str]:
    manifest = _manifest(directory)
    model_bytes = (directory / "model.onnx").read_bytes()
    model_sha256 = hashlib.sha256(model_bytes).hexdigest()
    manifest["model"]["byte_count"] = len(model_bytes)
    manifest["model"]["sha256"] = model_sha256
    manifest["artifact_id"] = f"resnet50-layer2-opset20-{model_sha256[:12]}"
    manifest.pop("artifact_digest")
    manifest["artifact_digest"] = hashlib.sha256(
        _canonical(manifest) + model_bytes
    ).hexdigest()
    _write_manifest(directory, manifest)
    return len(model_bytes), model_sha256


def test_canonical_two_file_round_trip_is_deterministic_and_immutable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    contract_model: Path,
) -> None:
    first_path, first = _publish(tmp_path / "first", contract_model, monkeypatch)
    second_path, second = _publish(tmp_path / "second", contract_model, monkeypatch)

    assert first == second == artifacts.load_onnx_feature_artifact(first_path)
    assert {path.name for path in first_path.iterdir()} == {
        "manifest.json",
        "model.onnx",
    }
    assert (first_path / "manifest.json").read_bytes() == (
        second_path / "manifest.json"
    ).read_bytes()
    manifest = _manifest(first_path)
    model_bytes = (first_path / "model.onnx").read_bytes()
    digest = manifest.pop("artifact_digest")
    assert digest == hashlib.sha256(_canonical(manifest) + model_bytes).hexdigest()
    assert first_path == (
        tmp_path
        / "first"
        / "artifacts"
        / "inspectrt_onnx_feature_portability_v1"
        / first.artifact_id
    )
    with pytest.raises(FrozenInstanceError):
        first.artifact_id = "changed"


def test_strict_inventory_and_symlink_boundaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    contract_model: Path,
) -> None:
    source, _ = _publish(tmp_path / "source", contract_model, monkeypatch)

    extra = _copy_artifact(source, tmp_path, "extra")
    (extra / "unexpected").write_bytes(b"x")
    missing = _copy_artifact(source, tmp_path, "missing")
    (missing / "model.onnx").unlink()
    linked_file = _copy_artifact(source, tmp_path, "linked-file")
    (linked_file / "manifest.json").unlink()
    (linked_file / "manifest.json").symlink_to(source / "manifest.json")
    linked_directory = tmp_path / "linked-directory"
    linked_directory.symlink_to(source, target_is_directory=True)

    for directory in (extra, missing, linked_file, linked_directory):
        with pytest.raises(ValueError):
            artifacts.load_onnx_feature_artifact(directory)
    with monkeypatch.context() as scoped:
        _assert_manifest_validation_classes(
            tmp_path / "manifest-cases", scoped, contract_model
        )


def _assert_manifest_validation_classes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    contract_model: Path,
) -> None:
    source, _ = _publish(tmp_path / "source", contract_model, monkeypatch)
    original = (source / "manifest.json").read_bytes()

    duplicate = _copy_artifact(source, tmp_path, "duplicate")
    (duplicate / "manifest.json").write_bytes(
        b'{"schema_version":1,' + original.removeprefix(b"{")
    )
    noncanonical = _copy_artifact(source, tmp_path, "noncanonical")
    (noncanonical / "manifest.json").write_bytes(b" " + original)
    unknown = _copy_artifact(source, tmp_path, "unknown")
    value = _manifest(unknown)
    value["source"]["unexpected"] = True
    _write_manifest(unknown, value)
    missing = _copy_artifact(source, tmp_path, "missing-field")
    value = _manifest(missing)
    del value["weights"]["enum"]
    _write_manifest(missing, value)
    wrong_type = _copy_artifact(source, tmp_path, "wrong-type")
    value = _manifest(wrong_type)
    value["feature_contract"]["patch_count"] = True
    _write_manifest(wrong_type, value)
    nonfinite = _copy_artifact(source, tmp_path, "nonfinite")
    (nonfinite / "manifest.json").write_bytes(
        original.replace(b'"model_version":0', b'"model_version":NaN')
    )

    for directory in (
        duplicate,
        noncanonical,
        unknown,
        missing,
        wrong_type,
        nonfinite,
    ):
        with pytest.raises((TypeError, ValueError)):
            artifacts.load_onnx_feature_artifact(directory)


def test_model_id_and_digest_integrity_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    contract_model: Path,
) -> None:
    source, _ = _publish(tmp_path / "source", contract_model, monkeypatch)

    changed_model = _copy_artifact(source, tmp_path, "changed-model")
    with (changed_model / "model.onnx").open("ab") as stream:
        stream.write(b"x")
    changed_hash = _copy_artifact(source, tmp_path, "changed-hash")
    value = _manifest(changed_hash)
    value["model"]["sha256"] = "0" * 64
    value.pop("artifact_digest")
    model_bytes = (changed_hash / "model.onnx").read_bytes()
    value["artifact_digest"] = hashlib.sha256(
        _canonical(value) + model_bytes
    ).hexdigest()
    _write_manifest(changed_hash, value)
    changed_id = _copy_artifact(source, tmp_path, "changed-id")
    value = _manifest(changed_id)
    value["artifact_id"] = "resnet50-layer2-opset20-000000000000"
    value.pop("artifact_digest")
    value["artifact_digest"] = hashlib.sha256(
        _canonical(value) + (changed_id / "model.onnx").read_bytes()
    ).hexdigest()
    _write_manifest(changed_id, value)
    changed_digest = _copy_artifact(source, tmp_path, "changed-digest")
    value = _manifest(changed_digest)
    value["artifact_digest"] = "0" * 64
    _write_manifest(changed_digest, value)

    for directory in (changed_model, changed_hash, changed_id, changed_digest):
        with pytest.raises(ValueError):
            artifacts.load_onnx_feature_artifact(directory)


def test_graph_contract_metadata_and_external_data_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    contract_model: Path,
) -> None:
    if find_spec("onnx") is None:
        pytest.skip("install inspectrt[onnx] to run ONNX graph inspection tests")
    import onnx
    from onnx import TensorProto, helper

    source, _ = _publish(tmp_path / "source", contract_model, monkeypatch)
    monkeypatch.setattr(artifacts, "_import_onnx", _REAL_IMPORT_ONNX)
    monkeypatch.setattr(artifacts, "_inspect_model", _REAL_INSPECT_MODEL)

    wrong_pool = _copy_artifact(source, tmp_path, "wrong-pool")
    model = onnx.load_model_from_string((wrong_pool / "model.onnx").read_bytes())
    pool = next(node for node in model.graph.node if node.op_type == "AveragePool")
    next(
        attribute
        for attribute in pool.attribute
        if attribute.name == "count_include_pad"
    ).i = 0
    (wrong_pool / "model.onnx").write_bytes(model.SerializeToString())
    size, digest = _rehash_artifact(wrong_pool)
    monkeypatch.setattr(artifacts, "_EXPECTED_MODEL_BYTES", size)
    monkeypatch.setattr(artifacts, "_EXPECTED_MODEL_SHA256", digest)
    with pytest.raises(ValueError, match="AveragePool"):
        artifacts.load_onnx_feature_artifact(wrong_pool)

    wrong_outputs = _copy_artifact(source, tmp_path, "wrong-outputs")
    model = onnx.load_model_from_string((wrong_outputs / "model.onnx").read_bytes())
    model.graph.output.reverse()
    (wrong_outputs / "model.onnx").write_bytes(model.SerializeToString())
    size, digest = _rehash_artifact(wrong_outputs)
    monkeypatch.setattr(artifacts, "_EXPECTED_MODEL_BYTES", size)
    monkeypatch.setattr(artifacts, "_EXPECTED_MODEL_SHA256", digest)
    with pytest.raises(ValueError, match="output contract"):
        artifacts.load_onnx_feature_artifact(wrong_outputs)

    external = _copy_artifact(source, tmp_path, "external")
    model = onnx.load_model_from_string((external / "model.onnx").read_bytes())
    values = helper.make_tensor("ghost-values", TensorProto.FLOAT, [1], [0.0])
    values.external_data.add(key="location", value="ghost.bin")
    indices = helper.make_tensor("ghost-indices", TensorProto.INT64, [1, 1], [0])
    model.graph.sparse_initializer.append(
        helper.make_sparse_tensor(values, indices, [1])
    )
    (external / "model.onnx").write_bytes(model.SerializeToString())
    size, digest = _rehash_artifact(external)
    monkeypatch.setattr(artifacts, "_EXPECTED_MODEL_BYTES", size)
    monkeypatch.setattr(artifacts, "_EXPECTED_MODEL_SHA256", digest)
    with pytest.raises(ValueError, match="external tensor data"):
        artifacts.load_onnx_feature_artifact(external)

    other_identity = _copy_artifact(source, tmp_path, "other-identity")
    model = onnx.load_model_from_string((other_identity / "model.onnx").read_bytes())
    model.doc_string = "different model identity"
    (other_identity / "model.onnx").write_bytes(model.SerializeToString())
    _rehash_artifact(other_identity)
    original_bytes = contract_model.read_bytes()
    monkeypatch.setattr(artifacts, "_EXPECTED_MODEL_BYTES", len(original_bytes))
    monkeypatch.setattr(
        artifacts,
        "_EXPECTED_MODEL_SHA256",
        hashlib.sha256(original_bytes).hexdigest(),
    )
    with pytest.raises(ValueError, match="accepted artifact model"):
        artifacts.load_onnx_feature_artifact(other_identity)

    metadata = _copy_artifact(source, tmp_path, "metadata")
    value = _manifest(metadata)
    value["onnx"]["producer_name"] = "other"
    value.pop("artifact_digest")
    model_bytes = (metadata / "model.onnx").read_bytes()
    value["artifact_digest"] = hashlib.sha256(
        _canonical(value) + model_bytes
    ).hexdigest()
    _write_manifest(metadata, value)
    with pytest.raises(ValueError):
        artifacts.load_onnx_feature_artifact(metadata)


def test_atomic_no_overwrite_and_failure_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    contract_model: Path,
) -> None:
    destination, _ = _publish(tmp_path / "existing", contract_model, monkeypatch)
    before = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in destination.iterdir()
    }
    with pytest.raises(FileExistsError):
        _publish(tmp_path / "existing", contract_model, monkeypatch)
    assert before == {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in destination.iterdir()
    }

    linked_root = tmp_path / "linked"
    milestone = linked_root / "artifacts" / "inspectrt_onnx_feature_portability_v1"
    milestone.mkdir(parents=True)
    linked_destination = milestone / destination.name
    linked_destination.symlink_to(tmp_path / "absent", target_is_directory=True)
    with pytest.raises(FileExistsError):
        _publish(linked_root, contract_model, monkeypatch)
    assert linked_destination.is_symlink()
    assert list(milestone.iterdir()) == [linked_destination]

    import inspectrt.artifacts as persistence

    monkeypatch.setattr(
        persistence,
        "_rename_without_overwrite",
        lambda source, target: (_ for _ in ()).throw(OSError("late failure")),
    )
    failed_root = tmp_path / "failed"
    with pytest.raises(OSError, match="late failure"):
        _publish(failed_root, contract_model, monkeypatch)
    milestone = failed_root / "artifacts" / "inspectrt_onnx_feature_portability_v1"
    assert list(milestone.iterdir()) == []


def test_weight_and_missing_extra_trust_boundaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = tmp_path / "weight.pth"
    expected.write_bytes(b"weight")
    monkeypatch.setattr(artifacts, "_WEIGHT_BYTES", 6)
    monkeypatch.setattr(
        artifacts,
        "_WEIGHT_SHA256",
        hashlib.sha256(b"weight").hexdigest(),
    )
    artifacts._check_cached_weight(expected)
    link = tmp_path / "linked.pth"
    link.symlink_to(expected)
    with pytest.raises(FileNotFoundError):
        artifacts._check_cached_weight(link)
    expected.write_bytes(b"changed")
    with pytest.raises(ValueError):
        artifacts._check_cached_weight(expected)

    monkeypatch.setattr(
        artifacts.importlib,
        "import_module",
        lambda name: (_ for _ in ()).throw(ModuleNotFoundError(name)),
    )
    output = tmp_path / "missing-extra"
    with pytest.raises(RuntimeError, match=r"inspectrt\[onnx\]"):
        artifacts.publish_onnx_feature_artifact(
            output,
            git_commit=_COMMIT,
            git_dirty=False,
            uv_lock_sha256=_LOCK,
        )
    assert not output.exists()
