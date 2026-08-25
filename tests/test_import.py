import hashlib
from pathlib import Path
import subprocess
import sys
import tomllib

import pytest


def test_package_import() -> None:
    import inspectrt
    from inspectrt import _resources

    assert inspectrt.__name__ == "inspectrt"
    baseline = _resources.bundled_baseline_bytes()
    assert len(baseline) == 303
    assert hashlib.sha256(baseline).hexdigest() == _resources.BASELINE_SHA256
    with _resources.bundled_fixture_directory() as directory:
        manifest = (directory / "manifest.json").read_bytes()
        payload = (directory / "tensors.bin").read_bytes()
        assert {path.name for path in directory.iterdir()} == {
            "manifest.json",
            "tensors.bin",
        }
    assert hashlib.sha256(manifest).hexdigest() == _resources.FIXTURE_MANIFEST_SHA256
    assert hashlib.sha256(payload).hexdigest() == _resources.FIXTURE_PAYLOAD_SHA256
    assert hashlib.sha256(manifest + payload).hexdigest() == _resources.FIXTURE_DIGEST
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        _resources._validate_bytes("changed", b"x", 1, "0" * 64)


def test_onnx_extra_is_published_and_package_import_isolated() -> None:
    root = Path(__file__).resolve().parents[1]
    with (root / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)["project"]

    assert project["optional-dependencies"]["onnx"] == [
        "onnx>=1.22.0",
        "onnxruntime==1.28.0",
        "onnxscript>=0.7.1",
    ]
    assert not any(
        requirement.lower().startswith("onnx")
        for requirement in project["dependencies"]
    )

    code = """
import sys
import inspectrt
import inspectrt.cli
import inspectrt._resources

for arguments in (
    ["evaluate", "--help"],
    ["benchmark", "--help"],
    ["fixture", "validate", "--help"],
):
    try:
        inspectrt.cli.main(arguments)
    except SystemExit as error:
        assert error.code == 0
    else:
        raise AssertionError("help did not exit")

for package in ("numpy", "torch", "torchvision", "onnx", "onnxruntime", "onnxscript"):
    assert not any(
        name == package or name.startswith(f"{package}.") for name in sys.modules
    )
"""
    result = subprocess.run(
        (sys.executable, "-c", code),
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
