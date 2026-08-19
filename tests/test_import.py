from pathlib import Path
import subprocess
import sys
import tomllib


def test_package_import() -> None:
    import inspectrt

    assert inspectrt.__name__ == "inspectrt"


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

for package in ("onnx", "onnxruntime", "onnxscript"):
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
