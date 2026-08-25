"""Private access to InspectRT's frozen bundled resources."""

from collections.abc import Iterator
from contextlib import contextmanager
import hashlib
from importlib import resources
from pathlib import Path
import tempfile

BASELINE_SHA256 = "8df093df5eb8e35f77e0e8c088746b34fe69023f115f89fb822a5682d66cdfb6"
FIXTURE_MANIFEST_SHA256 = (
    "9e72d4238ee0cae7f8236a82e50acf8f811c0e3f7b5e2815a11c56a9e1193c12"
)
FIXTURE_PAYLOAD_SHA256 = (
    "18c2c4333a060ff25b7304dd396cf4b292617c4593d7cbfc2576b406ed5a14bb"
)
FIXTURE_DIGEST = "ec30a68439f52051028a56cbd5a1c560edc2bccc4e77e603fa2d3355a26a4e9e"

_BASELINE = ("baseline.toml", 303, BASELINE_SHA256)
_MANIFEST = ("retrieval_v1/manifest.json", 1577, FIXTURE_MANIFEST_SHA256)
_PAYLOAD = ("retrieval_v1/tensors.bin", 416, FIXTURE_PAYLOAD_SHA256)
_SOURCE_PATHS = {
    "baseline.toml": "configs/baseline.toml",
    "retrieval_v1/manifest.json": "tests/fixtures/retrieval_v1/manifest.json",
    "retrieval_v1/tensors.bin": "tests/fixtures/retrieval_v1/tensors.bin",
}


def bundled_baseline_bytes() -> bytes:
    """Return the exact bundled baseline profile bytes."""
    return _resource_bytes(*_BASELINE)


@contextmanager
def bundled_fixture_directory() -> Iterator[Path]:
    """Yield the strict loader's two-file fixture directory."""
    manifest = _resource_bytes(*_MANIFEST)
    payload = _resource_bytes(*_PAYLOAD)
    if hashlib.sha256(manifest + payload).hexdigest() != FIXTURE_DIGEST:
        raise ValueError("bundled retrieval fixture digest mismatch")

    source_root = _source_root()
    if source_root is not None:
        yield source_root / "tests/fixtures/retrieval_v1"
        return

    with tempfile.TemporaryDirectory(prefix="inspectrt-fixture-") as temporary:
        directory = Path(temporary)
        (directory / "manifest.json").write_bytes(manifest)
        (directory / "tensors.bin").write_bytes(payload)
        yield directory


def source_checkout_root() -> Path | None:
    """Return the package's verified source checkout, never the caller's CWD."""
    root = _source_root()
    if root is None:
        return None
    git_marker = root / ".git"
    lockfile = root / "uv.lock"
    if (
        not git_marker.exists()
        or git_marker.is_symlink()
        or not lockfile.is_file()
        or lockfile.is_symlink()
    ):
        raise RuntimeError("InspectRT source checkout provenance is incomplete")
    return root


def _source_root() -> Path | None:
    module = Path(__file__)
    if not module.is_file():
        return None
    if module.is_symlink():
        raise ValueError("InspectRT resource helper must not be a symlink")
    root = module.resolve().parent.parent
    project = root / "pyproject.toml"
    if not project.exists():
        return None
    if not project.is_file() or project.is_symlink():
        raise ValueError("InspectRT source project marker must be a real file")
    return root


def _resource_bytes(name: str, size: int, digest: str) -> bytes:
    root = _source_root()
    if root is not None:
        path = root / _SOURCE_PATHS[name]
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"canonical InspectRT resource is missing: {name}")
        data = path.read_bytes()
    else:
        resource = resources.files("inspectrt").joinpath("_resources")
        for part in name.split("/"):
            resource = resource.joinpath(part)
        if isinstance(resource, Path) and resource.is_symlink():
            raise ValueError(
                f"bundled InspectRT resource must not be a symlink: {name}"
            )
        try:
            data = resource.read_bytes()
        except (FileNotFoundError, IsADirectoryError) as error:
            raise FileNotFoundError(
                f"bundled InspectRT resource is missing: {name}"
            ) from error
    _validate_bytes(name, data, size, digest)
    return data


def _validate_bytes(name: str, data: bytes, size: int, digest: str) -> None:
    if len(data) != size:
        raise ValueError(f"InspectRT resource byte count mismatch: {name}")
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError(f"InspectRT resource SHA-256 mismatch: {name}")
