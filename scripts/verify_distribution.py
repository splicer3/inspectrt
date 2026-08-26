"""Verify InspectRT's reviewed wheel and source-distribution boundary."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import stat
import sys
import tarfile
import zipfile
from email import policy
from email.parser import BytesParser
from pathlib import Path, PurePosixPath

NAME = "inspectrt"
VERSION = "0.1.0"
ROOT = f"{NAME}-{VERSION}"
DIST_INFO = f"{ROOT}.dist-info"

PACKAGE_FILES = {
    "inspectrt/__init__.py",
    "inspectrt/_resources.py",
    "inspectrt/artifacts.py",
    "inspectrt/benchmark.py",
    "inspectrt/cli.py",
    "inspectrt/data.py",
    "inspectrt/evaluation.py",
    "inspectrt/features.py",
    "inspectrt/fixtures.py",
    "inspectrt/metrics.py",
    "inspectrt/onnx_artifacts.py",
    "inspectrt/onnx_features.py",
    "inspectrt/onnx_runtime.py",
    "inspectrt/portability.py",
    "inspectrt/preprocessing.py",
    "inspectrt/retrieval.py",
}
RESOURCE_MAP = {
    "inspectrt/_resources/baseline.toml": "configs/baseline.toml",
    "inspectrt/_resources/retrieval_v1/manifest.json": (
        "tests/fixtures/retrieval_v1/manifest.json"
    ),
    "inspectrt/_resources/retrieval_v1/tensors.bin": (
        "tests/fixtures/retrieval_v1/tensors.bin"
    ),
}
LEGAL_FILES = {"LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md"}
WHEEL_FILES = (
    PACKAGE_FILES
    | set(RESOURCE_MAP)
    | {
        f"{DIST_INFO}/METADATA",
        f"{DIST_INFO}/WHEEL",
        f"{DIST_INFO}/entry_points.txt",
        *(f"{DIST_INFO}/licenses/{name}" for name in LEGAL_FILES),
        f"{DIST_INFO}/RECORD",
    }
)
SDIST_FILES = {
    *(f"{ROOT}/{name}" for name in PACKAGE_FILES),
    *(f"{ROOT}/{name}" for name in RESOURCE_MAP.values()),
    f"{ROOT}/uv.lock",
    f"{ROOT}/.gitignore",
    *(f"{ROOT}/{name}" for name in LEGAL_FILES),
    f"{ROOT}/README.md",
    f"{ROOT}/pyproject.toml",
    f"{ROOT}/PKG-INFO",
}

EXPECTED_HASHES = {
    "LICENSE": (
        11358,
        "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30",
    ),
    "NOTICE": (41, "47af3568a74d5c5e3d1a5ae5a29086aa0de962928f52ad344401cd61f74b70a6"),
    "THIRD_PARTY_NOTICES.md": (
        3534,
        "5c139e7e86ea88ce2a1b4930c6c2a25bbc9e26ee1a66aff9bb422453b0755a25",
    ),
    "baseline.toml": (
        303,
        "8df093df5eb8e35f77e0e8c088746b34fe69023f115f89fb822a5682d66cdfb6",
    ),
    "manifest.json": (
        1577,
        "9e72d4238ee0cae7f8236a82e50acf8f811c0e3f7b5e2815a11c56a9e1193c12",
    ),
    "tensors.bin": (
        416,
        "18c2c4333a060ff25b7304dd396cf4b292617c4593d7cbfc2576b406ed5a14bb",
    ),
    "uv.lock": (
        143826,
        "54553a9333abf06be92c7ca11fd1b35b1548ac7f50a7fb196897b4691ac73f5e",
    ),
}

EXPECTED_REQUIREMENTS = {
    "numpy>=2.4.6",
    "pillow>=12.3.0",
    "scikit-learn>=1.9.0",
    "torch>=2.13.0",
    "torchvision>=0.28.0",
    "onnx>=1.22.0; extra == 'onnx'",
    "onnxruntime==1.28.0; extra == 'onnx'",
    "onnxscript>=0.7.1; extra == 'onnx'",
}


class VerificationError(Exception):
    """A distribution violates the reviewed boundary."""


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def validate_member_name(name: str) -> None:
    path = PurePosixPath(name)
    require(not path.is_absolute(), f"absolute archive member: {name}")
    require(name == path.as_posix(), f"non-canonical archive member: {name}")
    require(".." not in path.parts, f"traversal archive member: {name}")


def read_wheel(path: Path) -> dict[str, bytes]:
    require(path.is_file(), f"wheel does not exist: {path}")
    try:
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            require(len(names) == len(set(names)), "wheel contains duplicate members")
            require(
                set(names) == WHEEL_FILES, inventory_error("wheel", names, WHEEL_FILES)
            )
            require(
                sum(info.file_size for info in infos) <= 5_000_000, "wheel is too large"
            )
            for info in infos:
                validate_member_name(info.filename)
                require(not info.is_dir(), f"wheel directory entry: {info.filename}")
                require(
                    not info.flag_bits & 1, f"encrypted wheel member: {info.filename}"
                )
                mode = info.external_attr >> 16
                require(
                    stat.S_IFMT(mode) in {0, stat.S_IFREG},
                    f"non-regular wheel member: {info.filename}",
                )
            return {info.filename: archive.read(info) for info in infos}
    except (OSError, RuntimeError, zipfile.BadZipFile) as error:
        raise VerificationError(f"cannot read wheel {path}: {error}") from error


def read_sdist(path: Path) -> dict[str, bytes]:
    require(path.is_file(), f"sdist does not exist: {path}")
    try:
        with tarfile.open(path, "r:gz") as archive:
            members = archive.getmembers()
            names = [member.name for member in members]
            require(len(names) == len(set(names)), "sdist contains duplicate members")
            require(
                set(names) == SDIST_FILES, inventory_error("sdist", names, SDIST_FILES)
            )
            require(
                sum(member.size for member in members) <= 5_000_000,
                "sdist is too large",
            )
            content: dict[str, bytes] = {}
            for member in members:
                validate_member_name(member.name)
                require(member.isfile(), f"non-regular sdist member: {member.name}")
                stream = archive.extractfile(member)
                require(stream is not None, f"unreadable sdist member: {member.name}")
                content[member.name] = stream.read()
            return content
    except (OSError, tarfile.TarError) as error:
        raise VerificationError(f"cannot read sdist {path}: {error}") from error


def inventory_error(kind: str, actual: list[str], expected: set[str]) -> str:
    actual_set = set(actual)
    missing = sorted(expected - actual_set)
    unexpected = sorted(actual_set - expected)
    return f"{kind} inventory mismatch; missing={missing}, unexpected={unexpected}"


def parse_metadata(data: bytes, label: str):
    message = BytesParser(policy=policy.default).parsebytes(data)
    require(
        not message.defects, f"{label} has metadata parser defects: {message.defects}"
    )
    expected_fields = {
        "Metadata-Version": "2.5",
        "Name": NAME,
        "Version": VERSION,
        "Requires-Python": "<3.13,>=3.11",
        "License-Expression": "Apache-2.0",
    }
    for field, expected in expected_fields.items():
        require(
            message.get_all(field, []) == [expected], f"{label} has invalid {field}"
        )
    license_files = message.get_all("License-File", [])
    require(
        len(license_files) == len(LEGAL_FILES) and set(license_files) == LEGAL_FILES,
        f"{label} has invalid license files",
    )
    require(
        message.get_all("Provides-Extra", []) == ["onnx"], f"{label} has invalid extras"
    )
    requirements = message.get_all("Requires-Dist", [])
    require(
        len(requirements) == len(EXPECTED_REQUIREMENTS)
        and set(requirements) == EXPECTED_REQUIREMENTS,
        f"{label} has invalid requirements",
    )
    return message


def verify_wheel_metadata(content: dict[str, bytes]) -> None:
    parse_metadata(content[f"{DIST_INFO}/METADATA"], "wheel METADATA")
    wheel = BytesParser(policy=policy.default).parsebytes(content[f"{DIST_INFO}/WHEEL"])
    require(not wheel.defects, f"WHEEL has parser defects: {wheel.defects}")
    require(wheel.get_all("Wheel-Version", []) == ["1.0"], "invalid Wheel-Version")
    require(
        wheel.get_all("Root-Is-Purelib", []) == ["true"], "wheel is not pure Python"
    )
    require(wheel.get_all("Tag", []) == ["py3-none-any"], "invalid wheel tag")
    require(
        content[f"{DIST_INFO}/entry_points.txt"]
        == b"[console_scripts]\ninspectrt = inspectrt.cli:main\n",
        "invalid console entry point",
    )


def verify_record(content: dict[str, bytes]) -> None:
    record_name = f"{DIST_INFO}/RECORD"
    try:
        rows = list(csv.reader(io.StringIO(content[record_name].decode("utf-8"))))
    except (UnicodeDecodeError, csv.Error) as error:
        raise VerificationError(f"invalid wheel RECORD: {error}") from error
    require(all(len(row) == 3 for row in rows), "wheel RECORD has malformed rows")
    names = [row[0] for row in rows]
    require(len(names) == len(set(names)), "wheel RECORD contains duplicate members")
    require(set(names) == WHEEL_FILES, "wheel RECORD inventory mismatch")
    for name, hash_field, size_field in rows:
        if name == record_name:
            require(not hash_field and not size_field, "RECORD must not hash itself")
            continue
        expected_hash = base64.urlsafe_b64encode(hashlib.sha256(content[name]).digest())
        expected_hash = expected_hash.rstrip(b"=").decode("ascii")
        require(
            hash_field == f"sha256={expected_hash}", f"RECORD hash mismatch: {name}"
        )
        require(size_field == str(len(content[name])), f"RECORD size mismatch: {name}")


def verify_expected_hashes(wheel: dict[str, bytes], sdist: dict[str, bytes]) -> None:
    paths = {
        "LICENSE": (f"{DIST_INFO}/licenses/LICENSE", f"{ROOT}/LICENSE"),
        "NOTICE": (f"{DIST_INFO}/licenses/NOTICE", f"{ROOT}/NOTICE"),
        "THIRD_PARTY_NOTICES.md": (
            f"{DIST_INFO}/licenses/THIRD_PARTY_NOTICES.md",
            f"{ROOT}/THIRD_PARTY_NOTICES.md",
        ),
        "baseline.toml": (
            "inspectrt/_resources/baseline.toml",
            f"{ROOT}/configs/baseline.toml",
        ),
        "manifest.json": (
            "inspectrt/_resources/retrieval_v1/manifest.json",
            f"{ROOT}/tests/fixtures/retrieval_v1/manifest.json",
        ),
        "tensors.bin": (
            "inspectrt/_resources/retrieval_v1/tensors.bin",
            f"{ROOT}/tests/fixtures/retrieval_v1/tensors.bin",
        ),
    }
    for label, (wheel_path, sdist_path) in paths.items():
        expected_size, expected_hash = EXPECTED_HASHES[label]
        for archive_label, data in (
            ("wheel", wheel[wheel_path]),
            ("sdist", sdist[sdist_path]),
        ):
            require(
                len(data) == expected_size, f"{archive_label} {label} size mismatch"
            )
            require(
                digest(data) == expected_hash, f"{archive_label} {label} hash mismatch"
            )
        require(
            wheel[wheel_path] == sdist[sdist_path], f"{label} differs between archives"
        )
    lock = sdist[f"{ROOT}/uv.lock"]
    expected_size, expected_hash = EXPECTED_HASHES["uv.lock"]
    require(len(lock) == expected_size, "sdist uv.lock size mismatch")
    require(digest(lock) == expected_hash, "sdist uv.lock hash mismatch")


def verify_source_mapping(wheel: dict[str, bytes], sdist: dict[str, bytes]) -> None:
    for wheel_path in PACKAGE_FILES:
        require(
            wheel[wheel_path] == sdist[f"{ROOT}/{wheel_path}"],
            f"package member differs from sdist source: {wheel_path}",
        )
    for wheel_path, source_path in RESOURCE_MAP.items():
        require(
            wheel[wheel_path] == sdist[f"{ROOT}/{source_path}"],
            f"resource differs from sdist source: {wheel_path}",
        )


def verify(wheel_path: Path, sdist_path: Path, rebuilt_path: Path | None) -> None:
    wheel = read_wheel(wheel_path)
    sdist = read_sdist(sdist_path)
    verify_wheel_metadata(wheel)
    verify_record(wheel)
    pkg_info = sdist[f"{ROOT}/PKG-INFO"]
    parse_metadata(pkg_info, "sdist PKG-INFO")
    require(
        wheel[f"{DIST_INFO}/METADATA"] == pkg_info, "wheel and sdist metadata differ"
    )
    verify_expected_hashes(wheel, sdist)
    verify_source_mapping(wheel, sdist)
    if rebuilt_path is not None:
        rebuilt = read_wheel(rebuilt_path)
        verify_wheel_metadata(rebuilt)
        verify_record(rebuilt)
        require(wheel == rebuilt, "sdist-built wheel differs from direct wheel")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("sdist", type=Path)
    parser.add_argument("sdist_wheel", nargs="?", type=Path)
    args = parser.parse_args()
    try:
        verify(args.wheel, args.sdist, args.sdist_wheel)
    except VerificationError as error:
        print(f"distribution verification failed: {error}", file=sys.stderr)
        return 1
    print("distribution verification passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
