"""Verify an exact pytest JUnit result and its authorized skips."""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path, PurePosixPath


class VerificationError(Exception):
    """A JUnit report violates the expected result."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def integer_attribute(element: ET.Element, name: str) -> int:
    value = element.get(name)
    require(value is not None and value.isdecimal(), f"invalid testsuite {name}")
    return int(value)


def node_id(case: ET.Element) -> str:
    file_name = case.get("file", "")
    class_name = case.get("classname", "")
    test_name = case.get("name", "")
    path = PurePosixPath(file_name)
    require(
        file_name == path.as_posix()
        and not path.is_absolute()
        and ".." not in path.parts
        and path.suffix == ".py",
        f"invalid testcase file: {file_name!r}",
    )
    require(
        test_name and "::" not in test_name, f"invalid testcase name: {test_name!r}"
    )
    module = ".".join(path.with_suffix("").parts)
    require(
        class_name == module or class_name.startswith(f"{module}."),
        f"invalid testcase classname: {class_name!r}",
    )
    class_parts = class_name[len(module) :].lstrip(".").split(".")
    components = [file_name, *(part for part in class_parts if part), test_name]
    return "::".join(components)


def verify(path: Path, expected_tests: int, expected_skips: set[str]) -> None:
    require(path.is_file(), f"report does not exist: {path}")
    require(path.stat().st_size <= 10_000_000, "report is too large")
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError) as error:
        raise VerificationError(f"cannot parse report: {error}") from error

    if root.tag == "testsuite":
        suite = root
    else:
        require(root.tag == "testsuites", f"unexpected XML root: {root.tag}")
        suites = root.findall("testsuite")
        require(
            len(suites) == 1 and len(root) == 1, "report must contain one testsuite"
        )
        suite = suites[0]

    require(
        integer_attribute(suite, "tests") == expected_tests, "unexpected test count"
    )
    require(integer_attribute(suite, "failures") == 0, "test failures are present")
    require(integer_attribute(suite, "errors") == 0, "test errors are present")
    require(
        integer_attribute(suite, "skipped") == len(expected_skips),
        "unexpected skip count",
    )
    if suite.get("disabled") is not None:
        require(integer_attribute(suite, "disabled") == 0, "disabled tests are present")

    cases = suite.findall("testcase")
    require(
        len(cases) == expected_tests, "testcase count does not match testsuite total"
    )
    all_ids = [node_id(case) for case in cases]
    require(len(all_ids) == len(set(all_ids)), "duplicate testcase node IDs")

    actual_skips: set[str] = set()
    for case, identifier in zip(cases, all_ids, strict=True):
        failures = case.findall("failure")
        errors = case.findall("error")
        skips = case.findall("skipped")
        require(not failures and not errors, f"failed testcase: {identifier}")
        require(len(skips) <= 1, f"duplicate skip result: {identifier}")
        if skips:
            require(
                skips[0].get("type") == "pytest.skip", f"non-pytest skip: {identifier}"
            )
            require(
                identifier not in actual_skips,
                f"duplicate skipped node ID: {identifier}",
            )
            actual_skips.add(identifier)

    missing = sorted(expected_skips - actual_skips)
    unexpected = sorted(actual_skips - expected_skips)
    require(
        not missing and not unexpected,
        f"skip mismatch; missing={missing}, unexpected={unexpected}",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--tests", required=True, type=int)
    parser.add_argument("--skip", action="append", default=[])
    args = parser.parse_args()
    try:
        require(args.tests >= 0, "expected test count must be non-negative")
        require(len(args.skip) == len(set(args.skip)), "duplicate expected skip")
        verify(args.report, args.tests, set(args.skip))
    except VerificationError as error:
        print(f"pytest report verification failed: {error}", file=sys.stderr)
        return 1
    print("pytest report verification passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
