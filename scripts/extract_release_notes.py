"""Extract one version section from the InspectRT changelog."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


def extract(text: str, version: str) -> str:
    if not version or any(character in version for character in "[]\r\n"):
        raise ValueError("invalid version")
    lines = text.splitlines()
    heading = f"## [{version}]"
    matches = [index for index, line in enumerate(lines) if line == heading]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {heading!r} section")
    start = matches[0] + 1
    end = next(
        (index for index in range(start, len(lines)) if lines[index].startswith("## ")),
        len(lines),
    )
    body = "\n".join(lines[start:end]).strip()
    if not body:
        raise ValueError(f"{heading!r} section is empty")
    return f"{body}\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("changelog", type=Path)
    parser.add_argument("version")
    args = parser.parse_args()
    try:
        notes = extract(args.changelog.read_text(encoding="utf-8"), args.version)
    except (OSError, UnicodeError, ValueError) as error:
        parser.error(str(error))
    sys.stdout.write(notes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
