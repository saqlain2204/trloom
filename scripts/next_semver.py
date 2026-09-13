"""Compute the next SemVer tag from existing git tags.

Bump kinds: patch (default), minor, major.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys


_TAG_RE = re.compile(r"^v?(\d+)\.(\d+)\.(\d+)$")


def list_version_tags(tags: list[str]) -> list[tuple[int, int, int]]:
    versions: list[tuple[int, int, int]] = []
    for tag in tags:
        match = _TAG_RE.match(tag.strip())
        if match:
            versions.append(tuple(int(part) for part in match.groups()))  # type: ignore[arg-type]
    return sorted(versions)


def latest_version(tags: list[str]) -> tuple[int, int, int] | None:
    versions = list_version_tags(tags)
    return versions[-1] if versions else None


def next_version(
    current: tuple[int, int, int] | None,
    kind: str,
) -> tuple[int, int, int]:
    if current is None:
        return (0, 1, 0)
    major, minor, patch = current
    if kind == "major":
        return (major + 1, 0, 0)
    if kind == "minor":
        return (major, minor + 1, 0)
    if kind == "patch":
        return (major, minor, patch + 1)
    raise ValueError(f"Unknown bump kind: {kind}")


def format_tag(version: tuple[int, int, int]) -> str:
    return f"v{version[0]}.{version[1]}.{version[2]}"


def git_tags() -> list[str]:
    result = subprocess.run(
        ["git", "tag", "-l", "v*.*.*"],
        check=True,
        capture_output=True,
        text=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bump",
        choices=("patch", "minor", "major"),
        default="patch",
        help="SemVer bump kind (default: patch).",
    )
    parser.add_argument(
        "--tags",
        nargs="*",
        default=None,
        help="Optional explicit tag list (defaults to git tag -l 'v*.*.*').",
    )
    args = parser.parse_args(argv)

    tags = args.tags if args.tags is not None else git_tags()
    nxt = next_version(latest_version(tags), args.bump)
    print(format_tag(nxt), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
