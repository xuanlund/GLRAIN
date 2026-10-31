"""Verify every released file against SHA256SUMS.txt."""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "SHA256SUMS.txt"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    failures = []
    checked = 0
    for line in MANIFEST.read_text(encoding="ascii").splitlines():
        expected, relative = line.split("  ", 1)
        path = ROOT / relative
        actual = sha256(path) if path.is_file() else "missing"
        checked += 1
        if actual != expected:
            failures.append((relative, expected, actual))
    if failures:
        details = "\n".join(f"{path}: expected {exp}, found {got}" for path, exp, got in failures)
        raise RuntimeError(f"Release verification failed:\n{details}")
    print(f"PASS: {checked} release files match SHA256SUMS.txt")


if __name__ == "__main__":
    main()
