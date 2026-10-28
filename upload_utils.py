"""Small, deterministic helpers for safer document uploads."""
from __future__ import annotations

import hashlib
from pathlib import Path


def content_sha256(payload: bytes) -> str:
    """Return a stable SHA-256 digest for upload content."""
    return hashlib.sha256(payload).hexdigest()


def unique_destination(directory: Path, filename: str, payload: bytes) -> tuple[Path, bool]:
    """Return a safe destination and whether identical content already exists.

    Same-content files are detected by hash across the directory. A same-name
    different-content file receives a numeric suffix rather than being overwritten.
    """
    directory.mkdir(parents=True, exist_ok=True)
    digest = content_sha256(payload)
    for existing in directory.iterdir():
        if existing.is_file():
            try:
                if content_sha256(existing.read_bytes()) == digest:
                    return existing, True
            except OSError:
                continue
    candidate = directory / Path(filename).name
    if not candidate.exists():
        return candidate, False
    stem, suffix = candidate.stem, candidate.suffix
    counter = 2
    while (directory / f"{stem}_{counter}{suffix}").exists():
        counter += 1
    return directory / f"{stem}_{counter}{suffix}", False
