"""Small, dependency-light version lineage for every durable artifact."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
from pathlib import Path


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_state() -> tuple[str | None, bool | None]:
    root = Path(__file__).resolve().parents[2]
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True,
            capture_output=True, text=True, timeout=2,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=root,
            check=True, capture_output=True, text=True, timeout=2,
        ).stdout.strip())
        return sha, dirty
    except (OSError, subprocess.SubprocessError):
        # Wheels, source archives, and managed GPU uploads normally omit `.git`.  The build
        # harness can still provide immutable provenance explicitly; never accept an
        # arbitrary label in a field that claims to be a Git object ID.
        sha = os.getenv("ACTUATE_BUILD_GIT_SHA", "").strip().lower()
        if not re.fullmatch(r"[0-9a-f]{40}(?:[0-9a-f]{24})?", sha):
            return None, None
        dirty_value = os.getenv("ACTUATE_BUILD_GIT_DIRTY", "").strip().lower()
        dirty = {"true": True, "1": True, "false": False, "0": False}.get(dirty_value)
        return sha, dirty


def build_lineage(profile: dict | None = None) -> dict:
    profile = profile or {}
    encoded = json.dumps(profile, sort_keys=True, default=str).encode("utf-8")
    sha, dirty = _git_state()
    return {
        "git_sha": sha,
        "git_dirty": dirty,
        "actuate_version": _package_version("actuate"),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "profile_sha256": hashlib.sha256(encoded).hexdigest(),
        "dependencies": {
            name: version for name in ("torch", "unidepth", "lerobot", "tensorflow")
            if (version := _package_version(name)) is not None
        },
    }
