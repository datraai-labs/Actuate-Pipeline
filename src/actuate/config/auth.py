"""User-level auth + defaults, stored at ~/.actuate/config.json (Phase 6 Part C).

Separate from `Settings` (env/AWS wiring): this is the per-user "who am I and what are my
defaults" file the SDK and the simplified CLI read. It lives in the user's HOME, never in the
repo and never in an env var, so a checked-out tree carries no credentials.

Two modes:
- **local** (default): no key needed. Processing runs on this machine. `aws_profile` names
  the AWS profile to use IF a step touches S3; nothing is validated against a server.
- **cloud** (future): an `api_key` authenticates against a DatraAI endpoint for managed
  processing. Scaffolded here but not wired -- `require_cloud()` raises a clear
  "coming soon" so the flow exists without pretending the backend does.

The api_key is WRITE-through-here-only: it is stored to disk and never printed. `redacted()`
is what any display path must use.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

DEFAULT_CONFIG = {
    "mode": "local",                       # "local" | "cloud"
    "aws_profile": None,                   # used only if a step touches S3
    "api_key": None,                       # populated in cloud mode; never printed raw
    "default_embodiment": "franka_panda",
    "default_export_format": "lerobot_v3",
}

#: Keys a user may set via `actuate config set`.
SETTABLE = ("mode", "aws_profile", "default_embodiment", "default_export_format")


def config_dir() -> Path:
    """~/.actuate (override with ACTUATE_HOME for tests / non-standard homes)."""
    root = os.environ.get("ACTUATE_HOME")
    return Path(root) if root else Path.home() / ".actuate"


def config_path() -> Path:
    return config_dir() / "config.json"


def load_config() -> dict:
    """The stored config merged over the defaults (so new keys appear even on old files)."""
    p = config_path()
    cfg = dict(DEFAULT_CONFIG)
    if p.exists():
        try:
            cfg.update(json.loads(p.read_text(encoding="utf-8")))
        except (ValueError, OSError):
            pass  # corrupt file -> fall back to defaults rather than crash
    return cfg


def save_config(cfg: dict) -> Path:
    d = config_dir()
    d.mkdir(parents=True, exist_ok=True)
    p = config_path()
    p.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    return p


def login(*, mode: str = "local", api_key: str | None = None,
          aws_profile: str | None = None) -> dict:
    """Write auth config. `mode='local'` needs no key; `mode='cloud'` stores the key.

    Returns the saved config (with the key present on disk but callers must `redacted()`
    before display).
    """
    if mode not in ("local", "cloud"):
        raise ValueError(f"mode must be 'local' or 'cloud', got {mode!r}")
    cfg = load_config()
    cfg["mode"] = mode
    if aws_profile is not None:
        cfg["aws_profile"] = aws_profile
    if mode == "cloud":
        if not api_key:
            raise ValueError("cloud mode needs an api_key")
        cfg["api_key"] = api_key
    save_config(cfg)
    return cfg


def is_authenticated() -> bool:
    """Local mode is always 'authenticated' (it's your own machine); cloud needs a key."""
    cfg = load_config()
    return cfg["mode"] == "local" or bool(cfg.get("api_key"))


def require_cloud() -> None:
    """Gate for managed/cloud features -- scaffolded, not live."""
    raise NotImplementedError(
        "Cloud features (remote processing, usage tracking, dataset hosting) are coming "
        "soon. Local processing needs no API key: run `actuate login --local`.")


def set_default(key: str, value: str) -> dict:
    if key not in SETTABLE:
        raise ValueError(f"cannot set {key!r}; settable keys: {', '.join(SETTABLE)}")
    cfg = load_config()
    cfg[key] = value
    save_config(cfg)
    return cfg


def redacted(cfg: dict | None = None) -> dict:
    """Config safe to print -- the api_key becomes a masked marker, never the value."""
    cfg = dict(cfg or load_config())
    if cfg.get("api_key"):
        cfg["api_key"] = "set (hidden)"
    return cfg
