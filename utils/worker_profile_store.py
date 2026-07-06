"""
DatraAI Pipeline — Worker Calibration Profile Store (v2 addendum §5)

Worker-linked hand-geometry calibration profiles are real, personally
identifying biometric-like data — a hand's grasp/pinch aperture range is
about as individual as gait, though far coarser than a fingerprint.
Storage, retention, and access here mirror the consent/retention
discipline already established for privacy redaction (v2 addendum §10),
not an afterthought bolted on after the fact: profile creation is
fail-closed on consent (save_profile refuses to write anything without an
explicit consent_granted=True), and profiles expire and are deleted on
read past config.WORKER_PROFILE_RETENTION_DAYS rather than being kept
around indefinitely.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import config as cfg


def profile_path(worker_id: str) -> Path:
    return cfg.CALIBRATION_DIR / f"{worker_id}_profile.json"


def save_profile(worker_id: str, thresholds: dict, consent_granted: bool) -> dict:
    """
    Persist a worker's calibration profile. Fail-closed: refuses to write
    anything without explicit consent — this is biometric-like data, never
    stored on an assumed-yes default (mirrors §10's
    BLOCK_DELIVERY_WITHOUT_CONSENT philosophy).
    """
    if not consent_granted:
        raise PermissionError(
            f"Refusing to store a calibration profile for worker_id={worker_id!r} "
            f"without explicit consent — worker hand-geometry data is real "
            f"biometric-like data (v2 addendum §5/§10)."
        )

    cfg.CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
    profile = {
        "worker_id": worker_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "consent_granted": True,
        **thresholds,
    }
    with open(profile_path(worker_id), "w") as f:
        json.dump(profile, f, indent=2)
    return profile


def load_profile(worker_id: str) -> Optional[dict]:
    """
    Load a worker's calibration profile if it exists AND hasn't expired
    per config.WORKER_PROFILE_RETENTION_DAYS. Returns None (not an
    exception) for both "no profile" and "expired profile" — both are the
    same "fall back to the glove-adjusted or raw default" case for
    callers. An expired profile is deleted on read rather than silently
    kept around — retention is enforced, not just documented.
    """
    path = profile_path(worker_id)
    if not path.exists():
        return None

    with open(path) as f:
        profile = json.load(f)

    created_at = datetime.fromisoformat(profile["created_at"])
    age_days = (datetime.now(timezone.utc) - created_at).days
    if age_days > cfg.WORKER_PROFILE_RETENTION_DAYS:
        path.unlink()
        return None

    return profile


def delete_profile(worker_id: str) -> bool:
    """Explicit deletion (e.g. on a worker's consent-withdrawal request). Returns True if a profile was actually deleted."""
    path = profile_path(worker_id)
    if not path.exists():
        return False
    path.unlink()
    return True
