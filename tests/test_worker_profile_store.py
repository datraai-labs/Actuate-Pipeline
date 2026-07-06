"""
DatraAI Pipeline — Tests for utils/worker_profile_store.py (v2 addendum §5)
"""

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config as cfg
from utils.worker_profile_store import save_profile, load_profile, delete_profile, profile_path


class TestSaveProfileConsentGate:
    def test_refuses_to_save_without_consent(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cfg, "CALIBRATION_DIR", tmp_path)
        with pytest.raises(PermissionError):
            save_profile("worker_001", {"power_grasp_dist": 0.18}, consent_granted=False)
        # Fail-closed must mean nothing was written at all.
        assert not profile_path("worker_001").exists()

    def test_saves_with_explicit_consent(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cfg, "CALIBRATION_DIR", tmp_path)
        profile = save_profile("worker_001", {"power_grasp_dist": 0.18, "lateral_pinch_dist": 0.09}, consent_granted=True)
        assert profile["worker_id"] == "worker_001"
        assert profile["consent_granted"] is True
        assert profile["power_grasp_dist"] == 0.18
        assert profile_path("worker_001").exists()


class TestLoadProfile:
    def test_missing_profile_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cfg, "CALIBRATION_DIR", tmp_path)
        assert load_profile("nonexistent_worker") is None

    def test_loads_fresh_profile(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cfg, "CALIBRATION_DIR", tmp_path)
        save_profile("worker_002", {"power_grasp_dist": 0.20}, consent_granted=True)
        profile = load_profile("worker_002")
        assert profile is not None
        assert profile["power_grasp_dist"] == 0.20

    def test_expired_profile_returns_none_and_is_deleted(self, tmp_path, monkeypatch):
        """Retention must be ENFORCED (deleted on read past the window), not just documented."""
        monkeypatch.setattr(cfg, "CALIBRATION_DIR", tmp_path)
        save_profile("worker_003", {"power_grasp_dist": 0.20}, consent_granted=True)

        # Backdate created_at past the retention window.
        import json
        path = profile_path("worker_003")
        with open(path) as f:
            profile = json.load(f)
        stale_date = datetime.now(timezone.utc) - timedelta(days=cfg.WORKER_PROFILE_RETENTION_DAYS + 1)
        profile["created_at"] = stale_date.isoformat()
        with open(path, "w") as f:
            json.dump(profile, f)

        assert path.exists()  # sanity: the file genuinely exists before the expiry check
        result = load_profile("worker_003")
        assert result is None
        assert not path.exists(), "expired profile must be deleted on read, not just ignored"

    def test_profile_just_under_retention_window_still_loads(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cfg, "CALIBRATION_DIR", tmp_path)
        save_profile("worker_004", {"power_grasp_dist": 0.20}, consent_granted=True)

        import json
        path = profile_path("worker_004")
        with open(path) as f:
            profile = json.load(f)
        fresh_date = datetime.now(timezone.utc) - timedelta(days=cfg.WORKER_PROFILE_RETENTION_DAYS - 1)
        profile["created_at"] = fresh_date.isoformat()
        with open(path, "w") as f:
            json.dump(profile, f)

        result = load_profile("worker_004")
        assert result is not None
        assert path.exists()


class TestDeleteProfile:
    def test_delete_existing_profile_returns_true(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cfg, "CALIBRATION_DIR", tmp_path)
        save_profile("worker_005", {"power_grasp_dist": 0.20}, consent_granted=True)
        assert delete_profile("worker_005") is True
        assert not profile_path("worker_005").exists()

    def test_delete_nonexistent_profile_returns_false(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cfg, "CALIBRATION_DIR", tmp_path)
        assert delete_profile("no_such_worker") is False
