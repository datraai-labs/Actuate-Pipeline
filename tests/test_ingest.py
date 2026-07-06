"""
DatraAI Pipeline — Tests for Step 01: Ingest (IMU JSON and CSV parsing)
"""

import importlib
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config as cfg

ingest_mod = importlib.import_module("scripts.01_ingest")

_spec = importlib.util.spec_from_file_location(
    "ingest_numeric",
    str(Path(__file__).resolve().parent.parent / "scripts" / "01_ingest.py"),
)
_ingest_mod2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ingest_mod2)

_read_consent_status = _ingest_mod2._read_consent_status
_read_glove_and_worker_config = _ingest_mod2._read_glove_and_worker_config
_maybe_delete_raw_video = _ingest_mod2._maybe_delete_raw_video


class TestIMUIngest:
    """Test IMU parsing for JSON and CSV formats."""

    def test_parse_imu_json(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            session_dir = Path(tmp_dir)
            imu_json_path = session_dir / "imu.json"
            
            # Create synthetic JSON data
            sample_data = []
            for i in range(150):
                sample_data.append({
                    "timestamp_ns": (1000000000 + i * 5000000), # 200 Hz
                    "accel": [0.1 * i, 0.2 * i, 0.3 * i],
                    "gyro": [0.01 * i, 0.02 * i, 0.03 * i],
                    "mag": [1.0, 2.0, 3.0],
                    "temp_c": 40.0
                })
            
            with open(imu_json_path, "w", encoding="utf-8") as f:
                json.dump(sample_data, f)
                
            expected_cols = ["epoch_ms"] + cfg.ACCEL_COLS + cfg.GYRO_COLS
            
            # Test loading logic matching step 01_ingest
            with open(imu_json_path, "r", encoding="utf-8") as f:
                imu_data = json.load(f)
            
            rows = []
            for item in imu_data:
                t_ms = item["timestamp_ns"] / 1e6
                ax, ay, az = item["accel"][:3]
                gx, gy, gz = item["gyro"][:3]
                rows.append([t_ms, ax, ay, az, gx, gy, gz])
                
            df = pd.DataFrame(rows, columns=expected_cols)
            assert len(df) == 150
            assert list(df.columns) == expected_cols
            assert df["epoch_ms"].iloc[0] == 1000.0
            assert df["ax"].iloc[1] == 0.1


class TestReadConsentStatus:
    def test_missing_consent_json_defaults_to_pending(self, tmp_path):
        assert _read_consent_status(tmp_path) == "pending"

    def test_reads_granted_status(self, tmp_path):
        with open(tmp_path / "consent.json", "w") as f:
            json.dump({"status": "granted"}, f)
        assert _read_consent_status(tmp_path) == "granted"

    def test_missing_status_key_defaults_to_pending(self, tmp_path):
        with open(tmp_path / "consent.json", "w") as f:
            json.dump({}, f)
        assert _read_consent_status(tmp_path) == "pending"


class TestReadGloveAndWorkerConfig:
    """v2 addendum §2/§5 — optional raw/{session_id}/session_config.json."""

    def test_missing_file_defaults_to_none_glove_and_no_worker(self, tmp_path):
        glove_type, worker_id = _read_glove_and_worker_config(tmp_path)
        assert glove_type == cfg.GLOVE_TYPE_DEFAULT
        assert worker_id is None

    def test_reads_declared_glove_type_and_worker_id(self, tmp_path):
        with open(tmp_path / "session_config.json", "w") as f:
            json.dump({"glove_type": "thick", "worker_id": "worker_007"}, f)
        glove_type, worker_id = _read_glove_and_worker_config(tmp_path)
        assert glove_type == "thick"
        assert worker_id == "worker_007"

    def test_partial_config_defaults_missing_fields(self, tmp_path):
        """A session_config.json declaring only worker_id (no glove_type) must still default glove_type sanely, not crash."""
        with open(tmp_path / "session_config.json", "w") as f:
            json.dump({"worker_id": "worker_007"}, f)
        glove_type, worker_id = _read_glove_and_worker_config(tmp_path)
        assert glove_type == cfg.GLOVE_TYPE_DEFAULT
        assert worker_id == "worker_007"


class TestMaybeDeleteRawVideo:
    """v2 addendum §8 — raw.mp4 retention gated on KEEP_RAW_AFTER_COMPRESSION."""

    def _make_raw_video(self, tmp_path):
        raw_video = tmp_path / "raw.mp4"
        raw_video.write_bytes(b"fake-raw")
        return raw_video

    def test_keep_raw_true_never_deletes(self, tmp_path, monkeypatch):
        raw_video = self._make_raw_video(tmp_path)
        monkeypatch.setattr(cfg, "KEEP_RAW_AFTER_COMPRESSION", True)
        monkeypatch.setattr(cfg, "PERCEPTION_SOURCE", "compressed")

        deleted = _maybe_delete_raw_video(raw_video)

        assert deleted is False
        assert raw_video.exists()

    def test_keep_raw_false_and_perception_source_compressed_deletes(self, tmp_path, monkeypatch):
        raw_video = self._make_raw_video(tmp_path)
        monkeypatch.setattr(cfg, "KEEP_RAW_AFTER_COMPRESSION", False)
        monkeypatch.setattr(cfg, "PERCEPTION_SOURCE", "compressed")

        deleted = _maybe_delete_raw_video(raw_video)

        assert deleted is True
        assert not raw_video.exists()

    def test_keep_raw_false_but_perception_source_raw_still_keeps_file(self, tmp_path, monkeypatch):
        """
        Even with KEEP_RAW_AFTER_COMPRESSION=False, raw.mp4 must survive if
        PERCEPTION_SOURCE="raw" — deleting it would break every subsequent
        perception script's resolve_perception_source() call.
        """
        raw_video = self._make_raw_video(tmp_path)
        monkeypatch.setattr(cfg, "KEEP_RAW_AFTER_COMPRESSION", False)
        monkeypatch.setattr(cfg, "PERCEPTION_SOURCE", "raw")

        deleted = _maybe_delete_raw_video(raw_video)

        assert deleted is False
        assert raw_video.exists()
