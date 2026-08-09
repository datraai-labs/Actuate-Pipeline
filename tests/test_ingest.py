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
    """Executes the PRODUCTION _parse_imu (the old version of this test copied the
    parsing logic into the test body — audit 2026-08-01 Group 3/4)."""

    def _write_json(self, session_dir, with_mag=True):
        sample_data = []
        for i in range(150):
            record = {
                "timestamp_ns": (1000000000 + i * 5000000),  # 200 Hz
                "accel": [0.1 * i, 0.2 * i, 0.3 * i],
                "gyro": [0.01 * i, 0.02 * i, 0.03 * i],
            }
            if with_mag:
                record["mag"] = [1.0, 2.0, 3.0]
                record["temp_c"] = 40.0
            sample_data.append(record)
        path = session_dir / "imu.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(sample_data, f)
        return path

    def test_parse_imu_json_keeps_all_measured_channels(self):
        """audit Group 4: the old parse coerced to 7 columns and silently DROPPED
        magnetometer + temperature — the two ingest paths disagreed about what an
        IMU is. Confirmed RED against the pre-fix inline parse."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            session_dir = Path(tmp_dir)
            imu_json = self._write_json(session_dir, with_mag=True)

            df = _ingest_mod2._parse_imu(imu_json, session_dir / "imu.csv")

            assert len(df) == 150
            assert df["epoch_ms"].iloc[0] == 1000.0
            assert df["ax"].iloc[1] == 0.1
            assert df["mx"].iloc[0] == 1.0        # kept, not dropped
            assert df["temp_c"].iloc[0] == 40.0   # kept, not dropped

    def test_absent_optional_channels_are_nan_not_zero(self):
        """NaN means NOT MEASURED; zeros would claim 'measured, field-free space'."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            session_dir = Path(tmp_dir)
            imu_json = self._write_json(session_dir, with_mag=False)

            df = _ingest_mod2._parse_imu(imu_json, session_dir / "imu.csv")

            assert len(df) == 150                  # rows NOT dropped by NaN optionals
            assert np.isnan(df["mx"]).all()
            assert np.isnan(df["temp_c"]).all()


class TestIMUSidecarAmbiguity:
    """audit 2026-08-01 Group 1: the legacy single-IMU path silently preferred
    imu.json when imu.csv also existed, and ignored imu_* extras entirely."""

    def test_both_json_and_csv_fail_loudly_before_any_processing(self, tmp_path):
        session = tmp_path / "session_x"
        session.mkdir()
        (session / "raw.mp4").write_bytes(b"\x00" * 128)
        (session / "imu.json").write_text("[]")
        (session / "imu.csv").write_text("epoch_ms,ax,ay,az,gx,gy,gz\n")

        with pytest.raises(ValueError) as excinfo:
            _ingest_mod2.run(session)
        message = str(excinfo.value)
        assert "imu.json" in message and "imu.csv" in message
        # raised before compression: no processed output may exist
        assert not (cfg.PROCESSED_DIR / "session_x").exists()

    def test_extra_imu_sidecar_fails_loudly(self, tmp_path):
        session = tmp_path / "session_y"
        session.mkdir()
        (session / "raw.mp4").write_bytes(b"\x00" * 128)
        (session / "imu.json").write_text("[]")
        (session / "imu_wrist.json").write_text("[]")

        with pytest.raises(ValueError, match="imu_wrist.json"):
            _ingest_mod2.run(session)


class TestTemporalAnchorAssessment:
    """audit 2026-08-01 Group 2: the IMU-t0 fallback anchor is tautological and the
    old code had nothing that could catch a wrong or cross-domain anchor."""

    _NOW_MS = 1_780_000_000_000.0  # ~2026, fixed so tests don't depend on wall clock

    def _assess(self, creation_ms, imu_t0, imu_t1, duration_s=95.0):
        return _ingest_mod2._assess_temporal_anchor(
            creation_ms, imu_t0, imu_t1, duration_s, now_epoch_ms=self._NOW_MS
        )

    def test_fallback_is_marked_unvalidated_not_clean(self):
        """The real corpus's case: no creation_time, uptime IMU clock."""
        anchor, fields = self._assess(None, 128_642.7, 223_640.6)
        assert anchor == 128_642.7
        assert fields["imu_clock_domain"] == "uptime"
        assert fields["temporal_alignment_validated"] is False
        assert "by construction" in fields["temporal_alignment_note"]

    def test_epoch_imu_with_overlapping_creation_time_validates(self):
        t0 = 1_750_000_000_000.0
        anchor, fields = self._assess(t0 + 500.0, t0, t0 + 95_000.0)
        assert anchor == t0 + 500.0
        assert fields["imu_clock_domain"] == "epoch"
        assert fields["temporal_alignment_validated"] is True

    def test_creation_time_against_uptime_imu_refuses_to_anchor(self):
        """A wall-clock video anchor + uptime IMU clock cannot be aligned; the old
        code would have produced a ~56-year drift instead of an explanation."""
        with pytest.raises(ValueError, match="[Cc]lock-domain"):
            self._assess(1_750_000_000_000.0, 128_642.7, 223_640.6)

    def test_disjoint_epoch_ranges_are_a_provably_wrong_anchor(self):
        t0 = 1_750_000_000_000.0
        with pytest.raises(ValueError, match="do not overlap"):
            self._assess(t0 + 10_000_000.0, t0, t0 + 95_000.0)

    def test_implausible_creation_time_is_rejected_to_fallback(self):
        """A 1970-epoch creation_time (zeroed container clock) must not become the
        anchor; it falls back, still marked unvalidated, with the rejection noted."""
        anchor, fields = self._assess(10_000.0, 128_642.7, 223_640.6)
        assert anchor == 128_642.7
        assert fields["temporal_alignment_validated"] is False
        assert "not a plausible wall-clock" in fields["temporal_alignment_note"]


class TestTimestampSemantics:
    def test_defaults_are_none_meaning_unknown_not_zero(self, tmp_path):
        semantics = _ingest_mod2._read_timestamp_semantics(tmp_path)
        assert semantics["video_frame_time_meaning"] is None
        assert semantics["imu_sample_time_meaning"] is None
        assert semantics["camera_to_imu_latency_ms"] is None

    def test_declared_semantics_are_read_from_session_config(self, tmp_path):
        (tmp_path / "session_config.json").write_text(json.dumps({
            "timestamp_semantics": {
                "video_frame_time_meaning": "start_of_exposure",
                "camera_to_imu_latency_ms": 18.5,
            }
        }))
        semantics = _ingest_mod2._read_timestamp_semantics(tmp_path)
        assert semantics["video_frame_time_meaning"] == "start_of_exposure"
        assert semantics["camera_to_imu_latency_ms"] == 18.5
        assert semantics["imu_sample_time_meaning"] is None  # still unknown


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
