"""
DatraAI Pipeline — Tests for utils/video_utils.py's resolve_perception_source
(v2 addendum §8: PERCEPTION_SOURCE toggle)

These tests prove the toggle actually changes which file gets resolved —
asserting on the resolved Path returned by the function, not merely that
the config constants exist. Same "wiring does what it claims" standard as
§2/§5's glove/calibration tests.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config as cfg
from utils.video_utils import resolve_perception_source


SESSION_ID = "session_toggle_test"


def _make_compressed(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "PROCESSED_DIR", tmp_path / "processed")
    proc_dir = cfg.PROCESSED_DIR / SESSION_ID
    proc_dir.mkdir(parents=True)
    compressed_path = proc_dir / "compressed.mp4"
    compressed_path.write_bytes(b"fake-compressed")
    return compressed_path


def _make_raw(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "RAW_DIR", tmp_path / "raw")
    raw_dir = cfg.RAW_DIR / SESSION_ID
    raw_dir.mkdir(parents=True)
    raw_path = raw_dir / "raw.mp4"
    raw_path.write_bytes(b"fake-raw")
    return raw_path


class TestResolvePerceptionSource:
    def test_compressed_source_resolves_to_compressed_path(self, tmp_path, monkeypatch):
        compressed_path = _make_compressed(tmp_path, monkeypatch)
        monkeypatch.setattr(cfg, "PERCEPTION_SOURCE", "compressed")

        resolved = resolve_perception_source(SESSION_ID)

        assert resolved == compressed_path
        assert resolved.name == "compressed.mp4"

    def test_raw_source_resolves_to_raw_path(self, tmp_path, monkeypatch):
        raw_path = _make_raw(tmp_path, monkeypatch)
        monkeypatch.setattr(cfg, "PERCEPTION_SOURCE", "raw")

        resolved = resolve_perception_source(SESSION_ID)

        assert resolved == raw_path
        assert resolved.name == "raw.mp4"

    def test_flipping_the_toggle_changes_the_resolved_path(self, tmp_path, monkeypatch):
        """
        The core §8 requirement: flipping PERCEPTION_SOURCE between "raw"
        and "compressed" must change which file gets opened — not just
        which config value is set.
        """
        compressed_path = _make_compressed(tmp_path, monkeypatch)
        raw_path = _make_raw(tmp_path, monkeypatch)

        monkeypatch.setattr(cfg, "PERCEPTION_SOURCE", "compressed")
        resolved_compressed = resolve_perception_source(SESSION_ID)

        monkeypatch.setattr(cfg, "PERCEPTION_SOURCE", "raw")
        resolved_raw = resolve_perception_source(SESSION_ID)

        assert resolved_compressed == compressed_path
        assert resolved_raw == raw_path
        assert resolved_compressed != resolved_raw

    def test_missing_compressed_file_raises_file_not_found(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cfg, "PROCESSED_DIR", tmp_path / "processed")
        monkeypatch.setattr(cfg, "PERCEPTION_SOURCE", "compressed")

        with pytest.raises(FileNotFoundError):
            resolve_perception_source(SESSION_ID)

    def test_missing_raw_file_raises_file_not_found(self, tmp_path, monkeypatch):
        """
        e.g. PERCEPTION_SOURCE="raw" after raw.mp4 was deleted
        (KEEP_RAW_AFTER_COMPRESSION=False) — must fail loudly, not
        silently fall back to compressed.mp4.
        """
        monkeypatch.setattr(cfg, "RAW_DIR", tmp_path / "raw")
        monkeypatch.setattr(cfg, "PERCEPTION_SOURCE", "raw")

        with pytest.raises(FileNotFoundError):
            resolve_perception_source(SESSION_ID)

    def test_unknown_source_raises_value_error(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cfg, "PERCEPTION_SOURCE", "bogus")

        with pytest.raises(ValueError):
            resolve_perception_source(SESSION_ID)
