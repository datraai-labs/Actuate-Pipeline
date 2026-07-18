"""Phase 6 Part F: source-resolver dispatch + the staging helpers. The real hf:// download
is exercised out-of-band (network); here we test dispatch, the URL rewriting, staging, and
the honest errors -- all offline."""

from __future__ import annotations

import numpy as np
import pytest

from actuate import sources
from actuate.sources import resolvers

cv2 = pytest.importorskip("cv2")


def _write_video(path, w=64, h=64, n=3):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (w, h))
    for _ in range(n):
        vw.write(np.zeros((h, w, 3), np.uint8))
    vw.release()


# ---------------------------------------------------------------- dispatch
def test_unknown_scheme_errors():
    with pytest.raises(ValueError, match="unrecognised source scheme"):
        sources.resolve("ftp://nope", "/tmp")


def test_dispatch_routes_by_prefix(monkeypatch, tmp_path):
    calls = {}
    for name, prefix in [("resolve_hf", "hf://x"), ("resolve_s3", "s3://x/y"),
                         ("resolve_http", "https://x/y.mp4"), ("resolve_openx", "openx://x")]:
        monkeypatch.setattr(resolvers, name,
                            lambda spec, wr, _n=name, **k: calls.setdefault(_n, spec) or tmp_path)
    for prefix in ("hf://x", "s3://x/y", "https://x/y.mp4", "openx://x"):
        sources.resolve(prefix, tmp_path)
    assert set(calls) == {"resolve_hf", "resolve_s3", "resolve_http", "resolve_openx"}


# ---------------------------------------------------------------- staging + find
def test_stage_video_creates_session_with_meta(tmp_path):
    vid = tmp_path / "clip.mp4"
    _write_video(vid)
    session = resolvers._stage_video(vid, tmp_path / "sess")
    assert (session / "clip.mp4").exists()
    assert (session / "session_meta.json").exists()


def test_find_video_digs_through_subdirs(tmp_path):
    deep = tmp_path / "videos" / "chunk-000" / "cam"
    deep.mkdir(parents=True)
    _write_video(deep / "episode_000000.mp4")
    assert resolvers._find_video(tmp_path).name == "episode_000000.mp4"


def test_find_video_prefers_non_depth_stream(tmp_path):
    _write_video(tmp_path / "observation.images.depth.mp4")
    _write_video(tmp_path / "observation.images.top.mp4")
    assert "depth" not in resolvers._find_video(tmp_path).name


def test_find_video_errors_when_none(tmp_path):
    with pytest.raises(FileNotFoundError, match="no video"):
        resolvers._find_video(tmp_path)


# ---------------------------------------------------------------- google drive rewrite
def test_google_drive_share_link_becomes_direct_download():
    url = "https://drive.google.com/file/d/ABC123xyz/view?usp=sharing"
    assert resolvers._direct_url(url) == \
        "https://drive.google.com/uc?export=download&id=ABC123xyz"


def test_plain_url_passes_through():
    url = "https://example.com/data/clip.mp4"
    assert resolvers._direct_url(url) == url


def test_url_filename_falls_back_when_pathless():
    assert resolvers._url_filename("https://example.com/download?id=x") == "download.mp4"
    assert resolvers._url_filename("https://example.com/a/clip.mp4") == "clip.mp4"


# ---------------------------------------------------------------- push (write side, no net)
def test_push_to_hub_refuses_a_non_directory(tmp_path):
    with pytest.raises(FileNotFoundError, match="not a directory"):
        sources.push_to_hub(tmp_path / "nope", "user/ds")
