"""S3 upload of processed runs (mocked backend -- no AWS): raw + canonical + export tree go
to the right buckets/keys, and clean_local removes the local copies."""

from __future__ import annotations

from pathlib import Path

from actuate.cloud import upload_run
from actuate.config.settings import Bucket


class _FakeBackend:
    def __init__(self):
        self.puts = []           # (bucket, key, src)

    def put_file(self, bucket, key, src):
        self.puts.append((bucket, key, Path(src)))
        return f"s3://actuate-{bucket.value}-dev/{key}"


def _run(tmp_path):
    canon = tmp_path / "canonical.json"
    canon.write_text("{}")
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"vid")
    export = tmp_path / "ds"
    (export / "meta").mkdir(parents=True)
    (export / "meta" / "info.json").write_text("{}")
    (export / "data.parquet").write_bytes(b"data")
    return canon, video, export


def test_upload_run_targets_correct_buckets(tmp_path):
    canon, video, export = _run(tmp_path)
    be = _FakeBackend()
    uris = upload_run(canon, "cap123", "ep00", video=video, export_dirs=[export],
                      backend=be, clean_local=False)
    buckets = {b for b, _, _ in be.puts}
    assert Bucket.RAW in buckets and Bucket.WORK in buckets
    assert uris["raw"].startswith("s3://actuate-raw-dev/cap123/")
    assert "canonical/ep00/canonical.json" in uris["canonical"]
    # export tree uploaded (both files under the export dir)
    assert len(uris["exports"]["ds"]) == 2
    keys = [k for _, k, _ in be.puts]
    assert any("ds/meta/info.json" in k for k in keys)


def test_clean_local_removes_copies(tmp_path):
    canon, video, export = _run(tmp_path)
    upload_run(canon, "cap", "ep", video=video, export_dirs=[export],
               backend=_FakeBackend(), clean_local=True)
    assert not canon.exists()          # canonical removed
    assert not export.exists()         # export tree removed
    assert video.exists()              # the user's INPUT video is never deleted


def test_upload_skips_missing_raw(tmp_path):
    canon = tmp_path / "canonical.json"
    canon.write_text("{}")
    be = _FakeBackend()
    uris = upload_run(canon, "cap", "ep", video=None, backend=be)
    assert "raw" not in uris            # no raw video -> not uploaded, not an error
    assert "canonical" in uris


def test_export_cache_dirs_are_skipped(tmp_path):
    canon = tmp_path / "canonical.json"
    canon.write_text("{}")
    export = tmp_path / "ds"
    (export / ".actuate_cache").mkdir(parents=True)
    (export / ".actuate_cache" / "junk.pkl").write_bytes(b"x")
    (export / "keep.json").write_text("{}")
    be = _FakeBackend()
    upload_run(canon, "cap", "ep", export_dirs=[export], backend=be)
    keys = [k for _, k, _ in be.puts]
    assert not any(".actuate_cache" in k for k in keys)   # scratch not uploaded
    assert any("keep.json" in k for k in keys)
