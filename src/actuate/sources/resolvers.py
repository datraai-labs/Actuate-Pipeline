"""Source resolvers -- turn a remote source spec into a local session directory (Part F).

Each resolver downloads/copies to a working dir, locates a processable video, stages it as a
session directory (video + auto session_meta, filenames normalised), and returns that path.
The pipeline then runs perception on the staged video -- a LeRobot/RLDS dataset is RE-PROCESSED
from its raw video, not blindly trusted.

Honesty labels (what actually ran):
- hf://    verified end-to-end on the real public `lerobot/aloha_static_battery_ep009`.
- http(s)://  verified on a direct URL (+ Google-Drive share-link rewriting).
- s3://    implemented via fsspec/s3fs; WRITTEN-ONLY (no deployed bucket to verify against).
- openx:// implemented via tfds; tested on a tiny slice (full Open-X sets are 100s of GB).
"""

from __future__ import annotations

import shutil
from pathlib import Path

_VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".webm")


def _stage_video(video: Path, session: Path) -> Path:
    """Stage a source as the session layout consumed by every perception backend.

    Remote datasets often keep calibration/IMU JSON beside the video, so retain those
    sidecars at this boundary. The shared ingest video resolver lets downstream stages
    accept the source filename without manufacturing a duplicate camera stream.
    """
    session.mkdir(parents=True, exist_ok=True)
    dest = session / video.name
    if not dest.exists():
        shutil.copy2(video, dest)

    for sidecar in video.parent.iterdir():
        if (
            sidecar.is_file()
            and sidecar != video
            and sidecar.suffix.lower() in {".json", ".csv", ".yaml", ".yml", ".h5", ".hdf5"}
        ):
            sidecar_dest = session / sidecar.name
            if not sidecar_dest.exists():
                shutil.copy2(sidecar, sidecar_dest)
    from actuate.ingest.run import ensure_session_meta
    from actuate.sources.detect import normalize_filenames

    normalize_filenames(session)
    ensure_session_meta(session)
    return session


def _find_video(root: Path) -> Path:
    """First video anywhere under `root` (LeRobot stores them a few levels deep)."""
    for ext in _VIDEO_EXTS:
        hits = sorted(root.rglob(f"*{ext}"))
        # prefer a non-redacted, non-depth stream
        hits = [h for h in hits if "depth" not in h.name.lower()] or hits
        if hits:
            return hits[0]
    raise FileNotFoundError(
        f"no video ({'/'.join(_VIDEO_EXTS)}) found under {root}. This source has no raw "
        "video for the perception pipeline to process.")


# ------------------------------------------------------------------ hf://
def resolve_hf(spec: str, work_root: Path, *, files: str | None = None,
               split: str | None = None) -> Path:
    """hf://<repo_id>[/<subpath>] -> a staged session directory.

    `files=` a CONCRETE path -> downloads just that ONE file (essential for large datasets:
    a 30 GB multimodal repo must not be pulled whole to process one clip). Without `files=`
    it snapshots the repo and finds a video (LeRobot datasets keep them under videos/…).
    `split=` is accepted for API compatibility (video processing is per-episode).
    """
    body = spec[len("hf://"):]
    repo_id = "/".join(body.split("/")[:2]) if body.count("/") >= 1 else body
    token = _hf_token()

    if files and "*" not in files and "?" not in files:
        # single-file download -- do NOT pull the whole repo
        from huggingface_hub import hf_hub_download

        video = Path(hf_hub_download(repo_id, filename=files, repo_type="dataset",
                                     token=token))
    else:
        from huggingface_hub import snapshot_download

        cache = work_root / "hf" / repo_id.replace("/", "__")
        local = Path(snapshot_download(repo_id, repo_type="dataset", local_dir=cache,
                                       token=token))
        video = (next(iter(local.rglob(files)), None) if files
                 else _find_video(local))
        if video is None:
            raise FileNotFoundError(f"{files!r} not found in {repo_id}")
    return _stage_video(video, work_root / repo_id.split("/")[-1])


def _hf_token() -> str | None:
    import os

    tok = os.environ.get("HF_TOKEN")
    if tok:
        return tok
    try:
        from huggingface_hub import HfFolder

        return HfFolder.get_token()
    except Exception:
        return None


# ------------------------------------------------------------------ s3://
def resolve_s3(spec: str, work_root: Path) -> Path:
    """s3://<bucket>/<key-or-prefix> -> staged session. Uses the configured AWS creds.

    WRITTEN-ONLY: there is no deployed Actuate bucket to verify against. The code path is
    real (fsspec/s3fs) and will work once credentials + a bucket exist.
    """
    try:
        import fsspec
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("s3:// needs the aws extra: pip install -e '.[aws]'") from exc

    fs = fsspec.filesystem("s3")
    key = spec[len("s3://"):]
    dest_root = work_root / "s3" / key.replace("/", "__")
    dest_root.mkdir(parents=True, exist_ok=True)
    if fs.isdir(spec):
        fs.get(spec.rstrip("/") + "/", str(dest_root), recursive=True)
        video = _find_video(dest_root)
    else:
        local_file = dest_root / Path(key).name
        fs.get(spec, str(local_file))
        video = local_file
    return _stage_video(video, work_root / Path(key).stem)


# ------------------------------------------------------------------ http(s)://
def resolve_http(spec: str, work_root: Path) -> Path:
    """http(s)://... -> staged session. Rewrites Google-Drive share links to direct
    downloads; otherwise a straight streamed download."""
    import requests

    url = _direct_url(spec)
    dest_root = work_root / "http"
    dest_root.mkdir(parents=True, exist_ok=True)
    name = _url_filename(url)
    local_file = dest_root / name
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with local_file.open("wb") as fh:
            for chunk in r.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
    return _stage_video(local_file, work_root / local_file.stem)


def _direct_url(url: str) -> str:
    """Google Drive share link -> direct-download URL; others pass through."""
    import re

    m = re.search(r"drive\.google\.com/file/d/([^/]+)", url)
    if m:
        return f"https://drive.google.com/uc?export=download&id={m.group(1)}"
    m = re.search(r"[?&]id=([^&]+)", url)
    if "drive.google.com" in url and m:
        return f"https://drive.google.com/uc?export=download&id={m.group(1)}"
    return url


def _url_filename(url: str) -> str:
    from urllib.parse import unquote, urlparse

    name = Path(unquote(urlparse(url).path)).name
    if not name or "." not in name:
        return "download.mp4"
    return name


# ------------------------------------------------------------------ openx://
def resolve_openx(spec: str, work_root: Path, *, max_episodes: int = 1) -> Path:
    """openx://<dataset_name> -> staged session, via tensorflow-datasets.

    Full Open-X datasets are hundreds of GB; this pulls a small slice and writes the first
    episode's frames out as an mp4 so the pipeline can process it. Tested on a tiny slice.
    """
    try:
        import tensorflow_datasets as tfds
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("openx:// needs tensorflow-datasets") from exc
    import cv2
    import numpy as np

    name = spec[len("openx://"):]
    ds = tfds.load(name, split=f"train[:{max_episodes}]", data_dir=str(work_root / "openx"))
    session = work_root / name.replace("/", "__")
    session.mkdir(parents=True, exist_ok=True)
    video_path = session / "compressed.mp4"

    writer = None
    for episode in ds:                                     # first episode only
        for step in episode["steps"]:
            img = step["observation"].get("image")
            if img is None:
                continue
            frame = np.asarray(img)[:, :, ::-1]            # RGB->BGR for OpenCV
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(str(video_path),
                                         cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (w, h))
            writer.write(frame)
        break
    if writer is not None:
        writer.release()
    if not video_path.exists():
        raise FileNotFoundError(
            f"openx dataset {name!r} yielded no image observations to build a video from.")
    from actuate.ingest.run import ensure_session_meta

    ensure_session_meta(session)
    return session
