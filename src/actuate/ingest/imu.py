"""Raw IMU ingestion and video-frame synchronization.

The modern pipeline receives source datasets as a directory containing a video and optional
sidecars. SLAM consumes frame-aligned IMU from ``session.h5``; this module is the boundary that
turns a raw ``imu.json``/``imu.csv`` sidecar into that contract.

The preferred source schema is one record per sensor sample:

.. code-block:: json

    {
      "timestamp_ns": 67710832865,
      "nearest_video_frame": 1,
      "video_frame_timestamp_ns": 67744472823,
      "accel": [0.97, 0.89, 10.0],
      "gyro": [-0.05, 0.16, 0.32]
    }

When ``nearest_video_frame`` is present, all samples assigned to a frame are averaged. Missing
frames are interpolated. Timestamp-only streams are interpolated onto the nominal video clock.
The result always has exactly ``session_meta.frame_count`` rows, which is the invariant SLAM
requires.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class IMUSyncResult:
    source_path: Path
    session_h5: Path
    raw_samples: int
    frame_count: int
    sync_method: str
    cached: bool = False

    def summary(self) -> str:
        source = "existing" if self.cached else self.sync_method
        return (
            f"{self.raw_samples} IMU samples -> {self.frame_count} video frames "
            f"({source})"
        )


def _source_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _find_source(session_dir: Path) -> Path | None:
    candidates = []
    for pattern in ("imu*.json", "imu*.csv"):
        candidates.extend(session_dir.glob(pattern))
    files = sorted(p for p in candidates if p.is_file())
    return files[0] if files else None


def _vector(value: Any) -> tuple[float, float, float] | None:
    if isinstance(value, dict):
        try:
            return float(value["x"]), float(value["y"]), float(value["z"])
        except (KeyError, TypeError, ValueError):
            return None
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        try:
            return float(value[0]), float(value[1]), float(value[2])
        except (TypeError, ValueError):
            return None
    return None


def _csv_vector(row: dict[str, Any], prefix: str) -> tuple[float, float, float] | None:
    for names in (
        (f"{prefix}_x", f"{prefix}_y", f"{prefix}_z"),
        (f"{prefix}.x", f"{prefix}.y", f"{prefix}.z"),
        (f"{prefix}X", f"{prefix}Y", f"{prefix}Z"),
    ):
        if all(row.get(n) not in (None, "") for n in names):
            try:
                return tuple(float(row[n]) for n in names)  # type: ignore[return-value]
            except (TypeError, ValueError):
                return None
    return None


def _load_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            for key in ("samples", "imu", "records", "data"):
                if isinstance(payload.get(key), list):
                    payload = payload[key]
                    break
        if not isinstance(payload, list):
            raise ValueError(
                f"{path.name}: expected a JSON list or an object containing "
                "`samples`/`imu`/`records`/`data`"
            )
        return [r for r in payload if isinstance(r, dict)]

    with path.open(newline="", encoding="utf-8") as fh:
        return [dict(row) for row in csv.DictReader(fh)]


def _number(row: dict[str, Any], *names: str) -> float:
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return float("nan")


def _timestamp_ns(row: dict[str, Any]) -> float:
    direct = _number(row, "timestamp_ns", "time_ns", "t_ns")
    if np.isfinite(direct):
        return direct
    us = _number(row, "timestamp_us", "time_us", "t_us")
    if np.isfinite(us):
        return us * 1_000.0
    ms = _number(row, "timestamp_ms", "time_ms", "t_ms")
    if np.isfinite(ms):
        return ms * 1_000_000.0
    sec = _number(row, "timestamp_s", "time_s", "timestamp", "time", "t")
    return sec * 1_000_000_000.0 if np.isfinite(sec) else float("nan")


def _columnar(records: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    rows = []
    for row in records:
        gyro = (
            _vector(row.get("gyro"))
            or _vector(row.get("gyroscope"))
            or _csv_vector(row, "gyro")
            or _csv_vector(row, "gyroscope")
        )
        if gyro is None:
            continue
        accel = (
            _vector(row.get("accel"))
            or _vector(row.get("accelerometer"))
            or _csv_vector(row, "accel")
            or _csv_vector(row, "accelerometer")
        )
        mag = (
            _vector(row.get("mag"))
            or _vector(row.get("magnetometer"))
            or _csv_vector(row, "mag")
            or _csv_vector(row, "magnetometer")
        )
        rows.append(
            {
                "timestamp_ns": _timestamp_ns(row),
                "frame": _number(
                    row,
                    "nearest_video_frame",
                    "video_frame",
                    "frame_index",
                    "frame",
                ),
                "video_timestamp_ns": _number(
                    row, "video_frame_timestamp_ns", "frame_timestamp_ns"
                ),
                "gyro": gyro,
                "accel": accel or (np.nan, np.nan, np.nan),
                "mag": mag or (np.nan, np.nan, np.nan),
                "temp_c": _number(row, "temp_c", "temperature_c", "temperature"),
            }
        )

    if not rows:
        raise ValueError("no IMU records with a valid 3-axis gyroscope were found")
    return {
        "timestamp_ns": np.asarray([r["timestamp_ns"] for r in rows], dtype=np.float64),
        "frame": np.asarray([r["frame"] for r in rows], dtype=np.float64),
        "video_timestamp_ns": np.asarray(
            [r["video_timestamp_ns"] for r in rows], dtype=np.float64
        ),
        "gyro": np.asarray([r["gyro"] for r in rows], dtype=np.float64),
        "accel": np.asarray([r["accel"] for r in rows], dtype=np.float64),
        "mag": np.asarray([r["mag"] for r in rows], dtype=np.float64),
        "temp_c": np.asarray([r["temp_c"] for r in rows], dtype=np.float64),
    }


def _fill_missing(values: np.ndarray) -> np.ndarray:
    """Interpolate each channel onto every frame; edge gaps use the nearest measurement."""
    values = np.asarray(values, dtype=np.float64)
    one_dimensional = values.ndim == 1
    if one_dimensional:
        values = values[:, None]
    x = np.arange(len(values), dtype=np.float64)
    out = values.copy()
    for axis in range(values.shape[1]):
        valid = np.isfinite(values[:, axis])
        if not valid.any():
            continue
        out[:, axis] = np.interp(x, x[valid], values[valid, axis])
    return out[:, 0] if one_dimensional else out


def _aggregate_by_frame(
    columns: dict[str, np.ndarray], frame_count: int, fps: float
) -> tuple[dict[str, np.ndarray], str]:
    raw_frame = columns["frame"]
    valid_frame = np.isfinite(raw_frame)
    observed = raw_frame[valid_frame].astype(np.int64)
    if observed.size == 0:
        raise ValueError("no frame assignments")

    # Camera tooling commonly numbers frames 1..N. Treat it as one-based only when the
    # observed range reaches N; a sparse zero-based recording may legitimately start after 0.
    frame_base = 1 if observed.min() >= 1 and observed.max() >= frame_count else 0
    frame_idx = raw_frame.astype(np.int64, casting="unsafe") - frame_base
    valid_frame &= (frame_idx >= 0) & (frame_idx < frame_count)

    aligned: dict[str, np.ndarray] = {}
    for key in ("gyro", "accel", "mag", "temp_c"):
        source = columns[key]
        source_2d = source[:, None] if source.ndim == 1 else source
        sums = np.zeros((frame_count, source_2d.shape[1]), dtype=np.float64)
        counts = np.zeros_like(sums)
        for axis in range(source_2d.shape[1]):
            valid = valid_frame & np.isfinite(source_2d[:, axis])
            np.add.at(sums[:, axis], frame_idx[valid], source_2d[valid, axis])
            np.add.at(counts[:, axis], frame_idx[valid], 1)
        values = np.divide(
            sums, counts, out=np.full_like(sums, np.nan), where=counts > 0
        )
        values = _fill_missing(values)
        aligned[key] = values[:, 0] if source.ndim == 1 else values

    sample_ts = np.full(frame_count, np.nan, dtype=np.float64)
    raw_sample_ts = columns["timestamp_ns"]
    video_ts = np.full(frame_count, np.nan, dtype=np.float64)
    raw_video_ts = columns["video_timestamp_ns"]
    for i in range(frame_count):
        frame_mask = valid_frame & (frame_idx == i)
        samples = raw_sample_ts[frame_mask & np.isfinite(raw_sample_ts)]
        videos = raw_video_ts[frame_mask & np.isfinite(raw_video_ts)]
        if videos.size:
            video_ts[i] = float(np.median(videos))
        if samples.size:
            # The averaged gyro represents the whole frame interval, but synchronization
            # integrity is the distance from the video timestamp to the nearest physical
            # sensor sample—not the distance to the interval mean (which is systematically
            # half a frame away when timestamps mark a frame boundary).
            reference = video_ts[i] if np.isfinite(video_ts[i]) else float(np.mean(samples))
            sample_ts[i] = float(samples[np.argmin(np.abs(samples - reference))])
    if not np.isfinite(video_ts).any():
        first = np.nanmin(columns["timestamp_ns"])
        if not np.isfinite(first):
            first = 0.0
        video_ts = first + np.arange(frame_count) * (1e9 / fps)
    if not np.isfinite(sample_ts).any():
        sample_ts = video_ts.copy()
    aligned["timestamp_ns"] = np.rint(_fill_missing(sample_ts)).astype(np.int64)
    if np.isfinite(raw_video_ts).any():
        aligned["video_timestamp_ns"] = np.rint(_fill_missing(video_ts)).astype(np.int64)
    return aligned, f"nearest_video_frame_mean_{frame_base}based"


def _aggregate_by_timestamp(
    columns: dict[str, np.ndarray], frame_count: int, fps: float
) -> tuple[dict[str, np.ndarray], str]:
    timestamps = columns["timestamp_ns"]
    valid_time = np.isfinite(timestamps)
    if valid_time.sum() < 2:
        raise ValueError(
            "IMU records need either `nearest_video_frame` or at least two timestamps"
        )
    order = np.argsort(timestamps[valid_time])
    sample_t = timestamps[valid_time][order]
    frame_t = sample_t[0] + np.arange(frame_count, dtype=np.float64) * (1e9 / fps)
    aligned: dict[str, np.ndarray] = {"timestamp_ns": np.rint(frame_t).astype(np.int64)}
    for key in ("gyro", "accel", "mag", "temp_c"):
        source = columns[key][valid_time][order]
        source_2d = source[:, None] if source.ndim == 1 else source
        values = np.full((frame_count, source_2d.shape[1]), np.nan)
        for axis in range(source_2d.shape[1]):
            valid = np.isfinite(source_2d[:, axis])
            if valid.any():
                values[:, axis] = np.interp(
                    frame_t, sample_t[valid], source_2d[valid, axis]
                )
        aligned[key] = values[:, 0] if source.ndim == 1 else values
    return aligned, "timestamp_interpolation"


def _write_h5(
    path: Path,
    aligned: dict[str, np.ndarray],
    *,
    source_path: Path,
    source_hash: str,
    raw_samples: int,
    sync_method: str,
) -> None:
    import h5py

    with h5py.File(path, "a") as h5:
        group = h5.require_group("imu")
        for key, values in aligned.items():
            if not np.isfinite(values).any():
                continue
            if key in group:
                del group[key]
            group.create_dataset(
                key,
                data=values.astype(np.int64 if key == "timestamp_ns" else np.float32),
                compression="gzip",
                shuffle=True,
            )
        group.attrs["source_file"] = source_path.name
        group.attrs["source_sha256"] = source_hash
        group.attrs["raw_samples"] = raw_samples
        group.attrs["sync_method"] = sync_method
        group.attrs["schema_version"] = 2
        group.attrs["gyro_units"] = "source_units"
        group.attrs["accel_units"] = "source_units"


def sync_imu(session_dir: Path, meta: dict[str, Any] | None = None) -> IMUSyncResult | None:
    """Synchronize a raw IMU sidecar to video frames and persist ``session.h5``.

    Returns ``None`` when no raw IMU sidecar exists. Existing output is reused only when it
    matches both the source SHA-256 and current video frame count.
    """
    session_dir = Path(session_dir)
    source = _find_source(session_dir)
    if source is None:
        return None
    if meta is None:
        meta = json.loads((session_dir / "session_meta.json").read_text(encoding="utf-8"))
    frame_count = int(meta.get("frame_count") or 0)
    fps = float(meta.get("fps_nominal", meta.get("fps", 30.0)))
    if frame_count <= 0 or fps <= 0:
        raise ValueError("session_meta.json needs positive frame_count and fps for IMU sync")

    source_hash = _source_hash(source)
    h5_path = session_dir / "session.h5"
    if h5_path.exists():
        import h5py

        try:
            with h5py.File(h5_path, "r") as h5:
                group = h5.get("imu")
                if (
                    group is not None
                    and "gyro" in group
                    and len(group["gyro"]) == frame_count
                    and group.attrs.get("source_sha256") == source_hash
                    and int(group.attrs.get("schema_version", 0)) == 2
                ):
                    return IMUSyncResult(
                        source,
                        h5_path,
                        int(group.attrs.get("raw_samples", 0)),
                        frame_count,
                        str(group.attrs.get("sync_method", "existing")),
                        cached=True,
                    )
        except OSError:
            pass

    records = _load_records(source)
    columns = _columnar(records)
    if np.isfinite(columns["frame"]).any():
        aligned, method = _aggregate_by_frame(columns, frame_count, fps)
    else:
        aligned, method = _aggregate_by_timestamp(columns, frame_count, fps)
    _write_h5(
        h5_path,
        aligned,
        source_path=source,
        source_hash=source_hash,
        raw_samples=len(columns["gyro"]),
        sync_method=method,
    )

    meta_path = session_dir / "session_meta.json"
    updated_meta = dict(meta)
    modalities = dict(updated_meta.get("modalities") or {})
    modalities["imu"] = True
    updated_meta["modalities"] = modalities
    updated_meta["imu"] = {
        "source": source.name,
        "raw_samples": len(columns["gyro"]),
        "aligned_frames": frame_count,
        "sync_method": method,
    }
    meta_path.write_text(json.dumps(updated_meta, indent=2), encoding="utf-8")
    return IMUSyncResult(
        source, h5_path, len(columns["gyro"]), frame_count, method, cached=False
    )


__all__ = ["IMUSyncResult", "sync_imu"]
