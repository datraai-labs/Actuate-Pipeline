"""
DatraAI Pipeline — HDF5 Session Writer / Reader
Handles write, read, and annotation append for session.h5 files.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

import h5py
import numpy as np


def write_session_h5(
    path: Path,
    video_timestamps_abs: np.ndarray,
    pts_relative: np.ndarray,
    accel: np.ndarray,
    gyro: np.ndarray,
    metadata_dict: Dict[str, Any],
    imu_timestamps_raw: Optional[np.ndarray] = None,
    imu_interpolated_over_dropout: Optional[np.ndarray] = None,
    imu_outside_range: Optional[np.ndarray] = None,
    imu_mag: Optional[np.ndarray] = None,
    imu_temp_c: Optional[np.ndarray] = None,
) -> None:
    """
    Write a complete session HDF5 file.

    HDF5 structure:
      session.h5/
      ├── video/
      │   ├── timestamps       [N_frames] float64, absolute epoch seconds
      │   └── pts_relative     [N_frames] float64, seconds from stream start
      ├── imu/
      │   ├── timestamps       [N_frames] float64 — the VIDEO FRAME AXIS the IMU
      │   │                    was resampled onto, NOT sensor sample times
      │   ├── timestamps_raw   [N_samples] float64, the original sensor sample
      │   │                    times exactly as recorded — the evidence needed to
      │   │                    diagnose dropouts/jitter/clock domain survives sync
      │   ├── accel            [N_frames, 3] float32: ax, ay, az (m/s²)
      │   ├── gyro             [N_frames, 3] float32: gx, gy, gz (rad/s)
      │   ├── interpolated_over_dropout  [N_frames] bool — value synthesized by
      │   │                    interpolating across a raw-sample dropout gap
      │   └── outside_imu_range [N_frames] bool — frame precedes/follows the raw
      │                        stream entirely (np.interp clamped to edge value)
      └── metadata             JSON string: session_meta + sync_stats
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Append-mode with per-dataset ownership (audit 2026-08-01, Group 4). This file
    # previously opened "w", truncating the whole file — running the legacy sync
    # after the modern one (actuate.ingest.imu, which writes its own keys under
    # `imu/`: timestamp_ns, timestamp_ns_raw, mag, temp_c, sample_count, ...)
    # DESTROYED the modern group. Each writer now deletes and recreates only the
    # datasets it owns; the modern path is the AUTHORITATIVE one (see
    # src/actuate/ingest/__init__.py) and its keys must survive this writer.
    def _replace(group, name, data, **kwargs):
        if name in group:
            del group[name]
        return group.create_dataset(name, data=data, compression="gzip", **kwargs)

    with h5py.File(str(path), "a") as f:
        # Video group
        video_grp = f.require_group("video")
        _replace(video_grp, "timestamps", video_timestamps_abs.astype(np.float64))
        _replace(video_grp, "pts_relative", pts_relative.astype(np.float64))

        # IMU group
        imu_grp = f.require_group("imu")
        ts = _replace(imu_grp, "timestamps", video_timestamps_abs.astype(np.float64))
        ts.attrs["semantics"] = (
            "video frame axis (== video/timestamps); original sensor sample times "
            "are in timestamps_raw"
        )
        _replace(imu_grp, "accel", accel.astype(np.float32))
        _replace(imu_grp, "gyro", gyro.astype(np.float32))
        if imu_timestamps_raw is not None:
            raw = _replace(
                imu_grp,
                "timestamps_raw",
                np.asarray(imu_timestamps_raw, dtype=np.float64),
            )
            raw.attrs["semantics"] = "original sensor sample times, absolute seconds"
        if imu_interpolated_over_dropout is not None:
            _replace(
                imu_grp,
                "interpolated_over_dropout",
                np.asarray(imu_interpolated_over_dropout, dtype=bool),
            )
        if imu_outside_range is not None:
            _replace(
                imu_grp,
                "outside_imu_range",
                np.asarray(imu_outside_range, dtype=bool),
            )
        if imu_mag is not None:
            _replace(imu_grp, "mag", np.asarray(imu_mag, dtype=np.float32))
        if imu_temp_c is not None:
            _replace(imu_grp, "temp_c", np.asarray(imu_temp_c, dtype=np.float32))

        # Metadata as JSON string attribute
        f.attrs["metadata"] = json.dumps(metadata_dict, default=str)


def read_session_h5(path: Path) -> Dict[str, Any]:
    """
    Read a session HDF5 file and return all arrays + metadata as a dict.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {path}")

    result = {}
    with h5py.File(str(path), "r") as f:
        # Video
        result["video_timestamps"] = f["video"]["timestamps"][:]
        result["pts_relative"] = f["video"]["pts_relative"][:]

        # IMU
        result["imu_timestamps"] = f["imu"]["timestamps"][:]
        result["accel"] = f["imu"]["accel"][:]
        result["gyro"] = f["imu"]["gyro"][:]
        for optional in (
            "timestamps_raw",
            "interpolated_over_dropout",
            "outside_imu_range",
            "mag",
            "temp_c",
        ):
            if optional in f["imu"]:
                result[f"imu_{optional}"] = f["imu"][optional][:]

        # Metadata
        metadata_str = f.attrs.get("metadata", "{}")
        result["metadata"] = json.loads(metadata_str)

        # Check for annotations
        if "annotations" in f:
            result["annotations"] = {}
            for key in f["annotations"]:
                data = f["annotations"][key][()]
                if isinstance(data, bytes):
                    data = data.decode("utf-8")
                result["annotations"][key] = data

    return result


def append_annotations(
    path: Path,
    annotation_key: str,
    annotation_data: Any,
) -> None:
    """
    Append an annotation dataset to an existing session HDF5 file.
    If the annotation already exists, it is overwritten.

    annotation_data can be:
      - A numpy array: stored as a dataset
      - A dict/list: stored as a JSON-encoded string dataset
      - A string: stored as-is
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {path}")

    with h5py.File(str(path), "a") as f:
        if "annotations" not in f:
            f.create_group("annotations")

        ann_grp = f["annotations"]

        # Remove existing if present
        if annotation_key in ann_grp:
            del ann_grp[annotation_key]

        if isinstance(annotation_data, np.ndarray):
            ann_grp.create_dataset(annotation_key, data=annotation_data, compression="gzip")
        elif isinstance(annotation_data, (dict, list)):
            json_str = json.dumps(annotation_data, default=str)
            ann_grp.create_dataset(annotation_key, data=json_str)
        elif isinstance(annotation_data, str):
            ann_grp.create_dataset(annotation_key, data=annotation_data)
        else:
            raise TypeError(
                f"Unsupported annotation_data type: {type(annotation_data)}. "
                "Expected np.ndarray, dict, list, or str."
            )


def update_metadata(path: Path, updates: Dict[str, Any]) -> None:
    """
    Merge updates into the existing metadata JSON in the HDF5 file.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {path}")

    with h5py.File(str(path), "a") as f:
        existing = json.loads(f.attrs.get("metadata", "{}"))
        existing.update(updates)
        f.attrs["metadata"] = json.dumps(existing, default=str)
