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
) -> None:
    """
    Write a complete session HDF5 file.

    HDF5 structure:
      session.h5/
      ├── video/
      │   ├── timestamps       [N_frames] float64, absolute epoch seconds
      │   └── pts_relative     [N_frames] float64, seconds from stream start
      ├── imu/
      │   ├── timestamps       [N_frames] float64, same axis as video
      │   ├── accel            [N_frames, 3] float32: ax, ay, az (m/s²)
      │   └── gyro             [N_frames, 3] float32: gx, gy, gz (rad/s)
      └── metadata             JSON string: session_meta + sync_stats
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(str(path), "w") as f:
        # Video group
        video_grp = f.create_group("video")
        video_grp.create_dataset(
            "timestamps", data=video_timestamps_abs.astype(np.float64), compression="gzip"
        )
        video_grp.create_dataset(
            "pts_relative", data=pts_relative.astype(np.float64), compression="gzip"
        )

        # IMU group
        imu_grp = f.create_group("imu")
        imu_grp.create_dataset(
            "timestamps", data=video_timestamps_abs.astype(np.float64), compression="gzip"
        )
        imu_grp.create_dataset(
            "accel", data=accel.astype(np.float32), compression="gzip"
        )
        imu_grp.create_dataset(
            "gyro", data=gyro.astype(np.float32), compression="gzip"
        )

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
