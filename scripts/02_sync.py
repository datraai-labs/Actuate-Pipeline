"""
DatraAI Pipeline — Step 02: Sync
Align IMU to video timestamps via interpolation → write HDF5.

Input:  processed/{session_id}/pts.npy
        processed/{session_id}/imu_raw.npy
        processed/{session_id}/session_meta.json
Output: processed/{session_id}/session.h5
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config as cfg
from utils.hdf5_writer import write_session_h5, update_metadata

STEP = "02_sync"


class SyncDriftError(Exception):
    """Raised when IMU-video sync drift exceeds threshold."""
    pass


def run(session_id: str) -> dict:
    """
    Synchronize IMU data to video frame timestamps and write session.h5.

    Returns:
        sync_stats dict.
    """
    t0 = time.time()
    print(f"[{STEP}] Starting {session_id}...")

    proc_dir = cfg.PROCESSED_DIR / session_id

    # Load inputs
    pts_path = proc_dir / "pts.npy"
    imu_path = proc_dir / "imu_raw.npy"
    meta_path = proc_dir / "session_meta.json"

    for p in [pts_path, imu_path, meta_path]:
        if not p.exists():
            raise FileNotFoundError(f"[{STEP}] Required file not found: {p}")

    pts = np.load(str(pts_path))            # relative, starts at 0.0
    imu_raw = np.load(str(imu_path))        # [N, 7]: epoch_ms, ax, ay, az, gx, gy, gz

    with open(meta_path) as f:
        session_meta = json.load(f)

    recording_start_epoch_ms = session_meta["recording_start_epoch_ms"]

    # ─── Step 1: Build absolute time axes ─────────────────────
    # Video: absolute epoch seconds
    video_t_abs = pts + (recording_start_epoch_ms / 1000.0)

    # IMU: already in epoch_ms, convert to epoch seconds
    imu_t_abs = imu_raw[:, 0] / 1000.0

    print(f"[{STEP}] Video: {len(pts)} frames, range [{video_t_abs[0]:.3f}, {video_t_abs[-1]:.3f}]s")
    print(f"[{STEP}] IMU:   {len(imu_raw)} samples, range [{imu_t_abs[0]:.3f}, {imu_t_abs[-1]:.3f}]s")

    # ─── Step 2: Interpolate IMU to video timestamps ──────────
    n_frames = len(pts)
    synced_accel = np.zeros((n_frames, 3), dtype=np.float32)
    synced_gyro = np.zeros((n_frames, 3), dtype=np.float32)

    # Accel channels: columns 1, 2, 3 of imu_raw
    for ch_idx in range(3):
        synced_accel[:, ch_idx] = np.interp(
            video_t_abs, imu_t_abs, imu_raw[:, ch_idx + 1]
        ).astype(np.float32)

    # Gyro channels: columns 4, 5, 6 of imu_raw
    for ch_idx in range(3):
        synced_gyro[:, ch_idx] = np.interp(
            video_t_abs, imu_t_abs, imu_raw[:, ch_idx + 4]
        ).astype(np.float32)

    print(f"[{STEP}] Interpolated IMU → {n_frames} frames (accel shape: {synced_accel.shape})")

    # ─── Step 3: Drift check ─────────────────────────────────
    # For each video frame, find the nearest IMU timestamp
    nearest_indices = np.searchsorted(imu_t_abs, video_t_abs)
    nearest_indices = np.clip(nearest_indices, 0, len(imu_t_abs) - 1)

    # Also check the index before (searchsorted gives insertion point)
    nearest_indices_prev = np.clip(nearest_indices - 1, 0, len(imu_t_abs) - 1)

    # Pick whichever is closer
    delta_right = np.abs(video_t_abs - imu_t_abs[nearest_indices])
    delta_left = np.abs(video_t_abs - imu_t_abs[nearest_indices_prev])
    use_left = delta_left < delta_right
    best_delta = np.where(use_left, delta_left, delta_right)

    delta_ms = best_delta * 1000.0
    max_drift_ms = float(np.max(delta_ms))
    mean_drift_ms = float(np.mean(delta_ms))

    # Count IMU dropout frames (where nearest IMU sample is far away)
    imu_dropout_frames = int(np.sum(delta_ms > 10.0))  # > 10ms is a dropout

    drift_warning = max_drift_ms > cfg.SYNC_DRIFT_THRESHOLD_MS

    if drift_warning:
        worst_frames = np.argsort(delta_ms)[-5:]
        print(
            f"[{STEP}] ⚠ DRIFT WARNING: max_drift={max_drift_ms:.2f}ms "
            f"(threshold: {cfg.SYNC_DRIFT_THRESHOLD_MS}ms)\n"
            f"  Worst frames: {worst_frames.tolist()}, "
            f"drifts: {delta_ms[worst_frames].tolist()}"
        )
    else:
        print(f"[{STEP}] Drift OK: max={max_drift_ms:.2f}ms, mean={mean_drift_ms:.3f}ms")

    # ─── Step 4: Write HDF5 ──────────────────────────────────
    sync_stats = {
        "max_drift_ms": round(max_drift_ms, 4),
        "mean_drift_ms": round(mean_drift_ms, 4),
        "imu_dropout_frame_count": imu_dropout_frames,
        "interpolation_method": "linear",
        "drift_warning": drift_warning,
    }

    combined_meta = {**session_meta, "sync_stats": sync_stats}

    h5_path = proc_dir / "session.h5"
    write_session_h5(
        path=h5_path,
        video_timestamps_abs=video_t_abs,
        pts_relative=pts,
        accel=synced_accel,
        gyro=synced_gyro,
        metadata_dict=combined_meta,
    )

    print(f"[{STEP}] HDF5 written: {h5_path} ({h5_path.stat().st_size / 1024:.0f} KB)")

    elapsed = time.time() - t0
    print(f"[{STEP}] ✓ Done ({elapsed:.1f}s)")

    return sync_stats


def main():
    parser = argparse.ArgumentParser(description="DatraAI Step 02: Sync IMU to video timestamps")
    parser.add_argument("--session", type=str, required=True, help="Session ID (e.g., session_001)")
    args = parser.parse_args()

    session_id = Path(args.session).name
    run(session_id)


if __name__ == "__main__":
    main()
