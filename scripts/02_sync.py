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
    # [N, 7] legacy or [N, 11] with mag+temp: epoch_ms, ax..az, gx..gz[, mx..mz, temp_c]
    imu_raw = np.load(str(imu_path))

    with open(meta_path) as f:
        session_meta = json.load(f)

    recording_start_epoch_ms = session_meta["recording_start_epoch_ms"]

    # ─── Step 1: Build absolute time axes ─────────────────────
    # Video: absolute epoch seconds
    video_t_abs = pts + (recording_start_epoch_ms / 1000.0)

    # Per-device latency (audit §3c): what a camera timestamp MEANS (start-of-
    # exposure vs end-of-readout) and the camera→IMU transport delay are recorded
    # in session_meta.timestamp_semantics. A declared latency is APPLIED here; an
    # undeclared one (None) is an ASSUMPTION we record, not silently make.
    semantics = session_meta.get("timestamp_semantics", {})
    declared_latency_ms = semantics.get("camera_to_imu_latency_ms")
    if isinstance(declared_latency_ms, (int, float)):
        video_t_abs = video_t_abs + declared_latency_ms / 1000.0
        latency_note = f"applied declared camera_to_imu_latency_ms={declared_latency_ms}"
    else:
        declared_latency_ms = None
        latency_note = (
            "camera_to_imu_latency_ms UNKNOWN — assumed 0. A rolling-shutter "
            "camera's exposure offset (typically 10–30ms) sits below this check's "
            "detection floor."
        )

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

    def _interp_optional(column: np.ndarray) -> np.ndarray | None:
        """Interpolate an optional channel over its finite samples; None if unmeasured."""
        finite = np.isfinite(column)
        if not finite.any():
            return None
        return np.interp(video_t_abs, imu_t_abs[finite], column[finite]).astype(np.float32)

    # Optional channels (audit Group 4): mag + temp are carried through when the
    # source measured them, instead of being silently dropped at parse time.
    synced_mag = None
    synced_temp = None
    if imu_raw.shape[1] >= 10:
        mag_channels = [_interp_optional(imu_raw[:, 7 + i]) for i in range(3)]
        if all(ch is not None for ch in mag_channels):
            synced_mag = np.stack(mag_channels, axis=1)
    if imu_raw.shape[1] >= 11:
        synced_temp = _interp_optional(imu_raw[:, 10])

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

    # ─── Step 3b: Fabrication accounting ─────────────────────
    # np.interp (Step 2) spans raw-sample gaps without complaint. A frame inside
    # a genuine dropout carries a synthesized value that must stay distinguishable
    # from a measured one at point of use — the None-over-plausible-default rule.
    # Flagged when the enclosing raw gap exceeds IMU_DROPOUT_GAP_FACTOR x the
    # median interval AND the frame doesn't sit essentially on a real sample
    # (within a quarter-median, that sample dominates the interpolation).
    raw_dt_ms = np.diff(imu_t_abs) * 1000.0
    raw_median_dt_ms = float(np.median(raw_dt_ms))
    raw_max_gap_ms = float(np.max(raw_dt_ms))
    raw_missing_estimate = (
        int(np.clip(np.rint(raw_dt_ms / raw_median_dt_ms) - 1, 0, None).sum())
        if raw_median_dt_ms > 0
        else 0
    )

    outside_imu_range = (video_t_abs < imu_t_abs[0]) | (video_t_abs > imu_t_abs[-1])
    interp_over_dropout = np.zeros(n_frames, dtype=bool)
    if raw_median_dt_ms > 0 and len(imu_t_abs) >= 2:
        right = np.clip(np.searchsorted(imu_t_abs, video_t_abs), 1, len(imu_t_abs) - 1)
        enclosing_gap_ms = (imu_t_abs[right] - imu_t_abs[right - 1]) * 1000.0
        interp_over_dropout = (
            ~outside_imu_range
            & (enclosing_gap_ms > cfg.IMU_DROPOUT_GAP_FACTOR * raw_median_dt_ms)
            & (delta_ms > 0.25 * raw_median_dt_ms)
        )

    n_fabricated = int(interp_over_dropout.sum())
    n_outside = int(outside_imu_range.sum())
    if n_fabricated or n_outside:
        print(
            f"[{STEP}] Fabrication flags: {n_fabricated} frames interpolated across "
            f"a dropout gap, {n_outside} outside the raw IMU range "
            f"(raw median dt {raw_median_dt_ms:.3f}ms, max gap {raw_max_gap_ms:.3f}ms, "
            f"~{raw_missing_estimate} samples missing)"
        )

    # ─── Step 3c: Separate alignment quality from transport jitter ──
    # max_drift_ms is the distance from each frame to the NEAREST raw sample — a
    # quantity bounded by the raw stream's gap structure, so on a dropout-y stream
    # it measures transport jitter, not clock alignment (audit §3a: the real
    # session's 3.94ms is half of one 7.97ms dropout gap). The CLEAN metric
    # excludes dropout-spanning and out-of-range frames and is what the sync-
    # quality gate should read.
    healthy = ~interp_over_dropout & ~outside_imu_range
    if healthy.any():
        max_drift_clean_ms = float(np.max(delta_ms[healthy]))
        mean_drift_clean_ms = float(np.mean(delta_ms[healthy]))
    else:
        max_drift_clean_ms = max_drift_ms
        mean_drift_clean_ms = mean_drift_ms

    drift_warning = max_drift_clean_ms > cfg.SYNC_DRIFT_THRESHOLD_MS

    if drift_warning:
        worst_frames = np.argsort(delta_ms)[-5:]
        print(
            f"[{STEP}] ⚠ DRIFT WARNING: max_drift_clean={max_drift_clean_ms:.2f}ms "
            f"(threshold: {cfg.SYNC_DRIFT_THRESHOLD_MS}ms, raw max incl. dropouts: "
            f"{max_drift_ms:.2f}ms)\n"
            f"  Worst frames: {worst_frames.tolist()}, "
            f"drifts: {delta_ms[worst_frames].tolist()}"
        )
    else:
        print(
            f"[{STEP}] Drift OK: clean max={max_drift_clean_ms:.2f}ms "
            f"(raw max incl. dropout-spanning frames: {max_drift_ms:.2f}ms, "
            f"mean={mean_drift_ms:.3f}ms)"
        )

    # Anchor provenance (audit §3b). When 01_ingest used the IMU-t0 fallback the
    # alignment holds BY CONSTRUCTION and this stage's drift numbers cannot
    # validate it — the certificate must carry that, not a clean score.
    alignment_validated = bool(session_meta.get("temporal_alignment_validated", False))
    alignment_note = session_meta.get("temporal_alignment_note")
    if alignment_note is None:
        source = session_meta.get("recording_start_source", "unknown")
        alignment_note = (
            f"session_meta predates anchor validation (audit 2026-08-01); "
            f"recording_start_source={source} — treated as UNVALIDATED"
        )

    # ─── Step 4: Write HDF5 ──────────────────────────────────
    sync_stats = {
        # Nearest-raw-sample distance. On a stream with dropouts this is bounded
        # by the gap structure — it quantifies TRANSPORT JITTER, not clock
        # divergence. Gate on max_drift_clean_ms, not this.
        "max_drift_ms": round(max_drift_ms, 4),
        "mean_drift_ms": round(mean_drift_ms, 4),
        # The same metric over frames NOT spanning a dropout / outside the raw
        # range — the actual sync-quality signal.
        "max_drift_clean_ms": round(max_drift_clean_ms, 4),
        "mean_drift_clean_ms": round(mean_drift_clean_ms, 4),
        "imu_dropout_frame_count": imu_dropout_frames,
        "interpolation_method": "linear",
        "drift_warning": drift_warning,
        # Raw-stream health: more diagnostic than the drift number, since the
        # drift metric is bounded by these gaps (see Step 3b).
        "raw_median_dt_ms": round(raw_median_dt_ms, 4),
        "raw_max_gap_ms": round(raw_max_gap_ms, 4),
        "raw_missing_sample_estimate": raw_missing_estimate,
        "dropout_gap_factor": cfg.IMU_DROPOUT_GAP_FACTOR,
        "frames_interpolated_over_dropout": n_fabricated,
        "frames_outside_imu_range": n_outside,
        "temporal_alignment_validated": alignment_validated,
        "temporal_alignment_note": alignment_note,
        "imu_clock_domain": session_meta.get("imu_clock_domain", "unknown"),
        "camera_to_imu_latency_ms_applied": declared_latency_ms,
        "camera_to_imu_latency_note": latency_note,
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
        imu_timestamps_raw=imu_t_abs,
        imu_interpolated_over_dropout=interp_over_dropout,
        imu_outside_range=outside_imu_range,
        imu_mag=synced_mag,
        imu_temp_c=synced_temp,
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
