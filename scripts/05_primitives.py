"""
DatraAI Pipeline — Step 05: Primitive Detection
12-primitive rule engine → L3 labels.

Detection strategy (wrist-IMU gyro/accel vs. hand-pose-vision vs. a
confidence-weighted fusion of both) is selected by config.IMU_SOURCE_MODE
and delegated to utils/imu_source_router.py (v2 addendum §1) — a
head-mounted IMU cannot see hand/finger motion, so "wrist_pronate" etc. must
come from vision when that's the only IMU available.

Each detector also gets a companion confidence float in [0,1] (v2 addendum
§9, utils/confidence.py) — CERTAINTY in the boolean call either way, not
"how likely True", surfaced per frame as "primitive_confidences".

Input:  processed/{session_id}/session.h5
        processed/{session_id}/hand_pose.json
        processed/{session_id}/object_tracks.json (optional — v2 addendum §3)
Output: processed/{session_id}/primitives.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config as cfg
from utils.hdf5_writer import read_session_h5
from utils.glove_profile import resolve_grasp_thresholds
from utils.worker_profile_store import load_profile as load_worker_profile
from utils.imu_source_router import (
    ALL_PRIMITIVES,
    # Re-exported for backward compatibility — existing callers/tests that
    # import these names directly from this module keep working unchanged.
    detect_wrist_pronate,
    detect_wrist_supinate,
    detect_wrist_flex,
    detect_reach_onset,
    detect_power_grasp,
    detect_lateral_pinch,
    detect_contact_onset,
    detect_contact_release,
    detect_finger_curl,
    detect_finger_extend,
    detect_idle,
    detect_transport,
    apply_minimum_duration_filter,
    get_primitive_strategy,
    check_imu_mount_plausibility,
)

STEP = "05_primitives"


# ═══════════════════════════════════════════════════════════════
# MAIN RUN
# ═══════════════════════════════════════════════════════════════


def run(session_id: str) -> list:
    """
    Detect 12 primitives per frame and write primitives.json.
    """
    t0 = time.time()
    print(f"[{STEP}] Starting {session_id}...")

    proc_dir = cfg.PROCESSED_DIR / session_id
    h5_path = proc_dir / "session.h5"
    pose_path = proc_dir / "hand_pose.json"

    if not h5_path.exists():
        raise FileNotFoundError(f"[{STEP}] session.h5 not found: {h5_path}")
    if not pose_path.exists():
        raise FileNotFoundError(f"[{STEP}] hand_pose.json not found: {pose_path}")

    # Load data
    h5_data = read_session_h5(h5_path)
    accel = h5_data["accel"]      # [N, 3]
    gyro = h5_data["gyro"]        # [N, 3]
    timestamps = h5_data["video_timestamps"]  # [N]
    interpolated_over_dropout = h5_data.get(
        "interpolated_over_dropout", np.zeros(len(gyro), dtype=bool)
    )

    with open(pose_path) as f:
        pose_data = json.load(f)

    # ─── Grasp/pinch threshold resolution (v2 addendum §2/§5) ─
    # Priority: a valid (consented, non-expired) per-worker calibration
    # profile > a glove-adjusted default for the session's declared
    # glove_type > the raw global default. A session with no session_meta.json
    # (shouldn't happen in practice — 01_ingest.py always writes one — but
    # tests/ad-hoc scripts sometimes construct a processed dir without one)
    # falls back to the raw default, matching pre-§2/§5 behavior exactly.
    glove_type = cfg.GLOVE_TYPE_DEFAULT
    worker_id = None
    session_meta_path = proc_dir / "session_meta.json"
    if session_meta_path.exists():
        with open(session_meta_path) as f:
            session_meta = json.load(f)
        glove_type = session_meta.get("glove_type", cfg.GLOVE_TYPE_DEFAULT)
        worker_id = session_meta.get("worker_id")

    glove_profile = resolve_grasp_thresholds(glove_type)
    power_grasp_dist = glove_profile["power_grasp_dist"]
    lateral_pinch_dist = glove_profile["lateral_pinch_dist"]
    threshold_source = f"glove_type={glove_profile['glove_type']!r}"

    if worker_id:
        worker_profile = load_worker_profile(worker_id)
        if worker_profile is not None:
            power_grasp_dist = worker_profile["power_grasp_dist"]
            lateral_pinch_dist = worker_profile["lateral_pinch_dist"]
            threshold_source = f"worker_id={worker_id!r} calibration profile"

    print(
        f"[{STEP}] Grasp thresholds: power_grasp_dist={power_grasp_dist:.4f} "
        f"lateral_pinch_dist={lateral_pinch_dist:.4f} (source: {threshold_source}, "
        f"bare defaults: {cfg.POWER_GRASP_DIST:.4f} / {cfg.LATERAL_PINCH_DIST:.4f})"
    )

    # Optional: object tracks (v2 addendum §3, not yet produced by any
    # script — loaded opportunistically so this stays forward-compatible).
    object_tracks_by_frame = {}
    object_tracks_path = proc_dir / "object_tracks.json"
    if object_tracks_path.exists():
        with open(object_tracks_path) as f:
            for entry in json.load(f):
                object_tracks_by_frame[entry.get("frame_idx")] = entry

    n_frames = len(timestamps)

    # Per-frame fps derived from the session's actual video_timestamps
    # (session.h5, written by 02_sync.py from real PTS) rather than assumed
    # from cfg.TARGET_FPS — sessions recorded at a different real frame
    # rate, or with minor frame-timing jitter/drops, would otherwise scale
    # every vision-derived deg/s value (compute_wrist_rotation_from_landmarks
    # / compute_wrist_flexion_from_landmarks) by the wrong factor, breaking
    # calibration against the WRIST_IMU_THRESHOLDS deg/s cutoffs.
    frame_dts = np.diff(timestamps) if n_frames > 1 else np.array([])
    valid_dts = frame_dts[frame_dts > 0]
    mean_fps = float(1.0 / np.mean(valid_dts)) if len(valid_dts) > 0 else cfg.TARGET_FPS
    fps = mean_fps  # session-level fallback for frame 0 and other edge cases

    print(f"[{STEP}] Loaded {n_frames} frames (IMU) + {len(pose_data)} frames (pose)")
    print(f"[{STEP}] Measured fps from video_timestamps: {mean_fps:.2f}")

    # Ensure alignment
    n_frames = min(n_frames, len(pose_data))

    strategy = get_primitive_strategy(cfg.IMU_SOURCE_MODE)
    print(f"[{STEP}] IMU_SOURCE_MODE={cfg.IMU_SOURCE_MODE!r} -> strategy={strategy.name}")
    if cfg.IMU_SOURCE_MODE == "dual":
        print(
            f"[{STEP}] NOTE: 'dual' fusion currently runs against a single "
            f"physical IMU stream duplicated into both the wrist and head "
            f"roles — true independent dual-sensor fusion requires "
            f"01_ingest.py/02_sync.py to parse and sync a second physical "
            f"IMU stream, which is out of scope for this pass."
        )

    # ─── IMU mount plausibility check (advisory) ─────────────
    # The pipeline trusts IMU_SOURCE_MODE to describe where the single
    # ingested stream is physically mounted — there's no second stream to
    # verify against, so a misconfigured device (e.g. a wrist strap on a
    # session labeled "head_mounted") would otherwise silently corrupt
    # idle detection (VisionPrimaryStrategy) or every gyro-based fine-motor
    # primitive (WristPrimaryStrategy) with no error at all.
    mount_check = check_imu_mount_plausibility(cfg.IMU_SOURCE_MODE, gyro, accel)
    if mount_check["checked"] and not mount_check["plausible"]:
        print(f"[{STEP}] ⚠ IMU MOUNT PLAUSIBILITY WARNING: {mount_check['reason']}")
    imu_mount_check = {
        "session_id": session_id,
        "imu_source_mode": cfg.IMU_SOURCE_MODE,
        "single_physical_stream": True,
        "mount_plausibility": mount_check,
    }
    with open(proc_dir / "imu_mount_check.json", "w") as f:
        json.dump(imu_mount_check, f, indent=2)

    # ─── Raw detection pass ──────────────────────────────────
    raw_flags = {prim: [False] * n_frames for prim in ALL_PRIMITIVES}
    # v2 addendum §9 — per-primitive confidence float alongside raw_flags,
    # computed on the RAW (pre-smoothing) detection for every frame, not
    # just frames where the primitive fired. See utils/confidence.py.
    raw_confidences = {prim: [0.0] * n_frames for prim in ALL_PRIMITIVES}
    disagreement_flags = [False] * n_frames
    primitive_source = strategy.name
    contact_onset_history = [False] * n_frames

    for i in range(n_frames):
        # IMU windows (current frame ± 2)
        w_start = max(0, i - 2)
        w_end = min(n_frames, i + 3)

        accel_window = accel[w_start:w_end]
        gyro_window = gyro[w_start:w_end]

        pose_frame = pose_data[i] if i < len(pose_data) else None
        prev_pose = pose_data[i - 1] if i > 0 and i - 1 < len(pose_data) else None
        object_track_frame = object_tracks_by_frame.get(i)

        # Instantaneous fps from this frame pair's real timestamp delta,
        # falling back to the session-mean measured fps at frame 0 or if a
        # non-positive/degenerate delta slips through (e.g. a PTS gap).
        if i > 0:
            instant_dt = float(timestamps[i] - timestamps[i - 1])
            frame_fps = (1.0 / instant_dt) if instant_dt > 0 else mean_fps
        else:
            frame_fps = mean_fps
        dt = 1.0 / frame_fps

        # Pose window for transport (last 10 frames)
        transport_start = max(0, i - 9)
        pose_window = [
            pose_data[j] if j < len(pose_data) else None
            for j in range(transport_start, i + 1)
        ]

        # The pipeline currently ingests a single physical IMU stream (see
        # the 'dual' mode note above) — it's exposed under both the
        # wrist-role and head-role keys so each strategy can consume
        # whichever role it needs.
        imu_window = {
            "accel_window": accel_window,
            "gyro_window": gyro_window,
            "head_accel_window": accel_window,
            "head_gyro_window": gyro_window,
            "dt": dt,
            "fps": frame_fps,
            "pose_window": pose_window,
            "contact_onset_history": contact_onset_history,
            # v2 addendum §2/§5 — glove-adjusted or per-worker-calibrated
            # grasp/pinch thresholds, resolved once per session above.
            "power_grasp_dist": power_grasp_dist,
            "lateral_pinch_dist": lateral_pinch_dist,
        }

        frame_result = strategy.detect_primitives(
            i, pose_frame, prev_pose, imu_window, object_track_frame
        )
        confidence_result = strategy.compute_confidences(
            i, pose_frame, prev_pose, imu_window, object_track_frame
        )

        for prim in ALL_PRIMITIVES:
            raw_flags[prim][i] = bool(frame_result.get(prim, False))
            raw_confidences[prim][i] = float(confidence_result.get(prim, 0.0))
        contact_onset_history[i] = raw_flags["contact_onset"][i]
        disagreement_flags[i] = bool(frame_result.get("disagreement", False))

    # ─── Temporal smoothing ──────────────────────────────────
    # contact_onset/contact_release are instantaneous transition markers,
    # not sustained states — smoothed separately with their own (much
    # shorter) minimum-duration filter so a real 1-frame event survives.
    # See config.CONTACT_EVENT_MIN_FRAMES's comment for why.
    sustained_flags = {p: f for p, f in raw_flags.items() if p not in cfg.CONTACT_EVENT_PRIMITIVES}
    contact_event_flags = {p: f for p, f in raw_flags.items() if p in cfg.CONTACT_EVENT_PRIMITIVES}

    print(f"[{STEP}] Applying minimum duration filter ({cfg.MIN_PRIMITIVE_FRAMES} frames)...")
    smoothed_flags = apply_minimum_duration_filter(sustained_flags, cfg.MIN_PRIMITIVE_FRAMES)
    smoothed_flags.update(
        apply_minimum_duration_filter(contact_event_flags, cfg.CONTACT_EVENT_MIN_FRAMES)
    )

    # ─── Build output ────────────────────────────────────────
    output = []
    for i in range(n_frames):
        active = [p for p in ALL_PRIMITIVES if smoothed_flags[p][i]]
        flags_dict = {p: smoothed_flags[p][i] for p in ALL_PRIMITIVES}

        accel_mag = float(np.linalg.norm(accel[i]))
        gyro_z_deg = float(np.degrees(gyro[i, 2]))

        confidences_dict = {p: round(raw_confidences[p][i], 4) for p in ALL_PRIMITIVES}

        frame_entry = {
            "frame_idx": i,
            "timestamp_sec": round(float(timestamps[i] - timestamps[0]), 4),
            "active_primitives": active,
            "raw_flags": flags_dict,
            "primitive_confidences": confidences_dict,
            "primitive_source": primitive_source,
            "interpolated_over_dropout": bool(interpolated_over_dropout[i]),
            "imu_snapshot": {
                "accel_mag": round(accel_mag, 4),
                "gyro_z_deg_s": round(gyro_z_deg, 2),
            },
        }
        if disagreement_flags[i]:
            frame_entry["disagreement"] = True
        output.append(frame_entry)

    # Stats
    for prim in ALL_PRIMITIVES:
        count = sum(1 for i in range(n_frames) if smoothed_flags[prim][i])
        if count > 0:
            print(f"[{STEP}]   {prim}: {count} frames ({count/n_frames*100:.1f}%)")

    disagreement_count = sum(disagreement_flags)
    if disagreement_count > 0:
        print(f"[{STEP}] ⚠ {disagreement_count} frames flagged with vision/IMU disagreement")

    # Write output
    output_path = proc_dir / "primitives.json"
    with open(output_path, "w") as f:
        json.dump(output, f, separators=(",", ":"))

    size_mb = output_path.stat().st_size / (1024 * 1024)
    elapsed = time.time() - t0
    print(f"[{STEP}] Wrote {len(output)} frames to primitives.json ({size_mb:.1f}MB)")
    print(f"[{STEP}] ✓ Done ({elapsed:.1f}s)")

    return output


def main():
    parser = argparse.ArgumentParser(description="DatraAI Step 05: Primitive Detection")
    parser.add_argument("--session", type=str, required=True, help="Session ID")
    args = parser.parse_args()
    run(Path(args.session).name)


if __name__ == "__main__":
    main()
