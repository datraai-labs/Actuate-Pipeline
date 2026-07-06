"""
DatraAI Pipeline — Step 08: Validate
Motion-label consistency cross-checks.

Input:  processed/{session_id}/session.h5
        processed/{session_id}/phases.json
        processed/{session_id}/task_label.json
        processed/{session_id}/primitives.json
        processed/{session_id}/hand_pose.json
Output: processed/{session_id}/validation_report.json
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

STEP = "08_validate"


def _validate_imu_spike_at_contact(
    primitives: list,
    accel: np.ndarray,
    n_frames: int,
) -> dict:
    """
    VALIDATION 1: For every contact_onset frame, check that there's an accel spike nearby.
    """
    contact_frames = [
        p["frame_idx"] for p in primitives
        if "contact_onset" in p.get("active_primitives", [])
    ]

    if len(contact_frames) == 0:
        return {
            "passed": True,
            "spike_found_rate": 1.0,
            "details": "No contact_onset frames found (vacuously true).",
        }

    spikes_found = 0
    for cf in contact_frames:
        w_start = max(0, cf - 5)
        w_end = min(n_frames, cf + 6)
        window = accel[w_start:w_end]
        accel_mags = np.linalg.norm(window, axis=1)
        if np.any(accel_mags > cfg.CONTACT_ACCEL_G):
            spikes_found += 1

    rate = spikes_found / len(contact_frames)
    passed = rate > cfg.VALIDATION_SPIKE_RATE_MIN

    return {
        "passed": passed,
        "spike_found_rate": round(rate, 4),
        "contact_onset_count": len(contact_frames),
        "spikes_found": spikes_found,
    }


def _validate_phase_boundary_velocity(
    phases: dict,
    hand_pose: list,
) -> dict:
    """
    VALIDATION 2: At reach→grasp transitions, wrist velocity should be decreasing.
    """
    segments = phases.get("segments", [])
    boundaries = []

    for i in range(len(segments) - 1):
        if segments[i]["phase"] == "reach" and segments[i + 1]["phase"] == "grasp":
            boundaries.append(segments[i]["end_frame"])

    if len(boundaries) == 0:
        return {
            "passed": True,
            "mean_decel_rate": 0.0,
            "details": "No reach→grasp transitions found.",
        }

    boundary_correct = []
    for bf in boundaries:
        # Get wrist velocity for 10 frames before boundary
        start_f = max(0, bf - 10)
        velocities = []
        for f_idx in range(start_f, bf + 1):
            if f_idx < len(hand_pose) and hand_pose[f_idx].get("derived"):
                vel = hand_pose[f_idx]["derived"].get("wrist_velocity_magnitude", 0.0)
                velocities.append(vel)

        if len(velocities) >= 2:
            decel_rate = (velocities[-1] - velocities[0]) / len(velocities)
            boundary_correct.append(decel_rate < 0)
        else:
            boundary_correct.append(True)  # Not enough data, assume ok

    correct_rate = sum(boundary_correct) / len(boundary_correct) if boundary_correct else 1.0
    mean_decel = np.mean(
        [
            (hand_pose[min(bf, len(hand_pose) - 1)].get("derived", {}).get("wrist_velocity_magnitude", 0) -
             hand_pose[max(0, bf - 10)].get("derived", {}).get("wrist_velocity_magnitude", 0)) / 10.0
            for bf in boundaries
            if bf < len(hand_pose) and hand_pose[bf].get("derived")
               and max(0, bf - 10) < len(hand_pose) and hand_pose[max(0, bf - 10)].get("derived")
        ]
    ) if boundaries else 0.0

    passed = correct_rate > cfg.VALIDATION_DECEL_RATE_MIN

    return {
        "passed": passed,
        "mean_decel_rate": round(float(mean_decel), 6),
        "boundary_count": len(boundaries),
        "correct_count": sum(boundary_correct),
    }


def _validate_causal_ordering(
    primitives: list,
    fps: float,
) -> dict:
    """
    VALIDATION 3: IMU contact spike should precede or closely follow grasp.
    """
    contact_frames = []
    grasp_frames = []

    for p in primitives:
        active = p.get("active_primitives", [])
        if "contact_onset" in active:
            contact_frames.append(p["frame_idx"])
        if "power_grasp" in active or "lateral_pinch" in active:
            grasp_frames.append(p["frame_idx"])

    if len(contact_frames) == 0:
        return {
            "passed": True,
            "inversion_count": 0,
            "total_events": 0,
            "details": "No contact events found.",
        }

    # For each contact event, find nearest grasp activation
    causal_inversions = []
    inversion_frames = []

    for cf in contact_frames:
        if len(grasp_frames) == 0:
            continue
        # Find nearest grasp frame
        idx = np.searchsorted(grasp_frames, cf)
        candidates = []
        if idx < len(grasp_frames):
            candidates.append(grasp_frames[idx])
        if idx > 0:
            candidates.append(grasp_frames[idx - 1])

        nearest_grasp = min(candidates, key=lambda gf: abs(gf - cf))
        time_diff_ms = (nearest_grasp - cf) / fps * 1000.0

        causal_ok = cfg.CAUSAL_MIN_OFFSET_MS < time_diff_ms < cfg.CAUSAL_MAX_OFFSET_MS
        if not causal_ok:
            causal_inversions.append(time_diff_ms)
            inversion_frames.append(cf)

    total_events = len(contact_frames)
    inversion_count = len(causal_inversions)
    inversion_rate = inversion_count / total_events if total_events > 0 else 0.0
    passed = inversion_rate < cfg.VALIDATION_CAUSAL_INVERSION_MAX_RATE

    return {
        "passed": passed,
        "inversion_count": inversion_count,
        "total_events": total_events,
        "inversion_rate": round(inversion_rate, 4),
        "inversion_frames": inversion_frames[:20],  # Cap for readability
    }


def _validate_task_phase_consistency(
    task_label: dict,
    phases: dict,
) -> dict:
    """
    VALIDATION 4: Check that phase proportions match task expectations.
    """
    task = task_label.get("L1_task", "unknown")
    frame_labels = phases.get("frame_labels", [])
    n_frames = len(frame_labels)

    if n_frames == 0 or task not in cfg.TASK_PHASE_EXPECTATIONS:
        return {
            "passed": True,
            "details": f"No phase expectations defined for task '{task}'.",
        }

    expectations = cfg.TASK_PHASE_EXPECTATIONS[task]
    results = {}
    all_met = True

    for phase, min_proportion in expectations.items():
        count = sum(1 for l in frame_labels if l == phase)
        proportion = count / n_frames if n_frames > 0 else 0.0
        met = proportion >= min_proportion
        results[f"{phase}_proportion"] = round(proportion, 4)
        results[f"{phase}_expected_min"] = min_proportion
        results[f"{phase}_met"] = met
        if not met:
            all_met = False

    return {
        "passed": all_met,
        **results,
    }


def run(session_id: str) -> dict:
    """
    Run all 4 cross-signal validation checks and write validation_report.json.
    """
    t0 = time.time()
    print(f"[{STEP}] Starting {session_id}...")

    proc_dir = cfg.PROCESSED_DIR / session_id

    # Load all inputs
    h5_data = read_session_h5(proc_dir / "session.h5")
    accel = h5_data["accel"]
    n_frames = len(accel)
    fps = h5_data["metadata"].get("fps_nominal", cfg.TARGET_FPS)

    with open(proc_dir / "phases.json") as f:
        phases = json.load(f)

    with open(proc_dir / "task_label.json") as f:
        task_label = json.load(f)

    with open(proc_dir / "primitives.json") as f:
        primitives = json.load(f)

    # Load hand_pose for velocity checks
    pose_path = proc_dir / "hand_pose.json"
    hand_pose = []
    if pose_path.exists():
        with open(pose_path) as f:
            hand_pose = json.load(f)

    # ─── Run validations ─────────────────────────────────────
    print(f"[{STEP}] Running IMU spike validation...")
    v1 = _validate_imu_spike_at_contact(primitives, accel, n_frames)
    print(f"[{STEP}]   Spike rate: {v1.get('spike_found_rate', 'N/A')} ({'PASS' if v1['passed'] else 'FAIL'})")

    print(f"[{STEP}] Running phase boundary velocity check...")
    v2 = _validate_phase_boundary_velocity(phases, hand_pose)
    print(f"[{STEP}]   Mean decel: {v2.get('mean_decel_rate', 'N/A')} ({'PASS' if v2['passed'] else 'FAIL'})")

    print(f"[{STEP}] Running causal ordering check...")
    v3 = _validate_causal_ordering(primitives, fps)
    print(f"[{STEP}]   Inversions: {v3.get('inversion_count', 0)}/{v3.get('total_events', 0)} ({'PASS' if v3['passed'] else 'FAIL'})")

    print(f"[{STEP}] Running task-phase consistency check...")
    v4 = _validate_task_phase_consistency(task_label, phases)
    print(f"[{STEP}]   Consistency: {'PASS' if v4['passed'] else 'FAIL'}")

    overall_valid = all([v1["passed"], v2["passed"], v3["passed"], v4["passed"]])

    flags = []
    if not v1["passed"]:
        flags.append("imu_spike_mismatch")
    if not v2["passed"]:
        flags.append("phase_boundary_velocity_issue")
    if not v3["passed"]:
        flags.append("causal_inversion")
    if not v4["passed"]:
        flags.append("task_phase_inconsistency")

    report = {
        "session_id": session_id,
        "overall_valid": overall_valid,
        "checks": {
            "imu_spike_at_contact": v1,
            "phase_boundary_velocity": v2,
            "causal_ordering": v3,
            "task_phase_consistency": v4,
        },
        "flags": flags,
        "causal_inversion_frames": v3.get("inversion_frames", []),
    }

    output_path = proc_dir / "validation_report.json"
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    elapsed = time.time() - t0
    status = "VALID" if overall_valid else f"INVALID (flags: {flags})"
    print(f"[{STEP}] Validation: {status}")
    print(f"[{STEP}] ✓ Done ({elapsed:.1f}s)")

    return report


def main():
    parser = argparse.ArgumentParser(description="DatraAI Step 08: Validation")
    parser.add_argument("--session", type=str, required=True, help="Session ID")
    args = parser.parse_args()
    run(Path(args.session).name)


if __name__ == "__main__":
    main()
