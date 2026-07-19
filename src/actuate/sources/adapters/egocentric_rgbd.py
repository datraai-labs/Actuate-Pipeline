"""Adapter: Lo6yu/egocentric_dataset (egocentric RGB-D + EMG + IMU) -> canonical episode.

This dataset already carries REAL sensor data our perception would only re-estimate worse:
per-frame metric 3D hand keypoints from a depth camera (`joints_3d_camera`), real depth,
per-finger contact force, and semantic action segments. WiLoR does not even detect hands on
this footage. So the right move is to INGEST its native annotations into our canonical schema
-- not run our L1 perception on it.

    dense hand_keypoints.jsonl (1229 frames, 21x3 metric)  -> keypoints_3d + wrist_pose
    semantic_segments.parquet (task/subtask labels + spans) -> task + subtasks
    the aligned concat RGB video                            -> observation.images

Frame index: the dense keypoints are on `global_frame_index` (0..N-1), and the overlay concat
video is exactly N frames -- they align 1:1. We use `global_frame_index` as the canonical
frame_idx so keypoints and RGB stay paired.

Provenance: keypoints are still camera-derived (vision_primary), but backed by a real depth
sensor -- far better than our monocular estimate. Licence: this dataset's own terms apply; it
is NOT MANO/WiLoR, so it does not inherit that non-commercial blocker.
"""

from __future__ import annotations

import json
from pathlib import Path

from actuate.config import Provenance, RigType, Side


def _wrist_pose(landmarks: list[list[float]]):
    """SE3 wrist pose from 21 metric hand landmarks -- reuses the io-layer hand frame.

    Inlined here (rather than importing canonical, which sits ABOVE sources in the layer
    graph) so the adapter stays a proper source-layer citizen: it depends only on io + schema.
    """
    import numpy as np

    from actuate.io.geometry import hand_frame_quaternion
    from actuate.schema.frame import SE3

    lm = np.asarray(landmarks, dtype=np.float64)
    q = hand_frame_quaternion(lm)
    if q is None:
        return None
    p = lm[0]
    return SE3(position_m=(float(p[0]), float(p[1]), float(p[2])), quaternion_wxyz=q)

_REPO = "Lo6yu/egocentric_dataset"
_KP_JSONL = "analysis/hand_keypoints/pose_jsonl/hand_keypoints.jsonl"
_SEGMENTS = "analysis/semantic_subtasks/subtask_segments.jsonl"
_OVERLAY = "analysis/hand_keypoints/preview/hand_cream_lossless_concat_hand_keypoints_overlay.mp4"
_HANDEDNESS = {"right": Side.RIGHT, "left": Side.LEFT}


def _hf(repo_id: str, rel: str):
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_id, filename=rel, repo_type="dataset"))


def adapt(package: str, work_root: Path, *, repo_id: str = _REPO,
          max_frames: int | None = None, clean_rgb: Path | None = None):
    """Build a CanonicalEpisode from this dataset's native annotations.

    Returns (episode, session_dir) -- the session dir holds the staged RGB video + meta so the
    existing exporters run unchanged. `clean_rgb` overrides the video (else the aligned concat).
    """
    from actuate.ingest.content_address import hash_file
    from actuate.ingest.run import ensure_session_meta
    from actuate.schema import CanonicalEpisode
    from actuate.schema.frame import CanonicalFrame, HandState

    pkg = f"packages/{package}"
    kp_lines = _hf(repo_id, f"{pkg}/{_KP_JSONL}").read_text(encoding="utf-8").splitlines()

    # stage the aligned RGB video into a session dir
    video_src = Path(clean_rgb) if clean_rgb else _hf(repo_id, f"{pkg}/{_OVERLAY}")
    session = Path(work_root) / package.replace("/", "_")
    session.mkdir(parents=True, exist_ok=True)
    video = session / "compressed.mp4"
    if not video.exists():
        import shutil

        shutil.copy2(video_src, video)
    ensure_session_meta(session)
    capture_id = hash_file(video)
    episode_id = f"{capture_id[:16]}_ep00"

    prov = {"hands": Provenance.VISION_PRIMARY,
            "hands.keypoints_3d": Provenance.VISION_PRIMARY}

    frames = []
    n = 0
    for line in kp_lines:
        rec = json.loads(line)
        idx = int(rec["global_frame_index"])
        if max_frames and n >= max_frames:
            break
        hands: dict = {}
        conf: dict = {}
        for h in rec.get("hands", []):
            side = _HANDEDNESS.get(str(h.get("handedness", "")).lower())
            kp = h.get("joints_3d_camera")
            if side is None or not kp or len(kp) != 21:
                continue
            kp3 = [[float(a) for a in p] for p in kp]
            hands[side] = HandState(
                keypoints_3d=tuple(tuple(p) for p in kp3),
                wrist_pose=_wrist_pose(kp3))
            conf["hands"] = float(h.get("detector_confidence", 0.5))
        if not hands:
            continue
        frames.append(CanonicalFrame(
            t=float(rec.get("time_s", idx / 30.0)), rig=RigType.HEAD_MOUNTED,
            episode_id=episode_id, frame_idx=idx, hands=hands,
            confidence=conf, provenance=prov))
        n += 1

    if not frames:
        raise ValueError(f"{package}: no frames with 21-joint hands in {_KP_JSONL}")

    # task + subtasks from the dataset's own semantic segments (real labels)
    task, subtasks = _load_segments(repo_id, pkg, {f.frame_idx for f in frames})

    ep = CanonicalEpisode(
        episode_id=episode_id, capture_id=capture_id, source_content_hash=capture_id,
        rig=RigType.HEAD_MOUNTED, frames=tuple(frames), task=task,
        subtasks=tuple(subtasks),
        derivation_notes={"source": f"adapted from {repo_id}/{package}: native metric 3D "
                          "hand keypoints (depth-camera), not our WiLoR/UniDepth perception."})
    return ep, session


def _load_segments(repo_id: str, pkg: str, frame_ids: set[int]):
    """(task string, [Subtask]) from subtask_segments.jsonl -- real labels + spans."""
    from actuate.schema.episode import Subtask

    try:
        lines = _hf(repo_id, f"{pkg}/{_SEGMENTS}").read_text(encoding="utf-8").splitlines()
    except Exception:
        return None, []
    lo, hi = (min(frame_ids), max(frame_ids)) if frame_ids else (0, 0)
    task = None
    subtasks = []
    for line in lines:
        s = json.loads(line)
        task = task or s.get("task_description_en")
        start = int(s.get("start_rgb_frame", s.get("start_frame", 0)))
        end = int(s.get("end_rgb_frame", s.get("end_frame", start)))
        # clamp to the keypoint frame range (segments are rgb-indexed; approximate overlap)
        start, end = max(start, lo), min(end, hi)
        if end < start:
            continue
        instr = s.get("subtask_label_en") or s.get("subtask_description_en")
        if instr:
            subtasks.append(Subtask(instruction=str(instr), start_frame=start,
                                    end_frame=end, confidence=0.9))
    return task, subtasks
