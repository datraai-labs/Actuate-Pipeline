"""
DatraAI Pipeline — Video Utilities
FFmpeg/ffprobe wrappers for compression, PTS extraction, metadata, and frame extraction.
"""

import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as cfg


def resolve_perception_source(session_id: str) -> Path:
    """
    Resolve which video file a perception script (04_hand_pose.py,
    04c_object_track.py, 04d_depth_estimate.py) should read, per
    config.PERCEPTION_SOURCE (v2 addendum §8) — so the choice lives in one
    place instead of each script hardcoding "compressed.mp4".

    "compressed" -> processed/{session_id}/compressed.mp4
    "raw"        -> raw/{session_id}/raw.mp4

    Raises FileNotFoundError if the resolved path doesn't exist — e.g.
    PERCEPTION_SOURCE="raw" after raw.mp4 was deleted
    (config.KEEP_RAW_AFTER_COMPRESSION=False). Silently falling back to
    the other source would defeat the point of the toggle (a caller who
    asked for raw wants raw, not an unannounced substitution) and could
    silently reintroduce compression artifacts the toggle exists to avoid.
    """
    source = getattr(cfg, "PERCEPTION_SOURCE", "compressed")
    if source == "raw":
        path = cfg.RAW_DIR / session_id / "raw.mp4"
    elif source == "compressed":
        path = cfg.PROCESSED_DIR / session_id / "compressed.mp4"
    else:
        raise ValueError(f"Unknown PERCEPTION_SOURCE: {source!r} — expected 'raw' or 'compressed'.")

    if not path.exists():
        raise FileNotFoundError(
            f"resolve_perception_source: PERCEPTION_SOURCE={source!r} resolved to "
            f"{path}, which does not exist."
        )
    return path


def _check_ffmpeg() -> None:
    """Verify ffmpeg and ffprobe are available on PATH."""
    for tool in ("ffmpeg", "ffprobe"):
        try:
            subprocess.run(
                [tool, "-version"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
        except FileNotFoundError:
            print(
                f"[video_utils] ERROR: '{tool}' not found on PATH.\n"
                f"  Install FFmpeg: https://ffmpeg.org/download.html\n"
                f"  On Ubuntu: sudo apt install ffmpeg\n"
                f"  On macOS: brew install ffmpeg",
                file=sys.stderr,
            )
            raise RuntimeError(f"'{tool}' is not installed or not on PATH.")


def compress_video(
    input_path: Path,
    output_path: Path,
    crf: int = 23,
    preset: str = "slow",
    codec: str = "libx265",
) -> Dict[str, float]:
    """
    Compress video to H.265 using FFmpeg.

    Returns dict with raw_size_mb, compressed_size_mb, reduction_pct.
    """
    _check_ffmpeg()
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    raw_size_mb = input_path.stat().st_size / (1024 * 1024)

    cmd = [
        "ffmpeg",
        "-y",                     # overwrite output
        "-i", str(input_path),
        "-c:v", codec,
        "-crf", str(crf),
        "-preset", preset,
        "-c:a", "aac",
        "-b:a", "128k",
        "-tag:v", "hvc1",         # compatibility tag for H.265
        "-progress", "pipe:1",
        str(output_path),
    ]

    process = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if process.returncode != 0:
        raise RuntimeError(
            f"FFmpeg compression failed (exit code {process.returncode}):\n"
            f"{process.stderr[-2000:]}"
        )

    if not output_path.exists():
        raise RuntimeError(f"FFmpeg ran but output file not found: {output_path}")

    compressed_size_mb = output_path.stat().st_size / (1024 * 1024)
    reduction_pct = (1 - compressed_size_mb / raw_size_mb) * 100 if raw_size_mb > 0 else 0.0

    return {
        "raw_size_mb": round(raw_size_mb, 2),
        "compressed_size_mb": round(compressed_size_mb, 2),
        "reduction_pct": round(reduction_pct, 1),
    }


def extract_pts(video_path: Path) -> np.ndarray:
    """
    Extract per-frame PTS (Presentation Timestamps) in seconds using ffprobe.

    Returns float64 array of PTS values, normalized to start at 0.0.
    Asserts monotonicity. Flags large gaps (> 200ms).
    """
    _check_ffmpeg()
    video_path = Path(video_path)

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-select_streams", "v:0",
        "-show_entries", "frame=pkt_pts_time,pts_time",
        "-of", "csv=p=0",
        str(video_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, check=True)

    pts_list = []
    for line in result.stdout.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        val = None
        for part in parts:
            if part and part.upper() != "N/A":
                try:
                    val = float(part)
                    break
                except ValueError:
                    continue
        if val is not None:
            pts_list.append(val)

    if len(pts_list) == 0:
        raise ValueError("No valid PTS values extracted from video.")

    pts = np.array(pts_list, dtype=np.float64)

    # Assert monotonicity
    diffs = np.diff(pts)
    if not np.all(diffs > 0):
        non_mono_indices = np.where(diffs <= 0)[0]
        raise ValueError(
            f"PTS is not monotonically increasing. "
            f"Non-monotonic at indices: {non_mono_indices[:10].tolist()} "
            f"(showing first 10). Values: {pts[non_mono_indices[:5]].tolist()}"
        )

    # Detect large gaps (> 200ms)
    gap_threshold = 0.200  # 200ms
    large_gaps = np.where(diffs > gap_threshold)[0]
    if len(large_gaps) > 0:
        print(
            f"[video_utils] WARNING: {len(large_gaps)} large PTS gaps (>200ms) detected "
            f"at frame indices: {large_gaps[:10].tolist()}"
        )

    # Normalize to start at 0.0
    pts = pts - pts[0]

    return pts


def get_video_metadata(video_path: Path) -> Dict:
    """
    Extract video metadata using ffprobe: duration, fps, frame count, creation_time.
    """
    _check_ffmpeg()
    video_path = Path(video_path)

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(video_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    probe = json.loads(result.stdout)

    # Find video stream
    video_stream = None
    for stream in probe.get("streams", []):
        if stream.get("codec_type") == "video":
            video_stream = stream
            break

    if video_stream is None:
        raise ValueError(f"No video stream found in {video_path}")

    # Parse FPS
    fps_str = video_stream.get("r_frame_rate", "30/1")
    if "/" in fps_str:
        num, den = fps_str.split("/")
        fps = float(num) / float(den) if float(den) != 0 else 30.0
    else:
        fps = float(fps_str)

    # Parse duration
    duration = float(video_stream.get("duration", probe.get("format", {}).get("duration", 0)))

    # Parse frame count
    frame_count = int(video_stream.get("nb_frames", 0))
    if frame_count == 0:
        frame_count = int(fps * duration) if duration > 0 else 0

    # Parse creation_time
    creation_time = None
    tags = video_stream.get("tags", {})
    if "creation_time" in tags:
        creation_time = tags["creation_time"]
    elif "creation_time" in probe.get("format", {}).get("tags", {}):
        creation_time = probe["format"]["tags"]["creation_time"]

    return {
        "fps": round(fps, 2),
        "duration_seconds": round(duration, 3),
        "frame_count": frame_count,
        "width": int(video_stream.get("width", 0)),
        "height": int(video_stream.get("height", 0)),
        "codec": video_stream.get("codec_name", "unknown"),
        "creation_time": creation_time,
    }


def extract_frame(video_path: Path, frame_idx: int) -> np.ndarray:
    """
    Extract a single frame by index from video. Returns BGR numpy array.
    """
    import cv2

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        raise ValueError(f"Failed to read frame {frame_idx} from {video_path}")

    return frame


def generate_preview_clip(
    video_path: Path,
    output_path: Path,
    start_sec: float = 60.0,
    duration_sec: float = 30.0,
) -> None:
    """
    Generate a short preview clip from the video.
    """
    _check_ffmpeg()
    video_path = Path(video_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-ss", str(start_sec),
        "-i", str(video_path),
        "-t", str(duration_sec),
        "-c:v", "libx264",
        "-crf", "28",
        "-preset", "fast",
        "-c:a", "aac",
        str(output_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Preview clip generation failed:\n{result.stderr[-1000:]}")


# ═══════════════════════════════════════════════════════════════
# VISION-DERIVED WRIST ROTATION (v2 addendum §1)
#
# When only a head-mounted IMU is available, gyro-based wrist
# pronation/supination/flexion detection is physically impossible (the
# sensor is on the head, not the wrist). These functions derive an
# equivalent angular-velocity signal from hand-pose landmarks so
# VisionPrimaryStrategy in utils/imu_source_router.py can substitute for
# the gyro-based detectors in scripts/05_primitives.py.
# ═══════════════════════════════════════════════════════════════


def _dominant_hand_wrist_mcp_vector(pose_frame: Optional[dict]):
    """
    Extract the (wrist landmark 0 -> middle-finger-MCP landmark 9) vector
    for the dominant hand of one hand_pose.json frame entry.

    Returns a 3-element list, or None if the frame has no usable dominant
    hand landmark set.
    """
    if not pose_frame or not pose_frame.get("hands_detected"):
        return None
    dom = pose_frame.get("dominant_hand")
    if not dom:
        return None
    hand = pose_frame.get(f"{dom}_hand")
    if not hand:
        return None
    landmarks = hand.get("landmarks")
    if not landmarks or len(landmarks) <= 9:
        return None
    wrist = landmarks[0]
    middle_mcp = landmarks[9]
    return [middle_mcp[i] - wrist[i] for i in range(3)]


def _wrap_angle(delta: float) -> float:
    """Wrap an angle delta (radians) to [-pi, pi]."""
    return (delta + math.pi) % (2 * math.pi) - math.pi


def compute_wrist_rotation_from_landmarks(
    pose_frame: Optional[dict],
    prev_pose_frame: Optional[dict],
    fps: float,
) -> Optional[float]:
    """
    Vision-derived proxy for wrist pronation/supination angular velocity
    (deg/s equivalent), substituting for gyro Z when only a head-mounted
    IMU is available.

    Derived from the frame-to-frame change in the azimuthal (x/y-plane)
    orientation of the wrist(0)->middle-MCP(9) vector — this vector rotates
    about the forearm axis as the wrist pronates/supinates. Sign follows the
    same convention as gyro_z: positive = pronation, negative = supination.

    Returns None if either frame lacks a usable dominant-hand landmark set,
    or if fps is non-positive.
    """
    if fps <= 0:
        return None

    curr_vec = _dominant_hand_wrist_mcp_vector(pose_frame)
    prev_vec = _dominant_hand_wrist_mcp_vector(prev_pose_frame)
    if curr_vec is None or prev_vec is None:
        return None

    curr_angle = math.atan2(curr_vec[1], curr_vec[0])
    prev_angle = math.atan2(prev_vec[1], prev_vec[0])
    delta = _wrap_angle(curr_angle - prev_angle)

    dt = 1.0 / fps
    return math.degrees(delta) / dt


def compute_wrist_flexion_from_landmarks(
    pose_frame: Optional[dict],
    prev_pose_frame: Optional[dict],
    fps: float,
) -> Optional[float]:
    """
    Vision-derived proxy for wrist flexion/extension angular velocity
    (deg/s equivalent), substituting for gyro X when only a head-mounted
    IMU is available.

    Derived from the frame-to-frame change in elevation angle (out-of-plane,
    using MediaPipe's relative z) of the wrist(0)->middle-MCP(9) vector —
    approximates the wrist bending toward/away from the camera as it flexes.
    Coarser than a true gyro signal since MediaPipe's z is a relative depth
    estimate, not a metric one.
    """
    if fps <= 0:
        return None

    curr_vec = _dominant_hand_wrist_mcp_vector(pose_frame)
    prev_vec = _dominant_hand_wrist_mcp_vector(prev_pose_frame)
    if curr_vec is None or prev_vec is None:
        return None

    def _elevation(vec):
        planar = math.sqrt(vec[0] ** 2 + vec[1] ** 2)
        if planar == 0.0 and vec[2] == 0.0:
            return 0.0
        return math.atan2(-vec[2], planar)

    delta = _wrap_angle(_elevation(curr_vec) - _elevation(prev_vec))

    dt = 1.0 / fps
    return math.degrees(delta) / dt
