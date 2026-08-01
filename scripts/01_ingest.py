"""
DatraAI Pipeline — Step 01: Ingest
FFmpeg compress + PTS extract + IMU parse + session metadata.

Input:  raw/{session_id}/raw.mp4 + imu.csv
Output: processed/{session_id}/compressed.mp4
        processed/{session_id}/pts.npy
        processed/{session_id}/imu_raw.npy
        processed/{session_id}/session_meta.json

raw/{session_id}/raw.mp4 is deleted after compression unless
config.KEEP_RAW_AFTER_COMPRESSION is True (default) or
config.PERCEPTION_SOURCE == "raw" (v2 addendum §8) — see run()'s "Step 1b".
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running standalone or as module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config as cfg
from utils.video_utils import compress_video, extract_pts, get_video_metadata

STEP = "01_ingest"


def _read_consent_status(session_path: Path) -> str:
    """
    Consent status (v2 addendum §10) — read from an optional
    raw/{session_id}/consent.json (e.g. {"status": "granted"}) produced by
    whatever intake/consent-form process precedes ingest. Defaults to
    "pending" — the SAFE default, since run_pipeline.py's delivery gate
    (config.BLOCK_DELIVERY_WITHOUT_CONSENT) blocks packaging on anything
    other than "granted". A session is never assumed consented by default.
    """
    consent_path = session_path / "consent.json"
    if not consent_path.exists():
        return "pending"
    with open(consent_path) as f:
        return json.load(f).get("status", "pending")


def _read_glove_and_worker_config(session_path: Path):
    """
    Glove type + worker ID (v2 addendum §2/§5) — read from an optional
    raw/{session_id}/session_config.json (mirrors _read_consent_status's
    pattern). glove_type defaults to config.GLOVE_TYPE_DEFAULT ("none") —
    a session is never assumed gloved. worker_id defaults to None — a
    session with no declared worker gets no per-worker calibration lookup
    (scripts/05_primitives.py falls back to the glove-adjusted or raw
    default in that case).

    Returns (glove_type, worker_id).
    """
    session_config_path = session_path / "session_config.json"
    if not session_config_path.exists():
        return cfg.GLOVE_TYPE_DEFAULT, None
    with open(session_config_path) as f:
        session_config = json.load(f)
    glove_type = session_config.get("glove_type", cfg.GLOVE_TYPE_DEFAULT)
    worker_id = session_config.get("worker_id")
    return glove_type, worker_id


_EPOCH_2000_MS = 946_684_800_000.0   # 2000-01-01T00:00:00Z — no capture predates this
_UPTIME_CEILING_MS = 1.0e10          # ~115 days — no plausible wall-clock is this SMALL


def _classify_clock_domain(t_ms: float) -> str:
    """'epoch' (Unix wall-clock ms), 'uptime' (device boot-relative ms), or 'ambiguous'.

    The real corpus's IMU starts at 128,642 ms = 128.6 s — an uptime clock. Read
    as epoch that is 1970-01-01T00:02:08; anchoring a wall-clock video against it
    would put the streams ~56 years apart.
    """
    if t_ms >= _EPOCH_2000_MS:
        return "epoch"
    if 0 <= t_ms < _UPTIME_CEILING_MS:
        return "uptime"
    return "ambiguous"


def _assess_temporal_anchor(
    creation_time_epoch_ms,
    imu_t0_ms: float,
    imu_t1_ms: float,
    video_duration_s: float,
    now_epoch_ms: float | None = None,
) -> tuple[float, dict]:
    """Derive the recording-start anchor and HONESTLY state how much it proves.

    Returns (recording_start_epoch_ms, meta_fields). Pure — unit-testable without
    ffmpeg. The audit (2026-08-01 §3b) showed the old fallback was tautological:
    defining video start := IMU t[0] forces alignment by construction, and the
    drift check then "validates" what the anchor guaranteed. Any constant anchor
    error under ~10 ms was invisible (and a 10 ms error even *improved* the drift
    number). So:

    - creation_time present + epoch-domain IMU + ranges overlap -> anchor is
      CROSS-VALIDATED at range level (constant sub-frame offsets remain invisible;
      the note says so).
    - creation_time present + uptime-domain IMU -> the two clock domains CANNOT be
      anchored against each other; raises rather than fabricating an alignment.
    - creation_time present + epoch IMU but DISJOINT ranges -> provably wrong
      anchor; raises.
    - no (or implausible) creation_time -> IMU-t0 fallback, explicitly marked
      temporal_alignment_validated=False. The drift check cannot detect a wrong
      anchor in this mode and downstream certificates must say so.
    """
    if now_epoch_ms is None:
        now_epoch_ms = datetime.now(timezone.utc).timestamp() * 1000.0

    imu_domain = _classify_clock_domain(imu_t0_ms)
    fields = {"imu_clock_domain": imu_domain}

    creation_rejected = None
    if creation_time_epoch_ms is not None and not (
        _EPOCH_2000_MS <= creation_time_epoch_ms <= now_epoch_ms + 86_400_000.0
    ):
        creation_rejected = (
            f"container creation_time {creation_time_epoch_ms:.0f}ms is not a "
            "plausible wall-clock value (before 2000 or in the future) — ignored"
        )
        creation_time_epoch_ms = None

    if creation_time_epoch_ms is not None:
        if imu_domain == "uptime":
            raise ValueError(
                f"[{STEP}] Clock-domain conflict: video creation_time is a wall-clock "
                f"epoch ({creation_time_epoch_ms:.0f}ms) but the IMU timestamps are an "
                f"uptime clock (t0={imu_t0_ms:.0f}ms ≈ {imu_t0_ms/1000.0:.1f}s after "
                "boot). These cannot be anchored against each other; 02_sync would "
                "align streams decades apart. Record the device's boot epoch (or a "
                "shared sync event) in session_config.json, or strip the container "
                "creation_time to use the IMU-t0 fallback deliberately."
            )
        video_t1 = creation_time_epoch_ms + video_duration_s * 1000.0
        overlap_ms = min(video_t1, imu_t1_ms) - max(creation_time_epoch_ms, imu_t0_ms)
        if imu_domain == "epoch" and overlap_ms <= 0:
            raise ValueError(
                f"[{STEP}] Anchor implausible: video range "
                f"[{creation_time_epoch_ms:.0f}, {video_t1:.0f}]ms and IMU range "
                f"[{imu_t0_ms:.0f}, {imu_t1_ms:.0f}]ms do not overlap at all — the "
                "creation_time anchor is provably wrong for this IMU stream."
            )
        fields.update(
            temporal_alignment_validated=True,
            temporal_alignment_note=(
                "anchor from container creation_time; cross-validated against an "
                f"independent epoch-domain IMU at range level (overlap {overlap_ms:.0f}ms). "
                "Constant sub-frame offsets (exposure/readout latency) remain invisible "
                "to this check — see timestamp_semantics."
            ),
        )
        return float(creation_time_epoch_ms), fields

    note = (
        "anchor DEFINED as IMU t[0] (no usable container creation_time): video/IMU "
        "alignment holds by construction, so the sync drift check cannot validate it "
        "and any constant anchor error is invisible. Temporal alignment is UNVALIDATED."
    )
    if creation_rejected:
        note = creation_rejected + "; " + note
    fields.update(temporal_alignment_validated=False, temporal_alignment_note=note)
    return float(imu_t0_ms), fields


#: What a timestamp MEANS — recorded explicitly, defaulting to None (= unknown),
#: never an implicit zero. A rolling-shutter phone camera's exposure-midpoint
#: offset alone is typically 10–30 ms — exactly the band the drift check cannot
#: see (audit §3c). Declared per-device via session_config.json's
#: "timestamp_semantics" object; 02_sync applies camera_to_imu_latency_ms when
#: it is a number and records the zero assumption when it is None.
_TIMESTAMP_SEMANTICS_DEFAULTS = {
    "video_frame_time_meaning": None,   # start_of_exposure | mid_exposure | end_of_readout
    "imu_sample_time_meaning": None,    # sample_time | arrival_time
    "camera_to_imu_latency_ms": None,   # None = UNKNOWN — not zero
}


def _read_timestamp_semantics(session_path: Path) -> dict:
    semantics = dict(_TIMESTAMP_SEMANTICS_DEFAULTS)
    config_path = session_path / "session_config.json"
    if config_path.exists():
        with open(config_path) as f:
            declared = json.load(f).get("timestamp_semantics", {})
        for key in semantics:
            if key in declared:
                semantics[key] = declared[key]
    return semantics


#: Optional IMU channels carried through when the source provides them (audit
#: 2026-08-01, Group 4): the legacy path used to coerce to 7 columns and silently
#: DROP magnetometer + temperature that the modern path (actuate.ingest.imu)
#: keeps — the two paths disagreed about what an IMU is. Absent channels are NaN
#: (= not measured), never zero.
IMU_OPTIONAL_COLS = ["mx", "my", "mz", "temp_c"]


def _parse_imu(raw_imu_json: Path, raw_imu_csv: Path) -> "pd.DataFrame":
    """Parse the IMU sidecar into columns [epoch_ms, ax..az, gx..gz, mx..mz, temp_c].

    Factored out of run() so the parse contract is unit-testable without ffmpeg.
    Required channels (time/accel/gyro) must be finite; optional channels stay NaN
    when the source doesn't provide them.
    """
    expected_cols = ["epoch_ms"] + cfg.ACCEL_COLS + cfg.GYRO_COLS
    all_cols = expected_cols + IMU_OPTIONAL_COLS

    if raw_imu_json.exists():
        with open(raw_imu_json, "r", encoding="utf-8") as f:
            imu_data = json.load(f)
        if isinstance(imu_data, list) and len(imu_data) > 0 and isinstance(imu_data[0], dict):
            rows = []
            for item in imu_data:
                if "timestamp_ns" in item:
                    t_ms = item["timestamp_ns"] / 1e6
                elif "epoch_ms" in item:
                    t_ms = item["epoch_ms"]
                elif "timestamp_ms" in item:
                    t_ms = item["timestamp_ms"]
                elif "timestamp" in item:
                    t_ms = item["timestamp"]
                else:
                    t_ms = 0.0

                if "accel" in item and isinstance(item["accel"], list):
                    ax, ay, az = item["accel"][:3]
                else:
                    ax, ay, az = item.get("ax", 0.0), item.get("ay", 0.0), item.get("az", 0.0)

                if "gyro" in item and isinstance(item["gyro"], list):
                    gx, gy, gz = item["gyro"][:3]
                else:
                    gx, gy, gz = item.get("gx", 0.0), item.get("gy", 0.0), item.get("gz", 0.0)

                if "mag" in item and isinstance(item["mag"], list):
                    mx, my, mz = item["mag"][:3]
                else:
                    mx = item.get("mx", np.nan)
                    my = item.get("my", np.nan)
                    mz = item.get("mz", np.nan)

                temp_c = item.get("temp_c", item.get("temperature_c", np.nan))

                rows.append([t_ms, ax, ay, az, gx, gy, gz, mx, my, mz, temp_c])
            imu_df = pd.DataFrame(rows, columns=all_cols)
        else:
            imu_df = pd.DataFrame(imu_data)
    else:
        imu_df = pd.read_csv(raw_imu_csv)
        for col in expected_cols:
            if col not in imu_df.columns:
                raise ValueError(
                    f"[{STEP}] IMU CSV missing column '{col}'. "
                    f"Expected: {expected_cols}. Got: {list(imu_df.columns)}"
                )

    for col in IMU_OPTIONAL_COLS:
        if col not in imu_df.columns:
            imu_df[col] = np.nan

    # NaN in a REQUIRED channel invalidates the row; NaN in an optional channel
    # means "not measured" and must not discard the sample.
    imu_df = imu_df[all_cols].astype(np.float64).dropna(subset=expected_cols)
    return imu_df


def _maybe_delete_raw_video(raw_video: Path) -> bool:
    """
    Delete raw_video if config.KEEP_RAW_AFTER_COMPRESSION is False and
    config.PERCEPTION_SOURCE isn't "raw" (v2 addendum §8) — pulled out of
    run() as its own function so the gating logic is unit-testable without
    invoking ffmpeg. Returns whether it deleted the file.
    """
    if cfg.KEEP_RAW_AFTER_COMPRESSION or cfg.PERCEPTION_SOURCE == "raw":
        return False
    raw_video.unlink()
    return True


def run(session_path: Path) -> dict:
    """
    Ingest a raw session: compress video, extract PTS, parse IMU, write metadata.

    Args:
        session_path: Path to raw session folder (e.g., raw/session_001).

    Returns:
        Dict with session_meta contents.
    """
    t0 = time.time()
    session_id = session_path.name
    print(f"[{STEP}] Starting {session_id}...")

    raw_video = session_path / "raw.mp4"
    raw_imu_json = session_path / "imu.json"
    raw_imu_csv = session_path / "imu.csv"

    if not raw_video.exists():
        raise FileNotFoundError(f"[{STEP}] raw.mp4 not found at {raw_video}")
    if not raw_imu_json.exists() and not raw_imu_csv.exists():
        raise FileNotFoundError(f"[{STEP}] Neither imu.json nor imu.csv found at {session_path}")

    # Never silently pick one IMU sidecar among several (e.g. both imu.json and
    # imu.csv, or an imu_wrist.json this single-stream path cannot represent).
    # A session must not ingest cleanly while a sensor stream is discarded.
    imu_sidecars = sorted(p.name for p in session_path.glob("imu*") if p.is_file())
    if len(imu_sidecars) > 1:
        raise ValueError(
            f"[{STEP}] {len(imu_sidecars)} IMU sidecars found at {session_path}: "
            f"{imu_sidecars}. This single-IMU path refuses to pick one silently — "
            "remove the extras or use the multi-stream ingest (actuate.ingest)."
        )

    out_dir = cfg.PROCESSED_DIR / session_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # ─── Step 1: Compress video ────────────────────────────────
    compressed_path = out_dir / "compressed.mp4"
    print(f"[{STEP}] Compressing video (H.265 CRF{cfg.FFMPEG_CRF})...")

    size_info = compress_video(
        input_path=raw_video,
        output_path=compressed_path,
        crf=cfg.FFMPEG_CRF,
        preset=cfg.FFMPEG_PRESET,
        codec=cfg.FFMPEG_CODEC,
    )

    assert compressed_path.exists(), f"Compressed video not created: {compressed_path}"
    print(
        f"[{STEP}] Compressed: {size_info['raw_size_mb']:.1f}MB → "
        f"{size_info['compressed_size_mb']:.1f}MB "
        f"({size_info['reduction_pct']:.0f}% reduction)"
    )

    # ─── Step 1b: Delete raw.mp4 (v2 addendum §8) ─────────────
    # Gated on config.KEEP_RAW_AFTER_COMPRESSION — defaults to True (never
    # delete) since that's the safe default. Even with
    # KEEP_RAW_AFTER_COMPRESSION=False, never delete raw.mp4 if
    # PERCEPTION_SOURCE="raw" — a perception script would otherwise fail
    # with FileNotFoundError on its very next run, for a reason that isn't
    # obvious from that error alone.
    if _maybe_delete_raw_video(raw_video):
        print(f"[{STEP}] Deleted {raw_video} (KEEP_RAW_AFTER_COMPRESSION=False)")

    # ─── Step 2: Extract PTS ──────────────────────────────────
    print(f"[{STEP}] Extracting PTS timestamps...")
    pts = extract_pts(compressed_path)
    pts_path = out_dir / "pts.npy"
    np.save(str(pts_path), pts)

    # Count large gaps
    diffs = np.diff(pts)
    pts_gap_count = int(np.sum(diffs > 0.200))  # > 200ms

    print(f"[{STEP}] PTS extracted: {len(pts):,} frames")
    if pts_gap_count > 0:
        print(f"[{STEP}] WARNING: {pts_gap_count} large PTS gaps (>200ms)")

    # ─── Step 3: Parse IMU ────────────────────────────────────
    print(f"[{STEP}] Parsing IMU data...")
    imu_df = _parse_imu(raw_imu_json, raw_imu_csv)

    if len(imu_df) < 100:
        raise ValueError(
            f"[{STEP}] IMU data too short: {len(imu_df)} rows (minimum 100 required)"
        )

    imu_raw = imu_df.values  # [N, 11]: epoch_ms, ax..az, gx..gz, mx..mz, temp_c
    imu_raw_path = out_dir / "imu_raw.npy"
    np.save(str(imu_raw_path), imu_raw)

    # Compute measured IMU Hz
    imu_duration_s = (imu_raw[-1, 0] - imu_raw[0, 0]) / 1000.0
    imu_hz_measured = len(imu_raw) / imu_duration_s if imu_duration_s > 0 else 0.0

    print(f"[{STEP}] IMU loaded: {len(imu_raw):,} rows at {imu_hz_measured:.1f}Hz")

    # ─── Step 4: Extract recording start epoch ────────────────
    video_meta = get_video_metadata(compressed_path)
    creation_time_epoch_ms = None

    if video_meta.get("creation_time"):
        try:
            ct = video_meta["creation_time"]
            # Parse ISO 8601 creation_time
            if ct.endswith("Z"):
                ct = ct[:-1] + "+00:00"
            dt = datetime.fromisoformat(ct)
            creation_time_epoch_ms = dt.timestamp() * 1000.0
        except (ValueError, TypeError):
            pass

    video_duration_s = float(pts[-1]) if len(pts) > 0 else 0.0
    recording_start_epoch_ms, anchor_fields = _assess_temporal_anchor(
        creation_time_epoch_ms,
        imu_t0_ms=float(imu_raw[0, 0]),
        imu_t1_ms=float(imu_raw[-1, 0]),
        video_duration_s=video_duration_s,
    )
    start_source = (
        "video_creation_time"
        if anchor_fields["temporal_alignment_validated"]
        else "imu_first_timestamp"
    )

    print(f"[{STEP}] Recording start: {recording_start_epoch_ms:.0f}ms (source: {start_source})")
    if not anchor_fields["temporal_alignment_validated"]:
        print(f"[{STEP}] ⚠ TEMPORAL ALIGNMENT UNVALIDATED: {anchor_fields['temporal_alignment_note']}")

    # ─── Step 5: Write session_meta.json ──────────────────────
    duration_seconds = video_duration_s

    session_meta = {
        "session_id": session_id,
        "fps_nominal": video_meta.get("fps", cfg.TARGET_FPS),
        "frame_count": len(pts),
        "duration_seconds": round(duration_seconds, 3),
        "imu_hz_measured": round(imu_hz_measured, 1),
        "recording_start_epoch_ms": recording_start_epoch_ms,
        "recording_start_source": start_source,
        **anchor_fields,
        "timestamp_semantics": _read_timestamp_semantics(session_path),
        "pts_gap_count": pts_gap_count,
        "compressed_size_mb": size_info["compressed_size_mb"],
        "raw_size_mb": size_info["raw_size_mb"],
        "video_width": video_meta.get("width", 0),
        "video_height": video_meta.get("height", 0),
        "video_codec": video_meta.get("codec", "unknown"),
        "imu_source_mode": cfg.IMU_SOURCE_MODE,
    }
    _IMU_MOUNT_LOCATIONS = {"wrist_mounted": "wrist", "dual": "head+wrist"}
    if cfg.IMU_SOURCE_MODE in _IMU_MOUNT_LOCATIONS:
        session_meta["imu_mount_location"] = _IMU_MOUNT_LOCATIONS[cfg.IMU_SOURCE_MODE]

    session_meta["consent_status"] = _read_consent_status(session_path)

    glove_type, worker_id = _read_glove_and_worker_config(session_path)
    session_meta["glove_type"] = glove_type
    session_meta["worker_id"] = worker_id

    meta_path = out_dir / "session_meta.json"
    with open(meta_path, "w") as f:
        json.dump(session_meta, f, indent=2)

    elapsed = time.time() - t0
    print(f"[{STEP}] ✓ Done ({elapsed:.1f}s)")

    return session_meta


def main():
    parser = argparse.ArgumentParser(description="DatraAI Step 01: Ingest raw session")
    parser.add_argument(
        "--session",
        type=str,
        required=True,
        help="Path to raw session folder (e.g., raw/session_001)",
    )
    args = parser.parse_args()
    run(Path(args.session))


if __name__ == "__main__":
    main()
