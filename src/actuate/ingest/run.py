"""L0 ingest.run -- MINIMAL, for the already-processed session layout (Phase 5 Part E).

Scope is deliberately small and stated: this handles a session directory that already has
`session_meta.json` + a video (what the real corpus is), reusing the existing
content-addressing. A raw JSON/CSV IMU sidecar is synchronized onto the video frame axis for
SLAM. It does NOT build the full L0: no MCAP container, no general multi-clock sync, none of
the six RigAdapters -- those stay unbuilt (see the package docstring).

### Aligned-capture mode (the Stage-II anchor gate, EgoScale/EgoVerse)

`aligned_robot=` claims this capture shares the target robot's camera configuration
(intrinsics, resolution). The claim is VERIFIED, never trusted:

- session and embodiment intrinsics both present and matching  -> tier = stage2_anchor
- both present, mismatched                                     -> stage1 + FLAG
- either side missing calibration                              -> stage1 + FLAG (an
  unverifiable claim is not a verified one -- exactly the auto-consent bug class, applied
  to alignment)

Stage-II is a CLAIM that makes the data command a premium; a pipeline that lets the flag
default on would be selling volume data at anchor prices. Today every registered embodiment
has `camera_intrinsics=None` (nobody has calibrated a robot camera), so on the real corpus
every aligned claim flags as unverifiable -- which is the true state of the world.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from actuate.config import RigType, Tier, all_sensor_patterns, get_embodiment, get_rig
from actuate.ingest.content_address import build_manifest, hash_file, write_manifest

#: Relative tolerance for fx/fy match. Beyond this, the cameras are not the same setup.
_INTRINSICS_RTOL = 0.05

#: Extensions accepted for a declared camera's video file on multi-camera rigs.
_CAMERA_EXTENSIONS = (".mp4", ".mov", ".avi")


class RigStreamError(ValueError):
    """The session's on-disk streams do not satisfy the declared rig's manifest.

    Raised for: a required stream with no matching file, a stream whose patterns
    match more than one file (ambiguous — we never pick one silently), a declared
    camera with no video, or a sensor-shaped file no declared stream claims. All
    are the same bug class: a session ingesting cleanly while a sensor is missing
    or discarded.
    """


class RigDeclarationError(ValueError):
    """The declared capture rig contradicts the source pixels/layout."""


def validate_declared_rig(src: Path, rig: RigType) -> str:
    """Validate active-rig declarations against source geometry.

    This check is deliberately fatal: processing a stereo composite as one monocular image
    produces plausible-looking but physically meaningless depth and duplicate hands.
    """
    from actuate.ingest.layout import detect_layout

    layout = detect_layout(src)
    if rig is RigType.HEAD_MOUNTED and layout in {"stereo_video", "multi_camera"}:
        raise RigDeclarationError(
            f"declared rig=head_mounted but the source layout is {layout}. A side-by-side "
            "stereo composite must not be processed as one monocular image. Declare "
            "rig=stereo and use a validated stereo adapter."
        )
    if rig is RigType.STEREO and layout not in {"stereo_video", "multi_camera"}:
        raise RigDeclarationError(
            f"declared rig=stereo but the source layout is {layout}; expected one "
            "side-by-side stereo video or both declared camera files. Refusing to invent "
            "the missing second view."
        )
    return layout


@dataclass
class IngestResult:
    session_dir: Path
    capture_id: str                       # SHA-256 of the video bytes -- THE identity
    rig: RigType
    tier: Tier
    aligned_robot: str | None
    intrinsics_match: bool | None         # None = could not verify (missing calibration)
    flags: list[str] = field(default_factory=list)
    manifest_path: Path | None = None
    imu_sync_path: Path | None = None
    imu_samples: int = 0
    imu_frames: int = 0

    def summary(self) -> str:
        lines = [(f"{self.session_dir.name}: capture {self.capture_id[:16]}… | "
                 f"rig {self.rig.value} | tier {self.tier.value}")]
        if self.aligned_robot:
            v = {True: "VERIFIED", False: "MISMATCH", None: "UNVERIFIABLE"}[
                self.intrinsics_match]
            lines.append(f"  aligned-robot claim [{self.aligned_robot}]: {v}")
        for f in self.flags:
            lines.append(f"  FLAG: {f}")
        if self.imu_sync_path is not None:
            lines.append(
                f"  IMU: {self.imu_samples} samples synchronized to "
                f"{self.imu_frames} video frames"
            )
        return "\n".join(lines)


def _session_video(src: Path) -> Path:
    # Once a redaction pass exists it is the safe/default view of this capture. Picking
    # compressed.mp4 first silently sent the unredacted source to visualization/export even
    # after --redact-pii had completed successfully.
    for name in ("redacted_compressed.mp4", "compressed.mp4", "raw.mp4"):
        p = src / name
        if p.exists():
            return p
    # any other single .mp4 the user dropped in -- accept it so "add my video and run"
    # doesn't require the exact filename
    others = sorted(src.glob("*.mp4"))
    if others:
        return others[0]
    raise FileNotFoundError(
        f"{src}: no video (*.mp4). ingest.run handles the processed session layout only -- "
        "see module docstring.")


def verify_rig_streams(rig: RigType, src: Path) -> None:
    """Assert the session's on-disk streams satisfy the declared rig's manifest.

    Master Spec §L0 gives ingest the rig; the registry (config/rigs.py) declares what
    that rig produces. Before this check, Layer 1 never read the registry — which is
    how a two-IMU session could ingest cleanly with one sensor silently discarded.
    Raises :class:`RigStreamError` listing EVERY problem found, or returns None.
    """
    spec = get_rig(rig)
    problems: list[str] = []

    # Cameras. A single-camera rig accepts the processed-session video names
    # (redacted_compressed/compressed/raw.mp4, or any sole .mp4). A multi-camera
    # rig requires one video per declared camera name — nothing else expresses
    # "which camera is which".
    if len(spec.cameras) <= 1:
        try:
            _session_video(src)
        except FileNotFoundError as exc:
            problems.append(str(exc))
    else:
        for cam in spec.cameras:
            names = [f"{cam}{ext}" for ext in _CAMERA_EXTENSIONS]
            names += [f"video_{cam}{ext}" for ext in _CAMERA_EXTENSIONS]
            if not any((src / n).exists() for n in names):
                problems.append(
                    f"declared camera '{cam}' has no video file "
                    f"(expected one of: {', '.join(names[:2])}, ...)"
                )

    claimed: set[Path] = set()
    for stream in spec.sensor_streams:
        matches = sorted(
            {p for pattern in stream.patterns for p in src.glob(pattern) if p.is_file()}
        )
        claimed.update(matches)
        if stream.required and not matches:
            problems.append(
                f"required stream '{stream.name}' is MISSING: no file matches "
                f"{list(stream.patterns)} in {src}"
            )
        if len(matches) > 1:
            problems.append(
                f"stream '{stream.name}' is AMBIGUOUS: {len(matches)} files match "
                f"{list(stream.patterns)}: {[m.name for m in matches]} — refusing "
                "to pick one silently"
            )

    # Anything sensor-shaped that no declared stream claims would be silently
    # discarded by every downstream stage. The recognizer is the union of every
    # registered rig's patterns, so registering a new rig protects its sidecars
    # on all rigs automatically.
    for pattern in sorted(all_sensor_patterns()):
        for p in sorted(src.glob(pattern)):
            if p.is_file() and p not in claimed:
                problems.append(
                    f"sensor sidecar '{p.name}' is present but not declared by rig "
                    f"'{spec.rig.value}' — a session must never ingest while a "
                    "sensor is discarded"
                )
                claimed.add(p)  # report each undeclared file once

    if problems:
        detail = "\n  - ".join(problems)
        raise RigStreamError(
            f"{src}: session does not satisfy rig '{spec.rig.value}' manifest:\n"
            f"  - {detail}"
        )


def ensure_session_meta(src: Path) -> dict:
    """Generate session_meta.json from the video itself if it is missing (OpenCV).

    The perception stages need frame_count/fps/resolution. A user who just drops a raw
    video into a folder has none of that; deriving it from the video removes the manual
    step so `actuate viz show <folder>` works on any clip. If a valid meta already exists
    it is returned untouched.
    """
    import json as _json

    src = Path(src)
    meta_p = src / "session_meta.json"
    if meta_p.exists():
        try:
            m = _json.loads(meta_p.read_text(encoding="utf-8"))
            if m.get("frame_count"):
                return m
        except (ValueError, OSError):
            pass  # malformed -> regenerate

    import cv2

    video = _session_video(src)
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    if fc <= 0:                       # some containers don't report a count -> decode-count
        cap = cv2.VideoCapture(str(video))
        while cap.read()[0]:
            fc += 1
        cap.release()
    meta = {
        "session_id": src.name,
        "frame_count": int(fc),
        "fps_nominal": round(float(fps), 3),
        "fps": round(float(fps), 3),
        "duration_seconds": round(fc / fps, 3) if fps else None,
        "resolution": [w, h],
        "width": w,
        "height": h,
        # Retain the explicit names used by the VIO/SLAM contract. The aliases above
        # remain for existing consumers.
        "video_width": w,
        "video_height": h,
        "source": "auto_from_video",
    }
    meta_p.write_text(_json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def _probe_video_fps(video: Path) -> float | None:
    """Container-reported fps, or None when it cannot be probed (then no claim is
    made either way — an unprobeable video must not manufacture a mismatch)."""
    try:
        import cv2

        cap = cv2.VideoCapture(str(video))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return float(fps) if fps and fps > 0 else None
    except Exception:
        return None


def _session_intrinsics(src: Path) -> tuple[float, float] | None:
    """(fx, fy) from camera_intrinsics.json when present; None = not calibrated."""
    K = load_camera_matrix(src)
    if K is None:
        return None
    return float(K[0, 0]), float(K[1, 1])


def load_camera_matrix(src: Path):
    """Load a valid 3x3 pinhole calibration supplied with the capture.

    Accepts either a full ``camera_matrix`` or the common ``fx/fy/cx/cy`` form. Returning
    ``None`` is deliberate: callers can then keep an estimated-intrinsics provenance instead
    of silently upgrading malformed metadata into a measured calibration claim.
    """
    import numpy as np

    src = Path(src)
    p = src / "camera_intrinsics.json"
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        if d.get("camera_matrix") is not None:
            K = np.asarray(d["camera_matrix"], dtype=np.float64)
        else:
            fx, fy = float(d["fx"]), float(d["fy"])
            meta_p = src / "session_meta.json"
            meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.exists() else {}
            width = float(meta.get("video_width", meta.get("width", 0)))
            height = float(meta.get("video_height", meta.get("height", 0)))
            cx = float(d.get("cx", width / 2.0))
            cy = float(d.get("cy", height / 2.0))
            K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
        return None
    if K.shape != (3, 3) or not np.all(np.isfinite(K)):
        return None
    if K[0, 0] <= 0 or K[1, 1] <= 0 or abs(K[2, 2] - 1.0) > 1e-6:
        return None
    return K


def _check_alignment(src: Path, aligned_robot: str,
                     flags: list[str]) -> tuple[Tier, bool | None]:
    spec = get_embodiment(aligned_robot)          # raises on unknown -- a typo'd robot
    robot_k = getattr(spec, "camera_intrinsics", None)
    session_k = _session_intrinsics(src)

    if robot_k is None or session_k is None:
        missing = ("embodiment has no calibrated camera_intrinsics registered"
                   if robot_k is None else "session has no camera_intrinsics.json")
        flags.append(
            f"aligned-robot claim UNVERIFIABLE: {missing}. An unverifiable claim is not "
            "a verified one -- tier stays stage1_volume. Register the calibration and "
            "re-ingest to earn stage2_anchor.")
        return Tier.STAGE1_VOLUME, None

    fx_r, fy_r = float(robot_k[0]), float(robot_k[1])
    fx_s, fy_s = session_k
    ok = (abs(fx_s - fx_r) <= _INTRINSICS_RTOL * fx_r
          and abs(fy_s - fy_r) <= _INTRINSICS_RTOL * fy_r)
    if not ok:
        flags.append(
            f"aligned-robot claim MISMATCH: session fx/fy=({fx_s:.0f},{fy_s:.0f}) vs "
            f"{aligned_robot} ({fx_r:.0f},{fy_r:.0f}) beyond {_INTRINSICS_RTOL:.0%}. "
            "NOT silently accepted as aligned -- tier stays stage1_volume.")
        return Tier.STAGE1_VOLUME, False
    return Tier.STAGE2_ANCHOR, True


def run(rig: RigType | str, src: Path, store: Path | None = None,
        aligned_robot: str | None = None) -> IngestResult:
    """Ingest one processed session: content-address it, verify any alignment claim.

    `store`: where the capture manifest is written (defaults to the session dir).
    """
    src = Path(src)
    rig = RigType(rig) if isinstance(rig, str) else rig
    verify_rig_streams(rig, src)
    video = _session_video(src)
    layout = validate_declared_rig(src, rig)

    flags: list[str] = []
    capture_id = hash_file(video)

    meta = {}
    meta_p = src / "session_meta.json"
    if meta_p.exists():
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
        # Master Spec §L0 fps-bug gate: a pre-existing meta's fps claim is checked
        # against the container, not silently trusted (the legacy QA stage enforces
        # the same gate from PTS; this covers sessions that skip legacy QA).
        claimed_fps = meta.get("fps_nominal") or meta.get("fps")
        if claimed_fps:
            probed_fps = _probe_video_fps(video)
            if probed_fps and abs(probed_fps - float(claimed_fps)) > 0.05 * probed_fps:
                flags.append(
                    f"fps mismatch: session_meta claims {claimed_fps} but the video "
                    f"container reports {probed_fps:.2f} — metadata NOT silently "
                    "trusted (Master Spec §L0 gate)"
                )
        # Persist the validated per-session fact. Canonical assembly must never infer a rig
        # from a process-global IMU setting.
        if meta.get("rig") != rig.value or meta.get("source_layout") != layout:
            meta = {**meta, "rig": rig.value, "source_layout": layout}
            meta_p.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    else:
        flags.append("no session_meta.json: frame_count/fps unverified")

    manifest = build_manifest(
        video, rig,
        frame_count=int(meta["frame_count"]) if meta.get("frame_count") else None,
        duration_sec=float(meta["duration_seconds"]) if meta.get("duration_seconds") else None,
        fps=float(meta["fps_nominal"]) if meta.get("fps_nominal") else None,
    )
    out_dir = Path(store) if store else src
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "capture_manifest.json"
    write_manifest(manifest_path, manifest)

    imu_result = None
    try:
        from actuate.ingest.imu import MultipleIMUSourcesError, sync_imu

        imu_result = sync_imu(src, meta) if meta else None
    except MultipleIMUSourcesError:
        # Ambiguity is never demoted to a flag: proceeding would ingest one sensor
        # and silently discard another. (verify_rig_streams should have caught this
        # already; this is defense in depth for direct callers.)
        raise
    except Exception as exc:
        # A malformed optional sensor sidecar must be visible, but it must not make a valid
        # video unprocessable. SLAM will honestly fall back to vision-only rotation.
        flags.append(f"IMU sync failed: {type(exc).__name__}: {exc}")

    tier, match = Tier.STAGE1_VOLUME, None
    if aligned_robot is not None:
        tier, match = _check_alignment(src, aligned_robot, flags)

    return IngestResult(
        session_dir=src, capture_id=capture_id, rig=rig, tier=tier,
        aligned_robot=aligned_robot, intrinsics_match=match, flags=flags,
        manifest_path=manifest_path,
        imu_sync_path=imu_result.session_h5 if imu_result else None,
        imu_samples=imu_result.raw_samples if imu_result else 0,
        imu_frames=imu_result.frame_count if imu_result else 0,
    )
