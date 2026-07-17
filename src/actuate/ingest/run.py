"""L0 ingest.run -- MINIMAL, for the already-processed session layout (Phase 5 Part E).

Scope is deliberately small and stated: this handles a session directory that already has
`session_meta.json` + a video (what the real corpus is), reusing the existing
content-addressing. It does NOT build the full L0: no MCAP container, no PyAV decode, no
Polars sync, none of the six RigAdapters -- those stay unbuilt (see the package docstring).

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

from actuate.config import RigType, Tier, get_embodiment
from actuate.ingest.content_address import build_manifest, hash_file, write_manifest

#: Relative tolerance for fx/fy match. Beyond this, the cameras are not the same setup.
_INTRINSICS_RTOL = 0.05


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

    def summary(self) -> str:
        lines = [f"{self.session_dir.name}: capture {self.capture_id[:16]}… | "
                 f"rig {self.rig.value} | tier {self.tier.value}"]
        if self.aligned_robot:
            v = {True: "VERIFIED", False: "MISMATCH", None: "UNVERIFIABLE"}[
                self.intrinsics_match]
            lines.append(f"  aligned-robot claim [{self.aligned_robot}]: {v}")
        for f in self.flags:
            lines.append(f"  FLAG: {f}")
        return "\n".join(lines)


def _session_video(src: Path) -> Path:
    for name in ("compressed.mp4", "redacted_compressed.mp4", "raw.mp4"):
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
        "source": "auto_from_video",
    }
    meta_p.write_text(_json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def _session_intrinsics(src: Path) -> tuple[float, float] | None:
    """(fx, fy) from camera_intrinsics.json when present; None = not calibrated."""
    p = src / "camera_intrinsics.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    fx = d.get("fx") or (d.get("camera_matrix") or [[None]])[0][0]
    fy = d.get("fy") or (d.get("camera_matrix") or [[None], [None, None]])[1][1] \
        if d.get("camera_matrix") else d.get("fy")
    if fx is None or fy is None:
        return None
    return float(fx), float(fy)


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
    video = _session_video(src)

    flags: list[str] = []
    capture_id = hash_file(video)

    meta = {}
    meta_p = src / "session_meta.json"
    if meta_p.exists():
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
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

    tier, match = Tier.STAGE1_VOLUME, None
    if aligned_robot is not None:
        tier, match = _check_alignment(src, aligned_robot, flags)

    return IngestResult(
        session_dir=src, capture_id=capture_id, rig=rig, tier=tier,
        aligned_robot=aligned_robot, intrinsics_match=match, flags=flags,
        manifest_path=manifest_path,
    )
