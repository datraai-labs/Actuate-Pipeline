"""Portable run preview for devices that cannot open an interactive Rerun viewer."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import unquote, urlparse

import numpy as np


def _write_browser_video(
    path: Path,
    frames: list[np.ndarray],
    *,
    hold_frames: int,
    fps: int,
) -> None:
    """Write H.264/yuv420p MP4, which browsers and phones can decode directly."""
    try:
        import av
    except ImportError as exc:  # pragma: no cover - depends on the selected install extra
        raise RuntimeError(
            "portable preview encoding needs PyAV; install `actuate[viz]`"
        ) from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    container = av.open(str(path), mode="w", options={"movflags": "faststart"})
    try:
        stream = container.add_stream("libx264", rate=fps)
        stream.width = int(frames[0].shape[1])
        stream.height = int(frames[0].shape[0])
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "23", "preset": "medium"}
        for frame in frames:
            video_frame = av.VideoFrame.from_ndarray(frame, format="bgr24")
            for _ in range(max(1, int(hold_frames))):
                for packet in stream.encode(video_frame):
                    container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()


def _local_uri(uri: str, base: Path) -> Path:
    if uri.startswith("file://"):
        return Path(unquote(urlparse(uri).path))
    p = Path(uri)
    return p if p.is_absolute() else base / p


def _decode_source_frames(video: Path, frame_ids: list[int]) -> dict[int, np.ndarray]:
    import cv2

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open source video {video}")
    frames = {}
    for i in frame_ids:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if ok:
            frames[i] = frame
    cap.release()
    return frames


def _depth_panel(depth_m: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    import cv2

    valid = depth_m[np.isfinite(depth_m) & (depth_m > 0)]
    if not valid.size:
        return np.zeros((size[1], size[0], 3), dtype=np.uint8)
    lo, hi = np.percentile(valid, [2, 98])
    norm = np.clip((depth_m - lo) / max(float(hi - lo), 1e-6), 0.0, 1.0)
    # Near = warm, far = cool.
    colored = cv2.applyColorMap(
        np.asarray((1.0 - norm) * 255.0, dtype=np.uint8), cv2.COLORMAP_TURBO
    )
    return cv2.resize(colored, size, interpolation=cv2.INTER_AREA)


def _annotate_rgb(
    frame: np.ndarray, frame_record: dict, object_records: list[dict]
) -> np.ndarray:
    import cv2

    from actuate.perception.objects.rle import decode_rle

    out = frame.copy()
    palette = [(41, 190, 255), (225, 90, 85), (90, 220, 130), (210, 120, 240)]
    by_track = {int(o["track_id"]): o for o in object_records}
    for track_id, obj in frame_record.get("objects", {}).items():
        track = int(track_id)
        color = palette[(track - 1) % len(palette)]
        mask_ref = obj.get("mask")
        if mask_ref and mask_ref.get("rle"):
            mask = decode_rle(mask_ref["rle"])
            if mask.shape == out.shape[:2]:
                overlay = np.zeros_like(out)
                overlay[mask] = color
                out = cv2.addWeighted(out, 1.0, overlay, 0.32, 0.0)
        info = by_track.get(track)
        if info:
            x0, y0, x1, y1 = [int(round(v)) for v in info["bbox_xyxy_px"]]
            cv2.rectangle(out, (x0, y0), (x1, y1), color, 4)
            label = f"{info['label']} #{track}  {info['score']:.2f}"
            cv2.putText(
                out, label, (x0, max(28, y0 - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, color, 2, cv2.LINE_AA,
            )
    return out


def _compose_frame(
    rgb: np.ndarray,
    depth_m: np.ndarray,
    frame_record: dict,
    object_records: list[dict],
    robot_q: list[float] | None,
    *,
    quality: int | None,
) -> np.ndarray:
    import cv2

    panel_size = (640, 360)
    rgb = _annotate_rgb(rgb, frame_record, object_records)
    rgb = cv2.resize(rgb, panel_size, interpolation=cv2.INTER_AREA)
    depth = _depth_panel(depth_m, panel_size)
    canvas = np.full((500, 1280, 3), 18, dtype=np.uint8)
    canvas[:360, :640] = rgb
    canvas[:360, 640:] = depth
    cv2.putText(canvas, "RGB + tracked masks", (20, 34), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, "Metric depth (UniDepth)", (660, 34), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2, cv2.LINE_AA)

    t = float(frame_record["t"])
    source_i = int(frame_record["frame_idx"])
    state = frame_record.get("interaction_state") or "not classified"
    cv2.putText(
        canvas,
        f"source frame {source_i}   t={t:6.2f}s   state={state}   quality={quality or '?'} / 5",
        (24, 402), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (235, 235, 235), 2, cv2.LINE_AA,
    )
    if robot_q is not None:
        cv2.putText(canvas, "Franka joint targets (rad)", (24, 442),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (180, 210, 255), 2, cv2.LINE_AA)
        for j, q in enumerate(robot_q):
            x = 290 + j * 135
            cv2.putText(canvas, f"q{j + 1} {q:+.2f}", (x, 442),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.56, (180, 210, 255), 1, cv2.LINE_AA)
            half = 50
            cv2.line(canvas, (x, 474), (x + half * 2, 474), (65, 65, 65), 3)
            end = int(x + half + np.clip(q / np.pi, -1.0, 1.0) * half)
            cv2.line(canvas, (x + half, 474), (end, 474), (80, 205, 255), 7)
    return canvas


def render_run_preview(
    run_dir: Path,
    *,
    video_out: Path | None = None,
    contact_sheet_out: Path | None = None,
    hold_frames: int = 8,
    fps: int = 10,
) -> tuple[Path, Path]:
    """Render an MP4 and contact sheet from a completed Actuate run."""
    import cv2

    run_dir = Path(run_dir)
    canonical = json.loads((run_dir / "canonical.json").read_text(encoding="utf-8"))
    records = canonical["frames"]
    if not records:
        raise ValueError("canonical episode has no frames")
    image_ref = next(iter(records[0]["images"].values()))
    source_video = _local_uri(image_ref["uri"], run_dir)
    frame_ids = [int(f["frame_idx"]) for f in records]
    source_frames = _decode_source_frames(source_video, frame_ids)

    depth_path = run_dir / records[0]["depth"]["head"]["uri"]
    with np.load(depth_path, allow_pickle=False) as depth_file:
        depth_maps = np.asarray(depth_file["depth_m"])

    objects_path = run_dir / "artifacts" / "perception" / "objects.json"
    objects = (
        json.loads(objects_path.read_text(encoding="utf-8"))
        if objects_path.exists() else {"frames": {}}
    )
    robot = canonical.get("action_robot", {}).get("franka_panda", {})
    robot_q = robot.get("joint_traj") or []
    quality = canonical.get("episode_meta", {}).get("quality")

    rendered = []
    for k, record in enumerate(records):
        source_i = int(record["frame_idx"])
        if source_i not in source_frames:
            continue
        depth_ref = record["depth"]["head"]
        depth = depth_maps[int(depth_ref["frame_index"])]
        rendered.append(_compose_frame(
            source_frames[source_i],
            depth,
            record,
            objects.get("frames", {}).get(str(source_i), []),
            robot_q[k] if k < len(robot_q) else None,
            quality=quality,
        ))
    if not rendered:
        raise ValueError("none of the canonical source frames could be decoded")

    video_out = Path(video_out or (run_dir / "preview.mp4"))
    contact_sheet_out = Path(contact_sheet_out or (run_dir / "contact-sheet.png"))
    video_out.parent.mkdir(parents=True, exist_ok=True)
    _write_browser_video(
        video_out,
        rendered,
        hold_frames=hold_frames,
        fps=fps,
    )

    # Two columns keeps object labels, timestamps, and seven joint values readable in a
    # normal image viewer; a 5-wide strip technically contained everything but was illegible.
    thumb_w, thumb_h = 640, 250
    cols = min(2, len(rendered))
    rows = int(np.ceil(len(rendered) / cols))
    sheet = np.full((rows * thumb_h, cols * thumb_w, 3), 18, dtype=np.uint8)
    for k, frame in enumerate(rendered):
        y, x = divmod(k, cols)
        sheet[y * thumb_h:(y + 1) * thumb_h, x * thumb_w:(x + 1) * thumb_w] = cv2.resize(
            frame, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA
        )
    if not cv2.imwrite(str(contact_sheet_out), sheet):
        raise RuntimeError(f"cannot create contact sheet {contact_sheet_out}")

    manifest_path = run_dir / "run_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        artifacts = manifest.setdefault("artifacts", {})
        artifacts["portable_preview"] = video_out.resolve().as_uri()
        artifacts["contact_sheet"] = contact_sheet_out.resolve().as_uri()
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return video_out, contact_sheet_out
