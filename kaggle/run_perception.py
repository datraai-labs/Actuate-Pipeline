"""Kaggle runner: full Actuate perception on the real capture, cached + benchmarked.

Handles a **raw Kaggle dataset** (read-only, filenames with spaces/parens and no session
metadata) OR an already-processed Actuate session -- detects which, and for the raw case copies
+ normalises files into a writable working dir and generates `session_meta.json` from the video
with OpenCV (no manual metadata editing). Then runs SLAM + WiLoR + UniDepth + Grounding DINO/SAM2
+ L2 fusion with `--cache`, writes the `.rrd`, and (with `--depth-ab`) the depth benchmark.

Why Kaggle: the 4 GB dev card thrashes (SAM2 propagation once took 86 min). A T4 (16 GB, free)
runs the whole thing in minutes and is the only place MoGe-2 / Video-Depth-Anything fit.

--------------------------------------------------------------------------------------------
CONSENT: the capture is consent-pending human data. Do NOT commit it or make the Kaggle dataset
public. Keep the dataset PRIVATE and delete it when done. On Kaggle, YOU are the consent
boundary. (The video is not face/OCR-redacted here -- this is R&D output that never ships.)
--------------------------------------------------------------------------------------------

Usage (Kaggle notebook, GPU on):
    python kaggle/run_perception.py \
        --session /kaggle/input/datasets/elon7069/session-001 \
        --max-frames 60 --depth-ab --out /kaggle/working/session_001.rrd

Download from the notebook's Output tab (everything under /kaggle/working):
    session_001.rrd                              -- open with `rerun session_001.rrd`
    depth_ab.txt                                 -- the depth benchmark table + gate verdict
    processed/session_001/.actuate_cache/*.pkl   -- copy into your local session, then
                                                    `actuate viz show <session> --cache`
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

#: raw Kaggle filename (glob) -> the normalised name the pipeline expects. The trailing
#: "(1)"/"(2)" Kaggle appends varies, so we match by PREFIX, not the exact string.
_RAW_MAP = {
    "compressed.mp4": "video*.mp4",
    "motion.json": "motion*.json",
    "timestamps.json": "timestamps*.json",
    "camera_intrinsics.json": "camera_intrinsic*.json",
    "metadata.json": "metadata*.json",
}


def _looks_processed(d: Path) -> bool:
    """An already-processed Actuate session has session_meta.json + a video next to it."""
    return (d / "session_meta.json").exists() and (
        (d / "redacted_compressed.mp4").exists() or (d / "compressed.mp4").exists()
    )


def _write_session_meta(dst: Path, video: Path) -> dict:
    """Generate a valid session_meta.json from the video itself (OpenCV). No manual editing."""
    import cv2

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    if fc <= 0:  # some containers don't report a frame count -> count by decoding
        cap = cv2.VideoCapture(str(video))
        fc = 0
        while cap.read()[0]:
            fc += 1
        cap.release()
    meta = {
        "session_id": "session_001",
        "frame_count": int(fc),
        "fps_nominal": round(float(fps), 3),
        "fps": round(float(fps), 3),
        "duration_seconds": round(fc / fps, 3) if fps else None,
        "resolution": [w, h],
        "width": w,
        "height": h,
        "source": "kaggle_raw_normalized",
    }
    (dst / "session_meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def prepare_session(src: Path, work_root: Path = Path("/kaggle/working/processed")) -> Path:
    """Return a WRITABLE, normalised session dir. Detects raw-Kaggle vs already-processed.

    - already-processed + writable -> use in place.
    - already-processed + read-only (e.g. mounted) -> copy to the working root.
    - raw Kaggle dataset -> copy + normalise filenames into the working root and synthesise
      session_meta.json from the video.
    """
    import os
    import shutil

    src = src.resolve()

    if _looks_processed(src):
        if os.access(src, os.W_OK):
            print(f"session: already-processed, writable -> using in place: {src}")
            return src
        dst = work_root / (src.name or "session_001")
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst, dirs_exist_ok=True)
        print(f"session: already-processed, read-only -> copied to {dst}")
        return dst

    # raw Kaggle dataset: normalise into a writable working dir
    dst = work_root / "session_001"
    dst.mkdir(parents=True, exist_ok=True)
    print(f"session: raw Kaggle dataset {src} -> normalising into {dst}")
    for target, pattern in _RAW_MAP.items():
        hits = sorted(src.glob(pattern))
        if hits:
            shutil.copy2(hits[0], dst / target)
            print(f"  {hits[0].name!r} -> {target}")
    video = dst / "compressed.mp4"
    if not video.exists():
        raise SystemExit(
            f"no video found in {src} (expected a file matching 'video*.mp4'). "
            f"Files present: {[p.name for p in src.iterdir()]}"
        )
    meta = _write_session_meta(dst, video)
    print(f"  generated session_meta.json: {meta['frame_count']} frames @ "
          f"{meta['fps_nominal']} fps, {meta['width']}x{meta['height']}")
    return dst


def _run_stage(session, stage, key, use_cache, force, run_fn):
    """Reuse the CLI's exact cache format so downloaded caches work with `actuate viz --cache`."""
    from actuate.cli.viz import _stage_cached

    t0 = time.time()
    result, src = _stage_cached(session, stage, key, use_cache=use_cache, force=force,
                                run_fn=run_fn)
    print(f"  {stage:8s} {src:6s}  {time.time() - t0:6.1f}s", flush=True)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True, type=Path,
                    help="raw Kaggle dataset dir OR a processed Actuate session dir")
    ap.add_argument("--max-frames", type=int, default=60)
    ap.add_argument("--prompts", default="stapler,paper,document,box")
    ap.add_argument("--task", default="handle paperwork on a desk")
    ap.add_argument("--no-cache", action="store_true", help="ignore/refresh the cache")
    ap.add_argument("--depth-ab", action="store_true",
                    help="run the depth benchmark (UniDepth/MoGe-2/VDA-anchored/flow-filter)")
    ap.add_argument("--out", type=Path, default=Path("/kaggle/working/session_001.rrd"),
                    help="write the Rerun .rrd here")
    args = ap.parse_args()

    session = prepare_session(args.session)
    nf = args.max_frames
    use_cache = not args.no_cache
    prompts = [p.strip() for p in args.prompts.split(",")]
    out_rrd = args.out
    out_rrd.parent.mkdir(parents=True, exist_ok=True)

    from actuate import fusion as fusionmod
    from actuate.canonical import build_from_perception
    from actuate.config import RigType
    from actuate.perception import depth as depthmod
    from actuate.perception import hands as handsmod
    from actuate.perception import objects as objmod
    from actuate.perception.slam import runner as slamrun
    from actuate.viz import log_episode

    print(f"\nmax_frames={nf}  cache={'on' if use_cache else 'off'}\nperception:")

    def _slam():
        try:
            return slamrun.run(session, max_frames=nf)
        except Exception as exc:
            print(f"    slam skipped (needs IMU/session.h5; raw motion.json not converted): {exc}")
            return None

    slam = _run_stage(session, "slam", f"n={nf}", use_cache, False, _slam)
    depth = _run_stage(session, "depth", f"n={nf}", use_cache, False,
                       lambda: depthmod.run(session, max_frames=nf))
    hands = _run_stage(session, "hands", f"n={nf}", use_cache, False,
                       lambda: handsmod.run(session, max_frames=nf, prefilter=False))
    objects = _run_stage(session, "objects", f"n={nf}|{','.join(prompts)}", use_cache, False,
                         lambda: objmod.run(session, prompts=prompts, max_frames=nf, chunk=nf))
    fusion = fusionmod.run(hands, rig=RigType.HEAD_MOUNTED, objects=objects)
    print("  fusion   ran")

    ep = build_from_perception(session, "kaggle" * 10 + "0000", hands=hands, depth=depth,
                               fusion=fusion, slam=slam, objects=objects,
                               rig=RigType.HEAD_MOUNTED, task=args.task, max_frames=nf)
    print(f"canonical: schema v{ep.schema_version}  frames={len(ep.frames)}")

    # read frames for the RGB / point-cloud overlay
    import cv2
    import rerun as rr

    video = session / "redacted_compressed.mp4"
    if not video.exists():
        video = session / "compressed.mp4"
    frames = []
    cap = cv2.VideoCapture(str(video))
    for _ in range(nf):
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()

    rr.init("actuate")
    log_episode(hands=hands, depth=depth, objects=objects, fusion=fusion, slam=slam,
                video_frames=frames, intrinsics=depth.intrinsics if depth else None,
                max_frames=nf)
    rr.save(str(out_rrd))
    print(f"wrote {out_rrd}")

    if args.depth_ab:
        ab_path = out_rrd.parent / "depth_ab.txt"
        _depth_ab(session, depth, hands, frames, nf, ab_path)
        print(f"wrote {ab_path}")

    print("\nDONE. Download from the Output tab:")
    print(f"  {out_rrd}                         (open: rerun {out_rrd.name})")
    if args.depth_ab:
        print(f"  {out_rrd.parent / 'depth_ab.txt'}   (the benchmark verdict)")
    print(f"  {session / '.actuate_cache'}/*.pkl   (copy into your local session -> "
          "actuate viz show <session> --cache)")


def _depth_ab(session, unidepth_result, hands, frames, nf, out_path):
    """Full depth benchmark: WRIST z-jitter (the gate) + static consistency, all models.

    Gate (fair baseline, Part C): < 4 mm/frame smoothed -- 3x below the 11.3 mm hand-cloud-fit
    baseline, NOT the 20.7 mm bbox strawman. Models: UniDepth (baseline), MoGe-2 (metric/focal
    anchor), Video-Depth-Anything anchored to the metric anchor (temporal + metric scale),
    flow-filter.
    """
    from actuate.perception import depth as depthmod
    from actuate.perception.depth import run_benchmark
    from actuate.perception.depth.temporal import anchor_scale

    models = {"UniDepthV2": unidepth_result}

    try:
        moge = depthmod.run(session, model="moge2", max_frames=nf)
        models["MoGe-2"] = moge
        print(f"  MoGe-2 focal fx={moge.intrinsics[0, 0]:.0f} (UniDepth guessed "
              f"{unidepth_result.intrinsics[0, 0]:.0f})")
    except Exception as exc:
        print(f"  MoGe-2 skipped: {exc}")
        moge = None

    try:
        models["UniDepth+flow_filter"] = depthmod.run(
            session, model="flow_filter", base=unidepth_result, max_frames=nf)
    except Exception as exc:
        print(f"  flow_filter skipped: {exc}")

    try:
        vda = depthmod.run(session, model="video_depth_anything",
                           intrinsics=unidepth_result.intrinsics, max_frames=nf)
        anchor = moge if moge is not None else unidepth_result
        models["VDA (anchored)"] = anchor_scale(vda, anchor, keyframe_stride=8)
    except Exception as exc:
        print(f"  Video-Depth-Anything skipped: {exc}")

    report = run_benchmark(models, hands, frames, baseline="UniDepthV2", max_frames=nf)
    print("\n" + report)
    out_path.write_text(report)


if __name__ == "__main__":
    main()
