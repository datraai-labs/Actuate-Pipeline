"""Kaggle runner: full Actuate perception on the real capture, cached for download.

Runs SLAM + WiLoR + UniDepth + Grounding DINO/SAM2 + L2 fusion on a session, writing
`<session>/.actuate_cache/*.pkl` in the exact format `actuate viz --cache` reads -- so you
download the cache and view locally with NO GPU. Also runs the depth A/B
(UniDepth vs Video-Depth-Anything vs flow-filter) and prints the temporal-consistency verdict.

Why Kaggle: the 4 GB dev card thrashes (SAM2 propagation once took 86 min). A T4 (16 GB, free)
runs the whole thing in minutes and is the only place FoundationPose / Video-Depth-Anything fit.

--------------------------------------------------------------------------------------------
CONSENT: the capture is consent-pending human data. Do NOT commit it or make the Kaggle dataset
public. Upload it as a PRIVATE Kaggle dataset and point --session at it. See kaggle/README.md.
--------------------------------------------------------------------------------------------

Usage (inside a Kaggle notebook, GPU on):
    !python run_perception.py --session /kaggle/input/<your-session>/session_001 --max-frames 60

Outputs (in /kaggle/working, downloadable from the notebook's Output tab):
    actuate_outputs.zip   -- the .actuate_cache/ pickles + episode.rrd + depth_ab.txt
"""

from __future__ import annotations

import argparse
import time
import zipfile
from pathlib import Path


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
    ap.add_argument("--session", required=True, type=Path, help="processed/<session> dir")
    ap.add_argument("--max-frames", type=int, default=60)
    ap.add_argument("--prompts", default="stapler,paper,document,box")
    ap.add_argument("--task", default="handle paperwork on a desk")
    ap.add_argument("--no-cache", action="store_true", help="ignore/refresh the cache")
    ap.add_argument("--depth-ab", action="store_true",
                    help="also run Video-Depth-Anything + flow-filter and score vs UniDepth")
    ap.add_argument("--out", type=Path, default=Path("/kaggle/working"))
    args = ap.parse_args()

    session = args.session.resolve()
    nf = args.max_frames
    use_cache = not args.no_cache
    prompts = [p.strip() for p in args.prompts.split(",")]

    from actuate import fusion as fusionmod
    from actuate.canonical import build_from_perception
    from actuate.config import RigType
    from actuate.perception import depth as depthmod
    from actuate.perception import hands as handsmod
    from actuate.perception import objects as objmod
    from actuate.perception.slam import runner as slamrun
    from actuate.viz import log_episode

    print(f"session={session}  max_frames={nf}  cache={'on' if use_cache else 'off'}")
    print("perception:")

    def _slam():
        try:
            return slamrun.run(session, max_frames=nf)
        except Exception as exc:
            print(f"    slam skipped: {exc}")
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

    # canonical episode (schema v3, MANO) -- proves the wiring end to end on the box that can
    # actually run it. Not deliverable (consent pending); this is R&D output.
    ep = build_from_perception(session, "kaggle" * 10 + "0000", hands=hands, depth=depth,
                               fusion=fusion, slam=slam, objects=objects,
                               rig=RigType.HEAD_MOUNTED, task=args.task, max_frames=nf)
    print(f"canonical: schema v{ep.schema_version}  frames={len(ep.frames)}")

    # write the .rrd too, so you can open it directly
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
    rrd = args.out / "episode.rrd"
    rr.save(str(rrd))
    print(f"wrote {rrd}")

    # optional depth A/B
    ab_path = args.out / "depth_ab.txt"
    if args.depth_ab:
        _depth_ab(session, depth, frames, nf, ab_path)

    # bundle for download
    zpath = args.out / "actuate_outputs.zip"
    cache_dir = session / ".actuate_cache"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for p in cache_dir.glob("*.pkl"):
            z.write(p, f".actuate_cache/{p.name}")
        if rrd.exists():
            z.write(rrd, rrd.name)
        if ab_path.exists():
            z.write(ab_path, ab_path.name)
    print(f"\nDOWNLOAD: {zpath}  (Output tab). Unzip .actuate_cache/ into your local "
          f"session dir, then: actuate viz show <session> --cache")


def _depth_ab(session, unidepth_result, frames, nf, out_path):
    """UniDepth vs Video-Depth-Anything vs flow-filter, scored by static-point wobble."""
    from actuate.perception import depth as depthmod
    from actuate.perception.depth import consistency_score, sample_tracked_depths

    lines = ["Depth temporal-consistency A/B (lower static-point wobble = more consistent)\n"]

    def score(name, dr):
        try:
            td = sample_tracked_depths(dr, frames, max_frames=nf)
            r = consistency_score(td)
            lines.append(f"{name:26s} {r.summary()}")
            return r
        except Exception as exc:
            lines.append(f"{name:26s} FAILED: {exc}")
            return None

    ru = score("UniDepthV2 (single-image)", unidepth_result)

    # flow-filtered UniDepth (runs anywhere)
    try:
        ff = depthmod.run(session, model="flow_filter", base=unidepth_result, max_frames=nf)
        score("UniDepth + flow_filter", ff)
    except Exception as exc:
        lines.append(f"UniDepth + flow_filter    FAILED: {exc}")

    # Video-Depth-Anything (needs the package + VDA_CKPT env var)
    try:
        vda = depthmod.run(session, model="video_depth_anything",
                           intrinsics=unidepth_result.intrinsics, max_frames=nf)
        rv = score("Video-Depth-Anything", vda)
        if ru and rv:
            better = rv.median_wobble_pct < ru.median_wobble_pct
            lines.append(
                f"\nVERDICT: VDA {'BEATS' if better else 'does NOT beat'} UniDepth "
                f"({rv.median_wobble_pct:.2f}% vs {ru.median_wobble_pct:.2f}% wobble)."
            )
    except Exception as exc:
        lines.append(f"Video-Depth-Anything      SKIPPED: {exc}")

    text = "\n".join(lines)
    print("\n" + text)
    out_path.write_text(text)


if __name__ == "__main__":
    main()
