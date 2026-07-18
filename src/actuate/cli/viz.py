"""`actuate viz <session>` -- see the pipeline. Master Spec §F.

Runs the perception stages on a session and logs every modality to Rerun on one scrubable
timeline. Default writes a self-contained `.rrd` (openable later, no GPU); `--live` streams to a
running viewer as each stage finishes, which is what you want when debugging a stage.

Perception is GPU-heavy and, on a 4 GB card, SAM2 propagation can thrash to system RAM (a 20-
frame chunk took 86 minutes once). `--cache` pickles each stage's result under the session and
reuses it on the next run -- so looking at the same episode again is a file load, not another
model run. A stage is re-run only when its inputs change (frame count, and object prompts) or
`--force` is passed.

This is a thin wrapper: all the logging logic is in `actuate.viz.log_episode`, importable
without a viewer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import typer

from actuate.pipeline.cache import stage_cached

viz_app = typer.Typer(help="F -- Rerun visualization: see the pipeline (Master Spec §F).")

_STAGES = ("depth", "hands", "objects", "fusion", "slam")


def _stage_cached(session: Path, stage: str, key: str, *, use_cache: bool, force: bool,
                  run_fn: Callable):
    """CLI adapter over the library `pipeline.cache.stage_cached` (cache logic lives there
    now so the SDK can reuse it). Prints a best-effort write miss with typer."""
    return stage_cached(session, stage, key, use_cache=use_cache, force=force,
                        run_fn=run_fn,
                        warn=lambda m: typer.secho(f"  ({m})", fg="yellow"))


@viz_app.command("show")
def show(
    session: Path = typer.Argument(..., help="processed/<session> directory"),
    out: Path = typer.Option(None, "--out", help="Write the .rrd here (default: <session>.rrd)."),
    live: bool = typer.Option(False, "--live", help="Stream to a running viewer as stages run."),
    stages: str = typer.Option(
        "depth,hands,objects,fusion",
        help="Comma-separated perception stages to run. SLAM/hands already ran in Parts A/B.",
    ),
    max_frames: int = typer.Option(60, help="Cap frames (perception is GPU-heavy)."),
    prompts: str = typer.Option("stapler,paper,document,box", help="Object detection prompts."),
    task: str = typer.Option(None, help="Operator-verified task (for the canonical episode)."),
    cache: bool = typer.Option(
        False, "--cache", help="Reuse cached perception outputs when inputs are unchanged."
    ),
    force: bool = typer.Option(
        False, "--force", help="Re-run every stage even if a matching cache exists (with --cache)."
    ),
) -> None:
    """Run perception on a session and visualize it in Rerun."""
    import rerun as rr

    from actuate.config import RigType
    from actuate.viz import log_episode

    want = {s.strip() for s in stages.split(",") if s.strip()}
    bad = want - set(_STAGES)
    if bad:
        raise typer.BadParameter(f"unknown stage(s) {sorted(bad)}; choose from {_STAGES}")
    if force and not cache:
        typer.secho("--force has no effect without --cache (nothing is cached).", fg="yellow")

    session = session.resolve()
    if not session.exists():
        raise typer.BadParameter(f"session dir not found: {session}")

    # A user who just dropped a raw video in has no session_meta.json; derive it from the
    # video so perception can read frame_count/fps without a manual step.
    from actuate.ingest.run import ensure_session_meta

    try:
        meta = ensure_session_meta(session)
        if meta.get("source") == "auto_from_video":
            typer.secho(f"generated session_meta.json: {meta['frame_count']} frames @ "
                        f"{meta['fps_nominal']} fps, {meta['width']}x{meta['height']}",
                        fg="cyan")
    except FileNotFoundError as exc:
        raise typer.BadParameter(str(exc)) from exc

    prompt_list = [p.strip() for p in prompts.split(",")]

    def report(stage: str, source: str) -> None:
        tag = "cached" if source == "cache" else "ran"
        typer.secho(f"{stage} ({tag}) ...", fg="green" if source == "cache" else "cyan")

    rr.init("actuate", spawn=live)

    hands = depth = objects = fusion = slam = None
    K = None

    if "slam" in want:
        from actuate.perception.slam import runner as slam_runner

        def _run_slam():
            try:
                return slam_runner.run(session, max_frames=max_frames)
            except Exception as exc:  # SLAM needs gyro; don't sink the whole viz if it's absent
                typer.secho(f"  slam skipped: {exc}", fg="yellow")
                return None

        slam, src = _stage_cached(session, "slam", f"n={max_frames}",
                                  use_cache=cache, force=force, run_fn=_run_slam)
        report("slam", src)

    if "depth" in want:
        from actuate.perception import depth as depthmod

        depth, src = _stage_cached(
            session, "depth", f"n={max_frames}", use_cache=cache, force=force,
            run_fn=lambda: depthmod.run(session, max_frames=max_frames),
        )
        report("depth", src)
        K = depth.intrinsics

    if "hands" in want:
        from actuate.perception import hands as handsmod

        hands, src = _stage_cached(
            session, "hands", f"n={max_frames}", use_cache=cache, force=force,
            run_fn=lambda: handsmod.run(session, max_frames=max_frames, prefilter=False),
        )
        report("hands", src)

    if "objects" in want:
        from actuate.perception import objects as objmod

        objects, src = _stage_cached(
            session, "objects", f"n={max_frames}|{','.join(prompt_list)}",
            use_cache=cache, force=force,
            run_fn=lambda: objmod.run(session, prompts=prompt_list,
                                      max_frames=max_frames, chunk=max_frames),
        )
        report("objects", src)

    if "fusion" in want and hands is not None:
        # Fusion is instant and derives from hands+objects -- always recompute, never cache.
        from actuate import fusion as fusionmod

        fusion = fusionmod.run(hands, rig=RigType.HEAD_MOUNTED, objects=objects)
        typer.secho("fusion (L2 arbiter) ...", fg="cyan")

    # read the video frames for the RGB/point-cloud overlay
    import cv2

    video = session / "redacted_compressed.mp4"
    if not video.exists():
        video = session / "compressed.mp4"
    frames = []
    cap = cv2.VideoCapture(str(video))
    for _ in range(max_frames):
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()

    typer.secho("logging to Rerun ...", fg="cyan")
    counts = log_episode(
        hands=hands, depth=depth, objects=objects, fusion=fusion, slam=slam,
        video_frames=frames, intrinsics=K, max_frames=max_frames,
    )
    typer.secho(f"logged modalities: {counts}", bold=True)

    if not live:
        target = out or session.with_suffix(".rrd")
        rr.save(str(target))
        typer.secho(f"wrote {target}", fg="green")
        typer.echo(f"open with:  rerun {target}")
    else:
        typer.secho("streaming to the live viewer; press Ctrl-C to stop.", fg="green")
