"""`actuate viz <session>` -- see the pipeline. Master Spec §F.

Runs the perception stages on a session and logs every modality to Rerun on one scrubable
timeline. Default writes a self-contained `.rrd` (openable later, no GPU); `--live` streams to a
running viewer as each stage finishes, which is what you want when debugging a stage.

This is a thin wrapper: all the logging logic is in `actuate.viz.log_episode`, importable
without a viewer.
"""

from __future__ import annotations

from pathlib import Path

import typer

viz_app = typer.Typer(help="F -- Rerun visualization: see the pipeline (Master Spec §F).")

_STAGES = ("depth", "hands", "objects", "fusion", "slam")


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
) -> None:
    """Run perception on a session and visualize it in Rerun."""
    import rerun as rr

    from actuate.config import RigType
    from actuate.viz import log_episode

    want = {s.strip() for s in stages.split(",") if s.strip()}
    bad = want - set(_STAGES)
    if bad:
        raise typer.BadParameter(f"unknown stage(s) {sorted(bad)}; choose from {_STAGES}")

    session = session.resolve()
    if not session.exists():
        raise typer.BadParameter(f"session dir not found: {session}")

    rr.init("actuate", spawn=live)

    hands = depth = objects = fusion = slam = None
    K = None

    if "slam" in want:
        typer.secho("slam (visual-inertial ego-motion) ...", fg="cyan")
        from actuate.perception.slam import runner as slam_runner

        try:
            slam = slam_runner.run(session, max_frames=max_frames)
        except Exception as exc:  # SLAM needs gyro; don't sink the whole viz if it's absent
            typer.secho(f"  slam skipped: {exc}", fg="yellow")
            slam = None
    if "depth" in want:
        typer.secho("depth (UniDepthV2) ...", fg="cyan")
        from actuate.perception import depth as depthmod

        depth = depthmod.run(session, max_frames=max_frames)
        K = depth.intrinsics
    if "hands" in want:
        typer.secho("hands (WiLoR) ...", fg="cyan")
        from actuate.perception import hands as handsmod

        hands = handsmod.run(session, max_frames=max_frames, prefilter=False)
    if "objects" in want:
        typer.secho("objects (Grounding DINO + SAM2) ...", fg="cyan")
        from actuate.perception import objects as objmod

        objects = objmod.run(
            session, prompts=[p.strip() for p in prompts.split(",")],
            max_frames=max_frames, chunk=max_frames,
        )
    if "fusion" in want and hands is not None:
        typer.secho("fusion (L2 arbiter) ...", fg="cyan")
        from actuate import fusion as fusionmod

        fusion = fusionmod.run(hands, rig=RigType.HEAD_MOUNTED, objects=objects)

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
