"""Cross-embodiment retargeting commands.

`arm` is the wrist-to-Franka-style path. `humanoid` is the full-body BVH-to-humanoid
path from the General Motion Retargeting paper (arXiv:2510.02252). They deliberately
remain separate because their inputs and target embodiments are different.
"""

# Typer's supported declaration style uses Option calls as defaults.
# ruff: noqa: B008

from __future__ import annotations

from pathlib import Path

import typer

retarget_app = typer.Typer(help="L5 -- cross-embodiment retargeting (Master Spec §L5).")


@retarget_app.command("setup-humanoid")
def setup_humanoid(
    out: Path = typer.Option(
        None,
        "--out",
        help="Asset cache destination. Default: ACTUATE_HOME/vendor/gmr/<commit>.",
    ),
) -> None:
    """Download pinned Unitree G1 models and BVH mappings omitted from GMR's wheel."""
    from actuate.retarget.humanoid import GMRDependencyError, install_reference_assets

    typer.echo("downloading pinned GMR robot models and IK mappings ...")
    try:
        root = install_reference_assets(out)
    except (GMRDependencyError, OSError, ValueError) as exc:
        typer.secho(str(exc), fg="red")
        raise typer.Exit(1) from exc
    typer.secho(f"GMR assets ready: {root}", fg="green")


@retarget_app.command("train-arm")
def train_arm(
    embodiment: str = typer.Option("franka_panda", help="Target robot (embodiment registry)."),
    out: Path = typer.Option(
        None,
        "--out",
        help="Write the estimator here. Default: ACTUATE_HOME/models/<embodiment>_root_frame.pt.",
    ),
    pairs: int = typer.Option(2000, help="Number of sim (wrist, root) training pairs."),
    epochs: int = typer.Option(2000, help="Training epochs (full run ~1.5-2 hrs on a T4)."),
    length: int = typer.Option(32, help="Trajectory length per pair."),
    device: str = typer.Option("cpu", help="cpu or cuda."),
) -> None:
    """Train the SE(3)-equivariant root-frame estimator in sim (no real capture needed)."""
    from actuate.retarget.arm import default_model_path, train_estimator

    out = out or default_model_path(embodiment)

    typer.secho(f"training {embodiment} root-frame estimator: {pairs} pairs, {epochs} epochs "
                f"on {device} ...", fg="cyan")
    train_estimator(embodiment, out, n_pairs=pairs, epochs=epochs, length=length, device=device)
    typer.secho(f"wrote {out}", fg="green")


@retarget_app.command("arm")
def arm(
    in_: Path = typer.Option(..., "--in", help="Canonical episode JSON (from `canonical build`)."),
    embodiment: str = typer.Option("franka_panda", help="Target robot."),
    model: Path = typer.Option(
        None,
        "--model",
        help="Trained estimator. Default: the user-local model written by train-arm.",
    ),
    out: Path = typer.Option(None, "--out", help="Write the episode + action.robot here."),
    candidates: int = typer.Option(16, help="Root-frame hypotheses to sample."),
) -> None:
    """Retarget a canonical episode's wrist trajectory to a robot joint trajectory."""
    from actuate.retarget.arm import attach_to_episode, default_model_path, run
    from actuate.schema import CanonicalEpisode

    episode = CanonicalEpisode.model_validate_json(Path(in_).read_text())
    model = model or default_model_path(embodiment)
    if not model.exists():
        typer.secho(
            f"no estimator at {model}. Train it first with "
            f"`actuate retarget train-arm --embodiment {embodiment}`.",
            fg="red",
        )
        raise typer.Exit(1)
    result = run(episode, embodiment, model, n_candidates=candidates)

    typer.secho(result.summary(), bold=True)
    if result.convergence < 0.9:
        typer.secho(
            "  IK convergence < 90% -- the estimator likely needs the full training run "
            "(Kaggle), or the wrist trajectory is not cleanly reachable.", fg="yellow")

    if out:
        ep2 = attach_to_episode(episode, result)
        Path(out).write_text(ep2.model_dump_json())
        typer.secho(f"wrote {out}  (action.robot.{embodiment} added)", fg="green")
    else:
        typer.echo("(no --out; not written. Pass --out to persist action.robot.)")


@retarget_app.command("humanoid")
def humanoid(
    in_: Path = typer.Option(..., "--in", help="Full-body BVH motion file."),
    out: Path = typer.Option(..., "--out", help="Output directory for motion.npz/report.json."),
    source_format: str = typer.Option(
        "xsens",
        "--source-format",
        help="BVH skeleton convention: xsens, lafan1, or nokov.",
    ),
    robot: str = typer.Option("unitree_g1", help="GMR target robot."),
    start: int = typer.Option(None, help="First source frame (inclusive)."),
    end: int = typer.Option(None, help="Last source frame (exclusive)."),
    max_frames: int = typer.Option(None, help="Limit frames for a quick test/demo."),
    scale: float = typer.Option(0.01, help="Xsens BVH position scale (centimetres -> metres)."),
    fps: float = typer.Option(None, help="Override the frame rate stored in the BVH."),
    human_height: float = typer.Option(None, help="Override inferred human height in metres."),
    solver: str = typer.Option("daqp", help="Mink/qpsolvers backend."),
    damping: float = typer.Option(0.5, help="Differential IK damping."),
    velocity_limit: bool = typer.Option(
        True,
        "--velocity-limit/--no-velocity-limit",
        help="Constrain joints to 3*pi rad/s during each IK solve.",
    ),
    ground_align: bool = typer.Option(
        True,
        "--ground-align/--no-ground-align",
        help="Apply the paper's whole-clip minimum-height correction.",
    ),
    reset_to_zero: bool = typer.Option(
        False,
        "--reset-to-zero",
        help="Remove initial X/Y displacement and heading in Xsens BVH.",
    ),
    preview: Path = typer.Option(
        None,
        "--preview",
        help="Optional headless .mp4 or .gif preview (no desktop viewer).",
    ),
    gmr_root: Path = typer.Option(
        None,
        "--gmr-root",
        help="GMR checkout/cache root. Normally set by `setup-humanoid`.",
    ),
    verbose: bool = typer.Option(False, "--verbose", help="Print GMR body/joint mappings."),
) -> None:
    """Retarget full-body BVH motion to a humanoid with the GMR paper method."""
    from actuate.retarget.humanoid import (
        GMRDependencyError,
        cli_progress_printer,
        render_preview,
        retarget_bvh,
    )

    try:
        result = retarget_bvh(
            in_,
            source_format=source_format,
            robot=robot,
            scale=scale,
            start=start,
            end=end,
            max_frames=max_frames,
            reset_to_zero=reset_to_zero,
            fps=fps,
            actual_human_height=human_height,
            solver=solver,
            damping=damping,
            velocity_limit=velocity_limit,
            ground_align=ground_align,
            verbose=verbose,
            progress=cli_progress_printer(),
            reference_root=gmr_root,
        )
        motion_path, report_path = result.write(out)
        if preview is not None:
            render_preview(result, preview)
    except (GMRDependencyError, FileNotFoundError, ValueError) as exc:
        typer.secho(str(exc), fg="red")
        raise typer.Exit(1) from exc

    typer.secho(result.summary(), bold=True, fg="green")
    typer.echo(f"motion: {motion_path}")
    typer.echo(f"report: {report_path}")
    if result.quality.warnings:
        for warning in result.quality.warnings:
            typer.secho(f"warning: {warning}", fg="yellow")
    if preview is not None:
        typer.echo(f"preview: {preview}")
