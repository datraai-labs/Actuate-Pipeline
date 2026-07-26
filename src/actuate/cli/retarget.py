"""`actuate retarget arm` -- wrist trajectory -> robot joint trajectory (Master Spec §L5).

Thin wrapper over `actuate.retarget.arm`. `train-arm` generates sim data and trains the
root-frame estimator (full run: Kaggle T4). `arm` loads a canonical episode, retargets its wrist
trajectory to the target robot, and writes the episode back with `action.robot.<embodiment>`.
"""

from __future__ import annotations

from pathlib import Path

import typer

retarget_app = typer.Typer(help="L5 -- cross-embodiment retargeting (Master Spec §L5).")


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
