"""`actuate ingest` -- L0 minimal ingestion + aligned-capture mode (Phase 5 Part E)."""

from __future__ import annotations

from pathlib import Path

import typer

ingest_app = typer.Typer(help="L0 — minimal ingest for processed sessions (Master Spec §L0).")


@ingest_app.command("run")
def run_cmd(
    in_path: Path = typer.Option(..., "--in", help="processed/<session> directory."),
    rig: str = typer.Option(..., help="Rig type (e.g. head_mounted)."),
    aligned_robot: str = typer.Option(
        None, help="Claim this capture shares the robot's camera config. The claim is "
        "VERIFIED against registered calibration — unverifiable or mismatching claims "
        "FLAG and stay stage1_volume."),
    store: Path = typer.Option(None, help="Where the capture manifest goes "
                                          "(default: the session dir)."),
) -> None:
    """Content-address one processed session and verify any aligned-capture claim."""
    from actuate.ingest import run_ingest

    try:
        res = run_ingest(rig, in_path, store=store, aligned_robot=aligned_robot)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        typer.secho(f"INGEST REFUSED\n\n{exc}", fg="red")
        raise typer.Exit(1) from exc

    color = "green" if not res.flags else "yellow"
    typer.secho(res.summary(), fg=color, bold=True)
    typer.echo(f"  manifest: {res.manifest_path}")
    typer.echo("  catalog registration: WRITTEN-ONLY (no Postgres on this machine — "
               "the tier travels on the manifest and the canonical episode)")
