"""`actuate run all` -- thin CLI over the library pipeline (Phase 6 refactor).

The orchestration now lives in `actuate.pipeline`; this file only parses the YAML profile,
wires a typer-printing reporter + an interactive confirm, and prints the final summary. The
SDK drives the exact same `run_pipeline` with its own reporter -- that is what makes "CLI
wraps SDK wraps library" real rather than aspirational.
"""

from __future__ import annotations

from pathlib import Path

import typer

from actuate.pipeline import run_pipeline

run_app = typer.Typer(help="One-command pipeline (Master Spec §2.2): "
                           "ingest -> perceive -> fuse -> canonical -> certify -> "
                           "retarget -> validate -> language -> package -> viz.")

_COLORS = {"done": "green", "skipped": "yellow", "flag": "yellow", "info": "cyan"}


@run_app.command("all")
def run_all(
    config: Path = typer.Option(..., help="YAML profile (see profiles/)."),
    in_path: Path = typer.Option(..., "--in", help="processed/<session> directory."),
    out: Path = typer.Option(..., help="Output directory (checkpoint, canonical, exports)."),
    resume: bool = typer.Option(True, help="Skip stages already checkpointed as done."),
) -> None:
    """Run the whole pipeline from a processed session to packaged datasets + viz."""
    import yaml

    profile = yaml.safe_load(config.read_text(encoding="utf-8"))

    def reporter(stage: str, status: str, note: str) -> None:
        if status == "flag":
            typer.secho(f"  [{stage}] FLAG: {note}", fg="yellow")
        elif status == "info":
            typer.secho(f"  [{stage}] {note}", fg="cyan")
        else:
            typer.secho(f"  [{stage}] {status.upper()}: {note}"
                        if status == "skipped" else f"  [{stage}] done  {note}",
                        fg=_COLORS.get(status, "white"))

    result = run_pipeline(in_path, out, profile, resume=resume,
                          reporter=reporter, confirm=typer.confirm)

    typer.secho(f"\npipeline complete in {result.seconds:.0f}s — stage summary:", bold=True)
    for stage, rec in result.checkpoint.items():
        if stage.startswith("_") or not isinstance(rec, dict):
            continue                       # _capture_id / _tier / _sim are metadata
        color = _COLORS.get(rec["status"], "red")
        typer.secho(f"  {stage:12s} {rec['status']:8s} {rec.get('note', '')}", fg=color)
    typer.echo(f"\ncanonical: {result.canonical_path}")
    typer.secho("nothing was delivered — `actuate deliver` is a separate, gated command "
                "(and the delivery bucket does not exist).", fg="yellow")
