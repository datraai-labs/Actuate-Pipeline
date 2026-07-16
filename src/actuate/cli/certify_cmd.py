"""`actuate certify` -- L4 quality scoring (Master Spec §4 L4)."""

from __future__ import annotations

from pathlib import Path

import typer

certify_app = typer.Typer(help="L4 — quality certification (Master Spec §4 L4).")


@certify_app.command("run")
def run(
    in_path: Path = typer.Option(..., "--in", help="Canonical episode JSON."),
    session: Path = typer.Option(
        None, help="processed/<session> dir with the L0 sync report (quality_certificate.json). "
        "Without it sync_integrity is reported as not measured."),
    embodiment: str = typer.Option(
        None, help="Attach L5 strategy/eligibility for this embodiment (needs --arm-model "
        "results already computed; today the CLI scores the perception-side certificate)."),
    out: Path = typer.Option(None, help="Write the certified episode JSON here "
                                        "(defaults to --in, updated in place)."),
) -> None:
    """Score one canonical episode and write the certificate into it.

    The consent/PII gate is untouched: certification never makes an episode deliverable --
    only consent does. A quality=5 episode with consent=pending still cannot ship.
    """
    from actuate.certify import score
    from actuate.schema import CanonicalEpisode

    episode = CanonicalEpisode.model_validate_json(in_path.read_text(encoding="utf-8"))
    report = score(episode, embodiment, session_dir=session)

    typer.secho(report.summary(), bold=True)
    for n in report.notes:
        typer.secho(f"  note: {n}", fg="yellow")
    for m in report.mistakes[:10]:
        typer.secho(f"  mistake: {m}", fg="yellow")
    if len(report.mistakes) > 10:
        typer.secho(f"  ... and {len(report.mistakes) - 10} more", fg="yellow")

    ep = report.episode
    typer.secho(
        f"  deliverable: {'YES' if ep.is_deliverable else 'NO'}"
        f"  ({ep.delivery_block_reason() or 'clear'}) — quality does not open this gate",
        fg="green" if ep.is_deliverable else "yellow",
    )

    target = out or in_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(ep.model_dump_json(indent=2), encoding="utf-8")
    typer.secho(f"wrote {target}", fg="green")
