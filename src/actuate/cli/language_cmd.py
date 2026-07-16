"""`actuate language` -- L6 rich-context annotation (Master Spec §L6)."""

from __future__ import annotations

from pathlib import Path

import typer

language_app = typer.Typer(help="L6 — language rich-context (Master Spec §L6).")


@language_app.command("annotate")
def annotate_cmd(
    in_path: Path = typer.Option(..., "--in", help="Canonical episode JSON."),
    video: Path = typer.Option(..., help="The REDACTED video (frames for the VLM+judge)."),
    session: Path = typer.Option(None, help="processed/<session> dir with phases.json "
                                            "(v1 phase boundaries = subgoal anchors)."),
    n_paraphrases: int = typer.Option(5, min=3, max=5),
    out: Path = typer.Option(None, help="Write the annotated episode JSON here "
                                        "(defaults to --in, updated in place)."),
    yes: bool = typer.Option(False, "--yes", "-y",
                             help="Skip the cost-estimate confirmation."),
) -> None:
    """Annotate one canonical episode: paraphrases, subtasks, subgoal frames, judge gate.

    Needs ANTHROPIC_API_KEY (env or .env.local). Without it: warns and exits 0 -- language
    annotation degrades gracefully, it never fails the pipeline.
    """
    from actuate.language import annotate, estimate_annotation_cost, get_api_key
    from actuate.language.annotate import _load_segments
    from actuate.schema import CanonicalEpisode

    episode = CanonicalEpisode.model_validate_json(in_path.read_text(encoding="utf-8"))

    if get_api_key() is None:
        typer.secho(
            "WARNING: no ANTHROPIC_API_KEY (env or .env.local). Skipping language "
            "annotation -- the pipeline continues without it.", fg="yellow", bold=True)
        raise typer.Exit(0)

    # cost estimate BEFORE any billed call (working discipline) -- uses the same merged
    # segment count annotate() will actually process
    n_segments = len(_load_segments(session)) if session else 0
    est = estimate_annotation_cost(n_segments)
    typer.echo(f"estimated cost: ~${est:.2f} "
               f"({n_segments} segments x (caption + judge) + 1 paraphrase call)")
    if not yes and not typer.confirm("proceed?"):
        raise typer.Exit(0)

    report = annotate(episode, session_dir=session, video=video,
                      n_paraphrases=n_paraphrases)
    typer.secho(report.summary(), bold=True)
    if report.skipped:
        raise typer.Exit(0)

    for i, p in enumerate(report.paraphrases, 1):
        typer.echo(f"  paraphrase {i}: {p}")
    for st in report.subtasks:
        typer.echo(f"  subtask [{st.start_frame}-{st.end_frame}] "
                   f"(conf {st.confidence:.2f}): {st.instruction}")
    if report.flagged_for_review:
        typer.secho(f"  {report.flagged_for_review} caption(s) FLAGGED FOR REVIEW "
                    "(judge below threshold) — shipped with low confidence, not hidden",
                    fg="yellow", bold=True)

    target = out or in_path
    target.write_text(report.episode.model_dump_json(indent=2), encoding="utf-8")
    typer.secho(f"wrote {target}  (actual cost ${report.cost_usd:.4f})", fg="green")
