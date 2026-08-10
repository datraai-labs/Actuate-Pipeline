"""`actuate language` -- L6 rich-context annotation (Master Spec section L6)."""

from __future__ import annotations

from pathlib import Path

import typer

language_app = typer.Typer(help="L6 -- language rich-context (Master Spec section L6).")


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
                    "(judge below threshold) -- shipped with low confidence, not hidden",
                    fg="yellow", bold=True)

    target = out or in_path
    target.write_text(report.episode.model_dump_json(indent=2), encoding="utf-8")
    typer.secho(f"wrote {target}  (actual cost ${report.cost_usd:.4f})", fg="green")


@language_app.command("label-actions")
def label_actions_cmd(
    in_path: Path = typer.Option(..., "--in", help="Canonical episode JSON."),
    out: Path = typer.Option(None, help="Write the labelled episode here "
                                        "(defaults to --in, updated in place)."),
    use_vlm: bool = typer.Option(False, help="Refine ambiguous intervals with a VLM "
                                             "(billed; off by default -- geometry is $0)."),
) -> None:
    """Detect fine-grained atomic action intervals (closed 20-verb vocab) per actor.

    Geometry-only by default: L2 states + wrist velocity + finger curl + object proximity.
    Costs nothing. Actions the monocular bare-hand rig cannot support (pour, insert, ...)
    are NOT fabricated -- they stay in the vocabulary for cross-dataset comparison.
    """
    from actuate.language import label_actions
    from actuate.schema import CanonicalEpisode

    if use_vlm:
        from actuate.language import get_api_key

        if get_api_key() is None:
            typer.secho("--use-vlm needs ANTHROPIC_API_KEY; falling back to geometry only.",
                        fg="yellow")
            use_vlm = False
        elif not typer.confirm("--use-vlm will call the Anthropic API on ambiguous "
                               "intervals (small cost). Proceed?"):
            use_vlm = False

    ep = CanonicalEpisode.model_validate_json(in_path.read_text(encoding="utf-8"))
    res = label_actions(ep, use_vlm=use_vlm)
    typer.secho(res.summary(), bold=True)
    for iv in res.intervals[:20]:
        flag = "  [FLAG]" if iv.confidence <= 0.5 else ""
        typer.echo(f"  {iv.actor.value:11s} {iv.action_label.value:10s} "
                   f"[{iv.start_frame:5d}-{iv.end_frame:5d}] conf {iv.confidence:.2f}{flag}")
    if len(res.intervals) > 20:
        typer.echo(f"  ... and {len(res.intervals) - 20} more")

    target = out or in_path
    target.write_text(res.episode.model_dump_json(indent=2), encoding="utf-8")
    typer.secho(f"wrote {target}", fg="green")
