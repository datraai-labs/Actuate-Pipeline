"""The simplified, SDK-backed CLI -- the common case in one command (Phase 6 Part B).

    actuate login                                  # one-time
    actuate process ./video.mp4 --task "pick up cup" --embodiment franka_panda
    actuate run ./video.mp4 --task "..." --out ./ds/ --format lerobot_v3   # process+export
    actuate export ./out/ --format lerobot_v3
    actuate status ; actuate report

These wrap `actuate.sdk` (CLI wraps SDK wraps library). The advanced tree (`actuate run all`,
`actuate ingest run`, `actuate certify`, ...) stays for power users. Errors say what to DO,
not just what broke; long operations show progress through the reporter.
"""

from __future__ import annotations

from pathlib import Path

import typer

# --- login / config -------------------------------------------------------------------
login_app = typer.Typer(help="Authenticate (one-time). Local by default; no key needed.")
config_app = typer.Typer(help="View / set defaults (~/.actuate/config.json).")


@login_app.callback(invoke_without_command=True)
def login(
    ctx: typer.Context,
    local: bool = typer.Option(False, "--local", help="Configure for local processing."),
    key: str = typer.Option(None, "--key", help="DatraAI API key for cloud mode (future)."),
) -> None:
    """Set up Actuate. `--local` (default) runs on this machine, no key. `--key` is cloud."""
    if ctx.invoked_subcommand is not None:
        return
    from actuate.config import auth

    if key:
        auth.login(mode="cloud", api_key=key)
        typer.secho("cloud mode configured. Note: managed processing is coming soon; local "
                    "processing works today.", fg="yellow")
    else:
        auth.login(mode="local")
        typer.secho("local mode ready — no API key needed. "
                    "Process data with `actuate process <source>`.", fg="green")


@config_app.command("show")
def config_show() -> None:
    """Print the current config (the API key is masked)."""
    import json

    from actuate.config import auth

    typer.echo(json.dumps(auth.redacted(), indent=2))


@config_app.command("set")
def config_set(key: str = typer.Argument(...), value: str = typer.Argument(...)) -> None:
    """Set a default, e.g. `actuate config set default_embodiment aloha_v2`."""
    from actuate.config import auth

    try:
        auth.set_default(key, value)
        typer.secho(f"set {key} = {value}", fg="green")
    except ValueError as exc:
        typer.secho(str(exc), fg="red")
        raise typer.Exit(1) from exc


# --- process / run / export -----------------------------------------------------------
def _reporter(stage: str, status: str, note: str) -> None:
    color = {"done": "green", "skipped": "yellow", "flag": "yellow", "info": "cyan"}
    if status == "flag":
        typer.secho(f"  [{stage}] {note}", fg="yellow")
    else:
        typer.secho(f"  [{stage}] {status}: {note}", fg=color.get(status, "white"))


def _run_summary(run) -> None:
    s = run.summary()
    typer.secho(f"\n{run.status}  quality {s['quality']}/5  |  {s['num_frames']} frames  "
                f"|  task: {s['task']!r}", bold=True)
    typer.echo(f"canonical: {s['canonical_path']}")


def process_cmd(
    source: str = typer.Argument(..., help="Video / session dir, or hf:// s3:// https:// "
                                          "openx:// source."),
    task: str = typer.Option(None, help="Task description. Auto-detected via VLM if a key "
                                        "is set, else you'll be prompted."),
    rig: str = typer.Option("auto", help="auto (detect from video) | head_mounted | stereo "
                                         "| teleop_robot | ..."),
    embodiment: str = typer.Option(None, help="Robot for retargeting/robot-space export "
                                              "(defaults to your config)."),
    arm_model: Path = typer.Option(
        None,
        "--arm-model",
        help="Optional trained root-frame estimator. Without this, Actuate uses the "
        "user-local model written by `actuate retarget train-arm`.",
    ),
    max_frames: int = typer.Option(None, help="Cap frames (default: full video)."),
    out: str = typer.Option(None, help="Working directory (default: ./actuate_runs)."),
    export: str = typer.Option(None, "--export", help="Also export in one shot: "
                                                      "lerobot_v3 | rlds."),
    dataset_out: str = typer.Option("./dataset/", "--dataset-out",
                                    help="Where --export writes the dataset."),
    to_s3: bool = typer.Option(None, "--to-s3/--local", help="Upload artifacts to S3 and "
                               "clean local (overrides `config storage`)."),
    redact_pii: bool = typer.Option(False, "--redact-pii", help="Blur faces and mark "
                                    "pii_status=passed (required, with consent, to deliver)."),
) -> None:
    """Process a source into a certified canonical episode + Rerun recording.

    Add `--export lerobot_v3` to process AND export in one command (the one-shot; the
    power-user config-driven form is `actuate run all --config`).
    """
    import actuate

    prompt_fn = (lambda msg: typer.prompt(msg, default="")) if task is None else None
    try:
        run = actuate.process(source, rig=rig, embodiment=embodiment, task=task,
                              max_frames=max_frames, out=out, reporter=_reporter,
                              prompt_fn=prompt_fn, redact_pii=redact_pii,
                              retarget={"arm_model": str(arm_model)} if arm_model else {})
    except FileNotFoundError as exc:
        typer.secho(f"cannot process: {exc}", fg="red")
        raise typer.Exit(1) from exc
    _run_summary(run)
    exported_dirs = []
    if export:
        try:
            res = run.export(export, path=dataset_out)
        except ValueError as exc:
            typer.secho(f"export failed: {exc}", fg="red")
            raise typer.Exit(1) from exc
        exported_dirs = [dataset_out]
        typer.secho(f"\nexported {res.n_frames} frames -> {res.path}  ({export})",
                    fg="green")
    else:
        typer.secho(f"export with:  actuate export {run._result.out} "
                    "--format lerobot_v3", fg="cyan")

    # S3 storage: --to-s3 flag wins, else the `config storage` default
    from actuate.config import auth

    use_s3 = to_s3 if to_s3 is not None else (auth.load_config().get("storage") == "s3")
    if use_s3:
        try:
            uris = run.upload_to_s3(export_dirs=exported_dirs, clean_local=True)
        except Exception as exc:
            typer.secho(f"S3 upload failed ({type(exc).__name__}: {exc}). Artifacts kept "
                        "locally. Check `actuate config` aws_profile + that the buckets "
                        "are deployed.", fg="red")
            raise typer.Exit(1) from exc
        typer.secho("\nstored in S3 (local copies cleaned):", fg="green", bold=True)
        typer.echo(f"  raw:       {uris.get('raw', '(no raw video)')}")
        typer.echo(f"  canonical: {uris['canonical']}")
        for name, files in uris.get("exports", {}).items():
            typer.echo(f"  export {name}: {len(files)} objects")


def export_cmd(
    processed: Path = typer.Argument(..., help="A run's output dir (has canonical.json)."),
    format: str = typer.Option("lerobot_v3", help="lerobot_v3 | rlds."),
    out: Path = typer.Option(None, help="Export path (default: <processed>/<format>)."),
    embodiment: str = typer.Option(None),
    push_hub: str = typer.Option(None, "--push-hub", help="Also push to this HF repo id."),
    private: bool = typer.Option(True, help="Push as a private HF dataset."),
) -> None:
    """Export an already-processed run to LeRobot v3 or RLDS (optionally push to HF)."""
    canon = processed / "canonical.json"
    if not canon.exists():
        typer.secho(f"no canonical.json in {processed}. Run `actuate process` first.",
                    fg="red")
        raise typer.Exit(1)
    session = _guess_session(processed)
    video = _first_video(session) if session else _first_video(processed)
    if video is None:
        typer.secho(
            "could not find the source video for export next to this run. Re-export from "
            f"the source in one shot:  actuate run <source> --out {out or './dataset'} "
            f"--format {format}", fg="red")
        raise typer.Exit(1)

    from actuate.pipeline.run import PipelineResult
    from actuate.sdk import ProcessingRun

    result = PipelineResult(out=processed, canonical_path=canon,
                            checkpoint={"canonical": {"status": "done"}}, seconds=0.0)
    run = ProcessingRun(session=video.parent,
                        profile={"video": video.name, "embodiment": embodiment},
                        _result=result)
    out = out or (processed / format)
    res = run.export(format, path=out, embodiment=embodiment)
    typer.secho(f"exported {res.n_frames} frames -> {res.path}  ({format})", fg="green")
    if push_hub:
        from actuate.sources import push_to_hub

        url = push_to_hub(out, push_hub, private=private)
        typer.secho(f"pushed -> {url}", fg="green")


def _guess_session(processed: Path) -> Path | None:
    """Find the staged session dir for an out dir (SDK names it <name>_out next to <name>)."""
    name = processed.name
    if name.endswith("_out"):
        cand = processed.parent / name[: -len("_out")]
        if cand.is_dir():
            return cand
    return None


def _first_video(session: Path):
    vids = [v for v in sorted(session.glob("*.mp4")) if "depth" not in v.name.lower()]
    return vids[0] if vids else None


def status_cmd() -> None:
    """Show auth + defaults (are you set up to process?)."""
    from actuate.config import auth

    cfg = auth.redacted()
    ready = auth.is_authenticated()
    typer.secho(f"mode: {cfg['mode']}  |  authenticated: {ready}", bold=True)
    typer.echo(f"default embodiment: {cfg['default_embodiment']}")
    typer.echo(f"default export:     {cfg['default_export_format']}")
    if not ready:
        typer.secho("run `actuate login --local` to get started.", fg="yellow")


def report_cmd(
    processed: Path = typer.Argument(..., help="A run's output dir (has canonical.json)."),
) -> None:
    """Print the quality certificate for a processed run."""
    from actuate.schema import CanonicalEpisode

    canon = processed / "canonical.json"
    if not canon.exists():
        typer.secho(f"no canonical.json in {processed}. Run `actuate process` first.",
                    fg="red")
        raise typer.Exit(1)
    ep = CanonicalEpisode.model_validate_json(canon.read_text(encoding="utf-8"))
    m = ep.episode_meta
    c = m.components
    fmt = lambda v: "not measured" if v is None else f"{v:.2f}"  # noqa: E731
    typer.secho(f"quality {m.quality}/5   speed {m.speed}   task {ep.task!r}", bold=True)
    for name in ("sync_integrity", "calibration_completeness", "perception_confidence",
                 "contact_consistency", "ik_convergence_rate"):
        typer.echo(f"  {name:26s} {fmt(getattr(c, name))}")
    typer.echo(f"  mistakes: {len(m.mistakes)}")
    typer.secho(f"  deliverable: {ep.is_deliverable}  "
                f"(consent={ep.consent.value}, pii={ep.pii_status.value})",
                fg="green" if ep.is_deliverable else "yellow")
