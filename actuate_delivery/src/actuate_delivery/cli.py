import csv
import json
import os
from pathlib import Path
from typing import Annotated

import typer

from actuate_delivery.qc import controlled_limitations
from actuate_delivery.run import (
    RunError,
    RunInputError,
    _delivery_review,
    _write_review,
    processing_progress,
    telemetry_policy,
)
from actuate_delivery.workflow import (
    STAGES,
    approve_stage,
    artifact_failures,
    bind_calibration,
    run_stage,
    workflow_state,
)

app = typer.Typer(no_args_is_help=True)


@app.callback()
def main() -> None:
    pass


def _show(stage, summary, run_dir):
    typer.echo(f"\n{dict(STAGES)[stage]}")
    for key, value in summary.items():
        typer.echo(f"  {key.replace('_', ' ')}: {value}")
    failures = artifact_failures(run_dir, stage)
    if failures:
        typer.echo("  Attention required:")
        for failure in failures:
            stream = f" / {failure['camera_stream_id']}" if failure["camera_stream_id"] else ""
            typer.echo(
                f"    {failure['episode']} - {failure['stage']}{stream}: {failure['message']}"
            )
    if stage == "inventory" and summary["incomplete_captures"]:
        typer.echo(f"  Attention required: {summary['incomplete_captures']} incomplete candidate(s)")
    if stage == "timing" and (summary["unavailable"] or summary["failed"]):
        typer.echo(
            f"  Attention required: {summary['unavailable']} unavailable and "
            f"{summary['failed']} failed timing result(s)"
        )
    if stage == "timing" and summary["unmatched_stereo_frames"]:
        typer.echo(
            f"  Attention required: {summary['unmatched_stereo_frames']} unmatched stereo frame(s)"
        )
    if stage == "qc" and (summary["blocking_checks"] or summary["material_checks"]):
        checks = summary["blocking_checks"] + summary["material_checks"]
        typer.echo(f"  Attention required: {', '.join(checks)}")
    progress = processing_progress(run_dir / "run.sqlite", stage)
    if progress["total"]:
        typer.echo(f"  Items: {progress['completed']} of {progress['total']} processed")
        for item in progress["items"]:
            elapsed = (f" in {item['elapsed_seconds']:.3f}s"
                       if item["elapsed_seconds"] is not None else "")
            typer.echo(f"    {item['label']}: {item['outcome'] or item['status']}{elapsed}")


def _review_in_terminal(run_dir: Path):
    review_path, entries = _delivery_review(run_dir / "run.sqlite", run_dir)
    limitations = {
        entry["row"]["capture_id"]: controlled_limitations(entry["internal_qc"])
        for entry in entries
    }
    with review_path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        if row["grouping_status"] != "complete" or row["decision"]:
            continue
        typer.echo(f"\n{row['episode_id']} - {row['source_relative_directory']}/{row['source_group']}")
        typer.echo(f"  layout: {row['capture_layout']}")
        typer.echo(f"  QC: {row['pass_count']} pass, {row['fail_count']} fail, "
                   f"{row['unknown_count']} unknown")
        if row["blocking_checks"]:
            typer.echo(f"  blocking checks: {row['blocking_checks']}")
        if row["material_checks"]:
            typer.echo(f"  material checks: {row['material_checks']}")
        decision = typer.prompt("Decision [include/exclude/stop]", default="stop").strip().lower()
        if decision == "stop":
            break
        if decision not in ("include", "exclude"):
            raise RunError(f"Invalid review decision: {decision}")
        if decision == "include" and limitations[row["capture_id"]]:
            typer.echo("  Measured issues recorded automatically:")
            for limitation in limitations[row["capture_id"]]:
                typer.echo(f"    - {limitation}")
        row.update({
            "decision": decision,
            "limitations_json": json.dumps(
                limitations[row["capture_id"]] if decision == "include" else []),
        })
    _write_review(review_path, rows)
    _delivery_review(run_dir / "run.sqlite", run_dir)


@app.command()
def run(
    source: str,
    run_dir: Path,
    ui: bool = typer.Option(False, "--ui"),
    output: Annotated[Path | None, typer.Option("--output")] = None,
    calibration: Annotated[Path | None, typer.Option("--calibration")] = None,
) -> None:
    if ui:
        import uvicorn

        from actuate_delivery.web import create_app

        ui_output = output or run_dir / "delivery/dataset"
        uvicorn.run(
            create_app(Path(source), run_dir, ui_output),
            host=os.environ.get("ACTUATE_UI_HOST", "127.0.0.1"),
            port=int(os.environ.get("PORT", "8000")),
        )
        return
    if output is None:
        raise typer.BadParameter("Required for the final delivery", param_hint="--output")
    if calibration is not None and not calibration.is_file():
        raise typer.BadParameter("Calibration file does not exist", param_hint="--calibration")
    try:
        for stage, name in STAGES:
            current = next(item for item in workflow_state(run_dir) if item["stage"] == stage)
            if stage == "delivery" and current["status"] == "complete":
                _show(stage, current["summary"], run_dir)
                return
            if stage != "delivery":
                result = run_stage(source, run_dir, stage)
                _show(stage, result.summary, run_dir)
                current = next(item for item in workflow_state(run_dir)
                               if item["stage"] == stage)
                if current["approved"]:
                    typer.echo("  Previously approved evidence remains unchanged.")
                    continue
                if stage == "review" and result.summary["pending"]:
                    _review_in_terminal(run_dir)
                    result = run_stage(source, run_dir, stage)
                    _show(stage, result.summary, run_dir)
                    if result.summary["pending"]:
                        typer.echo(f"Stopped safely. Resume with the same RUN_DIR: {run_dir}")
                        return
                if not typer.confirm(f"Approve {name} and continue?", default=False):
                    typer.echo(f"Stopped safely. Resume with the same RUN_DIR: {run_dir}")
                    return
                approved_by = ""
                if stage == "review":
                    approved_by = typer.prompt(
                        "Reviewer name (optional, press Enter to skip)",
                        default="", show_default=False,
                    ).strip()
                approve_stage(run_dir, stage, approved_by)
                continue
            policy = telemetry_policy(run_dir)
            if policy["requires_choice"]:
                typer.echo(
                    f"Telemetry is available for {policy['episodes_with_telemetry']} of "
                    f"{policy['included_episodes']} included episodes."
                )
                choice = typer.prompt(
                    "Telemetry [include available/exclude all/stop]", default="stop"
                ).strip().lower()
                if choice == "stop":
                    typer.echo(f"Stopped safely before delivery. Resume with the same RUN_DIR: {run_dir}")
                    return
                choices = {"include available": "include_available", "exclude all": "exclude_all"}
                if choice not in choices:
                    raise RunError(f"Invalid telemetry choice: {choice}")
                telemetry_policy(run_dir, choices[choice])
            if not typer.confirm("Build and validate the customer delivery now?", default=False):
                typer.echo(f"Stopped safely before delivery. Resume with the same RUN_DIR: {run_dir}")
                return
            if calibration is not None:
                bind_calibration(run_dir, json.loads(calibration.read_text()))
                typer.echo(f"Using explicitly selected calibration: {calibration}")
            result = run_stage(source, run_dir, stage, output)
            _show(stage, result.summary, run_dir)
    except RunInputError as error:
        raise typer.BadParameter(str(error), param_hint="SOURCE") from error
    except RunError as error:
        typer.echo(f"run_error={error}", err=True)
        raise typer.Exit(1) from error


@app.command()
def status(run_dir: Path) -> None:
    for item in workflow_state(run_dir):
        state = "approved" if item["approved"] else item["status"]
        typer.echo(f"{item['stage']}: {state}")
        if item["error"]:
            typer.echo(f"  error: {item['error']}")
        if item["summary"]:
            for key, value in item["summary"].items():
                typer.echo(f"  {key.replace('_', ' ')}: {value}")
        progress = item["progress"]
        if progress["total"]:
            typer.echo(f"  progress: {progress['completed']} of {progress['total']}")
            if progress["current"]:
                typer.echo(f"  current: {progress['current']['label']}")
            for work in progress["items"]:
                elapsed = (f" ({work['elapsed_seconds']:.3f}s)"
                           if work["elapsed_seconds"] is not None else "")
                typer.echo(f"    {work['label']}: {work['outcome'] or work['status']}{elapsed}")
    failures = artifact_failures(run_dir)
    if failures:
        typer.echo("failures:")
        for failure in failures:
            typer.echo(f"  {failure['episode']} - {failure['stage']}: {failure['message']}")
