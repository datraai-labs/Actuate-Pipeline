from pathlib import Path
from typing import Annotated

import typer

from trinet_delivery.run import (
    RunError,
    RunInputError,
    complete_local_delivery,
    prepare_local_run,
)

app = typer.Typer(no_args_is_help=True)


@app.callback()
def main() -> None:
    pass


@app.command()
def run(
    source: str,
    run_dir: Path,
    ui: bool = typer.Option(False, "--ui"),
    output: Annotated[Path | None, typer.Option("--output")] = None,
) -> None:
    try:
        result = prepare_local_run(source, run_dir)
    except RunInputError as error:
        raise typer.BadParameter(str(error), param_hint="SOURCE") from error
    except RunError as error:
        typer.echo(f"run_error={error}", err=True)
        raise typer.Exit(1) from error

    typer.echo(f"run_id={result.run_id}")
    typer.echo(f"ledger={run_dir / 'run.sqlite'}")
    typer.echo(f"files={result.files}")
    typer.echo(f"captures={result.captures}")
    typer.echo(f"preserved={result.files}")
    typer.echo(f"new={result.new}")
    typer.echo(f"changed={result.changed}")
    typer.echo(f"unchanged={result.unchanged}")
    typer.echo(f"removed={result.removed}")
    typer.echo(f"unique_captures={result.unique_captures}")
    typer.echo(f"duplicate_captures={result.duplicate_captures}")
    typer.echo(f"imu_decoded={result.imu_decoded}")
    typer.echo(f"imu_reused={result.imu_reused}")
    typer.echo(f"imu_failed={result.imu_failed}")
    typer.echo(f"vts_decoded={result.vts_decoded}")
    typer.echo(f"vts_reused={result.vts_reused}")
    typer.echo(f"vts_failed={result.vts_failed}")
    typer.echo(f"tel_decoded={result.tel_decoded}")
    typer.echo(f"tel_reused={result.tel_reused}")
    typer.echo(f"tel_failed={result.tel_failed}")
    typer.echo(f"video_verified={result.video_verified}")
    typer.echo(f"video_reused={result.video_reused}")
    typer.echo(f"video_failed={result.video_failed}")
    typer.echo(f"timing_created={result.timing_created}")
    typer.echo(f"timing_reused={result.timing_reused}")
    typer.echo(f"timing_unavailable={result.timing_unavailable}")
    typer.echo(f"timing_failed={result.timing_failed}")
    typer.echo(f"qc_created={result.qc_created}")
    typer.echo(f"qc_reused={result.qc_reused}")
    typer.echo(f"qc_failed={result.qc_failed}")
    if ui:
        typer.echo("ui=not_implemented_until_hosted_phase")
    if (result.imu_failed or result.vts_failed or result.tel_failed or result.video_failed
            or result.timing_failed or result.qc_failed):
        raise typer.Exit(1)
    if output is not None:
        try:
            delivery = complete_local_delivery(source, run_dir, output)
        except RunInputError as error:
            raise typer.BadParameter(str(error), param_hint="--output") from error
        except RunError as error:
            typer.echo(f"delivery_error={error}", err=True)
            raise typer.Exit(1) from error
        typer.echo(f"review_sheet={delivery.review_path}")
        typer.echo(f"delivery_status={delivery.status}")
        typer.echo(f"delivery_included={delivery.included}")
        typer.echo(f"delivery_excluded={delivery.excluded}")
        typer.echo(f"delivery_pending={delivery.pending}")
        if delivery.output is not None:
            typer.echo(f"delivery_output={delivery.output}")
