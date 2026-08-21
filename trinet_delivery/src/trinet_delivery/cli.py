from pathlib import Path

import typer

from trinet_delivery.run import RunError, RunInputError, prepare_local_run

app = typer.Typer(no_args_is_help=True)


@app.callback()
def main() -> None:
    pass


@app.command()
def run(
    source: str,
    run_dir: Path,
    ui: bool = typer.Option(False, "--ui"),
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
    if ui:
        typer.echo("ui=not_implemented_until_hosted_phase")
