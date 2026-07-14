"""`actuate schema` — freeze and verify the canonical contract."""

from __future__ import annotations

import typer

from actuate.schema import SCHEMA_VERSION, dump_json_schema, frozen_schema_path

schema_app = typer.Typer(help="Canonical schema (Master Spec §3) — the freeze point.")


@schema_app.command("freeze")
def freeze(
    check: bool = typer.Option(
        False, "--check", help="Exit non-zero if the frozen file is stale. Used by CI."
    ),
) -> None:
    """Write (or verify) the versioned JSON Schema for CanonicalEpisode."""
    path = frozen_schema_path()
    current = dump_json_schema()

    if check:
        if not path.exists():
            typer.secho(f"no frozen schema for v{SCHEMA_VERSION}: {path}", fg="red")
            raise typer.Exit(1)
        if path.read_text(encoding="utf-8") != current:
            typer.secho(
                f"the models have drifted from frozen schema v{SCHEMA_VERSION}.\n\n"
                "The canonical schema is the contract every exporter compiles against. "
                "Changing it silently breaks them against data they believe they "
                "understand. Bump SCHEMA_VERSION in actuate/schema/version.py, then:\n"
                "    actuate schema freeze",
                fg="red",
            )
            raise typer.Exit(1)
        typer.secho(f"frozen schema v{SCHEMA_VERSION} is current", fg="green")
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(current, encoding="utf-8")
    typer.secho(f"froze schema v{SCHEMA_VERSION} -> {path}", fg="green")


@schema_app.command("show")
def show() -> None:
    """Print the JSON Schema to stdout."""
    typer.echo(dump_json_schema())
