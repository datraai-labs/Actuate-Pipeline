"""The `actuate` Typer CLI — Master Spec §2.2.

**Thin.** Every command is a wrapper over a library call. Nothing here contains logic that
cannot be reached by importing `actuate` — that is the CLI/API-first rule, and the
import-linter contract in `.importlinter` enforces the direction (no layer may import
`actuate.cli`).

Command groups mirror the layers. Most print "not implemented" this increment: Increment 1
builds only the foundation (schema, io, catalog, infra) and stops for review.
"""

from __future__ import annotations

import typer

from actuate.cli.migrate import migrate_app
from actuate.cli.pipeline import canonical_app, package_app
from actuate.cli.schema_cmd import schema_app
from actuate.cli.storage import storage_app

app = typer.Typer(
    help="Actuate — multimodal capture to VLA-training-ready robot data.",
    no_args_is_help=True,
)

app.add_typer(schema_app, name="schema")
app.add_typer(storage_app, name="storage")
app.add_typer(migrate_app, name="migrate")
app.add_typer(canonical_app, name="canonical")
app.add_typer(package_app, name="package")


def _todo(group: str, spec: str) -> typer.Typer:
    """A command group that parses and honestly says it does nothing.

    Registering the full surface now means the shape of the CLI is reviewable before the
    layers land, and `--help` never advertises a capability that does not exist.
    """
    sub = typer.Typer(help=f"[NOT IMPLEMENTED — {spec}]", no_args_is_help=True)

    @sub.callback(invoke_without_command=True)
    def _cb(ctx: typer.Context) -> None:
        if ctx.invoked_subcommand is None:
            typer.secho(
                f"`actuate {group}` is not implemented. {spec}\n"
                "Increment 1 builds the foundation only (schema, io, catalog, infra).",
                fg="yellow",
            )
            raise typer.Exit(1)

    return sub


app.add_typer(_todo("ingest", "L0 — Master Spec §L0"), name="ingest")
app.add_typer(_todo("perceive", "L1 — Master Spec §L1; GPU"), name="perceive")
app.add_typer(_todo("fuse", "L2 — Master Spec §L2; net-new"), name="fuse")
app.add_typer(_todo("certify", "L4 — Master Spec §L4"), name="certify")
app.add_typer(_todo("retarget", "L5 — Master Spec §L5; net-new, GPU"), name="retarget")
app.add_typer(_todo("language", "L6 — Master Spec §L6"), name="language")


if __name__ == "__main__":
    app()
