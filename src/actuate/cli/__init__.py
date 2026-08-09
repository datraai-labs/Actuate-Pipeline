"""The `actuate` Typer CLI -- Master Spec section 2.2.

**Thin.** Every command is a wrapper over a library call. Nothing here contains logic that
cannot be reached by importing `actuate` -- that is the CLI/API-first rule, and the
import-linter contract in `.importlinter` enforces the direction (no layer may import
`actuate.cli`).

The supported customer path is ``status -> process -> report -> export -> deliver``.
Layer commands remain for operators; only the standalone ``perceive`` and ``fuse``
commands are placeholders because those stages run through ``actuate process``.
"""

from __future__ import annotations

import typer

from actuate.cli.certify_cmd import certify_app
from actuate.cli.deliver_cmd import deliver_cmd
from actuate.cli.ingest_cmd import ingest_app
from actuate.cli.language_cmd import language_app
from actuate.cli.migrate import migrate_app
from actuate.cli.pipeline import canonical_app, package_app
from actuate.cli.retarget import retarget_app
from actuate.cli.run_all import run_app
from actuate.cli.schema_cmd import schema_app
from actuate.cli.simple import (
    config_app,
    export_cmd,
    login_app,
    process_cmd,
    report_cmd,
    status_cmd,
)
from actuate.cli.storage import storage_app
from actuate.cli.viz import viz_app

app = typer.Typer(
    help="Actuate - multimodal capture to certified robot-learning data.",
    no_args_is_help=True,
)

app.add_typer(schema_app, name="schema")
app.add_typer(storage_app, name="storage")
app.add_typer(migrate_app, name="migrate")
app.add_typer(canonical_app, name="canonical")
app.add_typer(package_app, name="package")
app.add_typer(viz_app, name="viz")


def _todo(group: str, spec: str) -> typer.Typer:
    """A command group that parses and honestly says it does nothing.

    Registering the full surface now means the shape of the CLI is reviewable before the
    layers land, and `--help` never advertises a capability that does not exist.
    """
    sub = typer.Typer(help=f"[STANDALONE COMMAND NOT IMPLEMENTED - {spec}]", no_args_is_help=True)

    @sub.callback(invoke_without_command=True)
    def _cb(ctx: typer.Context) -> None:
        if ctx.invoked_subcommand is None:
            typer.secho(
                f"`actuate {group}` is not a standalone command. {spec}\n"
                "Use `actuate process <source>`; it runs perception and fusion in the live pipeline.",
                fg="yellow",
            )
            raise typer.Exit(1)

    return sub


app.add_typer(ingest_app, name="ingest")
app.add_typer(_todo("perceive", "L1 -- Master Spec section L1; GPU"), name="perceive")
app.add_typer(_todo("fuse", "L2 -- Master Spec section L2; net-new"), name="fuse")
app.add_typer(certify_app, name="certify")
app.add_typer(retarget_app, name="retarget")
app.add_typer(language_app, name="language")
app.add_typer(run_app, name="run")
app.command("deliver")(deliver_cmd)

# --- Phase 6: the simplified, SDK-backed surface (the common case) --------------------
app.add_typer(login_app, name="login")
app.add_typer(config_app, name="config")
app.command("process")(process_cmd)
app.command("export")(export_cmd)
app.command("status")(status_cmd)
app.command("report")(report_cmd)


if __name__ == "__main__":
    app()
