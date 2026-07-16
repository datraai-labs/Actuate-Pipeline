"""`actuate deliver` -- presigned-URL delivery, doubly gated (Phase 5 Part F)."""

from __future__ import annotations

import re
from pathlib import Path

import typer


def _parse_expires(s: str) -> int:
    m = re.fullmatch(r"(\d+)([dhm])", s.strip())
    if not m:
        raise typer.BadParameter("use e.g. 7d, 12h, 30m")
    n, unit = int(m.group(1)), m.group(2)
    return n * {"d": 86400, "h": 3600, "m": 60}[unit]


def deliver_cmd(
    dataset: Path = typer.Argument(..., help="Packaged dataset directory (lerobot/rlds)."),
    customer: str = typer.Option(..., help="Customer name (becomes the key prefix)."),
    episodes: Path = typer.Option(..., help="Canonical episode JSON the dataset was "
                                            "packaged from (consent+quality are read "
                                            "from it)."),
    expires: str = typer.Option("7d", help="Presigned URL lifetime (e.g. 7d, 12h)."),
    local_root: Path = typer.Option(None, help="Use a LocalBackend rooted here instead "
                                    "of S3 (the delivery bucket is NOT deployed; this is "
                                    "the only verifiable path today)."),
) -> None:
    """Deliver a packaged dataset: consent gate + quality floor, through DeliveryWriter.

    Refuses rather than ships anything questionable. With no --local-root this targets
    S3, which does not exist yet -- it will fail loudly, which is the honest outcome.
    """
    from actuate.io.backends import LocalBackend
    from actuate.io.consent import ConsentViolation
    from actuate.package.deliver import DeliveryRefused, deliver
    from actuate.schema import CanonicalEpisode

    ep = CanonicalEpisode.model_validate_json(episodes.read_text(encoding="utf-8"))
    backend = LocalBackend(local_root) if local_root else None
    try:
        rec = deliver(dataset, customer, [ep], backend=backend,
                      expires_s=_parse_expires(expires))
    except (DeliveryRefused, ConsentViolation) as exc:
        typer.secho(f"DELIVERY REFUSED\n\n{exc}", fg="red", bold=True)
        raise typer.Exit(1) from exc
    typer.secho(rec.summary(), fg="green", bold=True)
