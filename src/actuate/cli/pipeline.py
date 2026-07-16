"""`actuate canonical build` and `actuate package lerobot` — Master Spec §2.2.

Thin wrappers over the library. Nothing here contains logic unreachable by importing
`actuate`.
"""

from __future__ import annotations

from pathlib import Path

import typer

from actuate.canonical import build_episode
from actuate.package import ExportRefused, export_lerobot_v3

canonical_app = typer.Typer(help="L3 — canonical build (Master Spec §L3).")
package_app = typer.Typer(help="L7 — packaging & delivery (Master Spec §L7).")


@canonical_app.command("build")
def build(
    processed: Path = typer.Argument(..., help="processed/<session> directory"),
    capture_hash: str = typer.Option(..., help="SHA-256 of the raw capture (= capture_id)"),
    task: str = typer.Option(
        None,
        help="Operator-verified task. NOT taken from v1's language grounding, which "
        "wraps a failed classification in a fluent template.",
    ),
    out: Path = typer.Option(None, help="Write the canonical episode JSON here."),
) -> None:
    """Build a CanonicalEpisode from v1's processed outputs."""
    ep = build_episode(processed, capture_hash, task=task)

    typer.secho(f"episode {ep.episode_id}  schema v{ep.schema_version}", bold=True)
    typer.echo(f"  rig        : {ep.rig.value}")
    typer.echo(f"  frames     : {len(ep.frames)}")
    typer.echo(f"  task       : {ep.task!r}")
    typer.secho(
        f"  deliverable: {'YES' if ep.is_deliverable else 'NO'}"
        f"  ({ep.delivery_block_reason() or 'clear'})",
        fg="green" if ep.is_deliverable else "yellow",
    )

    if ep.derivation_notes:
        typer.secho("\n  KNOWN GAPS (these travel with the data):", fg="yellow", bold=True)
        for k, v in ep.derivation_notes.items():
            typer.secho(f"    [{k}] {v}", fg="yellow")

    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(ep.model_dump_json(indent=2), encoding="utf-8")
        typer.secho(f"\nwrote {out}", fg="green")


@package_app.command("lerobot")
def lerobot(
    processed: Path = typer.Argument(..., help="processed/<session> directory"),
    capture_hash: str = typer.Option(...),
    video: Path = typer.Option(..., help="The REDACTED video. Never the original."),
    out: Path = typer.Option(Path(".export/lerobot")),
    task: str = typer.Option(None, help="Required. The exporter fail-closes without it."),
    repo_id: str = typer.Option("actuate/dev"),
    tier: str = typer.Option("all", help="Tier FILTER: stage1 | stage2 | all. Episodes are "
                             "kept by their own episode.tier (unassigned = stage1)."),
    embodiment: str = typer.Option(None, help="Dual-space: also ship "
                                   "action.robot.<embodiment> (needs L5 attached)."),
    transform: list[str] = typer.Option([], help="Export-time co-training transforms: "
                                        "masked_hand, eef_overlay. Need --fx/--fy/--cx/--cy."),
    fx: float = typer.Option(None), fy: float = typer.Option(None),
    cx: float = typer.Option(None), cy: float = typer.Option(None),
    fps: int = typer.Option(30),
    overwrite: bool = typer.Option(False),
) -> None:
    """Export to LeRobot v3, through LeRobot's own writer.

    Not 'done' because it ran — done when LeRobot's own loader reads it and a real
    training step runs. See tests/integration/test_lerobot_gate.py.
    """
    ep = build_episode(processed, capture_hash, task=task)
    intrinsics = (fx, fy, cx, cy) if None not in (fx, fy, cx, cy) else None
    try:
        res = export_lerobot_v3(
            ep, out, repo_id=repo_id, fps=fps, tier=tier, overwrite=overwrite, video=video,
            embodiment=embodiment, transforms=tuple(transform), intrinsics=intrinsics,
        )
    except ExportRefused as exc:
        typer.secho(f"EXPORT REFUSED\n\n{exc}", fg="red")
        raise typer.Exit(1) from exc

    typer.secho(f"exported -> {res.root}", fg="green", bold=True)
    typer.echo(f"  frames kept    : {res.n_frames}")
    typer.echo(f"  frames dropped : {res.n_dropped}  (no hand, or no successor)")
    typer.echo(f"  tiers          : {res.tier_counts}")
    if res.embodiment:
        typer.echo(f"  dual-space     : action.robot.{res.embodiment} shipped")

    if res.ego_contaminated:
        typer.secho(
            "\n  WARNING: the action in this dataset is EGO-CONTAMINATED.\n"
            "  No camera_pose (L1 SLAM not built), so on this moving-camera rig the wrist\n"
            "  delta is hand motion + head motion. It is NOT yet correct to train on.\n"
            "  The warning ships with the dataset in meta/actuate_provenance.json.",
            fg="yellow",
            bold=True,
        )
    if not ep.is_deliverable:
        typer.secho(
            f"\n  This episode is NOT DELIVERABLE ({ep.delivery_block_reason()}).\n"
            "  Export is legal — consent gates DELIVERY, not internal processing — but\n"
            "  nothing here may be shipped to a customer.",
            fg="yellow",
        )


@package_app.command("rlds")
def rlds(
    in_path: Path = typer.Option(..., "--in", help="Canonical episode JSON."),
    video: Path = typer.Option(..., help="The REDACTED video. Never the original."),
    out: Path = typer.Option(Path(".export/rlds"), help="tfds data_dir for the export."),
    name: str = typer.Option("actuate_dataset", help="tfds dataset name (identifier)."),
    tier: str = typer.Option("all", help="Tier FILTER: stage1 | stage2 | all."),
    embodiment: str = typer.Option(None, help="Dual-space: add action_robot_<embodiment>."),
) -> None:
    """Export to RLDS/Open-X, through tfds's own writer.

    Gate: `tfds.load(name, data_dir=out)` reads it back and one episode iterates with
    Open-X step keys. See tests/integration/test_rlds_gate.py.
    """
    from actuate.package.rlds_export import export_rlds
    from actuate.schema import CanonicalEpisode

    ep = CanonicalEpisode.model_validate_json(in_path.read_text(encoding="utf-8"))
    try:
        res = export_rlds(ep, out, name=name, tier=tier, embodiment=embodiment, video=video)
    except ExportRefused as exc:
        typer.secho(f"EXPORT REFUSED\n\n{exc}", fg="red")
        raise typer.Exit(1) from exc

    typer.secho(f"exported -> {res.root} (tfds name={res.name} v{res.version})",
                fg="green", bold=True)
    typer.echo(f"  episodes : {res.n_episodes}")
    typer.echo(f"  steps    : {res.n_steps}  (dropped {res.n_dropped}: no hand/successor)")
    typer.echo(f"  tiers    : {res.tier_counts}")
    if res.embodiment:
        typer.echo(f"  dual-space: action_robot_{res.embodiment} shipped")
    typer.echo(f'  load with: tfds.load("{res.name}", data_dir=r"{res.root}", split="train")')
