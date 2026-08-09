"""`actuate migrate` -- move the local v1 corpus into S3 + the Postgres catalog.

The interesting part is not the upload. It is **capture identity**.

The local corpus contains the same 95-second recording under multiple session ids with
*conflicting consent records*, plus sessions whose raw video and processed artifacts come
from different recordings. Uploading those as independent sessions would carry both bugs
into the catalog permanently.

So migration is content-addressed (`actuate.ingest.content_address`): the capture id IS the
SHA-256 of the raw bytes. Identical footage computes an identical id and collapses to one
capture, one consent decision, one row. Legacy metadata is cross-checked against the bytes
it claims to describe, and a session whose claims contradict its payload is refused.

Consent is reconciled least-permissive-wins, and every observation is written to an
append-only event log that distinguishes a *duplicate-upload conflict* from a *withdrawal
of consent*. They resolve the same safe way; they are not the same event.

`plan` reads only. `apply` requires --yes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import typer

from actuate.config import (
    Bucket,
    ConsentStatus,
    Env,
    PiiStatus,
    RigType,
    StorageBackendKind,
    load_settings,
)
from actuate.ingest import CaptureManifest, build_manifest, check_legacy_claims

migrate_app = typer.Typer(help="Migrate the local v1 corpus into S3 + the catalog.")

_RAW_VIDEO = "raw.mp4"


@dataclass
class SessionClaim:
    """One session directory's claim about a capture."""

    session_id: str
    consent: ConsentStatus | None


@dataclass
class PlannedCapture:
    manifest: CaptureManifest
    video: Path
    claims: list[SessionClaim] = field(default_factory=list)

    @property
    def observed_consent(self) -> list[ConsentStatus]:
        return [c.consent for c in self.claims if c.consent is not None]

    @property
    def resolved_consent(self) -> ConsentStatus | None:
        obs = self.observed_consent
        if not obs:
            return None
        order = [
            ConsentStatus.DENIED,
            ConsentStatus.REVOKED,
            ConsentStatus.PENDING,
            ConsentStatus.GRANTED,
        ]
        return min(obs, key=order.index)

    @property
    def conflicted(self) -> bool:
        return len(set(self.observed_consent)) > 1

    @property
    def deliverable(self) -> bool:
        return self.resolved_consent is ConsentStatus.GRANTED


@dataclass
class Plan:
    captures: list[PlannedCapture] = field(default_factory=list)
    refused: list[tuple[str, str]] = field(default_factory=list)


def _read_consent(session_dir: Path) -> ConsentStatus | None:
    p = session_dir / "consent.json"
    if not p.exists():
        return None
    try:
        return ConsentStatus(json.loads(p.read_text()).get("status"))
    except (ValueError, json.JSONDecodeError):
        return None


def build_plan(raw: Path, processed: Path) -> Plan:
    """Hash every raw video, group by content, cross-check the legacy metadata. Reads only."""
    plan = Plan()
    by_hash: dict[str, PlannedCapture] = {}

    if not raw.is_dir():
        plan.refused.append((str(raw), "raw directory does not exist; nothing to migrate"))
        return plan

    for session_dir in sorted(d for d in raw.iterdir() if d.is_dir()):
        sid = session_dir.name
        video = session_dir / _RAW_VIDEO
        if not video.exists():
            plan.refused.append((sid, f"no {_RAW_VIDEO}"))
            continue

        proc = processed / sid
        meta_p = proc / "session_meta.json"
        if not meta_p.exists():
            plan.refused.append((sid, "no session_meta.json (never ingested)"))
            continue
        meta = json.loads(meta_p.read_text())

        hp = proc / "hand_pose_3d.json"
        if not hp.exists():
            plan.refused.append((sid, "no hand_pose_3d.json (incomplete run, no metric 3D)"))
            continue
        n_frames = len(json.loads(hp.read_text()))

        # rig_type was never captured by v1 ingestion. Inferred, and the inference is
        # recorded rather than presented as fact.
        rig = (
            RigType.GLOVE
            if meta.get("glove_type", "none") not in (None, "none")
            else RigType.UMI_GRIPPER
            if meta.get("imu_source_mode") == "wrist_mounted"
            else RigType.HEAD_MOUNTED
        )

        manifest = build_manifest(
            video,
            rig,
            frame_count=meta.get("frame_count"),
            duration_sec=meta.get("duration_seconds"),
            fps=meta.get("fps_nominal"),
        )

        # THE check content-addressing makes possible: do the bytes match the claims?
        problems = check_legacy_claims(manifest, meta, n_frames)
        if problems:
            plan.refused.append((sid, "; ".join(problems)))
            continue

        cap = by_hash.get(manifest.content_hash)
        if cap is None:
            cap = PlannedCapture(manifest=manifest, video=video)
            by_hash[manifest.content_hash] = cap
            plan.captures.append(cap)
        cap.claims.append(SessionClaim(session_id=sid, consent=_read_consent(session_dir)))

    return plan


def _render(plan: Plan, settings) -> None:
    store_raw = f"s3://{settings.bucket(Bucket.RAW)}"

    typer.secho(
        f"\n{len(plan.captures)} distinct capture(s) from "
        f"{sum(len(c.claims) for c in plan.captures)} session dir(s)\n",
        bold=True,
    )

    for cap in plan.captures:
        m = cap.manifest
        typer.secho(f"capture {m.capture_id[:16]}...  ({m.size_bytes / 1e6:.1f} MB)", bold=True)
        typer.echo(f"  sessions   : {', '.join(c.session_id for c in cap.claims)}")
        if len(cap.claims) > 1:
            typer.secho(
                f"  dedup      : {len(cap.claims)} session dirs hold identical bytes "
                "-> ONE capture (content-addressed)",
                fg="cyan",
            )
        typer.echo(f"  rig        : {m.rig.value}  (INFERRED - v1 never recorded rig type)")
        typer.echo(f"  frames     : {m.frame_count}   duration: {m.duration_sec}s")

        if cap.conflicted:
            typer.secho(
                f"  consent    : CONFLICT {sorted({c.value for c in cap.observed_consent})} "
                f"-> resolved LEAST-PERMISSIVE '{cap.resolved_consent.value}'",
                fg="yellow",
                bold=True,
            )
            typer.secho(
                "               logged as DUPLICATE_CONFLICT events (not a revocation)",
                fg="yellow",
            )
        elif cap.resolved_consent is None:
            typer.secho("  consent    : NO RECORD -> blocked", fg="red")
        else:
            typer.echo(f"  consent    : {cap.resolved_consent.value}")

        typer.secho(
            f"  DELIVERABLE: {'YES' if cap.deliverable else 'NO'}",
            fg="green" if cap.deliverable else "red",
            bold=True,
        )

        typer.echo("\n  WILL WRITE TO S3:")
        typer.echo(f"    {store_raw}/{m.prefix}/{_RAW_VIDEO}          ({m.size_bytes / 1e6:.1f} MB)")
        typer.echo(f"    {store_raw}/{m.prefix}/manifest.json         (binds metadata to the hash)")

        typer.echo("\n  WILL WRITE TO POSTGRES:")
        typer.echo(f"    captures        1 row  id={m.capture_id[:16]}... (= content hash)")
        typer.echo(
            f"    consent         1 row  status={cap.resolved_consent.value if cap.resolved_consent else 'NONE'}"
        )
        n_ev = (len(cap.observed_consent) + 1) if cap.conflicted else 1
        typer.echo(f"    consent_events  {n_ev} row(s)  append-only audit log")
        typer.echo("")

    if plan.refused:
        typer.secho("REFUSED (nothing uploaded, nothing registered):", fg="red", bold=True)
        for sid, why in plan.refused:
            typer.secho(f"  {sid}", fg="red")
            typer.secho(f"      {why}", fg="red")
        typer.echo("")


@migrate_app.command("plan")
def plan_cmd(
    raw: Path = typer.Option(Path("raw")),
    processed: Path = typer.Option(Path("processed")),
    env: Env = typer.Option(Env.DEV),
) -> None:
    """Show exactly what apply would write. Reads only; touches nothing."""
    settings = load_settings(env=env)
    _render(build_plan(raw, processed), settings)


@migrate_app.command("apply")
def apply_cmd(
    raw: Path = typer.Option(Path("raw")),
    processed: Path = typer.Option(Path("processed")),
    env: Env = typer.Option(Env.DEV),
    backend: StorageBackendKind = typer.Option(StorageBackendKind.S3),
    profile: str = typer.Option("datraai-admin"),
    yes: bool = typer.Option(False, "--yes", help="Required. Without it, this is a dry run."),
) -> None:
    """Upload the raw captures and register them in the catalog.

    Nothing local is deleted. Re-running is safe: content-addressed keys mean an identical
    capture lands in the identical place, so a second run is a no-op rather than a
    duplicate.
    """
    overrides = {"aws_profile": profile} if backend is StorageBackendKind.S3 else {}
    settings = load_settings(env=env, storage_backend=backend, **overrides)

    plan = build_plan(raw, processed)
    _render(plan, settings)

    if not yes:
        typer.secho("DRY RUN. Re-run with --yes to execute.", fg="yellow", bold=True)
        raise typer.Exit(0)

    try:
        from actuate.catalog import (
            Capture,
            Certification,
            Episode,
            make_engine,
            make_session_factory,
            reconcile_consent,
            session_scope,
        )
        from actuate.io import get_backend
        from actuate.schema import SCHEMA_VERSION
    except ImportError as exc:
        typer.secho(
            "migration apply needs the storage/catalog dependencies; install `actuate[aws]` "
            f"({exc})",
            fg="red",
        )
        raise typer.Exit(1) from exc

    if backend is StorageBackendKind.S3:
        import boto3

        ident = (
            boto3.Session(profile_name=settings.aws_profile, region_name=settings.aws_region)
            .client("sts")
            .get_caller_identity()
        )
        typer.echo(f"target account {ident['Account']} / {settings.aws_region}\n")

    store = get_backend(settings)
    factory = make_session_factory(make_engine(settings))

    for cap in plan.captures:
        m = cap.manifest
        raw_key = f"{m.prefix}/{_RAW_VIDEO}"

        if store.exists(Bucket.RAW, raw_key):
            typer.secho(f"already present (dedup hit): {m.capture_id[:16]}...", fg="cyan")
        else:
            # Re-verify before upload. The manifest was built from these bytes moments ago,
            # but the check costs nothing and the failure it guards against is permanent.
            m.verify(cap.video)
            store.put_file(Bucket.RAW, raw_key, cap.video)
            typer.secho(f"uploaded {m.size_bytes / 1e6:.1f} MB -> {store.uri(Bucket.RAW, raw_key)}", fg="green")

        store.put_bytes(
            Bucket.RAW,
            f"{m.prefix}/manifest.json",
            json.dumps(m.model_dump(mode="json"), indent=2, sort_keys=True).encode(),
        )

        with session_scope(factory) as s:
            if s.get(Capture, m.capture_id) is None:
                s.add(
                    Capture(
                        id=m.capture_id,
                        content_hash=m.content_hash,
                        size_bytes=m.size_bytes,
                        rig_type=m.rig,
                        raw_uri=store.uri(Bucket.RAW, raw_key),
                        frame_count=m.frame_count,
                        duration_sec=m.duration_sec,
                    )
                )
                s.flush()

            decision = reconcile_consent(
                s,
                m.capture_id,
                cap.observed_consent,
                sources=[c.session_id for c in cap.claims if c.consent is not None],
            )

            # One episode per capture for now. Episode segmentation is L3 (Increment 2 Part C).
            eid = f"{m.capture_id[:16]}_ep00"
            if s.get(Episode, eid) is None:
                s.add(
                    Episode(
                        id=eid,
                        capture_id=m.capture_id,
                        rig_type=m.rig,
                        schema_version=SCHEMA_VERSION,
                        frame_count=m.frame_count,
                    )
                )
                s.flush()
                # PII redaction has not been re-run under this pipeline, so pii_status is
                # PENDING -- not PASSED. Fail-closed: the delivery gate needs both.
                s.add(Certification(episode_id=eid, pii_status=PiiStatus.PENDING))

        typer.secho(
            f"  catalog: consent={decision.value}  "
            f"deliverable={'YES' if decision is ConsentStatus.GRANTED else 'NO'}",
            fg="green" if decision is ConsentStatus.GRANTED else "yellow",
        )

    typer.secho(
        f"\n{len(plan.captures)} capture(s) migrated, {len(plan.refused)} refused. "
        "Nothing deleted locally.",
        fg="green",
        bold=True,
    )
