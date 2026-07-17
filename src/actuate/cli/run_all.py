"""`actuate run all` -- the one-command full pipeline (Phase 5 Part F).

Config-driven (a YAML profile: rig, embodiment, stages, export formats, tier), with
checkpointing and resume: every stage writes its status to `<out>/checkpoint.json`, and a
re-run skips completed stages. Perception reuses the same sha256-keyed `.actuate_cache/`
spine as `actuate viz --cache`, so a crash after the 86-minute GPU stages never repeats
them.

### Honesty rules baked in

- A stage that cannot run says WHY (no trained arm model, no GPU stage cache, no API key)
  and the pipeline continues -- a skipped stage is reported, never papered over.
- The language stage defaults to REUSE: annotations already paid for on the same capture
  are attached by capture-id match. `mode: annotate` (billed) requires an explicit
  interactive confirmation -- this command never spends API money silently.
- Nothing here delivers. `actuate deliver` is a separate, doubly-gated command, and the
  delivery bucket does not exist anyway.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import typer

run_app = typer.Typer(help="One-command pipeline (Master Spec §2.2): "
                           "ingest -> perceive -> fuse -> canonical -> certify -> "
                           "retarget -> validate -> language -> package -> viz.")


@dataclass
class _Ctx:
    session: Path
    out: Path
    profile: dict
    checkpoint: dict = field(default_factory=dict)
    canonical_path: Path | None = None

    def save(self) -> None:
        (self.out / "checkpoint.json").write_text(
            json.dumps(self.checkpoint, indent=2), encoding="utf-8")

    def done(self, stage: str, note: str = "") -> None:
        self.checkpoint[stage] = {"status": "done", "note": note}
        self.save()
        typer.secho(f"  [{stage}] done  {note}", fg="green")

    def skip(self, stage: str, why: str) -> None:
        self.checkpoint[stage] = {"status": "skipped", "note": why}
        self.save()
        typer.secho(f"  [{stage}] SKIPPED: {why}", fg="yellow")

    def already(self, stage: str) -> bool:
        if self.checkpoint.get(stage, {}).get("status") == "done":
            typer.secho(f"  [{stage}] resume: already done", fg="cyan")
            return True
        return False


def _load_canonical(ctx: _Ctx):
    from actuate.schema import CanonicalEpisode

    return CanonicalEpisode.model_validate_json(
        ctx.canonical_path.read_text(encoding="utf-8"))


def _write_canonical(ctx: _Ctx, episode) -> None:
    ctx.canonical_path.write_text(episode.model_dump_json(indent=2), encoding="utf-8")


@run_app.command("all")
def run_all(
    config: Path = typer.Option(..., help="YAML profile (see profiles/)."),
    in_path: Path = typer.Option(..., "--in", help="processed/<session> directory."),
    out: Path = typer.Option(..., help="Output directory (checkpoint, canonical, exports)."),
    resume: bool = typer.Option(True, help="Skip stages already checkpointed as done."),
) -> None:
    """Run the whole pipeline from a processed session to packaged datasets + viz."""
    import yaml

    profile = yaml.safe_load(config.read_text(encoding="utf-8"))
    out.mkdir(parents=True, exist_ok=True)
    ctx = _Ctx(session=in_path, out=out, profile=profile)
    cp = out / "checkpoint.json"
    if resume and cp.exists():
        ctx.checkpoint = json.loads(cp.read_text(encoding="utf-8"))
    ctx.canonical_path = out / "canonical.json"

    t0 = time.time()
    _stage_ingest(ctx)
    perception = _stage_perceive(ctx)
    _stage_canonical(ctx, perception)
    _stage_label_actions(ctx)
    _stage_retarget(ctx)
    _stage_certify(ctx)
    _stage_language(ctx)
    _stage_package(ctx)
    _stage_viz(ctx, perception)

    typer.secho(f"\npipeline complete in {time.time() - t0:.0f}s — stage summary:",
                bold=True)
    for stage, rec in ctx.checkpoint.items():
        if stage.startswith("_") or not isinstance(rec, dict):
            continue                       # _capture_id / _tier / _sim are metadata, not stages
        color = {"done": "green", "skipped": "yellow"}.get(rec["status"], "red")
        typer.secho(f"  {stage:12s} {rec['status']:8s} {rec.get('note', '')}", fg=color)
    typer.echo(f"\ncanonical: {ctx.canonical_path}")
    typer.secho("nothing was delivered — `actuate deliver` is a separate, gated command "
                "(and the delivery bucket does not exist).", fg="yellow")


# ---------------------------------------------------------------- stages
def _stage_ingest(ctx: _Ctx) -> None:
    if ctx.already("ingest"):
        return
    from actuate.ingest import run_ingest

    res = run_ingest(ctx.profile.get("rig", "head_mounted"), ctx.session,
                     aligned_robot=ctx.profile.get("aligned_robot"))
    ctx.checkpoint["_capture_id"] = res.capture_id
    ctx.checkpoint["_tier"] = res.tier.value
    for f in res.flags:
        typer.secho(f"  [ingest] FLAG: {f}", fg="yellow")
    ctx.done("ingest", f"capture {res.capture_id[:12]}… tier {res.tier.value}")


def _stage_perceive(ctx: _Ctx) -> dict:
    """GPU stages via the same cache spine as `actuate viz --cache`."""
    stage = "perceive"
    cfg = ctx.profile.get("perception", {})
    if not cfg.get("enabled", True):
        ctx.skip(stage, "disabled in profile")
        return {}
    n = int(cfg.get("max_frames", 45))
    from actuate.cli.viz import _stage_cached

    results: dict = {}
    try:
        from actuate.perception import depth as depthmod
        from actuate.perception import hands as handsmod

        results["hands"], src_h = _stage_cached(
            ctx.session, "hands", f"n={n}", use_cache=True, force=False,
            run_fn=lambda: handsmod.run(ctx.session, max_frames=n, prefilter=False))
        results["depth"], src_d = _stage_cached(
            ctx.session, "depth", f"n={n}", use_cache=True, force=False,
            run_fn=lambda: depthmod.run(ctx.session, max_frames=n))
        try:
            from actuate.perception.slam import runner as slam_runner

            results["slam"], _ = _stage_cached(
                ctx.session, "slam", f"n={n}", use_cache=True, force=False,
                run_fn=lambda: slam_runner.run(ctx.session, max_frames=n))
        except Exception as exc:
            typer.secho(f"  [perceive] slam unavailable: {exc}", fg="yellow")
            results["slam"] = None
        prompts = cfg.get("prompts", ["stapler", "paper"])
        try:
            from actuate.perception import objects as objmod

            results["objects"], _ = _stage_cached(
                ctx.session, "objects", f"n={n}|{','.join(prompts)}",
                use_cache=True, force=False,
                run_fn=lambda: objmod.run(ctx.session, prompts=prompts,
                                          max_frames=n, chunk=n))
        except Exception as exc:
            typer.secho(f"  [perceive] objects unavailable: {exc}", fg="yellow")
            results["objects"] = None

        from actuate import fusion as fusionmod
        from actuate.config import RigType

        results["fusion"] = fusionmod.run(
            results["hands"], rig=RigType(ctx.profile.get("rig", "head_mounted")),
            objects=results["objects"])
        ctx.done(stage, f"{n} frames (hands {src_h}, depth {src_d})")
    except Exception as exc:
        ctx.skip(stage, f"GPU stages failed: {type(exc).__name__}: {exc}")
        return {}
    return results


def _stage_canonical(ctx: _Ctx, perception: dict) -> None:
    stage = "canonical"
    if ctx.already(stage) and ctx.canonical_path.exists():
        return
    # profile capture_id (the RAW-bytes hash, the canonical identity) wins over the
    # ingest hash, which for a processed session can only see the compressed payload
    capture_id = ctx.profile.get("capture_id") or ctx.checkpoint.get("_capture_id", "0" * 64)
    task = ctx.profile.get("task")
    if perception.get("hands") is not None:
        from actuate.canonical.from_perception import build_from_perception
        from actuate.config import RigType, Tier

        ep = build_from_perception(
            ctx.session, capture_id,
            hands=perception["hands"], depth=perception.get("depth"),
            fusion=perception.get("fusion"), slam=perception.get("slam"),
            objects=perception.get("objects"),
            rig=RigType(ctx.profile.get("rig", "head_mounted")), task=task)
        note = "from perception"
    else:
        from actuate.canonical import build_episode
        from actuate.config import Tier

        ep = build_episode(ctx.session, capture_id, task=task)
        note = "from v1-legacy outputs (perception unavailable)"
    from actuate.config import Tier

    tier = ctx.checkpoint.get("_tier")
    if tier:
        ep = ep.model_copy(update={"tier": Tier(tier)})
    _write_canonical(ctx, ep)
    ctx.done(stage, f"{note}, {len(ep.frames)} frames")


def _stage_label_actions(ctx: _Ctx) -> None:
    stage = "label_actions"
    if ctx.already(stage):
        return
    from actuate.language import label_actions

    ep = _load_canonical(ctx)
    fps = 30.0
    meta = ctx.session / "session_meta.json"
    if meta.exists():
        fps = float(json.loads(meta.read_text(encoding="utf-8")).get("fps_nominal", 30.0))
    res = label_actions(ep, fps=fps)          # geometry only, $0
    _write_canonical(ctx, res.episode)
    ctx.done(stage, f"{len(res.intervals)} intervals over {sorted(res.actors)}, "
                    f"{res.flagged_for_review} flagged")


def _stage_retarget(ctx: _Ctx) -> None:
    stage = "retarget"
    if ctx.already(stage):
        return
    cfg = ctx.profile.get("retarget", {})
    model = cfg.get("arm_model")
    if not model or not Path(model).exists():
        ctx.skip(stage, f"no trained arm estimator at {model!r} — train with "
                        "`actuate retarget train-arm` (Kaggle GPU)")
        return
    try:
        from actuate.retarget import arm as armmod
        from actuate.retarget import sim_validate as simval

        ep = _load_canonical(ctx)
        emb = ctx.profile.get("embodiment", "franka_panda")
        result = armmod.run(ep, emb, Path(model))
        ep = armmod.attach_to_episode(ep, result)
        sim = simval.run(ep, emb, result)
        ctx.checkpoint["_sim"] = {"eligible": sim.eligible,
                                  "ik": sim.ik_convergence_rate,
                                  "reasons": sim.reasons}
        _write_canonical(ctx, ep)
        ctx.done(stage, result.summary())
    except Exception as exc:
        ctx.skip(stage, f"{type(exc).__name__}: {exc}")


def _stage_certify(ctx: _Ctx) -> None:
    stage = "certify"
    if ctx.already(stage):
        return
    from actuate.certify import score

    ep = _load_canonical(ctx)
    sim_rec = ctx.checkpoint.get("_sim")
    sim = None
    if sim_rec:
        from types import SimpleNamespace

        sim = SimpleNamespace(eligible=sim_rec["eligible"],
                              ik_convergence_rate=sim_rec["ik"],
                              reasons=sim_rec["reasons"])
    report = score(ep, ctx.profile.get("embodiment"), session_dir=ctx.session,
                   sim_result=sim)
    _write_canonical(ctx, report.episode)
    ctx.done(stage, f"quality {report.quality}/5, {len(report.mistakes)} mistake flags")


def _stage_language(ctx: _Ctx) -> None:
    stage = "language"
    if ctx.already(stage):
        return
    cfg = ctx.profile.get("language", {})
    mode = cfg.get("mode", "reuse")
    ep = _load_canonical(ctx)

    if mode == "skip":
        ctx.skip(stage, "disabled in profile")
        return

    if mode == "reuse":
        src = Path(cfg.get("reuse_from", ""))
        if not src.exists():
            ctx.skip(stage, f"reuse source {src} missing; run `actuate language annotate` "
                            "once and point language.reuse_from at it")
            return
        from actuate.schema import CanonicalEpisode

        donor = CanonicalEpisode.model_validate_json(src.read_text(encoding="utf-8"))
        if donor.capture_id != ep.capture_id or not donor.task_paraphrases:
            ctx.skip(stage, "reuse source is a different capture or unannotated — "
                            "NOT attaching another episode's language")
            return
        ep = ep.model_copy(update={
            "task": ep.task or donor.task,
            "task_paraphrases": donor.task_paraphrases,
            "subtasks": donor.subtasks,
            "subgoal_frames": donor.subgoal_frames})
        _write_canonical(ctx, ep)
        ctx.done(stage, f"REUSED paid annotations ({len(donor.task_paraphrases)} "
                        f"paraphrases, {len(donor.subtasks)} subtasks) — $0 new API cost")
        return

    # mode == "annotate": billed — never silent
    from actuate.language import estimate_annotation_cost, get_api_key

    if get_api_key() is None:
        ctx.skip(stage, "no ANTHROPIC_API_KEY; language degrades gracefully")
        return
    est = estimate_annotation_cost(9)
    if not typer.confirm(f"language.mode=annotate will call the Anthropic API "
                         f"(~${est:.2f}). Proceed?"):
        ctx.skip(stage, "billed annotation declined at prompt")
        return
    from actuate.language import annotate as _annotate

    video = ctx.session / ctx.profile.get("video", "redacted_compressed.mp4")
    report = _annotate(ep, session_dir=ctx.session, video=video)
    if report.skipped:
        ctx.skip(stage, report.skip_reason)
        return
    _write_canonical(ctx, report.episode)
    ctx.done(stage, f"${report.cost_usd:.4f}, {report.flagged_for_review} flagged")


def _stage_package(ctx: _Ctx) -> None:
    stage = "package"
    if ctx.already(stage):
        return
    ep = _load_canonical(ctx)
    if not ep.task:
        ctx.skip(stage, "episode has no task; both exporters fail closed on that")
        return
    cfg = ctx.profile.get("export", {})
    video = ctx.session / ctx.profile.get("video", "redacted_compressed.mp4")
    emb = ctx.profile.get("embodiment")
    emb = emb if emb and emb in ep.action_robot else None
    notes = []
    if "lerobot" in cfg.get("formats", ["lerobot"]):
        from actuate.package import export_lerobot_v3

        res = export_lerobot_v3(ep, ctx.out / "lerobot", overwrite=True, video=video,
                                embodiment=emb, tier=cfg.get("tier", "all"))
        notes.append(f"lerobot {res.n_frames}f")
    if "rlds" in cfg.get("formats", []):
        from actuate.package.rlds_export import export_rlds

        res = export_rlds(ep, ctx.out / "rlds", embodiment=emb, video=video,
                          tier=cfg.get("tier", "all"))
        notes.append(f"rlds {res.n_steps}steps")
    ctx.done(stage, ", ".join(notes) + ("" if emb else " (human-space only: no robot "
                                        "action attached)"))


def _stage_viz(ctx: _Ctx, perception: dict) -> None:
    stage = "viz"
    if ctx.already(stage):
        return
    if not ctx.profile.get("viz", True):
        ctx.skip(stage, "disabled in profile")
        return
    try:
        import rerun as rr

        from actuate.viz import rerun_log

        ep = _load_canonical(ctx)
        rrd = ctx.out / "pipeline.rrd"
        rr.init("actuate-run-all")
        rr.save(str(rrd))
        # RGB context even when perception was skipped: read a bounded frame window
        video_frames = None
        video = ctx.session / ctx.profile.get("video", "redacted_compressed.mp4")
        if video.exists():
            import cv2

            cap = cv2.VideoCapture(str(video))
            video_frames = []
            for _ in range(min(300, len(ep.frames) or 300)):
                ok, f = cap.read()
                if not ok:
                    break
                video_frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
            cap.release()
        counts = rerun_log.log_episode(
            ep,
            video_frames=video_frames,
            hands=perception.get("hands"), depth=perception.get("depth"),
            objects=perception.get("objects"), fusion=perception.get("fusion"),
            slam=perception.get("slam"))
        ctx.done(stage, f"{rrd.name} ({counts})")
    except Exception as exc:
        ctx.skip(stage, f"{type(exc).__name__}: {exc}")
