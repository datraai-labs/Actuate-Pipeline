"""The full pipeline orchestration, as a LIBRARY function (Phase 6 refactor).

Lifted out of `cli.run_all` so the SDK, the CLI, and (later) the service can all drive the
same stages without inverting the import-linter contract. The CLI is now a thin caller that
passes a typer-printing `reporter`; the SDK passes its own. Behaviour is byte-for-byte the
same as the old `actuate run all` -- the integration gate proves it.

Stages: ingest -> perceive -> canonical -> label_actions -> retarget -> certify -> language
-> package -> viz. Every stage checkpoints to `<out>/checkpoint.json`; a re-run with
`resume=True` skips completed stages. A stage that cannot run reports WHY and the pipeline
continues -- a skip is surfaced, never papered over.

No typer, no prints of its own: progress goes through `reporter(stage, status, note)`, and
any money-spending confirmation goes through `confirm(prompt) -> bool` (default: deny, so a
library call never spends API money without an explicit confirmer).
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from actuate.pipeline.cache import stage_cached

#: reporter(stage, status, note): status in {"done","skipped","flag","info"}.
Reporter = Callable[[str, str, str], None]
Confirm = Callable[[str], bool]


def _noop_reporter(stage: str, status: str, note: str) -> None:  # pragma: no cover
    pass


def _deny(prompt: str) -> bool:  # pragma: no cover - default: never spend without a confirmer
    return False


@dataclass
class PipelineResult:
    out: Path
    canonical_path: Path
    checkpoint: dict
    seconds: float

    def stage(self, name: str) -> dict:
        return self.checkpoint.get(name, {})


@dataclass
class _Ctx:
    session: Path
    out: Path
    profile: dict
    reporter: Reporter
    confirm: Confirm
    checkpoint: dict = field(default_factory=dict)
    canonical_path: Path | None = None

    def save(self) -> None:
        (self.out / "checkpoint.json").write_text(
            json.dumps(self.checkpoint, indent=2), encoding="utf-8")

    def done(self, stage: str, note: str = "") -> None:
        self.checkpoint[stage] = {"status": "done", "note": note}
        self.save()
        self.reporter(stage, "done", note)

    def skip(self, stage: str, why: str) -> None:
        self.checkpoint[stage] = {"status": "skipped", "note": why}
        self.save()
        self.reporter(stage, "skipped", why)

    def flag(self, stage: str, msg: str) -> None:
        self.reporter(stage, "flag", msg)

    def already(self, stage: str) -> bool:
        if self.checkpoint.get(stage, {}).get("status") == "done":
            self.reporter(stage, "info", "resume: already done")
            return True
        return False


def _load_canonical(ctx: _Ctx):
    from actuate.schema import CanonicalEpisode

    return CanonicalEpisode.model_validate_json(
        ctx.canonical_path.read_text(encoding="utf-8"))


def _write_canonical(ctx: _Ctx, episode) -> None:
    ctx.canonical_path.write_text(episode.model_dump_json(indent=2), encoding="utf-8")


def run_pipeline(session: Path, out: Path, profile: dict, *, resume: bool = True,
                 reporter: Reporter | None = None,
                 confirm: Confirm | None = None) -> PipelineResult:
    """Run the whole pipeline from a processed session to packaged datasets + viz."""
    session, out = Path(session), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    ctx = _Ctx(session=session, out=out, profile=profile,
               reporter=reporter or _noop_reporter, confirm=confirm or _deny)
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
    return PipelineResult(out=out, canonical_path=ctx.canonical_path,
                          checkpoint=ctx.checkpoint, seconds=time.time() - t0)


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
        ctx.flag("ingest", f)
    imu_note = (
        f", IMU {res.imu_samples} samples -> {res.imu_frames} frames"
        if res.imu_sync_path is not None
        else ""
    )
    ctx.done(
        "ingest",
        f"capture {res.capture_id[:12]}… tier {res.tier.value}{imu_note}",
    )


def _stage_perceive(ctx: _Ctx) -> dict:
    """GPU stages via the shared sha256-keyed cache spine."""
    stage = "perceive"
    cfg = ctx.profile.get("perception", {})
    if not cfg.get("enabled", True):
        ctx.skip(stage, "disabled in profile")
        return {}
    n = int(cfg.get("max_frames", 45))
    warn = lambda m: ctx.flag(stage, m)  # noqa: E731

    results: dict = {}
    try:
        from actuate.perception import depth as depthmod
        from actuate.perception import hands as handsmod

        results["hands"], src_h = stage_cached(
            ctx.session, "hands", f"n={n}", use_cache=True, force=False, warn=warn,
            run_fn=lambda: handsmod.run(ctx.session, max_frames=n, prefilter=False))
        results["depth"], src_d = stage_cached(
            ctx.session, "depth", f"n={n}", use_cache=True, force=False, warn=warn,
            run_fn=lambda: depthmod.run(ctx.session, max_frames=n))
        try:
            from actuate.perception.slam import runner as slam_runner

            # SLAM output changes when a raw IMU sidecar is synchronized after an earlier
            # vision-only run. Include the aligned stream identity so that cache cannot
            # silently return the old modality result.
            imu_h5 = ctx.session / "session.h5"
            if imu_h5.exists():
                import hashlib

                h = hashlib.sha256()
                with imu_h5.open("rb") as fh:
                    for chunk in iter(lambda: fh.read(1 << 20), b""):
                        h.update(chunk)
                imu_key = h.hexdigest()[:12]
            else:
                imu_key = "none"
            results["slam"], _ = stage_cached(
                ctx.session, "slam", f"v2|n={n}|imu={imu_key}",
                use_cache=True, force=False, warn=warn,
                run_fn=lambda: slam_runner.run(ctx.session, max_frames=n))
        except Exception as exc:
            ctx.flag(stage, f"slam unavailable: {exc}")
            results["slam"] = None
        prompts = cfg.get("prompts", ["stapler", "paper"])
        try:
            from actuate.perception import objects as objmod

            results["objects"], _ = stage_cached(
                ctx.session, "objects", f"n={n}|{','.join(prompts)}",
                use_cache=True, force=False, warn=warn,
                run_fn=lambda: objmod.run(ctx.session, prompts=prompts,
                                          max_frames=n, chunk=n))
        except Exception as exc:
            ctx.flag(stage, f"objects unavailable: {exc}")
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
    capture_id = ctx.profile.get("capture_id") or ctx.checkpoint.get("_capture_id", "0" * 64)
    task = ctx.profile.get("task")
    if perception.get("hands") is not None:
        from actuate.canonical.from_perception import build_from_perception
        from actuate.config import RigType

        ep = build_from_perception(
            ctx.session, capture_id,
            hands=perception["hands"], depth=perception.get("depth"),
            fusion=perception.get("fusion"), slam=perception.get("slam"),
            objects=perception.get("objects"),
            rig=RigType(ctx.profile.get("rig", "head_mounted")), task=task)
        note = "from perception"
    elif (ctx.session / "hand_pose_3d.json").exists():
        # v1-legacy fallback ONLY when the session actually carries v1 artifacts (the bundled
        # demo). A fresh source has none, so never guess this path for it.
        from actuate.canonical import build_episode

        ep = build_episode(ctx.session, capture_id, task=task)
        note = "from v1-legacy outputs (perception unavailable)"
    else:
        # Perception failed AND there's nothing precomputed to fall back to. Surface the
        # ACTUAL perceive failure (in the checkpoint), not a downstream metric-depth crash.
        why = ctx.checkpoint.get("perceive", {}).get("note", "perception did not run")
        raise RuntimeError(
            "cannot build a canonical episode: the perception stage did not produce hands, "
            f"and this source has no pre-computed outputs to fall back to.\n\n  reason: {why}\n\n"
            "Perception (WiLoR hands + UniDepth depth) must run for a fresh source. If it "
            "failed on an import/dependency error, fix that environment; if it ran out of "
            "GPU memory, retry with a smaller --max-frames.")
    from actuate.config import Tier

    updates = {}
    tier = ctx.checkpoint.get("_tier")
    if tier:
        updates["tier"] = Tier(tier)
    # local/self-hosted processing may mark consent GRANTED (your own data). pii_status is
    # deliberately NOT touched here -- it stays PENDING, so is_deliverable() and the
    # DeliveryWriter gate still block. The consent boundary is unchanged; this only spares a
    # developer a PENDING-consent block on data they own.
    consent = ctx.profile.get("consent")
    if consent:
        from actuate.config import ConsentStatus

        updates["consent"] = ConsentStatus(consent)
    # PII redaction is what legitimately earns pii_status=PASSED. Only when the profile asks
    # for it: run a real redaction pass (-> redacted_compressed.mp4, the filename export +
    # canonical already point at) and record PASSED. Without this flag pii_status stays
    # PENDING and the delivery gate keeps blocking -- the boundary is unchanged.
    if ctx.profile.get("redact_pii"):
        from actuate.io import redact

        report = redact.redact_session(ctx.session)
        updates["pii_status"] = report.status
        ctx.flag(stage, f"PII redaction: blurred {report.regions_blurred} region(s) over "
                        f"{report.frames_scanned} frames -> pii_status={report.status.value}")
    if updates:
        ep = ep.model_copy(update=updates)
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
    emb = ctx.profile.get("embodiment", "franka_panda")
    configured_model = cfg.get("arm_model")
    if configured_model:
        model = Path(configured_model).expanduser()
    else:
        # Resolve the default without importing the optional MuJoCo/Torch retarget stack.
        # A core-only install with no trained model should skip cleanly, as it did before
        # user-local model discovery was added.
        from actuate.config.auth import config_dir

        model = config_dir() / "models" / f"{emb}_root_frame.pt"
    if not model.exists():
        ctx.skip(
            stage,
            f"no trained arm estimator at {str(model)!r} — train it with "
            f"`actuate retarget train-arm --embodiment {emb}`",
        )
        return
    try:
        from actuate.retarget import arm as armmod
        from actuate.retarget import sim_validate as simval

        ep = _load_canonical(ctx)
        result = armmod.run(ep, emb, model)
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
    if not ctx.confirm(f"language.mode=annotate will call the Anthropic API (~${est:.2f}). "
                       "Proceed?"):
        ctx.skip(stage, "billed annotation declined")
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
