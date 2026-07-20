"""Actuate-dashboard bridge -- the file `ACTUATE_PATH` should point at.

The Actuate-dashboard worker (apps/api/worker/pipeline_runner.py) runs the real pipeline by
shelling out to:

    python <ACTUATE_PATH> --session <id> --video <path> --output <dir>
                          --imu-source <x> --depth-mode <y>

...then (1) drives its live PhaseTimeline by parsing stdout for
    "Running stage: <token>"   and   "<token> ✓ <secs>s"
where <token> is one of its nine PIPELINE_STAGES, and (2) reads each episode's
    <output>/<episode_id>/quality_certificate.json  (+ language_grounding.json, phase_segmentation.json)
into its Episode table.

This bridge speaks exactly that contract on top of the real `actuate` SDK. Two honest design
points:

  * The dashboard's nine stage tokens are finer on perception and coarser elsewhere than the
    real pipeline's stages (ingest, perceive, canonical, label_actions, retarget, certify,
    language, package, viz). STAGE_MAP below is the reconciliation; a real `perceive` lights
    up depth+hand+object at once, which is truthful (they run inside that one stage).
  * When WiLoR detects 0 hands (common on real egocentric footage), the pipeline yields 0
    frames. We DO NOT invent an episode. We still emit ONE certificate with
    needs_human_review=true and a low score, so the dashboard shows an honest "needs review"
    card instead of silently nothing -- the dashboard already has a NeedsReviewBadge for this.

Nothing here claims managed-cloud processing (that is not live); this runs the local SDK.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# The dashboard's nine tokens, in order (must match worker PIPELINE_STAGES).
DASH_STAGES = [
    "01_ingest", "02_preprocess", "03_depth_estimation", "04_hand_pose",
    "05_object_detection", "06_phase_segmentation", "07_episode_extraction",
    "08_task_labeling", "09_quality_certificate",
]

# Real reporter stage -> the dashboard tokens it satisfies. Real stages not listed
# (retarget, package, viz) have no timeline row; they are logged, and retarget feeds the
# episode's retargeting_eligible field instead.
STAGE_MAP: dict[str, list[str]] = {
    "ingest": ["01_ingest", "02_preprocess"],
    "perceive": ["03_depth_estimation", "04_hand_pose", "05_object_detection"],
    "language": ["06_phase_segmentation"],
    "canonical": ["07_episode_extraction"],
    "label_actions": ["08_task_labeling"],
    "certify": ["09_quality_certificate"],
}


def _emit_start(token: str) -> None:
    print(f"Running stage: {token}", flush=True)


def _emit_done(token: str, secs: float) -> None:
    # the dashboard matches on the token + a "✓"/"Completed"/"completed" marker on the same
    # line. Use the ASCII word (not U+2713) so this never dies on a cp1252 Windows console.
    print(f"{token} completed {secs:.1f}s", flush=True)


def _make_reporter():
    """A reporter for actuate.process that translates real stage events into the dashboard's
    stdout vocabulary. Times each real stage by wall clock between events."""
    state = {"last": time.monotonic()}

    def reporter(stage: str, status: str, note: str) -> None:
        # Only terminal events advance the timeline. 'flag'/'info' (e.g. "slam unavailable",
        # "objects unavailable") are mid-stage notes -- log them, but don't re-fire the tokens,
        # or a stage with two flags would light up three times.
        if status not in ("done", "skipped"):
            print(f"[{stage}] {status}: {note}", flush=True)
            return
        now = time.monotonic()
        elapsed = now - state["last"]
        state["last"] = now
        tokens = STAGE_MAP.get(stage)
        if not tokens:
            # retarget / package / viz -> not a timeline row, but keep it in the log honestly
            print(f"[{stage}] {status}: {note}", flush=True)
            return
        per = elapsed / len(tokens)
        for tok in tokens:
            _emit_start(tok)
            if note:
                print(f"[{tok}] {status}: {note}", flush=True)
            _emit_done(tok, per)

    return reporter


def _confidence_tree(components, n_frames: int, quality: int) -> dict:
    return {
        "sync_integrity": components.sync_integrity,
        "calibration_completeness": components.calibration_completeness,
        "perception_confidence": components.perception_confidence,
        "contact_consistency": components.contact_consistency,
        "ik_convergence_rate": components.ik_convergence_rate,
        "n_frames": n_frames,
        "quality_1_to_5": quality,
    }


def _write_episode_outputs(ep, out_dir: Path) -> None:
    """Map a real CanonicalEpisode -> the dashboard's per-episode JSON contract."""
    d = out_dir / ep.episode_id
    d.mkdir(parents=True, exist_ok=True)

    m = ep.episode_meta
    c = m.components
    n_frames = len(ep.frames)
    quality = int(m.quality)
    # honest EIS on a 0-100 scale derived from the real 1-5 quality
    eis = round(quality / 5 * 100, 1)

    # WiLoR-0-hands / empty / weak episodes must surface for review, not masquerade as clean.
    needs_review = (n_frames == 0) or quality <= 2 or bool(m.mistakes)

    elig = "needs_review"
    re_map = getattr(ep, "retarget_eligibility", {}) or {}
    if re_map:
        elig = "eligible" if any(re_map.values()) else "ineligible"

    recommended = ("training" if quality >= 4
                   else "pretraining_only" if quality >= 3
                   else "needs_review")

    cert = {
        "episode_id": ep.episode_id,
        "task_label": ep.task,
        "task_confidence": None,                 # no separate classifier confidence today
        "eis_score": eis,
        "retargeting_eligible": elig,
        "recommended_use": recommended,
        "object_class": None,
        "needs_human_review": needs_review,
        "task_classification_disagreement": False,
        "confidence_tree": _confidence_tree(c, n_frames, quality),
        # extra, honest context the dashboard can show or ignore:
        "n_frames": n_frames,
        "consent": ep.consent.value,
        "pii_status": ep.pii_status.value,
        "is_deliverable": ep.is_deliverable,
        "note": ("no hands detected in this footage -- WiLoR found 0 hands; try the HaMeR "
                 "fallback on a GPU box or a dataset that ships its own keypoints."
                 if n_frames == 0 else ""),
    }
    (d / "quality_certificate.json").write_text(json.dumps(cert, indent=2), encoding="utf-8")

    lang = {"language_instruction": ep.task,
            "paraphrases": list(getattr(ep, "task_paraphrases", []) or [])}
    (d / "language_grounding.json").write_text(json.dumps(lang, indent=2), encoding="utf-8")

    phases = [{"phase": s.instruction, "start_frame": s.start_frame,
               "end_frame": s.end_frame, "confidence": s.confidence}
              for s in ep.subtasks]
    (d / "phase_segmentation.json").write_text(
        json.dumps({"phases": phases}, indent=2), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Actuate-dashboard pipeline bridge")
    ap.add_argument("--session", required=True, help="dashboard session id (used as run name)")
    ap.add_argument("--video", required=True, help="path to the raw capture video")
    ap.add_argument("--output", required=True, help="processed dir: <output>/<ep_id>/*.json")
    ap.add_argument("--task", default=None, help="task description (optional)")
    ap.add_argument("--rig", default="auto")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--redact-pii", action="store_true")
    ap.add_argument("--viz", action="store_true",
                    help="also write a Rerun .rrd (off by default: the viewer can segfault on "
                         "low-VRAM machines and the dashboard does not need it from here).")
    # accepted for dashboard-compat; recorded, not yet branched on
    ap.add_argument("--imu-source", default="internal")
    ap.add_argument("--depth-mode", default="monocular_estimate")
    args = ap.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    video = Path(args.video)
    if not video.exists():
        print(f"[error] video not found: {video}", file=sys.stderr, flush=True)
        return 2

    import actuate

    work = out_dir / "_work"
    run = None
    try:
        run = actuate.process(
            str(video), rig=args.rig, task=args.task, max_frames=args.max_frames,
            out=str(work), redact_pii=args.redact_pii, viz=args.viz,
            reporter=_make_reporter(),
        )
    except Exception as exc:
        # A late stage (e.g. the Rerun viewer) can die AFTER canonical.json is on disk. Don't
        # throw away a completed episode: fall through and emit outputs if canonical exists.
        print(f"[warn] pipeline raised: {type(exc).__name__}: {exc}", file=sys.stderr,
              flush=True)

    from actuate.schema import CanonicalEpisode

    canon = _find_canonical(work, run)
    if canon is None or not canon.exists():
        print("[error] no canonical.json was produced (perception did not yield an episode).",
              file=sys.stderr, flush=True)
        return 1
    ep = CanonicalEpisode.model_validate_json(canon.read_text(encoding="utf-8"))
    _write_episode_outputs(ep, out_dir)

    n = len(ep.frames)
    print(f"[done] session={args.session} quality={ep.episode_meta.quality}/5 frames={n} "
          f"task={ep.task!r} -> {out_dir / ep.episode_id}", flush=True)
    return 0


def _find_canonical(work: Path, run) -> Path | None:
    """The run's canonical.json -- from the run object if we have it, else by search (so a
    late-stage crash that lost the return value still finds the output)."""
    if run is not None:
        p = run._result.out / "canonical.json"
        if p.exists():
            return p
    hits = sorted(work.rglob("canonical.json")) if work.exists() else []
    return hits[0] if hits else None


if __name__ == "__main__":
    raise SystemExit(main())
