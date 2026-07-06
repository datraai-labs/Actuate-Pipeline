"""
DatraAI Pipeline — Main Entry Point
Run the complete action labeling pipeline on one or more sessions.

Usage:
  python run_pipeline.py --session raw/session_001
  python run_pipeline.py --batch raw/ --batch-id northstar_batch_001
  python run_pipeline.py --session raw/session_001 --upload
  python run_pipeline.py --session raw/session_001 --skip-qa --resume
"""

import argparse
import importlib.util
import logging
import sys
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Must happen before any step module prints — see utils/console_safety.py.
from utils.console_safety import install as _install_console_safety

_install_console_safety()

import time
import traceback
from datetime import datetime

import config as cfg


def _import_step(script_name: str):
    """
    Dynamically import a pipeline step script by filename.
    Handles filenames like '01_ingest.py' that aren't valid Python identifiers.
    """
    script_path = PROJECT_ROOT / "scripts" / f"{script_name}.py"
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")

    spec = importlib.util.spec_from_file_location(script_name, str(script_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ═══════════════════════════════════════════════════════════
# PIPELINE STEP DEFINITIONS
# ═══════════════════════════════════════════════════════════

# Lazy-load modules on first use
_step_modules = {}


def _get_step_module(name: str):
    if name not in _step_modules:
        _step_modules[name] = _import_step(name)
    return _step_modules[name]


PIPELINE_STEPS = [
    {
        "name": "01_ingest",
        "needs_raw_path": True,
        "outputs": ["compressed.mp4", "pts.npy", "imu_raw.npy", "session_meta.json"],
    },
    {
        "name": "02_sync",
        "outputs": ["session.h5"],
    },
    {
        "name": "03_qa",
        "outputs": ["qa_report.json"],
        "skippable": True,
    },
    {
        # Redacts the DELIVERABLE copy only (redacted_compressed.mp4) —
        # perception stages below keep reading the unredacted
        # compressed.mp4 for tracking fidelity. See
        # scripts/03b_privacy_redact.py module docstring. The actual
        # delivery-blocking consent gate is enforced separately, right
        # before the packaging step below, not here.
        "name": "03b_privacy_redact",
        "outputs": ["redacted_compressed.mp4", "privacy_report.json"],
        # Non-fatal: a redaction failure shouldn't block internal
        # processing (hand pose/primitives/etc. use the unredacted
        # compressed.mp4 regardless — see module docstring). Delivery is
        # still protected independently: 11_package.py refuses to package
        # a session with no redacted_compressed.mp4, and the consent gate
        # below blocks packaging outright without "granted" consent.
        "skippable": True,
    },
    {
        "name": "04_hand_pose",
        "outputs": ["hand_pose.json"],
    },
    {
        # STUB — see scripts/04c_object_track.py module docstring. Produces
        # a placeholder object_tracks.json (not real detection) so 04d and
        # 05_primitives' object-track-based contact detection have a
        # realistically-shaped input to run against.
        "name": "04c_object_track",
        "outputs": ["object_tracks.json"],
    },
    {
        "name": "04d_depth_estimate",
        "outputs": ["depth_data.json", "hand_pose_3d.json"],
    },
    {
        "name": "05_primitives",
        "outputs": ["primitives.json"],
    },
    {
        "name": "06_phase_segment",
        "outputs": ["phases.json"],
    },
    {
        # Splits phases.json into distinct task episodes (v2 addendum §6) —
        # 07/09/10 below run per episode, not per session.
        "name": "06b_episode_segment",
        "outputs": ["episodes.json"],
    },
    {
        "name": "07_task_classify",
        "outputs": ["task_label.json"],
    },
    {
        "name": "08_validate",
        "outputs": ["validation_report.json"],
    },
    {
        "name": "09_language_ground",
        "outputs": ["language_grounding.json"],
    },
    {
        "name": "10_eis",
        "outputs": ["quality_certificate.json"],
    },
]


def _run_step(step_name: str, session_id: str, session_path: Path):
    """
    Dynamically load and run a pipeline step.
    Steps that need the raw path receive it; others receive session_id.
    """
    mod = _get_step_module(step_name)
    step_def = next(s for s in PIPELINE_STEPS if s["name"] == step_name)

    if step_def.get("needs_raw_path"):
        return mod.run(session_path)
    else:
        return mod.run(session_id)


def _setup_logging(session_id: str) -> logging.Logger:
    """Set up file + console logging for a session."""
    log_dir = cfg.PROCESSED_DIR / session_id
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "pipeline.log"

    logger = logging.getLogger(f"datraai.{session_id}")
    logger.setLevel(logging.DEBUG)

    # Clear existing handlers
    logger.handlers.clear()

    # File handler
    fh = logging.FileHandler(str(log_path), mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    ))
    logger.addHandler(fh)

    # Console handler (INFO only)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)

    return logger


def _check_outputs_exist(session_id: str, outputs: list) -> bool:
    """Check if all output files for a step already exist."""
    proc_dir = cfg.PROCESSED_DIR / session_id
    return all((proc_dir / out).exists() for out in outputs)


def _step_is_skippable(step_name: str) -> bool:
    """Whether a given PIPELINE_STEPS entry is marked non-fatal (skippable)."""
    step_def = next((s for s in PIPELINE_STEPS if s["name"] == step_name), None)
    return bool(step_def and step_def.get("skippable"))


def _get_consent_status(session_id: str):
    """Read consent_status from a session's session_meta.json, or None if unavailable."""
    import json as _json

    meta_path = cfg.PROCESSED_DIR / session_id / "session_meta.json"
    if not meta_path.exists():
        return None
    with open(meta_path) as f:
        return _json.load(f).get("consent_status")


def run_session(
    session_path: Path,
    skip_qa: bool = False,
    resume: bool = False,
    upload: bool = False,
    batch_id: str = None,
) -> dict:
    """
    Run the complete pipeline for a single session.

    Returns:
        Summary dict with session results.
    """
    session_path = Path(session_path)
    session_id = session_path.name

    logger = _setup_logging(session_id)
    logger.info(f"{'═' * 60}")
    logger.info(f"DatraAI Pipeline — {session_id}")
    logger.info(f"Started: {datetime.now().isoformat()}")
    logger.info(f"{'═' * 60}")

    total_t0 = time.time()
    step_results = {}
    failed_step = None

    for step in PIPELINE_STEPS:
        step_name = step["name"]
        outputs = step.get("outputs", [])
        skippable = step.get("skippable", False)

        # Skip QA if flagged
        if skip_qa and step_name == "03_qa":
            logger.info(f"\n[{step_name}] SKIPPED (--skip-qa)")
            step_results[step_name] = "skipped"
            continue

        # Resume: skip if outputs already exist
        if resume and _check_outputs_exist(session_id, outputs):
            logger.info(f"\n[{step_name}] SKIPPED (outputs exist, --resume)")
            step_results[step_name] = "skipped (resume)"
            continue

        # Run step
        step_t0 = time.time()
        try:
            _run_step(step_name, session_id, session_path)
            step_elapsed = time.time() - step_t0

            # Verify outputs
            missing = [
                out for out in outputs
                if not (cfg.PROCESSED_DIR / session_id / out).exists()
            ]

            if missing:
                logger.warning(f"[{step_name}] ⚠ Missing outputs: {missing}")
                step_results[step_name] = f"completed ({step_elapsed:.1f}s) — missing: {missing}"
            else:
                logger.info(f"[{step_name}] ✓ ({step_elapsed:.1f}s)")
                step_results[step_name] = f"✓ ({step_elapsed:.1f}s)"

        except Exception as e:
            step_elapsed = time.time() - step_t0
            error_msg = f"{type(e).__name__}: {e}"
            logger.error(f"[{step_name}] ✗ FAILED ({step_elapsed:.1f}s): {error_msg}")
            logger.debug(traceback.format_exc())

            step_results[step_name] = f"✗ FAILED: {error_msg}"
            failed_step = step_name

            # Non-fatal for QA
            if skippable:
                logger.warning(f"[{step_name}] Non-fatal error, continuing...")
                continue

            # Fatal: stop pipeline
            logger.error(f"\n{'═' * 60}")
            logger.error(f"Pipeline STOPPED at {step_name}")
            logger.error(f"Error: {error_msg}")
            logger.error(f"Suggestion: Fix the issue and re-run with --resume")
            logger.error(f"{'═' * 60}")
            break

    # ─── Consent gate (v2 addendum §10) ──────────────────────
    # Hard block: a session never packages/ships without explicit consent.
    # This is enforced here (not just in 11_package.py) so the reason a
    # session didn't deliver is visible in the per-session summary, not
    # just an exception.
    consent_status = _get_consent_status(session_id)
    consent_blocked = cfg.BLOCK_DELIVERY_WITHOUT_CONSENT and consent_status != "granted"
    if consent_blocked:
        logger.error(
            f"\n{'═' * 60}\n"
            f"DELIVERY BLOCKED — consent_status={consent_status!r} (must be 'granted')\n"
            f"Session {session_id} was fully processed but will NOT be packaged for "
            f"delivery. Set consent_status to 'granted' in session_meta.json (or "
            f"raw/{session_id}/consent.json before re-ingesting) and re-run.\n"
            f"{'═' * 60}"
        )
        step_results["11_package"] = f"BLOCKED: consent_status={consent_status!r} (not 'granted')"

    # ─── Packaging step (runs separately) ─────────────────────
    if not consent_blocked and (failed_step is None or _step_is_skippable(failed_step)):
        try:
            step_t0 = time.time()
            batch = batch_id or f"batch_{datetime.now().strftime('%Y%m%d')}"
            pkg_mod = _get_step_module("11_package")
            pkg_mod.run([session_id], batch_id=batch, upload=upload)
            step_elapsed = time.time() - step_t0
            step_results["11_package"] = f"✓ ({step_elapsed:.1f}s)"
        except Exception as e:
            step_results["11_package"] = f"✗ FAILED: {e}"
            logger.error(f"[11_package] ✗ FAILED: {e}")

    total_elapsed = time.time() - total_t0

    # ─── Final summary ───────────────────────────────────────
    print(f"\n{'═' * 60}")
    print(f"PIPELINE SUMMARY — {session_id}")
    print(f"{'═' * 60}")
    for step_name, result in step_results.items():
        print(f"  {step_name}: {result}")
    print(f"{'─' * 60}")
    print(f"  Total time: {total_elapsed:.1f}s")

    # Print EIS and key metrics if available
    import json as _json

    cert_path = cfg.PROCESSED_DIR / session_id / "quality_certificate.json"
    if cert_path.exists():
        with open(cert_path) as f:
            cert = _json.load(f)
        print(f"  EIS: {cert.get('EIS', 'N/A')}/100")
        print(f"  Recommended use: {cert.get('recommended_use', [])}")

    qa_path = cfg.PROCESSED_DIR / session_id / "qa_report.json"
    if qa_path.exists():
        with open(qa_path) as f:
            qa = _json.load(f)
        print(f"  QA: {'PASS' if qa.get('overall_passed') else 'FAIL'} (score: {qa.get('qa_score', 'N/A')})")

    # List output files
    proc_dir = cfg.PROCESSED_DIR / session_id
    if proc_dir.exists():
        print(f"\n  Output files:")
        for f in sorted(proc_dir.iterdir()):
            if f.is_file():
                size = f.stat().st_size
                if size > 1024 * 1024:
                    size_str = f"{size / 1024 / 1024:.1f}MB"
                elif size > 1024:
                    size_str = f"{size / 1024:.1f}KB"
                else:
                    size_str = f"{size}B"
                print(f"    {f.name} ({size_str})")

    print(f"{'═' * 60}")
    print(f"\nTo re-run: python run_pipeline.py --session {session_path}")

    return {
        "session_id": session_id,
        "status": "completed" if failed_step is None else f"failed at {failed_step}",
        "duration_seconds": round(total_elapsed, 1),
        "steps": step_results,
    }


def run_batch(
    batch_path: Path,
    batch_id: str = None,
    upload: bool = False,
    skip_qa: bool = False,
    resume: bool = False,
) -> dict:
    """
    Run pipeline on all sessions in a batch folder.
    """
    batch_path = Path(batch_path)
    if not batch_path.is_dir():
        raise NotADirectoryError(f"Batch path is not a directory: {batch_path}")

    # Find session folders (containing raw.mp4)
    session_paths = sorted([
        d for d in batch_path.iterdir()
        if d.is_dir() and (d / "raw.mp4").exists()
    ])

    if len(session_paths) == 0:
        print(f"No sessions found in {batch_path}")
        return {"sessions": [], "batch_id": batch_id}

    print(f"\n{'═' * 60}")
    print(f"DatraAI BATCH PIPELINE")
    print(f"{'═' * 60}")
    print(f"Batch: {batch_id}")
    print(f"Sessions: {len(session_paths)}")
    print(f"{'═' * 60}\n")

    try:
        from tqdm import tqdm
        session_iter = tqdm(session_paths, desc="Sessions", unit="session")
    except ImportError:
        session_iter = session_paths

    results = []
    for sp in session_iter:
        try:
            result = run_session(
                sp,
                skip_qa=skip_qa,
                resume=resume,
                upload=False,  # Upload once at batch level
                batch_id=batch_id,
            )
            results.append(result)
        except Exception as e:
            print(f"\n✗ Session {sp.name} failed: {e}")
            results.append({
                "session_id": sp.name,
                "status": f"crashed: {e}",
                "duration_seconds": 0,
            })

    # Print batch summary table
    import json as _json

    print(f"\n{'═' * 80}")
    print(f"BATCH SUMMARY — {batch_id}")
    print(f"{'═' * 80}")
    print(f"{'Session':<20} {'Task':<22} {'EIS':>5} {'QA':>6} {'Duration':>10} {'Status':<12}")
    print(f"{'─' * 80}")

    for r in results:
        sid = r["session_id"]
        status = "✓" if "completed" in r.get("status", "") else "✗"

        task = "—"
        eis = "—"
        qa = "—"
        dur = f"{r.get('duration_seconds', 0):.1f}s"

        task_path = cfg.PROCESSED_DIR / sid / "task_label.json"
        if task_path.exists():
            with open(task_path) as f:
                task = _json.load(f).get("L1_task", "—")

        cert_path = cfg.PROCESSED_DIR / sid / "quality_certificate.json"
        if cert_path.exists():
            with open(cert_path) as f:
                eis = str(_json.load(f).get("EIS", "—"))

        qa_path = cfg.PROCESSED_DIR / sid / "qa_report.json"
        if qa_path.exists():
            with open(qa_path) as f:
                qa = "PASS" if _json.load(f).get("overall_passed") else "FAIL"

        print(f"{sid:<20} {task:<22} {eis:>5} {qa:>6} {dur:>10} {status:<12}")

    print(f"{'═' * 80}")

    # Upload batch if requested
    if upload:
        # Same consent gate as run_session() — a completed session still
        # isn't shippable without granted consent.
        consentable_ids = []
        for r in results:
            if "completed" not in r.get("status", ""):
                continue
            sid = r["session_id"]
            status = _get_consent_status(sid)
            if cfg.BLOCK_DELIVERY_WITHOUT_CONSENT and status != "granted":
                print(f"  ⚠ {sid}: excluded from batch upload — consent_status={status!r} (not 'granted')")
                continue
            consentable_ids.append(sid)

        if consentable_ids:
            try:
                pkg_mod = _get_step_module("11_package")
                pkg_mod.run(consentable_ids, batch_id=batch_id, upload=True)
            except Exception as e:
                print(f"Batch upload failed: {e}")
        else:
            print("Batch upload skipped — no sessions with granted consent.")

    return {"batch_id": batch_id, "sessions": results}


def main():
    parser = argparse.ArgumentParser(
        description="DatraAI Action Labeling Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py --session raw/session_001
  python run_pipeline.py --batch raw/ --batch-id northstar_batch_001
  python run_pipeline.py --session raw/session_001 --upload
  python run_pipeline.py --session raw/session_001 --skip-qa --resume
        """,
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--session",
        type=str,
        help="Path to raw session folder (e.g., raw/session_001)",
    )
    group.add_argument(
        "--batch",
        type=str,
        help="Path to folder containing multiple session folders",
    )

    parser.add_argument(
        "--batch-id",
        type=str,
        default=None,
        help="Name for delivery batch (default: batch_YYYYMMDD)",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload to S3 after packaging",
    )
    parser.add_argument(
        "--skip-qa",
        action="store_true",
        help="Skip QA check (for debugging)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip already-completed steps (check for output files)",
    )

    args = parser.parse_args()

    batch_id = args.batch_id or f"batch_{datetime.now().strftime('%Y%m%d')}"

    if args.session:
        run_session(
            Path(args.session),
            skip_qa=args.skip_qa,
            resume=args.resume,
            upload=args.upload,
            batch_id=batch_id,
        )
    elif args.batch:
        run_batch(
            Path(args.batch),
            batch_id=batch_id,
            upload=args.upload,
            skip_qa=args.skip_qa,
            resume=args.resume,
        )


if __name__ == "__main__":
    main()
