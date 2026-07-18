"""The high-level Actuate SDK -- `import actuate` and process data in under 10 lines.

    import actuate
    run = actuate.process(source="./my_video.mp4", rig="head_mounted", task="pick up cup")
    run.export("lerobot_v3", path="./dataset/")

This is a THIN wrapper over the library: `process` resolves the source to a session
directory, generates `session_meta.json` from the video if missing, builds a profile, and
drives `actuate.pipeline.run_pipeline`. The perception/retargeting/certification stack is
untouched -- the SDK is its simplest consumer, exactly as the CLI is.

Layer: `actuate.sdk` sits just below `actuate.cli` and above `actuate.pipeline`; it never
imports the CLI. The CLI's simplified commands (Part B) call THIS.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from actuate.config import auth
from actuate.ingest.run import ensure_session_meta
from actuate.pipeline import run_pipeline

#: Default object-detection prompts when the caller gives none (generic manipulation).
_DEFAULT_PROMPTS = ["hand", "object", "tool"]
#: Export-format aliases the SDK accepts.
_FORMATS = {"lerobot_v3": "lerobot", "lerobot": "lerobot", "rlds": "rlds",
            "openx": "rlds", "open_x": "rlds"}


def login(api_key: str | None = None, *, aws_profile: str | None = None) -> dict:
    """One-time setup. No key -> local mode (processing runs on this machine). A key ->
    cloud mode (managed features, currently a scaffolded 'coming soon')."""
    if api_key:
        return auth.login(mode="cloud", api_key=api_key, aws_profile=aws_profile)
    return auth.login(mode="local", aws_profile=aws_profile)


# ------------------------------------------------------------------ source resolution
def _resolve_source(source: str, work_root: Path) -> Path:
    """Resolve a source spec to a local session directory the pipeline can read.

    Part A handles LOCAL paths (a video file or an existing session dir). Remote schemes
    (hf://, s3://, http(s)://, openx://) are dispatched to `actuate.sources` when present;
    a clear error names what's missing otherwise.
    """
    if "://" in source:
        try:
            from actuate import sources
        except ImportError:
            raise ValueError(
                f"source {source!r} needs a resolver, but actuate.sources is not "
                "available. Use a local path for now, or install the source extras.")
        return sources.resolve(source, work_root)

    p = Path(source).expanduser()
    if not p.exists():
        raise FileNotFoundError(
            f"source not found: {p}. Give a local video file, a processed session "
            "directory, or a remote source (hf:// s3:// https:// openx://).")
    if p.is_dir():
        _normalize(p)
        return p                                   # already a session directory
    # a single video file -> stage it into its own session directory
    session = work_root / p.stem
    session.mkdir(parents=True, exist_ok=True)
    dest = session / p.name
    if not dest.exists():
        import shutil

        shutil.copy2(p, dest)
    _normalize(session)
    return session


def _normalize(session: Path) -> None:
    """Rename spaces/parens/unicode files to pipeline-safe names (the Kaggle bug)."""
    from actuate.sources.detect import normalize_filenames

    normalize_filenames(session)


def _video_name(session: Path) -> str:
    from actuate.ingest.run import _session_video

    return _session_video(session).name


def _detect_rig(session: Path, override: str) -> str:
    """`auto` -> infer from video geometry (Part D fills this in); otherwise pass through."""
    if override != "auto":
        return override
    try:
        from actuate.sources.detect import detect_rig

        return detect_rig(session)
    except Exception:
        return "head_mounted"                      # safe default until Part D lands


# ------------------------------------------------------------------ process
def process(
    source: str,
    rig: str = "auto",
    embodiment: str | None = None,
    task: str | None = None,
    max_frames: int | None = None,
    *,
    out: str | Path | None = None,
    prompts: list[str] | None = None,
    reporter=None,
    **kwargs,
) -> "ProcessingRun":
    """Process any source into a certified canonical episode + a Rerun recording.

    `source`: local video / session dir, or hf:// s3:// https:// openx:// (when
    actuate.sources is available). `rig='auto'` infers from the video. `max_frames=None`
    means the full video. `embodiment` defaults to your configured default (needed only for
    retargeting/robot-space export). Returns a ProcessingRun.
    """
    cfg = auth.load_config()
    embodiment = embodiment or cfg.get("default_embodiment")
    work_root = Path(out).expanduser() if out else Path("./actuate_runs")
    work_root.mkdir(parents=True, exist_ok=True)

    session = _resolve_source(source, work_root)
    meta = ensure_session_meta(session)
    rig = _detect_rig(session, rig)
    if max_frames is None:
        max_frames = int(meta.get("frame_count") or 45)

    # auto-task: a VLM keyframe call when a key is available and no task was given
    # ($ tiny, one image). No key -> stays None, and the exporter fail-closes on it.
    if task is None and kwargs.get("auto_task", True):
        from actuate.sources.detect import auto_task as _auto_task

        task = _auto_task(session)

    # local processing = your own data -> consent GRANTED. pii_status stays PENDING, so the
    # delivery gate still blocks (see pipeline._stage_canonical). Cloud mode keeps PENDING.
    from actuate.sources.detect import local_consent_default

    consent = (local_consent_default().value if cfg.get("mode", "local") == "local"
               else None)

    run_out = work_root / f"{session.name}_out"
    profile = {
        "rig": rig,
        "embodiment": embodiment,
        "task": task,
        "consent": consent,
        "video": _video_name(session),
        "perception": {"enabled": True, "max_frames": max_frames,
                       "prompts": prompts or _DEFAULT_PROMPTS},
        "retarget": kwargs.get("retarget", {}),          # no model -> stage skips itself
        "language": kwargs.get("language", {"mode": "skip"}),  # opt-in; SDK won't auto-bill
        "export": {"formats": [], "tier": "all"},        # export is explicit via .export()
        "viz": kwargs.get("viz", True),
    }
    result = run_pipeline(session, run_out, profile, reporter=reporter)
    return ProcessingRun(session=session, profile=profile, _result=result)


def process_and_export(
    source: str,
    export_format: str = "lerobot_v3",
    out: str | Path = "./output/",
    **kwargs,
) -> "ExportResult":
    """One-liner: process, then export in one call. Returns the ExportResult."""
    out = Path(out).expanduser()
    run = process(source, out=out / "_work", **kwargs)
    return run.export(export_format, path=out)


# ------------------------------------------------------------------ result objects
@dataclass
class ExportResult:
    format: str
    path: Path
    n_frames: int
    embodiment: str | None

    def __repr__(self) -> str:
        return (f"ExportResult(format={self.format!r}, path={str(self.path)!r}, "
                f"n_frames={self.n_frames})")


class ProcessingRun:
    """The handle a user holds after `process()`. Reads results off the canonical episode."""

    def __init__(self, session: Path, profile: dict, _result):
        self.session = Path(session)
        self.profile = profile
        self._result = _result
        self.canonical_path = str(_result.canonical_path)
        self._episode = None

    # -- lazy episode load --------------------------------------------------------------
    def _ep(self):
        if self._episode is None:
            from actuate.schema import CanonicalEpisode

            self._episode = CanonicalEpisode.model_validate_json(
                Path(self.canonical_path).read_text(encoding="utf-8"))
        return self._episode

    # -- status fields ------------------------------------------------------------------
    @property
    def status(self) -> str:
        if not Path(self.canonical_path).exists():
            return "failed"
        canon = self._result.checkpoint.get("canonical", {})
        return "completed" if canon.get("status") == "done" else "partial"

    @property
    def quality(self) -> int | None:
        return self._ep().episode_meta.quality

    @property
    def num_episodes(self) -> int:
        return 1

    @property
    def num_frames(self) -> int:
        return len(self._ep().frames)

    @property
    def duration_seconds(self) -> float | None:
        ts = [f.t for f in self._ep().frames]
        return round(max(ts) - min(ts), 3) if len(ts) >= 2 else None

    # -- actions ------------------------------------------------------------------------
    def export(self, format: str, path: str | Path, embodiment: str | None = None
               ) -> ExportResult:
        fmt = _FORMATS.get(format)
        if fmt is None:
            raise ValueError(f"unknown export format {format!r}; "
                             f"try one of {sorted(set(_FORMATS))}")
        ep = self._ep()
        path = Path(path).expanduser()
        video = self.session / self.profile["video"]
        emb = embodiment or self.profile.get("embodiment")
        emb = emb if emb and emb in ep.action_robot else None
        if fmt == "lerobot":
            from actuate.package import export_lerobot_v3

            res = export_lerobot_v3(ep, path, overwrite=True, video=video,
                                    embodiment=emb, tier="all")
            n = res.n_frames
        else:
            from actuate.package.rlds_export import export_rlds

            res = export_rlds(ep, path, embodiment=emb, video=video, tier="all")
            n = res.n_steps
        return ExportResult(format=format, path=path, n_frames=n, embodiment=emb)

    def viz(self, live: bool = False) -> None:
        """Open the pipeline recording in Rerun. `live` spawns a viewer."""
        rrd = self._result.out / "pipeline.rrd"
        import rerun as rr

        if rrd.exists() and not live:
            rr.init("actuate", spawn=True)
            rr.log_file_from_path(str(rrd))
        else:                                         # re-log live from the episode
            from actuate.viz import rerun_log

            rr.init("actuate", spawn=True)
            rerun_log.log_episode(self._ep())

    def summary(self) -> dict:
        return {
            "status": self.status,
            "quality": self.quality,
            "num_episodes": self.num_episodes,
            "num_frames": self.num_frames,
            "duration_seconds": self.duration_seconds,
            "canonical_path": self.canonical_path,
            "rig": self.profile.get("rig"),
            "embodiment": self.profile.get("embodiment"),
            "task": self._ep().task,
            "stages": {k: v.get("status") for k, v in self._result.checkpoint.items()
                       if not k.startswith("_") and isinstance(v, dict)},
        }

    def certificate(self) -> dict:
        m = self._ep().episode_meta
        c = m.components
        return {
            "quality": m.quality,
            "speed": m.speed,
            "mistakes": list(m.mistakes),
            "components": {
                "sync_integrity": c.sync_integrity,
                "calibration_completeness": c.calibration_completeness,
                "perception_confidence": c.perception_confidence,
                "contact_consistency": c.contact_consistency,
                "ik_convergence_rate": c.ik_convergence_rate,
            },
            "consent": self._ep().consent.value,
            "deliverable": self._ep().is_deliverable,
        }

    def __repr__(self) -> str:
        return (f"ProcessingRun(status={self.status!r}, quality={self.quality}, "
                f"frames={self.num_frames}, canonical={self.canonical_path!r})")
