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
def _resolve_source(source: str, work_root: Path, **kwargs) -> Path:
    """Resolve a source spec to a local session directory the pipeline can read.

    Local paths (a video file or an existing session dir) are handled here; remote schemes
    (hf://, s3://, http(s)://, openx://) dispatch to `actuate.sources`. `kwargs` like
    `files=`/`split=` pass through to the remote resolver.
    """
    if "://" in source:
        from actuate import sources

        return sources.resolve(source, work_root, **kwargs)

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


def auto_task(session: Path, api_key: str | None = None, *, client=None,
              prompt_fn=None) -> str | None:
    """Name the manipulation task from a keyframe (SDK layer -- it needs the VLM).

    With a VLM client/key: one call on the middle frame -> a short imperative task ($ tiny,
    one image). Without a key: call `prompt_fn(message)` if given (the CLI passes an
    interactive prompt), else None. Never fabricates a task silently -- None means "unknown",
    which the exporter fail-closes on.
    """
    from actuate.language import make_client

    client = client or make_client(api_key)
    if client is None:
        if prompt_fn is not None:
            ans = prompt_fn("No task given and no API key. Describe the manipulation task "
                            "(or leave blank to skip)")
            return (ans.strip() or None) if ans else None
        return None

    import json

    import cv2

    from actuate.ingest.run import _session_video
    from actuate.language.vlm import VLM_MODEL, encode_frame_base64

    video = _session_video(session)
    cap = cv2.VideoCapture(str(video))
    mid = int((cap.get(cv2.CAP_PROP_FRAME_COUNT) or 2) // 2)
    cap.release()
    b64 = encode_frame_base64(video, mid)
    if b64 is None:
        return None
    schema = {"type": "object",
              "properties": {"task": {"type": "string", "description":
                             "the manipulation task, imperative, e.g. 'pick up the cup'"}},
              "required": ["task"], "additionalProperties": False}
    resp = client.messages.create(
        model=VLM_MODEL, max_tokens=128,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                         "data": b64}},
            {"type": "text", "text": "What manipulation task is being performed? "
                                     "Answer as one short imperative instruction."}]}],
        output_config={"format": {"type": "json_schema", "schema": schema}})
    text = next(b.text for b in resp.content if b.type == "text")
    return json.loads(text)["task"].strip() or None


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
) -> ProcessingRun:
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

    src_kwargs = {k: kwargs[k] for k in ("files", "split", "max_episodes") if k in kwargs}
    session = _resolve_source(source, work_root, **src_kwargs)
    ensure_session_meta(session)
    rig = _detect_rig(session, rig)
    from actuate.config import product_capabilities

    rig_capability = next(
        (item for item in product_capabilities()["rigs"] if item["id"] == rig), None
    )
    if rig_capability is None:
        raise ValueError(f"unknown rig {rig!r}; read the capability registry before processing")
    if not rig_capability["enabled"]:
        raise ValueError(f"rig {rig!r} is not enabled: {rig_capability['status']}")
    # auto-task: a VLM keyframe call when a key is available and no task was given
    # ($ tiny, one image). No key -> stays None, and the exporter fail-closes on it.
    if task is None and kwargs.get("auto_task", True):
        task = auto_task(session, prompt_fn=kwargs.get("prompt_fn"))

    # Consent is never inferred from execution mode. Local ownership is not evidence that
    # every recorded subject granted permission. Callers must pass an explicit decision.
    consent = kwargs.get("consent")

    run_out = work_root / f"{session.name}_out"
    profile = {
        "rig": rig,
        "embodiment": embodiment,
        "task": task,
        "consent": consent,
        # PII redaction (face blur -> pii_status PASSED). Opt-in: only when you ask for it,
        # because PASSED is a delivery claim, not a default (see io.redact / io.consent).
        "redact_pii": bool(kwargs.get("redact_pii", False)),
        "video": _video_name(session),
        "perception": {
            "enabled": True,
            "max_frames": max_frames,
            # None lets the object stage derive vocabulary from the task and its broad
            # manipulation fallback. An explicit list remains an operator override.
            "prompts": prompts,
        },
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
) -> ExportResult:
    """Process, then invoke the requested writer in its isolated interpreter.

    This preserves the convenience API without loading WiLoR/UniDepth and LeRobot or
    TensorFlow in one process. The output dataset is written below ``out/<format>``; the
    durable canonical work directory remains alongside it.
    """
    import json
    import os
    import subprocess

    fmt = _FORMATS.get(export_format)
    if fmt is None:
        raise ValueError(f"unknown export format {export_format!r}")
    env_name = "ACTUATE_LEROBOT_PYTHON" if fmt == "lerobot" else "ACTUATE_RLDS_PYTHON"
    writer_python = os.getenv(env_name)
    if not writer_python:
        raise RuntimeError(
            f"{export_format} uses an isolated writer. Set {env_name} to that "
            "environment's Python executable; see docs/quickstart.md."
        )

    out = Path(out).expanduser().resolve()
    run = process(source, out=out / "work", **kwargs)
    destination = out / export_format
    command = [
        writer_python,
        "-m",
        "actuate.cli",
        "export",
        str(run._result.out.resolve()),
        "--format",
        export_format,
        "--out",
        str(destination),
    ]
    embodiment = kwargs.get("embodiment")
    if embodiment:
        command.extend(["--embodiment", str(embodiment)])
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode:
        detail = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(f"isolated {export_format} export failed: {detail}")

    if fmt == "lerobot":
        info = json.loads((destination / "meta" / "info.json").read_text(encoding="utf-8"))
        n_frames = int(info["total_frames"])
    else:
        manifest = json.loads(
            (destination / "actuate_manifest.json").read_text(encoding="utf-8")
        )
        n_frames = int(manifest.get("n_steps") or manifest.get("total_steps") or 0)
    return ExportResult(
        format=export_format,
        path=destination,
        n_frames=n_frames,
        embodiment=str(embodiment) if embodiment else None,
    )


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
        manifest_path = self._result.out / "run_manifest.json"
        if manifest_path.exists():
            import json

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            exports = manifest.setdefault("artifacts", {}).setdefault("exports", [])
            record = {
                "format": format,
                "path": path.resolve().as_uri(),
                "frames": int(n),
                "embodiment": emb,
            }
            exports[:] = [x for x in exports if not (
                x.get("format") == format and x.get("path") == record["path"]
            )]
            exports.append(record)
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return ExportResult(format=format, path=path, n_frames=n, embodiment=emb)

    def upload_to_s3(self, *, export_dirs: list | None = None,
                     clean_local: bool = False) -> dict:
        """Push this run's durable artifacts (raw video, canonical, exports) to S3.

        `clean_local=True` deletes the local copies after a verified upload, so the result
        lives in AWS, not on disk. Needs an `aws_profile` in `actuate config` (or AWS_PROFILE).
        """
        from actuate.cloud import upload_run

        ep = self._ep()
        video = self.session / self.profile["video"]
        return upload_run(self.canonical_path, ep.capture_id, ep.episode_id,
                          video=video if video.exists() else None,
                          export_dirs=[Path(d) for d in (export_dirs or [])],
                          clean_local=clean_local)

    def push_to_hub(self, repo_id: str, *, private: bool = True, token: str | None = None,
                    format: str = "lerobot_v3") -> str:
        """Export and push this run's dataset to the HuggingFace Hub. Returns the repo URL."""
        from actuate.sources import push_to_hub as _push

        export_dir = self._result.out / "hub_export"
        self.export(format, path=export_dir)
        return _push(export_dir, repo_id, private=private, token=token)

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
        from actuate.certify.score import THRESHOLD_SET_VERSION, THRESHOLDS_PROVISIONAL

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
            "thresholds_provisional": THRESHOLDS_PROVISIONAL,
            "threshold_set": THRESHOLD_SET_VERSION,
        }

    def __repr__(self) -> str:
        return (f"ProcessingRun(status={self.status!r}, quality={self.quality}, "
                f"frames={self.num_frames}, canonical={self.canonical_path!r})")
