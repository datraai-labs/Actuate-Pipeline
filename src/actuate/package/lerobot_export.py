"""L7 -- LeRobot v3 exporter.

Master Spec §L7: *"using the reference library itself so schema correctness is inherited,
not re-implemented."* So this writes through `LeRobotDataset.create/add_frame/save_episode`
rather than hand-rolling chunked Parquet + MP4 + metadata. If LeRobot changes its layout,
we get the new layout for free; if we had re-implemented it from memory, we would get a
schema-valid-looking dataset that its loader rejects.

The gate (§L7, non-negotiable): **load the export with LeRobot's own loader and run one
real training step**, including the normalization round-trip. Schema-correctness asserted
by our own test is not a substitute -- see `tests/integration/test_lerobot_gate.py`.

### Fail-closed on `task`

`task` is a required LeRobot field (`add_frame` does `frame.pop("task")` and will KeyError
without it). It is also the field v1 cannot honestly produce: its classifier returns
`unknown` and its language grounding wraps that in "Perform unknown task using right hand".

This exporter **refuses to export an episode with no task**. It does not substitute a
placeholder, does not fall back to the template, and does not let LeRobot fail with a
cryptic KeyError three layers down. The requirement is enforced here, where a human can
read the error.

### Normalization

1st/99th percentile -> [-1, 1] is the default (π0.5 / EgoVerse), and the raw percentiles
plus mean/std are shipped alongside so a customer on 2/98-per-timestep (TRI LBM) or z-score
(EgoMimic) can re-derive without recomputing over the whole dataset. TRI LBM's finding is
that normalization dominates downstream performance, so getting this wrong is not cosmetic.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from actuate.canonical.build import episode_dof_names, state_and_action_vectors
from actuate.config import Tier
from actuate.schema import CanonicalEpisode, FieldStats, NormStats


class ExportRefused(RuntimeError):
    """The episode cannot be honestly exported. Not a warning."""


@dataclass
class ExportResult:
    root: Path
    repo_id: str
    n_frames: int
    n_dropped: int
    norm_stats: NormStats
    tier: Tier
    ego_contaminated: bool


def compute_norm_stats(state: np.ndarray, action: np.ndarray) -> NormStats:
    """1/99 percentiles AND mean/std, both shipped (Master Spec §L7)."""

    def stats(a: np.ndarray) -> FieldStats:
        return FieldStats(
            p01=tuple(float(v) for v in np.percentile(a, 1, axis=0)),
            p99=tuple(float(v) for v in np.percentile(a, 99, axis=0)),
            mean=tuple(float(v) for v in a.mean(axis=0)),
            std=tuple(float(v) for v in a.std(axis=0)),
        )

    return NormStats(state=stats(state), action=stats(action))


def normalize_p01_p99(a: np.ndarray, s: FieldStats) -> np.ndarray:
    """-> [-1, 1]. The default the exported metadata declares."""
    lo, hi = np.asarray(s.p01), np.asarray(s.p99)
    span = np.where(np.abs(hi - lo) < 1e-8, 1.0, hi - lo)
    return np.clip(2.0 * (a - lo) / span - 1.0, -1.0, 1.0)


def denormalize_p01_p99(a: np.ndarray, s: FieldStats) -> np.ndarray:
    lo, hi = np.asarray(s.p01), np.asarray(s.p99)
    span = np.where(np.abs(hi - lo) < 1e-8, 1.0, hi - lo)
    return (a + 1.0) / 2.0 * span + lo


#: Training resolution. VLA backbones (SigLIP/PaliGemma-class) consume ~224px; shipping
#: 1080p would multiply dataset size for no training benefit. This IS a re-encode, which
#: AWS Architecture §1 warns against for the *raw* store — but a delivery dataset is a
#: different artifact from raw provenance, and LeRobot v3's contract is per-camera chunked
#: MP4 at training resolution. The raw 190 MB original is untouched in actuate-raw-dev.
_IMAGE_HW = (224, 224)


def _decode_frames(video: Path, indices: np.ndarray) -> dict[int, np.ndarray]:
    """Pull exactly the frames we keep, resized, RGB uint8 (H, W, C)."""
    import cv2

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ExportRefused(f"cannot open video {video}")

    wanted = set(int(i) for i in indices)
    out: dict[int, np.ndarray] = {}
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx in wanted:
            frame = cv2.resize(frame, (_IMAGE_HW[1], _IMAGE_HW[0]), interpolation=cv2.INTER_AREA)
            out[idx] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        idx += 1
    cap.release()
    return out


def export_lerobot_v3(
    episode: CanonicalEpisode,
    out: Path,
    *,
    repo_id: str = "actuate/dev",
    fps: int = 30,
    tier: Tier = Tier.STAGE1_VOLUME,
    overwrite: bool = False,
    video: Path | None = None,
) -> ExportResult:
    """Write `episode` as a LeRobot v3 dataset. Raises ExportRefused if it cannot be honest.

    `video` must be the REDACTED source. A VLA dataset without `observation.images.*` is
    not a VLA dataset — the models are vision-language-action, and LeRobot's own policies
    refuse a state-only dataset outright ("You must provide at least one image or the
    environment state among the inputs"). So images are required, not optional.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if video is None or not Path(video).exists():
        raise ExportRefused(
            f"episode {episode.episode_id}: no video supplied ({video}).\n\n"
            "A VLA dataset needs observation.images.* — LeRobot's own policies reject a "
            "state-only dataset. Pass the REDACTED video; the un-redacted original must "
            "never reach a delivery artifact."
        )

    # ---- fail-closed on the required task field ----
    if not episode.task:
        raise ExportRefused(
            f"episode {episode.episode_id} has no `task`. LeRobot requires it on every "
            "frame and most VLAs condition on it.\n\n"
            "v1's classifier returned 'unknown' and its language grounding emitted "
            "'Perform unknown task using right hand with power grasp.' -- a fluent sentence "
            "containing no task. Exporting that would launder a failed classification into "
            "a training label.\n\n"
            "Supply an operator-verified task explicitly, or fix classification. This "
            "exporter will not invent one."
        )

    state, action, valid = state_and_action_vectors(episode)
    dof_names = episode_dof_names(episode)  # 8 (wrist-only) or 53 (wrist + full MANO)
    n_total = len(state)
    keep = np.flatnonzero(valid)
    if keep.size == 0:
        raise ExportRefused(
            f"episode {episode.episode_id}: no frame has both a wrist pose and a successor. "
            "There is nothing to train on."
        )

    # Frames with no detected hand are DROPPED, not zero-filled. Zeroing would teach a
    # policy to drive the end-effector to the camera origin every time the hand left view.
    s, a = state[keep], action[keep]
    norm = compute_norm_stats(s, a)

    root = Path(out)
    if root.exists():
        if not overwrite:
            raise ExportRefused(f"{root} exists; pass overwrite=True")
        shutil.rmtree(root)

    # Camera name comes from the rig registry and is enforced consistent from ingestion —
    # a camera called "head" in one session and "head_cam" in the next silently splits a
    # training dataset in two (Master Spec §3).
    cam = next(iter(episode.frames[0].images), "head")
    image_key = f"observation.images.{cam}"

    frames_by_idx = _decode_frames(Path(video), np.array([episode.frames[i].frame_idx for i in keep]))

    features = {
        image_key: {
            "dtype": "video",
            "shape": (*_IMAGE_HW, 3),
            "names": ["height", "width", "channel"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (len(dof_names),),
            "names": dof_names,
        },
        "action": {
            "dtype": "float32",
            "shape": (len(dof_names),),
            "names": dof_names,
        },
    }

    ds = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=features,
        root=root,
        robot_type=episode.rig.value,
        use_videos=True,  # per-camera chunked MP4, LeRobot v3's own writer
    )

    for n, i in enumerate(keep):
        frame_idx = episode.frames[i].frame_idx
        img = frames_by_idx.get(frame_idx)
        if img is None:
            raise ExportRefused(
                f"video has no frame {frame_idx}; the video and the per-frame records do "
                "not describe the same recording."
            )
        ds.add_frame(
            {
                image_key: img,
                "observation.state": s[n].astype(np.float32),
                "action": a[n].astype(np.float32),
                "task": episode.task,
            }
        )
    ds.save_episode()

    # Ship the percentiles AND mean/std so a customer on a different normalization scheme
    # can re-derive without a full pass over the data.
    (root / "meta" / "actuate_norm_stats.json").write_text(
        norm.model_dump_json(indent=2), encoding="utf-8"
    )
    (root / "meta" / "actuate_provenance.json").write_text(
        __import__("json").dumps(
            {
                "schema_version": episode.schema_version,
                "capture_id": episode.capture_id,
                "source_content_hash": episode.source_content_hash,
                "tier": tier.value,
                "control_mode": episode.control_mode.value if episode.control_mode else None,
                "consent": episode.consent.value,
                "pii_status": episode.pii_status.value,
                "normalization": "p01_p99_to_pm1 (default); raw percentiles + mean/std shipped",
                "derivation_notes": episode.derivation_notes,
                "frames_dropped_no_hand": int(n_total - keep.size),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return ExportResult(
        root=root,
        repo_id=repo_id,
        n_frames=int(keep.size),
        n_dropped=int(n_total - keep.size),
        norm_stats=norm,
        tier=tier,
        ego_contaminated="action_semantics" in episode.derivation_notes,
    )
