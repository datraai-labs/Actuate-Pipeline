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

import json
import shutil
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from actuate.canonical.build import episode_dof_names, state_and_action_vectors
from actuate.config import Tier
from actuate.lineage import build_lineage
from actuate.package import normalize as _norm
from actuate.package import transforms as _tf
from actuate.package.manifest import generate as _generate_manifest
from actuate.schema import CanonicalEpisode, FieldStats, NormStats

# The normalization helpers moved to actuate.package.normalize (Phase 5 Part D) and are
# re-exported here because the load+train gate and downstream callers import them from
# this module. compute_norm_stats now ships the FULL percentile set (schema v4).
normalize_p01_p99 = _norm.normalize_p01_p99
denormalize_p01_p99 = _norm.denormalize_p01_p99
compute_norm_stats = _norm.compute


class ExportRefused(RuntimeError):
    """The episode cannot be honestly exported. Not a warning."""


def _assert_export_consent(ep: CanonicalEpisode) -> None:
    """Re-verify consent and PII at local export, not only at remote delivery."""
    if not ep.is_deliverable:
        raise ExportRefused(
            f"episode {ep.episode_id}: local export blocked: {ep.delivery_block_reason()}. "
            "An export is a portable copy of the data and enforces the same consent/PII "
            "boundary as delivery."
        )


@dataclass
class ExportResult:
    root: Path
    repo_id: str
    n_frames: int
    n_dropped: int
    norm_stats: NormStats                       # human-space (observation.state / action)
    tier_counts: dict[str, int]                 # tier value -> episodes exported
    ego_contaminated: bool
    n_episodes: int = 1
    embodiment: str | None = None               # dual-space partner, when exported
    robot_norm_stats: dict[str, FieldStats] = field(default_factory=dict)


def _tier_filter(tier) -> set[Tier] | None:
    """CLI/API tier filter -> the set of tiers kept (None = keep all)."""
    if tier in (None, "all", "ALL"):
        return None
    if isinstance(tier, Tier):
        return {tier}
    aliases = {
        "stage1": Tier.STAGE1_VOLUME, "stage1_volume": Tier.STAGE1_VOLUME,
        "stage2": Tier.STAGE2_ANCHOR, "stage2_anchor": Tier.STAGE2_ANCHOR,
    }
    if str(tier) in aliases:
        return {aliases[str(tier)]}
    raise ExportRefused(f"unknown tier filter {tier!r}; use stage1, stage2, or all")


def _episode_tier(ep: CanonicalEpisode) -> Tier:
    """An episode with no assigned tier is volume data by definition -- stage2 is a claim
    (matched viewpoint, verified alignment) that must be made explicitly, never defaulted."""
    return ep.tier or Tier.STAGE1_VOLUME


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


def _robot_action_rows(episode: CanonicalEpisode, embodiment: str,
                       keep: np.ndarray, n_total: int) -> np.ndarray:
    """Frame-align action_robot[embodiment].joint_traj with the exporter's kept frames.

    RobotAction carries no frame ids (schema limitation, recorded), so alignment is by
    LENGTH and refused when ambiguous: a trajectory the length of the FULL episode is
    indexed by `keep`; one already the length of the kept set is used as-is; anything else
    means the retarget and the export disagree about frame accounting, and shipping a
    misaligned robot action is worse than shipping none.
    """
    ra = episode.action_robot.get(embodiment)
    if ra is None:
        raise ExportRefused(
            f"episode {episode.episode_id} has no action_robot[{embodiment!r}]. Dual-space "
            "export needs the L5 retarget to have run and been attached "
            "(retarget.arm.attach_to_episode)."
        )
    verdict = episode.retarget_eligibility.get(embodiment)
    if verdict is not True:
        state = "missing" if verdict is None else "ineligible"
        raise ExportRefused(
            f"episode {episode.episode_id}: retarget eligibility for {embodiment!r} is "
            f"{state}. Robot-space data may only export after an explicit passing physics "
            "verdict."
        )
    if ra.joint_traj is None:
        raise ExportRefused(
            f"episode {episode.episode_id}: action_robot[{embodiment!r}] has no joint_traj "
            "(EE-only). The dual-space contract ships joint trajectories."
        )
    traj = np.asarray(ra.joint_traj, dtype=np.float64)
    if len(traj) == n_total:
        return traj[keep]
    if len(traj) == keep.size:
        return traj
    raise ExportRefused(
        f"episode {episode.episode_id}: action_robot[{embodiment!r}].joint_traj has "
        f"{len(traj)} steps but the episode has {n_total} frames ({keep.size} kept). "
        "The retarget and the export disagree about frame accounting -- refusing to guess."
    )


def _degenerate_dimensions(
    episode_id: str,
    state: np.ndarray,
    action: np.ndarray,
    keep: np.ndarray,
    names: list[str],
    robot: np.ndarray | None = None,
) -> list[str]:
    """Find zero-variance training columns and hard-block a degenerate grasp signal."""
    if keep.size < 2:
        raise ExportRefused(
            f"episode {episode_id}: only {keep.size} trainable transition(s); at least two "
            "are required to detect degenerate training signals."
        )
    warnings_out: list[str] = []
    for space, values in (("observation.state", state[keep]), ("action", action[keep])):
        for column, name in enumerate(names):
            values_col = values[:, column]
            if np.all(np.isfinite(values_col)) and float(np.ptp(values_col)) <= 1e-8:
                label = f"{space}.{name}"
                if name == "grasp":
                    raise ExportRefused(
                        f"episode {episode_id}: {label} has zero variance. A constant grasp "
                        "column is not useful manipulation supervision and commonly means the "
                        "signal was never measured; refusing to ship it."
                    )
                warnings_out.append(label)
    if robot is not None:
        for column in range(robot.shape[1]):
            if float(np.ptp(robot[:, column])) <= 1e-8:
                warnings_out.append(f"action.robot.j{column}")
    if warnings_out:
        import warnings

        warnings.warn(
            f"episode {episode_id} has zero-variance dimensions: "
            f"{', '.join(warnings_out)}",
            RuntimeWarning,
            stacklevel=2,
        )
    return warnings_out


def _hand_points_3d(frame) -> np.ndarray | None:
    """All hand keypoints in the camera frame, both sides pooled (for the mask bbox)."""
    pts = [np.asarray(h.keypoints_3d) for h in frame.hands.values()
           if h.keypoints_3d is not None]
    return np.concatenate(pts) if pts else None


def _apply_transforms(img, transforms, frame, next_frame, intrinsics, src_hw):
    if "masked_hand" in transforms:
        pts = _hand_points_3d(frame)
        px = None if pts is None else _tf.project_points(pts, intrinsics, src_hw, _IMAGE_HW)
        img = _tf.masked_hand(img, px)
    if "eef_overlay" in transforms:
        def wrist_px(f):
            if f is None:
                return None
            pts = _hand_points_3d(f)
            if pts is None:
                return None
            px = _tf.project_points(pts[:1], intrinsics, src_hw, _IMAGE_HW)
            return None if px is None else px[0]
        img = _tf.eef_overlay(img, wrist_px(frame), wrist_px(next_frame))
    return img


def _source_hw(video: Path) -> tuple[int, int]:
    import cv2

    cap = cv2.VideoCapture(str(video))
    hw = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    cap.release()
    return hw


def export_lerobot_v3(
    episodes: CanonicalEpisode | list[CanonicalEpisode],
    out: Path,
    *,
    repo_id: str = "actuate/dev",
    fps: int = 30,
    tier: Tier | str | None = "all",
    overwrite: bool = False,
    video: Path | dict[str, Path] | None = None,
    embodiment: str | None = None,
    transforms: tuple[str, ...] = (),
    intrinsics: tuple[float, float, float, float] | None = None,
) -> ExportResult:
    """Write episodes as ONE LeRobot v3 dataset. Raises ExportRefused if it cannot be honest.

    Phase 5 Part D additions on top of the v3 exporter:

    - **tier filter** (`tier="stage1" | "stage2" | "all"`): episodes are kept by their OWN
      `episode.tier` (unassigned = stage1_volume -- stage2 is a claim, never a default).
    - **dual-space** (`embodiment=`): ships `action.robot.<embodiment>` (retargeted joints)
      ALONGSIDE the human-space `action`, tagged by name so a training script selects a
      space instead of getting one imposed. Norm stats are computed PER SPACE (EgoMimic:
      38% drop without per-embodiment normalization).
    - **transforms** (`("masked_hand", "eef_overlay")`): EgoMimic co-training variants,
      applied at export time, never stored. Both need `intrinsics` (fx, fy, cx, cy at the
      source resolution) -- no intrinsics, no transform, because a mask drawn with guessed
      intrinsics hides the wrong pixels silently.

    `video` must be the REDACTED source (a dict keyed by episode_id for multi-episode
    exports). A VLA dataset without observation.images.* is not a VLA dataset.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    eps = [episodes] if isinstance(episodes, CanonicalEpisode) else list(episodes)
    if not eps:
        raise ExportRefused("no episodes given; nothing to export")

    keep_tiers = _tier_filter(tier)
    if keep_tiers is not None:
        eps = [e for e in eps if _episode_tier(e) in keep_tiers]
        if not eps:
            raise ExportRefused(
                f"tier filter {tier!r} excluded every episode -- nothing to export. "
                "That is the filter working, not an error to route around."
            )

    if transforms:
        unknown = set(transforms) - set(_tf.TRANSFORM_NAMES)
        if unknown:
            raise ExportRefused(f"unknown transform(s) {sorted(unknown)}; "
                                f"available: {_tf.TRANSFORM_NAMES}")
        if intrinsics is None:
            raise ExportRefused(
                "transforms need intrinsics (fx, fy, cx, cy at source resolution). A hand "
                "mask projected with guessed intrinsics blacks out the wrong region and "
                "silently trains the policy on exactly the pixels it was meant to hide."
            )

    def _video_for(ep: CanonicalEpisode) -> Path:
        v = video.get(ep.episode_id) if isinstance(video, dict) else video
        if v is None or not Path(v).exists():
            raise ExportRefused(
                f"episode {ep.episode_id}: no video supplied ({v}).\n\n"
                "A VLA dataset needs observation.images.* — LeRobot's own policies reject "
                "a state-only dataset. Pass the REDACTED video; the un-redacted original "
                "must never reach a delivery artifact."
            )
        return Path(v)

    # ---- per-episode fail-closed checks + vector extraction, BEFORE any writing ----
    dof_names = episode_dof_names(eps[0])
    prepared = []
    degenerate_by_episode: dict[str, list[str]] = {}
    for ep in eps:
        if not ep.task:
            raise ExportRefused(
                f"episode {ep.episode_id} has no `task`. LeRobot requires it on every "
                "frame and most VLAs condition on it.\n\n"
                "v1's classifier returned 'unknown' and its language grounding emitted "
                "'Perform unknown task using right hand with power grasp.' -- a fluent "
                "sentence containing no task. Exporting that would launder a failed "
                "classification into a training label.\n\n"
                "Supply an operator-verified task explicitly, or fix classification. This "
                "exporter will not invent one."
            )
        _assert_export_consent(ep)
        if episode_dof_names(ep) != dof_names:
            raise ExportRefused(
                f"episode {ep.episode_id} has a different state layout "
                f"({len(episode_dof_names(ep))} dof vs {len(dof_names)}). One dataset, one "
                "feature schema — export mixed layouts separately."
            )
        state, action, valid = state_and_action_vectors(ep)
        keep = np.flatnonzero(valid)
        if keep.size == 0:
            raise ExportRefused(
                f"episode {ep.episode_id}: no frame has both a wrist pose and a successor. "
                "There is nothing to train on."
            )
        robot = (_robot_action_rows(ep, embodiment, keep, len(state))
                 if embodiment is not None else None)
        degenerate_by_episode[ep.episode_id] = _degenerate_dimensions(
            ep.episode_id, state, action, keep, dof_names, robot
        )
        prepared.append((ep, _video_for(ep), state, action, keep, robot))

    # Frames with no detected hand are DROPPED, not zero-filled. Zeroing would teach a
    # policy to drive the end-effector to the camera origin every time the hand left view.
    s_all = np.concatenate([st[k] for _, _, st, _, k, _ in prepared])
    a_all = np.concatenate([ac[k] for _, _, _, ac, k, _ in prepared])
    norm = compute_norm_stats(s_all, a_all)
    _norm.verify_round_trip(s_all, norm.state)
    _norm.verify_round_trip(a_all, norm.action)

    robot_norm: dict[str, FieldStats] = {}
    if embodiment is not None:
        r_all = np.concatenate([r for *_, r in prepared])
        robot_norm[embodiment] = _norm.compute_field_stats(r_all)
        _norm.verify_round_trip(r_all, robot_norm[embodiment])

    root = Path(out)
    if root.exists():
        if not overwrite:
            raise ExportRefused(f"{root} exists; pass overwrite=True")
        shutil.rmtree(root)

    # Camera name comes from the rig registry and is enforced consistent from ingestion —
    # a camera called "head" in one session and "head_cam" in the next silently splits a
    # training dataset in two (Master Spec §3).
    cam = next(iter(eps[0].frames[0].images), "head")
    image_key = f"observation.images.{cam}"

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
    if embodiment is not None:
        n_joints = len(prepared[0][5][0])
        features[f"action.robot.{embodiment}"] = {
            "dtype": "float32",
            "shape": (n_joints,),
            "names": [f"{embodiment}_j{i}" for i in range(n_joints)],
        }

    ds = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=features,
        root=root,
        robot_type=eps[0].rig.value,
        use_videos=True,  # per-camera chunked MP4, LeRobot v3's own writer
    )

    n_frames = n_dropped = 0
    for ep, vid, state, action, keep, robot in prepared:
        s, a = state[keep], action[keep]
        src_hw = _source_hw(vid) if transforms else None
        frames_by_idx = _decode_frames(vid, np.array([ep.frames[i].frame_idx for i in keep]))
        for n, i in enumerate(keep):
            frame_idx = ep.frames[i].frame_idx
            img = frames_by_idx.get(frame_idx)
            if img is None:
                raise ExportRefused(
                    f"video has no frame {frame_idx}; the video and the per-frame records "
                    "do not describe the same recording."
                )
            if transforms:
                nxt = ep.frames[keep[n + 1]] if n + 1 < keep.size else None
                img = _apply_transforms(img, transforms, ep.frames[i], nxt,
                                        intrinsics, src_hw)
            row = {
                image_key: img,
                "observation.state": s[n].astype(np.float32),
                "action": a[n].astype(np.float32),
                "task": ep.task,
            }
            if robot is not None:
                row[f"action.robot.{embodiment}"] = robot[n].astype(np.float32)
            ds.add_frame(row)
        ds.save_episode()
        n_frames += int(keep.size)
        n_dropped += int(len(state) - keep.size)

    # Ship the percentiles AND mean/std so a customer on a different normalization scheme
    # can re-derive without a full pass over the data. Human space keeps its historical
    # filename (the load+train gate reads it); robot spaces are per-embodiment files.
    (root / "meta" / "actuate_norm_stats.json").write_text(
        norm.model_dump_json(indent=2), encoding="utf-8"
    )
    for emb, stats in robot_norm.items():
        (root / "meta" / f"actuate_norm_stats.{emb}.json").write_text(
            NormStats(action=stats).model_dump_json(indent=2), encoding="utf-8"
        )
    (root / "meta" / "actuate_manifest.json").write_text(
        _generate_manifest(eps).to_json(), encoding="utf-8"
    )

    tier_counts = Counter(_episode_tier(e).value for e in eps)
    ego = any("action_semantics" in e.derivation_notes for e in eps)
    (root / "meta" / "actuate_provenance.json").write_text(
        json.dumps(
            {
                "schema_version": eps[0].schema_version,
                "capture_ids": [e.capture_id for e in eps],
                "source_content_hashes": [e.source_content_hash for e in eps],
                "tier_filter": str(tier),
                "tier_counts": dict(tier_counts),
                "embodiment": embodiment,
                "transforms": list(transforms),
                "control_mode": eps[0].control_mode.value if eps[0].control_mode else None,
                "consent": [e.consent.value for e in eps],
                "pii_status": [e.pii_status.value for e in eps],
                "normalization": "p01_p99_to_pm1 (default); full percentiles + mean/std "
                                 "shipped, per space",
                "derivation_notes": {e.episode_id: e.derivation_notes for e in eps},
                "frames_dropped_no_hand": n_dropped,
                "zero_variance_dimensions": degenerate_by_episode,
                "lineage": build_lineage({
                    "format": "lerobot_v3", "embodiment": embodiment
                }),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    from actuate.package.delivery_docs import write_dataset_readme

    write_dataset_readme(root, format_name="LeRobot v3")

    return ExportResult(
        root=root,
        repo_id=repo_id,
        n_frames=n_frames,
        n_dropped=n_dropped,
        norm_stats=norm,
        tier_counts=dict(tier_counts),
        ego_contaminated=ego,
        n_episodes=len(eps),
        embodiment=embodiment,
        robot_norm_stats=robot_norm,
    )
