"""L7 -- RLDS secondary exporter (Open-X-Embodiment convention).

Same doctrine as the LeRobot exporter: write THROUGH the reference library
(`tfds.dataset_builders.store_as_tfds_dataset`) so the on-disk layout is inherited, not
re-implemented from memory -- v3 flags the RLDS field layout as "training knowledge only",
so every structural fact here was verified against the installed tfds (4.9.10) by writing
and re-loading a probe dataset first. Two facts that probe caught: iterable splits must
yield (key, example) PAIRS, and the nested `steps` structure is `tfds.features.Dataset`,
not a list feature.

### Step layout (Open-X)

    steps: {observation: {image, state}, action, language_instruction,
            reward, discount, is_first, is_last, is_terminal}

`reward` is 0.0 and `discount` 1.0 throughout: these are human demonstrations with no
reward signal, and shipping constants is the Open-X convention for teleop/demo data --
fabricating a shaped reward would be worse than honest constants.

Dual-space parity with Part D: `embodiment=` adds a per-step `action_robot_<embodiment>`
field (underscores -- tfds feature names are identifiers, the dotted LeRobot spelling is
not portable here).

The same refusals as LeRobot export apply: no task, no video, mixed state layouts, tier
filter excluding everything. Same reason: an RLDS dataset that silently lacks language or
images is not an Open-X dataset, it just looks like one.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from actuate.canonical.build import episode_dof_names, state_and_action_vectors
from actuate.package.lerobot_export import (
    _IMAGE_HW,
    ExportRefused,
    _decode_frames,
    _episode_tier,
    _robot_action_rows,
    _tier_filter,
)
from actuate.package.manifest import generate as _generate_manifest
from actuate.schema import CanonicalEpisode


@dataclass
class RldsExportResult:
    root: Path                 # data_dir handed to tfds; tfds.load(name, data_dir=root)
    name: str
    version: str
    n_episodes: int
    n_steps: int
    n_dropped: int
    tier_counts: dict[str, int]
    embodiment: str | None


def _features(dof_names: list[str], n_robot_joints: int | None, embodiment: str | None):
    import tensorflow_datasets as tfds

    step = {
        "observation": tfds.features.FeaturesDict({
            "image": tfds.features.Image(shape=(*_IMAGE_HW, 3), dtype=np.uint8),
            "state": tfds.features.Tensor(shape=(len(dof_names),), dtype=np.float32),
        }),
        "action": tfds.features.Tensor(shape=(len(dof_names),), dtype=np.float32),
        "language_instruction": tfds.features.Text(),
        "reward": tfds.features.Scalar(dtype=np.float32),
        "discount": tfds.features.Scalar(dtype=np.float32),
        "is_first": np.bool_,
        "is_last": np.bool_,
        "is_terminal": np.bool_,
    }
    if embodiment is not None:
        step[f"action_robot_{embodiment}"] = tfds.features.Tensor(
            shape=(n_robot_joints,), dtype=np.float32)
    return tfds.features.FeaturesDict({"steps": tfds.features.Dataset(step)})


def export_rlds(
    episodes: CanonicalEpisode | list[CanonicalEpisode],
    out: Path,
    *,
    name: str = "actuate_dataset",
    version: str = "1.0.0",
    tier: str | None = "all",
    embodiment: str | None = None,
    video: Path | dict[str, Path] | None = None,
) -> RldsExportResult:
    """Write episodes as ONE RLDS/Open-X dataset under `out` (the tfds data_dir).

    Load it back with ``tfds.load(name, data_dir=out, split="train")`` -- which is exactly
    what the verification gate does.
    """
    import tensorflow_datasets as tfds

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

    def _video_for(ep: CanonicalEpisode) -> Path:
        v = video.get(ep.episode_id) if isinstance(video, dict) else video
        if v is None or not Path(v).exists():
            raise ExportRefused(
                f"episode {ep.episode_id}: no video supplied ({v}). An Open-X dataset "
                "without observation.image is not an Open-X dataset. Pass the REDACTED "
                "video; the un-redacted original must never reach a delivery artifact."
            )
        return Path(v)

    dof_names = episode_dof_names(eps[0])
    prepared = []
    for ep in eps:
        if not ep.task:
            raise ExportRefused(
                f"episode {ep.episode_id} has no `task`. language_instruction is a "
                "required Open-X field and this exporter will not invent one -- the same "
                "fail-closed rule the LeRobot exporter enforces."
            )
        if episode_dof_names(ep) != dof_names:
            raise ExportRefused(
                f"episode {ep.episode_id} has a different state layout "
                f"({len(episode_dof_names(ep))} dof vs {len(dof_names)}). One dataset, "
                "one feature schema — export mixed layouts separately."
            )
        state, action, valid = state_and_action_vectors(ep)
        keep = np.flatnonzero(valid)
        if keep.size == 0:
            raise ExportRefused(
                f"episode {ep.episode_id}: no frame has both a wrist pose and a "
                "successor. There is nothing to train on."
            )
        robot = (_robot_action_rows(ep, embodiment, keep, len(state))
                 if embodiment is not None else None)
        prepared.append((ep, _video_for(ep), state, action, keep, robot))

    n_robot_joints = len(prepared[0][5][0]) if embodiment is not None else None

    def _episode_example(ep, vid, state, action, keep, robot):
        s, a = state[keep], action[keep]
        frames_by_idx = _decode_frames(vid, np.array([ep.frames[i].frame_idx for i in keep]))
        steps = []
        for n, i in enumerate(keep):
            frame_idx = ep.frames[i].frame_idx
            img = frames_by_idx.get(frame_idx)
            if img is None:
                raise ExportRefused(
                    f"video has no frame {frame_idx}; the video and the per-frame records "
                    "do not describe the same recording."
                )
            step = {
                "observation": {"image": img, "state": s[n].astype(np.float32)},
                "action": a[n].astype(np.float32),
                "language_instruction": ep.task,
                "reward": np.float32(0.0),
                "discount": np.float32(1.0),
                "is_first": n == 0,
                "is_last": n == keep.size - 1,
                "is_terminal": n == keep.size - 1,
            }
            if robot is not None:
                step[f"action_robot_{embodiment}"] = robot[n].astype(np.float32)
            steps.append(step)
        return {"steps": steps}

    root = Path(out)
    root.mkdir(parents=True, exist_ok=True)
    tfds.dataset_builders.store_as_tfds_dataset(
        name=name,
        version=version,
        features=_features(dof_names, n_robot_joints, embodiment),
        split_datasets={"train": [
            (ep.episode_id, _episode_example(ep, vid, st, ac, k, r))
            for ep, vid, st, ac, k, r in prepared
        ]},
        data_dir=root,
        description="Actuate canonical episodes, Open-X-Embodiment RLDS layout.",
    )

    n_steps = sum(int(k.size) for *_, k, _ in prepared)
    n_dropped = sum(len(st) - int(k.size) for _, _, st, _, k, _ in prepared)
    tier_counts = Counter(_episode_tier(e).value for e in eps)

    (root / "actuate_manifest.json").write_text(
        _generate_manifest(eps).to_json(), encoding="utf-8")
    (root / "actuate_provenance.json").write_text(
        json.dumps({
            "format": "rlds/open-x",
            "tfds_name": name,
            "tfds_version": version,
            "schema_version": eps[0].schema_version,
            "capture_ids": [e.capture_id for e in eps],
            "source_content_hashes": [e.source_content_hash for e in eps],
            "tier_filter": str(tier),
            "tier_counts": dict(tier_counts),
            "embodiment": embodiment,
            "consent": [e.consent.value for e in eps],
            "pii_status": [e.pii_status.value for e in eps],
            "reward_note": "reward=0, discount=1 throughout: human demonstrations carry "
                           "no reward signal (Open-X convention for demo data).",
            "derivation_notes": {e.episode_id: e.derivation_notes for e in eps},
            "frames_dropped_no_hand": n_dropped,
        }, indent=2),
        encoding="utf-8")

    return RldsExportResult(
        root=root, name=name, version=version, n_episodes=len(eps), n_steps=n_steps,
        n_dropped=n_dropped, tier_counts=dict(tier_counts), embodiment=embodiment,
    )
