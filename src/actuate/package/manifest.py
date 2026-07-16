"""L7 dataset manifest -- what a customer sees before loading a byte (Master Spec §L7).

EgoVerse's requirement drives the shape: scene diversity and demonstrator diversity are
reported SEPARATELY. Collapsing them into one "diversity" number hides which axis is thin --
100 episodes could be 100 scenes with 1 demonstrator, or 1 scene with 100 demonstrators, and
the two datasets train very different policies.

The manifest reports what IS, including the uncomfortable parts: unknown scene/demonstrator
ids count as unknown, a missing task shows up as untasked, and the certificate means include
the low scores. n=1 corpora look like n=1 corpora here.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field

import numpy as np

from actuate.schema import CanonicalEpisode


@dataclass
class DatasetManifest:
    episode_count: int
    total_frames: int
    effective_hours: float | None            # sum where recorded; None if nowhere recorded

    # diversity -- SEPARATE axes (EgoVerse)
    scene_count: int
    demonstrator_count: int
    episodes_with_unknown_scene: int
    episodes_with_unknown_demonstrator: int

    task_distribution: dict[str, int]        # task string -> episodes ("<untasked>" is honest)
    tier_distribution: dict[str, int]        # tier value -> episodes ("<unassigned>" likewise)
    mean_certificate: dict[str, float | None]  # per component + quality; None = never measured
    modality_inventory: dict[str, int]       # modality -> episodes carrying it
    rigs: dict[str, int]
    schema_versions: list[int]
    embodiments_with_actions: dict[str, int] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True) + "\n"


_CERT_COMPONENTS = ("sync_integrity", "calibration_completeness", "perception_confidence",
                    "contact_consistency", "ik_convergence_rate")


def _mean_or_none(vals: list[float]) -> float | None:
    return float(np.mean(vals)) if vals else None


def generate(episodes: CanonicalEpisode | list[CanonicalEpisode]) -> DatasetManifest:
    """Build the manifest for a set of canonical episodes."""
    eps = [episodes] if isinstance(episodes, CanonicalEpisode) else list(episodes)
    if not eps:
        raise ValueError("no episodes; a manifest for nothing describes nothing")

    scenes = {e.diversity.scene_id for e in eps if e.diversity.scene_id}
    demos = {e.diversity.demonstrator_id for e in eps if e.diversity.demonstrator_id}

    tasks = Counter((e.task or "<untasked>") for e in eps)
    tiers = Counter((e.tier.value if e.tier else "<unassigned>") for e in eps)
    rigs = Counter(e.rig.value for e in eps)

    cert: dict[str, float | None] = {}
    for name in _CERT_COMPONENTS:
        cert[name] = _mean_or_none(
            [getattr(e.episode_meta.components, name) for e in eps
             if getattr(e.episode_meta.components, name) is not None])
    cert["quality"] = _mean_or_none(
        [e.episode_meta.quality for e in eps if e.episode_meta.quality is not None])

    def _has(e: CanonicalEpisode, what: str) -> bool:
        checks = {
            "images": lambda f: bool(f.images),
            "depth": lambda f: bool(f.depth),
            "hands": lambda f: bool(f.hands),
            "mano": lambda f: any(h.mano is not None for h in f.hands.values()),
            "objects": lambda f: bool(f.objects),
            "contact": lambda f: f.contact is not None,
            "camera_pose": lambda f: f.camera_pose is not None,
            "interaction_state": lambda f: f.interaction_state is not None,
        }
        return any(checks[what](f) for f in e.frames)

    modality = {m: sum(1 for e in eps if _has(e, m))
                for m in ("images", "depth", "hands", "mano", "objects", "contact",
                          "camera_pose", "interaction_state")}
    modality["language_task"] = sum(1 for e in eps if e.task)
    modality["language_paraphrases"] = sum(1 for e in eps if e.task_paraphrases)
    modality["subtasks"] = sum(1 for e in eps if e.subtasks)

    hours = [e.effective_hours for e in eps if e.effective_hours is not None]

    return DatasetManifest(
        episode_count=len(eps),
        total_frames=sum(len(e.frames) for e in eps),
        effective_hours=float(sum(hours)) if hours else None,
        scene_count=len(scenes),
        demonstrator_count=len(demos),
        episodes_with_unknown_scene=sum(1 for e in eps if not e.diversity.scene_id),
        episodes_with_unknown_demonstrator=sum(
            1 for e in eps if not e.diversity.demonstrator_id),
        task_distribution=dict(tasks),
        tier_distribution=dict(tiers),
        mean_certificate=cert,
        modality_inventory=modality,
        rigs=dict(rigs),
        schema_versions=sorted({e.schema_version for e in eps}),
        embodiments_with_actions=dict(Counter(
            emb for e in eps for emb in e.action_robot)),
    )
