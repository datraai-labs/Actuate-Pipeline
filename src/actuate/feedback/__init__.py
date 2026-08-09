"""Minimum L8 review routing and version-aware reprocessing signals.

This is intentionally small: model retraining is still future work, but quality/privacy flags
now produce a queryable route instead of dying as unread JSON fields.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

from actuate.config import ConsentStatus, PiiStatus


@dataclass(frozen=True)
class ReviewRoute:
    queue: str
    reasons: tuple[str, ...]
    may_export: bool
    needs_human_review: bool

    def model_dump(self) -> dict:
        return asdict(self)


def route_episode(episode) -> ReviewRoute:
    reasons: list[str] = []
    if episode.consent is not ConsentStatus.GRANTED:
        reasons.append(f"consent={episode.consent.value}")
    if episode.pii_status is not PiiStatus.PASSED:
        reasons.append(f"pii_status={episode.pii_status.value}")
    if reasons:
        return ReviewRoute("privacy_block", tuple(reasons), False, True)

    if episode.episode_meta.quality is None:
        reasons.append("quality not measured")
    elif episode.episode_meta.quality < 3:
        reasons.append(f"provisional quality={episode.episode_meta.quality}/5")
    reasons.extend(episode.episode_meta.mistakes)
    reasons.extend(
        f"retarget[{name}]=ineligible"
        for name, eligible in episode.retarget_eligibility.items()
        if eligible is False
    )
    disagreement = any(
        "disagree" in str(note).lower() for note in episode.derivation_notes.values()
    )
    if disagreement:
        reasons.append("task classification disagreement")
    if reasons:
        queue = "robustness_review" if episode.episode_meta.quality in (1, 2) else "human_review"
        return ReviewRoute(queue, tuple(dict.fromkeys(reasons)), True, True)
    return ReviewRoute("delivery_candidate", (), True, False)


__all__ = ["ReviewRoute", "route_episode"]
