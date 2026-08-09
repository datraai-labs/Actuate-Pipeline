"""The L2 trust-weighted arbiter -- Master Spec §L2. The heart of fusion.

When two sources report the same channel (say a glove and a vision model both estimate finger
flexion), the arbiter picks ONE, by a fixed trust ordering:

    measured_robotspace  (DexUMI exoskeleton encoders)      -- ground truth in ROBOT space
      > measured_human   (glove flex / joint sensors)       -- ground truth in HUMAN space
      > gripper_aperture (UMI parallel-jaw width)           -- a 1-DoF hardware proxy
      > vision_primary   (a metric vision model, e.g. WiLoR)
      > vision_fallback  (a weaker vision inference)
      > approximated     (a heuristic / constant)

This ordering is not a preference -- it is the whole reason L2 exists. A 0.9 confidence from a
tactile sensor and a 0.9 from a vision heuristic are not the same claim, and the dexterous
branch (§L5) must be able to tell them apart. Getting the ordering backwards silently ships
vision guesses stamped as if they were hardware truth, which is the single worst failure this
layer can have. So the ordering lives in ONE place (`TRUST_RANK`), and the broken-priority test
(`tests/unit/test_fusion_arbiter.py`) exists specifically to fail if it is ever inverted.

Confidence is a TIE-BREAKER WITHIN a tier, never across tiers. A vision reading at confidence
1.0 never beats a glove reading at confidence 0.3 -- the tier dominates. This is deliberate:
confidence measures how sure a source is of itself, not how much we trust the source.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from actuate.config import Provenance

#: Trust rank, derived from the Provenance declaration order (higher = more trusted). Kept as
#: an explicit dict rather than an enum .value so the ordering is auditable at a glance and so
#: a test can pass a DIFFERENT (broken) rank to prove the arbiter actually depends on it.
TRUST_RANK: dict[Provenance, int] = {
    p: rank for rank, p in enumerate(reversed(list(Provenance)))
}
# reversed(list(Provenance)) => APPROXIMATED first (rank 0) ... MEASURED_ROBOTSPACE last
# (rank 5). So a higher rank == more trusted, matching the ordering in the docstring.


@dataclass(frozen=True)
class Candidate:
    """One source's reading of one channel, tagged with where it came from."""

    channel: str
    value: Any
    provenance: Provenance
    # No default: callers must either supply a real source score or say that the source
    # exposes no comparable score with ``None``.  Missing metadata must never become
    # perfect confidence merely because a key was absent.
    confidence: float | None

    def __post_init__(self) -> None:
        if self.provenance not in TRUST_RANK:
            raise ValueError(f"unknown provenance {self.provenance!r}")
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError("candidate confidence must be in [0, 1] or None (not measured)")


def arbitrate(
    candidates: list[Candidate],
    rank: Callable[[Provenance], int] | dict[Provenance, int] = TRUST_RANK,
) -> Candidate:
    """Pick the winning candidate: highest trust tier, then highest confidence within it.

    `rank` is injectable ONLY so the broken-priority test can pass an inverted ordering and
    demonstrate the arbiter genuinely obeys it (a test that can't fail proves nothing). Nothing
    in production passes anything but the default `TRUST_RANK`.
    """
    if not candidates:
        raise ValueError("arbitrate() needs at least one candidate")
    rank_fn = rank.__getitem__ if isinstance(rank, dict) else rank
    # Sort key: (trust tier, confidence). A missing source-specific confidence remains
    # unknown and loses only a tie within the same provenance tier; it never changes the
    # cross-tier trust order. Stable within equal keys -> first-listed wins.
    return max(
        candidates,
        key=lambda c: (
            rank_fn(c.provenance),
            float("-inf") if c.confidence is None else c.confidence,
        ),
    )


def resolve_channel(
    channel: str,
    candidates: list[Candidate],
    rank: Callable[[Provenance], int] | dict[Provenance, int] = TRUST_RANK,
) -> Candidate | None:
    """Arbitrate the candidates for one channel, or None if there are none.

    A None result is meaningful: it means the channel was not sourced at all on this rig (e.g.
    contact on a bare-hand head rig with no tactile hardware and no vision proxy). None must
    propagate as 'not measured', never as a zero -- see the schema's contact validators.
    """
    subset = [c for c in candidates if c.channel == channel]
    return arbitrate(subset, rank) if subset else None
