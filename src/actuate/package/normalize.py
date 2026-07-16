"""L7 normalization stats -- the full percentile set, per space (Master Spec §L7).

TRI LBM's finding is that normalization DOMINATES downstream performance, so the stats are a
first-class deliverable, not metadata garnish. The default transform is 1/99-percentile ->
[-1, 1] (π0.5 / EgoVerse), and the FULL raw percentile set (1, 2, 5, 25, 50, 75, 95, 98, 99)
plus mean/std ships alongside so a customer on 2/98-per-timestep (TRI LBM) or z-score
(EgoMimic) re-derives their scheme without a pass over the raw dataset.

Per-dimension, and -- for dual-space exports -- per SPACE: the human wrist+MANO stream and a
robot embodiment's joint stream have different ranges, and EgoMimic measured a 38% performance
drop without per-embodiment normalization. Mixing their stats would be exactly that bug.
"""

from __future__ import annotations

import numpy as np

from actuate.schema import FieldStats, NormStats

#: The raw percentiles shipped (schema v4 FieldStats). p01/p99 are the required pair the
#: default transform uses; the rest are optional in the schema and always filled here.
PERCENTILES = (1, 2, 5, 25, 50, 75, 95, 98, 99)


def compute_field_stats(a: np.ndarray) -> FieldStats:
    """Full per-dimension stats for one (T, D) stream."""
    a = np.asarray(a, dtype=np.float64)
    if a.ndim != 2 or len(a) == 0:
        raise ValueError(f"expected a non-empty (T, D) array, got shape {a.shape}")
    pct = {p: tuple(float(v) for v in np.percentile(a, p, axis=0)) for p in PERCENTILES}
    return FieldStats(
        p01=pct[1], p02=pct[2], p05=pct[5], p25=pct[25], p50=pct[50],
        p75=pct[75], p95=pct[95], p98=pct[98], p99=pct[99],
        mean=tuple(float(v) for v in a.mean(axis=0)),
        std=tuple(float(v) for v in a.std(axis=0)),
    )


def compute(state: np.ndarray, action: np.ndarray) -> NormStats:
    """Stats for a (state, action) pair -- one normalization space."""
    return NormStats(state=compute_field_stats(state), action=compute_field_stats(action))


def normalize_p01_p99(a: np.ndarray, s: FieldStats) -> np.ndarray:
    """-> [-1, 1], CLIPPED. The default the exported metadata declares.

    Clipping is part of the contract: the 1st/99th percentiles are the range by definition,
    so the outer 2% saturates. The round-trip guarantee below therefore holds for values
    INSIDE [p01, p99] -- outliers saturate by design, they do not round-trip.
    """
    lo, hi = np.asarray(s.p01), np.asarray(s.p99)
    span = np.where(np.abs(hi - lo) < 1e-8, 1.0, hi - lo)
    return np.clip(2.0 * (a - lo) / span - 1.0, -1.0, 1.0)


def denormalize_p01_p99(a: np.ndarray, s: FieldStats) -> np.ndarray:
    lo, hi = np.asarray(s.p01), np.asarray(s.p99)
    span = np.where(np.abs(hi - lo) < 1e-8, 1.0, hi - lo)
    return (a + 1.0) / 2.0 * span + lo


def verify_round_trip(a: np.ndarray, s: FieldStats, atol: float = 1e-9) -> None:
    """normalize -> denormalize must reproduce every in-range value (TRI LBM: non-negotiable).

    THE SUBTLETY THAT MAKES THIS A GATE: "in range" is judged by the DATA's own recomputed
    1/99 percentiles, never by the stat under test. Round-trip error only ever comes from
    clipping, so a check that trusts the shipped stat to define the clip region would let a
    corrupted stat shrink the region and hide its own damage -- it could never fail. Judged
    against the data's true range, a tampered p01/p99 clips values it shouldn't and fails
    loudly here.

    Scope: this exercises the p01/p99 -> [-1,1] transform. mean/std ship for customer
    re-derivation and are not exercised by it.
    """
    a = np.asarray(a, dtype=np.float64)
    lo_true = np.percentile(a, 1, axis=0)
    hi_true = np.percentile(a, 99, axis=0)
    back = denormalize_p01_p99(normalize_p01_p99(a, s), s)
    in_range = (a >= lo_true) & (a <= hi_true)
    err = np.abs(back - a)
    err[~in_range] = 0.0
    worst = float(err.max()) if err.size else 0.0
    if worst > atol:
        d = int(np.unravel_index(err.argmax(), err.shape)[1])
        raise ValueError(
            f"normalization round-trip failed: max error {worst:.3e} > {atol:.0e} "
            f"(dimension {d}). The shipped stats do not reproduce the data they claim to "
            "describe -- do not export."
        )
