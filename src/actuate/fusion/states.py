"""Interaction-state classification + Schmitt gating -- Master Spec §L2.

Turns per-frame hand geometry into a coherent STATIC / GRASPED_L / GRASPED_R / GRASPED_BOTH /
MOVING stream. The hard part is not the classification -- it is stopping the state from
FLICKERING. A raw per-frame threshold on a noisy grasp signal toggles GRASPED/STATIC many
times a second, which is meaningless as a label and poison for a policy. Two mechanisms fix it:

1. **Schmitt trigger** (hysteresis): entering GRASP needs the signal above a HIGH threshold;
   leaving it needs the signal below a separate LOW threshold. Between the two, the state
   holds. One noisy dip below the mean cannot flip a held grasp.
2. **Minimum dwell**: after gating, any state island shorter than `min_dwell` frames is
   absorbed into its neighbours. This removes the last single-frame blips the Schmitt trigger
   alone leaves at genuine transitions.

The grasp signal itself is a VISION proxy -- finger curl from WiLoR keypoints, optionally
gated by whether the hand is near a detected object. It is honestly weak (vision cannot feel
contact), which is why L2 stamps grasp `vision_fallback` on a bare-hand rig and keeps per-finger
contact confidence low. See fusion.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from actuate.config import InteractionState

# MediaPipe/MANO 21-keypoint topology (WiLoR uses it). Per finger: (mcp, pip, dip, tip).
_FINGERS = {
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}


def finger_curl(keypoints_3d: np.ndarray) -> dict[str, float]:
    """Per-finger curl in [0, 1): 1 - chord/path over the finger's phalanges.

    A straight finger has chord (tip-to-mcp straight line) ~= path (sum of segment lengths),
    so curl ~ 0. A curled finger's chord is much shorter than the path, so curl -> ~0.6-0.8.
    Scale-invariant (a ratio), so it does not care about hand size or the metric ambiguity in
    Part C's depth. Thumb is excluded -- its kinematics differ and it is not load-bearing for a
    power/precision grasp signal.
    """
    kp = np.asarray(keypoints_3d, dtype=np.float64)
    out: dict[str, float] = {}
    for name, (mcp, pip, dip, tip) in _FINGERS.items():
        path = (
            np.linalg.norm(kp[pip] - kp[mcp])
            + np.linalg.norm(kp[dip] - kp[pip])
            + np.linalg.norm(kp[tip] - kp[dip])
        )
        chord = np.linalg.norm(kp[tip] - kp[mcp])
        out[name] = float(max(0.0, 1.0 - chord / path)) if path > 1e-9 else 0.0
    return out


def grasp_signal(keypoints_3d: np.ndarray) -> float:
    """Scalar grasp proxy in [0, 1): mean finger curl. Higher == more closed."""
    curls = finger_curl(keypoints_3d)
    return float(np.mean(list(curls.values()))) if curls else 0.0


@dataclass
class SchmittTrigger:
    """Hysteresis gate. `high` to switch ON, `low` to switch OFF; holds in between."""

    high: float
    low: float
    state: bool = False

    def __post_init__(self) -> None:
        if self.low > self.high:
            raise ValueError(f"Schmitt low ({self.low}) must be <= high ({self.high})")

    def update(self, value: float) -> bool:
        if self.state and value < self.low:
            self.state = False
        elif not self.state and value >= self.high:
            self.state = True
        return self.state


def enforce_min_dwell(states: list, min_dwell: int) -> list:
    """Absorb any INTERIOR run shorter than `min_dwell` into the preceding run's value.

    This is the final de-flicker: after hysteresis, genuine transitions can still leave a
    1-2 frame island (the signal crossed both thresholds briefly). An interior short run --
    one with a run on BOTH sides -- is a flicker and is merged into its predecessor. The
    first and last runs are left alone: a short run at a boundary may be a real state that the
    frame window simply started or ended inside, not a flicker (which by definition is
    surrounded). Iterated to a fixed point so merges that expose new short runs also settle.
    """
    if not states:
        return states
    out = list(states)
    n = len(out)

    def runs(seq):
        bounds = []
        i = 0
        while i < n:
            j = i
            while j < n and seq[j] == seq[i]:
                j += 1
            bounds.append((i, j))
            i = j
        return bounds

    changed = True
    while changed:
        changed = False
        rs = runs(out)
        for idx, (a, b) in enumerate(rs):
            interior = 0 < idx < len(rs) - 1
            if interior and (b - a) < min_dwell:
                prev_val = out[rs[idx - 1][0]]
                out[a:b] = [prev_val] * (b - a)
                changed = True
                break  # re-scan from scratch after a merge
    return out


@dataclass
class HandGraspGate:
    """Per-hand Schmitt-gated grasp, optionally requiring object proximity."""

    trigger: SchmittTrigger
    require_object: bool = False

    def step(self, signal: float, near_object: bool) -> bool:
        grasped = self.trigger.update(signal)
        return grasped and (near_object or not self.require_object)


@dataclass
class StateClassifier:
    """Combines gated left/right grasp + wrist motion into one InteractionState stream.

    Priority when both hands grasp: GRASPED_BOTH. A single hand: GRASPED_L / GRASPED_R.
    Neither grasping: MOVING if the wrist is translating in-image faster than `motion_px`,
    else STATIC. Motion is measured in 2D pixels on purpose -- it does not depend on the
    metric depth that Part C showed is unreliable for the hand.
    """

    #: Enter/leave grasp on the mean-finger-curl signal. Calibrated on the real capture: an
    #: open hand over paperwork reads ~0.10 (max 0.15), a closed fist reads > 0.4, so 0.35/0.22
    #: sits cleanly between them. NOTE this is a POWER-grasp proxy -- a pinch grasp (thumb-index
    #: only) curls the other fingers little and can read below `high`; such grasps are
    #: under-detected, which is one more reason grasp is stamped vision_fallback on a bare-hand
    #: rig. On session_001 the hand never power-grasps (signal maxes at 0.15), so no GRASPED
    #: state fires -- that is the honest result for this footage, not a miscalibration.
    grasp_high: float = 0.35
    grasp_low: float = 0.22
    require_object: bool = False
    motion_px: float = 6.0
    min_dwell: int = 3

    _left: HandGraspGate = field(init=False)
    _right: HandGraspGate = field(init=False)

    def __post_init__(self) -> None:
        self._left = HandGraspGate(
            SchmittTrigger(self.grasp_high, self.grasp_low), self.require_object
        )
        self._right = HandGraspGate(
            SchmittTrigger(self.grasp_high, self.grasp_low), self.require_object
        )

    def run(
        self,
        left_signal: list[float | None],
        right_signal: list[float | None],
        left_near: list[bool],
        right_near: list[bool],
        wrist_xy: list[tuple[float, float] | None],
    ) -> list[InteractionState]:
        n = len(wrist_xy)
        raw: list[InteractionState] = []
        prev_xy: tuple[float, float] | None = None
        for i in range(n):
            lg = self._left.step(left_signal[i] or 0.0, left_near[i]) if left_signal[i] is not None else False
            rg = self._right.step(right_signal[i] or 0.0, right_near[i]) if right_signal[i] is not None else False

            xy = wrist_xy[i]
            moving = False
            if xy is not None and prev_xy is not None:
                moving = float(np.hypot(xy[0] - prev_xy[0], xy[1] - prev_xy[1])) > self.motion_px
            if xy is not None:
                prev_xy = xy

            if lg and rg:
                raw.append(InteractionState.GRASPED_BOTH)
            elif lg:
                raw.append(InteractionState.GRASPED_L)
            elif rg:
                raw.append(InteractionState.GRASPED_R)
            elif moving:
                raw.append(InteractionState.MOVING)
            else:
                raw.append(InteractionState.STATIC)
        return enforce_min_dwell(raw, self.min_dwell)
