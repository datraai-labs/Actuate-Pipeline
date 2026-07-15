"""L2 fusion -- `fusion.run(store) -> FusionReport`. Master Spec §L2.

Ties the arbiter and the state machine to real perception. On the head-mounted bare-hand rig
there is NO hardware sensor, so every channel resolves to vision, and the arbiter correctly
stamps grasp `vision_fallback`. That is not a degraded mode to apologise for -- it is the
honest answer for that rig, and the reason per-finger contact confidence stays low.

When a higher-trust source IS present (a glove's measured_human joints, a DexUMI's
measured_robotspace encoders), the arbiter picks it over vision automatically -- the same code
path, a different set of candidates. That is what makes this a fusion layer and not a vision
post-process.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from actuate.config import Finger, InteractionState, Provenance, RigType, Side
from actuate.config.enums import Channel
from actuate.config.rigs import RIG_REGISTRY
from actuate.fusion.arbiter import Candidate, resolve_channel
from actuate.fusion.states import StateClassifier, finger_curl, grasp_signal
from actuate.perception.objects.rle import bbox_iou

#: Vision cannot feel contact; it infers it from curl + proximity. Cap the confidence so a
#: vision contact reading can never masquerade as a confident tactile one.
_CONTACT_VISION_CAP = 0.40
#: Hand-to-object bbox IoU above which we call the hand "near an object" (2D, depth-free).
_NEAR_OBJECT_IOU = 0.05


@dataclass
class HardwareSources:
    """Optional higher-trust readings, per frame. Absent on a bare-hand rig.

    `finger_joints[frame][side]` -> per-finger angle vector (measured_human / robotspace).
    `contact[frame][side][finger]` -> confidence in [0,1]. Presence of these is what lets the
    arbiter override vision; a glove supplies them at MEASURED_HUMAN, a DexUMI at
    MEASURED_ROBOTSPACE.
    """

    provenance: Provenance
    finger_joints: dict[int, dict[Side, tuple[float, ...]]] = field(default_factory=dict)
    contact: dict[int, dict[Side, dict[Finger, float]]] = field(default_factory=dict)


@dataclass
class ContactPoint:
    confidence: float
    source: Provenance


@dataclass
class FrameFusion:
    interaction_state: InteractionState
    grasp: dict[Side, float]                       # arbitrated grasp scalar per detected hand
    grasp_provenance: Provenance
    contact: dict[Side, dict[Finger, ContactPoint]]
    finger_joints_provenance: dict[Side, Provenance]


@dataclass
class FusionReport:
    rig: RigType
    states: list[InteractionState] = field(default_factory=list)  # index == frame order
    frames: dict[int, FrameFusion] = field(default_factory=dict)
    provenance: dict[str, Provenance] = field(default_factory=dict)
    notes: dict[str, str] = field(default_factory=dict)

    def flicker_count(self) -> int:
        """Single-frame state islands remaining after gating. The gate wants this at 0."""
        s = self.states
        return sum(
            1 for i in range(1, len(s) - 1) if s[i] != s[i - 1] and s[i] != s[i + 1]
        )


def _near_object(hand_bbox, object_frames) -> bool:
    """Is this hand's box overlapping a NON-hand detected object (2D, depth-free)?"""
    for o in object_frames or []:
        if "hand" in o.label.lower():
            continue
        if bbox_iou(tuple(hand_bbox), tuple(o.bbox)) >= _NEAR_OBJECT_IOU:
            return True
    return False


def run(
    hands,                                   # HandResult from perception.hands.run
    *,
    rig: RigType,
    objects=None,                            # ObjectResult from perception.objects.run
    hardware: HardwareSources | None = None,
    n_frames: int | None = None,
    grasp_high: float = 0.35,
    grasp_low: float = 0.22,
) -> FusionReport:
    """Fuse per-frame perception into gated interaction states + arbitrated contact.

    `hardware` is the seam for measured sources: when present its readings enter the arbiter as
    higher-trust candidates and win over vision. On a bare-hand rig it is None and everything
    resolves to vision_fallback -- which is the correct, honest result, not a failure.
    """
    rig_entry = RIG_REGISTRY[rig]
    require_object = False  # bare-hand: many grasps happen over paperwork; don't force it

    frame_ids = sorted(hands.frames) if hasattr(hands, "frames") else sorted(hands)
    if n_frames is not None:
        frame_ids = frame_ids[:n_frames]

    # --- build per-frame vision signals -------------------------------------------------
    left_sig: list[float | None] = []
    right_sig: list[float | None] = []
    left_near: list[bool] = []
    right_near: list[bool] = []
    wrist_xy: list[tuple[float, float] | None] = []
    per_frame_hands: dict[int, dict[Side, object]] = {}

    for fid in frame_ids:
        hs = hands.frames[fid]
        by_side = {h.side: h for h in hs}
        per_frame_hands[fid] = by_side
        objf = objects.frames.get(fid) if objects is not None else None

        def sig(side):
            h = by_side.get(side)
            return grasp_signal(h.keypoints_3d) if h is not None else None

        def near(side):
            h = by_side.get(side)
            return _near_object(h.bbox, objf) if h is not None else False

        left_sig.append(sig(Side.LEFT))
        right_sig.append(sig(Side.RIGHT))
        left_near.append(near(Side.LEFT))
        right_near.append(near(Side.RIGHT))

        # wrist 2D for motion: prefer the right hand, else left.
        dom = by_side.get(Side.RIGHT) or by_side.get(Side.LEFT)
        wrist_xy.append(tuple(dom.keypoints_2d[0]) if dom is not None else None)

    clf = StateClassifier(
        grasp_high=grasp_high, grasp_low=grasp_low, require_object=require_object
    )
    states = clf.run(left_sig, right_sig, left_near, right_near, wrist_xy)

    # --- per-frame arbitration: grasp, contact, finger_joints ---------------------------
    report = FusionReport(rig=rig, states=states)
    for k, fid in enumerate(frame_ids):
        by_side = per_frame_hands[fid]
        objf = objects.frames.get(fid) if objects is not None else None

        grasp: dict[Side, float] = {}
        grasp_prov = Provenance.VISION_FALLBACK
        contact: dict[Side, dict[Finger, ContactPoint]] = {}
        fj_prov: dict[Side, Provenance] = {}

        for side, h in by_side.items():
            # GRASP channel: the scalar is vision-derived on this rig; the arbiter decides the
            # PROVENANCE (a glove/dexumi measuring grasp would flip it above vision_fallback).
            vision_grasp = grasp_signal(h.keypoints_3d)
            cands = [Candidate("grasp", vision_grasp, Provenance.VISION_FALLBACK, confidence=0.5)]
            hw_c = (hardware.contact.get(fid, {}).get(side) if hardware else None)
            hw_grasp = (hardware.finger_joints.get(fid, {}).get(side) if hardware else None)
            if hw_grasp is not None:
                # derive a hardware grasp scalar from measured joint angles (mean, normalised
                # to [0,1] against a ~90deg full-flex assumption) so value and provenance agree.
                hw_val = float(np.clip(np.mean(np.abs(hw_grasp)) / (np.pi / 2), 0.0, 1.0))
                cands.append(Candidate("grasp", hw_val, hardware.provenance, confidence=1.0))
            win = resolve_channel("grasp", cands)
            grasp[side] = float(win.value)
            grasp_prov = win.provenance

            # FINGER_JOINTS channel: glove/dexumi override vision when present.
            fj_cands = [Candidate("fj", None, Provenance.VISION_PRIMARY, 0.6)]
            if hardware and side in hardware.finger_joints.get(fid, {}):
                fj_cands.append(Candidate("fj", None, hardware.provenance, 1.0))
            fj_win = resolve_channel("fj", fj_cands)
            fj_prov[side] = fj_win.provenance if fj_win else Provenance.VISION_PRIMARY

            # CONTACT channel: per finger, arbitrated.
            near = _near_object(h.bbox, objf)
            curls = finger_curl(h.keypoints_3d)
            contact[side] = {}
            for finger in Finger:
                # vision proxy: curl (thumb has no curl entry -> reuse index as a stand-in),
                # scaled by proximity, capped LOW because vision cannot feel contact.
                cname = {Finger.THUMB: "index", Finger.INDEX: "index", Finger.MIDDLE: "middle",
                         Finger.RING: "ring", Finger.PINKY: "pinky"}[finger]
                v = curls.get(cname, 0.0) * (1.0 if near else 0.5)
                v = float(min(_CONTACT_VISION_CAP, max(0.0, v)))
                c_cands = [Candidate("contact", v, Provenance.VISION_FALLBACK, confidence=v)]
                if hw_c and finger in hw_c:
                    c_cands.append(
                        Candidate("contact", float(hw_c[finger]), hardware.provenance,
                                  confidence=1.0)
                    )
                cw = resolve_channel("contact", c_cands)
                contact[side][finger] = ContactPoint(
                    confidence=float(cw.value), source=cw.provenance
                )

        report.frames[fid] = FrameFusion(
            interaction_state=states[k],
            grasp=grasp,
            grasp_provenance=grasp_prov,
            contact=contact,
            finger_joints_provenance=fj_prov,
        )

    # --- report-level provenance + honesty notes ----------------------------------------
    has_hw = hardware is not None
    report.provenance = {
        "interaction_state": (hardware.provenance if has_hw else Provenance.VISION_FALLBACK),
        "grasp": (hardware.provenance if has_hw else Provenance.VISION_FALLBACK),
        "contact": (hardware.provenance if has_hw else Provenance.VISION_FALLBACK),
    }
    # Enforce the rig ceiling: a rig that measures nothing may NOT claim a measured provenance.
    ceiling = rig_entry.measured_provenance_ceiling
    if not has_hw or not rig_entry.measures(Channel.GRASP):
        report.provenance["grasp"] = Provenance.VISION_FALLBACK
        report.provenance["contact"] = Provenance.VISION_FALLBACK
    report.notes = {
        "mode": (
            "VISION-ONLY: this rig has no grasp/contact hardware, so grasp and contact are "
            "vision_fallback and per-finger contact confidence is capped low (<= "
            f"{_CONTACT_VISION_CAP}). This is the correct answer for a bare-hand rig, not a "
            "degraded one."
            if not has_hw
            else f"HARDWARE PRESENT ({hardware.provenance.value}): measured readings override "
            "vision for the channels the rig declares measured."
        ),
        "rig_ceiling": str(ceiling.value if ceiling else None),
    }
    return report
