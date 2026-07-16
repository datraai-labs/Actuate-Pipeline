"""GeoRT-style geometric finger retargeting (Master Spec §L5).

A fast, unsupervised, **contact-blind kinematic** map from human fingertip keypoints to a robot
hand's joint configuration. Per the GeoRT paper: minutes to train, a handful of hyperparameters,
assumes an anthropomorphic hand + finger correspondence.

### How it works here

There is no paired (human pose, robot pose) supervision -- nobody labels "this human fist equals
that Allegro config". So the map is assembled from two unsupervised halves:

1. **Robot C-space samples** (`hand.sample_cspace`): random in-limit configs -> forward
   kinematics -> fingertip positions in the palm frame. These (fingertips -> joints) pairs train
   an MLP that is, in effect, the hand's learned fingertip IK.
2. **Per-human calibration**: the human's fingertip cloud (from ~minutes of free finger motion)
   is aligned to the robot's fingertip cloud by a per-finger affine (centre + scale). This is the
   "geometric" part -- it absorbs hand-size and proportion differences so a human fingertip lands
   somewhere the robot can actually reach.

At inference: human fingertips -> per-finger affine -> MLP -> joints, clamped to limits.

**Honest scope.** This is kinematic and contact-blind by construction: it matches fingertip
GEOMETRY, it does not reason about grasp forces or contact. It is validated on sim + synthetic
MANO; **no real dexterous hand has been driven by it** (we have none), so real-hand validation is
deferred. And Allegro has no pinky -- the human pinky is dropped (see hand.py).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Calibration:
    """Similarity transform (rotation, uniform scale, translation) human frame -> robot frame.

    ### Why a rotation is required, not just per-finger scale+offset

    An earlier version aligned each fingertip's cloud by mean + scale only. That CANNOT fix a
    rotational misalignment between the human wrist frame and the robot palm frame -- and it
    measurably inverted the mapping: a human fist retargeted to a MORE extended robot hand than a
    human open hand (openness 0.110 vs 0.102 m -- backwards). Curl direction is a rotation, so
    the calibration must carry one.
    """

    R: np.ndarray            # (3, 3) rotation
    scale: float             # uniform scale (hand-size ratio)
    t: np.ndarray            # (3,) translation

    def apply(self, human_tips: np.ndarray) -> np.ndarray:
        """(n_tips,3) wrist-relative human fingertips -> the robot's palm-frame fingertips."""
        return self.scale * (np.asarray(human_tips) @ self.R.T) + self.t

    def to_dict(self):
        return {"R": self.R.tolist(), "scale": float(self.scale), "t": self.t.tolist()}

    @staticmethod
    def from_dict(d):
        return Calibration(R=np.asarray(d["R"]), scale=float(d["scale"]), t=np.asarray(d["t"]))


def _umeyama(src: np.ndarray, dst: np.ndarray):
    """Least-squares similarity transform (R, s, t) with s*R@src + t ~= dst. Kabsch + scale."""
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S, D = src - mu_s, dst - mu_d
    C = D.T @ S / len(src)
    U, sig, Vt = np.linalg.svd(C)
    d = np.sign(np.linalg.det(U @ Vt))
    W = np.diag([1.0, 1.0, d])
    R = U @ W @ Vt
    var_s = (S ** 2).sum() / len(src)
    scale = float((np.diag(np.diag(W) * sig)).trace() / max(var_s, 1e-12))
    t = mu_d - scale * (R @ mu_s)
    return R, scale, t


def fit_calibration(human_canonical: dict, robot_canonical: dict) -> Calibration:
    """Fit the human->robot similarity transform from CANONICAL POSE correspondences.

    `human_canonical` / `robot_canonical`: {pose_name: (n_tips, 3)} for the same pose names --
    e.g. {"open": ..., "fist": ...}. This is the calibration a human actually performs ("open
    your hand", "make a fist"), and pairing those poses is what pins the curl DIRECTION, which a
    statistics-only alignment cannot do.
    """
    keys = sorted(set(human_canonical) & set(robot_canonical))
    if not keys:
        raise ValueError("human and robot canonical poses share no pose names")
    H = np.concatenate([np.asarray(human_canonical[k]) for k in keys])
    Rb = np.concatenate([np.asarray(robot_canonical[k]) for k in keys])
    R, s, t = _umeyama(H, Rb)
    return Calibration(R=R, scale=s, t=t)


class GeoRT:
    """Fingertips -> joints, trained by GEOMETRIC RECONSTRUCTION. Lazy torch import.

    ### Why not regress joints directly

    The fingertip->joint map is **one-to-many**: Allegro has 16 DoF but fingertips impose only
    12 constraints, so many configs reach the same fingertips. Regressing joints against sampled
    (tips, joints) pairs averages those solutions into mush -- measured: the loss plateaus around
    0.14 rad^2 (~0.37 rad/joint) and every input pose collapses to nearly the same output. It is
    the same ill-posedness that forced flow-matching in the arm branch.

    So instead, two stages -- this is what makes it *geometric* retargeting:

    1. **Forward model** F: joints -> fingertips. Deterministic FK, so it is well-posed and
       trains cleanly. It exists to be DIFFERENTIABLE (MuJoCo's FK is not, from Python).
    2. **Inverse** G: fingertips -> joints, trained through the frozen F with a RECONSTRUCTION
       loss ||F(G(tips)) - tips||^2. Redundancy stops being a problem: any config that reaches
       the fingertips is accepted, so no averaging. A small mid-range prior breaks the remaining
       tie toward natural, non-extreme postures.

    G's output is sigmoid-squashed into the joint range, so predictions are ALWAYS within limits
    by construction -- not clipped after the fact.
    """

    def __init__(self, n_tips: int, n_joints: int, joint_limits: np.ndarray,
                 hidden: int = 128, device: str = "cpu") -> None:
        import torch
        import torch.nn as nn

        self._torch = torch
        self.device = device
        self.n_tips, self.n_joints = n_tips, n_joints
        self.joint_limits = np.asarray(joint_limits, dtype=np.float64)
        self.calibration: Calibration | None = None

        lo = torch.tensor(self.joint_limits[:, 0], dtype=torch.float32, device=device)
        hi = torch.tensor(self.joint_limits[:, 1], dtype=torch.float32, device=device)
        self._lo, self._hi = lo, hi

        class Inverse(nn.Module):
            def __init__(self):
                super().__init__()
                self.f = nn.Sequential(
                    nn.Linear(n_tips * 3, hidden), nn.SiLU(),
                    nn.Linear(hidden, hidden), nn.SiLU(),
                    nn.Linear(hidden, n_joints),
                )

            def forward(self, x):
                # sigmoid-squash into the joint range -> always within limits
                return lo + (hi - lo) * torch.sigmoid(self.f(x))

        class Forward(nn.Module):
            """Differentiable surrogate for the hand's FK: joints -> fingertips."""

            def __init__(self):
                super().__init__()
                self.f = nn.Sequential(
                    nn.Linear(n_joints, hidden), nn.SiLU(),
                    nn.Linear(hidden, hidden), nn.SiLU(),
                    nn.Linear(hidden, n_tips * 3),
                )

            def forward(self, q):
                return self.f(q)

        self.net = Inverse().to(device)
        self.fwd = Forward().to(device)

    # -- training -----------------------------------------------------------------------
    def train(self, robot_tips: np.ndarray, robot_joints: np.ndarray, *, epochs: int = 800,
              batch_size: int = 256, lr: float = 1e-3, seed: int = 0, log_every: int = 200,
              fwd_epochs: int | None = None, prior_weight: float = 1e-3):
        """Stage 1: fit the differentiable forward model. Stage 2: fit the inverse through it."""
        torch = self._torch
        torch.manual_seed(seed)
        X = torch.tensor(np.asarray(robot_tips).reshape(len(robot_tips), -1),
                         dtype=torch.float32, device=self.device)
        Y = torch.tensor(np.asarray(robot_joints), dtype=torch.float32, device=self.device)
        fwd_epochs = fwd_epochs if fwd_epochs is not None else epochs

        # --- stage 1: forward model (well-posed) ---
        optf = torch.optim.Adam(self.fwd.parameters(), lr=lr)
        self.fwd.train()
        for ep in range(fwd_epochs):
            idx = torch.randperm(len(X), device=self.device)[:batch_size]
            loss = ((self.fwd(Y[idx]) - X[idx]) ** 2).mean()
            optf.zero_grad()
            loss.backward()
            optf.step()
            if log_every and (ep % log_every == 0 or ep == fwd_epochs - 1):
                print(f"  [fwd] epoch {ep:4d}  fk-surrogate loss {loss.item():.6f}")
        for p in self.fwd.parameters():   # freeze: it is a fixed differentiable FK now
            p.requires_grad_(False)
        self.fwd.eval()

        # --- stage 2: inverse via geometric reconstruction through F ---
        q_mid = ((self._lo + self._hi) / 2).unsqueeze(0)
        opti = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.net.train()
        losses = []
        for ep in range(epochs):
            idx = torch.randperm(len(X), device=self.device)[:batch_size]
            tips = X[idx]
            q = self.net(tips)
            recon = ((self.fwd(q) - tips) ** 2).mean()
            prior = ((q - q_mid) ** 2).mean()          # prefer natural, non-extreme postures
            loss = recon + prior_weight * prior
            opti.zero_grad()
            loss.backward()
            opti.step()
            losses.append(recon.item())
            if log_every and (ep % log_every == 0 or ep == epochs - 1):
                print(f"  [inv] epoch {ep:4d}  fingertip recon {recon.item():.6f} m^2")
        return losses

    # -- inference ----------------------------------------------------------------------
    def predict(self, human_tips: np.ndarray) -> np.ndarray:
        """(n_tips,3) human fingertips -> (n_joints,) robot joints, within limits."""
        torch = self._torch
        tips = np.asarray(human_tips, dtype=np.float64)
        if self.calibration is not None:
            tips = self.calibration.apply(tips)
        self.net.eval()
        with torch.no_grad():
            x = torch.tensor(tips.reshape(1, -1), dtype=torch.float32, device=self.device)
            q = self.net(x).cpu().numpy()[0]
        return np.clip(q, self.joint_limits[:, 0], self.joint_limits[:, 1])

    # -- persistence --------------------------------------------------------------------
    def save(self, path):
        self._torch.save({
            "n_tips": self.n_tips, "n_joints": self.n_joints,
            "joint_limits": self.joint_limits,
            "calibration": self.calibration.to_dict() if self.calibration else None,
            "state": self.net.state_dict(),
            "fwd_state": self.fwd.state_dict(),
        }, str(path))

    @classmethod
    def load(cls, path, device="cpu"):
        import torch

        ck = torch.load(str(path), map_location=device, weights_only=False)
        m = cls(ck["n_tips"], ck["n_joints"], ck["joint_limits"], device=device)
        m.net.load_state_dict(ck["state"])
        if ck.get("fwd_state"):
            m.fwd.load_state_dict(ck["fwd_state"])
        if ck["calibration"]:
            m.calibration = Calibration.from_dict(ck["calibration"])
        return m
