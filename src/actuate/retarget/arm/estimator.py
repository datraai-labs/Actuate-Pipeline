"""Root-frame estimator: Vector-Neuron encoder + flow-matching (Master Spec §L5, EgoInfinity).

Predicts the robot BASE (root) frame from a wrist trajectory + gravity, in the camera frame.
Two design choices from the spec, both real here (not decoration):

**Vector Neurons (Deng et al.)** make the network exactly SO(3)-equivariant. A VN layer operates
on tensors of 3-vectors `(B, C, 3)`; rotating the input rotates the output. So if the camera
rotates, the predicted root frame rotates with it -- the network cannot learn a spurious
world-up bias, which a plain MLP on flattened coordinates would.

**Flow matching (not regression)** because the map wrist->root is one-to-many: the same wrist
motion is consistent with many base placements. A deterministic regressor would average them
into a mush; flow matching learns the whole conditional distribution, so we can SAMPLE several
candidate roots and score them (candidates.py). Isotropic-Gaussian noise is rotation-invariant,
so an equivariant velocity field integrated from it yields an equivariant sample distribution --
VN + flow matching is equivariant end to end.

The root pose is carried as (root_pos, rot_col1, rot_col2) = a (3, 3) tensor of 3-vectors; the
6D rotation is Gram-Schmidt'd back to SO(3) at the end (simdata.rot6d_to_matrix).

**Compute:** the encoder + flow head are small (fits 4 GB, CPU-trainable for a smoke run). Full
training (~1.5-2 hrs, more data/steps) is a Kaggle T4 job.
"""

from __future__ import annotations

import numpy as np


def _torch():
    import torch

    return torch


# --------------------------------------------------------------------------------------
# Vector-Neuron layers  (operate on (B, C, 3))
# --------------------------------------------------------------------------------------


def _make_layers():
    import torch
    import torch.nn as nn

    class VNLinear(nn.Module):
        """Rotation-equivariant linear map over channels: (B, C_in, 3) -> (B, C_out, 3)."""

        def __init__(self, c_in, c_out):
            super().__init__()
            self.w = nn.Linear(c_in, c_out, bias=False)

        def forward(self, x):  # x: (B, C_in, 3)
            return self.w(x.transpose(-2, -1)).transpose(-2, -1)

    class VNLeakyReLU(nn.Module):
        """Equivariant nonlinearity: cut the component along a learned direction if it points
        'negative', per channel. Preserves equivariance (uses only inner products + the vector)."""

        def __init__(self, c, slope=0.2):
            super().__init__()
            self.dir = VNLinear(c, c)
            self.slope = slope

        def forward(self, x):  # (B, C, 3)
            d = self.dir(x)
            dn = d / (d.norm(dim=-1, keepdim=True) + 1e-6)
            dot = (x * dn).sum(-1, keepdim=True)
            below = (dot < 0).float()
            # x_perp = x - dot*dn ; leaky: keep slope of the parallel part when below
            return x - (1 - self.slope) * below * dot * dn

    return torch, nn, VNLinear, VNLeakyReLU


# --------------------------------------------------------------------------------------
# The estimator
# --------------------------------------------------------------------------------------


class RootFrameEstimator:
    """Wraps a torch module + the flow-matching train/sample logic. Lazy torch import."""

    def __init__(self, hidden: int = 64, device: str = "cpu") -> None:
        torch, nn, VNLinear, VNLeakyReLU = _make_layers()
        self._torch = torch
        self.device = device
        self.hidden = hidden

        class Net(nn.Module):
            def __init__(self, H):
                super().__init__()
                # encoder: per-frame [wrist_pos, rot_c1, rot_c2, gravity] = 4 vectors -> H
                self.enc1 = VNLinear(4, H)
                self.enc_act = VNLeakyReLU(H)
                self.enc2 = VNLinear(H, H)
                # flow velocity field: [X_t (3) ++ cond (H)] -> H -> 3
                self.f1 = VNLinear(3 + H, H)
                self.f_act = VNLeakyReLU(H)
                self.f2 = VNLinear(H, 3)
                # t injected as invariant per-channel scales (scalar * vector = equivariant)
                self.t_mlp = nn.Sequential(nn.Linear(3, H), nn.SiLU(), nn.Linear(H, H))

            def encode(self, feats):           # feats: (B, T, 4, 3)
                B, T = feats.shape[:2]
                h = self.enc2(self.enc_act(self.enc1(feats.reshape(B * T, 4, 3))))
                return h.reshape(B, T, -1, 3).mean(1)   # temporal mean -> (B, H, 3)

            def velocity(self, x_t, cond, t):  # x_t:(B,3,3) cond:(B,H,3) t:(B,)
                temb = torch.stack([t, torch.sin(t), torch.cos(t)], -1)     # (B,3)
                scales = self.t_mlp(temb).unsqueeze(-1)                     # (B,H,1)
                h = self.f_act(self.f1(torch.cat([x_t, cond], dim=1)))      # (B,H,3)
                h = h * scales                                             # t-modulation
                return self.f2(h)                                          # (B,3,3)

        self.net = Net(hidden).to(device)

    # -- feature assembly ---------------------------------------------------------------
    def _features(self, wrist_pos, wrist_rot6d, gravity):
        """(T,3),(T,6),(3,) -> centered per-frame (T,4,3) feats + the wrist centroid (3,)."""
        torch = self._torch
        wp = torch.as_tensor(wrist_pos, dtype=torch.float32)
        centroid = wp.mean(0)
        wp = wp - centroid
        r = torch.as_tensor(wrist_rot6d, dtype=torch.float32).reshape(-1, 2, 3)
        g = torch.as_tensor(gravity, dtype=torch.float32).reshape(1, 1, 3).expand(wp.shape[0], 1, 3)
        feats = torch.cat([wp.unsqueeze(1), r, g], dim=1)  # (T,4,3)
        return feats, centroid

    def _batch(self, pairs):
        torch = self._torch
        feats, centroids, targets = [], [], []
        for p in pairs:
            f, c = self._features(p.wrist_pos_cam, p.wrist_rot6d_cam, p.gravity_cam)
            feats.append(f)
            centroids.append(c)
            # target root pose, position relative to the wrist centroid
            rp = torch.as_tensor(p.root_pos_cam, dtype=torch.float32) - c
            rr = torch.as_tensor(p.root_rot6d_cam, dtype=torch.float32).reshape(2, 3)
            targets.append(torch.cat([rp.unsqueeze(0), rr], dim=0))  # (3,3)
        return (torch.stack(feats).to(self.device),
                torch.stack(centroids).to(self.device),
                torch.stack(targets).to(self.device))

    # -- training (conditional flow matching) -------------------------------------------
    def train(self, pairs, epochs=200, batch_size=64, lr=1e-3, seed=0, log_every=50):
        torch = self._torch
        torch.manual_seed(seed)
        opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        feats_all, cent_all, tgt_all = self._batch(pairs)
        n = len(pairs)
        self.net.train()
        losses = []
        for ep in range(epochs):
            idx = torch.randperm(n)[:batch_size]
            feats, tgt = feats_all[idx], tgt_all[idx]
            cond = self.net.encode(feats)
            x1 = tgt
            x0 = torch.randn_like(x1)
            t = torch.rand(len(idx), device=self.device)
            x_t = (1 - t)[:, None, None] * x0 + t[:, None, None] * x1
            v_target = x1 - x0
            v_pred = self.net.velocity(x_t, cond, t)
            loss = ((v_pred - v_target) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
            if log_every and (ep % log_every == 0 or ep == epochs - 1):
                print(f"  epoch {ep:4d}  flow-matching loss {loss.item():.4f}")
        return losses

    # -- sampling -----------------------------------------------------------------------
    @staticmethod
    def _decode(x, centroid):
        """(3,3) sample + centroid -> (root_pos (3), root_R (3x3))."""
        from actuate.retarget.arm.simdata import rot6d_to_matrix

        pos = x[0] + centroid
        R = rot6d_to_matrix(np.concatenate([x[1], x[2]]))
        return pos, R

    def sample(self, wrist_pos, wrist_rot6d, gravity, n_samples=16, steps=20, seed=0):
        """Sample `n_samples` candidate root frames by integrating the flow ODE from noise.

        Returns a list of (root_pos (3,), root_R (3x3)). Multiple distinct candidates is the
        point -- the map is one-to-many and candidates.py scores them by IK.
        """
        torch = self._torch
        self.net.eval()
        feats, centroid = self._features(wrist_pos, wrist_rot6d, gravity)
        feats = feats.unsqueeze(0).to(self.device)
        with torch.no_grad():
            cond = self.net.encode(feats).expand(n_samples, -1, -1)
            g = torch.Generator(device=self.device).manual_seed(seed)
            x = torch.randn(n_samples, 3, 3, generator=g, device=self.device)
            dt = 1.0 / steps
            for i in range(steps):
                t = torch.full((n_samples,), i * dt, device=self.device)
                x = x + dt * self.net.velocity(x, cond, t)
        cen = centroid.cpu().numpy()
        return [self._decode(x[i].cpu().numpy(), cen) for i in range(n_samples)]

    # -- persistence --------------------------------------------------------------------
    def save(self, path):
        self._torch.save({"hidden": self.hidden, "state": self.net.state_dict()}, str(path))

    @classmethod
    def load(cls, path, device="cpu"):
        import torch

        ckpt = torch.load(str(path), map_location=device)
        est = cls(hidden=ckpt["hidden"], device=device)
        est.net.load_state_dict(ckpt["state"])
        return est
