"""FoundationPose -- zero-shot 6-DoF object pose (Master Spec §L1). INTERFACE ONLY.

FoundationPose gives full 6-DoF pose for objects with **known geometry** (a CAD mesh). Two
hard reasons it does not run in this environment, both worth stating plainly rather than
discovering at deploy:

1. **It needs a mesh we do not have.** FoundationPose is model-based: it renders the object's
   CAD mesh from hypothesised poses and scores against the observed crop + depth. The real
   capture shows paperwork and a stapler -- we have no mesh for either, so the algorithm's
   core input is missing. Without geometry there is no 6-DoF to estimate; the most you can
   recover is a 3D POSITION by back-projecting the mask through metric depth (which
   `perception.objects.run` does, via Part C's depth + intrinsics), not an orientation.

2. **It does not fit in 4 GB, and needs CUDA ops that are not installed.** FoundationPose
   depends on nvdiffrast (custom CUDA rasteriser) and runs a render-and-compare loop over
   hundreds of pose hypotheses; reference runs use >8 GB. On the RTX 2050 (4.29 GB) it will
   OOM. It belongs on a T4/A100 (Kaggle/cloud), and only once a mesh exists.

So this file is the **interface reservation**: the type an L1 objects pass would call if a
mesh and a bigger GPU were present, plus a synthetic-input smoke test that exercises the
interface shape without the model. It raises `FoundationPoseUnavailable` when actually asked
to run, naming the missing piece. This is the same discipline as the SLAM ORB-SLAM3 stub and
the depth FoundationStereo stub: reserve the seam, refuse to fake the capability.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class FoundationPoseUnavailable(NotImplementedError):
    """Raised when 6-DoF pose is requested but cannot be produced (no mesh, or won't fit)."""


@dataclass
class PoseHypothesis:
    """What a real FoundationPose call would return per object per frame."""

    #: 4x4 object->camera transform. Rotation is the part only a mesh-based method can give.
    T_obj_cam: np.ndarray
    score: float


class FoundationPoseEstimator:
    """Interface for zero-shot 6-DoF pose. Not runnable here -- see module docstring.

    A real implementation takes a CAD mesh, an RGB crop, a metric depth crop, and the camera
    intrinsics, and returns a `PoseHypothesis`. This class fixes that signature so the rest of
    the pipeline can be written against it, and fails loudly (not silently) when invoked.
    """

    def __init__(self, mesh_path: str | None = None, device: str = "cuda") -> None:
        self.mesh_path = mesh_path
        self.device = device

    def estimate(
        self,
        rgb: np.ndarray,          # (H, W, 3) object crop
        depth_m: np.ndarray,      # (H, W) metric depth crop, from Part C
        mask: np.ndarray,         # (H, W) boolean object mask
        intrinsics: np.ndarray,   # (3, 3) camera K
    ) -> PoseHypothesis:
        if self.mesh_path is None:
            raise FoundationPoseUnavailable(
                "FoundationPose needs a CAD mesh for the object; none was provided. The real "
                "capture (paperwork, stapler) has no mesh, so 6-DoF ORIENTATION is not "
                "recoverable. Use the mask + metric-depth back-projection for 3D POSITION "
                "instead (perception.objects.run does this). See Master Spec §L1."
            )
        raise FoundationPoseUnavailable(
            "FoundationPose is not installed/runnable here: it needs nvdiffrast (custom CUDA) "
            "and >8 GB VRAM for its render-and-compare loop; the RTX 2050 has 4.29 GB and will "
            "OOM. Run on a T4/A100. Interface reserved; mesh path was "
            f"{self.mesh_path!r}."
        )


def synthetic_interface_check() -> dict:
    """Exercise the interface shape without the model -- the smoke test the user asked for.

    Confirms: (a) the estimator instantiates, (b) it raises FoundationPoseUnavailable with the
    right message when called without a mesh, and (c) a PoseHypothesis round-trips the expected
    array shapes. This is a shape/contract test, NOT a claim the model runs.
    """
    est = FoundationPoseEstimator(mesh_path=None)
    H = W = 32
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    depth = np.full((H, W), 0.5, dtype=np.float64)
    mask = np.zeros((H, W), dtype=bool)
    mask[8:24, 8:24] = True
    K = np.array([[600, 0, W / 2], [0, 600, H / 2], [0, 0, 1]], dtype=np.float64)

    raised = None
    try:
        est.estimate(rgb, depth, mask, K)
    except FoundationPoseUnavailable as exc:
        raised = str(exc)

    # And confirm the return type is well-formed when a hypothesis IS constructed.
    hyp = PoseHypothesis(T_obj_cam=np.eye(4), score=0.0)
    ok_shape = hyp.T_obj_cam.shape == (4, 4)

    return {
        "instantiates": True,
        "raises_without_mesh": raised is not None,
        "message_names_the_gap": raised is not None and "mesh" in raised.lower(),
        "hypothesis_shape_ok": ok_shape,
    }
