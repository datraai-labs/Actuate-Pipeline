# GMR humanoid retargeting

Actuate has two retargeting paths with different contracts:

| Path | Input | Target | Command |
| --- | --- | --- | --- |
| Arm | Canonical egocentric wrist trajectory | Franka-style arm | `actuate retarget arm` |
| Humanoid/GMR | Full-body BVH skeleton motion | Supported humanoid | `actuate retarget humanoid` |

The humanoid path integrates the reference implementation for Araujo et al.,
["Retargeting Matters: General Motion Retargeting for Humanoid Motion
Tracking"](https://arxiv.org/abs/2510.02252), pinned to upstream commit
`bb1bbe40774794fceb2a7c579a3464a28e68c844`.

## Install

```bash
pip install -e '.[humanoid-retarget]'
actuate retarget setup-humanoid
```

The separate setup command is required because the upstream GMR `0.2.0` wheel omits its robot
MJCF/mesh assets and IK JSON package data. Actuate downloads the exact pinned commit into a
versioned cache under `ACTUATE_HOME` (normally `~/.actuate`) instead of depending on a temporary
Git checkout. Setup downloads only Unitree G1's MJCF, its referenced meshes, and the supported
BVH mappings—not the upstream repository's roughly 1.2 GB multi-robot asset directory.

## Quick local run

```bash
actuate retarget humanoid \
  --in motion.bvh \
  --source-format xsens \
  --robot unitree_g1 \
  --out out/gmr-run \
  --preview out/gmr-run/preview.mp4
```

For a fast smoke test, add `--max-frames 60`.

The output directory contains:

- `motion.npz`: numeric root pose and joint positions, with no executable pickle payload.
- `report.json`: method provenance, input/target metadata, ground correction, joint-limit,
  velocity-spike, and self-collision checks.
- the optional MP4/GIF preview, rendered off-screen without the MuJoCo desktop viewer.

## What is implemented

The official GMR backend owns the paper's five retargeting stages:

1. human/robot key-body matching;
2. Cartesian rest-pose alignment;
3. non-uniform local scaling;
4. rotation and endpoint differential IK;
5. rotation and translation fine-tuning.

Actuate processes frames sequentially so the previous robot pose warm-starts the next solve.
After the entire clip, it uses MuJoCo forward kinematics to find the minimum robot-body height
and subtracts that height from the root trajectory, matching the paper's global ground
post-process.

Velocity limiting is enabled by default. Actuate then reports non-finite output, joint-limit
violations, velocity spikes above `3*pi rad/s`, and robot self-collisions.

## Important boundary

An RGB egocentric video does not contain the full-body 3D skeleton required by GMR. The current
Actuate capture pipeline estimates hands, depth, objects, SLAM, and wrists, so it can feed the
arm retargeter but not the humanoid retargeter directly. A validated monocular full-body
reconstruction stage (for example, a GVHMR-to-SMPL-X adapter) is separate future work; this
integration does not invent that missing backend capability.
