"""05 — process for a specific robot embodiment (needed for robot-space export/retargeting).

The embodiment defaults to your `actuate config` setting; override it per-run here."""

import actuate

actuate.login()

run = actuate.process(
    source="./processed/session_001/compressed.mp4",
    rig="head_mounted",
    embodiment="franka_panda",                   # the retarget/export target
    task="handle paperwork",
    max_frames=20,
)
# without a trained arm model the retarget stage skips (human-space only) -- honest.
print("stages:", run.summary()["stages"])
run.export("lerobot_v3", path="./examples_out/05_embodiment/", embodiment="franka_panda")
