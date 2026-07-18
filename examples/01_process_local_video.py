"""01 — single local video -> LeRobot v3. The Refiner-equivalent first pipeline."""

import actuate

actuate.login()                                  # local mode, one-time

run = actuate.process(
    source="./processed/session_001/compressed.mp4",
    rig="head_mounted",
    task="handle paperwork on a desk",
    max_frames=20,                               # cap for a fast first run
)
print(f"status={run.status}  quality={run.quality}/5  frames={run.num_frames}")

res = run.export("lerobot_v3", path="./examples_out/01_lerobot/")
print(f"exported {res.n_frames} frames -> {res.path}")
