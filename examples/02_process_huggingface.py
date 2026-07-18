"""02 — a HuggingFace dataset -> processed. hf:// downloads + stages the raw video."""

import actuate

actuate.login()

# any public LeRobot dataset; pusht is small and free
run = actuate.process(
    source="hf://lerobot/pusht",
    rig="auto",                                  # detected from the video
    task="push the T block to the target",
    max_frames=30,
)
print(run.summary())
run.export("lerobot_v3", path="./examples_out/02_hf/")
