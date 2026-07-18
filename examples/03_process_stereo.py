"""03 — a side-by-side stereo video. `rig=auto` detects stereo from the >1.9 aspect ratio,
which unlocks real (not monocular) depth downstream."""

import actuate

actuate.login()

run = actuate.process(
    source="./my_stereo_video.mp4",              # e.g. 2560x720 side-by-side
    rig="auto",                                  # -> 'stereo' when width ~2x height
    task="assemble the bracket",
    max_frames=45,
)
print(f"detected rig via geometry; quality={run.quality}/5")
run.export("lerobot_v3", path="./examples_out/03_stereo/")
