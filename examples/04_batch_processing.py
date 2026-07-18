"""04 — a folder of videos -> one combined dataset (process each, export each)."""

from pathlib import Path

import actuate

actuate.login()

videos = sorted(Path("./my_videos/").glob("*.mp4"))
for i, video in enumerate(videos):
    run = actuate.process(source=str(video), rig="auto", max_frames=45)
    run.export("lerobot_v3", path=f"./examples_out/04_batch/{i:03d}/")
    print(f"{video.name}: {run.status} quality {run.quality}/5")
