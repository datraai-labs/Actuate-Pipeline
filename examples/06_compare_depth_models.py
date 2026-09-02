"""06 — inspect the depth a run produced (the depth A/B lives in the advanced CLI).

This shows how to reach the per-frame depth on a processed canonical."""

import actuate
from actuate.schema import CanonicalEpisode

actuate.login()
run = actuate.process(source="./processed/session_001/compressed.mp4",
                      rig="head_mounted", task="handle paperwork", max_frames=20)

ep = CanonicalEpisode.model_validate_json(open(run.canonical_path, encoding="utf-8").read())
with_depth = [f.frame_idx for f in ep.frames if f.depth]
print(f"{len(with_depth)}/{len(ep.frames)} frames carry a depth reference")
# Full model comparisons use `actuate.perception.depth.benchmark` in a GPU environment.
