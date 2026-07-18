"""07 — read the L4 quality certificate straight off a ProcessingRun."""

import json

import actuate

actuate.login()
run = actuate.process(source="./processed/session_001/compressed.mp4",
                      rig="head_mounted", task="handle paperwork", max_frames=20)

cert = run.certificate()
print(json.dumps(cert, indent=2, default=str))
# 'deliverable' is False until consent=granted AND pii=passed -- the fail-closed gate.
print("deliverable:", cert["deliverable"], "| consent:", cert["consent"])
