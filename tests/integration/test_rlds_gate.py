"""Part C gate: the RLDS export loads with tfds.load and one episode iterates with the
Open-X step layout -- shapes, dtypes, and field names checked against what the REAL library
returns, not against what this repo wrote (writer and reader agreeing with each other while
both being wrong is the failure mode a self-made assertion cannot catch; tfds.load is the
independent reader)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from actuate.canonical import build_episode
from actuate.config import Tier
from actuate.package import ExportRefused
from actuate.package.rlds_export import export_rlds
from actuate.schema.episode import RobotAction

tf = pytest.importorskip("tensorflow", reason="tensorflow not installed")
tfds = pytest.importorskip("tensorflow_datasets", reason="tensorflow-datasets not installed")

REPO = Path(__file__).resolve().parents[2]
PROCESSED = REPO / "processed" / "session_001"
VIDEO = PROCESSED / "redacted_compressed.mp4"
CAPTURE_HASH = "4cff6acb16f7f390a35034eb7ddc088d76e13f1e814214fb8e49180fc1d6bb83"
TASK = "Sort and staple paperwork at the workbench."

pytestmark = [
    pytest.mark.real_data,
    pytest.mark.skipif(not VIDEO.exists(), reason="real redacted capture not on disk"),
]

_N = 60
_EMB = "franka_panda"


@pytest.fixture(scope="module")
def episode():
    ep = build_episode(PROCESSED, CAPTURE_HASH, task=TASK)
    ep = ep.model_copy(update={"frames": ep.frames[:_N], "tier": Tier.STAGE1_VOLUME})
    traj = 0.1 * np.cumsum(np.full((_N, 7), 0.01), axis=0)   # synthetic, labelled so:
    action = RobotAction(embodiment=_EMB, control_mode="joint",  # proves plumbing only
                         joint_traj=tuple(tuple(map(float, q)) for q in traj))
    return ep.model_copy(update={"action_robot": {_EMB: action}})


@pytest.fixture(scope="module")
def exported(episode, tmp_path_factory):
    root = tmp_path_factory.mktemp("rlds")
    return export_rlds(episode, root, name="actuate_gate", embodiment=_EMB, video=VIDEO)


def test_tfds_load_iterates_one_episode_with_openx_layout(exported):
    """THE gate: tfds.load + iterate; every Open-X field, right shape, right dtype."""
    ds = tfds.load(exported.name, data_dir=exported.root, split="train")
    n_eps = 0
    for ep in ds:
        n_eps += 1
        steps = list(ep["steps"])
        assert len(steps) == exported.n_steps
        s0, s_last = steps[0], steps[-1]

        # Open-X step fields, names exactly
        expected = {"observation", "action", "language_instruction", "reward",
                    "discount", "is_first", "is_last", "is_terminal",
                    f"action_robot_{_EMB}"}
        assert set(s0.keys()) == expected

        obs = s0["observation"]
        assert set(obs.keys()) == {"image", "state"}
        assert obs["image"].dtype == tf.uint8 and obs["image"].shape == (224, 224, 3)
        assert obs["state"].dtype == tf.float32 and obs["state"].shape == (8,)
        assert s0["action"].dtype == tf.float32 and s0["action"].shape == (8,)
        assert s0[f"action_robot_{_EMB}"].shape == (7,)
        assert s0["language_instruction"].numpy().decode() == TASK
        assert float(s0["reward"]) == 0.0 and float(s0["discount"]) == 1.0

        # episode boundary flags
        assert bool(s0["is_first"]) and not bool(s0["is_last"])
        assert bool(s_last["is_last"]) and bool(s_last["is_terminal"])
        assert not bool(s_last["is_first"])

        # the image is a real video frame, not padding
        assert int(tf.reduce_max(obs["image"])) > 0
    assert n_eps == 1


def test_rlds_refuses_an_untasked_episode(episode, tmp_path):
    with pytest.raises(ExportRefused, match="no `task`"):
        export_rlds(episode.model_copy(update={"task": None}), tmp_path, video=VIDEO)


def test_rlds_tier_filter_excludes(episode, tmp_path):
    with pytest.raises(ExportRefused, match="excluded every episode"):
        export_rlds(episode, tmp_path, tier="stage2", video=VIDEO)   # episode is stage1


def test_provenance_and_manifest_ship(exported):
    import json

    prov = json.loads((exported.root / "actuate_provenance.json").read_text())
    assert prov["format"] == "rlds/open-x"
    assert prov["consent"] == ["pending"]
    assert "reward=0" in prov["reward_note"]
    m = json.loads((exported.root / "actuate_manifest.json").read_text())
    assert m["episode_count"] == 1
