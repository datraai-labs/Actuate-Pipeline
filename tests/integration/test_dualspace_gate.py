"""Part D gates on the REAL capture: dual-space export (human + robot actions in one
dataset, selectable by embodiment tag), tier filtering at the export boundary, per-space
norm stats, and the manifest -- all verified through LeRobot's own loader, not ours."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from actuate.canonical import build_episode
from actuate.config import Tier
from actuate.package import ExportRefused, export_lerobot_v3
from actuate.schema.episode import RobotAction

pytest.importorskip("lerobot", reason="lerobot not installed")

REPO = Path(__file__).resolve().parents[2]
PROCESSED = REPO / "processed" / "session_001"
VIDEO = PROCESSED / "redacted_compressed.mp4"
CAPTURE_HASH = "4cff6acb16f7f390a35034eb7ddc088d76e13f1e814214fb8e49180fc1d6bb83"
TASK = "Sort and staple paperwork at the workbench."

pytestmark = [
    pytest.mark.real_data,
    pytest.mark.skipif(not VIDEO.exists(), reason="real redacted capture not on disk"),
]

_N = 90            # enough real frames to be a real export, small enough to run in minutes
_EMB = "franka_panda"


@pytest.fixture(scope="module")
def episode():
    ep = build_episode(PROCESSED, CAPTURE_HASH, task=TASK)
    ep = ep.model_copy(update={"frames": ep.frames[:_N], "tier": Tier.STAGE2_ANCHOR})
    # a smooth synthetic 7-dof trajectory aligned to the FULL frame set. SYNTHETIC and
    # labelled so: this gate proves the dual-space PLUMBING on real video; the retargeted
    # values themselves are validated by the L5 gates, not here.
    traj = 0.1 * np.cumsum(np.full((_N, 7), 0.01), axis=0)
    action = RobotAction(embodiment=_EMB, control_mode="joint",
                         joint_traj=tuple(tuple(map(float, q)) for q in traj))
    return ep.model_copy(update={"action_robot": {_EMB: action}})


@pytest.fixture(scope="module")
def exported(episode, tmp_path_factory):
    root = tmp_path_factory.mktemp("dualspace") / "ds"
    res = export_lerobot_v3(
        episode, root, repo_id="actuate/dualspace-gate", overwrite=True, video=VIDEO,
        embodiment=_EMB, tier="stage2",
    )
    return res


def test_dual_space_loads_with_both_action_spaces(exported):
    """GATE: both spaces present, selectable by embodiment tag, via LeRobot's OWN loader."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(repo_id="actuate/dualspace-gate", root=exported.root)
    assert "action" in ds.meta.features                      # human space
    assert f"action.robot.{_EMB}" in ds.meta.features        # robot space, tagged
    sample = ds[0]
    assert sample["action"].shape[-1] == 8                   # wrist-only human layout
    assert sample[f"action.robot.{_EMB}"].shape[-1] == 7     # franka joints
    assert not bool(np.isnan(sample[f"action.robot.{_EMB}"].numpy()).any())


def test_per_space_norm_stats_ship(exported):
    human = json.loads((exported.root / "meta" / "actuate_norm_stats.json").read_text())
    robot = json.loads(
        (exported.root / "meta" / f"actuate_norm_stats.{_EMB}.json").read_text())
    assert human["state"]["p50"] is not None                 # v4 full percentile set
    assert robot["action"]["p01"] is not None
    assert robot["state"] is None                            # robot space ships action only
    # the two spaces are genuinely different stats, not one copied over the other
    assert human["action"]["p99"] != robot["action"]["p99"]


def test_manifest_ships_and_reports_this_dataset(exported):
    m = json.loads((exported.root / "meta" / "actuate_manifest.json").read_text())
    assert m["episode_count"] == 1
    assert m["tier_distribution"] == {"stage2_anchor": 1}
    assert m["embodiments_with_actions"] == {_EMB: 1}
    assert m["modality_inventory"]["language_task"] == 1
    # n=1 corpus looks like an n=1 corpus -- unknown axes reported, not hidden
    assert m["episodes_with_unknown_demonstrator"] == 1


def test_tier_filter_refuses_when_it_excludes_everything(episode, tmp_path):
    """GATE (other direction): --tier stage1 on a stage2-only set exports NOTHING."""
    with pytest.raises(ExportRefused, match="excluded every episode"):
        export_lerobot_v3(episode, tmp_path / "never", video=VIDEO,
                          embodiment=_EMB, tier="stage1")


def test_transforms_refuse_without_intrinsics(episode, tmp_path):
    with pytest.raises(ExportRefused, match="intrinsics"):
        export_lerobot_v3(episode, tmp_path / "never", video=VIDEO,
                          transforms=("masked_hand",))
