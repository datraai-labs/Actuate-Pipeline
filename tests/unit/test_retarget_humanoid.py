from __future__ import annotations

import json

import numpy as np
import pytest

from actuate.retarget.humanoid import (
    GMRQualityReport,
    GMRResult,
    _reference_root_is_complete,
    normalize_source_format,
    postprocess_and_measure,
    retarget_frames,
)


class _WarmStartRetargeter:
    def __init__(self):
        self.value = 0.0

    def retarget(self, frame):
        self.value += frame["increment"]
        return np.array([self.value, self.value + 1.0])


def test_retarget_frames_preserves_sequential_warm_start():
    retargeter = _WarmStartRetargeter()
    progress = []

    qpos = retarget_frames(
        [{"increment": 1.0}, {"increment": 2.0}, {"increment": -0.5}],
        retargeter,
        progress=lambda done, total: progress.append((done, total)),
    )

    np.testing.assert_allclose(qpos[:, 0], [1.0, 3.0, 2.5])
    assert progress == [(1, 3), (2, 3), (3, 3)]


def test_postprocess_aligns_global_minimum_and_reports_limits():
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody>
            <body name="root" pos="0 0 0">
              <freejoint/>
              <geom type="sphere" size="0.05"/>
              <body name="link" pos="0 0 -0.5">
                <joint name="hinge" type="hinge" range="-1 1"/>
                <geom type="sphere" size="0.05"/>
              </body>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    qpos = np.zeros((2, model.nq))
    qpos[:, 2] = [0.8, 0.7]
    qpos[:, 3] = 1.0
    qpos[:, 7] = [0.0, 0.01]

    corrected, report, joint_names, body_names = postprocess_and_measure(
        qpos, model, fps=10.0
    )

    assert report.deliverable == "PASS"
    assert report.ground_height_before_m == pytest.approx(0.2)
    assert report.ground_height_after_m == pytest.approx(0.0, abs=1e-9)
    np.testing.assert_allclose(corrected[:, 2], [0.6, 0.5])
    assert joint_names == ("hinge",)
    assert "link" in body_names
    assert report.joint_limit_violations == 0


def test_safe_artifacts_include_method_provenance(tmp_path):
    quality = GMRQualityReport(
        deliverable="PASS",
        finite=True,
        ground_height_before_m=0.1,
        ground_height_after_m=0.0,
        ground_correction_m=0.1,
        joint_limit_violations=0,
        max_joint_speed_rad_s=1.0,
        velocity_clipped_frames=(),
        velocity_spike_frames=(),
        self_collision_frames=(),
        self_collision_contacts=0,
        warnings=(),
    )
    result = GMRResult(
        qpos=np.zeros((3, 8)),
        fps=30.0,
        robot="unitree_g1",
        source_format="bvh_xsens",
        source_path="/motion.bvh",
        joint_names=("joint",),
        body_names=("pelvis",),
        model_xml="/g1.xml",
        backend_version="0.2.0",
        quality=quality,
    )

    motion_path, report_path = result.write(tmp_path)

    archive = np.load(motion_path)
    assert archive["qpos"].shape == (3, 8)
    report = json.loads(report_path.read_text())
    assert report["algorithm"]["paper"].endswith("2510.02252")
    assert report["algorithm"]["sequential_warm_start"] is True
    assert report["quality"]["deliverable"] == "PASS"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("xsens", "bvh_xsens"),
        ("lafan1", "bvh_lafan1"),
        ("bvh_nokov", "bvh_nokov"),
    ],
)
def test_source_format_aliases(value, expected):
    assert normalize_source_format(value) == expected


def test_unknown_source_format_is_actionable():
    with pytest.raises(ValueError, match="unsupported source format"):
        normalize_source_format("video")


def test_reference_asset_completeness_requires_model_and_config(tmp_path):
    assert not _reference_root_is_complete(tmp_path)
    (tmp_path / "assets/unitree_g1").mkdir(parents=True)
    (tmp_path / "assets/unitree_g1/g1_mocap_29dof.xml").write_text("<mujoco/>")
    assert not _reference_root_is_complete(tmp_path)
    config = tmp_path / "general_motion_retargeting/ik_configs"
    config.mkdir(parents=True)
    for name in (
        "bvh_lafan1_to_g1.json",
        "bvh_nokov_to_g1.json",
        "bvh_xsens_to_g1.json",
    ):
        (config / name).write_text("{}")
    assert _reference_root_is_complete(tmp_path)
