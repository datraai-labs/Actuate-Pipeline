from __future__ import annotations


def test_default_arm_model_is_scoped_to_actuate_home(tmp_path, monkeypatch):
    monkeypatch.setenv("ACTUATE_HOME", str(tmp_path / "local-auth"))
    from actuate.retarget.arm import default_model_path

    assert default_model_path("franka_panda") == (
        tmp_path / "local-auth" / "models" / "franka_panda_root_frame.pt"
    )
