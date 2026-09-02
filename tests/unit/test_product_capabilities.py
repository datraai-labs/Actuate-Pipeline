from actuate.config import product_capabilities


def test_product_capabilities_are_json_safe_and_registry_backed():
    capabilities = product_capabilities()

    rigs = {item["id"]: item for item in capabilities["rigs"]}
    assert rigs["head_mounted"]["enabled"] is True
    assert rigs["stereo"]["enabled"] is False
    assert rigs["glove"]["enabled"] is False
    assert rigs["stereo"]["cameras"] == ["stereo_left", "stereo_right"]
    assert rigs["stereo"]["measured_channels"] == ["depth"]

    embodiments = {item["id"]: item for item in capabilities["embodiments"]}
    assert embodiments["franka_panda"]["kinematic_model"] is True
    assert embodiments["franka_panda"]["enabled"] is True
    assert embodiments["franka_dual"]["enabled"] is False
