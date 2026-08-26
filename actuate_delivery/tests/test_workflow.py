import struct
import zipfile

import pytest
from actuate_delivery.run import RunError
from actuate_delivery.workflow import _build_archive, approve_stage, run_stage, workflow_state


def state(run_dir, stage):
    return next(item for item in workflow_state(run_dir) if item["stage"] == stage)


def test_stage_requires_previous_exact_approval(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    run_dir = tmp_path / "run"

    run_stage(str(source), run_dir, "inventory")

    with pytest.raises(RunError, match="Approve Inventory and preserve"):
        run_stage(str(source), run_dir, "sensors")
    assert state(run_dir, "inventory")["status"] == "complete"
    assert state(run_dir, "inventory")["approved"] is False


def test_unchanged_stage_keeps_approval_but_changed_source_clears_it(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "notes.txt").write_text("first")
    run_dir = tmp_path / "run"

    run_stage(str(source), run_dir, "inventory")
    approve_stage(run_dir, "inventory", "operator")
    run_stage(str(source), run_dir, "inventory")
    assert state(run_dir, "inventory")["approved"] is True

    (source / "notes.txt").write_text("changed")
    run_stage(str(source), run_dir, "inventory")

    inventory = state(run_dir, "inventory")
    assert inventory["approved"] is False
    assert inventory["summary"]["changed"] == 1
    assert state(run_dir, "sensors")["status"] == "waiting"


def test_empty_run_reaches_review_only_through_approved_stages(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    run_dir = tmp_path / "run"

    for stage in ("inventory", "sensors", "video", "timing", "qc", "review"):
        result = run_stage(str(source), run_dir, stage)
        assert result.status == "complete"
        approve_stage(run_dir, stage, "operator")

    review = state(run_dir, "review")
    assert review["approved"] is True
    assert review["summary"] == {
        "episodes": 0,
        "excluded": 0,
        "included": 0,
        "pending": 0,
        "review_path": str(run_dir / "review.csv"),
    }


def test_changed_approved_artifact_blocks_resume_and_clears_approval(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    header = bytearray(64)
    struct.pack_into("<8sIIHHQQI", header, 0, b"TRIMU001", 4, 400, 2, 3, 50, 75, 1)
    rows = [struct.pack("<Q18f", timestamp, *([float(index)] * 18))
            for index, timestamp in enumerate((100, 200))]
    (source / "take.imu").write_bytes(bytes(header) + b"".join(rows))
    run_dir = tmp_path / "run"

    run_stage(str(source), run_dir, "inventory")
    approve_stage(run_dir, "inventory", "operator")
    run_stage(str(source), run_dir, "sensors")
    approve_stage(run_dir, "sensors", "operator")
    (next((run_dir / "work").glob("*/imu.parquet"))).write_bytes(b"changed")

    with pytest.raises(RunError, match="changed after verification"):
        run_stage(str(source), run_dir, "sensors")
    assert state(run_dir, "sensors")["status"] == "failed"
    assert state(run_dir, "sensors")["approved"] is False
    assert state(run_dir, "video")["status"] == "waiting"


def test_delivery_archive_is_complete_and_crc_valid(tmp_path):
    output = tmp_path / "dataset"
    (output / "episodes/episode_000001").mkdir(parents=True)
    (output / "README.md").write_text("dataset")
    (output / "episodes/episode_000001/meta.json").write_text("{}")

    archive_path = _build_archive(output)

    with zipfile.ZipFile(archive_path) as archive:
        assert archive.testzip() is None
        assert archive.namelist() == [
            "dataset/README.md",
            "dataset/episodes/episode_000001/meta.json",
        ]
