import json
import sqlite3
from hashlib import sha256

import actuate_delivery.video as video_module
import pyarrow.parquet as pq
import pytest
from actuate_delivery.inventory import FileFact, PreservedFile, SourceInventory, group_captures
from actuate_delivery.run import (
    open_run,
    process_imus,
    process_sidecars,
    process_videos,
    store_inventory,
    store_preservation,
)
from actuate_delivery.video import VideoError, verify_video


def probe_json(video_count=1):
    videos = [
        {
            "index": index,
            "codec_type": "video",
            "codec_name": "hevc",
            "width": 1920,
            "height": 1080,
            "pix_fmt": "yuv420p",
            "time_base": "1/1000",
            "avg_frame_rate": "30/1",
            "nb_frames": "3",
            "duration": "0.067",
        }
        for index in range(video_count)
    ]
    audio = {
        "index": video_count,
        "codec_type": "audio",
        "codec_name": "aac",
        "sample_rate": "48000",
        "channels": 2,
        "channel_layout": "stereo",
        "time_base": "1/48000",
        "duration_ts": 3216,
        "duration": "0.067",
    }
    return json.dumps({"streams": [*videos, audio], "format": {"format_name": "mov,mp4"}})


def install_fake_tools(monkeypatch, probe=None, frames=None, decode_error=None):
    calls = []
    probe = probe_json() if probe is None else probe
    frames = {"frames": [{"pts": 0}, {"pts": 33}, {"pts": 67}]} if frames is None else frames
    monkeypatch.setattr(video_module, "_tool", lambda name: name)

    def run(command):
        calls.append(command)
        if "-show_streams" in command:
            return probe
        if "-show_frames" in command:
            return json.dumps(frames)
        if "-version" in command:
            return f"{command[0]} version test\n"
        if decode_error:
            raise VideoError(decode_error)
        return ""

    monkeypatch.setattr(video_module, "_run", run)
    return calls


def test_video_probe_full_decode_and_exact_frame_index(tmp_path, monkeypatch):
    source = tmp_path / "take.mp4"
    source.write_bytes(b"video")
    source_hash = sha256(source.read_bytes()).hexdigest()
    output = tmp_path / "frames.parquet"
    calls = install_fake_tools(monkeypatch)

    artifact = verify_video(source, output, source_hash)
    table = pq.read_table(output)

    assert artifact.frame_count == 3
    assert (artifact.codec, artifact.width, artifact.height) == ("hevc", 1920, 1080)
    assert artifact.duration_ns == 67_000_000
    assert artifact.audio_stream_count == 1
    assert table["mp4_pts"].to_pylist() == [0, 33, 67]
    assert table["mp4_pts_ns"].to_pylist() == [0, 33_000_000, 67_000_000]
    assert table.schema.metadata[b"source_sha256"].decode() == source_hash
    assert any("-xerror" in command and "0:a?" in command for command in calls)


def test_declared_frame_count_is_preserved_but_enumerated_frames_are_used(tmp_path, monkeypatch):
    source = tmp_path / "take.mp4"
    source.write_bytes(b"video")
    probe = json.loads(probe_json())
    probe["streams"][0]["nb_frames"] = "30"
    install_fake_tools(monkeypatch, json.dumps(probe))

    artifact = verify_video(
        source, tmp_path / "frames.parquet", sha256(source.read_bytes()).hexdigest()
    )
    facts = json.loads(artifact.facts_json)

    assert artifact.frame_count == 3
    assert facts["declared_frame_count"] == "30"
    assert facts["enumerated_frame_count"] == 3


@pytest.mark.parametrize(
    ("probe", "frames", "message"),
    [
        ("not-json", None, "invalid JSON"),
        (json.dumps({}), None, "no streams list"),
        (probe_json(0), None, "0 video streams"),
        (probe_json(2), None, "2 video streams"),
        (probe_json(), {"frames": []}, "zero video frames"),
        (probe_json(), {"frames": [{"pts": 1}, {"pts": 1}, {"pts": 2}]}, "strictly increasing"),
        (probe_json(), {"frames": [{"bad": 1}]}, "required video fact"),
    ],
)
def test_invalid_probe_fails_for_exact_reason(tmp_path, monkeypatch, probe, frames, message):
    source = tmp_path / "take.mp4"
    source.write_bytes(b"video")
    install_fake_tools(monkeypatch, probe, frames)

    with pytest.raises(VideoError, match=message):
        verify_video(source, tmp_path / "frames.parquet", sha256(b"video").hexdigest())


def test_missing_tool_decode_failure_and_wrong_hash_publish_nothing(tmp_path, monkeypatch):
    source = tmp_path / "take.mp4"
    source.write_bytes(b"video")
    output = tmp_path / "frames.parquet"
    source_hash = sha256(b"video").hexdigest()
    monkeypatch.setattr(video_module.shutil, "which", lambda name: None)
    with pytest.raises(VideoError, match="missing"):
        verify_video(source, output, source_hash)

    install_fake_tools(monkeypatch, decode_error="simulated decode failure")
    with pytest.raises(VideoError, match="decode failure"):
        verify_video(source, output, source_hash)
    with pytest.raises(VideoError, match="source SHA-256"):
        verify_video(source, output, "0" * 64)
    assert not output.exists()


def test_duplicate_video_stream_fails_without_choosing_a_member(tmp_path):
    run_dir = tmp_path / "run"
    first = FileFact(
        "a.mp4",
        "a.mp4",
        ".",
        "video",
        "take",
        "left",
        1,
        1,
        "local",
        ".",
        "video/mp4",
        None,
        None,
        True,
    )
    second = FileFact(
        "b.mp4",
        "b.mp4",
        ".",
        "video",
        "take",
        "left",
        1,
        1,
        "local",
        ".",
        "video/mp4",
        None,
        None,
        True,
    )
    inventory = SourceInventory("file:///source", (first, second), group_captures((first, second)))
    open_run(inventory.source_identity, run_dir)
    store_inventory(run_dir / "run.sqlite", inventory, inventory)
    preservation = (
        (PreservedFile("a.mp4", "a" * 64, "new"), PreservedFile("b.mp4", "b" * 64, "new")),
        ((".", "take", "c" * 64, True),),
        0,
    )
    store_preservation(run_dir / "run.sqlite", preservation)
    process_imus(run_dir / "run.sqlite", run_dir)
    process_sidecars(run_dir / "run.sqlite", run_dir)

    assert process_videos(run_dir / "run.sqlite", run_dir) == (0, 0, 1)
    error = (
        sqlite3.connect(run_dir / "run.sqlite")
        .execute("SELECT error FROM video_artifact")
        .fetchone()[0]
    )
    assert error == "Camera stream has 2 video members; expected one"
