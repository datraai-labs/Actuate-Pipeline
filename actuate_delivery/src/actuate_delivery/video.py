import json
import shutil
import subprocess
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from hashlib import sha256
from itertools import pairwise
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


class VideoError(ValueError):
    pass


@dataclass(frozen=True)
class VideoArtifact:
    parquet_sha256: str
    frame_count: int
    codec: str
    width: int
    height: int
    average_frame_rate: str
    duration_ns: int | None
    audio_stream_count: int
    facts_json: str


def _run(command: list[str]) -> str:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode:
        detail = result.stderr.strip().splitlines()
        raise VideoError(detail[-1] if detail else f"Command exited {result.returncode}")
    return result.stdout


def _tool(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise VideoError(f"Required video tool is missing: {name}")
    return path


def _source_hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def verify_video(source: Path, output: Path, expected_sha256: str) -> VideoArtifact:
    if _source_hash(source) != expected_sha256:
        raise VideoError("Video source SHA-256 does not match the preserved mapping")
    ffprobe, ffmpeg = _tool("ffprobe"), _tool("ffmpeg")
    try:
        probe = json.loads(_run([ffprobe, "-v", "error", "-show_streams",
                                 "-show_format", "-of", "json", str(source)]))
        frame_probe = json.loads(_run([
            ffprobe, "-v", "error", "-select_streams", "v:0", "-show_frames",
            "-show_entries", "frame=pts", "-of", "json", str(source)]))
    except json.JSONDecodeError as error:
        raise VideoError(f"ffprobe returned invalid JSON: {error}") from error
    streams = probe.get("streams")
    if not isinstance(streams, list):
        raise VideoError("ffprobe JSON has no streams list")
    videos = [stream for stream in streams if stream.get("codec_type") == "video"]
    if len(videos) != 1:
        raise VideoError(f"MP4 has {len(videos)} video streams; expected exactly one")
    video = videos[0]
    try:
        time_base_num, time_base_den = map(int, video["time_base"].split("/"))
        codec = str(video["codec_name"])
        width, height = int(video["width"]), int(video["height"])
        average_frame_rate = str(video["avg_frame_rate"])
        frames = frame_probe["frames"]
        pts = [int(frame["pts"]) for frame in frames]
    except (KeyError, TypeError, ValueError) as error:
        raise VideoError(f"ffprobe omitted a required video fact: {error}") from error
    if time_base_num <= 0 or time_base_den <= 0 or width <= 0 or height <= 0:
        raise VideoError("ffprobe returned an invalid time base or video dimension")
    if not pts:
        raise VideoError("ffprobe returned zero video frames")
    if any(after <= before for before, after in pairwise(pts)):
        raise VideoError("MP4 frame PTS values must be strictly increasing")
    declared_frames = video.get("nb_frames")
    _run([ffmpeg, "-v", "error", "-xerror", "-i", str(source),
          "-map", "0:v:0", "-map", "0:a?", "-f", "null", "-"])

    pts_ns = []
    for value in pts:
        quotient, remainder = divmod(value * time_base_num * 1_000_000_000, time_base_den)
        pts_ns.append(quotient + (remainder * 2 >= time_base_den))
    duration_raw = video.get("duration") or probe.get("format", {}).get("duration")
    try:
        duration_ns = None if duration_raw is None else round(Decimal(duration_raw) * 1_000_000_000)
    except InvalidOperation as error:
        raise VideoError(f"ffprobe returned an invalid duration: {duration_raw!r}") from error
    audio = [stream for stream in streams if stream.get("codec_type") == "audio"]
    fact_keys = ("index", "codec_name", "sample_rate", "channels", "channel_layout",
                 "time_base", "duration_ts", "duration")
    facts = {
        "format_name": probe.get("format", {}).get("format_name"),
        "pixel_format": video.get("pix_fmt"),
        "time_base": video["time_base"],
        "duration_raw": duration_raw,
        "declared_frame_count": declared_frames,
        "enumerated_frame_count": len(pts),
        "audio_streams": [{key: stream.get(key) for key in fact_keys} for stream in audio],
        "ffprobe_version": _run([ffprobe, "-version"]).splitlines()[0],
        "ffmpeg_version": _run([ffmpeg, "-version"]).splitlines()[0],
    }
    facts_json = json.dumps(facts, sort_keys=True, separators=(",", ":"))
    metadata = {"schema_version": "actuate_delivery.video_frames.v1",
                "source_sha256": expected_sha256,
                "pts_ns_method": "exact_time_base_nearest_ns;half_ties_toward_positive_infinity",
                "write_parameters": "parquet=2.6;compression=zstd;dictionary=false;statistics=true"}
    table = pa.table({"video_frame_index": range(len(pts)), "mp4_pts": pts,
                      "time_base_num": [time_base_num] * len(pts),
                      "time_base_den": [time_base_den] * len(pts), "mp4_pts_ns": pts_ns})
    table = table.replace_schema_metadata({k.encode(): v.encode() for k, v in metadata.items()})
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(f".{output.name}.staging")
    staging.unlink(missing_ok=True)
    try:
        pq.write_table(table, staging, version="2.6", compression="zstd",
                       use_dictionary=False, write_statistics=True)
        if not pq.read_table(staging).equals(table, check_metadata=True):
            raise VideoError("Video frame index does not match enumerated ffprobe frames")
        parquet_hash = sha256(staging.read_bytes()).hexdigest()
        staging.replace(output)
    except (OSError, pa.ArrowException) as error:
        raise VideoError(f"Video frame-index write or read failed: {error}") from error
    finally:
        staging.unlink(missing_ok=True)
    return VideoArtifact(parquet_hash, len(pts), codec, width, height,
                         average_frame_rate, duration_ns, len(audio), facts_json)
