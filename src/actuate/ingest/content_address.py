"""Content-addressed capture provenance — L0.

Increment 1 found three integrity failures in the real corpus, and all three share a root
cause: **nothing bound the metadata to the bytes it described.**

    1. Metadata declaring 1544 frames beside 2850 frames of perception output from a
       different recording.
    2. A `raw.mp4` of 1.49 MB whose metadata described a 181.84 MB source. The frame counts
       were internally consistent, so any check that looked only at those waved it through.
    3. The same footage under four session ids with conflicting consent records -- because
       "the same footage" was not a thing the system could express.

Those were caught by *ad-hoc checks* bolted onto the migration. Checks catch what you
thought to check for. This module removes the failure class instead:

**The capture id IS the SHA-256 of the raw bytes.**

From that one decision:

  - A payload/metadata mismatch becomes impossible to express. The manifest records the
    hash of the bytes it describes, and the bytes live at a path derived from that hash.
    Metadata cannot drift from its payload, because the payload's identity *is* the thing
    the metadata is filed under.
  - A duplicate upload becomes a dedup hit at write time, not a conflict discovered later.
    Two uploads of the same footage compute the same id and collide in S3 -- there is no
    second capture to disagree with the first.
  - Consent has an unambiguous subject. One recording, one id, one consent decision, no
    matter how many session directories it arrives in.

Derived artifacts carry `source_content_hash` back to the capture they came from, so the
chain is verifiable end to end rather than assumed.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from actuate.config import RigType

_CHUNK = 1 << 20


class IntegrityError(RuntimeError):
    """The bytes are not what the metadata says they are."""


def hash_file(path: Path) -> str:
    """SHA-256 of a file's bytes. THE capture identity.

    Streamed, because raw captures are hundreds of megabytes and will be gigabytes.
    """
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while chunk := fh.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


class CaptureManifest(BaseModel):
    """The binding between a capture's bytes and everything claimed about them.

    Written alongside the raw video at `raw/<rig>/<capture_id>/manifest.json`, where
    `capture_id == content_hash`. Anything that later claims to describe this capture can
    be checked against it by re-hashing -- the claim is falsifiable, which is the whole
    point.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    #: == content_hash. Named separately because downstream code talks about capture ids,
    #: and the identity of the two is a fact worth being able to assert.
    capture_id: str = Field(min_length=64, max_length=64)
    content_hash: str = Field(min_length=64, max_length=64)

    size_bytes: int = Field(gt=0)
    rig: RigType

    #: Everything below is a CLAIM about the bytes, not a property of them. It is recorded
    #: here so that it is bound to the hash and can be contradicted, rather than floating
    #: free in a session_meta.json that may describe some other video entirely.
    frame_count: int | None = Field(default=None, ge=0)
    duration_sec: float | None = Field(default=None, ge=0)
    fps: float | None = Field(default=None, gt=0)

    #: Where it came from. Provenance, not identity -- two identical files uploaded from
    #: different paths are still one capture.
    source_path: str | None = None
    ingested_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def verify(self, path: Path) -> None:
        """Re-hash the bytes and confirm they are the ones this manifest describes.

        This is the check that could not have been forgotten in Increment 1, because it is
        the only thing that establishes the manifest applies to this file at all.
        """
        p = Path(path)
        actual_size = p.stat().st_size
        if actual_size != self.size_bytes:
            raise IntegrityError(
                f"{p.name}: manifest says {self.size_bytes} bytes, file is {actual_size}. "
                "The metadata does not describe this payload."
            )
        actual = hash_file(p)
        if actual != self.content_hash:
            raise IntegrityError(
                f"{p.name}: content hash {actual[:12]}... does not match the manifest's "
                f"{self.content_hash[:12]}.... The metadata does not describe this payload."
            )

    @property
    def prefix(self) -> str:
        """`<rig>/<capture_id>/` -- the S3 key prefix. Content-addressed by construction:
        the same bytes always land in the same place, so a re-upload is a no-op rather
        than a second capture to reconcile."""
        return f"{self.rig.value}/{self.capture_id}"


def build_manifest(
    video: Path,
    rig: RigType,
    *,
    frame_count: int | None = None,
    duration_sec: float | None = None,
    fps: float | None = None,
) -> CaptureManifest:
    """Hash the bytes, then bind the claims to that hash."""
    video = Path(video)
    digest = hash_file(video)
    return CaptureManifest(
        capture_id=digest,
        content_hash=digest,
        size_bytes=video.stat().st_size,
        rig=rig,
        frame_count=frame_count,
        duration_sec=duration_sec,
        fps=fps,
        source_path=str(video),
    )


def check_legacy_claims(
    manifest: CaptureManifest, session_meta: dict, perception_frames: int | None
) -> list[str]:
    """Cross-check a v1 `session_meta.json` against the bytes it claims to describe.

    Returns the contradictions, empty if consistent. This is the *migration-time* bridge:
    legacy sessions have no manifest, so their claims must be checked against a freshly
    computed hash before they are allowed into a content-addressed world.

    Both real failures from Increment 1 are detected here:
      - a declared size that does not match the actual bytes;
      - a declared frame count that does not match the perception output.

    New ingestion does not need this. There, the manifest is written *from* the bytes, so
    the two cannot disagree in the first place.
    """
    problems: list[str] = []

    declared_mb = session_meta.get("raw_size_mb")
    if declared_mb:
        actual_mb = manifest.size_bytes / 1e6
        if abs(actual_mb - declared_mb) / declared_mb > 0.10:
            problems.append(
                f"raw.mp4 is {actual_mb:.2f} MB but session_meta declares a "
                f"{declared_mb:.2f} MB source -- the video and the metadata are from "
                "different recordings"
            )

    declared_frames = session_meta.get("frame_count")
    if (
        declared_frames
        and perception_frames is not None
        and perception_frames != declared_frames
    ):
        problems.append(
            f"session_meta declares {declared_frames} frames but {perception_frames} "
            "per-frame records exist -- the perception outputs are from a different "
            "recording"
        )

    return problems


def write_manifest(path: Path, manifest: CaptureManifest) -> None:
    Path(path).write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def read_manifest(path: Path) -> CaptureManifest:
    return CaptureManifest.model_validate_json(Path(path).read_text(encoding="utf-8"))
