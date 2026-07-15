"""Canonical store — Parquet (tabular) + Zarr (dense per-frame), per Master Spec §L3.

Layout mirrors AWS Architecture §2 exactly, on S3 and locally alike:

    work/canonical/<episode_id>/v<N>/
        episode.json     episode-level record (task, actions, certification, gates)
        frames.parquet   per-frame tabular: scalars, poses, contact, refs
        dense.zarr/      hand keypoints (N,2,21,3) and MANO params — the big arrays

**Video is never copied here.** `images.<cam>` and `depth.<cam>` hold *references* into
chunked MP4 / depth assets (AWS Architecture §1: video dominates cost; never duplicate,
never re-encode). The canonical representation points at pixels; it does not carry them.

The verification gate (Master Spec §3) is **bit-exact field recovery** on a real episode.
Two choices follow from taking that literally:

  - Dense float arrays go to Zarr as float64 and come back as float64. No downcasting to
    float32 "because it's close enough" — a metric hand keypoint that shifts in the
    seventh decimal on every round-trip is a schema that quietly loses information.
  - Absent values are written as NaN and read back as absent. NaN is not a value here; it
    is the encoding of "this frame had no hand", and the reader restores `None` rather
    than handing a downstream model a plausible-looking zero.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from actuate.config import Bucket, Finger, Provenance, RigType, Side
from actuate.io.backends import StorageBackend, StorageError
from actuate.schema import (
    SE3,
    CanonicalEpisode,
    CanonicalFrame,
    ContactReading,
    HandState,
    MANOParams,
)

SIDES: tuple[Side, ...] = (Side.LEFT, Side.RIGHT)
FINGERS: tuple[Finger, ...] = tuple(Finger)
_NA3 = [float("nan")] * 3
_NA4 = [float("nan")] * 4


def canonical_prefix(episode_id: str, version: int = 1) -> str:
    return f"canonical/{episode_id}/v{version}"


# --------------------------------------------------------------------------------------
# helpers — None <-> NaN, at exactly one place each
# --------------------------------------------------------------------------------------


def _se3_cols(pose: SE3 | None) -> tuple[list[float], list[float]]:
    if pose is None:
        return list(_NA3), list(_NA4)
    return list(pose.position_m), list(pose.quaternion_wxyz)


def _se3_from(pos: Any, quat: Any) -> SE3 | None:
    if pos is None or quat is None or any(np.isnan(pos)):
        return None
    return SE3(position_m=tuple(pos), quaternion_wxyz=tuple(quat))


def _jdump(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True)


# --------------------------------------------------------------------------------------
# write
# --------------------------------------------------------------------------------------


def _frames_table(frames: tuple[CanonicalFrame, ...]) -> pa.Table:
    cols: dict[str, list] = {
        "frame_idx": [],
        "t": [],
        "interaction_state": [],
        "camera_pose_pos": [],
        "camera_pose_quat": [],
        "images_json": [],
        "depth_json": [],
        "objects_json": [],
        "confidence_json": [],
        "provenance_json": [],
    }
    for side in SIDES:
        s = side.value
        cols[f"wrist_{s}_pos"] = []
        cols[f"wrist_{s}_quat"] = []
        cols[f"finger_joints_human_{s}"] = []
        cols[f"finger_joints_robotspace_{s}"] = []
        cols[f"contact_{s}_conf"] = []
        cols[f"contact_{s}_src"] = []

    for f in frames:
        cols["frame_idx"].append(f.frame_idx)
        cols["t"].append(f.t)
        cols["interaction_state"].append(
            f.interaction_state.value if f.interaction_state else None
        )
        pos, quat = _se3_cols(f.camera_pose)
        cols["camera_pose_pos"].append(pos)
        cols["camera_pose_quat"].append(quat)

        cols["images_json"].append(
            _jdump({k: v.model_dump(mode="json") for k, v in f.images.items()})
        )
        cols["depth_json"].append(
            _jdump({k: v.model_dump(mode="json") for k, v in f.depth.items()})
        )
        cols["objects_json"].append(
            _jdump({k: v.model_dump(mode="json") for k, v in f.objects.items()})
        )
        cols["confidence_json"].append(_jdump(f.confidence))
        cols["provenance_json"].append(_jdump({k: v.value for k, v in f.provenance.items()}))

        for side in SIDES:
            s = side.value
            hand = f.hands.get(side)
            wp, wq = _se3_cols(hand.wrist_pose if hand else None)
            cols[f"wrist_{s}_pos"].append(wp)
            cols[f"wrist_{s}_quat"].append(wq)

            fjh = (f.finger_joints_human or {}).get(side)
            cols[f"finger_joints_human_{s}"].append(list(fjh) if fjh is not None else None)
            fjr = (f.finger_joints_robotspace or {}).get(side)
            cols[f"finger_joints_robotspace_{s}"].append(list(fjr) if fjr is not None else None)

            readings = (f.contact or {}).get(side)
            if readings is None:
                # None means NOT MEASURED. It must survive the round-trip as None and
                # never become a vector of zeros, which would read as "measured, open".
                cols[f"contact_{s}_conf"].append(None)
                cols[f"contact_{s}_src"].append(None)
            else:
                cols[f"contact_{s}_conf"].append([readings[k].confidence for k in FINGERS])
                cols[f"contact_{s}_src"].append([readings[k].source.value for k in FINGERS])

    # Variable-length lists, not fixed-size: Arrow's fixed_size_list does not round-trip
    # nulls (a null becomes a zero-length list on read, and the reader then rejects it).
    # Lengths are already enforced by the schema validators, so Parquet need not re-assert
    # them — and a null here is load-bearing: it is how "not measured" survives the trip.
    schema = pa.schema(
        [
            ("frame_idx", pa.int32()),
            ("t", pa.float64()),
            ("interaction_state", pa.string()),
            ("camera_pose_pos", pa.list_(pa.float64())),
            ("camera_pose_quat", pa.list_(pa.float64())),
            ("images_json", pa.string()),
            ("depth_json", pa.string()),
            ("objects_json", pa.string()),
            ("confidence_json", pa.string()),
            ("provenance_json", pa.string()),
        ]
        + [
            field
            for side in SIDES
            for field in (
                (f"wrist_{side.value}_pos", pa.list_(pa.float64())),
                (f"wrist_{side.value}_quat", pa.list_(pa.float64())),
                (f"finger_joints_human_{side.value}", pa.list_(pa.float64())),
                (f"finger_joints_robotspace_{side.value}", pa.list_(pa.float64())),
                (f"contact_{side.value}_conf", pa.list_(pa.float64())),
                (f"contact_{side.value}_src", pa.list_(pa.string())),
            )
        ]
    )
    return pa.table(cols, schema=schema)


def _dense_arrays(frames: tuple[CanonicalFrame, ...]) -> dict[str, np.ndarray]:
    """Hand keypoints and MANO — the arrays too big for a Parquet row. float64 throughout."""
    n = len(frames)
    kp = np.full((n, 2, 21, 3), np.nan, dtype=np.float64)
    betas = np.full((n, 2, 10), np.nan, dtype=np.float64)
    theta = np.full((n, 2, 45), np.nan, dtype=np.float64)  # schema v3: full 45 axis-angle
    orient = np.full((n, 2, 3), np.nan, dtype=np.float64)

    for i, f in enumerate(frames):
        for j, side in enumerate(SIDES):
            hand = f.hands.get(side)
            if hand is None:
                continue
            if hand.keypoints_3d is not None:
                kp[i, j] = np.asarray(hand.keypoints_3d, dtype=np.float64)
            if hand.mano is not None:
                betas[i, j] = np.asarray(hand.mano.betas, dtype=np.float64)
                theta[i, j] = np.asarray(hand.mano.theta, dtype=np.float64)
                orient[i, j] = np.asarray(hand.mano.global_orient, dtype=np.float64)

    return {"keypoints_3d": kp, "mano_betas": betas, "mano_theta": theta, "mano_orient": orient}


def write_episode(
    backend: StorageBackend,
    episode: CanonicalEpisode,
    version: int = 1,
    bucket: Bucket = Bucket.WORK,
) -> str:
    """Write to work/canonical/<episode_id>/v<N>/. Returns the prefix URI.

    Note the bucket default: canonical artifacts are WORK, never DELIVERY. Getting to the
    delivery bucket requires passing the consent gate — see io/consent.py.
    """
    prefix = canonical_prefix(episode.episode_id, version)

    meta = episode.model_dump(mode="json", exclude={"frames"})
    backend.put_bytes(bucket, f"{prefix}/episode.json", _jdump(meta).encode())

    table = _frames_table(episode.frames)
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink, compression="zstd")
    backend.put_bytes(bucket, f"{prefix}/frames.parquet", sink.getvalue().to_pybytes())

    for name, arr in _dense_arrays(episode.frames).items():
        buf = arr.tobytes()
        backend.put_bytes(bucket, f"{prefix}/dense/{name}.f8", buf)
        backend.put_bytes(
            bucket, f"{prefix}/dense/{name}.shape.json", _jdump(list(arr.shape)).encode()
        )

    return backend.uri(bucket, prefix)


# --------------------------------------------------------------------------------------
# read
# --------------------------------------------------------------------------------------


def _hand_from(
    kp: np.ndarray, betas: np.ndarray, theta: np.ndarray, orient: np.ndarray,
    wrist: SE3 | None,
) -> HandState | None:
    has_kp = not np.isnan(kp).all()
    has_mano = not np.isnan(betas).all()
    if not (has_kp or has_mano or wrist is not None):
        return None
    return HandState(
        mano=(
            MANOParams(
                betas=tuple(betas), theta=tuple(theta), global_orient=tuple(orient)
            )
            if has_mano
            else None
        ),
        keypoints_3d=tuple(tuple(row) for row in kp) if has_kp else None,
        wrist_pose=wrist,
    )


def read_episode(
    backend: StorageBackend,
    episode_id: str,
    version: int = 1,
    bucket: Bucket = Bucket.WORK,
) -> CanonicalEpisode:
    """Reconstruct a validated CanonicalEpisode. Round-trips write_episode bit-exactly."""
    prefix = canonical_prefix(episode_id, version)
    try:
        meta = json.loads(backend.get_bytes(bucket, f"{prefix}/episode.json"))
    except StorageError as exc:
        raise StorageError(f"no canonical episode at {backend.uri(bucket, prefix)}") from exc

    table = pq.read_table(pa.BufferReader(backend.get_bytes(bucket, f"{prefix}/frames.parquet")))
    cols = table.to_pydict()

    dense = {}
    for name in ("keypoints_3d", "mano_betas", "mano_theta", "mano_orient"):
        shape = tuple(json.loads(backend.get_bytes(bucket, f"{prefix}/dense/{name}.shape.json")))
        raw = backend.get_bytes(bucket, f"{prefix}/dense/{name}.f8")
        dense[name] = np.frombuffer(raw, dtype=np.float64).reshape(shape)

    rig = RigType(meta["rig"])
    frames = []
    for i in range(table.num_rows):
        hands: dict[Side, HandState] = {}
        contact: dict[Side, dict[Finger, ContactReading]] = {}
        fj_human: dict[Side, tuple[float, ...]] = {}
        fj_robot: dict[Side, tuple[float, ...]] = {}

        for j, side in enumerate(SIDES):
            s = side.value
            wrist = _se3_from(cols[f"wrist_{s}_pos"][i], cols[f"wrist_{s}_quat"][i])
            hand = _hand_from(
                dense["keypoints_3d"][i, j],
                dense["mano_betas"][i, j],
                dense["mano_theta"][i, j],
                dense["mano_orient"][i, j],
                wrist,
            )
            if hand is not None:
                hands[side] = hand

            confs = cols[f"contact_{s}_conf"][i]
            if confs is not None:
                srcs = cols[f"contact_{s}_src"][i]
                contact[side] = {
                    f: ContactReading(confidence=confs[k], source=Provenance(srcs[k]))
                    for k, f in enumerate(FINGERS)
                }

            if (v := cols[f"finger_joints_human_{s}"][i]) is not None:
                fj_human[side] = tuple(v)
            if (v := cols[f"finger_joints_robotspace_{s}"][i]) is not None:
                fj_robot[side] = tuple(v)

        images = json.loads(cols["images_json"][i])
        depth = json.loads(cols["depth_json"][i])
        objects = json.loads(cols["objects_json"][i])
        provenance = {
            k: Provenance(v) for k, v in json.loads(cols["provenance_json"][i]).items()
        }

        frames.append(
            CanonicalFrame(
                t=cols["t"][i],
                rig=rig,
                episode_id=episode_id,
                frame_idx=cols["frame_idx"][i],
                images=images,
                camera_pose=_se3_from(cols["camera_pose_pos"][i], cols["camera_pose_quat"][i]),
                hands=hands,
                finger_joints_human=fj_human or None,
                finger_joints_robotspace=fj_robot or None,
                objects=objects,
                depth=depth,
                contact=contact or None,
                interaction_state=cols["interaction_state"][i],
                confidence=json.loads(cols["confidence_json"][i]),
                provenance=provenance,
            )
        )

    return CanonicalEpisode(**meta, frames=tuple(frames))
