"""Full-body human-to-humanoid retargeting through GMR.

This module integrates the reference implementation for:

    Araujo et al., "Retargeting Matters: General Motion Retargeting for
    Humanoid Motion Tracking", arXiv:2510.02252.

It intentionally does *not* replace :mod:`actuate.retarget.arm`.  The existing
Franka path consumes an egocentric wrist trajectory, whereas GMR requires a
full-body motion source such as BVH or SMPL-X.

The reference package owns the paper's IK implementation.  Actuate supplies a
headless source adapter, sequential execution, safe NPZ/JSON artifacts,
post-processing, diagnostics, and an off-screen preview.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import ssl
import tempfile
import urllib.request
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

PAPER_URL = "https://arxiv.org/abs/2510.02252"
REFERENCE_URL = "https://github.com/YanjieZe/GMR"
REFERENCE_COMMIT = "bb1bbe40774794fceb2a7c579a3464a28e68c844"
REFERENCE_RAW_URL = (
    f"https://raw.githubusercontent.com/YanjieZe/GMR/{REFERENCE_COMMIT}"
)
G1_IK_CONFIGS = (
    "bvh_lafan1_to_g1.json",
    "bvh_nokov_to_g1.json",
    "bvh_xsens_to_g1.json",
)

METHOD_STAGES = (
    "human_robot_key_body_matching",
    "cartesian_rest_pose_alignment",
    "non_uniform_local_scaling",
    "rotation_and_endpoint_differential_ik",
    "rotation_and_translation_fine_tuning",
)

SOURCE_ALIASES = {
    "lafan1": "bvh_lafan1",
    "nokov": "bvh_nokov",
    "xsens": "bvh_xsens",
    "bvh_lafan1": "bvh_lafan1",
    "bvh_nokov": "bvh_nokov",
    "bvh_xsens": "bvh_xsens",
}


class GMRDependencyError(RuntimeError):
    """The optional reference implementation is not installed."""


@dataclass(frozen=True)
class GMRQualityReport:
    """Artifact checks measured from the resulting MuJoCo trajectory."""

    deliverable: str
    finite: bool
    ground_height_before_m: float
    ground_height_after_m: float
    ground_correction_m: float
    joint_limit_violations: int
    max_joint_speed_rad_s: float
    velocity_clipped_frames: tuple[int, ...]
    velocity_spike_frames: tuple[int, ...]
    self_collision_frames: tuple[int, ...]
    self_collision_contacts: int
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class GMRResult:
    """A retargeted humanoid motion plus its provenance and diagnostics."""

    qpos: np.ndarray
    fps: float
    robot: str
    source_format: str
    source_path: str
    joint_names: tuple[str, ...]
    body_names: tuple[str, ...]
    model_xml: str
    backend_version: str
    quality: GMRQualityReport

    @property
    def root_position(self) -> np.ndarray:
        return self.qpos[:, :3]

    @property
    def root_quaternion_wxyz(self) -> np.ndarray:
        return self.qpos[:, 3:7]

    @property
    def joint_position(self) -> np.ndarray:
        return self.qpos[:, 7:]

    @property
    def frame_count(self) -> int:
        return int(self.qpos.shape[0])

    @property
    def duration_seconds(self) -> float:
        return self.frame_count / self.fps

    def summary(self) -> str:
        return (
            f"GMR {self.source_format} -> {self.robot}: {self.frame_count} frames, "
            f"{len(self.joint_names)} joints, {self.duration_seconds:.2f}s, "
            f"{self.quality.deliverable}"
        )

    def write(self, output_dir: str | Path) -> tuple[Path, Path]:
        """Write a non-executable motion archive and a human-readable report."""

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        motion_path = output_dir / "motion.npz"
        report_path = output_dir / "report.json"

        np.savez_compressed(
            motion_path,
            qpos=self.qpos,
            root_position=self.root_position,
            root_quaternion_wxyz=self.root_quaternion_wxyz,
            joint_position=self.joint_position,
            joint_names=np.asarray(self.joint_names),
            body_names=np.asarray(self.body_names),
            fps=np.asarray(self.fps),
        )

        report = {
            "algorithm": {
                "name": "General Motion Retargeting (GMR)",
                "paper": PAPER_URL,
                "reference_implementation": REFERENCE_URL,
                "reference_commit": REFERENCE_COMMIT,
                "backend_version": self.backend_version,
                "method_stages": list(METHOD_STAGES),
                "sequential_warm_start": True,
                "global_ground_postprocess": True,
            },
            "source": {
                "path": self.source_path,
                "format": self.source_format,
            },
            "target": {
                "robot": self.robot,
                "joint_names": list(self.joint_names),
                "body_names": list(self.body_names),
            },
            "motion": {
                "frames": self.frame_count,
                "fps": self.fps,
                "duration_seconds": self.duration_seconds,
                "quaternion_order": "wxyz",
                "artifact": motion_path.name,
            },
            "quality": {
                **asdict(self.quality),
                "velocity_clipped_frames": list(self.quality.velocity_clipped_frames),
                "velocity_spike_frames": list(self.quality.velocity_spike_frames),
                "self_collision_frames": list(self.quality.self_collision_frames),
                "warnings": list(self.quality.warnings),
            },
        }
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        return motion_path, report_path


def _import_reference() -> tuple[Any, Mapping[str, Mapping[str, Path]]]:
    try:
        from general_motion_retargeting.motion_retarget import GeneralMotionRetargeting
        from general_motion_retargeting.params import IK_CONFIG_DICT
    except ImportError as exc:
        raise GMRDependencyError(
            "The optional GMR backend is not installed. Run "
            "`pip install -e '.[humanoid-retarget]'` from Actuate-Pipeline."
        ) from exc
    return GeneralMotionRetargeting, IK_CONFIG_DICT


def default_reference_root() -> Path:
    """Versioned user cache for the assets omitted from GMR's Python wheel."""

    actuate_home = os.environ.get("ACTUATE_HOME")
    root = Path(actuate_home) if actuate_home else Path.home() / ".actuate"
    return root / "vendor" / "gmr" / REFERENCE_COMMIT


def _reference_root_is_complete(root: Path) -> bool:
    model = root / "assets" / "unitree_g1" / "g1_mocap_29dof.xml"
    config_root = root / "general_motion_retargeting" / "ik_configs"
    if not model.is_file() or not all(
        (config_root / config).is_file() for config in G1_IK_CONFIGS
    ):
        return False
    try:
        model_tree = ET.parse(model)
    except (ET.ParseError, OSError):
        return False
    mesh_root = model.parent / "meshes"
    return all(
        (mesh_root / mesh.attrib["file"]).is_file()
        for mesh in model_tree.findall(".//mesh")
        if "file" in mesh.attrib
    )


def install_reference_assets(destination: str | Path | None = None) -> Path:
    """Download the pinned upstream archive into Actuate's versioned asset cache.

    GMR 0.2.0's wheel does not include its MJCF, mesh, or IK JSON package data.
    This explicit setup operation keeps the runtime reproducible while avoiding a
    machine-specific checkout path.
    """

    destination = Path(destination) if destination else default_reference_root()
    if _reference_root_is_complete(destination):
        return destination
    if destination.exists() and any(destination.iterdir()):
        raise ValueError(
            f"refusing to replace incomplete non-empty GMR asset directory: {destination}"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(tempfile.mkdtemp(prefix="actuate-gmr-", dir=destination.parent))
    staged = temporary_root / "staged"

    try:
        try:
            import certifi
        except ImportError as exc:
            raise GMRDependencyError(
                "GMR asset setup requires certifi for HTTPS verification."
            ) from exc
        ca_file = certifi.where()

        def download(relative: str) -> None:
            target = staged / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            # Each worker gets its own SSL context; sharing one is unnecessary.
            context = ssl.create_default_context(cafile=ca_file)
            with (
                urllib.request.urlopen(
                    f"{REFERENCE_RAW_URL}/{relative}",
                    context=context,
                ) as source,
                target.open("wb") as sink,
            ):
                shutil.copyfileobj(source, sink)

        model_relative = "assets/unitree_g1/g1_mocap_29dof.xml"
        initial_files = [
            model_relative,
            "assets/unitree_g1/LICENSE",
            *[
                f"general_motion_retargeting/ik_configs/{name}"
                for name in G1_IK_CONFIGS
            ],
        ]
        with ThreadPoolExecutor(max_workers=4) as executor:
            list(executor.map(download, initial_files))

        model_tree = ET.parse(staged / model_relative)
        mesh_files = sorted(
            {
                mesh.attrib["file"]
                for mesh in model_tree.findall(".//mesh")
                if "file" in mesh.attrib
            }
        )
        with ThreadPoolExecutor(max_workers=8) as executor:
            list(
                executor.map(
                    download,
                    [
                        f"assets/unitree_g1/meshes/{mesh_file}"
                        for mesh_file in mesh_files
                    ],
                )
            )

        if not _reference_root_is_complete(staged):
            raise ValueError("downloaded GMR data is missing required G1 assets")
        if destination.exists():
            destination.rmdir()
        staged.replace(destination)
    finally:
        shutil.rmtree(temporary_root, ignore_errors=True)
    return destination


def _tail_after(path: Path, marker: str) -> Path:
    parts = path.parts
    indices = [index for index, part in enumerate(parts) if part == marker]
    if not indices:
        return Path(path.name)
    return Path(*parts[indices[-1] + 1 :])


def configure_reference_assets(
    reference_root: str | Path | None,
    *,
    source_format: str,
    robot: str,
) -> Path:
    """Point the installed GMR code at a complete upstream asset checkout/cache."""

    from general_motion_retargeting import params

    candidates: list[Path] = []
    if reference_root is not None:
        candidates.append(Path(reference_root))
    configured = os.environ.get("ACTUATE_GMR_ROOT")
    if configured:
        candidates.append(Path(configured))

    # Editable upstream installs already have package data next to the source tree.
    installed_root = Path(params.ASSET_ROOT).parent
    candidates.extend((installed_root, default_reference_root()))
    root = next((candidate for candidate in candidates if _reference_root_is_complete(candidate)), None)
    if root is None:
        checked = ", ".join(str(candidate) for candidate in candidates)
        raise GMRDependencyError(
            "GMR's Python package is installed, but its wheel omits robot/IK assets. "
            "Run `actuate retarget setup-humanoid`, or pass `--gmr-root` pointing to "
            f"a GMR checkout. Checked: {checked}"
        )

    asset_root = root / "assets"
    config_root = root / "general_motion_retargeting" / "ik_configs"
    for target_robot, old_path in list(params.ROBOT_XML_DICT.items()):
        params.ROBOT_XML_DICT[target_robot] = asset_root / _tail_after(
            Path(old_path), "assets"
        )
    for robot_configs in params.IK_CONFIG_DICT.values():
        for target_robot, old_path in list(robot_configs.items()):
            robot_configs[target_robot] = config_root / _tail_after(
                Path(old_path), "ik_configs"
            )
    params.ASSET_ROOT = asset_root
    params.IK_CONFIG_ROOT = config_root

    robot_xml = Path(params.ROBOT_XML_DICT.get(robot, ""))
    ik_config = Path(params.IK_CONFIG_DICT.get(source_format, {}).get(robot, ""))
    if not robot_xml.is_file() or not ik_config.is_file():
        raise ValueError(
            f"asset cache has no complete {source_format} -> {robot} mapping "
            f"(robot={robot_xml}, config={ik_config})"
        )
    return root


def _backend_version() -> str:
    try:
        return metadata.version("general-motion-retargeting")
    except metadata.PackageNotFoundError:
        return "unknown"


def normalize_source_format(source_format: str) -> str:
    try:
        return SOURCE_ALIASES[source_format.lower()]
    except KeyError as exc:
        supported = ", ".join(sorted(SOURCE_ALIASES))
        raise ValueError(f"unsupported source format {source_format!r}; choose one of: {supported}") from exc


def _frame_time_from_bvh(path: Path) -> float | None:
    """Read BVH timing without loading the complete motion a second time."""

    with path.open(errors="replace") as handle:
        for _ in range(10_000):
            line = handle.readline()
            if not line:
                break
            match = re.match(r"\s*Frame\s+Time\s*:\s*([0-9.eE+-]+)", line)
            if match:
                value = float(match.group(1))
                return value if value > 0 else None
    return None


def _load_lafan_family(path: Path, source_format: str) -> tuple[list[dict[str, Any]], float, float]:
    try:
        from general_motion_retargeting.utils.lafan1 import load_bvh_file
    except ImportError as exc:
        raise GMRDependencyError(
            "GMR's BVH loader is unavailable; reinstall `.[humanoid-retarget]`."
        ) from exc

    family = source_format.removeprefix("bvh_")
    frames, human_height = load_bvh_file(str(path), format=family)
    frame_time = _frame_time_from_bvh(path)
    return frames, float(human_height), frame_time or (1.0 / 30.0)


def _load_xsens_headless(
    path: Path,
    *,
    scale: float,
    start: int | None,
    end: int | None,
    reset_to_zero: bool,
) -> tuple[list[dict[str, Any]], float, float]:
    """Load the official Xsens sample without importing its PyQt curve editor.

    GMR's current ``utils.xsens`` module imports PyQt6 at module load time even
    for non-interactive conversion.  This is the same parser/FK path with zero
    editor offsets, which keeps server, CI, and headless Mac execution possible.
    """

    try:
        import general_motion_retargeting.utils.lafan_vendor.utils as quat_utils
        from general_motion_retargeting.utils.xsens_vendor.BVHParser import Anim, BVHParser
    except ImportError as exc:
        raise GMRDependencyError(
            "GMR's Xsens BVH parser is unavailable; reinstall `.[humanoid-retarget]`."
        ) from exc

    parser = BVHParser(axis_order="zxy", scale=scale)
    rotations, _positions = parser.parse(
        path.read_text(), start=start, end=end, reset_to_zero=reset_to_zero
    )
    zero_offsets = np.zeros_like(rotations)
    quats, positions, offsets, parents = parser._MOTION_data_post_processing(
        rotations + zero_offsets,
        np.copy(parser.positions),
        reset_to_zero=reset_to_zero,
    )
    animation = Anim(quats, positions, offsets, parents, parser.names)
    global_quats, global_positions = quat_utils.quat_fk(
        animation.quats, animation.pos, animation.parents
    )

    frames: list[dict[str, Any]] = []
    for frame_index in range(animation.pos.shape[0]):
        frame = {
            bone: (
                np.asarray(global_positions[frame_index, index]),
                np.asarray(global_quats[frame_index, index]),
            )
            for index, bone in enumerate(animation.bones)
        }
        frame["LeftFootMod"] = (
            np.asarray(frame["LeftAnkle"][0]),
            np.asarray(frame["LeftAnkle"][1]),
        )
        frame["RightFootMod"] = (
            np.asarray(frame["RightAnkle"][0]),
            np.asarray(frame["RightAnkle"][1]),
        )
        frames.append(frame)

    heights = [
        float(frame["Head_end_site"][0][2])
        - min(
            float(frame["LeftToe_end_site"][0][2]),
            float(frame["RightToe_end_site"][0][2]),
        )
        for frame in frames
    ]
    valid_heights = [height for height in heights if np.isfinite(height) and height > 0.5]
    if not valid_heights:
        raise ValueError("could not infer a valid human height from the Xsens BVH")
    return frames, float(np.median(valid_heights)), float(parser.frame_time)


def load_bvh_frames(
    path: str | Path,
    *,
    source_format: str,
    scale: float = 0.01,
    start: int | None = None,
    end: int | None = None,
    reset_to_zero: bool = False,
    max_frames: int | None = None,
) -> tuple[list[dict[str, Any]], float, float]:
    """Load a supported BVH file as GMR body-pose frames."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"BVH source does not exist: {path}")
    if path.suffix.lower() != ".bvh":
        raise ValueError(f"expected a .bvh full-body motion file, got: {path.name}")
    if start is not None and start < 0:
        raise ValueError("start must be >= 0")
    if end is not None and start is not None and end <= start:
        raise ValueError("end must be greater than start")
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be greater than zero")

    source_format = normalize_source_format(source_format)
    if source_format == "bvh_xsens":
        frames, height, frame_time = _load_xsens_headless(
            path,
            scale=scale,
            start=start,
            end=end,
            reset_to_zero=reset_to_zero,
        )
    else:
        frames, height, frame_time = _load_lafan_family(path, source_format)
        frames = frames[slice(start, end)]

    if max_frames is not None:
        frames = frames[:max_frames]
    if not frames:
        raise ValueError("the selected BVH range contains no motion frames")
    return frames, height, frame_time


def retarget_frames(
    frames: Iterable[Mapping[str, Any]],
    retargeter: Any,
    *,
    progress: Callable[[int, int], None] | None = None,
) -> np.ndarray:
    """Run frames in sequence so every solve warm-starts from the previous pose."""

    frame_list = list(frames)
    qpos: list[np.ndarray] = []
    for index, frame in enumerate(frame_list):
        # NumPy/OpenBLAS on Apple Silicon can leave benign floating-point status
        # flags after a finite matrix multiply inside Mink. The result is checked
        # for finiteness below and again after the complete trajectory.
        with np.errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
            pose = np.asarray(retargeter.retarget(dict(frame)), dtype=np.float64)
        if pose.ndim != 1:
            raise ValueError(f"GMR returned a non-vector pose at frame {index}: {pose.shape}")
        qpos.append(pose.copy())
        if progress is not None:
            progress(index + 1, len(frame_list))
    result = np.stack(qpos)
    if result.ndim != 2 or result.shape[0] == 0:
        raise ValueError("GMR returned no robot poses")
    return result


def _joint_metadata(model: Any, mj: Any) -> tuple[tuple[str, ...], tuple[int, ...]]:
    names: list[str] = []
    addresses: list[int] = []
    for joint_id in range(model.njnt):
        joint_type = int(model.jnt_type[joint_id])
        if joint_type == int(mj.mjtJoint.mjJNT_FREE):
            continue
        width = 4 if joint_type == int(mj.mjtJoint.mjJNT_BALL) else 1
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, joint_id)
        for component in range(width):
            names.append(name if width == 1 else f"{name}[{component}]")
            addresses.append(int(model.jnt_qposadr[joint_id]) + component)
    return tuple(names), tuple(addresses)


def _body_names(model: Any, mj: Any) -> tuple[str, ...]:
    return tuple(
        mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, body_id) or f"body_{body_id}"
        for body_id in range(1, model.nbody)
    )


def _trajectory_measurements(
    qpos: np.ndarray,
    model: Any,
    mj: Any,
) -> tuple[float, tuple[int, ...], int]:
    data = mj.MjData(model)
    minimum_height = np.inf
    collision_frames: list[int] = []
    collision_contacts = 0

    for frame_index, pose in enumerate(qpos):
        data.qpos[:] = pose
        mj.mj_forward(model, data)
        minimum_height = min(minimum_height, float(np.min(data.xpos[1:, 2])))

        frame_collisions = 0
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            body1 = int(model.geom_bodyid[int(contact.geom1)])
            body2 = int(model.geom_bodyid[int(contact.geom2)])
            if body1 > 0 and body2 > 0 and body1 != body2 and float(contact.dist) < -1e-4:
                frame_collisions += 1
        if frame_collisions:
            collision_frames.append(frame_index)
            collision_contacts += frame_collisions

    return float(minimum_height), tuple(collision_frames), collision_contacts


def postprocess_and_measure(
    qpos: np.ndarray,
    model: Any,
    *,
    fps: float,
    ground_align: bool = True,
    velocity_spike_threshold: float = 3.0 * np.pi,
    velocity_limit_rad_s: float | None = None,
) -> tuple[np.ndarray, GMRQualityReport, tuple[str, ...], tuple[str, ...]]:
    """Apply the paper's global ground correction and measure common artifacts."""

    try:
        import mujoco as mj
    except ImportError as exc:
        raise GMRDependencyError("MuJoCo is required for GMR post-processing.") from exc

    qpos = np.asarray(qpos, dtype=np.float64)
    if qpos.ndim != 2 or qpos.shape[1] != model.nq:
        raise ValueError(f"expected qpos with shape (frames, {model.nq}), got {qpos.shape}")
    if fps <= 0:
        raise ValueError("fps must be greater than zero")

    finite = bool(np.all(np.isfinite(qpos)))
    if not finite:
        raise ValueError("GMR produced NaN or infinite joint values")

    corrected = qpos.copy()
    joint_names, joint_addresses = _joint_metadata(model, mj)
    clipped_frames: list[int] = []
    if velocity_limit_rad_s is not None:
        if velocity_limit_rad_s <= 0:
            raise ValueError("velocity_limit_rad_s must be greater than zero")
        max_delta = velocity_limit_rad_s / fps
        for frame_index in range(1, len(corrected)):
            previous = corrected[frame_index - 1, joint_addresses]
            desired = corrected[frame_index, joint_addresses]
            clipped = previous + np.clip(desired - previous, -max_delta, max_delta)
            if not np.allclose(clipped, desired, atol=1e-12, rtol=0):
                clipped_frames.append(frame_index)
            corrected[frame_index, joint_addresses] = clipped

    height_before, _pre_collision_frames, _pre_collision_count = _trajectory_measurements(
        corrected, model, mj
    )
    correction = height_before if ground_align else 0.0
    if ground_align:
        corrected[:, 2] -= correction
    height_after, collision_frames, collision_count = _trajectory_measurements(
        corrected, model, mj
    )

    limit_violations = 0
    for joint_id in range(model.njnt):
        if not bool(model.jnt_limited[joint_id]):
            continue
        address = int(model.jnt_qposadr[joint_id])
        lower, upper = (float(value) for value in model.jnt_range[joint_id])
        values = corrected[:, address]
        limit_violations += int(np.count_nonzero((values < lower - 1e-6) | (values > upper + 1e-6)))

    if len(corrected) > 1 and joint_addresses:
        velocity = np.diff(corrected[:, joint_addresses], axis=0) * fps
        speed_by_frame = np.max(np.abs(velocity), axis=1)
        max_speed = float(np.max(speed_by_frame))
        spike_frames = tuple(
            (
                np.flatnonzero(speed_by_frame > velocity_spike_threshold + 1e-9)
                + 1
            ).tolist()
        )
    else:
        max_speed = 0.0
        spike_frames = ()

    warnings: list[str] = []
    if limit_violations:
        warnings.append(f"{limit_violations} joint-limit violations")
    if spike_frames:
        warnings.append(
            f"{len(spike_frames)} frames exceed {velocity_spike_threshold:.3f} rad/s"
        )
    if collision_frames:
        warnings.append(f"self-collision detected in {len(collision_frames)} frames")
    if abs(height_after) > 1e-5:
        warnings.append(f"ground alignment residual is {height_after:.6f} m")

    deliverable = (
        "PASS"
        if finite and limit_violations == 0 and not collision_frames
        else "BLOCKED"
    )
    quality = GMRQualityReport(
        deliverable=deliverable,
        finite=finite,
        ground_height_before_m=height_before,
        ground_height_after_m=height_after,
        ground_correction_m=correction,
        joint_limit_violations=limit_violations,
        max_joint_speed_rad_s=max_speed,
        velocity_clipped_frames=tuple(clipped_frames),
        velocity_spike_frames=spike_frames,
        self_collision_frames=collision_frames,
        self_collision_contacts=collision_count,
        warnings=tuple(warnings),
    )
    return corrected, quality, joint_names, _body_names(model, mj)


def retarget_bvh(
    source: str | Path,
    *,
    source_format: str,
    robot: str = "unitree_g1",
    scale: float = 0.01,
    start: int | None = None,
    end: int | None = None,
    max_frames: int | None = None,
    reset_to_zero: bool = False,
    fps: float | None = None,
    actual_human_height: float | None = None,
    solver: str = "daqp",
    damping: float = 0.5,
    velocity_limit: bool = True,
    ground_align: bool = True,
    verbose: bool = False,
    progress: Callable[[int, int], None] | None = None,
    reference_root: str | Path | None = None,
) -> GMRResult:
    """Retarget a full-body BVH motion to a supported humanoid."""

    source = Path(source)
    source_format = normalize_source_format(source_format)
    retargeter_class, ik_configs = _import_reference()
    configure_reference_assets(
        reference_root,
        source_format=source_format,
        robot=robot,
    )
    if source_format not in ik_configs or robot not in ik_configs[source_format]:
        supported = ", ".join(sorted(ik_configs.get(source_format, {}))) or "none"
        raise ValueError(
            f"GMR has no {source_format} -> {robot} mapping; supported robots: {supported}"
        )

    frames, inferred_height, frame_time = load_bvh_frames(
        source,
        source_format=source_format,
        scale=scale,
        start=start,
        end=end,
        reset_to_zero=reset_to_zero,
        max_frames=max_frames,
    )
    motion_fps = float(fps) if fps is not None else 1.0 / frame_time
    if motion_fps <= 0:
        raise ValueError("fps must be greater than zero")

    retargeter = retargeter_class(
        src_human=source_format,
        tgt_robot=robot,
        actual_human_height=actual_human_height or inferred_height,
        solver=solver,
        damping=damping,
        verbose=verbose,
        use_velocity_limit=velocity_limit,
    )
    raw_qpos = retarget_frames(frames, retargeter, progress=progress)
    qpos, quality, joint_names, body_names = postprocess_and_measure(
        raw_qpos,
        retargeter.model,
        fps=motion_fps,
        ground_align=ground_align,
        velocity_limit_rad_s=3.0 * np.pi if velocity_limit else None,
    )
    return GMRResult(
        qpos=qpos,
        fps=motion_fps,
        robot=robot,
        source_format=source_format,
        source_path=str(source.resolve()),
        joint_names=joint_names,
        body_names=body_names,
        model_xml=str(Path(retargeter.xml_file).resolve()),
        backend_version=_backend_version(),
        quality=quality,
    )


def render_preview(
    result: GMRResult,
    output: str | Path,
    *,
    width: int = 720,
    height: int = 720,
    max_frames: int = 300,
) -> Path:
    """Render an MP4/GIF off-screen; no desktop MuJoCo viewer is required."""

    try:
        import imageio.v2 as imageio
        import mujoco as mj
    except ImportError as exc:
        raise GMRDependencyError("Preview rendering requires MuJoCo and imageio[ffmpeg].") from exc

    output = Path(output)
    if output.suffix.lower() not in {".mp4", ".gif"}:
        raise ValueError("preview output must end in .mp4 or .gif")
    if max_frames <= 0:
        raise ValueError("max_frames must be greater than zero")
    output.parent.mkdir(parents=True, exist_ok=True)

    model = mj.MjModel.from_xml_path(result.model_xml)
    data = mj.MjData(model)
    camera = mj.MjvCamera()
    mj.mjv_defaultCamera(camera)
    camera.type = mj.mjtCamera.mjCAMERA_FREE
    camera.distance = 2.2
    camera.azimuth = 135
    camera.elevation = -12

    stride = max(1, int(np.ceil(result.frame_count / max_frames)))
    indices = range(0, result.frame_count, stride)
    preview_fps = result.fps / stride
    writer_kwargs = {"fps": preview_fps} if output.suffix.lower() == ".mp4" else {
        "duration": 1000.0 / preview_fps,
        "loop": 0,
    }

    with (
        mj.Renderer(model, height=height, width=width) as renderer,
        imageio.get_writer(output, **writer_kwargs) as writer,
    ):
        for frame_index in indices:
            data.qpos[:] = result.qpos[frame_index]
            mj.mj_forward(model, data)
            camera.lookat[:] = result.qpos[frame_index, :3] + np.array([0.0, 0.0, 0.15])
            renderer.update_scene(data, camera=camera)
            writer.append_data(renderer.render())
    return output


def cli_progress_printer(every: int = 25) -> Callable[[int, int], None]:
    """Small reusable progress callback for non-interactive terminals."""

    def report(done: int, total: int) -> None:
        if done == 1 or done == total or done % every == 0:
            print(f"retargeting {done}/{total} frames")

    return report
