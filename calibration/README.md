# Camera Calibration

Per-**device** (not per-session) camera intrinsics, consumed by
[`scripts/04d_depth_estimate.py`](../scripts/04d_depth_estimate.py) to lift
2D hand-pose landmarks and object-track centroids into metric 3D.

## File naming

```
calibration/{device_id}_intrinsics.json
```

`device_id` is read from `session_meta.json`'s optional `"device_id"` field
(defaults to `"default"` if absent — see `config.CAMERA_INTRINSICS_PATH`).
One file per physical camera rig, reused across every session recorded on
that rig.

## Format

See [`example_device_intrinsics.json`](./example_device_intrinsics.json):

```json
{
  "fx": 600.0,
  "fy": 605.0,
  "cx": 320.0,
  "cy": 240.0
}
```

Standard pinhole intrinsics in pixels: `fx`/`fy` are the focal lengths,
`cx`/`cy` are the principal point. Obtain these from a standard camera
calibration procedure (e.g. OpenCV's `cv2.calibrateCamera` on a checkerboard
pattern) for each physical device.

## What happens if a device's file is missing

`04d_depth_estimate.py` does **not** fail — it falls back to an
approximated pinhole model derived from the video's resolution and
`config.CAMERA_DEFAULT_HFOV_DEG` (assumed horizontal field of view). This
keeps the pipeline runnable end-to-end without calibration data, but the
resulting metric 3D coordinates should **not** be trusted for anything
requiring true precision (e.g. real retargeting) — check
`depth_data.json`'s / the intrinsics dict's `"source"` field:

| `source` | Meaning |
|---|---|
| `"calibration_file"` | Real per-device intrinsics were used. |
| `"approximated_no_calibration_file"` | No calibration file found — coordinates are a rough approximation only. |

## Adding a new device

1. Calibrate the rig (checkerboard + OpenCV, or your vendor's tool).
2. Save the four values as `calibration/{your_device_id}_intrinsics.json`.
3. Set `"device_id": "{your_device_id}"` in that device's recorded sessions'
   `session_meta.json` (or have `01_ingest.py` populate it automatically if
   your capture rig reports its own ID).
