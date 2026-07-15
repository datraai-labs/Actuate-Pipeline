"""Part D model-free pieces: the RLE codec, IoU, and the FoundationPose interface stub.

The heavy Grounding DINO + SAM2 tracking gate needs a GPU and the real capture, so it lives in
integration (test_objects_gate.py, GPU-guarded). These are the deterministic parts that must
hold on every commit -- a mask codec that silently corrupts a mask is a data-integrity bug, so
it gets a real bit-exactness test plus a demonstration that a WRONG codec fails it.
"""

from __future__ import annotations

import numpy as np
import pytest

from actuate.perception.objects import (
    FoundationPoseUnavailable,
    bbox_iou,
    decode_rle,
    encode_rle,
    mask_iou,
    synthetic_interface_check,
)
from actuate.perception.objects.foundationpose import FoundationPoseEstimator


@pytest.mark.parametrize(
    "mask",
    [
        np.zeros((10, 12), dtype=bool),                       # empty
        np.ones((8, 9), dtype=bool),                          # full
        np.eye(16, dtype=bool),                               # diagonal
        (np.random.default_rng(0).random((40, 55)) > 0.6),    # random
    ],
)
def test_rle_round_trip_is_bit_exact(mask):
    assert np.array_equal(decode_rle(encode_rle(mask)), mask)


def test_rle_handles_mask_that_starts_true():
    """The leading-run convention (sequence starts with a False run) is the easy thing to get
    wrong -- a mask whose (0,0) pixel is True must still round-trip."""
    m = np.zeros((6, 6), dtype=bool)
    m[0, 0] = True
    assert np.array_equal(decode_rle(encode_rle(m)), m)


def test_a_row_major_decoder_would_FAIL_the_round_trip():
    """Broken-vs-fixed: the codec is column-major (COCO axis order). A decoder that read the
    runs row-major would reconstruct a transposed-ish mask and fail. This proves the
    round-trip test above actually constrains the axis order, rather than passing vacuously."""
    m = np.zeros((5, 7), dtype=bool)
    m[1, 4] = True  # asymmetric so row- vs column-major differ
    s = encode_rle(m)
    shape, _, body = s.partition(":")
    h, w = (int(x) for x in shape.split("x"))
    runs = [int(x) for x in body.split(",")]
    flat = np.zeros(h * w, dtype=bool)
    pos, val = 0, False
    for r in runs:
        if val:
            flat[pos : pos + r] = True
        pos += r
        val = not val
    wrong = flat.reshape((h, w), order="C")  # WRONG order on purpose
    assert not np.array_equal(wrong, m), "row-major decode should not match a column-major encode"


def test_mask_iou_separates_identical_from_disjoint():
    a = np.zeros((10, 10), dtype=bool)
    a[2:6, 2:6] = True
    b = np.zeros((10, 10), dtype=bool)
    b[6:9, 6:9] = True
    assert mask_iou(a, a) == 1.0
    assert mask_iou(a, b) == 0.0
    # partial overlap is strictly between
    c = np.zeros((10, 10), dtype=bool)
    c[4:8, 4:8] = True
    assert 0.0 < mask_iou(a, c) < 1.0


def test_bbox_iou():
    assert bbox_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert bbox_iou((0, 0, 5, 5), (10, 10, 15, 15)) == 0.0
    assert 0.0 < bbox_iou((0, 0, 10, 10), (5, 5, 15, 15)) < 1.0


def test_foundationpose_refuses_without_a_mesh_and_says_why():
    est = FoundationPoseEstimator(mesh_path=None)
    with pytest.raises(FoundationPoseUnavailable, match="mesh"):
        est.estimate(
            np.zeros((8, 8, 3), np.uint8),
            np.full((8, 8), 0.5),
            np.ones((8, 8), bool),
            np.eye(3),
        )


def test_foundationpose_interface_smoke():
    r = synthetic_interface_check()
    assert r["instantiates"]
    assert r["raises_without_mesh"]
    assert r["message_names_the_gap"]
    assert r["hypothesis_shape_ok"]
