"""PII redaction: the pass actually blurs the region it's told to, earns pii_status=PASSED,
and that is exactly what flips an owned episode from blocked to deliverable. The detector is
injected so the blur+gate logic is tested deterministically (Haar recall is a separate axis)."""

from __future__ import annotations

import numpy as np
import pytest

from actuate.config import ConsentStatus, PiiStatus
from actuate.io import redact

cv2 = pytest.importorskip("cv2")


def _noisy_video(path, w=64, h=64, n=5):
    """A video whose top-left 32x32 is high-frequency noise -- blur must visibly smooth it."""
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (w, h))
    rng = np.random.default_rng(0)
    for _ in range(n):
        frame = np.zeros((h, w, 3), np.uint8)
        frame[:32, :32] = rng.integers(0, 255, (32, 32, 3), np.uint8)
        vw.write(frame)
    vw.release()


def test_redaction_blurs_the_targeted_region(tmp_path):
    src = tmp_path / "in.mp4"
    _noisy_video(src)
    fixed = lambda frame: [(0, 0, 32, 32)]                       # inject one region
    report = redact.redact_video(src, tmp_path / "out.mp4", detector=fixed)

    assert report.regions_blurred == 5                           # one per frame
    assert report.output.exists()
    # the redacted region must be smoother than the original noisy input
    cap = cv2.VideoCapture(str(report.output))
    ok, frame = cap.read(); cap.release()
    assert ok
    assert frame[:32, :32].var() < 4000                          # noise (var ~5000) knocked down


def test_status_is_passed_after_a_pass(tmp_path):
    src = tmp_path / "in.mp4"
    _noisy_video(src, n=2)
    report = redact.redact_video(src, tmp_path / "out.mp4", detector=lambda f: [])
    assert report.status is PiiStatus.PASSED                     # a completed pass earns PASSED
    assert report.regions_blurred == 0                           # ...even with nothing detected


def test_redact_session_writes_the_downstream_filename(tmp_path):
    sess = tmp_path / "sess"
    sess.mkdir()
    _noisy_video(sess / "compressed.mp4", n=2)
    report = redact.redact_session(sess, detector=lambda f: [(0, 0, 32, 32)])
    assert (sess / "redacted_compressed.mp4").exists()           # the name perception prefers
    assert report.output.name == "redacted_compressed.mp4"


def test_redaction_flips_the_deliverable_gate(canonical_episode_factory=None):
    """The whole point: PASSED + GRANTED is what makes is_deliverable True."""
    from actuate.io.consent import check_deliverable, ConsentViolation

    # pending pii blocks even with consent granted...
    with pytest.raises(ConsentViolation, match="pii_status"):
        check_deliverable("ep0", ConsentStatus.GRANTED, PiiStatus.PENDING)
    # ...and a completed redaction pass (PASSED) is what lets it through.
    check_deliverable("ep0", ConsentStatus.GRANTED, PiiStatus.PASSED)   # no raise
