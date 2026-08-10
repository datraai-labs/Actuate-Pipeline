from __future__ import annotations

import numpy as np

from actuate.io.redact import haar_face_detector


def test_packaged_haar_detector_loads_and_runs() -> None:
    """The supported OpenCV wheel must include the cascade used by live redaction."""

    detector = haar_face_detector()
    assert detector(np.zeros((64, 64, 3), dtype=np.uint8)) == []
