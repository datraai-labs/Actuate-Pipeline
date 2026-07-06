"""
DatraAI Pipeline — Tests for utils/s3_utils.py
Tests the pure S3-key-prefix sanitization logic (no AWS calls / boto3 required).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.s3_utils import _sanitize_s3_prefix


class TestSanitizeS3Prefix:
    """Traversal segments in a caller-supplied prefix must not survive sanitization."""

    def test_plain_prefix_unchanged(self):
        assert _sanitize_s3_prefix("deliveries/batch_001") == "deliveries/batch_001"

    def test_strips_parent_traversal(self):
        assert _sanitize_s3_prefix("../../etc/passwd") == "etc/passwd"

    def test_strips_embedded_traversal(self):
        assert _sanitize_s3_prefix("deliveries/../../secret") == "deliveries/secret"

    def test_strips_dot_segments(self):
        assert _sanitize_s3_prefix("./deliveries/./batch_001") == "deliveries/batch_001"

    def test_normalizes_backslashes(self):
        assert _sanitize_s3_prefix("deliveries\\batch_001") == "deliveries/batch_001"

    def test_empty_prefix(self):
        assert _sanitize_s3_prefix("") == ""

    def test_all_traversal_collapses_to_empty(self):
        assert _sanitize_s3_prefix("../..") == ""

    def test_leading_and_trailing_slashes_stripped(self):
        assert _sanitize_s3_prefix("/deliveries/batch_001/") == "deliveries/batch_001"
