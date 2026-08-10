"""
DatraAI Pipeline — Tests for service/api.py
Uses FastAPI's TestClient to verify the wrapper REST endpoints.
"""

import shutil
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from fastapi.testclient import TestClient
from service.api import FRONTEND_STAGES, app, session_jobs


class TestPipelineAPI:
    """Verifies behavior and routing logic of the thin FastAPI wrapper."""

    @pytest.fixture(autouse=True)
    def setup_client(self, tmp_path, monkeypatch):
        # The old test silently depended on one developer's uncommitted
        # ``processed/session_001`` tree. Build the contract fixture here so a fresh clone
        # and CI exercise the same endpoints deterministically.
        monkeypatch.chdir(tmp_path)
        proc = tmp_path / "processed" / "session_001"
        proc.mkdir(parents=True)
        raw = tmp_path / "raw" / "session_001"
        raw.mkdir(parents=True)

        completed = [
            ("01_ingest", 242.3),
            ("02_sync", 1.0),
            ("04d_depth_estimate", 1.0),
            ("04_hand_pose", 1.0),
            ("04c_object_track", 1.0),
            ("05_primitives", 1.0),
            ("06b_episode_segment", 1.0),
            ("07_task_classify", 1.0),
        ]
        lines = [
            f"2026-08-10T00:00:00+00:00 | INFO | [{stage}] ✓ ({duration:.1f}s)"
            for stage, duration in completed
        ]
        lines.append(
            "2026-08-10T00:00:01+00:00 | ERROR | DELIVERY BLOCKED: consent=pending"
        )
        (proc / "pipeline.log").write_text("\n".join(lines), encoding="utf-8")
        (proc / "quality_certificate.json").write_text(
            '{"session_id":"session_001"}', encoding="utf-8"
        )
        (proc / "language_grounding.json").write_text('{}', encoding="utf-8")
        (proc / "task_label.json").write_text(
            '{"episodes":[{"L1_task":"unknown"}]}', encoding="utf-8"
        )
        (proc / "episodes.json").write_text('[]', encoding="utf-8")
        (proc / "hand_pose.json").write_text('[]', encoding="utf-8")
        (proc / "object_tracks.json").write_text('[]', encoding="utf-8")
        (proc / "depth_data.json").write_text('[]', encoding="utf-8")
        (proc / "redacted_compressed.mp4").write_bytes(b"fixture-video")
        session_jobs.clear()
        self.client = TestClient(app)

    def test_health_check(self):
        """Verifies root route returns the HTML dashboard."""
        response = self.client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "DatraAI Console" in response.text

    def test_status_endpoint_existing_session(self):
        """Verifies status endpoint correctly parses existing processed session_001 log."""
        response = self.client.get("/sessions/session_001/status")
        assert response.status_code == 200
        data = response.json()
        
        assert data["session_id"] == "session_001"
        assert data["status"] in ("completed", "failed")  # session_001 was processed but delivery-blocked
        assert data["overall_progress"] == 88
        
        runs = data["pipeline_runs"]
        assert len(runs) == len(FRONTEND_STAGES)
        stages = [r["stage"] for r in runs]
        assert "01_ingest" in stages
        assert "04_hand_pose" in stages
        
        # Ingest should have parsed duration
        ingest_run = next(r for r in runs if r["stage"] == "01_ingest")
        assert ingest_run["status"] == "completed"
        assert ingest_run["duration_seconds"] == 242.3

    def test_status_endpoint_non_existent_session(self):
        """Verifies status endpoint returns not_started for empty/new sessions."""
        response = self.client.get("/sessions/non_existent_session_xyz/status")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "not_started"
        assert data["overall_progress"] == 0
        assert data["current_stage"] is None

    def test_results_endpoint_existing_session(self):
        """Verifies results endpoint aggregates output JSON reports from session_001."""
        response = self.client.get("/sessions/session_001/results")
        assert response.status_code == 200
        data = response.json()
        
        assert "quality_certificate" in data
        assert "language_grounding" in data
        assert "task_label" in data
        assert "episodes" in data
        
        assert data["quality_certificate"]["session_id"] == "session_001"
        assert data["task_label"]["episodes"][0]["L1_task"] == "unknown"

    def test_results_endpoint_missing_session(self):
        """Verifies results endpoint yields 404 for unprocessed sessions."""
        response = self.client.get("/sessions/non_existent_session_xyz/results")
        assert response.status_code == 404

    def test_overlays_endpoint_existing_session(self):
        """Verifies overlays endpoint aggregates perception outputs."""
        response = self.client.get("/sessions/session_001/overlays")
        assert response.status_code == 200
        data = response.json()
        
        assert "hand_pose" in data
        assert "object_tracks" in data
        assert "depth_data" in data
        
        # hand_pose and depth_data exist for session_001
        assert data["hand_pose"] is not None
        assert data["depth_data"] is not None

    def test_video_endpoint_stream(self):
        """Verifies video endpoint serves processed files successfully."""
        response = self.client.get("/sessions/session_001/video")
        assert response.status_code == 200
        assert response.headers["content-type"] == "video/mp4"

    def test_video_endpoint_missing_session(self):
        """Verifies video endpoint yields 404 for unprocessed sessions."""
        response = self.client.get("/sessions/non_existent_session_xyz/video")
        assert response.status_code == 404

    def test_run_endpoint_validation(self):
        """Verifies run endpoint requires either uploads or pre-existing files."""
        # Clean non_existent run folder if it exists
        raw_xyz = Path("raw") / "non_existent_session_xyz"
        if raw_xyz.exists():
            shutil.rmtree(raw_xyz)
            
        # Try running without any files
        response = self.client.post(
            "/sessions/non_existent_session_xyz/run",
            json={"imu_source": "head_mounted", "depth_mode": "stereo"}
        )
        assert response.status_code == 400
        assert "Incomplete raw data" in response.json()["detail"]

    def test_consent_get_and_patch(self):
        """Verifies GET and PATCH /sessions/{session_id}/consent endpoints work as expected."""
        session_id = "test_consent_session"
        raw_dir = Path("raw") / session_id
        raw_dir.mkdir(parents=True, exist_ok=True)
        consent_file = raw_dir / "consent.json"
        
        # Cleanup
        if consent_file.exists():
            consent_file.unlink()
            
        # 1. GET returns pending by default
        response = self.client.get(f"/sessions/{session_id}/consent")
        assert response.status_code == 200
        assert response.json()["consent_status"] == "pending"
        
        # 2. PATCH updates consent to granted
        response = self.client.patch(
            f"/sessions/{session_id}/consent",
            json={"status": "granted"}
        )
        assert response.status_code == 200
        assert response.json()["consent_status"] == "granted"
        
        # 3. GET now returns granted
        response = self.client.get(f"/sessions/{session_id}/consent")
        assert response.status_code == 200
        assert response.json()["consent_status"] == "granted"
        
        # 4. PATCH updates consent to denied
        response = self.client.patch(
            f"/sessions/{session_id}/consent",
            json={"consent_status": "denied"}
        )
        assert response.status_code == 200
        assert response.json()["consent_status"] == "denied"
        
        # Cleanup
        if raw_dir.exists():
            shutil.rmtree(raw_dir)

    def test_consent_blocking_behavior_in_status(self):
        """Verifies that a session with blocked delivery status reports status failed & quality_certificate failed."""
        response = self.client.get("/sessions/session_001/status")
        assert response.status_code == 200
        data = response.json()
        
        # session_001 has consent='pending' in raw and was delivery-blocked in pipeline.log
        assert data["status"] == "failed"
        runs = data["pipeline_runs"]
        
        qc_run = next(r for r in runs if r["stage"] == "09_quality_certificate")
        assert qc_run["status"] == "failed"
        assert "DELIVERY BLOCKED" in qc_run["error_message"]
