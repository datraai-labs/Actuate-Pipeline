"""
DatraAI Pipeline — Thin FastAPI Wrapper Service
Provides endpoints to trigger, monitor, stream, and retrieve pipeline data.
"""

import os
import re
import sys
import json
import time
import shutil
import logging
import asyncio
import threading
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta, timezone

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Add parent directory to path to allow importing config and run_pipeline
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config as cfg
import run_pipeline

app = FastAPI(
    title="DatraAI Pipeline wrapper",
    description="Thin REST API wrapper around the robot-learning egocentric video + IMU processing pipeline.",
    version="1.0.0"
)

# Enable CORS for frontend dashboard communication (e.g. localhost:3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ═══════════════════════════════════════════════════════════
# STATE & LOCKING
# ═══════════════════════════════════════════════════════════

# Global locks and in-memory tracker
session_jobs: Dict[str, Dict[str, Any]] = {}
session_jobs_lock = threading.Lock()
pipeline_execution_lock = threading.Lock()

# 9 Frontend stages and the mapping from 15 internal pipeline steps
FRONTEND_STAGES = [
    "01_ingest",
    "02_preprocess",
    "03_depth_estimation",
    "04_hand_pose",
    "05_object_detection",
    "06_phase_segmentation",
    "07_episode_extraction",
    "08_task_labeling",
    "09_quality_certificate"
]

INTERNAL_TO_FRONTEND = {
    "01_ingest": "01_ingest",
    "02_sync": "02_preprocess",
    "03_qa": "02_preprocess",
    "03b_privacy_redact": "02_preprocess",
    "04d_depth_estimate": "03_depth_estimation",
    "04_hand_pose": "04_hand_pose",
    "04c_object_track": "05_object_detection",
    "05_primitives": "06_phase_segmentation",
    "06_phase_segment": "06_phase_segmentation",
    "06b_episode_segment": "07_episode_extraction",
    "07_task_classify": "08_task_labeling",
    "08_validate": "08_task_labeling",
    "09_language_ground": "08_task_labeling",
    "10_eis": "09_quality_certificate",
    "11_package": "09_quality_certificate"
}

PIPELINE_STAGES_LIST = [
    "01_ingest",
    "02_sync",
    "03_qa",
    "03b_privacy_redact",
    "04_hand_pose",
    "04c_object_track",
    "04d_depth_estimate",
    "05_primitives",
    "06_phase_segment",
    "06b_episode_segment",
    "07_task_classify",
    "08_validate",
    "09_language_ground",
    "10_eis",
    "11_package"
]

STAGE_REGEX = re.compile(r"\[([\w_]+)\]\s+([^\s]+)(?:\s+\(([\d\.]+)s\))?")

# Helper to modify job tracker in thread-safe manner
def update_job_state(session_id: str, **kwargs):
    with session_jobs_lock:
        if session_id not in session_jobs:
            session_jobs[session_id] = {
                "status": "queued",
                "current_stage": None,
                "overall_progress": 0,
                "started_at": None,
                "completed_at": None,
                "error": None
            }
        session_jobs[session_id].update(kwargs)

# ═══════════════════════════════════════════════════════════
# LOG PARSING & RECOVERY
# ═══════════════════════════════════════════════════════════

def get_session_status_from_log(session_id: str) -> Dict[str, Any]:
    """
    Parses pipeline.log to reconstruct per-stage timing metrics, log outputs,
    and stage statuses. Resolves current executing stage and overall progress percentage.
    """
    log_path = Path("processed") / session_id / "pipeline.log"
    
    # Check active state in-memory
    with session_jobs_lock:
        in_memory_job = session_jobs.get(session_id)
        
    overall_status = "not_started"
    if in_memory_job:
        overall_status = in_memory_job["status"]
        
    stage_timings: Dict[str, float] = {}
    stage_statuses: Dict[str, str] = {s: "pending" for s in FRONTEND_STAGES}
    log_lines: List[str] = []
    
    if log_path.exists():
        try:
            with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
            log_lines = [line.strip() for line in lines]
            
            for line in lines:
                parts = line.strip().split(" | ", 2)
                if len(parts) != 3:
                    continue
                ts_str, level, msg = parts
                
                # Parse timestamps (INFO | [01_ingest] ✓ (242.3s))
                match = STAGE_REGEX.match(msg)
                if match:
                    internal_stage = match.group(1)
                    status_char = match.group(2)
                    duration_str = match.group(3)
                    
                    duration = float(duration_str) if duration_str else 0.0
                    frontend_stage = INTERNAL_TO_FRONTEND.get(internal_stage)
                    
                    if frontend_stage:
                        # Accumulate duration for grouped stages
                        stage_timings[frontend_stage] = stage_timings.get(frontend_stage, 0.0) + duration
                        
                        # Decide status
                        if status_char in ("✓", "⚠"):
                            if stage_statuses[frontend_stage] != "failed":
                                stage_statuses[frontend_stage] = "completed"
                        elif status_char in ("✗", "FAILED"):
                            stage_statuses[frontend_stage] = "failed"
                        elif "SKIP" in status_char or "skipped" in status_char:
                            stage_statuses[frontend_stage] = "skipped"
        except Exception as e:
            print(f"[API] Error reading logs for {session_id}: {e}", file=sys.stderr)

    # Reconstruct final state when server is restarted or pipeline finishes
    completed_stages = [s for s, stat in stage_statuses.items() if stat in ("completed", "skipped")]
    
    # Check if delivery was blocked due to consent gate
    is_blocked = any("DELIVERY BLOCKED" in line for line in log_lines)
    
    if overall_status in ("not_started", "completed") and log_path.exists():
        if is_blocked:
            overall_status = "failed"
        elif "09_quality_certificate" in completed_stages:
            overall_status = "completed"
        elif any("Pipeline STOPPED" in l for l in log_lines):
            overall_status = "failed"
        else:
            # Stale or interrupted
            overall_status = "failed"
            
    # Override status if consent is blocked
    if is_blocked:
        stage_statuses["09_quality_certificate"] = "failed"
        if overall_status in ("completed", "not_started", "running"):
            overall_status = "failed"

    # Resolve running/progress properties
    current_stage = None
    if overall_status == "running":
        # Find first stage that is still pending
        for s in FRONTEND_STAGES:
            if stage_statuses[s] == "pending":
                current_stage = s
                stage_statuses[s] = "running"
                break
        if not current_stage:
            current_stage = FRONTEND_STAGES[-1]

    # Calculate progress
    if overall_status == "completed":
        overall_progress = 100
    elif overall_status == "queued":
        overall_progress = 0
    else:
        completed_count = sum(1 for s in FRONTEND_STAGES if stage_statuses[s] in ("completed", "skipped"))
        overall_progress = int((completed_count / len(FRONTEND_STAGES)) * 100)

    # Format runs array conforming to PipelineRunOut
    pipeline_runs = []
    for stage in FRONTEND_STAGES:
        # Estimate started/completed times based on completion timestamp inside log
        # Or placeholder datetime
        completed_time = None
        started_time = None
        
        # Look for the last line matching this stage inside the log file to get timestamp
        stage_log_lines = []
        last_matching_ts = None
        
        for line in log_lines:
            for internal, front in INTERNAL_TO_FRONTEND.items():
                if front == stage and f"[{internal}]" in line:
                    stage_log_lines.append(line)
                    parts = line.split(" | ", 2)
                    if len(parts) >= 1:
                        last_matching_ts = parts[0]
                        
        if last_matching_ts:
            try:
                completed_time = datetime.fromisoformat(last_matching_ts.replace("Z", "+00:00"))
                if completed_time.tzinfo is None:
                    completed_time = completed_time.replace(tzinfo=timezone.utc)
                dur = stage_timings.get(stage, 0.0)
                started_time = completed_time - timedelta(seconds=dur)
            except Exception:
                pass
                
        # Fill in current running timestamps
        if stage_statuses[stage] == "running" and in_memory_job and in_memory_job.get("started_at"):
            started_time = datetime.fromtimestamp(in_memory_job["started_at"], timezone.utc)

        # Resolve correct error message
        err_msg = None
        if stage_statuses[stage] == "failed":
            if stage == "09_quality_certificate" and is_blocked:
                # Extract the DELIVERY BLOCKED reason line
                blocked_line = next((l for l in log_lines if "DELIVERY BLOCKED" in l), None)
                if not blocked_line:
                    # Look at adjacent lines
                    for idx, l in enumerate(log_lines):
                        if "DELIVERY BLOCKED" in l:
                            blocked_line = l
                            break
                err_msg = blocked_line or "DELIVERY BLOCKED — consent_status (must be 'granted')"
            elif in_memory_job:
                err_msg = in_memory_job.get("error")

        pipeline_runs.append({
            "id": f"{session_id}_{stage}",
            "session_id": session_id,
            "stage": stage,
            "status": stage_statuses[stage],
            "started_at": started_time.isoformat() if started_time else None,
            "completed_at": completed_time.isoformat() if completed_time else None,
            "duration_seconds": round(stage_timings.get(stage, 0.0), 1) if stage in stage_timings else None,
            "error_message": err_msg,
            "log_output": "\n".join(stage_log_lines) if stage_log_lines else None
        })

    return {
        "session_id": session_id,
        "status": overall_status,
        "current_stage": current_stage,
        "overall_progress": overall_progress,
        "pipeline_runs": pipeline_runs,
        "logs": "\n".join(log_lines)
    }

# ═══════════════════════════════════════════════════════════
# BACKGROUND RUNNER TASK
# ═══════════════════════════════════════════════════════════

def run_pipeline_task(session_id: str, imu_source: str, depth_mode: str, resume: bool, skip_qa: bool):
    """
    Thread-safe synchronous pipeline execution wrapper.
    Modifies configuration attributes globally for the execution duration.
    """
    update_job_state(session_id, status="running", started_at=time.time(), error=None)
    
    # Acquire locks sequentially (only 1 pipeline runs on system hardware)
    with pipeline_execution_lock:
        print(f"[API Background] Thread acquired execution lock for session {session_id}")
        
        # Override config.py globals dynamically
        original_imu_source = cfg.IMU_SOURCE_MODE
        original_depth_mode = cfg.DEPTH_MODE
        
        cfg.IMU_SOURCE_MODE = imu_source
        cfg.DEPTH_MODE = depth_mode
        
        session_path = Path("raw") / session_id
        
        try:
            # Invoke run_session from run_pipeline.py
            run_pipeline.run_session(
                session_path=session_path,
                resume=resume,
                skip_qa=skip_qa,
                upload=False
            )
            
            update_job_state(
                session_id,
                status="completed",
                completed_at=time.time(),
                current_stage=None,
                overall_progress=100
            )
            print(f"[API Background] Finished pipeline successfully for {session_id}")
        except Exception as e:
            import traceback
            err_msg = f"{type(e).__name__}: {str(e)}"
            print(f"[API Background] Pipeline execution failed for {session_id}: {err_msg}", file=sys.stderr)
            traceback.print_exc()
            
            update_job_state(
                session_id,
                status="failed",
                completed_at=time.time(),
                current_stage=None,
                error=err_msg
            )
        finally:
            # Restore original config modes
            cfg.IMU_SOURCE_MODE = original_imu_source
            cfg.DEPTH_MODE = original_depth_mode

# ═══════════════════════════════════════════════════════════
# HTTP ROUTE HANDLERS
# ═══════════════════════════════════════════════════════════

@app.post("/sessions/{session_id}/run", status_code=202)
async def run_pipeline_endpoint(
    session_id: str,
    request: Request,
    background_tasks: BackgroundTasks
):
    """
    Accepts config JSON (IMU source, depth mode) or form configurations, and either
    uploaded raw video/IMU files or a staged folder path. Invokes the pipeline asynchronously.
    """
    # 1. Parse config variables from body content type
    imu_source = "head_mounted"
    depth_mode = "stereo"
    staged_path = None
    resume = False
    skip_qa = False
    
    video_bytes: Optional[bytes] = None
    imu_bytes: Optional[bytes] = None
    
    content_type = request.headers.get("content-type", "")
    if "multipart/form-data" in content_type:
        form = await request.form()
        imu_source = str(form.get("imu_source", imu_source))
        depth_mode = str(form.get("depth_mode", depth_mode))
        staged_path = form.get("staged_path")
        if staged_path:
            staged_path = str(staged_path)
            
        resume = str(form.get("resume", "false")).lower() == "true"
        skip_qa = str(form.get("skip_qa", "false")).lower() == "true"
        
        # Read uploaded files if any
        video_upload = form.get("video")
        if video_upload and isinstance(video_upload, UploadFile):
            video_bytes = await video_upload.read()
        
        imu_upload = form.get("imu")
        if imu_upload and isinstance(imu_upload, UploadFile):
            imu_bytes = await imu_upload.read()
    else:
        # Assume application/json
        try:
            body = await request.json()
            imu_source = body.get("imu_source", imu_source)
            depth_mode = body.get("depth_mode", depth_mode)
            staged_path = body.get("staged_path", staged_path)
            resume = bool(body.get("resume", resume))
            skip_qa = bool(body.get("skip_qa", skip_qa))
        except Exception:
            pass

    # 2. Stage raw input files
    raw_dir = Path("raw") / session_id
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    raw_video_path = raw_dir / "raw.mp4"
    raw_imu_path = raw_dir / "imu.csv"
    
    # Save files if uploaded
    if video_bytes:
        with open(raw_video_path, "wb") as f:
            f.write(video_bytes)
    if imu_bytes:
        with open(raw_imu_path, "wb") as f:
            f.write(imu_bytes)
            
    # Copy files if staged_path was specified
    if staged_path:
        staged_dir = Path(staged_path)
        if staged_dir.exists() and staged_dir.is_dir():
            s_video = staged_dir / "raw.mp4"
            s_imu_csv = staged_dir / "imu.csv"
            s_imu_json = staged_dir / "imu.json"
            
            if s_video.exists():
                shutil.copy2(s_video, raw_video_path)
            if s_imu_csv.exists():
                shutil.copy2(s_imu_csv, raw_imu_path)
            elif s_imu_json.exists():
                shutil.copy2(s_imu_json, raw_dir / "imu.json")
            s_consent = staged_dir / "consent.json"
            if s_consent.exists():
                shutil.copy2(s_consent, raw_dir / "consent.json")
        else:
            # Try to resolve relative to workspace
            resolved_staged = PROJECT_ROOT / staged_path
            if resolved_staged.exists() and resolved_staged.is_dir():
                s_video = resolved_staged / "raw.mp4"
                s_imu_csv = resolved_staged / "imu.csv"
                s_imu_json = resolved_staged / "imu.json"
                
                if s_video.exists():
                    shutil.copy2(s_video, raw_video_path)
                if s_imu_csv.exists():
                    shutil.copy2(s_imu_csv, raw_imu_path)
                elif s_imu_json.exists():
                    shutil.copy2(s_imu_json, raw_dir / "imu.json")
                s_consent = resolved_staged / "consent.json"
                if s_consent.exists():
                    shutil.copy2(s_consent, raw_dir / "consent.json")
            else:
                raise HTTPException(status_code=400, detail=f"Staged path not found: {staged_path}")
                
    # 3. Check that raw inputs are present
    has_video = raw_video_path.exists()
    has_imu = raw_imu_path.exists() or (raw_dir / "imu.json").exists()
    
    if not has_video or not has_imu:
        raise HTTPException(
            status_code=400,
            detail=f"Incomplete raw data in session folder raw/{session_id}. Video exists: {has_video}, IMU exists: {has_imu}"
        )
        
    # 4. Check consent status
    consent_status = "pending"
    consent_file = raw_dir / "consent.json"
    if consent_file.exists():
        try:
            with open(consent_file, "r") as f:
                data = json.load(f)
                consent_status = data.get("status") or data.get("consent_status") or "pending"
        except Exception:
            pass
    if consent_status != "granted":
        raise HTTPException(
            status_code=400,
            detail=f"Pipeline run rejected: consent status is '{consent_status}' (must be 'granted')"
        )

    # 5. Check if currently executing
    with session_jobs_lock:
        job = session_jobs.get(session_id)
        if job and job["status"] in ("queued", "running"):
            return {
                "message": "Pipeline already active for this session",
                "session_id": session_id,
                "status": job["status"]
            }

    # 6. Initialize queued status & dispatch background thread
    update_job_state(session_id, status="queued")
    background_tasks.add_task(
        run_pipeline_task, 
        session_id, 
        imu_source, 
        depth_mode, 
        resume, 
        skip_qa
    )
    
    return {
        "message": "Pipeline run initiated",
        "session_id": session_id,
        "status": "queued"
    }


@app.get("/sessions/{session_id}/status")
def get_session_status(session_id: str):
    """
    Returns stage statuses, durations, and log dumps.
    """
    status_info = get_session_status_from_log(session_id)
    return status_info


@app.get("/sessions/{session_id}/results")
def get_session_results(session_id: str):
    """
    Retrieves and parses completed pipeline metadata reports:
    quality_certificate.json, language_grounding.json, task_label.json, and episodes.json
    """
    proc_dir = Path("processed") / session_id
    if not proc_dir.exists():
        raise HTTPException(status_code=404, detail="Session processed directory not found")
        
    cert_path = proc_dir / "quality_certificate.json"
    lang_path = proc_dir / "language_grounding.json"
    task_path = proc_dir / "task_label.json"
    episodes_path = proc_dir / "episodes.json"
    
    results = {}
    
    # Load and parse each file if available
    for name, path in [("quality_certificate", cert_path), 
                       ("language_grounding", lang_path), 
                       ("task_label", task_path), 
                       ("episodes", episodes_path),
                       ("phases", proc_dir / "phases.json")]:
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    results[name] = json.load(f)
            except Exception as e:
                results[name] = {"error": f"Failed to parse JSON: {e}"}
        else:
            results[name] = None
            
    if not any(results.values()):
        raise HTTPException(status_code=404, detail="No processing results available yet for this session")
    return results


@app.get("/sessions/{session_id}/video")
def stream_session_video(session_id: str):
    """
    Streams the redacted compressed video file (or unredacted fallback) 
    using FileResponse to support chunked seek operations.
    """
    proc_dir = Path("processed") / session_id
    redacted_path = proc_dir / "redacted_compressed.mp4"
    compressed_path = proc_dir / "compressed.mp4"
    
    if redacted_path.exists():
        return FileResponse(redacted_path, media_type="video/mp4")
    elif compressed_path.exists():
        return FileResponse(compressed_path, media_type="video/mp4")
    else:
        raise HTTPException(status_code=404, detail="Processed video file not found")


@app.get("/sessions/{session_id}/overlays")
def get_session_overlays(session_id: str):
    """
    Exposes hand landmarks, object detections, and estimated depths.
    """
    proc_dir = Path("processed") / session_id
    if not proc_dir.exists():
        raise HTTPException(status_code=404, detail="Session processed directory not found")
        
    hand_path = proc_dir / "hand_pose.json"
    tracks_path = proc_dir / "object_tracks.json"
    depth_path = proc_dir / "depth_data.json"
    
    overlays = {}
    
    for name, path in [("hand_pose", hand_path), 
                       ("object_tracks", tracks_path), 
                       ("depth_data", depth_path)]:
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    overlays[name] = json.load(f)
            except Exception as e:
                overlays[name] = {"error": f"Failed to parse JSON: {e}"}
        else:
            overlays[name] = None
    return overlays


@app.patch("/sessions/{session_id}/consent")
async def update_session_consent(session_id: str, request: Request):
    """
    Explicitly updates the consent status for a session.
    Writes the value to raw/{session_id}/consent.json and mirrors it 
    in processed/{session_id}/session_meta.json (if already processed).
    """
    content_type = request.headers.get("content-type", "")
    consent_val = None
    
    if "multipart/form-data" in content_type:
        form = await request.form()
        consent_val = form.get("status") or form.get("consent_status")
    else:
        try:
            body = await request.json()
            consent_val = body.get("status") or body.get("consent_status")
        except Exception:
            pass
            
    if not consent_val:
        raise HTTPException(status_code=400, detail="Missing 'status' or 'consent_status' in request body")
        
    consent_val = str(consent_val).lower().strip()
    if consent_val not in ("pending", "granted", "denied", "revoked"):
        raise HTTPException(status_code=400, detail=f"Invalid consent status value: {consent_val}")
        
    # Write to raw consent file
    raw_dir = Path("raw") / session_id
    raw_dir.mkdir(parents=True, exist_ok=True)
    consent_file = raw_dir / "consent.json"
    
    try:
        with open(consent_file, "w") as f:
            json.dump({"status": consent_val}, f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to write consent file: {e}")
        
    # Also update processed metadata file if it exists
    meta_path = Path("processed") / session_id / "session_meta.json"
    if meta_path.exists():
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
            meta["consent_status"] = consent_val
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)
        except Exception as e:
            print(f"Warning: failed to update metadata consent for {session_id}: {e}", file=sys.stderr)
            
    return {"session_id": session_id, "consent_status": consent_val}


@app.get("/sessions/{session_id}/consent")
def get_session_consent(session_id: str):
    """
    Returns the current consent status of a session from raw storage or metadata.
    Defaults to 'pending' if no consent files exist.
    """
    # 1. Check raw consent file
    consent_file = Path("raw") / session_id / "consent.json"
    if consent_file.exists():
        try:
            with open(consent_file, "r") as f:
                data = json.load(f)
                val = data.get("status") or data.get("consent_status")
                if val:
                    return {"session_id": session_id, "consent_status": val}
        except Exception:
            pass
            
    # 2. Check processed metadata
    meta_path = Path("processed") / session_id / "session_meta.json"
    if meta_path.exists():
        try:
            with open(meta_path, "r") as f:
                data = json.load(f)
                val = data.get("consent_status") or data.get("status")
                if val:
                    return {"session_id": session_id, "consent_status": val}
        except Exception:
            pass
            
    return {"session_id": session_id, "consent_status": "pending"}


# ═══════════════════════════════════════════════════════════
# PREMIUM LANDING & JOB DASHBOARD PAGE (/)
# ═══════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
def get_api_dashboard():
    """
    Returns a glassmorphic dashboard visualizing session logs and queues.
    """
    # Scan raw/processed session folders
    sessions = set()
    if os.path.exists("raw"):
        for name in os.listdir("raw"):
            if os.path.isdir(os.path.join("raw", name)):
                sessions.add(name)
    if os.path.exists("processed"):
        for name in os.listdir("processed"):
            if os.path.isdir(os.path.join("processed", name)):
                sessions.add(name)
                
    sessions_data = []
    for s_id in sorted(sessions):
        status_info = get_session_status_from_log(s_id)
        sessions_data.append({
            "id": s_id,
            "status": status_info["status"],
            "progress": status_info["overall_progress"],
            "current_stage": status_info["current_stage"] or "—"
        })
        
    # Build a premium Glassmorphism page
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>DatraAI Pipeline Console</title>
        <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap" rel="stylesheet">
        <style>
            :root {{
                --bg-primary: #0b0f19;
                --card-bg: rgba(255, 255, 255, 0.03);
                --card-border: rgba(255, 255, 255, 0.08);
                --glow-color: rgba(59, 130, 246, 0.5);
                --text-primary: #f3f4f6;
                --text-secondary: #9ca3af;
                --accent-blue: #3b82f6;
                --accent-green: #10b981;
                --accent-red: #ef4444;
                --accent-purple: #8b5cf6;
            }}
            * {{
                box-sizing: border-box;
                margin: 0;
                padding: 0;
            }}
            body {{
                font-family: 'Outfit', sans-serif;
                background-color: var(--bg-primary);
                color: var(--text-primary);
                min-height: 100vh;
                display: flex;
                flex-direction: column;
                overflow-x: hidden;
                background-image: 
                    radial-gradient(at 10% 10%, rgba(59, 130, 246, 0.15) 0px, transparent 50%),
                    radial-gradient(at 90% 90%, rgba(139, 92, 246, 0.15) 0px, transparent 50%);
            }}
            header {{
                padding: 2rem;
                border-bottom: 1px solid var(--card-border);
                display: flex;
                justify-content: space-between;
                align-items: center;
                backdrop-filter: blur(10px);
            }}
            header h1 {{
                font-weight: 700;
                font-size: 1.8rem;
                background: linear-gradient(to right, #3b82f6, #8b5cf6);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                display: flex;
                align-items: center;
                gap: 0.5rem;
            }}
            .container {{
                max-width: 1200px;
                margin: 2rem auto;
                padding: 0 1.5rem;
                width: 100%;
                flex-grow: 1;
            }}
            .card {{
                background: var(--card-bg);
                border: 1px solid var(--card-border);
                border-radius: 16px;
                padding: 2rem;
                margin-bottom: 2rem;
                backdrop-filter: blur(16px);
                box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
            }}
            .card-title {{
                font-size: 1.4rem;
                font-weight: 600;
                margin-bottom: 1.5rem;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }}
            .session-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
                gap: 1.5rem;
            }}
            .session-card {{
                background: rgba(255, 255, 255, 0.015);
                border: 1px solid var(--card-border);
                border-radius: 12px;
                padding: 1.25rem;
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                position: relative;
                overflow: hidden;
            }}
            .session-card:hover {{
                transform: translateY(-4px);
                border-color: rgba(59, 130, 246, 0.4);
                box-shadow: 0 4px 20px 0 rgba(59, 130, 246, 0.1);
            }}
            .session-header {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 1rem;
            }}
            .session-id {{
                font-weight: 600;
                font-size: 1.1rem;
            }}
            .badge {{
                padding: 0.25rem 0.6rem;
                border-radius: 9999px;
                font-size: 0.75rem;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.05em;
            }}
            .badge-completed {{ background: rgba(16, 185, 129, 0.15); color: var(--accent-green); }}
            .badge-failed {{ background: rgba(239, 68, 68, 0.15); color: var(--accent-red); }}
            .badge-running {{ background: rgba(59, 130, 246, 0.15); color: var(--accent-blue); animation: pulse 2s infinite; }}
            .badge-queued {{ background: rgba(139, 92, 246, 0.15); color: var(--accent-purple); }}
            .badge-not_started {{ background: rgba(255, 255, 255, 0.08); color: var(--text-secondary); }}
            
            .progress-bar-container {{
                width: 100%;
                height: 6px;
                background: rgba(255, 255, 255, 0.08);
                border-radius: 999px;
                margin-bottom: 0.75rem;
                overflow: hidden;
            }}
            .progress-bar {{
                height: 100%;
                background: linear-gradient(to right, #3b82f6, #8b5cf6);
                border-radius: 999px;
                transition: width 0.4s ease;
            }}
            .meta-row {{
                display: flex;
                justify-content: space-between;
                font-size: 0.85rem;
                color: var(--text-secondary);
            }}
            .links-row {{
                display: flex;
                gap: 0.5rem;
                margin-top: 1rem;
            }}
            .btn {{
                background: rgba(255, 255, 255, 0.05);
                color: var(--text-primary);
                border: 1px solid var(--card-border);
                padding: 0.4rem 0.8rem;
                border-radius: 6px;
                font-size: 0.8rem;
                cursor: pointer;
                transition: all 0.2s ease;
                text-decoration: none;
                display: inline-flex;
                align-items: center;
                gap: 0.25rem;
            }}
            .btn:hover {{
                background: rgba(59, 130, 246, 0.2);
                border-color: var(--accent-blue);
            }}
            .btn-primary {{
                background: var(--accent-blue);
                border-color: var(--accent-blue);
            }}
            .btn-primary:hover {{
                background: #2563eb;
            }}
            
            form.grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 1rem;
                align-items: flex-end;
            }}
            .form-group {{
                display: flex;
                flex-direction: column;
                gap: 0.5rem;
            }}
            .form-group label {{
                font-size: 0.85rem;
                color: var(--text-secondary);
                font-weight: 500;
            }}
            .form-control {{
                background: rgba(0, 0, 0, 0.2);
                border: 1px solid var(--card-border);
                padding: 0.6rem;
                border-radius: 8px;
                color: var(--text-primary);
                font-family: inherit;
                outline: none;
            }}
            .form-control:focus {{
                border-color: var(--accent-blue);
            }}
            footer {{
                padding: 2rem;
                text-align: center;
                color: var(--text-secondary);
                font-size: 0.85rem;
                border-top: 1px solid var(--card-border);
            }}
            
            @keyframes pulse {{
                0% {{ opacity: 0.6; }}
                50% {{ opacity: 1; }}
                100% {{ opacity: 0.6; }}
            }}
        </style>
    </head>
    <body>
        <header>
            <h1>⚡ DatraAI Console</h1>
            <div>
                <a href="/docs" class="btn btn-primary" target="_blank">Swagger Docs</a>
            </div>
        </header>
        
        <div class="container">
            <div class="card">
                <div class="card-title">Trigger Pipeline Run</div>
                <form id="runForm" class="grid">
                    <div class="form-group">
                        <label for="session_id">Session ID</label>
                        <input type="text" id="session_id" class="form-control" placeholder="e.g. session_001" required>
                    </div>
                    <div class="form-group">
                        <label for="imu_source">IMU Source</label>
                        <select id="imu_source" class="form-control">
                            <option value="head_mounted">Head Mounted</option>
                            <option value="wrist_mounted">Wrist Mounted</option>
                            <option value="dual">Dual</option>
                            <option value="none">None</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label for="depth_mode">Depth Mode</label>
                        <select id="depth_mode" class="form-control">
                            <option value="stereo">Stereo</option>
                            <option value="monocular_estimated">Monocular Estimated</option>
                            <option value="none">None</option>
                        </select>
                    </div>
                    <div>
                        <button type="submit" class="btn btn-primary" style="height: 38px; width: 100%; display: justify-content; align-items: center; justify-content: center; font-weight: 600;">Launch Run</button>
                    </div>
                </form>
            </div>
            
            <div class="card">
                <div class="card-title">Sessions Pipeline Registry</div>
                <div class="session-grid" id="sessionGrid">
    """
    
    # Append session items
    for s in sessions_data:
        html_content += f"""
                    <div class="session-card" id="session-{s['id']}">
                        <div class="session-header">
                            <span class="session-id">{s['id']}</span>
                            <span class="badge badge-{s['status']}">{s['status']}</span>
                        </div>
                        <div class="progress-bar-container">
                            <div class="progress-bar" style="width: {s['progress']}%"></div>
                        </div>
                        <div class="meta-row">
                            <span>Stage: {s['current_stage']}</span>
                            <span>{s['progress']}%</span>
                        </div>
                        <div class="links-row">
                            <a href="/sessions/{s['id']}/status" class="btn" target="_blank">Status JSON</a>
                            {"<a href='/sessions/" + s['id'] + "/results' class='btn' target='_blank'>Results</a>" if s['status'] == 'completed' else ''}
                            {"<a href='/sessions/" + s['id'] + "/overlays' class='btn' target='_blank'>Overlays</a>" if s['status'] == 'completed' else ''}
                            {"<a href='/sessions/" + s['id'] + "/video' class='btn' target='_blank'>Stream Video</a>" if s['status'] == 'completed' else ''}
                        </div>
                    </div>
        """
        
    html_content += """
                </div>
            </div>
        </div>
        
        <footer>
            DatraAI Pipeline Core Wrapper Service v1.0.0
        </footer>
        
        <script>
            document.getElementById('runForm').addEventListener('submit', async (e) => {
                e.preventDefault();
                const session_id = document.getElementById('session_id').value;
                const imu_source = document.getElementById('imu_source').value;
                const depth_mode = document.getElementById('depth_mode').value;
                
                try {
                    const response = await fetch(`/sessions/${session_id}/run`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ imu_source, depth_mode })
                    });
                    const res = await response.json();
                    alert(res.message || 'Error occurred');
                    window.location.reload();
                } catch (err) {
                    alert('Failed to connect to API: ' + err.message);
                }
            });
            
            // Poll running sessions automatically
            setInterval(async () => {
                const badgeRunning = document.querySelectorAll('.badge-running, .badge-queued');
                if (badgeRunning.length > 0) {
                    window.location.reload();
                }
            }, 5000);
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)
