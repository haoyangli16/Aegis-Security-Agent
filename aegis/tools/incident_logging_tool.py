"""
Incident Logging Tool for Aegis Security Agent System.

This tool handles logging of security incidents with evidence capture,
timestamps, and structured incident records for later review and analysis.
"""

import base64
import json
import os
import sys
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from google.adk.tools import ToolContext

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from aegis.config.camera_sources import get_camera_info
from aegis.config.settings import get_setting

from aegis.tools.object_detection_tool import _capture_frame_from_source


def log_security_incident(camera_id: str, incident_type: str, description: str, 
                         tool_context: ToolContext, severity: Optional[str] = None) -> Dict[str, Any]:
    """
    Log a security incident with evidence capture and structured record keeping.
    
    This tool creates a permanent record of security incidents including timestamps,
    camera snapshots, incident details, and generates unique incident IDs for tracking.
    It's essential for maintaining security audit trails and evidence collection.
    
    Args:
        camera_id: The camera where the incident was detected
        incident_type: Type of incident (e.g., 'threat_detection', 'crowd_control', 'abandoned_object')
        description: Detailed description of the incident
        tool_context: ADK tool context for session state and services
        severity: Incident severity level ('low', 'medium', 'high', 'critical'). Optional.
        
    Returns:
        Dict containing:
        - status: Success/error status
        - incident_id: Unique identifier for the incident
        - log_file_path: Path to the incident log file
        - evidence_captured: Whether visual evidence was captured
        - timestamp: When the incident was logged
    """
    
    try:
        # Set default severity if not provided
        incident_severity = severity or "medium"

        # Generate unique incident ID
        incident_id = f"INC_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
        
        # Create incident timestamp
        incident_timestamp = datetime.now().isoformat()

        # Get camera information
        # assert the camera_id looks like cam1, cam2, cam3, etc.
        if not camera_id.startswith("cam"):
            raise ValueError("Camera ID must start with 'cam'")

        camera_info = get_camera_info(camera_id)
        camera_name = camera_info.get("name", "Unknown Camera") if camera_info else "Unknown Camera"
        camera_location = camera_info.get("location", "Unknown Location") if camera_info else "Unknown Location"
        
        # Capture evidence frame if possible
        evidence_captured = False
        evidence_path = None
        evidence_base64 = None
        
        try:
            if camera_info:
                camera_url = camera_info.get("url")
                if camera_url:
                    evidence_frame = _capture_frame_from_source(camera_url)
                    if evidence_frame is not None:
                        # Save evidence frame
                        evidence_dir = get_setting("incident_logging.alert_frame_directory", "incident_evidence")
                        os.makedirs(evidence_dir, exist_ok=True)
                        
                        evidence_filename = f"evidence_{incident_id}.jpg"
                        evidence_path = os.path.join(evidence_dir, evidence_filename)
                        
                        import cv2
                        cv2.imwrite(evidence_path, evidence_frame)
                        
                        # Also encode as base64 for immediate use
                        _, buffer = cv2.imencode('.jpg', evidence_frame)
                        evidence_base64 = base64.b64encode(buffer).decode('utf-8')
                        
                        evidence_captured = True
        except Exception as e:
            print(f"Warning: Could not capture evidence frame: {e}")
        
        # Create incident record
        incident_record = {
            "incident_id": incident_id,
            "timestamp": incident_timestamp,
            "camera_id": camera_id,
            "camera_name": camera_name,
            "camera_location": camera_location,
            "incident_type": incident_type,
            "severity": incident_severity,
            "description": description,
            "evidence_captured": evidence_captured,
            "evidence_path": evidence_path,
            "logged_by": "aegis_agent",
            "session_id": tool_context.state.get("session_id", "unknown"),
            "additional_context": {
                "previous_detections": tool_context.state.get(f"detections_{camera_id}", {}),
                "previous_analysis": tool_context.state.get(f"security_analysis_{camera_id}", {}),
                "system_state": {
                    "active_cameras": len([k for k in tool_context.state.keys() if k.startswith("camera_access_")]),
                    "total_incidents_logged": tool_context.state.get("total_incidents_logged", 0) + 1
                }
            }
        }
        
        # Write to log file
        log_dir = get_setting("incident_logging.log_directory", "security_logs")
        os.makedirs(log_dir, exist_ok=True)
        
        # Create daily log file
        log_date = datetime.now().strftime("%Y-%m-%d")
        log_filename = f"security_incidents_{log_date}.jsonl"
        log_file_path = os.path.join(log_dir, log_filename)
        
        # Append incident to log file (JSONL format for easy parsing)
        with open(log_file_path, 'a', encoding='utf-8') as log_file:
            log_file.write(json.dumps(incident_record) + '\n')
        
        # Update session state
        tool_context.state["total_incidents_logged"] = tool_context.state.get("total_incidents_logged", 0) + 1
        tool_context.state[f"last_incident_{camera_id}"] = {
            "incident_id": incident_id,
            "timestamp": incident_timestamp,
            "severity": incident_severity
        }
        
        # Create response based on severity
        if incident_severity in ["high", "critical"]:
            alert_message = f"🚨 {incident_severity.upper()} INCIDENT LOGGED: {incident_id}"
        else:
            alert_message = f"📝 Incident logged: {incident_id}"
        
        return {
            "status": "success",
            "message": alert_message,
            "incident_id": incident_id,
            "timestamp": incident_timestamp,
            "camera_id": camera_id,
            "camera_name": camera_name,
            "incident_type": incident_type,
            "severity": incident_severity,
            "description": description,
            "evidence_captured": evidence_captured,
            "evidence_path": evidence_path,
            "log_file_path": log_file_path,
            "total_incidents_today": _count_todays_incidents(log_dir),
            "next_actions": _get_incident_next_actions(incident_severity, incident_type)
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to log incident: {str(e)}",
            "incident_id": None,
            "evidence_captured": False,
            "error_details": str(e)
        }


def get_recent_incidents(tool_context: ToolContext, camera_id: Optional[str] = None, 
                        limit: Optional[int] = None) -> Dict[str, Any]:
    """
    Retrieve recent security incidents, optionally filtered by camera.
    
    Args:
        tool_context: ADK tool context for session state and services  
        camera_id: Optional camera ID to filter by, or None for all cameras
        limit: Maximum number of incidents to return. Optional.
        
    Returns:
        Dict containing recent incident records
    """
    
    try:
        # Set default limit if not provided
        incident_limit = limit or 10

        log_dir = get_setting("incident_logging.log_directory", "security_logs")
        
        if not os.path.exists(log_dir):
            return {
                "status": "success",
                "message": "No incident logs found",
                "incidents": [],
                "total_incidents": 0
            }
        
        incidents = []
        
        # Read incidents from recent log files (last 7 days)
        for days_back in range(7):
            log_date = datetime.now()
            log_date = log_date.replace(day=log_date.day - days_back)
            log_filename = f"security_incidents_{log_date.strftime('%Y-%m-%d')}.jsonl"
            log_file_path = os.path.join(log_dir, log_filename)
            
            if os.path.exists(log_file_path):
                try:
                    with open(log_file_path, 'r', encoding='utf-8') as log_file:
                        for line in log_file:
                            if line.strip():
                                incident = json.loads(line.strip())
                                
                                # Filter by camera if specified
                                if camera_id is None or incident.get("camera_id") == camera_id:
                                    incidents.append(incident)
                except Exception as e:
                    print(f"Error reading log file {log_file_path}: {e}")
        
        # Sort by timestamp (most recent first) and limit
        incidents.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        incidents = incidents[:incident_limit]
        
        return {
            "status": "success",
            "message": f"Retrieved {len(incidents)} recent incidents",
            "incidents": incidents,
            "total_incidents": len(incidents),
            "camera_filter": camera_id,
            "summary": _summarize_incidents(incidents)
        }
        
    except Exception as e:
        return {
            "status": "error", 
            "message": f"Failed to retrieve incidents: {str(e)}",
            "incidents": [],
            "total_incidents": 0,
            "error_details": str(e)
        }


def _count_todays_incidents(log_dir: str) -> int:
    """Count incidents logged today."""
    
    try:
        log_date = datetime.now().strftime("%Y-%m-%d")
        log_filename = f"security_incidents_{log_date}.jsonl"
        log_file_path = os.path.join(log_dir, log_filename)
        
        if not os.path.exists(log_file_path):
            return 0
        
        with open(log_file_path, 'r', encoding='utf-8') as log_file:
            return sum(1 for line in log_file if line.strip())
    
    except Exception:
        return 0


def _get_incident_next_actions(severity: str, incident_type: str) -> list:
    """Get recommended next actions based on incident severity and type."""
    
    actions = []
    
    if severity == "critical":
        actions.extend([
            "Immediate emergency response required",
            "Notify security management", 
            "Consider evacuation procedures"
        ])
    elif severity == "high":
        actions.extend([
            "Dispatch security personnel immediately",
            "Monitor situation closely",
            "Prepare incident report"
        ])
    elif severity == "medium":
        actions.extend([
            "Investigate further",
            "Increase monitoring frequency",
            "Document findings"
        ])
    else:  # low
        actions.append("Continue monitoring")
    
    # Add incident-type specific actions
    if incident_type == "threat_detection":
        actions.append("Verify threat assessment")
    elif incident_type == "crowd_control":
        actions.append("Consider crowd management measures")
    elif incident_type == "abandoned_object":
        actions.append("Investigate object origin")
    
    return actions


def _summarize_incidents(incidents: list) -> Dict[str, Any]:
    """Create a summary of incident statistics."""
    
    if not incidents:
        return {"total": 0}
    
    severity_counts = {}
    type_counts = {}
    camera_counts = {}
    
    for incident in incidents:
        # Count by severity
        severity = incident.get("severity", "unknown")
        severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        # Count by type
        incident_type = incident.get("incident_type", "unknown")
        type_counts[incident_type] = type_counts.get(incident_type, 0) + 1
        
        # Count by camera
        camera_id = incident.get("camera_id", "unknown")
        camera_counts[camera_id] = camera_counts.get(camera_id, 0) + 1
    
    return {
        "total": len(incidents),
        "by_severity": severity_counts,
        "by_type": type_counts, 
        "by_camera": camera_counts,
        "most_recent": incidents[0].get("timestamp") if incidents else None
    } 