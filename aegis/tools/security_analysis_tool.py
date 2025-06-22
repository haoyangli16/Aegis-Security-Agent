"""
Security Analysis Tool for Aegis Security Agent System.

This tool provides comprehensive security situation analysis by combining
YOLO object detection, VLM scene understanding, and security decision logic.
"""

import time
from typing import Dict, Any
from google.adk.tools import ToolContext
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))
from aegis.core.analyze_frame import analyze_frame, clean_result_for_json
from aegis.config.camera_sources import get_camera_url
from aegis.tools.object_detection_tool import _capture_frame_from_source


def analyze_security_situation(video_feed_id: str, tool_context: ToolContext) -> Dict[str, Any]:
    """
    Perform comprehensive security analysis of a video feed.
    
    This tool combines YOLO object detection, VLM scene analysis, and security
    decision logic to provide a complete security assessment. It identifies
    threats, assesses crowd situations, and provides actionable recommendations.
    
    Args:
        video_feed_id: The camera ID or video source identifier (e.g., 'cam1', 'gate_2')
        tool_context: ADK tool context for session state and services
        
    Returns:
        Dict containing:
        - status: Success/error status
        - threat_level: Security threat level (low/medium/high)
        - security_label: Overall security assessment (safe/abnormal)
        - recommendations: Specific action recommendations
        - detected_objects: Objects found in the scene
        - scene_analysis: VLM description of the scene
        - incident_detected: Whether an incident was detected
    """
    
    try:
        from aegis.tools.incident_logging_tool import log_security_incident

        # Get camera URL from video_feed_id
        camera_url = get_camera_url(video_feed_id)
        if not camera_url:
            return {
                "status": "error",
                "message": f"Camera '{video_feed_id}' not found in system",
                "threat_level": "unknown",
                "security_label": "error",
                "recommendations": ["Check camera configuration"],
                "incident_detected": False
            }
        
        # Capture frame for analysis
        frame = _capture_frame_from_source(video_feed_id)
        if frame is None:
            return {
                "status": "error",
                "message": f"Failed to capture frame from camera '{video_feed_id}'",
                "threat_level": "unknown",
                "security_label": "error", 
                "recommendations": ["Check camera connection"],
                "incident_detected": False
            }
        
        # Run comprehensive security analysis
        analysis_result = analyze_frame(frame, camera_id=video_feed_id, use_vlm=True)
        
        # Process the analysis results
        threat_level = analysis_result.get("threat_level", "low")
        security_label = analysis_result.get("label", "safe")
        command = analysis_result.get("command", "No action needed")
        description = analysis_result.get("description", "Normal activity")
        matched_threat = analysis_result.get("matched", None)
        
        yolo_analysis = analysis_result.get("yolo_analysis", {})
        vlm_analysis = analysis_result.get("vlm_analysis", {})
        vlm_description = analysis_result.get("raw_vlm_description", "")
        
        # Generate specific recommendations based on analysis
        recommendations = _generate_security_recommendations(
            threat_level, security_label, command, yolo_analysis, vlm_analysis, matched_threat
        )
        
        # Determine if this constitutes a security incident
        incident_detected = (
            security_label == "abnormal" or 
            threat_level in ["medium", "high"] or
            len(yolo_analysis.get("abnormal_objects", [])) > 0
        )
        
        # Store comprehensive analysis in session state
        security_key = f"security_analysis_{video_feed_id}"
        tool_context.state[security_key] = {
            "timestamp": time.time(),
            "threat_level": threat_level,
            "security_label": security_label,
            "incident_detected": incident_detected,
            "analysis_details": clean_result_for_json(analysis_result)
        }
        
        # Create structured response
        response = {
            "status": "success",
            "message": f"Security analysis completed for {video_feed_id}",
            "camera_id": video_feed_id,
            "timestamp": time.time(),
            
            # Core Security Assessment
            "threat_level": threat_level,
            "security_label": security_label,
            "incident_detected": incident_detected,
            "threat_description": matched_threat or "No specific threats",
            
            # Detailed Analysis
            "scene_description": description,
            "vlm_description": vlm_description,
            "detected_objects": yolo_analysis.get("detected_objects", []),
            "object_counts": yolo_analysis.get("bounding_boxes", []),
            "person_count": yolo_analysis.get("person_count", 0),
            "vehicle_count": yolo_analysis.get("car_count", 0),
            
            # Security Intelligence
            "abnormal_objects": yolo_analysis.get("abnormal_objects", []),
            "suspicious_behavior": vlm_analysis.get("has_suspicious_behavior", False),
            "abnormal_behavior": vlm_analysis.get("has_abnormal_behavior", False),
            "matched_keywords": vlm_analysis.get("matched_keywords", []),
            
            # Actionable Recommendations
            "recommendations": recommendations,
            "priority_action": _get_priority_action(threat_level, security_label),
            
            # Technical Details
            "buffer_frames": analysis_result.get("buffer_size", 0),
            "analysis_confidence": _calculate_overall_confidence(yolo_analysis, vlm_analysis)
        }
        
        if incident_detected:
            log_result = log_security_incident(
                camera_id=video_feed_id,
                incident_type="threat_detection",
                description=description,
                tool_context=tool_context,
                severity=threat_level
            )
            response["incident_log"] = log_result
            response["alert_message"] = log_result.get("message", "Incident logged.")

        return response

    except Exception as e:
        return {
            "status": "error",
            "message": f"Security analysis failed: {str(e)}",
            "threat_level": "unknown",
            "security_label": "error",
            "recommendations": ["System error - check logs"],
            "incident_detected": False,
            "error_details": str(e)
        }


def _generate_security_recommendations(threat_level: str, security_label: str, command: str, 
                                     yolo_analysis: Dict, vlm_analysis: Dict, matched_threat: str) -> list:
    """Generate specific security recommendations based on analysis results."""
    
    recommendations = []
    
    # High-priority recommendations for abnormal situations
    if security_label == "abnormal":
        if matched_threat:
            recommendations.append(f"ALERT: {matched_threat} detected - Immediate response required")
        
        if yolo_analysis.get("abnormal_objects"):
            recommendations.append("Dangerous objects detected - Dispatch security immediately")
        
        if vlm_analysis.get("has_abnormal_behavior"):
            recommendations.append("Abnormal behavior pattern detected - Investigate immediately")
    
    # Crowd management recommendations
    person_count = yolo_analysis.get("person_count", 0)
    if person_count >= 10:
        recommendations.append("High crowd density - Consider crowd control measures")
        if vlm_analysis.get("has_suspicious_behavior"):
            recommendations.append("Suspicious activity in crowd - Increase security presence")
    elif person_count >= 5:
        recommendations.append("Monitor crowd movement and behavior")
    
    # Traffic management recommendations  
    vehicle_count = yolo_analysis.get("car_count", 0)
    if vehicle_count >= 5:
        recommendations.append("High vehicle traffic - Deploy traffic management")
    elif vehicle_count >= 3:
        recommendations.append("Monitor vehicle flow")
    
    # General security recommendations
    if threat_level == "high":
        recommendations.append("High threat level - Maintain heightened security posture")
    elif threat_level == "medium":
        recommendations.append("Elevated threat - Increase monitoring frequency")
    
    # Default recommendation for normal situations
    if not recommendations:
        recommendations.append("Continue normal monitoring")
    
    return recommendations


def _get_priority_action(threat_level: str, security_label: str) -> str:
    """Determine the priority action based on threat assessment."""
    
    if security_label == "abnormal":
        if threat_level == "high":
            return "IMMEDIATE_RESPONSE_REQUIRED"
        elif threat_level == "medium":
            return "DISPATCH_SECURITY"
        else:
            return "INVESTIGATE"
    else:
        return "CONTINUE_MONITORING"


def _calculate_overall_confidence(yolo_analysis: Dict, vlm_analysis: Dict) -> float:
    """Calculate overall confidence score for the security analysis."""
    
    confidence = 0.7  # Base confidence
    
    # Increase confidence based on detection count
    detections = len(yolo_analysis.get("bounding_boxes", []))
    if detections > 0:
        confidence += min(detections * 0.05, 0.2)
    
    # Consider VLM analysis quality
    if vlm_analysis.get("matched_keywords"):
        confidence += 0.1
    
    return min(confidence, 1.0) 