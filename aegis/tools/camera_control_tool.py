"""
Camera Control Tool for Aegis Security Agent System.

This tool manages camera sources and provides URL mapping for video feeds.
It acts as a directory service for logical camera IDs to physical sources.
"""

import os
import sys
from typing import Any, Dict, Optional

from google.adk.tools import ToolContext

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from aegis.config.camera_sources import (
    CAMERA_SOURCES,
    get_camera_info,
    get_camera_url,
    get_cameras_by_type,
    list_available_cameras,
)


def get_camera_info(camera_id: str) -> Dict[str, Any]:
    """
    Get the camera information for a specific camera.
    """
    return CAMERA_SOURCES[camera_id]

def get_camera_stream_url(camera_id: str, tool_context: ToolContext) -> Dict[str, Any]:
    """
    Get the video stream URL and information for a specific camera.
    
    This tool provides a clean interface for the agent to obtain camera stream
    URLs and metadata. It maps logical camera names (like 'cam1', 'cam2', 'cam3', 'cam4') to
    actual video source URLs and provides detailed camera information.
    
    Args:
        camera_id: The logical camera identifier (e.g., 'cam1', 'cam2', 'cam3', 'cam4')
        tool_context: ADK tool context for session state and services
        
    Returns:
        Dict containing:
        - status: Success/error status  
        - camera_url: The video stream URL
        - camera_info: Detailed camera metadata
        - available_cameras: List of available cameras if requested camera not found
    """
    
    try:
        # Get camera URL
        camera_url = get_camera_url(camera_id)
        
        if not camera_url:
            # Camera not found, provide helpful alternatives
            available_cameras = list_available_cameras()
            
            # Try to find similar camera IDs
            suggestions = _find_similar_cameras(camera_id, available_cameras)
            
            return {
                "status": "error",
                "message": f"Camera '{camera_id}' not found in system",
                "camera_url": None,
                "camera_info": None,
                "available_cameras": available_cameras,
                "suggestions": suggestions,
                "total_cameras": len(available_cameras)
            }
        
        # Get detailed camera information
        camera_info = get_camera_info(camera_id)
        
        # Store camera access in session state for tracking
        access_key = f"camera_access_{camera_id}"
        tool_context.state[access_key] = {
            "last_accessed": tool_context.state.get("current_time", "unknown"),
            "access_count": tool_context.state.get(access_key, {}).get("access_count", 0) + 1
        }
        
        return {
            "status": "success",
            "message": f"Camera '{camera_id}' accessed successfully",
            "camera_id": camera_id,
            "camera_url": camera_url,
            "camera_info": {
                "name": camera_info.get("name", "Unknown"),
                "location": camera_info.get("location", "Unknown"),
                "type": camera_info.get("type", "unknown"),
                "description": camera_info.get("description", "No description")
            },
            "stream_available": True,
            "access_count": tool_context.state.get(access_key, {}).get("access_count", 1)
        }
        
    except Exception as e:
        return {
            "status": "error", 
            "message": f"Camera control error: {str(e)}",
            "camera_url": None,
            "camera_info": None,
            "error_details": str(e)
        }


def list_all_cameras(tool_context: ToolContext) -> Dict[str, Any]:
    """
    List all available cameras in the system.
    
    This tool provides a complete inventory of available cameras with their
    metadata. Useful for operators to understand available surveillance coverage.
    
    Args:
        tool_context: ADK tool context for session state and services
        
    Returns:
        Dict containing all camera information
    """
    
    try:
        available_cameras = list_available_cameras()
        
        # Get detailed info for each camera
        camera_details = {}
        cameras_by_type = {}
        
        for camera_id in available_cameras.keys():
            info = get_camera_info(camera_id)
            camera_details[camera_id] = info
            
            # Group by type
            camera_type = info.get("type", "unknown")
            if camera_type not in cameras_by_type:
                cameras_by_type[camera_type] = []
            cameras_by_type[camera_type].append(camera_id)
        
        return {
            "status": "success",
            "message": f"Found {len(available_cameras)} cameras in system",
            "total_cameras": len(available_cameras),
            "cameras": camera_details,
            "cameras_by_type": cameras_by_type,
            "available_types": list(cameras_by_type.keys())
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to list cameras: {str(e)}",
            "total_cameras": 0,
            "cameras": {},
            "error_details": str(e)
        }


def get_cameras_by_location_type(location_type: str, tool_context: ToolContext) -> Dict[str, Any]:
    """
    Get cameras filtered by location type (e.g., 'entrance_monitor', 'security_gate').
    
    Args:
        location_type: Type of camera location to filter by
        tool_context: ADK tool context for session state and services
        
    Returns:
        Dict containing filtered camera information
    """
    
    try:
        filtered_cameras = get_cameras_by_type(location_type)
        
        if not filtered_cameras:
            # Get all available types for helpful response
            all_cameras = {k: v for k, v in CAMERA_SOURCES.items()}
            available_types = list(set(cam.get("type", "unknown") for cam in all_cameras.values()))
            
            return {
                "status": "error",
                "message": f"No cameras found for type '{location_type}'",
                "cameras": {},
                "available_types": available_types,
                "total_cameras": 0
            }
        
        return {
            "status": "success", 
            "message": f"Found {len(filtered_cameras)} cameras of type '{location_type}'",
            "location_type": location_type,
            "cameras": filtered_cameras,
            "total_cameras": len(filtered_cameras)
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Camera filtering failed: {str(e)}",
            "cameras": {},
            "total_cameras": 0,
            "error_details": str(e)
        }


def _find_similar_cameras(camera_id: str, available_cameras: Dict[str, str]) -> list:
    """
    Find cameras with similar names to the requested camera_id.
    
    Args:
        camera_id: The requested camera ID
        available_cameras: Dict of available camera IDs to names
        
    Returns:
        List of similar camera IDs
    """
    
    suggestions = []
    camera_id_lower = camera_id.lower()
    
    for cam_id, cam_name in available_cameras.items():
        # Check for partial matches in ID or name
        if (camera_id_lower in cam_id.lower() or 
            cam_id.lower() in camera_id_lower or
            camera_id_lower in cam_name.lower()):
            suggestions.append(cam_id)
    
    # Limit suggestions to top 3
    return suggestions[:3] 