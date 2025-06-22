"""
Tools package for Aegis Security Co-Pilot System.
Provides comprehensive security monitoring tools for Google ADK integration.
"""

# Import all tools for easy access
from .object_detection_tool import detect_objects
from .vlm_analysis_tool import analyze_scene_with_vlm
from .security_analysis_tool import analyze_security_situation
from .camera_control_tool import get_camera_stream_url,get_camera_info,list_all_cameras,get_cameras_by_location_type,get_cameras_by_type
from .incident_logging_tool import log_security_incident,get_recent_incidents
from .update_server_data_agent import update_detection_result

# Export all tools for easy import
__all__ = [
    'detect_objects',
    'analyze_scene_with_vlm', 
    'analyze_security_situation',
    'get_camera_stream_url',
    'log_security_incident',
    'get_camera_info',
    'list_all_cameras',
    'get_cameras_by_location_type',
    'get_cameras_by_type',
    'update_detection_result',
    'get_recent_incidents'
]

# List of all tool functions for agent registration
SECURITY_TOOLS = [
    detect_objects,
    analyze_scene_with_vlm,
    analyze_security_situation, 
    get_camera_stream_url,
    log_security_incident,
    get_camera_info,
    get_recent_incidents,
    update_detection_result,
    list_all_cameras,
    get_cameras_by_location_type,
    get_cameras_by_type
] 