"""
Security settings and configuration for Aegis Security Agent System.
"""

import os
from typing import Dict, Any

# Security analysis settings
SECURITY_SETTINGS = {
    "threat_detection": {
        "high_threat_objects": [
            "knife", "gun", "weapon", "fighting", "violence", 
            "punching", "kicking", "blood", "screaming"
        ],
        "suspicious_behaviors": [
            "sitting on the ground", "squatting", "crouching",
            "loitering", "abandoned object"
        ],
        "crowd_thresholds": {
            "low_density": 5,
            "medium_density": 15, 
            "high_density": 30
        },
        "vehicle_thresholds": {
            "traffic_alert": 5,
            "congestion_alert": 10
        }
    },
    
    "analysis_parameters": {
        "frame_buffer_size": 90,  # frames to keep in buffer
        "analysis_interval": 2.0,  # seconds between VLM analysis
        "min_frames_for_vlm": 8,   # minimum frames needed for VLM analysis
        "yolo_confidence_threshold": 0.5,
        "enable_vlm_analysis": True,
        "save_alert_frames": True
    },
    
    "incident_logging": {
        "log_directory": "security_logs",
        "alert_frame_directory": "alert_frames",
        "max_log_files": 100,
        "log_retention_days": 30
    },
    
    "audio_settings": {
        "whisper_model": "small",
        "sample_rate": 16000,
        "chunk_duration": 5.0,  # seconds
        "language": "en"
    }
}

# ADK Agent settings
ADK_SETTINGS = {
    "agent": {
        "name": "aegis_agent",
        "model": "gemini-2.0-flash",
        "description": "AI Security Co-Pilot for real-time surveillance monitoring",
        "session_timeout": 3600,  # 1 hour
        "max_tool_calls_per_session": 100
    },
    
    "api": {
        "host": "0.0.0.0",
        "port": 8000,
        "debug": False,
        "cors_enabled": True
    }
}

# VLM Debug Configuration
VLM_DEBUG_LOGGING = False  # Set to True to enable VLM debug prints

# Environment-based overrides
def get_setting(key_path: str, default: Any = None) -> Any:
    """
    Get a setting value, with environment variable override support.
    
    Args:
        key_path: Dot-separated path to setting (e.g., 'threat_detection.crowd_thresholds.low_density')
        default: Default value if setting not found
        
    Returns:
        The setting value or default
    """
    # Check for environment variable override
    env_key = f"AEGIS_{key_path.upper().replace('.', '_')}"
    env_value = os.getenv(env_key)
    
    if env_value is not None:
        # Try to convert to appropriate type
        try:
            # Handle boolean values
            if env_value.lower() in ('true', 'false'):
                return env_value.lower() == 'true'
            # Handle numeric values
            if env_value.isdigit():
                return int(env_value)
            if '.' in env_value and env_value.replace('.', '').isdigit():
                return float(env_value)
            return env_value
        except ValueError:
            return env_value
    
    # Navigate through nested settings
    keys = key_path.split('.')
    current = SECURITY_SETTINGS
    
    try:
        for key in keys:
            current = current[key]
        return current
    except (KeyError, TypeError):
        # Try ADK settings
        current = ADK_SETTINGS
        try:
            for key in keys:
                current = current[key]
            return current
        except (KeyError, TypeError):
            return default

# Convenience functions
def get_threat_keywords() -> list:
    """Get list of all threat-related keywords"""
    return (SECURITY_SETTINGS["threat_detection"]["high_threat_objects"] + 
            SECURITY_SETTINGS["threat_detection"]["suspicious_behaviors"])

def get_crowd_threshold(level: str) -> int:
    """Get crowd threshold for specific level"""
    return SECURITY_SETTINGS["threat_detection"]["crowd_thresholds"].get(level, 5)

def is_vlm_enabled() -> bool:
    """Check if VLM analysis is enabled"""
    return get_setting("analysis_parameters.enable_vlm_analysis", True)