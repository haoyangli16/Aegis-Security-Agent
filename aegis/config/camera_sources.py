"""
Camera source configuration for Aegis Security Agent System.

This module manages the mapping between logical camera IDs and actual video sources.
For the hackathon demo, we use YouTube live streams that simulate security cameras.
"""

from typing import Dict, Optional

# Demo camera sources using YouTube live streams
# These URLs represent different types of surveillance scenarios

CAMERA_SOURCES = {
    "cam1": {
        "name": "Main Entrance", 
        "location": "Building Lobby Kensington, Philadelphia PA, USA",
        "url": "https://www.youtube.com/watch?v=9WmysKoxh8w",  # City walking stream
        "type": "entrance_monitor",
        "description": "Primary entrance monitoring for pedestrian traffic"
    },
    "cam2": {
        "name": "Gate Security",
        "location": "Security Checkpoint Ponte delle Guglie and Strada Nova, Bologna, Italy", 
        "url": "https://www.youtube.com/watch?v=yGQepsHPcN4",  # Stadium/crowd stream
        "type": "security_gate",
        "description": "Security gate monitoring for crowd control"
    },
    "cam3": {
        "name": "Parking Area",
        "location": "Vehicle Entry Nanai Road, Patong, Phuket, Thailand",
        "url": "https://www.youtube.com/watch?v=R3H5NtS_OeQ",  # Traffic/vehicle stream
        "type": "vehicle_monitor", 
        "description": "Parking area surveillance for vehicle monitoring"
    },
    "cam4": {
        "name": "Public Plaza",
        "location": "Central Plaza Street Davao City, Philippines",
        "url": "https://www.youtube.com/watch?v=i3w7qZVSAsY",  # Public space stream
        "type": "public_space",
        "description": "Philippines Live Street BBQ Agdao"
        
    },
}

def get_camera_url(camera_id: str) -> Optional[str]:
    """
    Get the video stream URL for a given camera ID.
    
    Args:
        camera_id: The logical camera identifier
        
    Returns:
        str: The video stream URL, or None if camera ID not found
    """
    camera_info = CAMERA_SOURCES.get(camera_id)
    return camera_info.get("url") if camera_info else None

def get_camera_info(camera_id: str) -> Optional[Dict]:
    """
    Get complete camera information for a given camera ID.
    
    Args:
        camera_id: The logical camera identifier
        
    Returns:
        dict: Complete camera information, or None if camera ID not found
    """
    return CAMERA_SOURCES.get(camera_id)

def list_available_cameras() -> Dict[str, str]:
    """
    Get a list of all available cameras with their names.
    
    Returns:
        dict: Mapping of camera_id to camera name
    """
    return {
        camera_id: info["name"] 
        for camera_id, info in CAMERA_SOURCES.items()
    }

def get_cameras_by_type(camera_type: str) -> Dict[str, Dict]:
    """
    Get all cameras of a specific type.
    
    Args:
        camera_type: The type of camera to filter by
        
    Returns:
        dict: Cameras matching the specified type
    """
    return {
        camera_id: info 
        for camera_id, info in CAMERA_SOURCES.items()
        if info.get("type") == camera_type
    } 