"""
VLM Analysis Tool for Aegis Security Agent System.

This tool provides both SmolVLM and Seed-VL-1.5 pro capabilities for video scene analysis.
- SmolVLM: Fast analysis with local models
- Seed-VL-1.5 pro: Accurate analysis with cloud-based models (slower but more detailed)
"""

import os
import sys
from typing import Any, Dict

import cv2
import numpy as np
from google.adk.tools import ToolContext

from aegis.core.security_context import get_security_context

# Try to import SmolVLM
try:
    from viclab.video.realtime_video import SmolVLMRealtimeProcessor
    SMOLVLM_AVAILABLE = True
    print("✅ SmolVLM analysis available")
except ImportError as e:
    SMOLVLM_AVAILABLE = False
    print(f"⚠️ SmolVLM analysis not available: {e}")

# Try to import Seed-VL-1.5 pro (ComplexVideoUnderstander)
try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../VicLab")))
    from viclab.video.doubao_video import ComplexVideoUnderstander
    SEEDVL_AVAILABLE = True
    print("✅ Seed-VL-1.5 pro analysis available")
except ImportError as e:
    SEEDVL_AVAILABLE = False
    print(f"⚠️ Seed-VL-1.5 pro analysis not available: {e}")

VLM_AVAILABLE = SMOLVLM_AVAILABLE or SEEDVL_AVAILABLE

# Global VLM processor instances (initialized once for performance)
smolvlm_processor = None
seedvl_processor = None


def init_smolvlm_processor():
    """Initialize SmolVLM processor if not already done"""
    global smolvlm_processor
    if SMOLVLM_AVAILABLE and smolvlm_processor is None:
        try:
            smolvlm_processor = SmolVLMRealtimeProcessor(max_frames_buffer=100)
            print("✅ SmolVLM processor initialized for tools")
            return True
        except Exception as e:
            print(f"❌ Failed to initialize SmolVLM processor: {e}")
            return False
    return smolvlm_processor is not None


def init_seedvl_processor():
    """Initialize Seed-VL-1.5 pro processor if not already done"""
    global seedvl_processor
    if SEEDVL_AVAILABLE and seedvl_processor is None:
        try:
            seedvl_processor = ComplexVideoUnderstander(max_frames_buffer=100)
            print("✅ Seed-VL-1.5 pro processor initialized for tools")
            return True
        except Exception as e:
            print(f"❌ Failed to initialize Seed-VL-1.5 pro processor: {e}")
            return False
    return seedvl_processor is not None


def analyze_scene_with_vlm(
    video_feed_id: str, 
    question: str, 
    tool_context: ToolContext,
    vlm_method: str = "smolvlm"
) -> Dict[str, Any]:
    """
    Analyze a video scene using a Vision Language Model to answer natural language questions.

    Args:
        video_feed_id: The camera ID or video source identifier
        question: Natural language question about the scene
        tool_context: ADK tool context for session state and services
        vlm_method: VLM method to use ("smolvlm" for speed, "seedvl" for accuracy)

    Returns:
        Dict containing analysis results
    """

    try:
        # Validate VLM method
        if vlm_method not in ["smolvlm", "seedvl"]:
            return {
                "status": "error",
                "message": f"Invalid VLM method '{vlm_method}'. Use 'smolvlm' or 'seedvl'",
                "analysis": "Invalid VLM method specified",
                "confidence": 0.0,
            }

        # Check if requested VLM method is available
        if vlm_method == "smolvlm" and not SMOLVLM_AVAILABLE:
            return {
                "status": "error",
                "message": "SmolVLM requested but not available. Please install dependencies or use 'seedvl'",
                "analysis": "SmolVLM analysis unavailable",
                "confidence": 0.0,
            }
        elif vlm_method == "seedvl" and not SEEDVL_AVAILABLE:
            return {
                "status": "error",
                "message": "Seed-VL-1.5 pro requested but not available. Please install dependencies or use 'smolvlm'",
                "analysis": "Seed-VL-1.5 pro analysis unavailable",
                "confidence": 0.0,
            }

        # Get current frame from monitor server
        try:
            context = get_security_context()
            monitor_server = context.monitor_server
        except RuntimeError:
            return {
                "status": "error",
                "message": "SecurityContext not initialized",
                "analysis": "Cannot access camera feed",
                "confidence": 0.0,
            }

        # Convert camera_id to proper format
        try:
            camera_id = (
                int(video_feed_id)
                if video_feed_id.isdigit()
                else int(video_feed_id.replace("cam", "").replace("camera", ""))
            )
            source_index = monitor_server.selected_sources[camera_id - 1]
            current_frame = monitor_server.webcam_loader.get_frame(source_index)
        except (ValueError, IndexError, AttributeError):
            return {
                "status": "error",
                "message": f"Invalid camera ID: {video_feed_id}",
                "analysis": "Cannot access specified camera",
                "confidence": 0.0,
            }

        if current_frame is None:
            return {
                "status": "error",
                "message": f"No frame available from camera {video_feed_id}",
                "analysis": "Camera feed not available",
                "confidence": 0.0,
            }

        # Run analysis based on VLM method
        if vlm_method == "smolvlm":
            analysis, confidence, method_info = _analyze_with_smolvlm(
                monitor_server, source_index, question
            )
        else:  # seedvl
            analysis, confidence, method_info = _analyze_with_seedvl(
                monitor_server, source_index, question
            )

        if not analysis:
            analysis = "VLM analysis completed but no specific observations to report."

        # Store analysis in session state
        analysis_key = f"vlm_analysis_{video_feed_id}"
        tool_context.state[analysis_key] = {
            "question": question,
            "analysis": analysis,
            "vlm_method": vlm_method,
            "confidence": confidence,
            "method_info": method_info,
        }

        return {
            "status": "success",
            "message": f"VLM analysis completed using {method_info} for {video_feed_id}",
            "vlm_method": method_info,
            "camera_id": video_feed_id,
            "question_asked": question,
            "analysis": analysis,
            "scene_description": analysis,
            "confidence": confidence,
        }

    except Exception as e:
        print(f"VLM analysis error: {e}")
        return {
            "status": "error",
            "message": f"VLM analysis failed: {str(e)}",
            "vlm_method": vlm_method,
            "analysis": f"Analysis failed due to technical error: {str(e)}",
            "confidence": 0.0,
        }


def _analyze_with_smolvlm(monitor_server, source_index, question) -> tuple:
    """
    Analyze video scene using SmolVLM (fast local analysis).
    
    Args:
        monitor_server: Monitor server instance
        source_index: Camera source index
        question: Question to ask about the scene
        
    Returns:
        tuple: (analysis text, confidence, method_info)
    """
    # Initialize SmolVLM processor if not done
    if not init_smolvlm_processor():
        return "SmolVLM processor not available", 0.0, "SmolVLM (unavailable)"

    # Get recent frames for temporal analysis
    recent_frames = []
    for i in range(10):  # Get last 10 frames
        frame = monitor_server.webcam_loader.get_frame(source_index)
        if frame is not None:
            # Convert BGR to RGB for VLM
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            recent_frames.append(rgb_frame)

    if len(recent_frames) < 3:
        return "Insufficient frames for SmolVLM analysis", 0.0, "SmolVLM (insufficient frames)"

    # Clear SmolVLM processor buffer and add our frames
    smolvlm_processor.frame_buffer.clear()
    smolvlm_processor.frame_timestamps.clear()

    # Add frames to VLM processor
    import time

    current_time = time.time()
    for i, frame in enumerate(recent_frames):
        smolvlm_processor.frame_buffer.append(frame)
        smolvlm_processor.frame_timestamps.append(
            current_time - (len(recent_frames) - i) * 0.1
        )

    # Analyze with SmolVLM using the provided question
    prompt = f"{question}. Provide specific observations about what you see in this security camera footage."
    analysis = smolvlm_processor.analyze_recent_frames(
        prompt=prompt, n_frames=min(8, len(recent_frames)), min_frames=3
    )

    return analysis or "No specific observations", 0.9, "SmolVLM (fast)"


def _analyze_with_seedvl(monitor_server, source_index, question) -> tuple:
    """
    Analyze video scene using Seed-VL-1.5 pro (accurate cloud-based analysis).
    
    Args:
        monitor_server: Monitor server instance  
        source_index: Camera source index
        question: Question to ask about the scene
        
    Returns:
        tuple: (analysis text, confidence, method_info)
    """
    # Initialize Seed-VL-1.5 pro processor if not done
    if not init_seedvl_processor():
        return "Seed-VL-1.5 pro processor not available", 0.0, "Seed-VL-1.5 pro (unavailable)"

    # Get recent frames for temporal analysis
    recent_frames = []
    for i in range(15):  # Get more frames for better Seed-VL analysis
        frame = monitor_server.webcam_loader.get_frame(source_index)
        if frame is not None:
            recent_frames.append(frame)

    if len(recent_frames) < 8:
        return "Insufficient frames for Seed-VL-1.5 pro analysis", 0.0, "Seed-VL-1.5 pro (insufficient frames)"

    # Clear Seed-VL processor buffer and add our frames
    seedvl_processor.frame_buffer.clear()
    seedvl_processor.frame_timestamps.clear()

    # Add frames to Seed-VL processor
    import time

    current_time = time.time()
    for i, frame in enumerate(recent_frames):
        # Convert BGR to RGB for Seed-VL
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        seedvl_processor.frame_buffer.append(rgb_frame)
        seedvl_processor.frame_timestamps.append(
            current_time - (len(recent_frames) - i) * 0.1
        )

    # Analyze with Seed-VL-1.5 pro using the provided question
    prompt = f"{question}. Provide detailed and specific observations about what you see in this security camera footage. Focus on behaviors, objects, and any potential security concerns."
    
    try:
        analysis = seedvl_processor.analyze_recent_frames(
            prompt=prompt, 
            n_frames=min(12, len(recent_frames)), 
            min_frames=8
        )
        return analysis or "No specific observations", 0.95, "Seed-VL-1.5 pro (accurate)"
        
    except Exception as e:
        print(f"Seed-VL-1.5 pro analysis failed: {e}")
        return f"Seed-VL-1.5 pro analysis failed: {str(e)}", 0.0, "Seed-VL-1.5 pro (error)"
