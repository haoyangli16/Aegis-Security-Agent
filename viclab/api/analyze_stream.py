from fastapi import APIRouter, UploadFile, Form, File, HTTPException
from fastapi.responses import StreamingResponse
import sys
import os
import shutil
import asyncio
import json
from uuid import uuid4
from typing import Optional
import logging
import time
from collections import deque
import atexit
import glob
from PIL import Image
import io

# Add the viclab package to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "viclab"))

try:
    from viclab.video.realtime_video import SmolVLMRealtimeProcessor
except ImportError as e:
    logging.error(f"Failed to import SmolVLMRealtimeProcessor: {e}")
    raise

router = APIRouter()

# Global processor instance
processor = None
# Frontend frame buffer (since webcam is on frontend, not backend)
# Now storing PIL Images directly in memory instead of file paths
MAX_BUFFER_SIZE = 20  # Reduced buffer size to limit memory usage
frontend_frame_buffer = deque(maxlen=MAX_BUFFER_SIZE)
frontend_frame_timestamps = deque(maxlen=MAX_BUFFER_SIZE + 10)

def get_processor():
    global processor
    if processor is None:
        try:
            logging.info("Initializing SmolVLMRealtimeProcessor...")
            processor = SmolVLMRealtimeProcessor(max_frames_buffer=200)
            logging.info("SmolVLMRealtimeProcessor initialized successfully")
        except Exception as e:
            logging.error(f"Failed to initialize processor: {e}")
            processor = None  # Reset to None so we can try again later
            raise HTTPException(status_code=500, detail=f"Failed to initialize video processor: {str(e)}")
    return processor

async def upload_file_to_pil_image(file: UploadFile) -> Image.Image:
    """Convert uploaded file to PIL Image without saving to disk."""
    # Validate file type
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Read file content into memory
    file_content = await file.read()
    
    # Convert to PIL Image
    pil_image = Image.open(io.BytesIO(file_content))
    
    # Convert to RGB if necessary (handles RGBA, grayscale, etc.)
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')
    
    return pil_image

@router.post("/analyze-frame")
async def analyze_frame(
    file: UploadFile = File(...),
    prompt: str = Form(...)
):
    """Analyze a single frame/image with the given prompt."""
    try:
        # Convert uploaded file to PIL Image (no disk I/O)
        pil_image = await upload_file_to_pil_image(file)
        
        # Get processor and analyze directly with PIL Image
        proc = get_processor()
        
        # Use the processor's process_image method with PIL Image directly
        result = proc.process_image(pil_image, prompt)
        
        return {
            "result": result,
            "prompt": prompt,
            "status": "success"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error in analyze_frame: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@router.post("/add-frame-to-buffer")
async def add_frame_to_buffer(
    file: UploadFile = File(...)
):
    """Add a frame to the frontend buffer for later analysis."""
    try:
        # Convert uploaded file to PIL Image (no disk I/O)
        pil_image = await upload_file_to_pil_image(file)
        
        # Add PIL Image directly to buffer (deque will automatically remove oldest if at maxlen)
        frontend_frame_buffer.append(pil_image)
        frontend_frame_timestamps.append(time.time())
        
        return {
            "status": "success",
            "buffer_size": len(frontend_frame_buffer),
            "message": "Frame added to buffer"
        }
        
    except Exception as e:
        logging.error(f"Error adding frame to buffer: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to add frame: {str(e)}")

@router.post("/analyze-stream")
async def analyze_stream(
    file: UploadFile = File(...),
    prompt: str = Form(...)
):
    """Legacy endpoint - redirects to analyze-frame for compatibility."""
    return await analyze_frame(file, prompt)

@router.get("/stream-status")
async def get_stream_status():
    """Get the current status of the video stream buffer."""
    try:
        buffer_info = {
            "buffer_size": len(frontend_frame_buffer),
            "max_buffer_size": MAX_BUFFER_SIZE,
            "is_streaming": len(frontend_frame_buffer) > 0,
            "latest_timestamp": frontend_frame_timestamps[-1] if frontend_frame_timestamps else None,
            "buffer_time_span": (
                frontend_frame_timestamps[-1] - frontend_frame_timestamps[0] 
                if len(frontend_frame_timestamps) > 1 else 0
            )
        }
        return {
            "status": "success",
            "buffer_info": buffer_info
        }
    except Exception as e:
        logging.error(f"Error getting stream status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stream status: {str(e)}")

@router.post("/start-stream")
async def start_stream(
    video_source: int = Form(0),  # Default to webcam
    fps_limit: int = Form(10)
):
    """Start real-time video stream processing (frontend-based)."""
    try:
        # Clear existing buffer
        frontend_frame_buffer.clear()
        frontend_frame_timestamps.clear()
        
        return {
            "status": "success",
            "message": f"Frontend stream buffer initialized and ready"
        }
    except Exception as e:
        logging.error(f"Error starting stream: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start stream: {str(e)}")

@router.post("/stop-stream")
async def stop_stream():
    """Stop the real-time video stream."""
    try:
        # Clear buffer (no temp files to clean up anymore)
        frontend_frame_buffer.clear()
        frontend_frame_timestamps.clear()
        
        return {
            "status": "success",
            "message": "Stream stopped and buffer cleared"
        }
    except Exception as e:
        logging.error(f"Error stopping stream: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to stop stream: {str(e)}")

@router.post("/analyze-recent-frames")
async def analyze_recent_frames(
    prompt: str = Form(...),
    n_frames: int = Form(5),
    min_frames: int = Form(1)  # Reduced minimum for testing
):
    """Analyze recent frames from the frontend buffer using video analysis."""
    try:
        if len(frontend_frame_buffer) < min_frames:
            return {
                "status": "insufficient_frames",
                "message": f"Not enough frames in buffer (have {len(frontend_frame_buffer)}, need at least {min_frames})",
                "buffer_info": {
                    "buffer_size": len(frontend_frame_buffer),
                    "max_buffer_size": MAX_BUFFER_SIZE
                }
            }
        
        # Get recent PIL Images from buffer
        recent_pil_images = list(frontend_frame_buffer)[-n_frames:]
        
        if not recent_pil_images:
            return {
                "status": "insufficient_frames",
                "message": "No frames available in buffer"
            }
        
        # Get processor and add PIL Images to its internal buffer
        proc = get_processor()
        
        # Clear processor's buffer and add our PIL Images
        proc.frame_buffer.clear()
        proc.frame_timestamps.clear()
        
        # Convert PIL Images to OpenCV format for the processor's internal buffer
        import cv2
        import numpy as np
        for i, pil_image in enumerate(recent_pil_images):
            # Convert PIL Image to OpenCV format (RGB -> BGR)
            frame_rgb = np.array(pil_image)
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            
            proc.frame_buffer.append(frame_bgr)
            proc.frame_timestamps.append(frontend_frame_timestamps[-(len(recent_pil_images)-i)])
        
        # Use the processor's analyze_recent_frames method for proper video analysis
        if len(proc.frame_buffer) > 0:
            result = proc.analyze_recent_frames(
                prompt=prompt, 
                n_frames=min(n_frames, len(proc.frame_buffer)),
                min_frames=1
            )
            
            if result:
                return {
                    "status": "success",
                    "result": result,
                    "prompt": prompt,
                    "frames_analyzed": len(proc.frame_buffer),
                    "buffer_info": {
                        "buffer_size": len(frontend_frame_buffer),
                        "max_buffer_size": MAX_BUFFER_SIZE
                    }
                }
            else:
                return {
                    "status": "analysis_failed",
                    "message": "Video analysis returned no result"
                }
        else:
            return {
                "status": "insufficient_frames",
                "message": "No valid frames could be loaded for analysis"
            }
        
    except Exception as e:
        logging.error(f"Error analyzing recent frames: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@router.post("/clear-buffer")
async def clear_buffer():
    """Clear the frame buffer."""
    try:
        # Clear buffer (no temp files to clean up anymore)
        frontend_frame_buffer.clear()
        frontend_frame_timestamps.clear()
        
        return {
            "status": "success",
            "message": "Buffer cleared"
        }
        
    except Exception as e:
        logging.error(f"Error clearing buffer: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear buffer: {str(e)}")

@router.post("/cleanup-temp")
async def cleanup_temp():
    """Legacy endpoint - no longer needed since we don't use temp files."""
    try:
        return {
            "status": "success",
            "message": "No temp files to clean up (using in-memory processing)"
        }
        
    except Exception as e:
        logging.error(f"Error in manual cleanup: {e}")
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")
