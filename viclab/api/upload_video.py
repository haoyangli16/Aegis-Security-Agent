from fastapi import APIRouter, File, UploadFile, HTTPException, Form
import shutil
import os
import sys
from uuid import uuid4
import logging
from typing import Optional

# Add the viclab package to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "viclab"))

try:
    from viclab.video.realtime_video import SmolVLMRealtimeProcessor
except ImportError as e:
    logging.error(f"Failed to import SmolVLMRealtimeProcessor: {e}")

router = APIRouter()

VIDEO_SAVE_PATH = "./videos"

@router.post("/upload-video")
async def upload_video(
    file: UploadFile = File(...),
    prompt: Optional[str] = Form(None)
):
    """Upload a video file and optionally analyze it."""

    # Validate file type
    if not file.content_type or not file.content_type.startswith('video/'):
        raise HTTPException(status_code=400, detail="File must be a video")        
    # Generate unique filename
    file_extension = os.path.splitext(file.filename or "video.mp4")[1]
    filename = f"video_{uuid4().hex[:8]}{file_extension}"

    file_path = os.path.join(VIDEO_SAVE_PATH, filename)

        # Save uploaded file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

        response = {
            "status": "success",
            "message": "Video uploaded successfully", 
            "path": file_path,
            "filename": filename
        }
        
        # If prompt provided, analyze the video
        if prompt:
            try:
                processor = SmolVLMRealtimeProcessor()
                analysis_result = processor.process_video(file_path, prompt)
                response["analysis"] = {
                    "prompt": prompt,
                    "result": analysis_result
                }
            except Exception as e:
                logging.error(f"Video analysis failed: {e}")
                response["analysis_error"] = f"Video analysis failed: {str(e)}"
        
        return response
        


@router.get("/videos")
async def list_videos():
    """List all uploaded videos."""
    try:
        if not os.path.exists(VIDEO_SAVE_PATH):
            return {"videos": []}
        
        videos = []
        for filename in os.listdir(VIDEO_SAVE_PATH):
            if filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.wmv')):
                file_path = os.path.join(VIDEO_SAVE_PATH, filename)
                file_size = os.path.getsize(file_path)
                videos.append({
                    "filename": filename,
                    "path": file_path,
                    "size": file_size
                })
        
        return {"videos": videos}
        
    except Exception as e:
        logging.error(f"Error listing videos: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list videos: {str(e)}")

@router.delete("/videos/{filename}")
async def delete_video(filename: str):
    """Delete a specific video file."""
    try:
        file_path = os.path.join(VIDEO_SAVE_PATH, filename)
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Video not found")
        
        os.remove(file_path)
        return {"status": "success", "message": f"Video {filename} deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error deleting video: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete video: {str(e)}")
