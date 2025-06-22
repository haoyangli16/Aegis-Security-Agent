import base64
import logging
import textwrap
import threading
import time
from typing import Any, Dict, Optional

import cv2
import numpy as np
from yt_dlp import YoutubeDL

from aegis.config.camera_sources import CAMERA_SOURCES, get_camera_info, get_camera_url

# Import our core modules
from aegis.core.analyze_frame import (
    analyze_frame,
    init_analysis_engine,
)
# Lazy import to avoid circular dependency - import moved to function level

# Configure logging
logging.basicConfig(level=logging.WARNING)  # Only show warnings and errors
logger = logging.getLogger(__name__)


class WebcamLoader:
    def __init__(self, camera_sources: Dict[str, Dict[str, Any]]):
        self.sources = []  # List of resolved stream URLs
        self.frames = {}  # Latest frame per source index
        self.captures = {}  # OpenCV VideoCapture objects
        self.running = {}  # Track whether each stream is running

        self.camera_sources = camera_sources
        self._load_stream_urls()

    def _get_youtube_stream_url(self, video_url: str) -> Optional[str]:
        try:
            ydl_opts = {
                "format": "best[height<=480][fps<=30]/best",
                "quiet": True,
                "nocheckcertificate": True,
            }
            with YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(video_url, download=False)
                return info.get("url")
        except Exception as e:
            logger.error(f"Failed to extract stream URL for {video_url}: {e}")
            return None

    def _load_stream_urls(self):
        for cam_id, cam_info in self.camera_sources.items():
            url = cam_info.get("url")
            resolved_url = self._get_youtube_stream_url(url) if url else None
            if resolved_url:
                self.sources.append(resolved_url)
                logger.info(f"✅ Resolved stream for {cam_id} in {cam_info.get('location')}")
            else:
                logger.warning(f"❌ Failed to resolve stream for {cam_id}")

    def start_all_streams(self):
        for i, url in enumerate(self.sources):
            thread = threading.Thread(
                target=self._capture_stream, args=(i, url), daemon=True
            )
            self.running[i] = True
            thread.start()

    def stop_all_streams(self):
        for i in self.running:
            self.running[i] = False
        for cap in self.captures.values():
            cap.release()

    def _capture_stream(self, index: int, stream_url: str):
        cap = cv2.VideoCapture(stream_url)
        self.captures[index] = cap
        # logger.info(f"🎥 Started capture thread for stream {index}")

        while self.running.get(index, False):
            ret, frame = cap.read()
            if ret:
                self.frames[index] = frame
            else:
                logger.warning(f"⚠️ Frame read failed for stream {index}")
            cv2.waitKey(50)

        cap.release()
        logger.info(f"🛑 Stopped stream capture for index {index}")

    def get_frame(self, index: int) -> Optional[np.ndarray]:
        return self.frames.get(index)


class WebMonitorServer:
    """Web-based webcam monitoring server for ADK integration"""

    def __init__(self, port=4000):
        self.port = port
        self.selected_sources = [0, 1, 2, 3]  # Default to first 4 cameras
        self.running = True
        self.show_bounding_boxes = False  # Hidden by default for ADK

        self.webcam_loader = WebcamLoader(CAMERA_SOURCES)

        # Initialize analysis engine
        init_analysis_engine()
        
        # Analysis throttling - only analyze every N seconds per camera
        self.analysis_interval = 10.0  # Analyze every 10 seconds (configurable)
        self.last_analysis_time = {}  # Track last analysis time per camera
        self.cached_results = {}  # Cache analysis results per camera

        self.webcam_loader.start_all_streams()

    def toggle_bounding_boxes(self, enabled: bool):
        self.show_bounding_boxes = enabled
        logger.info(f"Bounding boxes {'enabled' if enabled else 'disabled'}")
    
    def set_analysis_interval(self, interval_seconds: float):
        """Set the analysis interval (how often to analyze frames per camera)"""
        self.analysis_interval = max(1.0, interval_seconds)  # Minimum 1 second
        logger.info(f"Analysis interval set to {self.analysis_interval} seconds")
        
    def disable_background_analysis(self):
        """Disable automatic background analysis entirely"""
        self.analysis_interval = float('inf')  # Never analyze automatically
        logger.info("Background analysis disabled - analysis only on demand")

    def get_frame_as_jpeg(self, source_index: int, camera_id: int) -> Optional[bytes]:
        raw_frame = self.webcam_loader.get_frame(source_index)

        if not isinstance(raw_frame, np.ndarray) or raw_frame.size == 0:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(
                frame,
                "No Signal",
                (200, 240),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (255, 255, 255),
                3,
            )
            ret, buffer = cv2.imencode(".jpg", frame)
            return buffer.tobytes() if ret else None

        frame = raw_frame.copy()

        try:
            # Check if we should analyze this frame (throttling)
            current_time = time.time()
            should_analyze = (
                camera_id not in self.last_analysis_time or 
                current_time - self.last_analysis_time[camera_id] >= self.analysis_interval
            )
            
            if should_analyze:
                # Perform analysis and cache result
                result = analyze_frame(frame, camera_id=camera_id)
                self.last_analysis_time[camera_id] = current_time
                self.cached_results[camera_id] = result
                logger.info(f"🔍 Analyzed camera {camera_id} (next analysis in {self.analysis_interval}s)")
                
                # Update detection result
                try:
                    from aegis.tools.update_server_data_agent import update_detection_result
                    update_detection_result(camera_id, result)
                except ImportError:
                    logger.warning("Could not import update_detection_result - skipping update")
            else:
                # Use cached result to avoid repeated analysis
                result = self.cached_results.get(camera_id, {
                    "label": "safe", 
                    "description": "Awaiting analysis...",
                    "annotated_frame": frame
                })
            
            if self.show_bounding_boxes and "annotated_frame" in result:
                frame = result["annotated_frame"]
            self._add_status_overlay(frame, result)
        except Exception as e:
            logger.error(f"Error analyzing frame for camera {camera_id}: {e}")

        ret, buffer = cv2.imencode(".jpg", frame)
        return buffer.tobytes() if ret else None

    def _add_status_overlay(self, frame: np.ndarray, analysis_result: Dict[str, Any]):
        h, w, _ = frame.shape
        info_bar_height = 80
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, info_bar_height), (0, 0, 0), -1)
        alpha = 0.6
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        is_safe = analysis_result.get("label") == "safe"
        description = analysis_result.get("description", "Analysis running...")

        status_text = "Status: SAFE" if is_safe else "Status: ALERT"
        status_color = (0, 255, 0) if is_safe else (0, 0, 255)

        cv2.putText(
            frame, status_text, (60, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2
        )

        if description:
            wrapper = textwrap.TextWrapper(width=90)
            lines = wrapper.wrap(text=str(description))
            y_text = 65
            for i, line in enumerate(lines[:2]):
                cv2.putText(
                    frame,
                    line,
                    (60, y_text + i * 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )

    def get_frame_as_base64(self, source_index: int, camera_id: int) -> Optional[str]:
        jpeg_bytes = self.get_frame_as_jpeg(source_index, camera_id)
        return base64.b64encode(jpeg_bytes).decode("utf-8") if jpeg_bytes else None

    def get_stream_info(self) -> list:
        streams = []
        for i, source_index in enumerate(self.selected_sources):
            if source_index < len(self.webcam_loader.sources):
                streams.append(
                    {
                        "camera_id": i + 1,
                        "url": self.webcam_loader.sources[source_index],
                        "active": self.webcam_loader.running.get(source_index, False),
                        "has_frame": self.webcam_loader.get_frame(source_index)
                        is not None,
                        "type": "YouTube",
                    }
                )
            else:
                streams.append(
                    {
                        "camera_id": i + 1,
                        "url": "None",
                        "active": False,
                        "has_frame": False,
                        "type": "none",
                    }
                )
        return streams

    def stop_monitoring(self):
        self.running = False
        self.webcam_loader.stop_all_streams()
