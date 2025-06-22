"""
Advanced frame analysis engine using YOLO object detection and VLM scene understanding.

This module provides the core computer vision and AI analysis capabilities for the
Aegis Security Agent System.
"""

import os
import time
import cv2
import base64
import numpy as np
from collections import Counter, deque
from ultralytics import YOLO
from PIL import Image
import sys
import threading
from typing import Dict, List, Any, Optional

# Try to import VLM processor - make it optional for ADK deployment
try:
    # Add viclab to path if available
    from viclab.video.realtime_video import SmolVLMRealtimeProcessor
    VLM_AVAILABLE = True
except ImportError:
    VLM_AVAILABLE = False
    print("Warning: VLM dependencies not available. VLM analysis will be disabled.")

# Security keyword definitions
ABNORMAL_KEYWORDS = [
    "knife", "gun", "weapon", "fighting", "violence", "punching",
    "kicking", "blood", "screaming", "running aggressively",
    "pepper spray", "falling", "breaking", "fire", "explosion"
]

SUSPICIOUS_BEHAVIOR_KEYWORDS = [
    "sitting", "sitting on the ground", "squatting", "on the ground", "crouching"
]

# Global instances - will be initialized in init_analysis_engine()
yolo_model = None
vlm_processor = None
vlm_lock = threading.Lock()

def init_analysis_engine(model_path: Optional[str] = None) -> bool:
    """
    Initialize the analysis engine with YOLO and VLM models.
    
    Args:
        model_path: Optional path to YOLO model. If None, uses default.
        
    Returns:
        bool: True if initialization successful, False otherwise.
    """
    global yolo_model, vlm_processor
    
    try:
        # Initialize YOLO model
        if model_path is None:
            # Try to find YOLO model in common locations
            possible_paths = [
                "../checkpoints/yolo11n.pt",
                "../../checkpoints/yolo11n.pt", 
                "yolo11n.pt"
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    model_path = path
                    break
        
        if model_path and os.path.exists(model_path):
            yolo_model = YOLO(model_path)
            print(f"✅ YOLO model loaded from: {model_path}")
        else:
            # Download default model
            yolo_model = YOLO("yolo11n.pt")
            print("✅ YOLO model downloaded and loaded (yolo11n.pt)")
        
        # Initialize VLM processor if available
        if VLM_AVAILABLE:
            vlm_processor = SmolVLMRealtimeProcessor(max_frames_buffer=100)
            print("✅ VLM processor initialized")
        else:
            print("⚠️ VLM processor not available")
            
        return True
        
    except Exception as e:
        print(f"❌ Error initializing analysis engine: {e}")
        return False

class FrameBuffer:
    """Manages frame buffer for video stream analysis"""
    
    def __init__(self, max_frames=90, target_fps=30):  # 3 seconds at 30fps
        self.max_frames = max_frames
        self.target_fps = target_fps
        self.frames = deque(maxlen=max_frames)
        self.timestamps = deque(maxlen=max_frames)
        self.lock = threading.Lock()
        
    def add_frame(self, frame):
        """Add frame to buffer with timestamp"""
        with self.lock:
            self.frames.append(frame.copy())
            self.timestamps.append(time.time())
        
    def get_recent_frames(self, duration_seconds=3.0):
        """Get frames from the last N seconds"""
        with self.lock:
            if not self.frames:
                return []
                
            current_time = time.time()
            cutoff_time = current_time - duration_seconds
            
            recent_frames = []
            for i, timestamp in enumerate(self.timestamps):
                if timestamp >= cutoff_time:
                    recent_frames.extend(list(self.frames)[i:])
                    break
                    
            return recent_frames[-min(len(recent_frames), self.max_frames):]
    
    def get_latest_frame(self):
        """Get the most recent frame"""
        with self.lock:
            return self.frames[-1] if self.frames else None
    
    def clear(self):
        """Clear the buffer"""
        with self.lock:
            self.frames.clear()
            self.timestamps.clear()

class SecurityDecisionEngine:
    """Handles security decision logic based on YOLO and VLM results"""
    
    def __init__(self):
        self.abnormal_keywords = ABNORMAL_KEYWORDS
        self.suspicious_keywords = SUSPICIOUS_BEHAVIOR_KEYWORDS
        
    def analyze_yolo_results(self, yolo_results) -> Dict[str, Any]:
        """Analyze YOLO detection results"""
        if not yolo_results or not hasattr(yolo_results, 'boxes') or yolo_results.boxes is None:
            return {
                "person_count": 0,
                "car_count": 0,
                "motorcycle_count": 0,
                "detected_objects": [],
                "abnormal_objects": [],
                "bounding_boxes": []
            }
            
        # Extract labels and bounding boxes
        boxes_data = yolo_results.boxes.data.cpu().numpy() if yolo_results.boxes.data is not None else []
        labels = []
        bounding_boxes = []
        
        for box in boxes_data:
            if len(box) >= 6:  # x1, y1, x2, y2, confidence, class_id
                class_id = int(box[5])
                confidence = float(box[4])
                
                if class_id < len(yolo_model.names):
                    label = yolo_model.names[class_id]
                    labels.append(label)
                    
                    # Store bounding box info
                    bounding_boxes.append({
                        "label": label,
                        "confidence": confidence,
                        "bbox": [float(box[0]), float(box[1]), float(box[2]), float(box[3])]  # x1, y1, x2, y2
                    })
        
        counts = Counter(labels)
        
        # Check for abnormal objects
        abnormal_objects = [label for label in labels if label.lower() in self.abnormal_keywords]
        
        return {
            "person_count": counts.get("person", 0),
            "car_count": counts.get("car", 0),
            "motorcycle_count": counts.get("motorcycle", 0),
            "bicycle_count": counts.get("bicycle", 0),
            "detected_objects": list(counts.keys()),
            "abnormal_objects": abnormal_objects,
            "bounding_boxes": bounding_boxes,
            "total_detections": len(labels)
        }
    
    def analyze_vlm_description(self, vlm_description: str) -> Dict[str, Any]:
        """Analyze VLM description for suspicious behavior"""
        if not vlm_description:
            return {
                "has_abnormal_behavior": False,
                "has_suspicious_behavior": False,
                "matched_keywords": []
            }
            
        description_lower = vlm_description.lower()
        
        # Check for abnormal keywords
        abnormal_matches = [kw for kw in self.abnormal_keywords if kw in description_lower]
        suspicious_matches = [kw for kw in self.suspicious_keywords if kw in description_lower]
        
        return {
            "has_abnormal_behavior": len(abnormal_matches) > 0,
            "has_suspicious_behavior": len(suspicious_matches) > 0,
            "matched_keywords": abnormal_matches + suspicious_matches
        }
    
    def make_security_decision(self, yolo_analysis: Dict, vlm_analysis: Dict, vlm_description: str = "") -> Dict[str, Any]:
        """Make final security decision based on all analysis"""
        label = "safe"
        command = "No action needed"
        matched = None
        description = vlm_description or "No abnormal behavior detected"
        threat_level = "low"
        
        # Check YOLO abnormal objects
        if yolo_analysis["abnormal_objects"]:
            label = "abnormal"
            threat_level = "high"
            matched = yolo_analysis["abnormal_objects"][0]
            command = "Security alert: Dangerous object detected"
        
        # Check VLM abnormal behavior
        elif vlm_analysis["has_abnormal_behavior"]:
            label = "abnormal"
            threat_level = "high"
            matched = vlm_analysis["matched_keywords"][0] if vlm_analysis["matched_keywords"] else "abnormal behavior"
            command = "Security alert: Abnormal behavior detected"
        
        # Check crowd control scenarios
        elif yolo_analysis["person_count"] >= 5:
            if vlm_analysis["has_suspicious_behavior"]:
                label = "abnormal"
                threat_level = "medium"
                command = "Send police"
                matched = "crowd suspicious behavior"
            else:
                threat_level = "low"
                command = "Let people keep walking"
        
        # Check traffic scenarios
        elif yolo_analysis["car_count"] >= 5:
            threat_level = "low"
            command = "Send security guard to run the traffic"
        
        return {
            "label": label,
            "description": description,
            "matched": matched,
            "command": command,
            "threat_level": threat_level,
            "yolo_analysis": yolo_analysis,
            "vlm_analysis": vlm_analysis
        }

# Global instances
frame_buffers = {}  # One buffer per camera
decision_engine = SecurityDecisionEngine()

def get_frame_buffer(camera_id: str) -> FrameBuffer:
    """Get or create frame buffer for camera"""
    if camera_id not in frame_buffers:
        frame_buffers[camera_id] = FrameBuffer()
    return frame_buffers[camera_id]

def analyze_frame(frame, camera_id: str = "default", use_vlm: bool = True) -> Dict[str, Any]:
    """
    Analyze frame using SmolVLM for video analysis and YOLO for object detection
    
    Args:
        frame: Frame from camera. Can be PIL Image or OpenCV frame (BGR)
        camera_id: Camera identifier for frame buffer management
        use_vlm: Whether to use VLM analysis (can be disabled for performance)
        
    Returns:
        Dict containing analysis results
    """
    
    # Initialize engine if not done
    if yolo_model is None:
        if not init_analysis_engine():
            return {
                "label": "error",
                "description": "Analysis engine not initialized",
                "error": "Failed to initialize YOLO model"
            }
    
    try:
        # Ensure frame is in OpenCV format (BGR numpy array)
        if isinstance(frame, Image.Image):
            frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
        
        # Get frame buffer for this camera
        buffer = get_frame_buffer(camera_id)
        buffer.add_frame(frame)
        
        # Run YOLO on latest frame
        yolo_results = yolo_model(frame, verbose=False, classes=list(yolo_model.names.keys()))[0]
        annotated_frame = yolo_results.plot()
        
        # Analyze YOLO results
        yolo_analysis = decision_engine.analyze_yolo_results(yolo_results)
        
        # VLM analysis on recent frames
        vlm_description = ""
        vlm_analysis = {"has_abnormal_behavior": False, "has_suspicious_behavior": False, "matched_keywords": []}
        
        if use_vlm and VLM_AVAILABLE and vlm_processor:
            recent_frames = buffer.get_recent_frames(duration_seconds=3.0)
            
            if len(recent_frames) >= 8:  # Need at least 8 frames for video analysis
                with vlm_lock:
                    try:
                        # Clear VLM processor buffer and add our frames
                        vlm_processor.frame_buffer.clear()
                        vlm_processor.frame_timestamps.clear()
                        
                        # Add frames to VLM processor (convert BGR to RGB)
                        current_time = time.time()
                        for i, bgr_frame in enumerate(recent_frames):
                            bgr_frame = np.array(bgr_frame)
                            rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
                            vlm_processor.frame_buffer.append(rgb_frame)
                            vlm_processor.frame_timestamps.append(current_time - (len(recent_frames) - i) * 0.1)
                        
                        # Analyze with VLM
                        prompt = "Describe what is happening in this video. Focus on any unusual, dangerous, or suspicious behavior. Response in 25 words or less."
                        vlm_description = vlm_processor.analyze_recent_frames(
                            prompt=prompt,
                            n_frames=min(15, len(recent_frames)),
                            min_frames=8
                        )
                        
                        if vlm_description:
                            vlm_analysis = decision_engine.analyze_vlm_description(vlm_description)
                        
                    except Exception as e:
                        print(f"VLM analysis error: {e}")
                        vlm_description = f"VLM analysis failed: {str(e)}"
        
        # Make final decision
        decision = decision_engine.make_security_decision(yolo_analysis, vlm_analysis, vlm_description)
        
        # Save alert frames if needed
        if decision["label"] == "abnormal" or decision["command"] != "No action needed":
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            os.makedirs("alert_frames", exist_ok=True)
            alert_path = os.path.join("alert_frames", f"alert_{timestamp}_cam{camera_id}.jpg")
            cv2.imwrite(alert_path, annotated_frame)
        
        return {
            "label": decision["label"],
            "description": decision["description"],
            "matched": decision["matched"],
            "command": decision["command"],
            "threat_level": decision["threat_level"],
            "annotated_frame": annotated_frame,
            "yolo_analysis": decision["yolo_analysis"],
            "raw_vlm_description": vlm_description,
            "vlm_analysis": decision["vlm_analysis"],
            "buffer_size": len(buffer.frames),
            "timestamp": time.time(),
            "camera_id": camera_id
        }
        
    except Exception as e:
        print(f"Frame analysis error: {e}")
        return {
            "label": "error",
            "description": f"Analysis failed: {str(e)}",
            "matched": None,
            "command": "No action due to error",
            "threat_level": "unknown",
            "annotated_frame": frame,
            "yolo_analysis": {},
            "vlm_analysis": {},
            "buffer_size": 0,
            "error": str(e)
        }

def clean_result_for_json(data: Any) -> Any:
    """Clean result for JSON serialization"""
    if isinstance(data, dict):
        return {k: clean_result_for_json(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [clean_result_for_json(item) for item in data]
    elif isinstance(data, (np.integer, np.int32, np.int64)):
        return int(data)
    elif isinstance(data, (np.floating, np.float32, np.float64)):
        return float(data)
    elif isinstance(data, np.ndarray):
        return data.tolist()
    else:
        return data

def clear_camera_buffer(camera_id: str):
    """Clear frame buffer for specific camera"""
    if camera_id in frame_buffers:
        frame_buffers[camera_id].clear()

def get_buffer_status() -> Dict[str, Dict]:
    """Get status of all frame buffers"""
    return {
        camera_id: {
            "buffer_size": len(buffer.frames),
            "latest_timestamp": buffer.timestamps[-1] if buffer.timestamps else None
        }
        for camera_id, buffer in frame_buffers.items()
    } 