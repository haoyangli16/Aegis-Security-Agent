"""
Object Detection Tool for Aegis Security Agent System.

This tool provides both YOLO and OWLv2 object detection capabilities for video streams.
- YOLO: Fast detection with predefined classes
- OWLv2: Accurate detection with natural language prompts (slower but more flexible)
"""

import os
import sys
from typing import Any, Dict, List

import cv2
import numpy as np
from google.adk.tools import ToolContext
from ultralytics import YOLO

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from aegis.config.camera_sources import get_camera_url
from aegis.core.security_context import get_security_context

# Try to import OWLv2 dependencies
try:
    sys.path.append(
        os.path.abspath(os.path.join(os.path.dirname(__file__), "../../VicLab"))
    )
    from viclab.image.det_seg import OwlV2SAM

    OWLV2_AVAILABLE = True
    print("✅ OWLv2 detection available")
except ImportError as e:
    OWLV2_AVAILABLE = False
    print(f"⚠️ OWLv2 detection not available: {e}")

MONITOR_AVAILABLE = True

# Global OWLv2 model instance (initialized once for performance)
owlv2_detector = None


def detect_objects(
    video_feed_id: str,
    object_list: List[str],
    tool_context: ToolContext,
    detection_method: str = "yolo",
) -> Dict[str, Any]:
    """
    Detect specific objects in a video feed using YOLO or OWLv2 object detection.

    This tool captures a frame from the specified video feed and runs object detection
    to find the requested objects. It returns bounding box coordinates, confidence scores,
    and detection counts.

    Args:
        video_feed_id: The camera ID or video source identifier (e.g., 'cam1', 'gate_2')
        object_list: List of objects to detect (e.g., ['person', 'backpack', 'car'])
        tool_context: ADK tool context for session state and services
        detection_method: Detection method to use ("yolo" for speed, "owlv2" for accuracy)

    Returns:
        Dict containing:
        - status: Success/error status
        - detection_method: Method used for detection
        - detections: List of detected objects with bounding boxes
        - counts: Count of each detected object type
        - total_objects: Total number of objects detected
        - camera_info: Information about the camera source
    """

    try:
        # Validate detection method
        if detection_method not in ["yolo", "owlv2"]:
            return {
                "status": "error",
                "message": f"Invalid detection method '{detection_method}'. Use 'yolo' or 'owlv2'",
                "detections": [],
                "counts": {},
                "total_objects": 0,
            }

        # Check if OWLv2 is requested but not available
        if detection_method == "owlv2" and not OWLV2_AVAILABLE:
            return {
                "status": "error",
                "message": "OWLv2 detection requested but not available. Please install dependencies or use 'yolo'",
                "detections": [],
                "counts": {},
                "total_objects": 0,
            }

        # Get camera URL from video_feed_id
        camera_url = get_camera_url(video_feed_id)
        if not camera_url:
            return {
                "status": "error",
                "message": f"Camera '{video_feed_id}' not found in system",
                "available_cameras": list(
                    get_camera_url.__globals__["CAMERA_SOURCES"].keys()
                ),
                "detections": [],
                "counts": {},
                "total_objects": 0,
            }

        # Capture real frame from video source using monitor server
        frame = _capture_frame_from_source(video_feed_id)

        if frame is None:
            return {
                "status": "error",
                "message": f"Failed to capture frame from camera '{video_feed_id}'",
                "detections": [],
                "counts": {},
                "total_objects": 0,
            }

        # Run detection based on method
        if detection_method == "yolo":
            detections, object_counts = _detect_with_yolo(frame, object_list)
            method_info = "YOLO (fast)"
        else:  # owlv2
            detections, object_counts = _detect_with_owlv2(frame, object_list)
            method_info = "OWLv2 (accurate)"

        # Store detection results in session state for other tools to use
        detection_key = f"detections_{video_feed_id}"
        tool_context.state[detection_key] = {
            "timestamp": tool_context.state.get("current_time", "unknown"),
            "detection_method": detection_method,
            "detections": detections,
            "counts": object_counts,
        }

        return {
            "status": "success",
            "message": f"Detected {len(detections)} objects using {method_info} in {video_feed_id}",
            "detection_method": method_info,
            "camera_id": video_feed_id,
            "requested_objects": object_list,
            "detections": detections,
            "counts": object_counts,
            "total_objects": len(detections),
            "frame_analyzed": True,
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Object detection failed: {str(e)}",
            "detection_method": detection_method,
            "detections": [],
            "counts": {},
            "total_objects": 0,
            "error_details": str(e),
        }


def _detect_with_yolo(frame: np.ndarray, object_list: List[str]) -> tuple:
    """
    Detect objects using YOLO (fast detection with predefined classes).

    Args:
        frame: Input frame for detection
        object_list: List of objects to detect

    Returns:
        tuple: (detections list, object_counts dict)
    """
    yolo_model_path = os.path.join(
        os.path.dirname(__file__), "../checkpoints/yolo11n.pt"
    )
    yolo_model = YOLO(yolo_model_path)

    print("Running YOLO detection")
    cv2.imwrite("frame_before.jpg", frame)
    print("Frame saved to frame_before.jpg")

    results = yolo_model(frame, verbose=False, classes=list(yolo_model.names.keys()))[0]

    cv2.imwrite("frame_after.jpg", frame)
    print("Frame saved to frame_after.jpg")

    # Process detections
    detections = []
    object_counts = {}

    if results.boxes is not None:
        boxes_data = results.boxes.data.cpu().numpy()

        for box in boxes_data:
            if len(box) >= 6:  # x1, y1, x2, y2, confidence, class_id
                class_id = int(box[5])
                confidence = float(box[4])

                if class_id < len(yolo_model.names):
                    detected_object = yolo_model.names[class_id]

                    # Check if this object is in our requested list
                    if (
                        not object_list
                        or detected_object in object_list
                        or any(
                            obj.lower() in detected_object.lower()
                            for obj in object_list
                        )
                    ):
                        detection = {
                            "object": detected_object,
                            "confidence": round(confidence, 3),
                            "method": "YOLO",
                            "bounding_box": {
                                "x1": float(box[0]),
                                "y1": float(box[1]),
                                "x2": float(box[2]),
                                "y2": float(box[3]),
                                "width": float(box[2] - box[0]),
                                "height": float(box[3] - box[1]),
                            },
                        }

                        detections.append(detection)
                        object_counts[detected_object] = (
                            object_counts.get(detected_object, 0) + 1
                        )

    return detections, object_counts


def _detect_with_owlv2(frame: np.ndarray, object_list: List[str]) -> tuple:
    """
    Detect objects using OWLv2 (accurate detection with natural language prompts).

    Args:
        frame: Input frame for detection
        object_list: List of objects to detect (natural language descriptions)

    Returns:
        tuple: (detections list, object_counts dict)
    """
    global owlv2_detector

    # Initialize OWLv2 detector if not already done
    if owlv2_detector is None:
        try:
            print("Initializing OWLv2 detector...")
            sam_checkpoint = os.path.join(
                os.path.dirname(__file__),
                "../../VicLab/viclab/image/checkpoints/sam_vit_h_4b8939.pth",
            )
            owlv2_detector = OwlV2SAM(sam_checkpoint=sam_checkpoint)
            print("✅ OWLv2 detector initialized")
        except Exception as e:
            print(f"❌ Failed to initialize OWLv2 detector: {e}")
            return [], {}

    print("Running OWLv2 detection")
    cv2.imwrite("frame_before_owlv2.jpg", frame)
    print("Frame saved to frame_before_owlv2.jpg")

    # Convert BGR to RGB for OWLv2
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Prepare text prompts - if object_list is empty, use common security objects
    if not object_list:
        text_prompts = ["a person", "a car", "a bag", "a weapon", "suspicious object"]
    else:
        # Convert object names to more descriptive prompts for better OWLv2 performance
        text_prompts = []
        for obj in object_list:
            if obj.lower() in ["person", "people", "human"]:
                text_prompts.append("a person")
            elif obj.lower() in ["car", "vehicle", "automobile"]:
                text_prompts.append("a car")
            elif obj.lower() in ["bag", "backpack", "luggage", "suitcase"]:
                text_prompts.append("a bag")
            elif obj.lower() in ["weapon", "gun", "knife"]:
                text_prompts.append("a weapon")
            else:
                text_prompts.append(f"a {obj.lower()}")

    # Run OWLv2 detection
    try:
        results = owlv2_detector.detect_and_segment(
            image=rgb_frame,
            text_prompts=text_prompts,
            detection_threshold=0.15,  # Lower threshold for security use case
            return_all_detections=True,
        )

        cv2.imwrite("frame_after_owlv2.jpg", frame)
        print("Frame saved to frame_after_owlv2.jpg")

    except Exception as e:
        print(f"OWLv2 detection failed: {e}")
        return [], {}

    # Process OWLv2 detections
    detections = []
    object_counts = {}

    if results.get("detected", False) and "detections" in results:
        for detection in results["detections"]:
            box = detection["box"]  # [x1, y1, x2, y2]
            confidence = detection["score"]
            text_prompt = detection["text_prompt"]

            # Extract object name from text prompt
            object_name = text_prompt.replace("a ", "").replace("an ", "").strip()

            detection_data = {
                "object": object_name,
                "confidence": round(confidence, 3),
                "method": "OWLv2",
                "text_prompt": text_prompt,
                "bounding_box": {
                    "x1": float(box[0]),
                    "y1": float(box[1]),
                    "x2": float(box[2]),
                    "y2": float(box[3]),
                    "width": float(box[2] - box[0]),
                    "height": float(box[3] - box[1]),
                },
            }

            detections.append(detection_data)
            object_counts[object_name] = object_counts.get(object_name, 0) + 1

    return detections, object_counts


def _capture_frame_from_source(video_feed_id: str) -> np.ndarray:
    """
    Capture a real frame from the video source using the monitor server.

    Args:
        video_feed_id: Camera ID (e.g., 'cam1', '1', 'camera1')

    Returns:
        numpy.ndarray: Real captured frame from the camera, or None if capture failed
    """

    try:
        if not MONITOR_AVAILABLE:
            print("Monitor server not available for frame capture")
            return None

        # Get the security context
        try:
            context = get_security_context()
            monitor_server = context.monitor_server
        except RuntimeError:
            print("SecurityContext not initialized")
            return None

        # Convert video_feed_id to proper camera_id format (same as VLM tool)
        try:
            if video_feed_id.startswith("cam"):
                camera_id = int(video_feed_id.replace("cam", ""))
            elif video_feed_id.startswith("camera"):
                camera_id = int(video_feed_id.replace("camera", ""))
            elif video_feed_id.isdigit():
                camera_id = int(video_feed_id)
            else:
                print(f"Invalid camera ID format: {video_feed_id}")
                return None

            # Get source index from monitor server (same as VLM tool)
            source_index = monitor_server.selected_sources[camera_id - 1]

            # Get real frame from webcam loader
            frame = monitor_server.webcam_loader.get_frame(source_index)
            if frame is None:
                print(
                    f"❌ No frame available from {video_feed_id} (source_index: {source_index})"
                )

            # save the frame to a file
            cv2.imwrite("frame_after_capture.jpg", frame)
            print("Frame saved to frame_after_capture.jpg")

            if frame is not None:
                print(
                    f"✅ Successfully captured real frame from {video_feed_id} (source_index: {source_index})"
                )
                return frame
            else:
                print(
                    f"❌ No frame available from {video_feed_id} (source_index: {source_index})"
                )
                return None

        except (ValueError, IndexError, AttributeError) as e:
            print(f"Error mapping camera ID {video_feed_id}: {e}")
            return None

    except Exception as e:
        print(f"Error capturing real frame from {video_feed_id}: {e}")
        return None
