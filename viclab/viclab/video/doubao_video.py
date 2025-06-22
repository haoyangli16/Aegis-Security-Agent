# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
from typing import Optional, Tuple, List, Dict
from enum import Enum
import os
import base64
import shutil
import time
import threading
from collections import deque

import cv2
import numpy as np
from openai import OpenAI


class Strategy(Enum):
    """
    Enumeration for video frame extraction strategies.
    """
    CONSTANT_INTERVAL = "constant_interval"
    EVEN_INTERVAL = "even_interval"


class ComplexVideoUnderstander:
    """
    A class to understand video content using the Seed-1.5-VL model.
    It handles video preprocessing, frame extraction, real-time streaming, and interaction with the OpenAI API.
    """
    def __init__(
        self,
        base_url: str = "https://ark.cn-beijing.volces.com/api/v3",
        seed_vl_version: str = "doubao-1-5-thinking-vision-pro-250428",
        openai_api_key: Optional[str] = "ffb01810-79f6-402d-9ee6-46c109b3f93f",
        default_output_dir: str = "video_frames",
        max_frames_buffer: int = 100,
    ):
        if openai_api_key is None:
            openai_api_key = os.environ.get("OPENAI_API_KEY")

        self.client = OpenAI(base_url=base_url, api_key=openai_api_key)
        self.seed_vl_version = seed_vl_version
        self.default_output_dir = default_output_dir
        
        # Frame buffer for streaming (similar to SmolVLMRealtimeProcessor)
        self.max_frames_buffer = max_frames_buffer
        self.frame_buffer = deque(maxlen=max_frames_buffer)
        self.frame_timestamps = deque(maxlen=max_frames_buffer)
        self.is_streaming = False
        self.stream_thread = None
        self.capture = None

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        if height < width:
            target_height, target_width = 480, 640
        else:
            target_height, target_width = 640, 480
        if height <= target_height and width <= target_width:
            return image
        if height / target_height < width / target_width:
            new_width = target_width
            new_height = int(height * (new_width / width))
        else:
            new_height = target_height
            new_width = int(width * (new_height / height))
        return cv2.resize(image, (new_width, new_height))

    def _encode_image_to_base64(self, image_path: str) -> str:
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {image_path}")
        image_resized = self._resize_image(image)
        _, encoded_image_bytes = cv2.imencode(".jpg", image_resized)
        return base64.b64encode(encoded_image_bytes).decode("utf-8")

    def _encode_frame_to_base64(self, frame: np.ndarray) -> str:
        """Encode a frame (numpy array) to base64 string"""
        frame_resized = self._resize_image(frame)
        _, encoded_image_bytes = cv2.imencode(".jpg", frame_resized)
        return base64.b64encode(encoded_image_bytes).decode("utf-8")

    def _construct_llm_messages(
        self, image_paths: List[str], timestamps: Optional[List[float]], prompt: str
    ) -> List[Dict]:
        content = []
        for idx, image_path in enumerate(image_paths):
            if timestamps:
                content.append({"type": "text", "text": f"[{timestamps[idx]} second]"})
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{self._encode_image_to_base64(image_path)}",
                        "detail": "low",
                    },
                }
            )
        content.append({"type": "text", "text": prompt})
        return [{"role": "user", "content": content}]

    def _construct_llm_messages_from_frames(
        self, frames: List[np.ndarray], timestamps: Optional[List[float]], prompt: str
    ) -> List[Dict]:
        """Construct LLM messages from frame arrays instead of file paths"""
        content = []
        for idx, frame in enumerate(frames):
            if timestamps:
                content.append({"type": "text", "text": f"[{timestamps[idx]:.2f} second]"})
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{self._encode_frame_to_base64(frame)}",
                        "detail": "low",
                    },
                }
            )
        content.append({"type": "text", "text": prompt})
        return [{"role": "user", "content": content}]

    def _call_api(self, messages: List[Dict]):
        response = self.client.chat.completions.create(
            model=self.seed_vl_version, messages=messages
        )
        if response.choices:
            return response.choices[0]
        return None

    def _capture_frames(self, video_source, fps_limit: Optional[int] = None):
        """
        Continuously capture frames from video source in a separate thread.
        Similar to SmolVLMRealtimeProcessor._capture_frames
        """
        self.capture = cv2.VideoCapture(video_source)
        if not self.capture.isOpened():
            raise ValueError(f"Could not open video source: {video_source}")
            
        original_fps = self.capture.get(cv2.CAP_PROP_FPS)
        if fps_limit and fps_limit > 0:
            frame_interval = 1.0 / fps_limit
        else:
            frame_interval = 1.0 / original_fps if original_fps > 0 else 1.0 / 30
            
        print(f"Starting frame capture at {1/frame_interval:.1f} FPS")
        
        last_frame_time = 0
        
        while self.is_streaming:
            current_time = time.time()
            
            # Control frame rate
            if current_time - last_frame_time < frame_interval:
                time.sleep(0.001)  # Small sleep to prevent busy waiting
                continue
                
            ret, frame = self.capture.read()
            if not ret:
                print("Failed to read frame from video source")
                if isinstance(video_source, str):  # If it's a file, we might have reached the end
                    break
                continue
                
            # Add frame to buffer with timestamp
            self.frame_buffer.append(frame.copy())
            self.frame_timestamps.append(current_time)
            last_frame_time = current_time
            
        self.capture.release()
        print("Frame capture stopped")

    def start_stream(self, video_source, fps_limit: Optional[int] = 10):
        """
        Start capturing frames from video source.
        Similar to SmolVLMRealtimeProcessor.start_stream
        """
        if self.is_streaming:
            print("Stream is already running")
            return
            
        self.is_streaming = True
        self.frame_buffer.clear()
        self.frame_timestamps.clear()
        
        # Start frame capture in a separate thread
        self.stream_thread = threading.Thread(
            target=self._capture_frames, 
            args=(video_source, fps_limit)
        )
        self.stream_thread.start()
        print(f"Started video stream from: {video_source}")

    def stop_stream(self):
        """Stop the video stream. Similar to SmolVLMRealtimeProcessor.stop_stream"""
        if not self.is_streaming:
            return
            
        self.is_streaming = False
        if self.stream_thread:
            self.stream_thread.join(timeout=2.0)
        if self.capture:
            self.capture.release()
        print("Video stream stopped")

    def analyze_recent_frames(
        self, 
        prompt: str, 
        n_frames: int = 10,
        min_frames: int = 3
    ) -> Optional[str]:
        """
        Analyze the most recent N frames from the stream.
        Similar to SmolVLMRealtimeProcessor.analyze_recent_frames
        """
        if len(self.frame_buffer) < min_frames:
            print(f"Not enough frames in buffer: {len(self.frame_buffer)}/{min_frames}")
            return None
            
        # Get the latest n_frames from the buffer
        latest_frames = list(self.frame_buffer)[-n_frames:]
        latest_timestamps = list(self.frame_timestamps)[-n_frames:]
        
        if not latest_frames:
            return None
            
        print(f"Analyzing {len(latest_frames)} recent frames")
        
        try:
            # Adjust timestamps to be relative to the first frame
            if latest_timestamps:
                base_time = latest_timestamps[0]
                relative_timestamps = [t - base_time for t in latest_timestamps]
            else:
                relative_timestamps = None
            
            # Construct messages from frame arrays
            messages = self._construct_llm_messages_from_frames(
                latest_frames, relative_timestamps, prompt
            )
            
            # Call API
            result = self._call_api(messages)
            
            if result and hasattr(result, "message") and hasattr(result.message, "content"):
                return result.message.content
            return None
                
        except Exception as e:
            print(f"Error in frame analysis: {e}")
            return None

    def get_buffer_info(self) -> dict:
        """Get information about the current frame buffer. Similar to SmolVLMRealtimeProcessor."""
        return {
            "buffer_size": len(self.frame_buffer),
            "max_buffer_size": self.max_frames_buffer,
            "is_streaming": self.is_streaming,
            "latest_timestamp": self.frame_timestamps[-1] if self.frame_timestamps else None,
            "buffer_time_span": (
                self.frame_timestamps[-1] - self.frame_timestamps[0] 
                if len(self.frame_timestamps) > 1 else 0
            )
        }

    def process_video_stream(
        self,
        video_source,
        prompt: str,
        analysis_interval: float = 2.0,
        n_frames: int = 8,
        fps_limit: int = 10,
        callback=None
    ):
        """
        Continuously process video stream and analyze recent frames at regular intervals.
        Similar to SmolVLMRealtimeProcessor.process_video_stream
        """
        print(f"Starting real-time video stream analysis")
        
        try:
            # Start the stream
            self.start_stream(video_source, fps_limit)
            
            # Wait a bit for frames to accumulate
            time.sleep(1.0)
            
            analysis_count = 0
            
            while self.is_streaming:
                start_time = time.time()
                
                # Analyze recent frames
                result = self.analyze_recent_frames(prompt, n_frames)
                
                if result:
                    analysis_count += 1
                    end_time = time.time()
                    analysis_time = end_time - start_time
                    
                    # Get timestamps of analyzed frames
                    recent_timestamps = list(self.frame_timestamps)[-n_frames:] if len(self.frame_timestamps) >= n_frames else list(self.frame_timestamps)
                    
                    # print(f"\n--- Analysis #{analysis_count} ---")  # Removed debug prints to prevent terminal spam
                    # print(f"Frames in buffer: {len(self.frame_buffer)}")
                    # print(f"Analysis time: {analysis_time:.2f}s")
                    # print(f"Result: {result}")
                    # print("-" * 50)
                    
                    # Call callback if provided
                    if callback:
                        callback(result, recent_timestamps)
                        
                # Wait for next analysis
                elapsed = time.time() - start_time
                sleep_time = max(0, analysis_interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
        except KeyboardInterrupt:
            print("Stream analysis interrupted by user")
        except Exception as e:
            print(f"Error in stream processing: {e}")
        finally:
            self.stop_stream()

    def _preprocess_video(
        self,
        video_file_path: str,
        output_dir: str,
        extraction_strategy: Strategy = Strategy.EVEN_INTERVAL,
        interval_in_seconds: float = 1.0,
        max_frames: int = 10,
        use_timestamp: bool = True,
        keyframe_naming_template: str = "frame_{:04d}.jpg",
    ) -> Tuple[List[str], Optional[List[float]]]:
        if not os.path.exists(video_file_path):
            raise FileNotFoundError(f"Video file not found at {video_file_path}")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        else:
            shutil.rmtree(output_dir)
            os.makedirs(output_dir)
        cap = cv2.VideoCapture(video_file_path)
        if not cap.isOpened():
            raise IOError(f"Could not open video file: {video_file_path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            print(f"Warning: FPS of video {video_file_path} is 0. Defaulting to 25 FPS for frame interval calculation.")
            fps = 25.0
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if length == 0:
            raise ValueError(f"Video {video_file_path} has no frames.")
        if extraction_strategy == Strategy.CONSTANT_INTERVAL:
            frame_interval = int(fps * interval_in_seconds)
        elif extraction_strategy == Strategy.EVEN_INTERVAL:
            frame_interval = max(1, int(length / max_frames))
        else:
            raise ValueError("Invalid extraction strategy")
        frame_count = 0
        keyframes = []
        timestamps = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % frame_interval == 0:
                image_path = os.path.join(
                    output_dir, keyframe_naming_template.format(len(keyframes))
                )
                cv2.imwrite(image_path, frame)
                keyframes.append(image_path)
                if use_timestamp:
                    timestamps.append(frame_count / fps)
                if len(keyframes) >= max_frames:
                    break
            frame_count += 1
        cap.release()
        if use_timestamp:
            return keyframes, timestamps
        else:
            return keyframes, None

    def analyze_video(
        self,
        video_path: str,
        prompt: str,
        extraction_strategy: Strategy = Strategy.CONSTANT_INTERVAL,
        sampling_fps: float = 1.0,
        max_frames: int = 30,
        use_timestamp: bool = True,
        cleanup_frames: bool = True,
    ) -> Optional[str]:
        output_dir_to_use = self.default_output_dir
        keyframes, timestamps = self._preprocess_video(
            video_file_path=video_path,
            output_dir=output_dir_to_use,
            extraction_strategy=extraction_strategy,
            interval_in_seconds=1.0 / sampling_fps if extraction_strategy == Strategy.CONSTANT_INTERVAL and sampling_fps > 0 else 1.0,
            max_frames=max_frames,
            use_timestamp=use_timestamp,
        )
        messages = self._construct_llm_messages(keyframes, timestamps if use_timestamp else None, prompt)
        result = self._call_api(messages)
        if cleanup_frames:
            try:
                shutil.rmtree(output_dir_to_use)
            except Exception as e:
                print(f"Error cleaning up frame directory {output_dir_to_use}: {e}")
        if result and hasattr(result, "message") and hasattr(result.message, "content"):
            return result.message.content
        return None 