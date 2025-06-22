import logging
import time
from typing import List, Union, Optional, Callable
import os
from collections import deque
import threading

import cv2
import torch
import numpy as np
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SmolVLMRealtimeProcessor:
    """
    A processor for handling images, videos, and real-time video streams
    using the SmolVLM model.
    """

    def __init__(
        self,
        model_id: str = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
        device: str = "cuda",
        max_frames_buffer: int = 100,
    ):
        """
        Initializes the SmolVLMRealtimeProcessor.

        Args:
            model_id (str): The ID of the SmolVLM model to use.
            device (str): The device to run the model on ('cuda' or 'cpu').
            max_frames_buffer (int): Maximum number of frames to keep in the rolling buffer.
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map=self.device
        )
        logger.info(f"Loaded model {model_id}")
        
        # Frame buffer for streaming
        self.max_frames_buffer = max_frames_buffer
        self.frame_buffer = deque(maxlen=max_frames_buffer)
        self.frame_timestamps = deque(maxlen=max_frames_buffer)
        self.is_streaming = False
        self.stream_thread = None
        self.capture = None

    def _extract_frames_from_buffer(self, n_frames: int = 8) -> List[Image.Image]:
        """
        Extract the latest N frames from the buffer and convert them to PIL Images.
        
        Args:
            n_frames (int): Number of recent frames to extract.
            
        Returns:
            List[Image.Image]: List of PIL Images.
        """
        if len(self.frame_buffer) == 0:
            return []
            
        # Get the latest n_frames from the buffer
        latest_frames = list(self.frame_buffer)[-n_frames:]
        
        pil_frames = []
        for frame in latest_frames:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Convert to PIL Image
            pil_image = Image.fromarray(frame_rgb)
            # Resize to a standard size for the model
            pil_image = pil_image.resize((384, 384), Image.Resampling.LANCZOS)
            pil_frames.append(pil_image)
            
        return pil_frames

    def _capture_frames(self, video_source: Union[int, str], fps_limit: Optional[int] = None):
        """
        Continuously capture frames from video source in a separate thread.
        
        Args:
            video_source (Union[int, str]): Video source (webcam index or video file path).
            fps_limit (Optional[int]): Limit the capture FPS to reduce processing load.
        """
        self.capture = cv2.VideoCapture(video_source)
        if not self.capture.isOpened():
            raise ValueError(f"Could not open video source: {video_source}")
            
        original_fps = self.capture.get(cv2.CAP_PROP_FPS)
        if fps_limit and fps_limit > 0:
            frame_interval = 1.0 / fps_limit
        else:
            frame_interval = 1.0 / original_fps if original_fps > 0 else 1.0 / 30
            
        logger.info(f"Starting frame capture at {1/frame_interval:.1f} FPS")
        
        last_frame_time = 0
        
        while self.is_streaming:
            current_time = time.time()
            
            # Control frame rate
            if current_time - last_frame_time < frame_interval:
                time.sleep(0.001)  # Small sleep to prevent busy waiting
                continue
                
            ret, frame = self.capture.read()
            if not ret:
                logger.warning("Failed to read frame from video source")
                if isinstance(video_source, str):  # If it's a file, we might have reached the end
                    break
                continue
                
            # Add frame to buffer with timestamp
            self.frame_buffer.append(frame.copy())
            self.frame_timestamps.append(current_time)
            last_frame_time = current_time
            
        self.capture.release()
        logger.info("Frame capture stopped")

    def _generate_response(
        self, conversation: List[dict], max_new_tokens: int = 128
    ) -> str:
        """
        Generates a response from the model for a given conversation.

        Args:
            conversation (List[dict]): The conversation history.
            max_new_tokens (int): The maximum number of new tokens to generate.

        Returns:
            str: The generated text response.
        """
        try:
            # Extract images and prepare inputs
            images = []
            for msg in conversation:
                if "content" in msg:
                    for content_item in msg["content"]:
                        if isinstance(content_item, dict) and content_item.get("type") == "image":
                            # Handle both direct PIL Images and image references
                            if "image" in content_item:
                                images.append(content_item["image"])
                        elif isinstance(content_item, Image.Image):
                            # Direct PIL Image in content
                            images.append(content_item)
            
            # Apply chat template
            prompt = self.processor.apply_chat_template(
                conversation, add_generation_prompt=True
            )
            
            # Process inputs - handle both single image and multiple images
            inputs = self.processor(
                text=prompt, 
                images=images if images else None, 
                return_tensors="pt"
            )
            inputs = inputs.to(self.model.device, dtype=torch.bfloat16)

            generated_ids = self.model.generate(
                **inputs, do_sample=False, max_new_tokens=max_new_tokens
            )
            generated_texts = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )

            results = generated_texts[0].strip()

            # print(f"{results}")  # Removed debug print to prevent terminal spam
            # Optional: Use proper logging instead
            logger.debug(f"VLM raw response: {results}")
            return self._post_process_response(results)
            
        except Exception as e:
            logger.error(f"Error in response generation: {e}")
            raise

    def _post_process_response(self, raw_response: str) -> str:
        """
        Post-process the VLM response to extract only the assistant's answer.
        
        Args:
            raw_response (str): Raw response from the VLM model
            
        Returns:
            str: Clean assistant response without user prompt
        """
        try:
            # The VLM typically returns format like:
            # "User:\n\n\n\n<prompt>\nAssistant: <actual_response>"
            # We want to extract only the part after "Assistant:"
            
            # Find the last occurrence of "Assistant:" (case-insensitive)
            import re
            
            # Look for various patterns where assistant response starts
            patterns = [
                r'Assistant:\s*(.*?)(?:\n\n|$)',  # "Assistant: <response>"
                r'assistant:\s*(.*?)(?:\n\n|$)',  # "assistant: <response>"
                r'ASSISTANT:\s*(.*?)(?:\n\n|$)',  # "ASSISTANT: <response>"
            ]
            
            for pattern in patterns:
                match = re.search(pattern, raw_response, re.DOTALL | re.IGNORECASE)
                if match:
                    assistant_response = match.group(1).strip()
                    if assistant_response:  # Make sure we got something
                        return assistant_response
            
            # Fallback: If no "Assistant:" pattern found, try to split by common separators
            # and take the last meaningful part
            lines = raw_response.split('\n')
            
            # Remove empty lines and find content after user prompt
            meaningful_lines = [line.strip() for line in lines if line.strip()]
            
            if len(meaningful_lines) > 0:
                # Try to find where the actual response starts
                # Look for lines that don't look like prompts or system messages
                for i, line in enumerate(meaningful_lines):
                    # Skip lines that look like user prompts or system messages
                    if (not line.lower().startswith(('user:', 'prompt:', 'describe', 'analyze', 'what', 'how', 'where', 'when', 'why'))
                        and len(line) > 10):  # Reasonable length for a response
                        # Return everything from this point onwards
                        return ' '.join(meaningful_lines[i:])
                
                # If all else fails, return the last substantial line
                for line in reversed(meaningful_lines):
                    if len(line) > 20:  # Substantial content
                        return line
            
            # Last resort: return the original response cleaned up
            return raw_response.strip()
            
        except Exception as e:
            logger.warning(f"Error in post-processing response: {e}")
            # If post-processing fails, return the original response
            return raw_response.strip()

    def process_image(self, image_source: Union[str, Image.Image], prompt: str) -> str:
        """
        Processes a single image and returns a description.

        Args:
            image_source (Union[str, Image.Image]): Path/URL to the image or a PIL Image object.
            prompt (str): The prompt to ask the model about the image.

        Returns:
            str: The model's response.
        """
        logger.info(
            f"Processing image: {image_source if isinstance(image_source, str) else 'PIL Image'}"
        )

        try:
            # Handle different image source types and create conversation
            if isinstance(image_source, str):
                # For URL or file path, load the image
                if image_source.startswith(('http://', 'https://')):
                    from transformers.image_utils import load_image
                    image = load_image(image_source)
                else:
                    image = Image.open(image_source)
                
                # Create conversation with loaded image
                content = [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            elif isinstance(image_source, Image.Image):
                # Create conversation with PIL Image directly
                content = [
                    {"type": "image", "image": image_source},
                    {"type": "text", "text": prompt}
                ]
            else:
                raise TypeError("image_source must be a string (path/URL) or a PIL.Image object.")

            conversation = [{"role": "user", "content": content}]
            
            # Use the unified _generate_response method
            return self._generate_response(conversation, max_new_tokens=128)
            
        except Exception as e:
            logger.error(f"Error in image processing: {e}")
            raise

    def process_video(self, video_path: str, prompt: str) -> str:
        """
        Processes a whole video file and returns a description.

        Args:
            video_path (str): The path to the video file.
            prompt (str): The prompt to ask the model about the video.

        Returns:
            str: The model's response.
        """
        logger.info(f"Processing video: {video_path}")
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "path": video_path},
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        return self._generate_response(conversation, max_new_tokens=256)

    def start_stream(
        self, 
        video_source: Union[int, str], 
        fps_limit: Optional[int] = 10
    ):
        """
        Start capturing frames from video source.
        
        Args:
            video_source (Union[int, str]): Video source (0 for webcam, or video file path).
            fps_limit (Optional[int]): Limit capture FPS to reduce load.
        """
        if self.is_streaming:
            logger.warning("Stream is already running")
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
        logger.info(f"Started video stream from: {video_source}")

    def stop_stream(self):
        """Stop the video stream."""
        if not self.is_streaming:
            return
            
        self.is_streaming = False
        if self.stream_thread:
            self.stream_thread.join(timeout=2.0)
        if self.capture:
            self.capture.release()
        logger.info("Video stream stopped")

    def analyze_recent_frames(
        self, 
        prompt: str, 
        n_frames: int = 10,
        min_frames: int = 3
    ) -> Optional[str]:
        """
        Analyze the most recent N frames from the stream.
        
        Args:
            prompt (str): The question to ask about the frames.
            n_frames (int): Number of recent frames to analyze.
            min_frames (int): Minimum number of frames required for analysis.
            
        Returns:
            Optional[str]: Analysis result or None if insufficient frames.
        """
        if len(self.frame_buffer) < min_frames:
            logger.warning(f"Not enough frames in buffer: {len(self.frame_buffer)}/{min_frames}")
            return None
            
        frames = self._extract_frames_from_buffer(n_frames)
        if not frames:
            return None
            
        logger.info(f"Analyzing {len(frames)} recent frames")
        
        # For multi-frame analysis, we'll use the first frame as primary and mention multiple frames in prompt
        # This is a workaround since the model may have issues with multiple images
        try:
            if len(frames) == 1:
                # Single frame analysis
                content = [
                    {"type": "image", "image": frames[0]},
                    {"type": "text", "text": prompt}
                ]
                conversation = [{"role": "user", "content": content}]
                
                # Use the unified _generate_response method
                return self._generate_response(conversation, max_new_tokens=256)
            else:
                # Multiple frames - use the most recent frame with modified prompt
                enhanced_prompt = f"Analyze this frame from a video sequence (frame {n_frames} of {n_frames} recent frames). {prompt}"
                content = [
                    {"type": "image", "image": frames[-1]},
                    {"type": "text", "text": enhanced_prompt}
                ]
                conversation = [{"role": "user", "content": content}]
                
                # Use the unified _generate_response method
                return self._generate_response(conversation, max_new_tokens=256)
                
        except Exception as e:
            logger.error(f"Error in frame analysis: {e}")
            return None

    def process_video_stream(
        self,
        video_source: Union[int, str],
        prompt: str,
        analysis_interval: float = 2.0,
        n_frames: int = 8,
        fps_limit: int = 10,
        callback: Optional[Callable[[str, List[float]], None]] = None
    ):
        """
        Continuously process video stream and analyze recent frames at regular intervals.

        Args:
            video_source (Union[int, str]): Video source (0 for webcam, or video file path).
            prompt (str): The prompt to ask about each analysis.
            analysis_interval (float): Time between analyses in seconds.
            n_frames (int): Number of recent frames to analyze each time.
            fps_limit (int): Limit capture FPS to reduce processing load.
            callback (Optional[Callable]): Callback function to handle results.
        """
        logger.info(f"Starting real-time video stream analysis")
        
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
            logger.info("Stream analysis interrupted by user")
        except Exception as e:
            logger.error(f"Error in stream processing: {e}")
        finally:
            self.stop_stream()

    def get_buffer_info(self) -> dict:
        """Get information about the current frame buffer."""
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


def example_callback(result: str, timestamps: List[float]):
    """Example callback function for handling analysis results."""
    print(f"[CALLBACK] Received analysis at {time.strftime('%H:%M:%S')}")
    print(f"[CALLBACK] Analyzed frames from timestamps: {[f'{t:.2f}' for t in timestamps[-3:]]}")  # Show last 3 timestamps


def run_examples():
    """Function to run examples for the SmolVLMRealtimeProcessor."""
    sample_video_path = os.path.join(os.path.dirname(__file__), "../../samples/world_cup_example.mov")
    if not os.path.exists(sample_video_path):
        print(
            f"\nWarning: Sample video not found at '{sample_video_path}'. "
            "Video examples will fail. Please provide a valid path."
        )

    video_processor = SmolVLMRealtimeProcessor(max_frames_buffer=50)

    # --- Image Example ---
    print("\n--- Testing Image Processing ---")
    try:
        image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image_prompt = "Describe this image."
        response = video_processor.process_image(image_url, image_prompt)
        print(f"Prompt: {image_prompt}\nResponse: {response}")
    except Exception as e:
        print(f"Image processing failed: {e}")

    # --- Real-time Stream Analysis Example ---
    print("\n--- Testing Real-time Stream Analysis ---")
    try:
        stream_prompt = "What is currently happening? Describe any movement or changes."
        
        # For webcam, use: video_source = 0
        # For video file, use: video_source = sample_video_path
        video_source = sample_video_path if os.path.exists(sample_video_path) else 0
        
        print(f"Starting stream analysis for 15 seconds...")
        print("Press Ctrl+C to stop early")
        
        # Run for a limited time as example
        def limited_time_stream():
            video_processor.process_video_stream(
                video_source=video_source,
                prompt=stream_prompt,
                analysis_interval=3.0,  # Analyze every 3 seconds
                n_frames=5,  # Use 5 recent frames
                fps_limit=10,  # Capture at 10 FPS
                callback=example_callback
            )
        
        # Start stream in a thread and stop after 15 seconds
        import threading
        stream_thread = threading.Thread(target=limited_time_stream)
        stream_thread.start()
        
        # Let it run for 15 seconds then stop
        time.sleep(15)
        video_processor.stop_stream()
        stream_thread.join(timeout=2.0)
        
    except Exception as e:
        print(f"Real-time stream processing failed: {e}")
    
    # --- Buffer Info Example ---
    print("\n--- Buffer Information ---")
    buffer_info = video_processor.get_buffer_info()
    for key, value in buffer_info.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    run_examples()