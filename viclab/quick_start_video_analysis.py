#!/usr/bin/env python3
"""
Quick start guide for the SmolVLMRealtimeProcessor class (with real-time video buffer).
Demonstrates image, video, and real-time stream analysis using the new buffer features.
"""

import os
import sys
import time

# Add the VicLab package to the path if needed
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "viclab"))

from viclab.video.realtime_video import SmolVLMRealtimeProcessor

def quick_start_example():
    """Quick start example showing basic usage of SmolVLMRealtimeProcessor with real-time buffer."""

    print("=== SmolVLM Video Analysis Quick Start (Realtime Buffer) ===\n")

    # 1. Initialize the SmolVLMRealtimeProcessor
    print("1. Initializing SmolVLMRealtimeProcessor...")
    try:
        processor = SmolVLMRealtimeProcessor()
        print("   Initialization complete.\n")
    except Exception as e:
        print(f"Error initializing processor: {e}")
        return

    # 2. Process a single image
    print("2. Processing a single image...")
    try:
        image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image_prompt = "Describe what you see in this image."
        response = processor.process_image(image_url, image_prompt)
        print(f"   Prompt: {image_prompt}")
        print(f"   Response: {response}\n")
    except Exception as e:
        print(f"   Image processing failed: {e}\n")

    # 3. Process a video file
    print("3. Processing a video file...")
    try:
        # Replace with your video path
        video_path = os.path.join(os.path.dirname(__file__), "samples", "world_cup_example.mov")
        if not os.path.exists(video_path):
            print(f"   Warning: Sample video not found at {video_path}")
            print("   Please provide a valid video path to test this feature.\n")
        else:
            video_prompt = "Describe the main events in this video."
            response = processor.process_video(video_path, video_prompt)
            print(f"   Prompt: {video_prompt}")
            print(f"   Response: {response}\n")
    except Exception as e:
        print(f"   Video processing failed: {e}\n")

    # 4. Real-time video stream analysis with buffer
    print("4. Real-time video stream analysis (with buffer)...")
    try:
        # Use webcam (0) or a video file as stream source
        stream_source = video_path if os.path.exists(video_path) else 0
        stream_prompt = "What is happening in this clip?"
        print(f"   Using {'video file' if isinstance(stream_source, str) else 'webcam'} as stream source")
        print("   Starting real-time stream analysis for 10 seconds...\n")

        # Define a simple callback to print results
        def print_callback(result, timestamps):
            print(f"[Callback] Analysis result: {result}")
            print(f"[Callback] Frame timestamps: {[f'{t:.2f}' for t in timestamps]}")

        # Start stream analysis in a thread and stop after 10 seconds
        import threading
        def run_stream():
            processor.process_video_stream(
                video_source=stream_source,
                prompt=stream_prompt,
                analysis_interval=3.0,  # Analyze every 3 seconds
                n_frames=5,             # Use 5 recent frames
                fps_limit=10,           # Capture at 10 FPS
                callback=print_callback
            )

        stream_thread = threading.Thread(target=run_stream)
        stream_thread.start()
        time.sleep(10)
        processor.stop_stream()
        stream_thread.join(timeout=2.0)
        print("   Real-time stream analysis stopped.\n")
    except Exception as e:
        print(f"   Stream processing failed: {e}\n")

    # 5. Buffer info
    print("5. Buffer Information:")
    buffer_info = processor.get_buffer_info()
    for key, value in buffer_info.items():
        print(f"   {key}: {value}")

    print("\n=== Quick Start Complete! ===")


def simple_image_analysis():
    """Simple example for image analysis."""
    processor = SmolVLMRealtimeProcessor()
    image_source = "your_image.jpg"  # Replace with your image path or URL
    prompt = "What objects do you see in this image?"
    response = processor.process_image(image_source, prompt)
    print(f"Prompt: {prompt}")
    print(f"Response: {response}")
    return response


def simple_video_analysis():
    """Simple example for video analysis."""
    processor = SmolVLMRealtimeProcessor()
    video_path = "your_video.mp4"  # Replace with your video path
    prompt = "Describe the main events in this video."
    response = processor.process_video(video_path, prompt)
    print(f"Prompt: {prompt}")
    print(f"Response: {response}")
    return response


def error_handling_example():
    """Example showing error handling."""
    try:
        processor = SmolVLMRealtimeProcessor()
        # This will fail if image doesn't exist
        result = processor.process_image("nonexistent.jpg", "Describe this image")
        print("This shouldn't print if file doesn't exist")
    except FileNotFoundError:
        print("Error: Image file not found!")
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    # Run the quick start example
    quick_start_example()

    print("\n" + "=" * 50)
    print("More examples:")
    print("- simple_image_analysis()")
    print("- simple_video_analysis()")
    print("- error_handling_example()") 