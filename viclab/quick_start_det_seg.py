#!/usr/bin/env python3
"""
Quick start guide for the OwlV2SAM class.
Simple examples for combined object detection and segmentation.
"""

import os
import sys
import numpy as np

# Add the viclab package to the path
# This assumes the script is run from the 'viclab' directory
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "thirdparty", "viclab"))

from viclab.image.det_seg import OwlV2SAM


def quick_start_example():
    """Quick start example showing basic usage of OwlV2SAM."""

    # --- Setup ---
    # Define paths relative to this script's location
    base_dir = os.path.dirname(os.path.abspath(__file__))
    sam_checkpoint = os.path.join(base_dir, "viclab", "image", "checkpoints", "sam_vit_h_4b8939.pth")
    image_path = "/share/project/lhy/CoT-Vision/examples/samples/000000001000.jpeg"  # Replace with your image path
    output_dir = os.path.join(base_dir, "det_seg_results")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    print("=== OwlV2SAM Quick Start ===\n")

    # 1. Initialize the OwlV2SAM detector
    print("1. Initializing OwlV2SAM...")
    if not os.path.exists(sam_checkpoint):
        print(f"Error: SAM checkpoint not found at {sam_checkpoint}")
        print("Please ensure the checkpoint file is correctly placed.")
        return
        
    try:
        detector = OwlV2SAM(sam_checkpoint=sam_checkpoint)
    except Exception as e:
        print(f"Error initializing models: {e}")
        return
    print("   Initialization complete.\n")

    if not os.path.exists(image_path):
        print(f"Error: Example image not found at {image_path}")
        return

    # 2. Detect and Segment objects with text prompts
    print("2. Running Detection and Segmentation...")
    prompts = ["person", "tennis racket"]
    detect_results = detector.detect_and_segment(
        image=image_path,
        text_prompts=prompts,
        detection_threshold=0.1,
        return_all_detections=True,
    )
    
    if detect_results["detected"]:
        print(f"   Detected {len(detect_results['detections'])} objects.")
        detector.visualize(
            results=detect_results,
            save_path=output_dir,
            save_name="detect_and_segment_result.jpg"
        )
        print(f"   Visualization saved to: {os.path.join(output_dir, 'detect_and_segment_result.jpg')}\n")
    else:
        print("   No objects detected with the given prompts.\n")


    # 3. Segment with a point prompt
    print("3. Running Segmentation with a point prompt...")
    # Use the center of the first detected box as a point prompt
    if detect_results["detected"] and len(detect_results["detections"]) > 0:
        first_box = detect_results["detections"][0]['box']
        point_coord = np.array([[(first_box[0] + first_box[2]) / 2, (first_box[1] + first_box[3]) / 2]])
        point_label = np.array([1]) # 1 for foreground

        point_results = detector.segment_with_points(
            image=image_path,
            points=point_coord,
            point_labels=point_label,
        )
        print(f"   Generated {len(point_results['masks'])} masks from the point.")
        detector.visualize(
            results=point_results,
            save_path=output_dir,
            save_name="segment_with_point_result.jpg"
        )
        print(f"   Visualization saved to: {os.path.join(output_dir, 'segment_with_point_result.jpg')}\n")
    else:
        print("   Skipping point segmentation because no objects were detected first.\n")

    # 4. Segment with a bounding box prompt
    print("4. Running Segmentation with a bounding box prompt...")
    # Use the first detected box as a bbox prompt
    if detect_results["detected"] and len(detect_results["detections"]) > 0:
        first_box = detect_results["detections"][0]['box']
        
        box_results = detector.segment_with_bbox(
            image=image_path,
            bbox=first_box,
        )
        print(f"   Generated {len(box_results['masks'])} masks from the bounding box.")
        detector.visualize(
            results=box_results,
            show_box=True, # Explicitly show the input box
            save_path=output_dir,
            save_name="segment_with_box_result.jpg"
        )
        print(f"   Visualization saved to: {os.path.join(output_dir, 'segment_with_box_result.jpg')}\n")
    else:
        print("   Skipping bbox segmentation because no objects were detected first.\n")

    print("=== Quick Start Complete! ===")

if __name__ == "__main__":
    # Add parent directory to path to find the 'viclab' package
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    quick_start_example() 