#!/usr/bin/env python3
"""
Quick start guide for the Dou2DTools class.
Simple examples to get you started with image perception tasks.
"""

import os
import sys

# Add the VicLab package to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "thirdparty", "VicLab"))

from viclab.image.perception import Dou2DTools


def quick_start_example():
    """Quick start example showing basic usage."""

    # 1. Initialize the Dou2DTools
    # Make sure to set OPENAI_API_KEY environment variable
    perceptor = Dou2DTools()

    # 2. Set your image path
    image_path = "/share/project/lhy/CoT-Vision/examples/samples/000000001000.jpeg"  # Replace with your image path

    print("=== Dou2DTools Quick Start ===\n")

    # 3. Phrase Grounding - Find specific objects
    print("1. Phrase Grounding:")
    result = perceptor.phrase_grounding(image_path, "person")
    print(f"   Found {len(result['bboxes'])} person(s)")

    # 4. Object Detection - Find all objects
    print("\n2. Object Detection:")
    result = perceptor.object_detection(
        image_path, categories=["person", "Tennis racket", "hat"]
    )
    print(f"   Detected {len(result['bboxes'])} objects")

    # 5. Counting - Count specific objects
    print("\n3. Object Counting:")
    result = perceptor.count_objects(image_path, "person wear red")
    print(f"   Counted {result['count']} person(s)")

    # 6. Text Spotting - Find text in image
    print("\n4. Text Spotting:")
    text_image_path = "/share/project/lhy/CoT-Vision/examples/samples/ocr.png"
    result = perceptor.text_spotting(text_image_path)
    print(f"   Found {len(result['polygons'])} text region(s)")
    if result["texts"]:
        print("   Detected texts:", result["texts"][:3])  # Show first 3 texts

        # Visualize text detection with labels
        vis_image = perceptor.visualize_text(
            text_image_path,
            result["polygons"],
            result["texts"],
            output_path="text_spotting_result.jpg",
        )
        print("   Text spotting visualization saved to: text_spotting_result.jpg")

    # 7. Visualize results (saves to file)
    phrase_result = perceptor.phrase_grounding(image_path, "person")
    # print(phrase_result)

    if phrase_result["bboxes"].any():
        vis_image = perceptor.visualize(
            image_path,
            bboxes=phrase_result["bboxes"],
            labels=["person"] * len(phrase_result["bboxes"]),
            output_path="quick_start_result.jpg",
        )
        print("\n5. Visualization saved to: quick_start_result.jpg")

    # 8. Visualize object detection results
    object_result = perceptor.object_detection(
        image_path, categories=["Tennis racket", "hat"]
    )
    if object_result["bboxes"].any():
        vis_image = perceptor.visualize(
            image_path,
            bboxes=object_result["bboxes"],
            labels=object_result["categories"],
            output_path="object_detection_result.jpg",
        )
        print(
            "\n6. Object detection visualization saved to: object_detection_result.jpg"
        )

    # 9. Open-ended detection
    print("\n7. Open-ended detection:")
    result = perceptor.open_ended_detection(image_path)
    print(f"   Detected {len(result['bboxes'])} objects")
    if result["bboxes"].any():
        vis_image = perceptor.visualize(
            image_path,
            bboxes=result["bboxes"],
            labels=result["categories"],
            output_path="open_ended_detection_result.jpg",
        )
        print(
            "\n7. Open-ended detection visualization saved to: open_ended_detection_result.jpg"
        )

    # 10. Visual Prompt Grounding
    print("\n8. Visual Prompt Grounding:")
    blueberry_path = "/share/project/lhy/CoT-Vision/examples/samples/blueberry.png"
    example_path = "/share/project/lhy/CoT-Vision/examples/samples/exemplar.png"
    result = perceptor.visual_prompt_grounding(
        exemplar_path=example_path, image_path=blueberry_path
    )
    print(f"   Detected {len(result['bboxes'])} objects")
    if result["bboxes"].any():
        vis_image = perceptor.visualize(
            blueberry_path,
            bboxes=result["bboxes"],
            output_path="visual_prompt_grounding_result.jpg",
        )
        print(
            "\n8. Visual Prompt Grounding visualization saved to: visual_prompt_grounding_result.jpg"
        )

    print("\n=== Quick Start Complete! ===")


def simple_phrase_grounding():
    """Simple phrase grounding example."""
    perceptor = Dou2DTools()

    # Replace with your image and what you want to find
    image_path = "your_image.jpg"
    phrase = "red car"

    result = perceptor.phrase_grounding(image_path, phrase)

    print(f"Looking for: {phrase}")
    print(f"Found {len(result['bboxes'])} instances")

    # Show bounding box coordinates
    for i, bbox in enumerate(result["bboxes"]):
        x1, y1, x2, y2 = bbox
        print(f"  {phrase} {i + 1}: ({x1}, {y1}) to ({x2}, {y2})")

    return result


def simple_counting():
    """Simple counting example."""
    perceptor = Dou2DTools()

    image_path = "your_image.jpg"
    object_type = "people"

    result = perceptor.count_objects(image_path, object_type)

    print(f"How many {object_type}? {result['count']}")

    return result


def error_handling_example():
    """Example showing error handling."""
    try:
        perceptor = Dou2DTools()

        # This will fail if image doesn't exist
        result = perceptor.phrase_grounding("nonexistent.jpg", "car")
        print("This shouldn't print if file doesn't exist")

    except FileNotFoundError:
        print("Error: Image file not found!")
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")


def simple_text_spotting():
    """Simple text spotting example showing both text and polygon detection."""
    perceptor = Dou2DTools()

    image_path = "your_text_image.jpg"  # Replace with an image containing text

    result = perceptor.text_spotting(image_path)

    print(f"Text Spotting Results:")
    print(f"Found {len(result['text_detections'])} text regions")
    print(f"Detected {len(result['texts'])} text strings")

    # Show detected texts
    for i, text in enumerate(result["texts"]):
        if text.strip():
            print(f"  Text {i + 1}: '{text.strip()}'")

    # Show polygon coordinates for first detection
    if result["text_detections"]:
        polygon = result["text_detections"][0]
        print(f"  First polygon coordinates: {polygon}")

    return result


if __name__ == "__main__":
    # Check if API key is set
    # if not os.environ.get("OPENAI_API_KEY"):
    #     print("Please set the OPENAI_API_KEY environment variable:")
    #     print("export OPENAI_API_KEY='your-api-key'")
    #     sys.exit(1)

    # Run the quick start example
    quick_start_example()

    print("\n" + "=" * 50)
    print("More examples:")
    print("- simple_phrase_grounding()")
    print("- simple_counting()")
    print("- error_handling_example()")
    print("- simple_text_spotting()")
