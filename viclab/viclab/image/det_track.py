import os
from typing import List, Optional, Tuple, Union

import cv2
import imageio
import numpy as np
import supervision as sv
import torch
from tqdm import tqdm
from ultralytics import YOLO

# NOTE: The visualization utilities below are adapted from viclab/image/det_seg.py
# to ensure a consistent visualization style across the project.
BOX_ANNOTATOR = sv.BoxAnnotator(thickness=2)


class LabelAnnotator(sv.LabelAnnotator):
    """Custom label annotator for better text positioning."""

    @staticmethod
    def resolve_text_background_xyxy(
        center_coordinates,
        text_wh,
        position,
    ):
        center_x, center_y = center_coordinates
        text_w, text_h = text_wh
        return center_x, center_y, center_x + text_w, center_y + text_h


LABEL_ANNOTATOR = LabelAnnotator(text_padding=4, text_scale=0.5, text_thickness=1)

class YOLOv11Detector:
    """
    A class for object detection and tracking using YOLO models from Ultralytics.

    Note: While the original class name mentioned YOLOv11, this class utilizes
    the 'ultralytics' library, which supports YOLOv8 and other models. The
    default model is 'yolov8n.pt'.
    """

    def __init__(self, model_path: str = "yolov8n.pt"):
        """
        Initializes the YOLO detector.

        Args:
            model_path (str): Path to the YOLO model file (e.g., 'yolov8n.pt').
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        self.model = YOLO(model_path)
        self.model.to(self.device)

    def detect_image(
        self, image_source: Union[str, np.ndarray]
    ) -> Tuple[sv.Detections, np.ndarray]:
        """
        Performs object detection on a single image.

        Args:
            image_source (Union[str, np.ndarray]): Path to the image or the image as a numpy array.

        Returns:
            A tuple containing:
                - sv.Detections: The detected objects.
                - np.ndarray: The annotated image.
        """
        if isinstance(image_source, str):
            image = cv2.imread(image_source)
            if image is None:
                raise FileNotFoundError(f"Image not found at {image_source}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = image_source.copy()

        results = self.model(image)[0]
        detections = sv.Detections.from_ultralytics(results)

        labels = [
            f"{self.model.names[class_id]} {confidence:.2f}"
            for _, _, confidence, class_id, _, _ in detections
        ]

        annotated_image = image.copy()
        annotated_image = BOX_ANNOTATOR.annotate(
            scene=annotated_image, detections=detections
        )
        annotated_image = LABEL_ANNOTATOR.annotate(
            scene=annotated_image, detections=detections, labels=labels
        )

        return detections, annotated_image

    def process_video(self, video_path: str, output_path: str, track: bool = True):
        """
        Processes a video from a file path for object detection or tracking.

        Args:
            video_path (str): Path to the input video file.
            output_path (str): Path to save the processed video file.
            track (bool): If True, performs object tracking. Otherwise, performs detection.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error opening video file: {video_path}")
            return

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        writer = imageio.get_writer(output_path, fps=fps)

        if track:
            results_generator = self.model.track(source=video_path, stream=True, verbose=False)
        else:
            results_generator = self.model(video_path, stream=True, verbose=False)

        print("Processing video from file...")
        for result in tqdm(results_generator, total=frame_count, desc="Processing video"):
            annotated_frame = self._annotate_frame(result, track)
            writer.append_data(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))

        writer.close()
        print(f"Processed video saved to: {output_path}")

    def process_image_stream(
        self,
        image_stream: List[np.ndarray],
        output_path: str,
        save_frames: bool = False,
        fps: int = 30,
        track: bool = True,
    ):
        """
        Processes a video stream (list of image arrays) for detection or tracking.

        Args:
            image_stream (List[np.ndarray]): A list of images (as NumPy arrays in BGR format)
                                              representing the video stream.
            output_path (str): Path to save the processed video file.
            fps (int): Frames per second for the output video.
            track (bool): If True, performs object tracking. Otherwise, performs detection.
        """
        if not image_stream:
            print("Error: The image stream is empty.")
            return

        if save_frames:
            writer = imageio.get_writer(output_path, fps=fps)
        else:
            writer = None

        frame_count = len(image_stream)

        # The model can directly take an iterable of numpy arrays
        if track:
            results_generator = self.model.track(source=image_stream, stream=True, verbose=False)
        else:
            results_generator = self.model(image_stream, stream=True, verbose=False)

        print("Processing image stream...")
        for result in tqdm(results_generator, total=frame_count, desc="Processing stream"):
            annotated_frame = self._annotate_frame(result, track)
            if save_frames:
                writer.append_data(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))

        if save_frames:
            writer.close()
        print(f"Processed stream saved to: {output_path}")

    def _annotate_frame(self, result, track: bool) -> np.ndarray:
        """Helper function to annotate a single frame."""
        frame = result.orig_img  # BGR format
        detections = sv.Detections.from_ultralytics(result)

        if track and hasattr(result.boxes, "id") and result.boxes.id is not None:
            detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)

        labels = []
        for _, _, confidence, class_id, tracker_id, _ in detections:
            label = ""
            if track and tracker_id is not None:
                label += f"#{tracker_id} "
            label += f"{self.model.names[class_id]} {confidence:.2f}"
            labels.append(label)

        annotated_frame = frame.copy()
        annotated_frame = BOX_ANNOTATOR.annotate(
            scene=annotated_frame, detections=detections
        )
        annotated_frame = LABEL_ANNOTATOR.annotate(
            scene=annotated_frame, detections=detections, labels=labels
        )
        return annotated_frame

    @staticmethod
    def display_image(image: np.ndarray, window_name: str = "Image") -> None:
        """Display image in a window."""
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow(window_name, image_bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# --- Example Usage ---
if __name__ == '__main__':
    # Initialize the detector
    detector = YOLOv11Detector("yolov8n.pt")

    # 1. Example with a single image
    # Create a dummy black image for demonstration
    dummy_image_bgr = np.zeros((640, 480, 3), dtype=np.uint8)
    cv2.putText(dummy_image_bgr, "Image Test", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    dummy_image_rgb = cv2.cvtColor(dummy_image_bgr, cv2.COLOR_BGR2RGB)

    _, annotated_img = detector.detect_image(dummy_image_rgb)
    # detector.display_image(annotated_img, "Single Image Detection")


    # 2. Example with an image stream (list of numpy arrays)
    print("\n--- Testing Image Stream Processing ---")
    # Create a dummy video stream (list of 100 black frames)
    video_stream = []
    for i in range(100):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Add a moving element to simulate a video
        cv2.circle(frame, (50 + i*3, 240), 20, (0, 0, 255), -1)
        cv2.putText(frame, f"Frame: {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        video_stream.append(frame)

    # Process the stream and save it as a video
    detector.process_image_stream(
        image_stream=video_stream,
        output_path="output_stream.mp4",
        save_frames=False,
        fps=25,
        track=True
    )