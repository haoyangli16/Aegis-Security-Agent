import os

import cv2
import numpy as np
import torch
from PIL import Image
import supervision as sv
from segment_anything import SamPredictor, sam_model_registry
from transformers import Owlv2ForObjectDetection, Owlv2Processor
from typing import Dict, List, Tuple, Optional, Union
import transformers

BOX_ANNOTATOR = sv.BoxAnnotator(thickness=2)
POINT_ANNOTATOR = sv.DotAnnotator(radius=6)

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

def draw_boxes_points_with_labels(
    image: np.ndarray,
    boxes: Optional[np.ndarray] = None,
    points: Optional[np.ndarray] = None,
    labels: Optional[List[str]] = None,
    output_path: Optional[str] = None,
) -> np.ndarray:
    annotated_image = image.copy()
    detections = None
    # Draw boxes
    if boxes is not None and len(boxes) > 0:
        boxes = np.array(boxes)
        detections = sv.Detections(
            xyxy=boxes,
            class_id=np.arange(len(boxes)),
            confidence=np.ones(len(boxes))
        )
        annotated_image = BOX_ANNOTATOR.annotate(annotated_image, detections)
    # Draw points
    if points is not None and len(points) > 0:
        points = np.array(points)
        points_xyxy = np.concatenate([points, points], axis=1)
        detections = sv.Detections(
            xyxy=points_xyxy,
            class_id=np.arange(len(points)),
            confidence=np.ones(len(points))
        )
        annotated_image = POINT_ANNOTATOR.annotate(annotated_image, detections)
    # Draw labels
    if labels is not None and detections is not None:
        annotated_image = LABEL_ANNOTATOR.annotate(
            annotated_image, detections, labels=labels
        )
    if output_path:
        cv2.imwrite(
            output_path,
            cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
        )
    return annotated_image

class OwlV2SAM:
    def __init__(
        self,
        sam_checkpoint: str = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        sam_predictor: SamPredictor = None,
    ):
        """Initialize the detector with OWLv2 and SAM models.

        Args:
            sam_checkpoint: Path to SAM model checkpoint (ignored if sam_predictor is provided)
            device: Device to run models on ("cuda" or "cpu")
            sam_predictor: Existing SAM predictor instance
        """
        self.device = device

        # Use provided SAM predictor or initialize a new one
        if sam_predictor is not None:
            self.sam_predictor = sam_predictor
            self.sam = self.sam_predictor.model
        else:
            # Initialize SAM
            self.sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
            self.sam.to(device=self.device)
            self.sam_predictor = SamPredictor(self.sam)

        # Initialize OWL-ViT
        print("Loading OWLv2 model...")
        owlv2_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints/models--google--owlv2-base-patch16-ensemble")
        if not os.path.exists(owlv2_path):
            self.owlv2_processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
            self.owlv2_model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")
        else:
            self.owlv2_processor = Owlv2Processor.from_pretrained(owlv2_path)
            self.owlv2_model = Owlv2ForObjectDetection.from_pretrained(owlv2_path)

        self.owlv2_model.to(self.device)
        print("Models loaded successfully!")

    def detect_and_segment(
        self,
        image: Union[np.ndarray, str],
        text_prompts: List[str],
        detection_threshold: float = 0.1,
        return_all_detections: bool = False,
    ) -> Dict:
        """Detect objects with OWLv2 and segment with SAM.

        Args:
            image: Input image (numpy array in RGB format) or path to image
            text_prompts: List of text prompts for detection
            detection_threshold: Confidence threshold for detection
            return_all_detections: If True, return all detections; otherwise, return best detection

        Returns:
            Dictionary containing detection and segmentation results

            "detected": True if objects are detected, False otherwise
            "detections": List of detections if return_all_detections is True, otherwise None
            "box": Bounding box of the detected object if return_all_detections is False, otherwise None, [x1, y1, x2, y2]
            "score": Score of the detected object
            "label": Label of the detected object
            "text_prompt": Text prompt of the detected object
            "mask": Mask of the detected object, shape: (H, W)
            "image": Original image
            "text_prompts": Text prompts used for detection
        """
        # Load image if path is provided
        if isinstance(image, str):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        pil_image = Image.fromarray(image)

        # Detect with OWLv2
        inputs = self.owlv2_processor(
            text=text_prompts, images=pil_image, return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.owlv2_model(**inputs)

        target_sizes = torch.Tensor([pil_image.size[::-1]]).to(self.device)
        """
        NOTE(Haoyang): transformers version (≥4.37), the method post_process_grounded_object_detection has been deprecated or removed
        - post_process_object_detection works with text + image
        - post_process_image_guided_detection works with query_image + image
        """
        # if transformers version >= 4.37, use post_process_object_detection
        if transformers.__version__ >= "4.37":
            results = self.owlv2_processor.post_process_object_detection(
                outputs=outputs, target_sizes=target_sizes, threshold=detection_threshold
            )[0]
        else:
            results = self.owlv2_processor.post_process_grounded_object_detection(
                outputs=outputs, target_sizes=target_sizes, threshold=detection_threshold
            )[0]

        if len(results["boxes"]) == 0:
            return {
                "detected": False,
                "message": "No objects detected",
                "image": image,
                "text_prompts": text_prompts,
            }

        # Prepare SAM for segmentation
        self.sam_predictor.set_image(image)

        if return_all_detections:
            # Process all detections
            detections = []
            for i, (box, score, label) in enumerate(
                zip(results["boxes"], results["scores"], results["labels"])
            ):
                box_np = box.detach().cpu().numpy()
                score_np = score.item()
                label_idx = label.item()
                text_prompt = text_prompts[label_idx % len(text_prompts)]

                # Generate mask with SAM
                masks, _, _ = self.sam_predictor.predict(
                    box=box_np, multimask_output=False
                )

                detections.append({
                    "box": box_np,
                    "score": score_np,
                    "label": label_idx,
                    "text_prompt": text_prompt,
                    "mask": masks[0],
                })

            return {
                "detected": True,
                "detections": detections,
                "image": image,
                "text_prompts": text_prompts,
            }
        else:
            # Get best detection
            best_idx = torch.argmax(results["scores"])
            best_box = results["boxes"][best_idx].detach().cpu().numpy()
            best_score = results["scores"][best_idx].item()
            best_label = results["labels"][best_idx].item()
            text_prompt = text_prompts[best_label % len(text_prompts)]

            # Generate mask with SAM
            masks, _, _ = self.sam_predictor.predict(
                box=best_box, multimask_output=False
            )

            return {
                "detected": True,
                "box": best_box,
                "score": best_score,
                "label": best_label,
                "text_prompt": text_prompt,
                "mask": masks[0],
                "image": image,
                "text_prompts": text_prompts,
            }

    def segment_with_points(
        self,
        image: Union[np.ndarray, str],
        points: List[List[int]],  # List of [x, y] coordinates
        point_labels: List[int],  # 1 for foreground, 0 for background
        multimask_output: bool = True,
    ) -> Dict:
        """Segment objects with SAM using point prompts.

        Args:
            image: Input image (numpy array in RGB format) or path to image
            points: List of point coordinates [[x1, y1], [x2, y2], ...]
            point_labels: List of point labels (1 for foreground, 0 for background)
            multimask_output: If True, return multiple masks; otherwise, return single best mask

        Returns:
            Dictionary containing segmentation results

            "masks": Array of predicted masks, shape (N, H, W)
            "scores": Array of mask confidence scores
            "points": Input points
            "point_labels": Input point labels
            "image": Original image
        """
        # Load image if path is provided
        if isinstance(image, str):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert points and labels to numpy arrays if they aren't already
        points_np = np.array(points)
        point_labels_np = np.array(point_labels)

        # Prepare SAM for segmentation
        self.sam_predictor.set_image(image)

        # Generate masks with SAM
        masks, scores, logits = self.sam_predictor.predict(
            point_coords=points_np,
            point_labels=point_labels_np,
            multimask_output=multimask_output,
        )

        return {
            "masks": masks,
            "scores": scores,
            "logits": logits,
            "points": points_np,
            "point_labels": point_labels_np,
            "image": image,
        }

    def segment_with_bbox(
        self,
        image: Union[np.ndarray, str],
        bbox: List[int],  # [x1, y1, x2, y2]
        multimask_output: bool = True,
    ) -> Dict:
        """Segment objects with SAM using a bounding box prompt.

        Args:
            image: Input image (numpy array in RGB format) or path to image
            bbox: Bounding box in format [x1, y1, x2, y2]
            multimask_output: If True, return multiple masks; otherwise, return single best mask

        Returns:
            Dictionary containing segmentation results

            "masks": Array of predicted masks, shape (N, H, W)
            "scores": Array of mask confidence scores
            "logits": Low resolution logits
            "box": Input bounding box
            "image": Original image
        """
        # Load image if path is provided
        if isinstance(image, str):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert bbox to numpy array if it isn't already
        bbox_np = np.array(bbox)

        # Prepare SAM for segmentation
        self.sam_predictor.set_image(image)

        # Generate masks with SAM
        masks, scores, logits = self.sam_predictor.predict(
            box=bbox_np,
            multimask_output=multimask_output,
        )

        return {
            "masks": masks,
            "scores": scores,
            "logits": logits,
            "box": bbox_np,
            "image": image,
        }

    def segment(
        self,
        image: Union[np.ndarray, str],
        point_coords: Optional[List[List[int]]] = None,  # List of [x, y] coordinates
        point_labels: Optional[List[int]] = None,  # 1 for foreground, 0 for background
        bbox: Optional[List[int]] = None,  # [x1, y1, x2, y2]
        mask_input: Optional[np.ndarray] = None,  # Previous mask prediction
        multimask_output: bool = True,
    ) -> Dict:
        """Segment objects with SAM using multiple types of prompts.

        Args:
            image: Input image (numpy array in RGB format) or path to image
            point_coords: List of point coordinates [[x1, y1], [x2, y2], ...]
            point_labels: List of point labels (1 for foreground, 0 for background)
            bbox: Bounding box in format [x1, y1, x2, y2]
            mask_input: A low resolution mask input, typically from a previous prediction
            multimask_output: If True, return multiple masks; otherwise, return single best mask

        Returns:
            Dictionary containing segmentation results

            "masks": Array of predicted masks, shape (N, H, W)
            "scores": Array of mask confidence scores
            "logits": Low resolution logits
            "points": Input points (if provided)
            "point_labels": Input point labels (if provided)
            "box": Input bounding box (if provided)
            "image": Original image
        """
        # Load image if path is provided
        if isinstance(image, str):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert inputs to numpy arrays
        point_coords_np = np.array(point_coords) if point_coords is not None else None
        point_labels_np = np.array(point_labels) if point_labels is not None else None
        bbox_np = np.array(bbox) if bbox is not None else None

        # Prepare SAM for segmentation
        self.sam_predictor.set_image(image)

        # Generate masks with SAM
        masks, scores, logits = self.sam_predictor.predict(
            point_coords=point_coords_np,
            point_labels=point_labels_np,
            box=bbox_np,
            mask_input=mask_input,
            multimask_output=multimask_output,
        )

        # Prepare result dictionary
        result = {
            "masks": masks,
            "scores": scores,
            "logits": logits,
            "image": image,
        }

        # Add optional inputs to result
        if point_coords is not None:
            result["points"] = point_coords_np
        if point_labels is not None:
            result["point_labels"] = point_labels_np
        if bbox is not None:
            result["box"] = bbox_np

        return result

    def visualize(
        self,
        results: Dict,
        show_box: bool = True,
        show_mask: bool = True,
        show_points: bool = True,
        mask_idx: int = None,
        save_path: Optional[str] = None,
        save_name: Optional[str] = "det_seg.png",
    ) -> np.ndarray:
        """Visualize detection and segmentation results using supervision."""
        is_detection = "detected" in results

        if is_detection and not results["detected"]:
            print(results.get("message", "No objects detected"))
            return results["image"]

        vis_image = results["image"].copy()

        # 1. Apply masks
        if show_mask:
            # mask_color = np.array([0, 255, 0], dtype=np.uint8)  # Green
            # random color
            mask_color = np.random.randint(0, 255, 3).astype(np.uint8)
            masks_to_draw = []
            if is_detection:
                if "detections" in results:
                    masks_to_draw = [d["mask"] for d in results["detections"] if "mask" in d]
                elif "mask" in results:
                    masks_to_draw = [results["mask"]]
            elif "masks" in results:
                masks = results["masks"]
                if mask_idx is not None and 0 <= mask_idx < len(masks):
                    masks_to_draw = [masks[mask_idx]]
                elif mask_idx is None and len(masks) > 0:
                    scores = results.get("scores", [])
                    best_idx = np.argmax(scores) if len(scores) > 0 else 0
                    if 0 <= best_idx < len(masks):
                         masks_to_draw = [masks[best_idx]]
            
            for mask in masks_to_draw:
                vis_image = self._apply_mask(vis_image, mask, mask_color, alpha=0.5)

        # 2. Prepare boxes, points, labels for draw_boxes_points_with_labels
        boxes_to_draw = []
        points_to_draw = []
        labels_to_draw = []
        
        if show_box:
            if is_detection:
                if "detections" in results:
                    boxes_to_draw = [d["box"] for d in results["detections"]]
                    labels_to_draw = [f"{d['text_prompt']}: {d['score']:.2f}" for d in results["detections"]]
                elif "box" in results:
                    boxes_to_draw = [results["box"]]
                    labels_to_draw = [f"{results['text_prompt']}: {results['score']:.2f}"]
            elif "box" in results:
                boxes_to_draw = [results["box"]]
            
        if show_points and "points" in results:
            points_to_draw = results["points"]
            # To prevent box labels from being drawn on points
            if len(points_to_draw) > 0:
                labels_to_draw = []

        # 3. Call unified drawing function and save
        output_path = os.path.join(save_path, save_name) if save_path else None
        vis_image = draw_boxes_points_with_labels(
            image=vis_image,
            boxes=np.array(boxes_to_draw),
            points=np.array(points_to_draw),
            labels=labels_to_draw,
            output_path=output_path
        )

        if output_path:
            print(f"Visualization saved to {output_path}")

        return vis_image

    def _apply_mask(
        self, image: np.ndarray, mask: np.ndarray, color: np.ndarray, alpha: float = 0.5
    ) -> np.ndarray:
        """Apply colored mask to image.

        Args:
            image: Input image
            mask: Binary mask
            color: Color for mask
            alpha: Transparency of mask

        Returns:
            Image with mask applied
        """
        mask = mask.astype(bool)
        colored_mask = np.zeros_like(image)
        colored_mask[mask] = color

        # Blend the mask with the image
        return cv2.addWeighted(image, 1, colored_mask, alpha, 0)

    def display_image(self, image: np.ndarray, window_name: str = "Image") -> None:
        """Display image in a window.

        Args:
            image: Image to display
            window_name: Name of the window
        """
        cv2.imshow(window_name, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# Example usage
if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(
        description="Detect and segment objects in an image"
    )
    parser.add_argument(
        "--image",
        type=str,
        default=f"{os.path.dirname(os.path.abspath(__file__))}/example/image.jpg",
        help="Path to input image",
    )
    parser.add_argument(
        "--sam_checkpoint",
        type=str,
        default=f"{os.path.dirname(os.path.abspath(__file__))}/checkpoints/sam_vit_h_4b8939.pth",
        help="Path to SAM model checkpoint",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["detect", "point"],
        default="detect",
        help="Segmentation mode: detection-based or point-based",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        nargs="+",
        default=["a photo of a cat"],
        help="Text prompts for detection mode",
    )
    parser.add_argument(
        "--points",
        type=int,
        nargs="+",
        default=[],
        help="Point coordinates for point-based mode [x1 y1 x2 y2 ...]",
    )
    parser.add_argument(
        "--point_labels",
        type=int,
        nargs="+",
        default=[],
        help="Point labels for point-based mode (1=foreground, 0=background)",
    )
    parser.add_argument(
        "--multimask",
        action="store_true",
        help="Return multiple masks for point-based segmentation",
    )
    parser.add_argument(
        "--mask_idx",
        type=int,
        default=None,
        help="Index of mask to visualize when multiple masks are available",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.3, help="Detection confidence threshold"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=f"{os.path.dirname(os.path.abspath(__file__))}/results",
        help="Path to save visualization",
    )
    parser.add_argument("--show_box", action="store_true", help="Show bounding boxes")
    parser.add_argument(
        "--show_mask", action="store_true", help="Show segmentation masks"
    )
    parser.add_argument("--show_points", action="store_true", help="Show input points")
    parser.add_argument(
        "--all_detections",
        action="store_true",
        help="Return all detections instead of best",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize the results",
    )

    args = parser.parse_args()

    # Set defaults for visualization if none specified
    if not args.show_box and not args.show_mask and not args.show_points:
        args.show_box = True
        args.show_mask = True
        args.show_points = True

    # Initialize detector
    detector = OwlV2SAM(sam_checkpoint=args.sam_checkpoint)

    # Process based on mode
    if args.mode == "detect":
        # Detect and segment
        results = detector.detect_and_segment(
            image=args.image,
            text_prompts=args.prompts,
            detection_threshold=args.threshold,
            return_all_detections=args.all_detections,
        )

        # Print detection results
        if results["detected"]:
            if "detections" in results:
                print(f"Found {len(results['detections'])} objects")
                for i, det in enumerate(results["detections"]):
                    print(
                        f"Detection {i + 1}: {det['text_prompt']} (score: {det['score']:.3f})"
                    )
            else:
                print(
                    f"Detected {results['text_prompt']} with confidence {results['score']:.3f}"
                )
        else:
            print("No objects detected")

    elif args.mode == "point":
        # Check if points are provided
        if not args.points:
            print("Error: No points provided for point-based segmentation.")
            exit(1)

        # Reshape points from flat list [x1, y1, x2, y2, ...] to [[x1, y1], [x2, y2], ...]
        point_coords = []
        for i in range(0, len(args.points), 2):
            if i + 1 < len(args.points):
                point_coords.append([args.points[i], args.points[i + 1]])

        # If point labels not provided, default to all foreground (1)
        if not args.point_labels:
            point_labels = [1] * len(point_coords)
        else:
            point_labels = args.point_labels

        if len(point_labels) != len(point_coords):
            print("Error: Number of point labels must match number of points.")
            exit(1)

        # Segment with points
        results = detector.segment_with_points(
            image=args.image,
            points=point_coords,
            point_labels=point_labels,
            multimask_output=args.multimask,
        )

        # Print results
        num_masks = len(results["masks"])
        print(f"Generated {num_masks} mask{'s' if num_masks != 1 else ''}")
        for i, score in enumerate(results["scores"]):
            print(f"Mask {i + 1} score: {score:.3f}")

    # Visualize results if requested
    if args.visualize:
        vis_image = detector.visualize(
            results=results,
            show_box=args.show_box,
            show_mask=args.show_mask,
            show_points=args.show_points,
            mask_idx=args.mask_idx,
            save_path=args.save_path,
        )

        # Display image
        detector.display_image(vis_image)
