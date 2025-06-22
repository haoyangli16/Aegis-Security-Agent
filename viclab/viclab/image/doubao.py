# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
from typing import List, Union, Optional, Tuple, Dict, Any
import os
import re
import json
import base64

import cv2
import numpy as np
from PIL import Image, ImageDraw
import supervision as sv
from openai import OpenAI
import matplotlib.pyplot as plt

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
    image_path,
    boxes=None,
    points=None,
    labels=None,
    output_path=None,
):
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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

class Dou2DTools:
    """
    A class for image perception tasks using the Seed-1.5-VL Pro model.
    Supports phrase grounding, object detection, counting, visual prompts, text spotting, and more.
    """
    
    def __init__(
        self,
        base_url: str = "https://ark.cn-beijing.volces.com/api/v3",
        seed_vl_version: str = "doubao-1-5-thinking-vision-pro-250428",
        openai_api_key: Optional[str] = "ffb01810-79f6-402d-9ee6-46c109b3f93f",
    ):
        """
        Initialize the Dou2DTools.
        
        Args:
            base_url: Base URL for the API endpoint.
            seed_vl_version: Model version to use.
            openai_api_key: OpenAI API key. If None, will try to get from OPENAI_API_KEY environment variable.
        """
        if openai_api_key is None:
            openai_api_key = os.environ.get("OPENAI_API_KEY")
        
        self.client = OpenAI(base_url=base_url, api_key=openai_api_key)
        self.seed_vl_version = seed_vl_version
    
    def _get_image_format(self, image_path: str) -> str:
        image_format = image_path.split('.')[-1].lower()
        if image_format == 'jpg':
            image_format = 'jpeg'
        assert image_format in ['jpeg', 'jpg', 'png', 'webp'], f"Unsupported image format: {image_format}"
        return image_format

    def _encode_image_to_base64(self, image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def _call_api(self, messages: list, image_path: str = None) -> str:
        # If image_path is provided, fix the image format in the data URL
        if image_path is not None:
            image_format = self._get_image_format(image_path)
            for msg in messages:
                if 'content' in msg:
                    for content in msg['content']:
                        if content.get('type') == 'image_url' and 'image_url' in content:
                            url = content['image_url']['url']
                            # Replace the format in the data URL
                            content['image_url']['url'] = re.sub(
                                r'data:image/[^;]+;',
                                f'data:image/{image_format};',
                                url
                            )
        try:
            response = self.client.chat.completions.create(
                model=self.seed_vl_version,
                messages=messages
            )
            if response.choices:
                return response.choices[0].message.content
            return None
        except Exception as e:
            print(f"API call failed: {e}")
            return None
    
    def _parse_transform_bboxes_vanilla(self, image_path: str, message: str) -> np.ndarray:
        pattern = r'<bbox>(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)</bbox>'
        matches = re.finditer(pattern, message)
        bboxes = [
            [float(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4))]
            for m in matches
        ]
        image = Image.open(image_path)
        w, h = image.size
        if not bboxes:
            return np.zeros((0, 4), dtype='float')
        bboxes = np.array(bboxes, dtype='float') / 1000
        bboxes[:, 0::2] *= w
        bboxes[:, 1::2] *= h
        return bboxes

    def _parse_transform_points_vanilla(self, image_path: str, message: str) -> np.ndarray:
        pattern = r'<point>(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)</point>'
        matches = re.finditer(pattern, message)
        points = [
            [float(m.group(1)), float(m.group(2))]
            for m in matches
        ]
        image = Image.open(image_path)
        w, h = image.size
        if not points:
            return np.zeros((0, 2), dtype='float')
        points = np.array(points, dtype='float') / 1000
        points[:, 0] *= w
        points[:, 1] *= h
        return points

    def _parse_transform_bboxes_json(self, image_path: str, message: str) -> (np.ndarray, list):
        pattern = r'<bbox>(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)</bbox>'
        items = json.loads(message)
        bboxes = []
        categories = []
        for item in items:
            try:
                cat, bbox_str = tuple(item.items())[0]
                bbox = re.match(pattern, bbox_str)
                bbox = [float(bbox.group(i + 1)) for i in range(4)]
                bboxes.append(bbox)
                categories.append(cat)
            except Exception as e:
                print("error parsing: {}, error: {}".format(item, e))
        image = Image.open(image_path)
        w, h = image.size
        # normalize the bbox to [0, 1]
        bboxes = np.array(bboxes, dtype='float') / 1000
        bboxes[:, 0::2] *= w
        bboxes[:, 1::2] *= h
        return bboxes, categories

    def phrase_grounding(self, image_path: str, phrase: str) -> dict:
        base64_image = self._encode_image_to_base64(image_path)
        image_format = self._get_image_format(image_path)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/{image_format};base64,{base64_image}",
                            "detail": "high"
                        }
                    },
                    {
                        "type": "text",
                        "text": f"Please locate the {phrase} in the image using <bbox>x1 y1 x2 y2</bbox> format."
                    }
                ]
            }
        ]
        response = self._call_api(messages, image_path=image_path)
        if not response:
            return {"bboxes": [], "raw_response": ""}
        bboxes = self._parse_transform_bboxes_vanilla(image_path, response)
        return {"bboxes": bboxes, "raw_response": response}

    def object_detection(self, image_path: str, categories: Union[str, list] = None) -> dict:
        base64_image = self._encode_image_to_base64(image_path)
        image_format = self._get_image_format(image_path)
        if categories is not None:
            if isinstance(categories, list):
                categories_str = ", ".join(categories)
            else:
                categories_str = categories
            prompt = f"Please detect all '{categories_str}' in the image."
            prompt += """
            Please output all objects in the JSON format: [{"category1": "<bbox1>"}, {"category2": "<bbox2>"}].
            For each object, provide its category and bounding box in the format: <bbox>x1 y1 x2 y2</bbox>. And don't output {'category': 'category1', 'bbox': '642 438 818 996'} format.
            The correct format should be: [{"category1": "<bbox> a1 b1 c1 d1 </bbox>"}, {"category2": "<bbox2> a2 b2 c2 d2 </bbox>"}]
            """
        else:
            prompt = "Please detect all objects in this image and provide their locations using <bbox>x1 y1 x2 y2</bbox> format. List the object name before each bbox."
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/{image_format};base64,{base64_image}",
                            "detail": "high"
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]
        response = self._call_api(messages, image_path=image_path)
        if not response:
            return {"bboxes": [], "categories": [], "raw_response": ""}
        bboxes, categories = self._parse_transform_bboxes_json(image_path, response)
        return {"bboxes": bboxes, "categories": categories, "raw_response": response}
    
    def open_ended_detection(self, image_path: str) -> Dict[str, Any]:
        """
        Perform open-ended object detection based on a query.
        
        Args:
            image_path: Path to the image file.
            
        Returns:
            Dictionary containing detection results.
        """
        base64_image = self._encode_image_to_base64(image_path)
        text_prompts = "please detect all objects in the image"
        text_prompts += """
        Please output all objects in the JSON format: [{"category1": "<bbox1>"}, {"category2": "<bbox2>"}].
        For each object, provide its category and bounding box in the format: <bbox>x1 y1 x2 y2</bbox>. And don't output {'category': 'category1', 'bbox': 'x1 y1 x2 y2'} format.
        The correct format should be: [{"category1": "<bbox> x11 y11 x21 y21 </bbox>"}, {"category2": "<bbox2> x12 y12 x22 y22 </bbox>"},...,]
        """

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "high"
                        }
                    },
                    {
                        "type": "text",
                        "text": text_prompts
                    }
                ]
            }
        ]
        
        response = self._call_api(messages)
        if not response:
            return {"bboxes": [], "categories": [], "raw_response": ""}
        
        # Get image dimensions
        with Image.open(image_path) as img:
            image_width, image_height = img.size

        bboxes, categories = self._parse_transform_bboxes_json(image_path, response)
        
        return {
            "bboxes": bboxes,
            "categories": categories,
            "raw_response": response
        }
    
    def count_objects(self, image_path: str, with_bbox: bool = True, object_type: str = None) -> Dict[str, Any]:
        """
        Count specific objects in the image.
        
        Args:
            image_path: Path to the image file.
            object_type: Type of object to count.
            
        Returns:
            Dictionary containing count and detection results.
        """
        base64_image = self._encode_image_to_base64(image_path)
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "high"
                        }
                    },  
                    {
                        "type": "text",
                        "text": f"Output the points (<point>x y</point>) for each {object_type} in the image."
                    }
                ]
            }
        ]
        
        response = self._call_api(messages)
        if not response:
            return {"count": 0, "bboxes": [], "points": [], "raw_response": ""}
        
        # Get image dimensions
        with Image.open(image_path) as img:
            image_width, image_height = img.size
        
        if with_bbox:
            bboxes = self._parse_transform_bboxes_vanilla(image_path, response)
        else:
            bboxes = []

        points = self._parse_transform_points_vanilla(image_path, response)
        count = len(bboxes)
        
        # Try to extract count from text
        count_match = re.search(r'(\d+)', response)
        if count_match:
            text_count = int(count_match.group(1))
            if text_count > count:  # Use text count if it's higher (more reliable)
                count = text_count
        
        return {
            "count": count,
            "bboxes": bboxes,
            "points": points,
            "raw_response": response
        }
    
    def visual_prompt_grounding(self, exemplar_path: str, image_path: str, prompt: str = None) -> dict:
        assert exemplar_path is not None, "exemplar_path is required"
        assert image_path is not None, "image_path is required"
        base64_image1 = self._encode_image_to_base64(exemplar_path)
        image_format1 = self._get_image_format(exemplar_path)
        base64_image2 = self._encode_image_to_base64(image_path)
        image_format2 = self._get_image_format(image_path)
        if prompt is None:
            text_prompts = "Given the first image, please recognize the main object in the first image. Please detect all the similar objects in the second image. Please output all bounding boxes of the object in the second image with the format <bbox>x1 y1 x2 y2</bbox>"
        else:
            text_prompts = prompt
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/{image_format1};base64,{base64_image1}",
                            "detail": "high"
                        }
                    },
                    {
                        "type": "text",
                        "text": text_prompts
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/{image_format2};base64,{base64_image2}",
                            "detail": "high"
                        }
                    },
                ]
            }
        ]
        response = self._call_api(messages)
        if not response:
            return {"bboxes": [], "raw_response": ""}
        bboxes = self._parse_transform_bboxes_vanilla(image_path, response)
        return {"bboxes": bboxes, "raw_response": response}
    
    def _parse_text_polygons(self, text: str, image_width: int, image_height: int) -> Tuple[List[List[List[int]]], List[str]]:
        """Parse text and polygons from text spotting response."""
        # Extract text content
        text_pattern = r'<text>(.*?)</text>'
        texts = re.findall(text_pattern, text)
        
        # Extract polygons
        polygon_pattern = r'<polygon>(.*?)</polygon>'
        polygon_matches = re.findall(polygon_pattern, text, re.DOTALL)
        
        polygons = []
        for match in polygon_matches:
            # Parse coordinate pairs - handle both comma and space separated
            coord_pairs = re.findall(r'(\d+)\s*,?\s*(\d+)', match)
            polygon = []
            for x, y in coord_pairs:
                actual_x = int(int(x) * image_width / 999)
                actual_y = int(int(y) * image_height / 999)
                polygon.append([actual_x, actual_y])
            if len(polygon) >= 3:  # Valid polygon needs at least 3 points
                polygons.append(polygon)
        
        # Ensure we have matching number of texts and polygons
        if len(texts) < len(polygons):
            texts.extend([''] * (len(polygons) - len(texts)))
        elif len(texts) > len(polygons):
            texts = texts[:len(polygons)]
            
        return polygons, texts

    def text_spotting(self, image_path: str) -> Dict[str, Any]:
        base64_image = self._encode_image_to_base64(image_path)
        image_format = self._get_image_format(image_path)
        text_prompts = "Grounding text in the image in the format of <text>text</text><polygon>x1 y1, x2 y2, x3 y3, x4 y4</polygon>"
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/{image_format};base64,{base64_image}",
                            "detail": "high"
                        }
                    },
                    {
                        "type": "text",
                        "text": text_prompts
                    }
                ]
            }
        ]
        response = self._call_api(messages, image_path=image_path)
        if not response:
            return {"polygons": [], "texts": [], "raw_response": ""}
        image_width, image_height = Image.open(image_path).size 
        polygons, texts = self._parse_text_polygons(response, image_width, image_height)
        return {"polygons": polygons, "texts": texts, "raw_response": response}
    
    def document_to_html(self, image_path: str) -> Dict[str, Any]:
        """
        Convert document image to HTML format.
        
        Args:
            image_path: Path to the document image file.
            
        Returns:
            Dictionary containing HTML conversion results.
        """
        base64_image = self._encode_image_to_base64(image_path)
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "high"
                        }
                    },
                    {
                        "type": "text",
                        "text": "Please convert this document image to HTML format, preserving the structure and formatting as much as possible."
                    }
                ]
            }
        ]
        
        response = self._call_api(messages)
        if not response:
            return {"html": "", "raw_response": ""}
        
        return {
            "html": response,
            "raw_response": response
        }
    
    def visualize(
        self,
        image_path: str,
        bboxes: Optional[np.ndarray] = None,
        points: Optional[np.ndarray] = None,
        labels: Optional[List[str]] = None,
        output_path: Optional[str] = None,
    ) -> np.ndarray:
        """
        Visualize detection results on the image.
        Args:
            image_path: Path to the input image
            bboxes: Bounding boxes to draw
            points: Points to draw
            classes: Class labels for the detections
            output_path: Optional path to save the visualization
        Returns:
            Annotated image as numpy array
        """
        return draw_boxes_points_with_labels(
            image_path=image_path,
            boxes=bboxes,
            points=points,
            labels=labels,
            output_path=output_path,
        )
    
    def visualize_text(self, image_path: str, polygons: List[List[List[int]]], texts: List[str] = None, output_path: str = None) -> np.ndarray:
        """
        Visualize text detection results on the image with text labels.
        
        Args:
            image_path: Path to the image file.
            polygons: List of polygons representing text regions.
            texts: List of text strings for each polygon.
            save_path: Path to save the visualized image.
            
        Returns:
            Annotated image as numpy array.
        """
        # Load image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Use PIL for text rendering as it handles fonts better
        pil_image = Image.fromarray(image)
        draw = ImageDraw.Draw(pil_image)
        
        # If no texts provided, create empty list
        if texts is None or len(texts) == 0:
            texts = [''] * len(polygons)
        
        # Draw polygons and text
        for idx, (polygon, text) in enumerate(zip(polygons, texts)):
            if len(polygon) >= 3:  # Need at least 3 points for a polygon
                # Convert to tuple format for PIL
                polygon_points = [(point[0], point[1]) for point in polygon]
                
                # Draw polygon outline
                draw.polygon(polygon_points, outline='red', width=2)
                
                # Draw text label if available
                if text and text.strip():
                    # Position text at the top-left of the polygon
                    text_x, text_y = polygon_points[0]
                    # Offset text slightly above the polygon
                    text_y = max(0, text_y - 20)
                    draw.text((text_x, text_y), text.strip(), fill='red')
        
        # Convert back to numpy array
        image = np.array(pil_image)
        
        # Save if path provided
        if output_path:
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, image_bgr)
        
        return image
