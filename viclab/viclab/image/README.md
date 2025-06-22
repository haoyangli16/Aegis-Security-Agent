# Dou2DTools Class

The `Dou2DTools` class provides comprehensive image perception capabilities using the Seed-1.5-VL Pro model. It supports various computer vision tasks including object detection, phrase grounding, counting, text spotting, and more.

## Features

- **Phrase Grounding**: Locate objects described by natural language phrases
- **Object Detection**: Detect and locate all objects in an image
- **Open-ended Detection**: Custom detection queries for specific requirements
- **Object Counting**: Count specific types of objects with visual verification
- **Visual Prompt Grounding**: Use exemplar images to find similar objects
- **Text Spotting**: Detect and recognize text in images with polygon coordinates
- **Document to HTML**: Convert document images to structured HTML
- **Advanced Visualization**: Rich visualization tools using supervision library

## Installation

Ensure you have the required dependencies:

```bash
pip install openai pillow numpy supervision opencv-python matplotlib


# download the checkpoints

mkdir checkpoints/
cd checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

## Quick Start

```python
from viclab.image.perception import Dou2DTools

# Initialize (set OPENAI_API_KEY environment variable)
perceptor = Dou2DTools()

# Basic phrase grounding
result = perceptor.phrase_grounding("image.jpg", "red car")
print(f"Found {len(result['bboxes'])} red cars")

# Visualize results
vis_image = perceptor.visualize("image.jpg", bboxes=result['bboxes'])
```

## API Reference

### Constructor

```python
Dou2DTools(
    api_key: Optional[str] = None,
    base_url: str = "https://ark.cn-beijing.volces.com/api/v3",
    seed_vl_version: str = "doubao-1-5-thinking-vision-pro-250428"
)
```

**Parameters:**
- `api_key`: OpenAI API key (uses `OPENAI_API_KEY` env var if None)
- `base_url`: API endpoint base URL
- `seed_vl_version`: Model version to use

### Core Methods

#### `phrase_grounding(image_path: str, phrase: str) -> Dict[str, Any]`

Locate objects described by a phrase.

```python
result = perceptor.phrase_grounding("image.jpg", "person wearing red shirt")
# Returns: {"bboxes": [[x1, y1, x2, y2], ...], "raw_response": "..."}
```

#### `object_detection(image_path: str) -> Dict[str, Any]`

Detect all objects in the image.

```python
result = perceptor.object_detection("image.jpg")
# Returns: {"detections": [[x1, y1, x2, y2], ...], "raw_response": "..."}
```

#### `count_objects(image_path: str, object_type: str) -> Dict[str, Any]`

Count specific objects with bounding box verification.

```python
result = perceptor.count_objects("image.jpg", "cars")
# Returns: {"count": 5, "bboxes": [...], "raw_response": "..."}
```

#### `visual_prompt_grounding(image_path: str, exemplar_path: str, prompt: str) -> Dict[str, Any]`

Find objects using visual examples.

```python
result = perceptor.visual_prompt_grounding(
    "target.jpg", 
    "exemplar.jpg", 
    "Find similar objects"
)
# Returns: {"bboxes": [...], "points": [...], "raw_response": "..."}
```

#### `text_spotting(image_path: str) -> Dict[str, Any]`

Detect and recognize text with polygon coordinates.

```python
result = perceptor.text_spotting("document.jpg")
# Returns: {"text_detections": [[[x1,y1], [x2,y2], ...], ...], "raw_response": "..."}
```

### Visualization Methods

#### `visualize(image_path, bboxes=None, points=None, labels=None, save_path=None)`

Visualize detection results with bounding boxes and points.

```python
vis_image = perceptor.visualize(
    "image.jpg",
    bboxes=[[100, 100, 200, 200]],
    labels=["car"],
    save_path="result.jpg"
)
```

#### `visualize_text(image_path, polygons, save_path=None)`

Visualize text detection results with polygons.

```python
vis_image = perceptor.visualize_text("image.jpg", text_polygons)
```

## Coordinate System

The model uses a normalized coordinate system (0-999) which is automatically converted to actual image coordinates:

- **Input**: Model returns coordinates in 0-999 range
- **Output**: Automatically converted to actual pixel coordinates
- **Format**: `[x1, y1, x2, y2]` for bounding boxes, `[x, y]` for points

## Examples

### Basic Usage

```python
# Initialize
perceptor = Dou2DTools()

# Find all people
people_result = perceptor.phrase_grounding("crowd.jpg", "people")

# Count cars
car_count = perceptor.count_objects("street.jpg", "cars")
print(f"Found {car_count['count']} cars")

# Detect text
text_result = perceptor.text_spotting("document.jpg")
```

### Advanced Usage

```python
# Multi-task analysis
def analyze_image(image_path):
    perceptor = Dou2DTools()
    
    # Object detection
    objects = perceptor.object_detection(image_path)
    
    # Count people
    people = perceptor.count_objects(image_path, "people")
    
    # Find text
    text = perceptor.text_spotting(image_path)
    
    # Create combined visualization
    vis = perceptor.visualize(
        image_path,
        bboxes=objects['detections'] + people['bboxes'],
        labels=["object"] * len(objects['detections']) + 
               ["person"] * len(people['bboxes'])
    )
    
    return {
        "objects": len(objects['detections']),
        "people": people['count'],
        "text_regions": len(text['text_detections']),
        "visualization": vis
    }
```

### Batch Processing

```python
import os

def process_directory(image_dir, target_object):
    perceptor = Dou2DTools()
    results = []
    
    for filename in os.listdir(image_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(image_dir, filename)
            result = perceptor.count_objects(image_path, target_object)
            results.append({
                "filename": filename,
                "count": result['count']
            })
    
    return results

# Process all images in a directory
car_counts = process_directory("images/", "cars")
```

## Error Handling

```python
try:
    perceptor = Dou2DTools()
    result = perceptor.phrase_grounding("image.jpg", "cats")
except ValueError as e:
    print(f"Configuration error: {e}")
except FileNotFoundError:
    print("Image file not found")
except Exception as e:
    print(f"API error: {e}")
```

## Environment Setup

Set your API key:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

Or pass it directly:

```python
perceptor = Dou2DTools(api_key="your-api-key")
```

## Performance Tips

1. **Batch Processing**: Reuse the same `Dou2DTools` instance for multiple images
2. **Image Size**: Large images may take longer to process
3. **Coordinate Conversion**: Coordinates are automatically converted - no manual scaling needed
4. **Visualization**: Use `save_path` parameter to save visualizations instead of returning large arrays

## Troubleshooting

### Common Issues

1. **API Key Error**: Ensure `OPENAI_API_KEY` is set correctly
2. **Import Error**: Check that all dependencies are installed
3. **Image Loading Error**: Verify image paths and file formats
4. **Empty Results**: Check image quality and adjust prompts

### Supported Image Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- WebP (.webp)

### Dependencies

- `openai`: API client
- `PIL`: Image processing
- `numpy`: Array operations
- `supervision`: Visualization annotations
- `opencv-python`: Image I/O and processing
- `matplotlib`: Plotting and display 


### For the checkpoint:
yolov11: wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt