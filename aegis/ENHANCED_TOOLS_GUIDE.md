# Enhanced Aegis Tools Guide

## Overview

The Aegis Security Agent now supports **dual-method** capabilities for both object detection and VLM analysis, giving you the flexibility to choose between **speed** and **accuracy** based on your specific security needs.

## 🔍 Object Detection Enhancement

### Methods Available

| Method | Speed | Accuracy | Best Use Case |
|--------|-------|----------|---------------|
| **YOLO** (Default) | ⚡ Very Fast (~50-100ms) | Good for common objects | Real-time monitoring, continuous surveillance |
| **OWLv2** | 🎯 Slower (~1-3 seconds) | Excellent for any object | Incident investigation, custom object detection |

### Usage Examples

#### YOLO Detection (Fast)
```python
# Default - backward compatible
result = detect_objects("cam1", ["person", "car"], tool_context)

# Explicit YOLO
result = detect_objects("cam1", ["person", "car"], tool_context, detection_method="yolo")
```

#### OWLv2 Detection (Accurate)
```python
# Natural language object descriptions
result = detect_objects("cam2", ["weapon", "suspicious bag", "unattended luggage"], 
                       tool_context, detection_method="owlv2")
```

### Object Detection Capabilities

#### YOLO Strengths:
- ✅ 80 predefined COCO classes
- ✅ Extremely fast inference
- ✅ Low resource usage
- ✅ Perfect for continuous monitoring

#### OWLv2 Strengths:
- ✅ Unlimited object classes via natural language
- ✅ Superior accuracy for security objects
- ✅ Better handling of unusual/custom objects
- ✅ Excellent for detailed forensic analysis

## 🧠 VLM Analysis Enhancement

### Methods Available

| Method | Speed | Accuracy | Best Use Case |
|--------|-------|----------|---------------|
| **SmolVLM** (Default) | ⚡ Fast (~1-2 seconds) | Good scene understanding | Quick scene descriptions, real-time analysis |
| **Seed-VL-1.5 pro** | 🎯 Slower (~3-5 seconds) | Excellent detailed analysis | Incident investigation, detailed security assessment |

### Usage Examples

#### SmolVLM Analysis (Fast)
```python
# Default - backward compatible
result = analyze_scene_with_vlm("cam1", "What's happening in this area?", tool_context)

# Explicit SmolVLM
result = analyze_scene_with_vlm("cam1", "Describe the scene", tool_context, vlm_method="smolvlm")
```

#### Seed-VL-1.5 pro Analysis (Accurate)
```python
# Detailed security analysis
result = analyze_scene_with_vlm("cam2", "Analyze this scene for security threats and suspicious behavior", 
                               tool_context, vlm_method="seedvl")
```

### VLM Analysis Capabilities

#### SmolVLM Strengths:
- ✅ Local processing (no API calls)
- ✅ Fast response time
- ✅ Good for basic scene understanding
- ✅ Lower resource requirements

#### Seed-VL-1.5 pro Strengths:
- ✅ Superior understanding of complex scenes
- ✅ Better at detecting subtle security concerns
- ✅ More detailed and accurate descriptions
- ✅ Excellent for incident documentation

## 🚀 Strategic Usage Recommendations

### Real-time Monitoring Setup
```python
# For continuous surveillance - prioritize speed
detection = detect_objects("cam1", ["person", "vehicle"], context, detection_method="yolo")
analysis = analyze_scene_with_vlm("cam1", "Any unusual activity?", context, vlm_method="smolvlm")
```

### Incident Investigation Setup
```python
# For detailed analysis - prioritize accuracy
detection = detect_objects("cam2", ["weapon", "suspicious object"], context, detection_method="owlv2")
analysis = analyze_scene_with_vlm("cam2", "Detailed security assessment needed", context, vlm_method="seedvl")
```

### Balanced Performance Setup
```python
# Mixed approach for important locations
detection = detect_objects("cam3", ["person", "bag"], context, detection_method="yolo")  # Fast detection
analysis = analyze_scene_with_vlm("cam3", "Security analysis", context, vlm_method="seedvl")  # Detailed analysis
```

## 🛡️ Security Agent Integration

### Agent Commands

The Aegis agent can intelligently choose methods based on context:

```
User: "Quick scan of camera 3 for people"
→ Uses YOLO + SmolVLM for speed

User: "Detailed analysis of suspicious activity on camera 2" 
→ Uses OWLv2 + Seed-VL-1.5 pro for accuracy

User: "Emergency - check camera 1 for weapons immediately"
→ Uses OWLv2 for accurate weapon detection
```

### Natural Language Method Selection

The agent understands these keywords for method selection:

**For Speed:**
- "quick", "fast", "immediate", "real-time"
- "continuous monitoring", "regular check"

**For Accuracy:**
- "detailed", "thorough", "comprehensive", "forensic"
- "investigation", "incident", "suspicious"

## 📊 Performance Comparison

### Speed Benchmarks (Approximate)
```
YOLO Detection:        50-100ms   🚀🚀🚀🚀🚀
OWLv2 Detection:       1-3 sec    🚀🚀
SmolVLM Analysis:      1-2 sec    🚀🚀🚀🚀
Seed-VL-1.5 Analysis:  3-5 sec    🚀🚀
```

### Accuracy Levels
```
YOLO:        ⭐⭐⭐⭐     (Excellent for COCO objects)
OWLv2:       ⭐⭐⭐⭐⭐   (Superior for any object)
SmolVLM:     ⭐⭐⭐⭐     (Good scene understanding)
Seed-VL-1.5: ⭐⭐⭐⭐⭐   (Excellent detailed analysis)
```

## 🔧 Configuration and Dependencies

### Requirements

**YOLO + SmolVLM (Default):**
- ✅ Included in standard Aegis installation
- ✅ Local processing only

**OWLv2 Enhancement:**
- 📦 Requires VicLab package
- 📦 Requires SAM model checkpoint
- 🔧 GPU recommended for best performance

**Seed-VL-1.5 pro Enhancement:**
- 📦 Requires VicLab package  
- 🌐 Requires API access to Doubao service
- 🔑 Requires API key configuration

### Installation Check

Run the test script to verify which methods are available:
```bash
cd aegis
python test_enhanced_tools.py
```

## ⚠️ Error Handling

The tools gracefully handle missing dependencies:

```python
# If OWLv2 not available, returns error with suggestion
result = detect_objects("cam1", ["person"], context, detection_method="owlv2")
# Returns: {"status": "error", "message": "OWLv2 requested but not available. Use 'yolo'"}

# If Seed-VL not available, returns error with fallback
result = analyze_scene_with_vlm("cam1", "Question", context, vlm_method="seedvl") 
# Returns: {"status": "error", "message": "Seed-VL-1.5 pro requested but not available. Use 'smolvlm'"}
```

## 🎯 Best Practices

### 1. Method Selection Strategy
- **Continuous Monitoring:** YOLO + SmolVLM
- **Alert Verification:** OWLv2 + SmolVLM  
- **Incident Investigation:** OWLv2 + Seed-VL-1.5 pro
- **Emergency Response:** Choose based on urgency vs accuracy needs

### 2. Resource Management
- Use fast methods for high-frequency checks
- Reserve accurate methods for triggered events
- Monitor system performance and adjust accordingly

### 3. Fallback Strategy
- Always have YOLO + SmolVLM as backup
- Test enhanced methods availability at startup
- Implement graceful degradation in agent logic

## 🔄 Backward Compatibility

All existing code continues to work unchanged:
```python
# These calls work exactly as before
detect_objects("cam1", ["person"], context)  # Uses YOLO by default
analyze_scene_with_vlm("cam1", "Question", context)  # Uses SmolVLM by default
```

The new method parameters are optional and default to the original fast methods.

## 📞 Support and Troubleshooting

### Common Issues

1. **OWLv2 not available:** Install VicLab dependencies
2. **Seed-VL-1.5 pro not available:** Configure API access
3. **Slow performance:** Check GPU availability and model loading
4. **Memory issues:** Monitor resource usage with accurate methods

### Debug Mode

Enable debug logging to see method selection and performance:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

This enhanced dual-method system gives you the best of both worlds: **speed when you need it, accuracy when it matters most**. 