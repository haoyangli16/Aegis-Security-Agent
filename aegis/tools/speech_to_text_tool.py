"""
Speech-to-Text Tool for Aegis Security Agent System.

This tool provides audio transcription capabilities for voice commands
from security operators using the Faster Whisper model.
"""

import base64
import os
import sys
from typing import Any, Dict, Optional, Union

import numpy as np
from google.adk.tools import ToolContext

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from aegis.config.settings import get_setting

# Global Whisper model instance
whisper_model = None


def init_whisper_model():
    """Initialize the Whisper model for speech recognition."""
    global whisper_model

    if whisper_model is None:
        try:
            from faster_whisper import WhisperModel

            model_size = get_setting("audio_settings.whisper_model", "small")
            whisper_model = WhisperModel(model_size, compute_type="int8")
            print(f"✅ Whisper model '{model_size}' initialized")
            return True
        except ImportError:
            print("❌ faster_whisper not available. Audio transcription disabled.")
            return False
        except Exception as e:
            print(f"❌ Error initializing Whisper model: {e}")
            return False

    return True


def transcribe_audio(
    audio_data: Union[str, bytes],
    tool_context: ToolContext,
    language: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Transcribe audio data to text using Faster Whisper.

    This tool converts voice commands from security operators into text that
    the agent can understand and process. It supports various audio formats
    and provides confidence scores for transcription quality.

    Args:
        audio_data: Audio data as base64 string or raw bytes
        tool_context: ADK tool context for session state and services
        language: Language code for transcription (e.g., 'en'). Optional.

    Returns:
        Dict containing:
        - status: Success/error status
        - transcribed_text: The transcribed text from audio
        - confidence: Confidence score of transcription
        - language_detected: Detected language of the audio
        - processing_time: Time taken for transcription
    """

    try:
        # Set default language if not provided
        transcription_language = language or "en"

        # Initialize Whisper model if needed
        if not init_whisper_model():
            return {
                "status": "error",
                "message": "Speech recognition not available. Whisper model not initialized.",
                "transcribed_text": "",
                "confidence": 0.0,
            }

        # Handle different audio data formats
        audio_array = _process_audio_input(audio_data)

        if audio_array is None:
            return {
                "status": "error",
                "message": "Invalid audio data format",
                "transcribed_text": "",
                "confidence": 0.0,
            }

        # Check audio quality
        if len(audio_array) < 1000:  # Too short
            return {
                "status": "error",
                "message": "Audio clip too short for transcription",
                "transcribed_text": "",
                "confidence": 0.0,
            }

        import time

        start_time = time.time()

        # Transcribe audio using Whisper
        try:
            segments, info = whisper_model.transcribe(
                audio_array,
                language=transcription_language,
                beam_size=5,
                temperature=0.0,
            )

            # Combine all segments into full text
            transcribed_text = " ".join(segment.text for segment in segments)

            # Calculate average confidence
            segment_list = list(segments)
            if segment_list:
                avg_confidence = sum(
                    segment.avg_logprob for segment in segment_list
                ) / len(segment_list)
                # Convert log probability to approximate confidence (0-1)
                confidence = max(0.0, min(1.0, (avg_confidence + 5.0) / 5.0))
            else:
                confidence = 0.0

            processing_time = time.time() - start_time

            # Store transcription in session state
            transcription_key = "last_transcription"
            tool_context.state[transcription_key] = {
                "text": transcribed_text,
                "confidence": confidence,
                "language": info.language,
                "timestamp": tool_context.state.get("current_time", "unknown"),
            }

            # Update transcription statistics
            total_transcriptions = tool_context.state.get("total_transcriptions", 0) + 1
            tool_context.state["total_transcriptions"] = total_transcriptions

            return {
                "status": "success",
                "message": f"Audio transcribed successfully in {processing_time:.2f} seconds",
                "transcribed_text": transcribed_text.strip(),
                "confidence": round(confidence, 3),
                "language_detected": info.language,
                "language_probability": round(info.language_probability, 3),
                "processing_time": round(processing_time, 3),
                "audio_duration": len(audio_array)
                / get_setting("audio_settings.sample_rate", 16000),
                "total_transcriptions": total_transcriptions,
            }

        except Exception as whisper_error:
            return {
                "status": "error",
                "message": f"Whisper transcription failed: {str(whisper_error)}",
                "transcribed_text": "",
                "confidence": 0.0,
                "error_details": str(whisper_error),
            }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Audio transcription tool failed: {str(e)}",
            "transcribed_text": "",
            "confidence": 0.0,
            "error_details": str(e),
        }


def transcribe_voice_command(
    audio_data: Union[str, bytes], tool_context: ToolContext
) -> Dict[str, Any]:
    """
    Specialized transcription for security voice commands with command parsing.

    This is a higher-level tool that not only transcribes audio but also
    analyzes the content for security-specific commands and contexts.

    Args:
        audio_data: Audio data as base64 string or raw bytes
        tool_context: ADK tool context for session state and services

    Returns:
        Dict containing transcription plus command analysis
    """

    try:
        # First, perform standard transcription
        transcription_result = transcribe_audio(audio_data, tool_context)

        if transcription_result["status"] != "success":
            return transcription_result

        transcribed_text = transcription_result["transcribed_text"]

        # Analyze the transcribed text for security commands
        command_analysis = _analyze_security_command(transcribed_text)

        # Enhance the response with command analysis
        enhanced_result = transcription_result.copy()
        enhanced_result.update(
            {
                "command_type": command_analysis["command_type"],
                "camera_mentioned": command_analysis["camera_mentioned"],
                "objects_to_detect": command_analysis["objects_to_detect"],
                "action_required": command_analysis["action_required"],
                "confidence_level": command_analysis["confidence_level"],
                "parsed_intent": command_analysis["intent"],
            }
        )

        return enhanced_result

    except Exception as e:
        return {
            "status": "error",
            "message": f"Voice command processing failed: {str(e)}",
            "transcribed_text": "",
            "confidence": 0.0,
            "error_details": str(e),
        }


def _process_audio_input(audio_data: Union[str, bytes]) -> np.ndarray:
    """
    Process different audio input formats into numpy array.

    Args:
        audio_data: Audio as base64 string or raw bytes

    Returns:
        numpy.ndarray: Audio data as float32 array, or None if invalid
    """

    try:
        if isinstance(audio_data, str):
            # Assume base64 encoded audio
            try:
                audio_bytes = base64.b64decode(audio_data)
            except Exception:
                return None
        else:
            # Raw bytes
            audio_bytes = audio_data

        # Convert bytes to numpy array
        # Assume 16-bit signed integers (common format)
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16)

        # Convert to float32 and normalize to [-1, 1]
        audio_array = audio_array.astype(np.float32) / 32768.0

        return audio_array

    except Exception as e:
        print(f"Error processing audio input: {e}")
        return None


def _analyze_security_command(text: str) -> Dict[str, Any]:
    """
    Analyze transcribed text for security-specific commands and intents.

    Args:
        text: Transcribed text to analyze

    Returns:
        Dict containing command analysis
    """

    text_lower = text.lower()

    # Initialize analysis result
    analysis = {
        "command_type": "unknown",
        "camera_mentioned": None,
        "objects_to_detect": [],
        "action_required": "monitor",
        "confidence_level": "medium",
        "intent": text,
    }

    # Detect camera references
    camera_patterns = ["cam", "camera", "gate", "entrance", "parking", "plaza"]
    for pattern in camera_patterns:
        if pattern in text_lower:
            # Try to extract camera number/ID
            import re

            camera_match = re.search(rf"{pattern}[\s]*(\d+|[a-z_]+)", text_lower)
            if camera_match:
                analysis["camera_mentioned"] = f"{pattern}{camera_match.group(1)}"
            break

    # Detect objects to look for
    object_keywords = {
        "person": ["person", "people", "individual", "someone"],
        "backpack": ["backpack", "bag", "luggage"],
        "car": ["car", "vehicle", "auto"],
        "weapon": ["weapon", "gun", "knife"],
        "suspicious": ["suspicious", "strange", "unusual"],
    }

    for obj_type, keywords in object_keywords.items():
        if any(keyword in text_lower for keyword in keywords):
            analysis["objects_to_detect"].append(obj_type)

    # Determine command type
    if any(word in text_lower for word in ["scan", "detect", "look for", "find"]):
        analysis["command_type"] = "detection"
        analysis["action_required"] = "analyze"
    elif any(word in text_lower for word in ["switch", "show", "display", "view"]):
        analysis["command_type"] = "camera_control"
        analysis["action_required"] = "switch_camera"
    elif any(word in text_lower for word in ["analyze", "assess", "check"]):
        analysis["command_type"] = "analysis"
        analysis["action_required"] = "comprehensive_analysis"
    elif any(word in text_lower for word in ["monitor", "watch", "observe"]):
        analysis["command_type"] = "monitoring"
        analysis["action_required"] = "monitor"

    # Determine confidence level based on clarity
    if len(analysis["objects_to_detect"]) > 0 and analysis["camera_mentioned"]:
        analysis["confidence_level"] = "high"
    elif len(analysis["objects_to_detect"]) > 0 or analysis["camera_mentioned"]:
        analysis["confidence_level"] = "medium"
    else:
        analysis["confidence_level"] = "low"

    return analysis
