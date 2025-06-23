#!/usr/bin/env python3
"""
Aegis Security Co-Pilot - ADK Integration Server
Hybrid FastAPI server combining Google ADK agent capabilities with real-time video monitoring
"""
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, Response
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from google.adk.cli.fast_api import get_fast_api_app


# Add current directory to path for imports
# sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import our core modules
from aegis.core.analyze_frame import (
    analyze_frame,
    clean_result_for_json,
)
from aegis.core.security_context import initialize_global_context, get_security_context, cleanup_global_context

# Configure logging - Disable verbose ADK logs
logging.basicConfig(level=logging.WARNING)  # Only show warnings and errors
logger = logging.getLogger(__name__)

# Disable verbose ADK framework logging
logging.getLogger('google_adk').setLevel(logging.ERROR)  # Only show errors from ADK
logging.getLogger('google_genai').setLevel(logging.ERROR)  # Only show errors from GenAI
logging.getLogger('httpx').setLevel(logging.WARNING)  # Reduce HTTP request logs
logging.getLogger('uvicorn.access').setLevel(logging.WARNING)  # Reduce server access logs
logging.getLogger('aegis.server.web_monitor_server').setLevel(logging.ERROR)  # Reduce supervision logs
# logging.getLogger('aegis.core.security_context').setLevel(logging.ERROR)  # Reduce supervision logs

logger.disabled = True

def create_adk_app() -> FastAPI:
    """Create the ADK FastAPI application with video streaming capabilities"""

    # Get the current directory for agent configuration
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Create base ADK FastAPI app
    app = get_fast_api_app(
        agents_dir=current_dir,
        session_service_uri="sqlite:///./sessions.db",  # Correct parameter name
        allow_origins=["*"],  # Allow all origins for development
        web=True,  # Enable ADK web interface
    )

    # Create static directory for our frontend
    static_dir = Path(current_dir) / "static"
    static_dir.mkdir(exist_ok=True)

    # Mount static files
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    return app


def add_video_routes(app: FastAPI):
    """Add video streaming routes to the FastAPI app"""

    @app.get("/video")
    async def video_interface():
        """Serve the video monitoring interface"""
        static_dir = Path(__file__).parent / "static"
        index_file = static_dir / "index.html"

        if index_file.exists():
            return FileResponse(index_file)
        else:
            # Create the interface if it doesn't exist
            create_video_interface()
            return FileResponse(index_file)
        
    #@app.get("/api/analyze_security")
    #async def analyze_security(camera_id: str):
    #    tool_context = ToolContext()
    #    return analyze_security_situation(camera_id, tool_context)

    @app.get("/api/streams")
    async def get_streams():
        """Get camera stream information"""
        return JSONResponse(get_security_context().monitor_server.get_stream_info())

    @app.get("/api/frame/{camera_id}")
    async def get_frame(camera_id: int):
        """Get a single frame from a camera as base64"""
        if not (1 <= camera_id <= 4):
            return JSONResponse({"error": "Invalid camera ID"}, status_code=400)

        source_index = get_security_context().monitor_server.selected_sources[camera_id - 1]
        frame_data = get_security_context().monitor_server.get_frame_as_base64(source_index, camera_id)

        if frame_data is None:
            return JSONResponse({"error": "No frame available"}, status_code=404)

        return JSONResponse({"image": f"data:image/jpeg;base64,{frame_data}"})

    @app.get("/api/label/{camera_id}")
    async def label_camera_frame(camera_id: int):
        """Get analysis results for a camera frame"""
        if not (1 <= camera_id <= 4):
            return JSONResponse({"error": "Invalid camera ID"}, status_code=400)

        source_index = get_security_context().monitor_server.selected_sources[camera_id - 1]
        frame = get_security_context().monitor_server.webcam_loader.get_frame(source_index)

        if not isinstance(frame, np.ndarray) or frame.size == 0:
            return JSONResponse({"error": "No valid frame available"}, status_code=404)

        try:
            result = analyze_frame(frame, camera_id=camera_id)
            return JSONResponse(clean_result_for_json(result))
        except Exception as e:
            logger.error(f"Error analyzing frame: {e}")
            return JSONResponse({"error": "Analsis failed"}, status_code=500)

    @app.get("/stream/{camera_id}")
    async def video_stream(camera_id: int):
        """Stream video from a camera as MJPEG"""
        if not (1 <= camera_id <= 4):
            return Response("Invalid camera ID", status_code=400)

        def generate():
            source_index = get_security_context().monitor_server.selected_sources[camera_id - 1]
            frame_duration = 1.0 / 20  # Target 20 FPS

            while get_security_context().monitor_server.running:
                start_time = time.time()
                frame_bytes = get_security_context().monitor_server.get_frame_as_jpeg(source_index, camera_id)

                if frame_bytes:
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
                    )

                # Frame rate limiting
                elapsed = time.time() - start_time
                if (frame_duration - elapsed) > 0:
                    time.sleep(frame_duration - elapsed)

        return StreamingResponse(
            generate(), media_type="multipart/x-mixed-replace; boundary=frame"
        )

    @app.post("/api/toggle_bounding_boxes")
    async def toggle_bounding_boxes(enabled: bool = True):
        """Toggle YOLO bounding box display"""
        get_security_context().monitor_server.toggle_bounding_boxes(enabled)
        return JSONResponse(
            {
                "status": "success",
                "bounding_boxes_enabled": enabled,
                "message": f"Bounding boxes {'enabled' if enabled else 'disabled'}",
            }
        )

    @app.get("/api/test_sources")
    async def test_sources():
        """Test all camera sources"""
        sources = get_security_context().monitor_server.webcam_loader.sources
        results = []

        for i, source in enumerate(sources):
            results.append(
                {
                    "index": i,
                    "name": f"Camera {i+1}",
                    "location": "YouTube Stream",
                    "type": "YouTube",
                    "working": get_security_context().monitor_server.webcam_loader.get_frame(i) is not None,
                }
            )

        return JSONResponse(results)


def create_video_interface():
    """Create the video monitoring interface HTML"""
    static_dir = Path(__file__).parent / "static"
    static_dir.mkdir(exist_ok=True)

    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AEGIS Security Co-Pilot - Real-time Surveillance</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); color: white; min-height: 100vh; }
        .header { text-align: center; padding: 20px; background: linear-gradient(135deg, #1e293b 0%, #334155 100%); border-bottom: 2px solid #3b82f6; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); }
        .header h1 { font-size: 2.5em; margin-bottom: 10px; display: flex; align-items: center; justify-content: center; gap: 15px; background: linear-gradient(135deg, #60a5fa, #3b82f6); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
        .header h1::before {
            content: '';
            display: inline-block;
            background-image: url('/static/img/logo.png');
            background-size: contain;
            background-repeat: no-repeat;
            width: 24px;
            height: 24px;
            margin-right: 8px;
            vertical-align: middle;
        }
        .header p { font-size: 1.1em; color: #cbd5e0; }
        .main-container { display: grid; grid-template-columns: 2fr 1fr; gap: 20px; padding: 20px; height: calc(100vh - 140px); }
        .video-section { display: flex; flex-direction: column; }
        .status-bar { display: flex; justify-content: space-between; align-items: center; padding: 15px 20px; background: rgba(30, 41, 59, 0.8); border-radius: 10px; margin-bottom: 20px; backdrop-filter: blur(10px); border: 1px solid rgba(59, 130, 246, 0.2); }
        .status-info { display: flex; gap: 20px; align-items: center; }
        .status-item { display: flex; align-items: center; gap: 8px; color: #a0aec0; }
        .status-item span#active-cameras, .status-item span#last-update { color: #cbd5e0; font-weight: 500; }
        .status-dot { width: 12px; height: 12px; border-radius: 50%; background: #ef4444; box-shadow: 0 0 10px rgba(239, 68, 68, 0.5); }
        .status-dot.active { background: #10b981; animation: pulse 2s infinite; box-shadow: 0 0 10px rgba(16, 185, 129, 0.5); }
        @keyframes pulse { 0% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); } 70% { box-shadow: 0 0 0 10px rgba(16, 185, 129, 0); } 100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); } }
        .video-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; flex: 1; }
        .camera-container { background: rgba(30, 41, 59, 0.6); border-radius: 15px; padding: 15px; border: 1px solid rgba(59, 130, 246, 0.3); transition: all 0.3s ease; display: flex; flex-direction: column; backdrop-filter: blur(10px); }
        .camera-container:hover { border-color: rgba(59, 130, 246, 0.6); box-shadow: 0 8px 32px rgba(59, 130, 246, 0.1); transform: translateY(-2px); }
        .camera-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }
        .camera-title { font-size: 1.2em; font-weight: bold; color: #f1f5f9; }
        .camera-status { width: 10px; height: 10px; border-radius: 50%; background: #ef4444; }
        .camera-status.active { background: #10b981; }
        .camera-status.connecting { background: #f59e0b; animation: pulse-orange 1.5s infinite; }
        @keyframes pulse-orange { 0% { opacity: 1; } 50% { opacity: 0.5; } 100% { opacity: 1; } }
        .video-frame { width: 100%; flex-grow: 1; background: #0f172a; border-radius: 8px; overflow: hidden; position: relative; min-height: 200px; }
        .video-stream { width: 100%; height: 100%; object-fit: cover; }
        .camera-info { position: absolute; bottom: 10px; left: 10px; background: rgba(0, 0, 0, 0.8); padding: 5px 10px; border-radius: 7px; font-size: 0.9em; backdrop-filter: blur(5px); }
        .vlm-description-box { margin-top: 15px; padding: 12px; background: rgba(15, 23, 42, 0.8); border-radius: 8px; border: 1px solid rgba(59, 130, 246, 0.2); height: 100px; overflow-y: auto; font-family: 'SF Mono', 'Courier New', Courier, monospace; line-height: 1.5; backdrop-filter: blur(5px); }
        .vlm-header { font-size: 0.9em; font-weight: bold; color: #60a5fa; margin-bottom: 8px; border-bottom: 1px solid rgba(59, 130, 246, 0.3); padding-bottom: 5px; }
        .vlm-text { font-size: 0.85em; color: #cbd5e0; white-space: pre-wrap; word-wrap: break-word; }
        .agent-section { display: flex; flex-direction: column; gap: 20px; }
        .controls-panel { background: rgba(30, 41, 59, 0.8); border-radius: 15px; padding: 20px; border: 1px solid rgba(59, 130, 246, 0.3); backdrop-filter: blur(10px); }
        .controls-header { font-size: 1.3em; font-weight: bold; color: #60a5fa; margin-bottom: 15px; display: flex; align-items: center; gap: 10px; }
        .controls-header::before { content: '⚙️'; }
        .controls { display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 15px; }
        .btn { padding: 8px 16px; background: linear-gradient(135deg, #3b82f6, #2563eb); border: none; border-radius: 8px; color: white; cursor: pointer; transition: all 0.2s ease; font-size: 0.9em; display: flex; align-items: center; gap: 8px; font-weight: 500; }
        .btn:hover { background: linear-gradient(135deg, #2563eb, #1d4ed8); transform: translateY(-1px); box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3); }
        .btn:disabled { background: #374151; color: #6b7280; cursor: not-allowed; transform: none; box-shadow: none; }
        .btn.secondary { background: linear-gradient(135deg, #6b7280, #4b5563); }
        .btn.secondary:hover { background: linear-gradient(135deg, #4b5563, #374151); }
        .toggle-switch { display: flex; align-items: center; gap: 10px; margin-bottom: 10px; }
        .switch { position: relative; width: 50px; height: 24px; background: #374151; border-radius: 12px; cursor: pointer; transition: all 0.3s ease; }
        .switch.active { background: #3b82f6; }
        .switch::after { content: ''; position: absolute; width: 20px; height: 20px; border-radius: 50%; background: white; top: 2px; left: 2px; transition: all 0.3s ease; }
        .switch.active::after { transform: translateX(26px); }
        .agent-dialogue { background: rgba(30, 41, 59, 0.8); border-radius: 15px; border: 1px solid rgba(59, 130, 246, 0.3); backdrop-filter: blur(10px); flex: 1; display: flex; flex-direction: column; }
        .dialogue-header { padding: 20px; border-bottom: 1px solid rgba(59, 130, 246, 0.3); font-size: 1.3em; font-weight: bold; color: #60a5fa; display: flex; align-items: center; gap: 10px; }
        .dialogue-header::before { content: '🤖'; }
        .dialogue-content { flex: 1; padding: 20px; overflow-y: auto; font-family: 'SF Mono', 'Courier New', Courier, monospace; font-size: 0.9em; line-height: 1.6; }
        .dialogue-input { padding: 20px; border-top: 1px solid rgba(59, 130, 246, 0.3); }
        .input-group { display: flex; gap: 10px; }
        .dialogue-input input { flex: 1; padding: 12px; background: rgba(15, 23, 42, 0.8); border: 1px solid rgba(59, 130, 246, 0.3); border-radius: 8px; color: white; font-size: 1em; outline: none; transition: all 0.3s ease; }
        .dialogue-input input:focus { border-color: #3b82f6; box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1); }
        .dialogue-input input::placeholder { color: #6b7280; }
        .message { margin-bottom: 15px; padding: 12px; border-radius: 8px; max-width: 80%; clear: both; }
        .message.user { background: rgba(59, 130, 246, 0.2); border: 1px solid rgba(59, 130, 246, 0.3); margin-left: auto; text-align: right; float: right; }
        .message.agent, .message.system { background: rgba(16, 185, 129, 0.2); border: 1px solid rgba(16, 185, 129, 0.3); float: left; text-align: left; }
        .message-header { font-size: 0.8em; color: #a0aec0; margin-bottom: 5px; }
        .message-text { color: #f1f5f9; }
        @media (max-width: 1200px) { .main-container { grid-template-columns: 1fr; grid-template-rows: 2fr 1fr; } .agent-section { flex-direction: row; } .controls-panel { min-width: 300px; } }
        @media (max-width: 768px) { .video-grid { grid-template-columns: 1fr; } .agent-section { flex-direction: column; } .main-container { grid-template-columns: 1fr; padding: 10px; } }
    </style>
</head>
<body>
    <div class="header">
        <h1>AEGIS Security Co-Pilot</h1>
        <p>AI-Powered Real-time Surveillance with Advanced Threat Detection</p>
    </div>
    <div class="main-container">
        <div class="video-section">
            <div class="status-bar">
                <div class="status-info">
                    <div class="status-item"><div class="status-dot active" id="system-status"></div><span>System Status</span></div>
                    <div class="status-item"><span>Active Cameras: <span id="active-cameras">4/4</span></span></div>
                    <div class="status-item"><span>Last Update: <span id="last-update">...</span></span></div>
                </div>
                <div class="controls">
                    <button class="btn" onclick="refreshStreams()">🔄 Refresh</button>
                    <button class="btn secondary" onclick="testSources()">🧪 Test Sources</button>
                </div>
            </div>
            <div class="video-grid">
                <div class="camera-container" id="camera-1">
                    <div class="camera-header"><div class="camera-title">Camera 1</div><div class="camera-status active" id="status-1"></div></div>
                    <div class="video-frame"><img class="video-stream" id="stream-1" src="/stream/1"><div class="camera-info" id="info-1">...</div></div>
                    <div class="vlm-description-box"><div class="vlm-header">🔍 Security Analysis</div><div class="vlm-text" id="vlm-text-1">Awaiting analysis...</div></div>
                </div>
                <div class="camera-container" id="camera-2">
                    <div class="camera-header"><div class="camera-title">Camera 2</div><div class="camera-status active" id="status-2"></div></div>
                    <div class="video-frame"><img class="video-stream" id="stream-2" src="/stream/2"><div class="camera-info" id="info-2">...</div></div>
                    <div class="vlm-description-box"><div class="vlm-header">🔍 Security Analysis</div><div class="vlm-text" id="vlm-text-2">Awaiting analysis...</div></div>
                </div>
                <div class="camera-container" id="camera-3">
                    <div class="camera-header"><div class="camera-title">Camera 3</div><div class="camera-status active" id="status-3"></div></div>
                    <div class="video-frame"><img class="video-stream" id="stream-3" src="/stream/3"><div class="camera-info" id="info-3">...</div></div>
                    <div class="vlm-description-box"><div class="vlm-header">🔍 Security Analysis</div><div class="vlm-text" id="vlm-text-3">Awaiting analysis...</div></div>
                </div>
                <div class="camera-container" id="camera-4">
                    <div class="camera-header"><div class="camera-title">Camera 4</div><div class="camera-status active" id="status-4"></div></div>
                    <div class="video-frame"><img class="video-stream" id="stream-4" src="/stream/4"><div class="camera-info" id="info-4">...</div></div>
                    <div class="vlm-description-box"><div class="vlm-header">🔍 Security Analysis</div><div class="vlm-text" id="vlm-text-4">Awaiting analysis...</div></div>
                </div>
            </div>
        </div>
        <div class="agent-section">
            <div class="controls-panel">
                <div class="controls-header">System Controls</div>
                <div class="toggle-switch"><span>YOLO Detection Overlay:</span><div class="switch" id="bbox-switch" onclick="toggleBoundingBoxes()"></div></div>
                <div class="controls">
                    <button class="btn" onclick="openAdkInterface()">🔗 ADK Interface</button>
                    <button class="btn secondary" onclick="exportReport()">📊 Export Report</button>
                </div>
                <div style="margin-top: 15px; padding-top: 15px; border-top: 1px solid rgba(59, 130, 246, 0.3);">
                    <div style="font-size: 0.9em; color: #a0aec0; margin-bottom: 10px;">Quick Commands:</div>
                    <div class="controls">
                        <button class="btn secondary" onclick="sendQuickCommand('scan all cameras for threats')">🔍 Scan All</button>
                        <button class="btn secondary" onclick="sendQuickCommand('check camera 1 for objects')">📹 Check C1</button>
                        <button class="btn secondary" onclick="sendQuickCommand('analyze crowd density')">👥 Crowd Check</button>
                    </div>
                </div>
            </div>
            <div class="agent-dialogue">
                <div class="dialogue-header">AI Security Assistant</div>
                <div class="dialogue-content" id="dialogue-content">
                    <div class="message agent">
                        <div class="message-header">AEGIS • System Ready</div>
                        <div class="message-text">Hello! I'm AEGIS, your AI Security Co-Pilot. Ready to assist with security monitoring and analysis.</div>
                    </div>
                </div>
                <div class="dialogue-input">
                    <div class="input-group">
                        <input type="text" id="command-input" placeholder="Ask AEGIS to analyze cameras, detect objects, or assess security..." onkeypress="handleKeyPress(event)">
                        <button class="btn" onclick="sendCommand()">Send</button>
                    </div>
                </div>
            </div>
        </div>
    </div>
    <script>
        const streamStates = {}; let boundingBoxesEnabled = false;
        function initMonitor() { for (let i = 1; i <= 4; i++) { streamStates[i] = { mode: 'mjpeg', retries: 0, element: document.getElementById(`stream-${i}`) }; streamStates[i].element.src = `/stream/${i}?t=${Date.now()}`; streamStates[i].element.addEventListener('error', () => switchToFallback(i)); } updateStreamInfo(); setInterval(updateAnalysisData, 3000); setInterval(updateStreamInfo, 5000); }
        async function updateStreamInfo() { try { const response = await fetch('/api/streams'); const streams = await response.json(); let activeCount = 0; streams.forEach(s => { const statusEl = document.getElementById(`status-${s.camera_id}`); const infoEl = document.getElementById(`info-${s.camera_id}`); if(statusEl) statusEl.className = 'camera-status ' + (s.active && s.has_frame ? 'active' : (s.active ? 'connecting' : '')); if(infoEl) infoEl.textContent = `${s.name} - ${s.location}`; if(s.active && s.has_frame) activeCount++; }); document.getElementById('active-cameras').textContent = `${activeCount}/4`; document.getElementById('last-update').textContent = new Date().toLocaleTimeString(); document.getElementById('system-status').classList.toggle('active', activeCount > 0); } catch (error) { console.error('Error updating stream info:', error); } }
        async function updateAnalysisData() { for (let i = 1; i <= 4; i++) { try { const response = await fetch(`/api/label/${i}`); const vlmTextElement = document.getElementById(`vlm-text-${i}`); if (!response.ok || !vlmTextElement) continue; const data = await response.json(); const newText = data.description || 'Awaiting analysis...'; if (vlmTextElement.textContent !== newText) vlmTextElement.textContent = newText; } catch (error) { console.error(`Error updating analysis for camera ${i}:`, error); } } }
        function switchToFallback(id) { const state = streamStates[id]; if (state.mode === 'fallback') return; state.mode = 'fallback'; const fetchAndUpdate = () => { if(state.mode !== 'fallback') return; fetch(`/api/frame/${id}`).then(r => r.ok ? r.json() : Promise.reject()).then(d => { if(d.image) state.element.src = d.image; }).catch(() => {}); }; fetchAndUpdate(); state.fallbackInterval = setInterval(fetchAndUpdate, 3000); }
        function refreshStreams() { window.location.reload(); }
        async function testSources() { try { const response = await fetch('/api/test_sources'); const results = await response.json(); alert('Source Test Results:\\n\\n' + results.map(r => `${r.working ? '✅' : '❌'} ${r.name} (${r.location})`).join('\\n')); } catch (error) { alert('Error testing sources: ' + error.message); } }
        async function toggleBoundingBoxes() { boundingBoxesEnabled = !boundingBoxesEnabled; const switchEl = document.getElementById('bbox-switch'); try { const response = await fetch('/api/toggle_bounding_boxes', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ enabled: boundingBoxesEnabled }) }); if (response.ok) { switchEl.classList.toggle('active', boundingBoxesEnabled); addMessage('system', `YOLO detection overlay ${boundingBoxesEnabled ? 'enabled' : 'disabled'}`); } } catch (error) { console.error('Error toggling bounding boxes:', error); } }
        function openAdkInterface() { window.open('/', '_blank'); }
        function exportReport() { addMessage('system', 'Security report export functionality will be available soon.'); }
        function handleKeyPress(event) { if (event.key === 'Enter') sendCommand(); }
        async function sendCommand() { const input = document.getElementById('command-input'); const command = input.value.trim(); if (!command) return; input.value = ''; addMessage('user', command); const loadingId = addMessage('agent', 'Processing...'); try { await sendRealAgentCommand(command, loadingId); } catch (error) { updateMessage(loadingId, 'Error processing command: ' + error.message); } }
        function sendQuickCommand(command) { document.getElementById('command-input').value = command; sendCommand(); }
        function addMessage(sender, text) { const content = document.getElementById('dialogue-content'); const messageId = 'msg-' + Date.now(); const messageDiv = document.createElement('div'); messageDiv.className = `message ${sender}`; messageDiv.id = messageId; const timestamp = new Date().toLocaleTimeString(); const senderName = sender === 'user' ? 'Security Operator' : sender === 'agent' ? 'AEGIS' : 'System'; messageDiv.innerHTML = `<div class="message-header">${senderName} • ${timestamp}</div><div class="message-text">${text}</div>`; content.appendChild(messageDiv); content.scrollTop = content.scrollHeight; return messageId; }
        function updateMessage(messageId, newText) { const messageEl = document.getElementById(messageId); if (messageEl) { const textEl = messageEl.querySelector('.message-text'); if (textEl) textEl.innerHTML = newText; } }
        async function sendRealAgentCommand(command, loadingId) { 
            try {
                // Create session if needed and send command to ADK agent
                const sessionResponse = await fetch('/apps/aegis_agent/users/user/sessions', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({})
                });
                
                if (!sessionResponse.ok) {
                    console.error('Session creation failed:', sessionResponse.status, sessionResponse.statusText);
                    throw new Error('Failed to create session: ' + sessionResponse.status);
                }
                
                const session = await sessionResponse.json();
                const sessionId = session.id;
                console.log('Session created:', sessionId);
                
                // Send command to agent with correct ADK API format
                const requestBody = {
                    appName: 'aegis_agent',        // MUST use directory name, not agent's internal name
                    userId: 'user',               // camelCase as per API schema  
                    sessionId: sessionId,         // camelCase as per API schema
                    newMessage: {
                        parts: [
                            {
                                text: command      // Correct ADK message format
                            }
                        ]
                    }
                };
                
                console.log('Sending request with correct ADK format:', requestBody);
                
                response = await fetch('/run', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(requestBody)
                });

                console.log('Response:', response);
                
                if (!response.ok) {
                    console.error('Agent request failed:', response.status, response.statusText);
                    const errorText = await response.text();
                    console.error('Error details:', errorText);
                    throw new Error('Agent request failed: ' + response.status);
                }
                
                const events = await response.json();
                console.log('ADK Events:', events); // Debug logging
                
                // Extract agent response from ADK events - parse content.parts[0].text
                let agentResponse = 'I\\'ve processed your request.';
                if (Array.isArray(events)) {
                    for (const event of events) {
                        console.log('Checking event:', event);
                        if (event && event.author && (
                            event.author === 'aegis_security_copilot' || 
                            event.author === 'aegis_agent' ||
                            event.author === 'root_agent'
                        )) {
                            // Extract text from ADK event structure: event.content.parts[0].text
                            if (event.content && event.content.parts && event.content.parts.length > 0 && event.content.parts[0].text) {
                                agentResponse = event.content.parts[0].text.trim();
                                console.log('Found agent response from', event.author, ':', agentResponse);
                                break;
                            }
                        }
                    }
                } else {
                    console.warn('Events is not an array:', events);
                }

                
                // Ensure we have a meaningful response
                if (!agentResponse || agentResponse.trim() === '' || agentResponse === 'I\\'ve processed your request.') {
                    agentResponse = 'AEGIS has analyzed your request. The security monitoring system is active and processing your command. Please check the camera feeds and analysis results above.';
                }
                
                updateMessage(loadingId, agentResponse);
                
            } catch (error) {
                console.error('Agent command error:', error);
                updateMessage(loadingId, 'Sorry, I\\'m having trouble processing your request. Please try again.');
            }
        }
        window.addEventListener('load', initMonitor);
    </script>
</body>
</html>"""

    index_file = static_dir / "index.html"
    with open(index_file, "w", encoding="utf-8") as f:
        f.write(html_content)

    logger.info("Video monitoring interface created successfully")


def main():
    """Main server entry point"""
    print("=" * 60)
    print(" AEGIS SECURITY CO-PILOT - ADK SERVER")
    print("=" * 60)

    # Initialize the global SecurityContext
    print("🔧 Initializing SecurityContext...")
    context = initialize_global_context()
    if not context.is_initialized():
        print("❌ Failed to initialize SecurityContext")
        return

    # Configure analysis interval to reduce resource usage
    # Options:
    # - 10.0: Analyze every 10 seconds (recommended for balanced performance)
    # - 30.0: Analyze every 30 seconds (good for low resource usage)
    # - float('inf'): Disable background analysis entirely (only on-demand via chat)
    analysis_interval = 30.0  # Change this value to adjust analysis frequency
    context.monitor_server.set_analysis_interval(analysis_interval)
    print(f"🔍 Background analysis interval set to {analysis_interval} seconds")

    # Create ADK FastAPI app
    app = create_adk_app()

    # Add video streaming routes
    add_video_routes(app)
    

    print("✅ ADK agent system initialized")
    print("📹 Video monitoring system ready")
    # Use the PORT environment variable if available (for Cloud Run compatibility)
    #port = int(os.environ.get("PORT", 4000))
    port = 4001

    print(f"🌐 Server starting at: http://localhost:{port}")
    print(f"🔧 ADK Interface: http://localhost:{port}")
    print(f"📺 Video Interface: http://localhost:{port}/video")
    print("🛑 Press Ctrl+C to stop the server")
    print("=" * 60)

    try:
        uvicorn.run(app, host="0.0.0.0", port=port)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down server...")
        cleanup_global_context()
        print("✅ Server stopped successfully")
    except Exception as e:
        print(f"❌ Server error: {e}")
        cleanup_global_context()


if __name__ == "__main__":
    main()
