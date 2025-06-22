// src/App.tsx
import React, { useEffect, useRef, useState } from 'react';

export default function App() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [prompt, setPrompt] = useState('');
  const [result, setResult] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isStreamActive, setIsStreamActive] = useState(false);
  const [streamStatus, setStreamStatus] = useState<any>(null);
  const [error, setError] = useState('');
  const [captureInterval, setCaptureInterval] = useState(200); // Default 200ms (5 fps)
  const intervalRef = useRef<NodeJS.Timeout | null>(null);
  const statusIntervalRef = useRef<NodeJS.Timeout | null>(null);

  // API base URL
  const API_BASE = 'http://localhost:8000/api';

  // Start camera stream on mount
  useEffect(() => {
    const enableCamera = async () => {
      try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
          
          // Add event listeners for better error handling
          videoRef.current.onloadedmetadata = () => {
            videoRef.current?.play().catch(err => {
              console.error('Video play error:', err);
              // Try to play again after a short delay
              setTimeout(() => {
                videoRef.current?.play().catch(console.error);
              }, 100);
            });
          };
          
          videoRef.current.onerror = (e) => {
            console.error('Video element error:', e);
            setError('Video stream error. Please refresh the page.');
          };
        }
      } catch (err) {
        setError('Failed to access camera. Please ensure camera permissions are granted.');
        console.error('Camera error:', err);
      }
    };
    enableCamera();
  }, []);

  // Cleanup intervals on unmount
  useEffect(() => {
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
      if (statusIntervalRef.current) {
        clearInterval(statusIntervalRef.current);
      }
    };
  }, []);

  // Cleanup on page unload
  useEffect(() => {
    const handleBeforeUnload = () => {
      // Try to cleanup temp files when page is unloaded
      fetch(`${API_BASE}/cleanup-temp`, { method: "POST" }).catch(() => {});
    };

    window.addEventListener('beforeunload', handleBeforeUnload);
    return () => {
      window.removeEventListener('beforeunload', handleBeforeUnload);
    };
  }, []);

  // Function to capture and send frame to buffer
  const sendFrameToBuffer = async () => {
    try {
    const canvas = document.createElement("canvas");
    const video = videoRef.current;
    if (!video) return;
  
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
      if (!ctx) return;
      
      ctx.drawImage(video, 0, 0);
  
      // Convert canvas to blob and send to backend buffer
      const blob = await new Promise<Blob>((resolve) => {
        canvas.toBlob((blob) => {
          if (blob) resolve(blob);
        }, "image/jpeg", 0.8);
      });

      const formData = new FormData();
      formData.append("file", new File([blob], "buffer_frame.jpg", { type: "image/jpeg" }));

      const response = await fetch(`${API_BASE}/add-frame-to-buffer`, {
        method: "POST",
        body: formData,
      });

      if (response.ok) {
        const data = await response.json();
        console.log('Frame added to buffer:', data.buffer_size, 'frames total');
      } else {
        console.error('Failed to add frame to buffer:', response.status);
      }

    } catch (err) {
      console.error('Error capturing/sending frame:', err);
    }
  };

  // Capture frame from video and analyze
  const captureAndAnalyze = async () => {
    if (!prompt.trim()) {
      setError('Please enter a prompt');
      return;
    }

    setIsLoading(true);
    setError('');

    try {
      const canvas = document.createElement("canvas");
      const video = videoRef.current;
      if (!video) {
        throw new Error('Video not available');
      }

      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext("2d");
      if (!ctx) {
        throw new Error('Canvas context not available');
      }
      
      ctx.drawImage(video, 0, 0);

      // Convert canvas to blob
      const blob = await new Promise<Blob>((resolve) => {
        canvas.toBlob((blob) => {
          if (blob) resolve(blob);
        }, "image/jpeg", 0.8);
      });

      // Create form data
      const formData = new FormData();
      formData.append("file", new File([blob], "frame.jpg", { type: "image/jpeg" }));
      formData.append("prompt", prompt);
  
      // Send to API
      const response = await fetch(`${API_BASE}/analyze-frame`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
      }

      const data = await response.json();
      
      if (data.status === 'success') {
        setResult(data.result);
      } else {
        throw new Error(data.message || 'Analysis failed');
      }

    } catch (err) {
      setError(`Analysis failed: ${err instanceof Error ? err.message : 'Unknown error'}`);
      console.error('Analysis error:', err);
    } finally {
      setIsLoading(false);
    }
  };

  // Start real-time stream analysis
  const startStreamAnalysis = async () => {
    try {
      setError('');
      console.log('Starting stream analysis...');
      console.log('API_BASE:', API_BASE);
      
      const url = `${API_BASE}/start-stream`;
      const body = new URLSearchParams({
        video_source: '0',
        fps_limit: '10'
      });
      
      console.log('Request URL:', url);
      console.log('Request body:', body.toString());
      
      const response = await fetch(url, {
        method: "POST",
        headers: { 
          'Content-Type': 'application/x-www-form-urlencoded',
          'Accept': 'application/json'
        },
        body: body
      });

      console.log('Response status:', response.status);
      console.log('Response ok:', response.ok);

      if (!response.ok) {
        const errorText = await response.text();
        console.error('Response error text:', errorText);
        throw new Error(`Failed to start stream: ${response.status} - ${errorText}`);
      }

      const data = await response.json();
      console.log('Response data:', data);
      
      if (data.status === 'success') {
        setIsStreamActive(true);
        
        // Start sending frames to buffer at selected interval
        intervalRef.current = setInterval(sendFrameToBuffer, captureInterval);
        
        // Start polling stream status every 2 seconds
        statusIntervalRef.current = setInterval(pollStreamStatus, 2000);
        
        // Get initial status
        pollStreamStatus();
      } else {
        throw new Error(data.message || 'Failed to start stream');
      }
    } catch (err) {
      console.error('Stream start error:', err);
      if (err instanceof TypeError && err.message.includes('fetch')) {
        setError('Network error: Cannot connect to backend. Make sure backend is running on port 8000.');
      } else {
        setError(`Failed to start stream: ${err instanceof Error ? err.message : 'Unknown error'}`);
      }
    }
  };

  // Stop real-time stream analysis
  const stopStreamAnalysis = async () => {
    try {
      // Clear intervals
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
      if (statusIntervalRef.current) {
        clearInterval(statusIntervalRef.current);
        statusIntervalRef.current = null;
      }

      const response = await fetch(`${API_BASE}/stop-stream`, {
        method: "POST"
      });

      if (!response.ok) {
        throw new Error(`Failed to stop stream: ${response.status}`);
      }

      setIsStreamActive(false);
      setStreamStatus(null);
    } catch (err) {
      setError(`Failed to stop stream: ${err instanceof Error ? err.message : 'Unknown error'}`);
    }
  };

  // Analyze recent frames from stream buffer
  const analyzeRecentFrames = async () => {
    if (!prompt.trim()) {
      setError('Please enter a prompt');
      return;
    }

    setIsLoading(true);
    setError('');

    try {
      const formData = new FormData();
      formData.append("prompt", prompt);
      formData.append("n_frames", "5");  // Analyze last 5 frames
      formData.append("min_frames", "1");

      const response = await fetch(`${API_BASE}/analyze-recent-frames`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
      }

      const data = await response.json();
      
      if (data.status === 'success') {
        setResult(data.result);
      } else if (data.status === 'insufficient_frames') {
        setError(`${data.message}. Please wait for more frames to be captured.`);
      } else {
        throw new Error(data.message || 'Analysis failed');
      }

    } catch (err) {
      setError(`Analysis failed: ${err instanceof Error ? err.message : 'Unknown error'}`);
    } finally {
      setIsLoading(false);
    }
  };

  // Poll stream status
  const pollStreamStatus = async () => {
    try {
      const response = await fetch(`${API_BASE}/stream-status`);
      if (response.ok) {
        const data = await response.json();
        setStreamStatus(data.buffer_info);
      }
    } catch (err) {
      console.error('Failed to get stream status:', err);
    }
  };

  // Health check
  const checkAPIHealth = async () => {
    try {
      const response = await fetch('http://localhost:8000/health');
      return response.ok;
    } catch {
      return false;
    }
  };

  // Test backend connectivity
  const testBackendConnection = async () => {
    try {
      console.log('Testing backend connection...');
      const response = await fetch('http://localhost:8000/health');
      console.log('Health check response:', response);
      if (response.ok) {
        const data = await response.json();
        console.log('Health check data:', data);
        setError(`✅ Backend connected successfully! Response: ${JSON.stringify(data)}`);
      } else {
        setError(`❌ Backend returned error: ${response.status}`);
      }
    } catch (err) {
      console.error('Connection test error:', err);
      setError(`❌ Cannot connect to backend: ${err instanceof Error ? err.message : 'Unknown error'}`);
    }
  };

  // Clean up temp files
  const cleanupTempFiles = async () => {
    try {
      const response = await fetch(`${API_BASE}/cleanup-temp`, {
        method: "POST"
      });

      if (response.ok) {
        const data = await response.json();
        setError(`✅ ${data.message}`);
      } else {
        setError(`❌ Cleanup failed: ${response.status}`);
      }
    } catch (err) {
      setError(`❌ Cleanup failed: ${err instanceof Error ? err.message : 'Unknown error'}`);
    }
  };

  // Update capture interval
  const updateCaptureInterval = (newInterval: number) => {
    setCaptureInterval(newInterval);
    
    // If streaming is active, restart with new interval
    if (isStreamActive && intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = setInterval(sendFrameToBuffer, newInterval);
      console.log(`Updated capture interval to ${newInterval}ms (${(1000/newInterval).toFixed(1)} fps)`);
    }
  };

  // Get FPS from interval
  const getFpsFromInterval = (interval: number) => {
    return (1000 / interval).toFixed(1);
  };

  return (
    <div className="app-container">
      <h1 className="app-title">
        viclab Real-time Video Analysis
      </h1>

      {/* Error Display */}
      {error && (
        <div className="error-message">
          <strong>Error:</strong> {error}
        </div>
      )}

      <div className="main-grid">
        
        {/* Video Stream Section */}
        <div className="video-section">
          <video 
            ref={videoRef} 
            className="video-element" 
            muted 
            autoPlay 
          />
          <p className="video-caption">Live camera stream</p>
          
          {/* Stream Controls */}
          <div className="stream-controls">
            {!isStreamActive ? (
              <button
                onClick={startStreamAnalysis}
                className="btn-start-stream"
              >
                Start Stream Buffer
              </button>
            ) : (
              <button
                onClick={stopStreamAnalysis}
                className="btn-stop-stream"
              >
                Stop Stream Buffer
              </button>
            )}
            <button
              onClick={testBackendConnection}
              className="btn-start-stream"
              style={{ marginLeft: '10px', backgroundColor: '#2563eb' }}
            >
              Test Backend Connection
            </button>
            <button
              onClick={cleanupTempFiles}
              className="btn-start-stream"
              style={{ marginLeft: '10px', backgroundColor: '#dc2626' }}
            >
              Clean Temp Files
            </button>
          </div>

          {/* Capture Interval Controls */}
          <div className="capture-controls">
            <label htmlFor="capture-interval" className="label">
              Frame Capture Rate
            </label>
            <div className="capture-interval-container">
              <select
                id="capture-interval"
                value={captureInterval}
                onChange={(e) => updateCaptureInterval(Number(e.target.value))}
                className="interval-select"
              >
                <option value={50}>50ms (20.0 fps)</option>
                <option value={100}>100ms (10.0 fps)</option>
                <option value={200}>200ms (5.0 fps)</option>
                <option value={500}>500ms (2.0 fps)</option>
                <option value={1000}>1s (1.0 fps)</option>
                <option value={2000}>2s (0.5 fps)</option>
                <option value={5000}>5s (0.2 fps)</option>
              </select>
              <span className="fps-display">
                Current: {getFpsFromInterval(captureInterval)} fps
                {isStreamActive && <span className="active-indicator"> (Active)</span>}
              </span>
            </div>
          </div>

          {/* Stream Status */}
          {streamStatus && (
            <div className="stream-status">
              <h3>Stream Buffer Status</h3>
              <p>Buffer Size: {streamStatus.buffer_size}/{streamStatus.max_buffer_size}</p>
              <p>Stream Active: {streamStatus.is_streaming ? 'Yes' : 'No'}</p>
              <p>Time Span: {streamStatus.buffer_time_span?.toFixed(2)}s</p>
            </div>
          )}
        </div>

        {/* Controls Section */}
        <div className="controls-section">
          <label htmlFor="prompt" className="label">
            Analysis Prompt
          </label>
          <input
            id="prompt"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            className="input-field"
            placeholder="e.g., Describe what you see in the video"
            disabled={isLoading}
          />

          {/* Analysis Buttons */}
          <div className="button-group">
          <button
              onClick={captureAndAnalyze}
              disabled={isLoading || !prompt.trim()}
              className="btn-analyze"
          >
              {isLoading ? 'Analyzing...' : 'Analyze Current Frame'}
          </button>

            {isStreamActive && (
              <button
                onClick={analyzeRecentFrames}
                disabled={isLoading || !prompt.trim()}
                className="btn-analyze-buffer"
              >
                {isLoading ? 'Analyzing...' : 'Analyze Recent Frames'}
              </button>
            )}
          </div>

          {/* Results Section */}
          <div className="results-section">
            <h2 className="results-title">Analysis Result</h2>
            <div className="results-box">
              {result || 'No analysis yet. Enter a prompt and click analyze.'}
            </div>
          </div>

          {/* Instructions */}
          <div className="instructions-section">
            <h3 className="instructions-title">Instructions:</h3>
            <ul className="instructions-list">
              <li>Select your preferred frame capture rate from the dropdown (50ms to 5s)</li>
              <li>Click "Start Stream Buffer" to begin collecting frames automatically</li>
              <li>You can change the capture rate even while streaming is active</li>
              <li>Use "Analyze Current Frame" for immediate single-frame analysis</li>
              <li>Use "Analyze Recent Frames" to analyze frames from the buffer</li>
              <li>Higher fps (shorter intervals) = more responsive but more processing</li>
              <li>Make sure your backend is running on port 8000</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}
